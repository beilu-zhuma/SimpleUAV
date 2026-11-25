"""UAV-ON主方法批量执行、有限恢复与完整评分。"""

import argparse
import hashlib
import json
import concurrent.futures
import queue
import re
import signal
import socket
import subprocess
import sys
import time
import threading
from collections import Counter
from pathlib import Path

from lightning_vln.core.rpc import RPC
from lightning_vln.core.errors import SimulationUnavailable
from lightning_vln.core.simulator import DECISION_CAPTURE_ATTEMPTS, COLLISION_PROTOCOL, SPAWN_COLLISION_POLICY
from lightning_vln.adapters.uavon.trainset import TRAIN_SCENES, load_train_sample
from lightning_vln.core.frames import load_scene_offset
from lightning_vln.adapters.uavon.benchmark import TEST_SCENES, load_full_testset
from lightning_vln.core.vlm import PROMPT_VERSION
from lightning_vln.core.evidence import PROMPT_VERSION as EVIDENCE_PROMPT_VERSION
from lightning_vln.core.motion_options import PROMPT_VERSION as MOTION_OPTIONS_PROMPT_VERSION
from lightning_vln.adapters.uavon.evaluation import PROBLEM_SCENES, UPSTREAM_COMMIT, run_official_metrics, export_episode
from lightning_vln.adapters.uavon.scene import interrupt, stop_process_group as stop_scene

from lightning_vln.adapters.uavon.tuning import METHOD_VERSION
from lightning_vln.adapters.uavon.minimal.sampling import load_test_sample

PROTOCOL = METHOD_VERSION

DATASET_SHA256 = '97d6aff50832a73302b4b1d801a73b2f71cf0354f7ad3a1cbb7008f9c7831ace'
RECOVERY_POLICY = {'version': 3, 'scene_restarts': 1, 'decision_capture_attempts': DECISION_CAPTURE_ATTEMPTS}
SCENES = TEST_SCENES


def routed_agent_args(agent_args, endpoints, gpu):
    """只路由同模型服务与硬件标签，不改变导航和评分参数。"""
    result = agent_args.copy()
    if endpoints is None:
        return result
    endpoint = endpoints[str(gpu)]
    for flag, value in (('--base-url', endpoint['base_url']), ('--model', endpoint['model'])):
        result[result.index(flag)+1] = value
    for flag in ('--model-hardware', '--sim-hardware'):
        if flag in result:
            i = result.index(flag)+1
            result[i] = re.sub(r'GPU[0-3]', 'GPU'+str(gpu), result[i])
    return result


def recovery_upgrade_allowed(previous, current):
    """仅允许明确的基础设施恢复升级，导航和传感器校验不在例外内。"""
    if previous.get('version')==5 and current.get('version')==6:
        new_fields={'version','model_request_attempts','model_retry_timeout_s','model_retry_backoff_s','model_fault_isolation'}
        return (current.get('model_request_attempts')==3 and current.get('model_retry_timeout_s')==180
                and current.get('model_retry_backoff_s')==[5,15] and current.get('model_fault_isolation')=='episode'
                and {k:v for k,v in previous.items() if k not in new_fields}
                  =={k:v for k,v in current.items() if k not in new_fields})
    if current.get('version') == 3 and previous in [dict(current, version=v) for v in (1, 2)]:
        return True
    changed = {'version', 'scene_restarts', 'restart_backoff_s', 'max_restart_backoff_s', 'model_transport_recoverable'}
    return (previous.get('version') in (4, 5) and current.get('version') == 5
            and 0 <= previous.get('scene_restarts', -1) <= current.get('scene_restarts', -1) <= 5
            and {k:v for k,v in previous.items() if k not in changed}
                == {k:v for k,v in current.items() if k not in changed}
            and current.get('restart_backoff_s') == 10 and current.get('max_restart_backoff_s') == 30
            and current.get('model_transport_recoverable') is True)


def check_scene_port(port):
    """允许已关闭连接的 TIME_WAIT，仍拒绝已有监听场景。"""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('127.0.0.1', port))
        except OSError as error:
            raise OSError(error.errno, f'场景端口预检失败：{port}；{error}') from error


def selected_tasks(dataset: Path, episode_ids: list[str], allow_problem_scenes: bool = False) -> list[dict]:
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError('数据不匹配固定 random50；禁止用重新抽样的 50 条代替')
    tasks = json.loads(dataset.read_text())
    ids = [str(t['episode_id']) for t in tasks]
    if len(tasks) != 50 or len(set(ids)) != 50:
        raise ValueError('random50 必须恰好包含 50 个唯一 episode')
    if len(set(episode_ids)) != len(episode_ids) or set(episode_ids)-set(ids):
        raise ValueError('门禁 episode-id 重复或不在固定 subset 中')
    excluded = {str(t['episode_id']) for t in tasks if t['map_name'] in PROBLEM_SCENES}
    if not allow_problem_scenes and excluded.intersection(episode_ids):
        raise ValueError('所选任务位于官方1.0问题场景；历史诊断需 --allow-v1-problem-scenes')
    if not allow_problem_scenes:
        tasks = [t for t in tasks if t['map_name'] not in PROBLEM_SCENES]
    return [t for t in tasks if not episode_ids or str(t['episode_id']) in episode_ids]


def aggregate(tasks: list[dict], results: list[dict]) -> dict:
    if not tasks:
        raise ValueError('不能结算空任务集')
    ids = [str(r['episode_id']) for r in results]
    if Counter(ids) != Counter(str(t['episode_id']) for t in tasks):
        raise ValueError('结果覆盖不完整或存在重复 episode；不能结算 SR')
    terminal = ('stop', 'decision_limit', 'collision', 'ascent_guard_exhausted', 'initial_yaw_not_reached')
    if any(r['status'] not in terminal for r in results):
        raise ValueError('结果中存在未结算任务')
    if any(r['success'] for r in results if r['status'] in ('ascent_guard_exhausted', 'initial_yaw_not_reached')):
        raise ValueError('保护耗尽或初始转向失败不能标为成功')
    success = sum(r['success'] for r in results)
    if any(r.get('record_mode') == 'results-only' for r in results):
        if not all(r.get('record_mode') == 'results-only' for r in results):
            raise ValueError('不能混合仅结果与完整记录的结算')
        return dict(status='completed',total_finished=len(results),success=success,sr=success/len(results),
                    record_mode='results-only',official_comparable=False,results=results)
    if any(r.get('record_mode') == 'metrics-only' for r in results):
        if not all(r.get('record_mode') == 'metrics-only' for r in results):
            raise ValueError('不能混合metrics-only与其他留档模式')
        if not all(all(k in r and r[k] is not None for k in ('oracle_success','spl','ne_m','sampled_path_length_m')) for r in results):
            raise ValueError('metrics-only缺少必需指标，不得结算')
    oracle = sum(r['oracle_success'] for r in results)
    return {**({'record_mode':'metrics-only'} if all(r.get('record_mode')=='metrics-only' for r in results) else {}),
            'ne_m':sum(r['final_distance_m'] for r in results)/len(results) if all('final_distance_m' in r for r in results) else None,
            'status': 'completed', 'total_finished': len(results), 'success': success,
            'oracle_success': oracle, 'sr': success/len(results), 'osr': oracle/len(results),
            'collisions': sum(r['status'] == 'collision' for r in results),
            'spl': sum(r['spl'] for r in results)/len(results) if all(r.get('spl') is not None for r in results) else None,
            'dts_m': sum(r['final_distance_m'] for r in results)/len(results) if all('final_distance_m' in r for r in results) else None,
            'diagnostic_only': False, 'official_comparable': False, 'results': results}


def agent_args_without_resources(args):
    remaining = []
    index = 0
    while index < len(args):
        if args[index] in ('--base-url', '--model-hardware', '--sim-hardware'):
            if index+1 >= len(args): raise ValueError('资源参数缺少值')
            index += 2
        else:
            remaining.append(args[index]); index += 1
    return remaining


def reusable_results(source: Path, plan: dict, tasks: list[dict], allow_recovery_upgrade=False,
                     allow_resource_relocation=False, allow_spawn_collision_upgrade=False) -> list[tuple[dict, Path]]:
    previous = json.loads((source/'manifest.json').read_text())
    default_module = 'lightning_vln.adapters.uavon.tuning.cli'
    if previous.get('agent_module',default_module) != plan.get('agent_module',default_module):
        raise ValueError('接续运行设置不一致：agent_module')
    if previous.get('finite_actions_ablation') != plan.get('finite_actions_ablation'):
        raise ValueError('接续运行设置不一致：finite_actions_ablation')
    for key in ('dataset_sha256', 'prompt_version', 'protocol', 'upstream_commit', 'agent_args', 'recovery_policy', 'split', 'benchmark','frame_manifest_sha256', 'collision_detection_protocol', 'spawn_collision_policy'):
        if previous.get(key) != plan.get(key):
            if (key == 'spawn_collision_policy' and allow_spawn_collision_upgrade
                    and previous.get(key) is None and plan.get(key) == SPAWN_COLLISION_POLICY
                    and previous.get('collision_detection_protocol') == COLLISION_PROTOCOL):
                continue
            if (key == 'recovery_policy' and allow_recovery_upgrade
                    and recovery_upgrade_allowed(previous.get(key) or {}, plan.get(key) or {})):
                continue
            if (key == 'agent_args' and allow_resource_relocation
                    and agent_args_without_resources(previous[key]) == agent_args_without_resources(plan[key])):
                continue
            if key=='agent_args' and allow_recovery_upgrade:
                before=[v for v in previous[key] if v!='--isolated-model-recovery']
                after=[v for v in plan[key] if v!='--isolated-model-recovery']
                if allow_resource_relocation:before,after=map(agent_args_without_resources,(before,after))
                if before==after:continue
            raise ValueError(f'接续运行设置不一致：{key}')
    expected = {str(t['episode_id']): t for t in tasks}
    completed = []
    seen = set()
    def accept(result, episode):
        identity = str(result['episode_id'])
        if identity not in expected:
            return
        if identity in seen:
            raise ValueError('接续结果重复：'+identity)
        task_identity = json.loads((episode/'task.json').read_text())
        wanted = (dict(episode_id=identity,task_sha256=hashlib.sha256(json.dumps(
            expected[identity],sort_keys=True,ensure_ascii=False).encode()).hexdigest())
            if result.get('record_mode') in ('results-only', 'metrics-only') else expected[identity])
        if (task_identity != wanted
                or json.loads((episode/'summary.json').read_text()) != result):
            raise ValueError('接续任务或评分内容不一致：'+identity)
        aggregate([expected[identity]], [result])
        completed.append((result, episode.resolve()))
        seen.add(identity)
    for identity, location in previous.get('reused_episodes', {}).items():
        episode = Path(location)
        result = json.loads((episode/'summary.json').read_text())
        if str(result['episode_id']) != str(identity):
            raise ValueError('接续来源身份不一致：'+str(identity))
        accept(result, episode)
    # results.json 在逐任务评分和导出都完成后写入；不接续未结算的中间 summary。
    for results_path in sorted([*source.glob('*/agent/results.json'),
                                *source.glob('*/attempt_*/agent/results.json')]):
        directories = {}
        for episode in results_path.parent.glob('episode_*'):
            task_path = episode/'task.json'
            if task_path.exists():
                task = json.loads(task_path.read_text())
                directories[str(task['episode_id'])] = episode
        for result in json.loads(results_path.read_text()):
            identity = str(result['episode_id'])
            accept(result, directories[identity])
    return completed


def settled_results(root: Path, tasks: list[dict]) -> list[dict]:
    """只接纳已导出并落盘的结算，不采纳异常 attempt 的孤立 summary。"""
    path = root/'agent/results.json'
    if not path.exists():
        return []
    results = json.loads(path.read_text())
    expected = {str(t['episode_id']): t for t in tasks}
    ids = [str(r['episode_id']) for r in results]
    if len(set(ids)) != len(ids) or set(ids)-expected.keys():
        raise ValueError('attempt 结果身份重复或越界')
    episodes = {}
    for episode in (root/'agent').glob('episode_*'):
        task = json.loads((episode/'task.json').read_text())
        identity = str(task['episode_id'])
        if identity in episodes:
            raise ValueError('attempt 任务身份重复')
        episodes[identity] = episode
    for result in results:
        identity = str(result['episode_id'])
        episode = episodes[identity]
        task_identity = json.loads((episode/'task.json').read_text())
        wanted = (dict(episode_id=identity,task_sha256=hashlib.sha256(json.dumps(
            expected[identity],sort_keys=True,ensure_ascii=False).encode()).hexdigest())
            if result.get('record_mode') in ('results-only', 'metrics-only') else expected[identity])
        if (task_identity != wanted
                or json.loads((episode/'summary.json').read_text()) != result):
            raise ValueError('attempt 任务或评分内容不一致')
        aggregate([expected[identity]], [result])
    return results


class DeferredModelTasks(SimulationUnavailable):
    """模型传输耗尽的任务已隔离；本组其他任务已继续，不重建UE重试这些任务。"""


def recover_scene(root, tasks, run_attempt, commit, restarts=1, cancelled=None):
    """重建故障场景并接续未结算任务；每次 attempt 原样保留，不按成绩重试。"""
    pending = list(tasks)
    deferred = {}
    for number in range(restarts + 1):
        if cancelled is not None and cancelled.is_set():
            raise RuntimeError('并发评测已取消，不重建场景')
        attempt = root if number == 0 else root/f'attempt_{number:02d}'
        attempt.mkdir(parents=True, exist_ok=False)
        error = None
        try:
            run_attempt(attempt, pending)
        except SimulationUnavailable as caught:
            error = caught
            (attempt/'recovery.json').write_text(json.dumps({
                'error': repr(caught), 'attempt': number,
                'pending_ids': [str(t['episode_id']) for t in pending]}, ensure_ascii=False, indent=2))
        completed = settled_results(attempt, pending)
        commit(completed)
        ids = {str(r['episode_id']) for r in completed}
        deferred_path=attempt/'agent/deferred.json'
        if deferred_path.exists():
            expected={str(t['episode_id']) for t in pending}
            for item in json.loads(deferred_path.read_text()):
                key=str(item['episode_id'])
                if key not in expected or key in ids or key in deferred or item.get('not_scored') is not True:
                    raise ValueError('延期任务身份不一致或已被结算')
                deferred[key]=item
            (root/'model-deferred.json').write_text(json.dumps(list(deferred.values()),ensure_ascii=False,indent=2))
        pending = [t for t in pending if str(t['episode_id']) not in ids and str(t['episode_id']) not in deferred]
        if not pending:
            if deferred:raise DeferredModelTasks('模型传输耗尽，其他任务已完成；未评分任务：'+','.join(deferred))
            return
        if error is None:
            raise ValueError('agent 成功退出但结果未完整结算')
        if number == restarts:
            raise error
        delay = min(RECOVERY_POLICY.get('max_restart_backoff_s', 30),
                    RECOVERY_POLICY.get('restart_backoff_s', 0) * (2 ** number))
        if delay:
            (attempt/'restart-wait.json').write_text(json.dumps(
                {'delay_s': delay, 'next_attempt': number+1, 'pending_ids': [str(t['episode_id']) for t in pending]}))
            if cancelled is not None:
                if cancelled.wait(delay): raise RuntimeError('重建等待已取消')
            else:
                time.sleep(delay)


def parallel_scenes(scenes, slots, work):
    """仿真故障隔离至场景，其他错误取消全局；返回耗尽恢复的场景。"""
    pending = queue.Queue()
    for scene in scenes:
        pending.put(scene)
    cancelled = threading.Event()
    failures = {}
    failure_lock = threading.Lock()
    def worker(slot):
        while not cancelled.is_set():
            try:
                scene = pending.get_nowait()
            except queue.Empty:
                return
            try:
                work(scene, slot, cancelled)
            except SimulationUnavailable as error:
                with failure_lock:
                    failures[scene] = repr(error)
            except BaseException:
                cancelled.set()
                raise
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(slots)))
    try:
        futures = [executor.submit(worker, slot) for slot in slots]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    finally:
        cancelled.set()
        executor.shutdown(wait=True, cancel_futures=True)
    return failures


def scene_jobs(tasks, batch_size=0):
    jobs = {}
    for scene in dict.fromkeys(t['map_name'] for t in tasks):
        rows = [t for t in tasks if t['map_name'] == scene]
        size = batch_size or len(rows)
        for offset in range(0, len(rows), size):
            name = scene if not batch_size else f'{scene}_part{offset//size:03d}'
            jobs[name] = rows[offset:offset+size]
    return dict(sorted(jobs.items(), key=lambda item: -len(item[1])))


def agent_failure(agent, root):
    path = root/'failure.json'
    failure = json.loads(path.read_text()) if path.exists() else {}
    if failure.get('recoverable_simulation') is True:
        raise SimulationUnavailable(f"{failure['error_type']}: {failure['error']}")
    if (RECOVERY_POLICY.get('model_transport_recoverable') is True
            and failure.get('error_type') in ('TimeoutError', 'URLError', 'ConnectionError', 'ConnectionResetError')):
        raise SimulationUnavailable(f"ModelTransportFailure: {failure['error_type']}: {failure.get('error')}")
    if (RECOVERY_POLICY.get('model_transport_recoverable') is True
            and failure.get('error_type') == 'HTTPError'
            and str(failure.get('error', '')).startswith(tuple(f'HTTP Error {code}:' for code in (500, 502, 503, 504)))):
        raise SimulationUnavailable(f"ModelServiceFailure: {failure['error_type']}: {failure.get('error')}")
    raise subprocess.CalledProcessError(agent.returncode, agent.args)


def acquire_initialization(lock, cancelled):
    while not lock.acquire(timeout=.1):
        if cancelled.is_set():
            raise RuntimeError('并发评测已取消')
    if cancelled.is_set():
        lock.release()
        raise RuntimeError('并发评测已取消')


def wait_for_initialization(agent, path, cancelled, timeout=180):
    deadline = time.monotonic() + timeout
    while True:
        if cancelled.is_set():
            raise RuntimeError('并发评测已取消')
        if agent.poll() is not None and agent.returncode:
            agent_failure(agent, path.parent.parent)
        try:
            initialization = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if agent.poll() is not None:
                raise RuntimeError('agent 已退出但缺少初始化证据')
        else:
            if initialization.get('spawn_collision') is True:
                if initialization.get('collision_phase') != 'spawn':
                    raise ValueError('出生碰撞初始化标注不一致')
                return
            tolerance = float(initialization['position_tolerance_m'])
            stop = initialization['stop']
            if (float(initialization['error_m']) > tolerance or float(stop['speed_m_s']) > .1
                    or abs(float(stop['z_error_m'])) > tolerance):
                raise ValueError(f'初始化门禁未通过：{initialization}')
            return initialization
        if time.monotonic() >= deadline:
            raise SimulationUnavailable(f'agent 初始化未在 {timeout:g}s 内完成：{path}')
        cancelled.wait(.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/uavon_random50_stratified_seed20260811.json'))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--sample-manifest', type=Path, help='显式使用新的trainset抽样清单；默认仍固定random50')
    selection.add_argument('--benchmark-manifest', type=Path, help='官方完整 test 健康集；不抽样')
    selection.add_argument('--test-sample-manifest', type=Path, help='独立健康test随机抽样清单，不替换旧random50/37')
    parser.add_argument('--scene-batch-size', type=int, default=0, help='每个UE任务组最多几条；0为每场景一组')
    parser.add_argument('--scene-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-selection', action='store_true',
                        help='从完整trainset清单选择已知病例，单独标为诊断，禁止复用正式成绩')
    parser.add_argument('--episode-id', action='append', default=[], help='可重复；省略则健康37条')
    parser.add_argument('--allow-v1-problem-scenes', action='store_true', help='仅历史诊断；不传episode-id则原50条')
    parser.add_argument('--port', type=int, default=30310)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpus', help='逗号分隔的GPU编号；省略时使用--gpu')
    parser.add_argument('--isolate-gpu', action='store_true', help='每个 UE 仅暴露其指定物理 GPU，适用于共享服务器')
    parser.add_argument('--workers', type=int, default=1, help='每张GPU的独立仿真环境数')
    parser.add_argument('--gpu-model-endpoints', type=Path, help='同模型多卡服务映射：GPU到base_url/model；仅改变资源路由')
    parser.add_argument('--agent-module',choices=('lightning_vln.adapters.uavon.tuning.cli',),
                        default='lightning_vln.adapters.uavon.tuning.cli',help='默认保持原两阶段导航入口')
    parser.add_argument('--resume-from', type=Path, help='接续同设置下已结算任务，写入新的output目录')
    parser.add_argument('--resume-recovery-upgrade', action='store_true',
                        help='显式允许已知恢复分类或v4→v5有界重建升级，仍严格验证其他配置')
    parser.add_argument('--resume-resource-relocation', action='store_true',
                        help='显式允许服务地址和硬件描述变化；模型、提示、动作及评分必须相同')
    parser.add_argument('--resume-spawn-collision-fix', action='store_true', help='显式接续修复前已正常结算任务，仅将出生碰撞从报错改为单任务失败')
    parser.add_argument('--scene-restarts', type=int, help='仿真故障最多重建0—5次；省略保持既有策略')
    parser.add_argument('--restart-backoff', type=float, help='重建退避基础秒数；本轮支持10秒，最多30秒')
    parser.add_argument('--isolated-model-recovery',action='store_true',help='模型HTTP有界重试并按单任务隔离，不因模型超时重建UE')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('agent_args', nargs=argparse.REMAINDER, help='-- 后传入 CLI 模型/硬件/运动参数')
    args = parser.parse_args()
    if args.scene_restarts is not None or args.restart_backoff is not None:
        if args.scene_restarts is None or not 0 <= args.scene_restarts <= 5 or args.restart_backoff != 10:
            parser.error('显式重建策略须同时提供 --scene-restarts 0至5 和 --restart-backoff 10')
        RECOVERY_POLICY.update(version=5, scene_restarts=args.scene_restarts,
                               restart_backoff_s=10, max_restart_backoff_s=30, model_transport_recoverable=True)
    if args.isolated_model_recovery:
        if args.scene_restarts is None:parser.error('单任务模型恢复须显式保留场景重建策略')
        RECOVERY_POLICY.update(version=6,model_request_attempts=3,model_retry_timeout_s=180,
                               model_retry_backoff_s=[5,15],model_fault_isolation='episode')
    if args.resume_recovery_upgrade and not args.resume_from:
        parser.error('恢复分类升级必须提供 --resume-from')
    if args.resume_resource_relocation and not args.resume_from:
        parser.error('资源迁移必须提供 --resume-from')
    if args.resume_spawn_collision_fix and not args.resume_from:
        parser.error('出生碰撞修复接续必须提供 --resume-from')
    try:
        gpus = [int(v) for v in args.gpus.split(',')] if args.gpus else [args.gpu]
    except ValueError:
        parser.error('gpus 必须为逗号分隔的整数')
    if args.workers < 1 or min(gpus) < 0 or len(set(gpus)) != len(gpus) or args.scene_batch_size < 0:
        parser.error('workers必须为正，GPU编号必须非负且不重复')
    slots = [{'gpu': gpu, 'port': args.port+i} for i, gpu in enumerate(
        gpu for _ in range(args.workers) for gpu in gpus)]
    if not 1 <= args.port <= slots[-1]['port'] <= 65535:
        parser.error('并发端口超出有效范围')
    agent_args = args.agent_args[1:] if args.agent_args[:1] == ['--'] else args.agent_args
    if args.isolated_model_recovery and '--isolated-model-recovery' not in agent_args:
        agent_args=[*agent_args,'--isolated-model-recovery']
    endpoints = json.loads(args.gpu_model_endpoints.read_text()) if args.gpu_model_endpoints else None
    if endpoints is not None:
        if set(endpoints) != {str(g) for g in gpus} or any(
                not isinstance(v.get('base_url'), str) or not isinstance(v.get('model'), str)
                for v in endpoints.values()):
            parser.error('模型服务映射必须恰好覆盖全部GPU并提供base_url/model')
    reserved = {'--dataset', '--scene', '--output', '--metrics-output', '--limit', '--episode-id', '--airsim-port', '--airsim-host', '--allow-v1-problem-scenes'}
    if any(a.split('=')[0] in reserved for a in agent_args):
        parser.error('dataset/scene/output/limit/episode-id/AirSim地址由批量入口管理')
    if args.diagnostic_selection and (not args.sample_manifest or not args.episode_id or args.resume_from):
        parser.error('诊断选择须提供trainset清单和episode-id，且不能resume')
    sampling = None
    benchmark = None
    scene_scripts = SCENES
    if args.test_sample_manifest:
        if args.episode_id or args.diagnostic_selection or args.allow_v1_problem_scenes:
            parser.error('新test抽样须按完整固定清单运行，不接受临时挑选或问题场景')
        tasks, benchmark = load_test_sample(args.dataset, args.test_sample_manifest)
        excluded_ids = benchmark['excluded_ids']
    elif args.benchmark_manifest:
        if args.episode_id or args.diagnostic_selection or args.allow_v1_problem_scenes:
            parser.error('完整test必须运行清单全部健康任务；不允许临时筛选或恢复问题场景')
        tasks, benchmark = load_full_testset(args.dataset, args.benchmark_manifest)
        excluded_ids = benchmark['excluded_ids']
    elif args.sample_manifest:
        if args.allow_v1_problem_scenes or (args.episode_id and not args.diagnostic_selection):
            parser.error('trainset抽样必须按完整清单运行，不允许临时筛选或补抽')
        tasks, sampling = load_train_sample(args.dataset, args.sample_manifest)
        if args.diagnostic_selection:
            requested = set(args.episode_id)
            if len(requested) != len(args.episode_id) or not requested <= {str(t['episode_id']) for t in tasks}:
                parser.error('诊断任务身份重复或不在原始抽样清单中')
            tasks = [t for t in tasks if str(t['episode_id']) in requested]
        excluded_ids = sampling['excluded_ids']
        scene_scripts = TRAIN_SCENES
    else:
        tasks = selected_tasks(args.dataset, args.episode_id, args.allow_v1_problem_scenes)
        excluded_ids = [str(t['episode_id']) for t in selected_tasks(args.dataset, [], True)
                        if t['map_name'] in PROBLEM_SCENES]
    scenes = list(dict.fromkeys(t['map_name'] for t in tasks))
    frame_path = None
    for i,arg in enumerate(agent_args):
        if arg=='--frame-manifest':
            if i+1>=len(agent_args):parser.error('frame-manifest缺少路径')
            frame_path=Path(agent_args[i+1])
        elif arg.startswith('--frame-manifest='):
            frame_path=Path(arg.split('=',1)[1])
    if frame_path:
        if not sampling:parser.error('当前坐标清单只支持trainset')
        for scene in scenes:load_scene_offset(frame_path,scene)
    for scene in scenes:
        if not (args.scene_root/scene_scripts[scene]).is_file():
            parser.error('场景不存在：'+str(args.scene_root/scene_scripts[scene]))
    if '--motion-options' in agent_args and '--current-target-evidence' not in agent_args:
        parser.error('--motion-options 需要 --current-target-evidence')
    if any(flag in agent_args for flag in ('--initial-yaw90', '--ascent-guard')) and '--current-target-evidence' not in agent_args:
        parser.error('SimpleUAV主方法需要 --current-target-evidence')
    prompt_version = METHOD_VERSION
    plan = {'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(), 'prompt_version': prompt_version,
            'collision_detection_protocol': COLLISION_PROTOCOL,
            'spawn_collision_policy': SPAWN_COLLISION_POLICY,
            'gpu_model_endpoints': endpoints,
            'frame_manifest_sha256': hashlib.sha256(frame_path.read_bytes()).hexdigest() if frame_path else None,
            'split': 'trainset' if sampling else 'valset', 'sampling': sampling,
            'benchmark': benchmark,
            'diagnostic_selection': args.diagnostic_selection,
            'recovery_policy': RECOVERY_POLICY,
            'selected_ids': [str(t['episode_id']) for t in tasks], 'scenes': scenes,
            'gpu': args.gpu, 'port': args.port, 'agent_args': agent_args,
            'slots': slots, 'workers_per_gpu': args.workers, 'isolate_gpu': args.isolate_gpu,
            'protocol': PROTOCOL, 'upstream_commit': UPSTREAM_COMMIT, 'official_comparable': False,
            'historical_problem_scenes': args.allow_v1_problem_scenes,
            'excluded_ids': [] if args.allow_v1_problem_scenes else excluded_ids,
            'exclusion_reason': 'UAV-ON 1.0 depth/visibility issues', 'expected_tasks': len(tasks)}
    if args.agent_module != 'lightning_vln.adapters.uavon.tuning.cli':plan.update(agent_module=args.agent_module,protocol=prompt_version)
    reused = reusable_results(args.resume_from, plan, tasks, args.resume_recovery_upgrade,
                              args.resume_resource_relocation, args.resume_spawn_collision_fix) if args.resume_from else []
    if args.resume_spawn_collision_fix:
        plan['spawn_collision_policy_upgrade'] = dict(source=str(args.resume_from.resolve()),
                                                      policy=SPAWN_COLLISION_POLICY)
    if args.resume_resource_relocation:
        plan['resource_relocation'] = dict(source=str(args.resume_from.resolve()),
                                           allowed_fields=['base-url', 'model-hardware', 'sim-hardware'])
    if args.resume_recovery_upgrade:
        previous = json.loads((args.resume_from/'manifest.json').read_text())
        plan['resume_recovery_upgrade'] = {'source': str(args.resume_from.resolve()),
                                           'from_version': previous['recovery_policy']['version'],
                                           'to_version': RECOVERY_POLICY['version'],
                                           'reason': 'explicit infrastructure recovery upgrade; navigation, sensor limits and scoring unchanged'}
    plan['reused_episodes'] = {r['episode_id']: str(path) for r, path in reused}
    reused_ids = set(plan['reused_episodes'])
    remaining = [t for t in tasks if str(t['episode_id']) not in reused_ids]
    scenes = list(dict.fromkeys(t['map_name'] for t in remaining))
    # 长场景先启动，减少最后只剩一个环境工作的尾部时间。
    scenes.sort(key=lambda s: sum(t['map_name'] == s for t in remaining), reverse=True)
    plan['pending_scenes'] = scenes
    jobs = scene_jobs(remaining, args.scene_batch_size)
    plan['scene_batch_size'] = args.scene_batch_size
    plan['task_groups'] = {name: [str(t['episode_id']) for t in rows] for name, rows in jobs.items()}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'manifest.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    dataset = (args.output/'tasks.json').resolve()
    dataset.write_text(json.dumps(remaining, ensure_ascii=False, indent=2))
    results = [r for r, _ in reused]
    expected = {str(t['episode_id']): t for t in tasks}
    for result, episode in reused:
        if result.get('record_mode') == 'metrics-only':
            from lightning_vln.adapters.uavon.evaluation import export_metric_scalars
            export_metric_scalars(expected[str(result['episode_id'])], result, args.output/'uavon_metrics')
        elif result.get('record_mode') != 'results-only':
            export_episode(expected[str(result['episode_id'])], episode, result, args.output/'uavon_metrics')
    (args.output/'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, interrupt)
    lock = threading.Lock()
    initialization_lock = threading.Lock()
    def status(scene, state):
        with lock, (args.output/'status.tsv').open('a') as f:
            f.write(f'{time.time():.3f}\t{scene}\t{state}\n')
    def run_attempt(scene, slot, cancelled, root, pending):
        process = agent = None
        initialization_owned = False
        def check_cancelled():
            if cancelled.is_set():
                raise RuntimeError('并发评测已取消')
        try:
            check_cancelled()
            acquire_initialization(initialization_lock, cancelled)
            initialization_owned = True
            check_cancelled()
            attempt_dataset = (root/'tasks.json').resolve()
            attempt_dataset.write_text(json.dumps(pending, ensure_ascii=False, indent=2))
            check_scene_port(slot['port'])  # 不接管已有场景。
            command = [sys.executable, '-m', 'lightning_vln.adapters.uavon.scene', '--script', str(args.scene_root/scene_scripts[scene]),
                       '--output', str(root/'scene'), '--port', str(slot['port']), '--gpu', str(slot['gpu']),
                       *(['--isolate-gpu'] if args.isolate_gpu else [])]
            status(scene, 'starting')
            with (root/'scene.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    deadline = time.monotonic()+180
                    while True:
                        check_cancelled()
                        if process.poll() is not None:
                            raise SimulationUnavailable(f'{scene} 场景启动退出：{process.returncode}')
                        try:
                            rpc = RPC('127.0.0.1', slot['port'], 2)
                            try:
                                if rpc.call('ping'):
                                    break
                            finally:
                                rpc.close()
                        except (OSError, RuntimeError):
                            pass
                        if time.monotonic() >= deadline:
                            raise SimulationUnavailable(f'{scene} RPC 未就绪')
                        cancelled.wait(.5)
                    status(scene, 'running')
                    with (root/'agent.log').open('w') as agent_log:
                        slot_args = routed_agent_args(agent_args, endpoints, slot['gpu'])
                        agent = subprocess.Popen([sys.executable, '-u', '-m', args.agent_module,
                                        '--dataset', str(attempt_dataset), '--scene', scene, '--limit', '0',
                                        '--airsim-port', str(slot['port']), '--output', str(root/'agent'),
                                        '--metrics-output', str(args.output/'uavon_metrics'),
                                        *(['--allow-v1-problem-scenes'] if args.allow_v1_problem_scenes else []),
                                        *slot_args], stdout=agent_log, stderr=subprocess.STDOUT, start_new_session=True)
                        wait_for_initialization(agent, root/'agent/episode_0000/initialization.json', cancelled)
                        status(root.relative_to(args.output), 'initialized')
                        initialization_lock.release()
                        initialization_owned = False
                        while agent.poll() is None:
                            check_cancelled()
                            if process.poll() is not None:
                                raise SimulationUnavailable(f'{scene} 运行中退出：{process.returncode}')
                            cancelled.wait(.2)
                        if agent.returncode:
                            agent_failure(agent, root/'agent')
                finally:
                    if agent is not None and agent.poll() is None:
                        stop_scene(agent)
                    stop_scene(process)
                    status(scene, 'scene_stopped')
        except BaseException as error:
            (root/'failure.json').write_text(json.dumps({'error': repr(error)}))
            status(scene, 'failed')
            raise
        finally:
            if initialization_owned:
                initialization_lock.release()
    def commit(scene_results):
        with lock:
            results.extend(scene_results)
            checkpoint = args.output/'results.json.tmp'
            checkpoint.write_text(json.dumps(results, ensure_ascii=False, indent=2))
            checkpoint.replace(args.output/'results.json')
    def run_scene(job, slot, cancelled):
        pending = jobs[job]
        scene = pending[0]['map_name']
        recover_scene(args.output/job, pending,
                      lambda root, pending: run_attempt(scene, slot, cancelled, root, pending),
                      commit, RECOVERY_POLICY['scene_restarts'], cancelled=cancelled)
        status(job, 'completed')
    try:
        failures = parallel_scenes(jobs, slots[:len(jobs)], run_scene)
        if failures:
            completed_ids = {str(r['episode_id']) for r in results}
            incomplete = {'status': 'incomplete', 'expected_tasks': len(tasks),
                          'total_finished': len(results), 'failed_scenes': failures,
                          'missing_ids': [str(t['episode_id']) for t in tasks
                                          if str(t['episode_id']) not in completed_ids]}
            (args.output/'incomplete.json').write_text(json.dumps(incomplete, ensure_ascii=False, indent=2))
            raise RuntimeError('存在未结算任务；详见 incomplete.json，不发布完整评分')
        summary = aggregate(tasks, results)
        summary["diagnostic_only"] = args.diagnostic_selection
        if summary.get('record_mode') == 'results-only':
            (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            status('subset', 'completed')
            return
        official = run_official_metrics(args.output/'uavon_metrics', len(tasks))
        # 官方stdout有四舍五入，精确值保留在summary中。
        for metric in ('sr', 'osr', 'spl', 'dts_m'):
            if abs(summary[metric]-official[metric]) > (.000501 if metric == 'dts_m' else .000051):
                raise ValueError(f'官方评分与适配层不一致：{metric}')
        summary['official_script_metrics'] = official
        (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        status('subset', 'completed')
    except BaseException as error:
        state = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        (args.output/'failure.json').write_text(json.dumps({'status': state, 'error': str(error)}, ensure_ascii=False))
        status('subset', state)
        raise


if __name__ == '__main__':
    main()
