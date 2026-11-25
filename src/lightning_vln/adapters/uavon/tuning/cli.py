"""独立参数对照入口，保持已验证方法与默认行为不变。"""

import argparse
import hashlib
import importlib.metadata
import json
import io
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from lightning_vln.core.agent import hold_after_motion, Limits
from lightning_vln.adapters.uavon.minimal.agent import run_agent
from lightning_vln.adapters.uavon.minimal.policy import Options
from lightning_vln.adapters.uavon.tuning import METHOD_VERSION
from lightning_vln.core.geometry import vector
from lightning_vln.core.frames import DatasetFrameWorld, load_scene_offset
from lightning_vln.core.errors import SimulationUnavailable
from lightning_vln.core.model_transport import ModelTransportFailure, RECOVERY_POLICY as MODEL_TRANSPORT_POLICY
from lightning_vln.adapters.uavon.evaluation import PROBLEM_SCENES, PROTOCOL, UPSTREAM_COMMIT, export_episode, score_episode, score_success, export_metric_scalars, run_official_metrics
from lightning_vln.core.simulator import AirSimWorld, COLLISION_PROTOCOL, SPAWN_COLLISION_POLICY
from lightning_vln.core.vlm import PROMPT_VERSION
from lightning_vln.adapters.uavon.tuning.vlm import TuningVLM as VLM
from lightning_vln.core.evidence import PROMPT_VERSION as EVIDENCE_PROMPT_VERSION
from lightning_vln.core.motion_options import PROMPT_VERSION as MOTION_OPTIONS_PROMPT_VERSION


def tasks_from_args(args) -> list[dict]:
    if args.instruction:
        task = {"episode_id": "manual", "instruction": args.instruction}
        if args.start_position is not None:
            task["start_pose"] = {"start_position": vector(args.start_position, 3).tolist(),
                                  "start_quaternionr": vector(args.start_quaternion, 4).tolist()}
        return [task]
    if not args.scene:
        raise ValueError("使用 dataset 时必须提供 --scene，确认当前启动的是同一场景")
    if args.scene in PROBLEM_SCENES and not args.allow_v1_problem_scenes:
        raise ValueError("UAV-ON 1.0 问题场景默认排除；历史诊断需 --allow-v1-problem-scenes")
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("UAV-ON dataset 必须是 JSON 列表")
    rows = [row for row in rows if row["map_name"] == args.scene]
    if args.episode_id is not None:
        rows = [row for row in rows if str(row["episode_id"]) == args.episode_id]
    if not rows:
        raise ValueError("没有匹配的任务")
    return rows[:args.limit] if args.limit else rows


def task_instruction(task: dict) -> str:
    if task.get("instruction"):
        return str(task["instruction"])
    name = task.get("true_name") or task.get("object_name") or "目标物体"
    return f"寻找 {name}。目标描述：{task.get('description', '')}"


def main():
    parser = argparse.ArgumentParser(description="独立UAV-ON方法：可分别启用开局90度观察和重复上升保护")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--instruction")
    source.add_argument("--dataset", type=Path)
    parser.add_argument("--scene", help="当前运行的场景名；dataset 模式必填")
    parser.add_argument("--episode-id")
    parser.add_argument("--allow-v1-problem-scenes", action="store_true", help="仅用于显式历史诊断")
    parser.add_argument("--metrics-output", type=Path, help="由串行入口管理的官方日志根目录")
    parser.add_argument("--limit", type=int, default=1, help="0 表示匹配场景的全部任务")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-hardware", required=True, help="模型 GPU 型号/数量/显存；远端需手动确认")
    parser.add_argument("--sim-hardware", required=True, help="仿真 GPU 型号/数量/显存")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thinking", action="store_true", help="默认关闭 thinking")
    parser.add_argument("--current-target-evidence", action="store_true",
                        help="每轮先独立核查当前四图目标，再将目标动作绑定本轮证据；增加一次模型调用")
    parser.add_argument("--model-timeout", type=float, default=90)
    parser.add_argument('--isolated-model-recovery',action='store_true',help='模型HTTP有界重试，耗尽只延期该任务并继续后续任务')
    parser.add_argument("--motion-options", action="store_true", help="连续无进展时约束为当前深度可尝试移动；需要当前目标证据")
    parser.add_argument("--airsim-host", default="127.0.0.1")
    parser.add_argument("--airsim-port", type=int, default=30100)
    parser.add_argument("--vehicle", default="Drone_1")
    parser.add_argument("--start-position", type=float, nargs=3, help="手动任务的空中出生点，NED 米")
    parser.add_argument("--start-quaternion", type=float, nargs=4, default=[0, 0, 0, 1], help="xyzw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spawn-up", type=float, default=2, help="出生点上调米数；NED z 减去此值")
    parser.add_argument('--frame-manifest',type=Path,help='离线独立对象验证的train场景坐标清单；原任务坐标/标签不改写')
    parser.add_argument("--success-distance", type=float, default=20, help="官方1.0阈值；改动必须单独标明")
    for name, default in asdict(Limits()).items():
        flags = ["--" + name.replace("_", "-")]
        if name == "max_step_len":
            flags.append("--max-leg")  # 旧启动命令兼容；manifest 仅记录新名称。
        parser.add_argument(*flags, dest=name, type=type(default), default=default)
    parser.add_argument("--initial-yaw90", action="store_true", help="启用首动作原地转90度，计入总决策预算；默认关闭")
    parser.add_argument("--ascent-guard", action="store_true", help="同一水平位置重复上升保护")
    parser.add_argument("--max-up-explores", type=int, default=2)
    parser.add_argument("--reset-horizontal-m", type=float, default=3.0)
    parser.add_argument("--text-memory-mode", choices=("on", "off"), default="on", help="仅控制文字搜索摘要是否进入导航输入，历史窗口独立控制")
    parser.add_argument('--record-mode', choices=('full', 'results-only', 'metrics-only'), default='full')
    parser.add_argument('--record-step-timing', action='store_true',
                        help='额外保存逐决策计时标量，不保存图像、模型原文或位姿')
    args = parser.parse_args()
    try:
        options = Options(args.initial_yaw90, args.ascent_guard, args.max_up_explores, args.reset_horizontal_m)
    except ValueError as error:
        parser.error(str(error))
    if (options.initial_yaw90 or options.ascent_guard) and not args.current_target_evidence:
        parser.error("小改动实验须保留原两阶段方法，提供 --current-target-evidence")
    if args.motion_options and not args.current_target_evidence:
        parser.error("--motion-options 需要 --current-target-evidence")
    if args.frame_manifest and (not args.dataset or not args.scene):
        parser.error('场景坐标清单只用于指定scene的dataset评测')
    frame_offset=load_scene_offset(args.frame_manifest,args.scene) if args.frame_manifest else None
    if args.dataset and args.start_position is not None:
        parser.error("dataset 使用任务出生点；--start-position 仅适用于 --instruction")
    if (args.limit < 0 or args.max_tokens <= 0 or not np.isfinite(args.model_timeout)
            or args.model_timeout <= 0):
        parser.error("limit 必须非负，max-tokens/model-timeout 必须为正")
    if not np.isfinite([args.spawn_up, args.success_distance]).all() or args.success_distance <= 0:
        parser.error("spawn-up 必须有限，success-distance 必须为有限正数")
    limits = Limits(**{name: getattr(args, name) for name in asdict(Limits())})
    tasks = tasks_from_args(args)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
        dirty = None
    versions = {}
    for package in ("numpy", "Pillow", "msgpack"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    manifest = {"git_commit": commit, "git_dirty": dirty,
                "collision_detection_protocol": COLLISION_PROTOCOL,
                "spawn_collision_policy": SPAWN_COLLISION_POLICY,
                'frame_manifest_sha256': hashlib.sha256(args.frame_manifest.read_bytes()).hexdigest() if args.frame_manifest else None,
                'dataset_to_airsim_offset_ned_m': frame_offset.tolist() if frame_offset is not None else None,
                "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest() if args.dataset else None,
                "args": {k: str(v) if isinstance(v, Path) else v
                                                for k, v in vars(args).items()},
                "limits": asdict(limits), "versions": versions,
                "protocol": METHOD_VERSION, "base_protocol": PROTOCOL, "upstream_commit": UPSTREAM_COMMIT,
                "method_options": dict(options.manifest(), text_memory_mode=args.text_memory_mode),
                "prompt_version": METHOD_VERSION,
                "official_comparable": False}
    source_root = Path(__file__).resolve().parents[3]
    manifest["source_sha256"] = {
        str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source_root.glob("*.py")) + sorted((source_root/"uavon_minimal").glob("*.py")) + sorted(Path(__file__).resolve().parent.glob("*.py"))}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    world = None
    index = None
    active_error = None
    results = []
    deferred = []
    try:
        world = AirSimWorld(args.airsim_host, args.airsim_port, args.vehicle)
        if frame_offset is not None:
            world = DatasetFrameWorld(world,frame_offset)
        for index, task in enumerate(tasks):
            trace = None
            try:
                output = args.output / f"episode_{index:04d}"
                output.mkdir()
                identity = (dict(episode_id=str(task['episode_id']), task_sha256=hashlib.sha256(
                    json.dumps(task, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
                    if args.record_mode != 'full' else task)
                (output / "task.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
                start = task.get("start_pose")
                if start:
                    position = vector(start["start_position"], 3)
                    position[2] -= args.spawn_up
                    world.prepare(position, start["start_quaternionr"])
                else:
                    world.prepare()
                if getattr(world, "last_prepare", None):
                    initial = world.last_prepare
                    spawn_collision = bool(initial.get('spawn_collision', False))
                    if args.record_mode != 'full':
                        initial = {k:initial[k] for k in ('position_tolerance_m','error_m','stop')}
                        initial['stop'] = {k:initial['stop'][k] for k in ('speed_m_s','z_error_m')}
                    initial = dict(initial, spawn_collision=spawn_collision,
                                   collision_phase='spawn' if spawn_collision else None)
                    (output / "initialization.json").write_text(json.dumps(initial, indent=2))
                model = VLM(args.base_url, args.model, None if args.record_mode != 'full' else output / "model.jsonl",
                            os.environ.get("OPENAI_API_KEY", ""), args.model_timeout,
                            args.max_tokens, args.thinking, args.current_target_evidence, args.motion_options)
                model.isolated_model_recovery = args.isolated_model_recovery
                model.minimal_options = options
                model.text_memory_mode = args.text_memory_mode
                def progress(decisions):
                    if decisions == 1:
                        (output/'navigation-started.json').write_text(json.dumps(dict(actual_navigation=True)))
                        print('NAVIGATION_STARTED', task['episode_id'], flush=True)
                trace = io.StringIO() if args.record_mode == 'metrics-only' else None
                timing_options = {'step_timing_path': output/'step-timings.jsonl'} if args.record_step_timing else {}
                summary = run_agent(world, model, task_instruction(task), output, limits, options,
                    record_details=args.record_mode == 'full', progress=progress if args.record_mode != 'full' else None,
                    metric_trace=trace, **timing_options)
                if args.record_step_timing:
                    timing_rows = [json.loads(line) for line in (output/'step-timings.jsonl').read_text().splitlines()]
                    if [row['iteration'] for row in timing_rows] != list(range(summary['decisions'])):
                        raise ValueError('逐步计时条数或步号与实际决策不一致')
                    summary['step_timing'] = {'file': 'step-timings.jsonl', 'rows': len(timing_rows),
                                              'schema': 'decision_timing_v1', 'unit': 's'}
                summary.update(method_version=METHOD_VERSION, method_options=dict(options.manifest(), text_memory_mode=args.text_memory_mode))
                summary['collision_detection_protocol'] = COLLISION_PROTOCOL
                summary['spawn_collision'] = bool(getattr(world, 'last_prepare', {}).get('spawn_collision', False))
                summary['collision_phase'] = ('spawn' if summary['spawn_collision'] else
                                              'navigation' if summary['status'] == 'collision' else None)
                if callable(getattr(world, 'collision_record', None)):
                    summary['collision'] = world.collision_record()
                if args.record_mode == 'results-only':
                    summary = dict(status=summary['status'], success=score_success(task, summary, args.success_distance),
                                   record_mode='results-only', spawn_collision=summary['spawn_collision'],
                                   collision_phase=summary['collision_phase'])
                elif args.record_mode == 'metrics-only':
                    rows = [json.loads(line) for line in trace.getvalue().splitlines()]
                    summary.update(score_episode(task, output, summary, args.success_distance, trajectory_rows=rows))
                    trace.close()
                    summary.pop('start_position_ned_m', None); summary.pop('position_ned_m', None)
                    summary['record_mode'] = 'metrics-only'
                else:
                    summary.update(score_episode(task, output, summary, args.success_distance))
                if args.isolated_model_recovery:
                    summary['model_transport'] = dict(http_attempts=getattr(model,'http_attempts',0),policy=MODEL_TRANSPORT_POLICY)
                summary["metric_protocol"] = METHOD_VERSION
                evaluation_path = output/"evaluation.json"
                if evaluation_path.exists():
                    evaluation = json.loads(evaluation_path.read_text())
                    evaluation.update(metric_protocol=METHOD_VERSION, base_protocol=PROTOCOL,
                                      method_options=dict(options.manifest(), text_memory_mode=args.text_memory_mode))
                    evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2)+"\n")
                summary["episode_id"] = str(task["episode_id"])
                if summary.get("spl") is not None:
                    if args.record_mode == 'metrics-only':
                        export_metric_scalars(task, summary, args.metrics_output or args.output / "uavon_metrics")
                    else:
                        export_episode(task, output, summary, args.metrics_output or args.output / "uavon_metrics")
                (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                results.append(summary)
                # 逐任务持久保存已完成结果；后续失败不会抹掉前面的结算。
                checkpoint = args.output / "results.json.tmp"
                checkpoint.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                checkpoint.replace(args.output / "results.json")
            except ModelTransportFailure as error:
                # 不伪造导航失败、不重放运动；保留此任务未结算身份，继续本UE下一条。
                hold_after_motion(world)
                item=dict(episode_id=str(task['episode_id']),error_type=type(error).__name__,
                          error=str(error),http_attempts=getattr(model,'http_attempts',error.attempts),
                          failed_request_attempts=error.attempts,transport_policy=MODEL_TRANSPORT_POLICY,
                          status='deferred_model_transport',not_scored=True)
                (output/'deferred.json').write_text(json.dumps(item,ensure_ascii=False,indent=2))
                deferred.append(item)
                checkpoint=args.output/'deferred.json.tmp'
                checkpoint.write_text(json.dumps(deferred,ensure_ascii=False,indent=2))
                checkpoint.replace(args.output/'deferred.json')
                print('TASK_DEFERRED_MODEL',task['episode_id'],flush=True)
                if trace is not None:trace.close()
                continue
        if args.metrics_output is None and results and all(r.get("spl") is not None for r in results):
            run_official_metrics(args.output / "uavon_metrics", len(results))
    except BaseException as error:
        active_error = error
        failure = {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                   "episode_index": index, "error_type": type(error).__name__, "error": str(error),
                   "recoverable_simulation": isinstance(error, SimulationUnavailable),
                   "sensor": getattr(world, "last_observe", {}),
                   "prepare": getattr(world, "last_prepare", {}),
                   "stop": getattr(world, "last_stop", {})}
        if args.record_mode != 'full':
            for key in ('sensor','prepare','stop'):failure.pop(key,None)
        (args.output / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    finally:
        if world is not None:
            try:
                hold_after_motion(world)
            except Exception as cleanup_error:
                (args.output / "cleanup_failure.json").write_text(json.dumps({"error": repr(cleanup_error)}))
                if active_error is None:
                    raise
            finally:
                if hasattr(world, "close"):
                    world.close()


if __name__ == "__main__":
    main()
