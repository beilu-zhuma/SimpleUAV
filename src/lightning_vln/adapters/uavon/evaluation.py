"""结束后评分与官方日志适配；本模块绝不参与模型或运动决策。"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

UPSTREAM_COMMIT = "d7ccf795033a577549d12fae173c149ef38d08a7"
PROBLEM_SCENES = frozenset({"CityPark_test", "BrushifyRoad_test", "Slum_test", "DownTown_test"})
PROBLEM_SCENES |= frozenset({"CityPark_TrainSets", "BrushifyRoad_train", "Slum_train", "DownTown_train"})
PROTOCOL = "simple_agent_physics_v6_directional_explore"


def score_success(task: dict, summary: dict, threshold: float = 20) -> bool:
    """仅SR模式使用相同终点条件；不写入位置、轨迹或其他指标。"""
    targets = np.asarray(task['pose'], dtype=float)
    if targets.ndim == 1:
        targets = targets.reshape(1, -1)
    if targets.ndim != 2 or targets.shape[1] != 3 or not len(targets) or not np.isfinite(targets).all():
        raise ValueError('GT pose 必须为非空 N×3 有限坐标')
    position = np.asarray(summary['position_ned_m'], dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('SR评分终点及阈值必须有效')
    return bool(summary['status'] == 'stop' and np.linalg.norm(targets-position, axis=1).min() <= threshold)


from lightning_vln.core.metrics import sampled_trajectory


def score_episode(task: dict, output: Path, summary: dict, threshold: float = 20, *, trajectory_rows=None) -> dict:
    if "pose" not in task:
        return {}
    targets = np.asarray(task["pose"], dtype=float)
    if targets.ndim == 1:
        targets = targets.reshape(1, -1)
    if targets.ndim != 2 or targets.shape[1] != 3 or not len(targets) or not np.isfinite(targets).all():
        raise ValueError("GT pose 必须为非空 N×3 有限坐标")
    rows = sampled_trajectory(output, trajectory_rows)
    if not np.allclose(rows[0]["position_ned_m"], summary["start_position_ned_m"], atol=1e-6, rtol=0):
        raise ValueError("轨迹起点与实际出生点不一致")
    if not np.allclose(rows[-1]["position_ned_m"], summary["position_ned_m"], atol=1e-6, rtol=0):
        raise ValueError("轨迹终点与任务结算不一致")
    for row in rows:
        row["distance_to_end"] = float(np.linalg.norm(targets-np.asarray(row["position_ned_m"]), axis=1).min())
    endpoints = [r for r in rows if r["event"] == "decision_end"]
    final = rows[-1]["distance_to_end"]
    collision = summary["status"] == "collision"
    success = not collision and summary["status"] == "stop" and final <= threshold
    # 官方在 makeActions 使用 <20，在成功/预算终点使用 <=20；保留此边界。
    oracle = not collision and (success or any(r["distance_to_end"] < threshold for r in endpoints)
                               or (summary["status"] == "decision_limit" and final <= threshold))
    geodesic = task.get("info", {}).get("geodesic_distance")
    if geodesic is not None and (not np.isfinite(geodesic) or geodesic <= 0):
        raise ValueError("geodesic_distance 必须为有限正数；不能用直线距离替代")
    travelled = rows[-1]["move_distance"]
    # 官方 classify_metric 对零路径任务贡献 0；不擅自改成标准 SPL 的 1。
    spl = None if geodesic is None else (float(geodesic/max(travelled, geodesic)) if success and travelled > 0 else 0.)
    result = {"metric_protocol": PROTOCOL, "upstream_commit": UPSTREAM_COMMIT,
              "diagnostic_only": False, "comparable_official_metrics": False,
              "success_distance_m": threshold, "final_distance_m": final, "ne_m": final,
              "min_sampled_distance_m": min(r["distance_to_end"] for r in rows),
              "success": success, "oracle_success": oracle, "spl": spl,
              "geodesic_distance_m": geodesic, "sampled_path_length_m": travelled,
              "path_length_reference": "dataset_original_spawn_geodesic",
              "oracle_sample_basis": "decision_endpoints", "trajectory_samples": len(rows),
              "diagnostic_15m": {"success": not collision and summary["status"] == "stop" and final <= 15,
                                 "oracle_success": not collision and any(r["distance_to_end"] <= 15 for r in rows)}}
    (output / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def export_metric_scalars(task: dict, summary: dict, metrics_root: Path):
    """官方评分所需的单行标量输入；不是逐步轨迹，不伪造位置。"""
    if summary.get('spl') is None:
        raise ValueError('标量评分输入需要原始geodesic')
    scene=str(task['map_name']);episode=str(task['episode_id'])
    if any(Path(v).name!=v or v in ('.','..') for v in (scene,episode)):
        raise ValueError('非法评分任务身份')
    prefix='success_' if summary['success'] else 'oracle_' if summary['oracle_success'] else ''
    root=metrics_root/scene/f'{prefix}{scene}.json'/f'task_{episode}'
    (root/'log').mkdir(parents=True,exist_ok=False)
    (root/'object_description.json').write_text(json.dumps(task,ensure_ascii=False))
    scalar=dict(metric_input_kind='scalar_summary_not_trajectory',distance_to_end=summary['final_distance_m'],
        move_distance=summary['sampled_path_length_m'],frame=summary['decisions'],
        is_collision=summary['status']=='collision',termination_status=summary['status'])
    (root/'log/trajectory.jsonl').write_text(json.dumps(scalar)+'\n')


def export_episode(task: dict, output: Path, summary: dict, metrics_root: Path):
    """官方脚本按 success/oracle/fail 目录分类；GT 只写入独立评分产物。"""
    if summary.get("spl") is None:
        raise ValueError("导出官方评分日志需要原始 geodesic_distance")
    scene = str(task["map_name"])
    episode = str(task["episode_id"])
    if any(Path(v).name != v or v in (".", "..") for v in (scene, episode)):
        raise ValueError("非法场景或任务身份")
    prefix = "success_" if summary["success"] else "oracle_" if summary["oracle_success"] else ""
    root = metrics_root / scene / f"{prefix}{scene}.json" / f"task_{episode}"
    (root / "log").mkdir(parents=True, exist_ok=False)
    (root / "object_description.json").write_text(json.dumps(task, ensure_ascii=False, indent=2))
    samples = sampled_trajectory(output)
    targets = np.asarray(task["pose"], dtype=float).reshape(-1, 3)
    # frame 是决策索引；路径长度来自完整控制采样，而非指令目标点。
    endpoints = [samples[0], *[r for r in samples if r["event"] == "decision_end"], samples[-1]]
    with (root / "log/trajectory.jsonl").open("w") as f:
        for index, row in enumerate(endpoints):
            position = row["position_ned_m"]
            f.write(json.dumps({"frame": index, "is_collision": summary["status"] == "collision" and index == len(endpoints)-1,
                                "action": row.get("action"), "steps_size": None,
                                "termination_status": summary["status"] if index == len(endpoints)-1 else None,
                                "move_distance": row["move_distance"],
                                "distance_to_end": float(np.linalg.norm(targets-position, axis=1).min()),
                                "sensors": {"state": {"position": position}}})+"\n")


def run_official_metrics(root: Path, expected: int) -> dict:
    """执行原样保留的官方脚本；只解析总指标，其150帧终止猜测不适用本协议。"""
    descriptions = list(root.glob("*/*.json/task_*/object_description.json"))
    if len(descriptions) != expected:
        raise ValueError("官方日志任务覆盖不完整")
    script = Path(__file__).resolve().parents[4] / "third_party/UAV-ON/utils/classify_metric.py"
    process = subprocess.run([sys.executable, str(script), "--base_root", str(root)],
                             capture_output=True, text=True, check=True, timeout=60)
    (root / "official_metrics.txt").write_text(process.stdout)
    head = process.stdout.split("===== Termination Type Statistics =====")[0]
    total = int(re.search(r"Total Tasks\s*:\s*(\d+)", head)[1])
    if total != expected:
        raise ValueError("官方脚本统计分母不匹配")
    result = {"total_finished": total, "upstream_commit": UPSTREAM_COMMIT,
              "official_comparable": False, "source": "unmodified_upstream_classify_metric"}
    for key, pattern in {"sr": r"Success Rate \(SR\):\s*([\d.]+)%",
                         "osr": r"Oracle\s+Rate \(OSR\):\s*([\d.]+)%",
                         "spl": r"SPL \(Success \+ Failure\):\s*([\d.]+)%",
                         "dts_m": r"- All\s*:\s*([\d.]+)"}.items():
        result[key] = float(re.search(pattern, head)[1]) / (1 if key == "dts_m" else 100)
    (root / "official_metrics.json").write_text(json.dumps(result, indent=2))
    return result
