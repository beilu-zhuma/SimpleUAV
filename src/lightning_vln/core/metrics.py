"""基准无关的实际轨迹累计；成功/碰撞评分留在适配器。"""
import json
from pathlib import Path
import numpy as np

def sampled_trajectory(output: Path, trajectory_rows=None) -> list[dict]:
    rows = (trajectory_rows if trajectory_rows is not None else
            [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()])
    positions = np.asarray([r["position_ned_m"] for r in rows], dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(rows) < 2 or not np.isfinite(positions).all():
        raise ValueError("评分需要完整、有序的实际采样轨迹")
    if rows[0]["event"] != "start" or rows[-1]["event"] != "final":
        raise ValueError("轨迹缺少实际出生点或最终位置，不能结算")
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    return [dict(r, move_distance=float(length)) for r, length in zip(rows, lengths)]
