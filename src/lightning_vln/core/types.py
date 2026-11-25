"""Agent 的输入与输出；坐标统一为 AirSim 世界 NED，单位米。"""

from dataclasses import dataclass

import numpy as np


VIEWS = {"front": "0", "left": "1", "right": "2", "down": "3"}


@dataclass
class View:
    rgb: bytes
    depth: np.ndarray
    position: np.ndarray
    quaternion: np.ndarray
    fov: float
    timestamp: int


@dataclass
class Observation:
    position: np.ndarray
    views: dict[str, View]


@dataclass
class Decision:
    action: str
    view: str
    bbox: tuple[float, float, float, float]
    reason: str
    memory: str
    vertical_direction: str | None = None
    target_id: str | None = None
