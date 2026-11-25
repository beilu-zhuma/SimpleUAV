"""仅保护同一水平位置的重复上升探索，不改变原停滞、STOP或几何规则。"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Options:
    initial_yaw90: bool = False
    ascent_guard: bool = False
    max_up_explores: int = 2
    reset_horizontal_m: float = 3.0

    def __post_init__(self):
        if type(self.max_up_explores) is not int or self.max_up_explores < 1:
            raise ValueError("max_up_explores必须为正整数")
        if not np.isfinite(self.reset_horizontal_m) or self.reset_horizontal_m <= 0:
            raise ValueError("reset_horizontal_m必须为有限正数")

    def manifest(self):
        return {**asdict(self), "initial_rotation_budget": "one_of_total_decisions",
                "guard_scope": "vertical_explore_up_only",
                "height_envelope": False}


class AscentGuard:
    def __init__(self, start, options):
        self.options = options
        self.start_z = float(start[2])
        self.anchor = np.asarray(start[:2], dtype=float).copy()
        self.up_count = 0
        self.up_gain_m = 0.0
        self.position = np.asarray(start, dtype=float).copy()
        self.blocks = 0
        self.resets = 0

    def observe(self, position):
        self.position = np.asarray(position, dtype=float).copy()
        if np.linalg.norm(self.position[:2]-self.anchor) >= self.options.reset_horizontal_m:
            if self.up_count:
                self.resets += 1
            self.anchor = self.position[:2].copy()
            self.up_count = 0
            self.up_gain_m = 0.0

    @property
    def blocked(self):
        return self.options.ascent_guard and self.up_count >= self.options.max_up_explores

    def validate(self, decision):
        if self.blocked and decision.action == "vertical_explore" and decision.vertical_direction == "up":
            self.blocks += 1
            raise ValueError("重复上升保护：同一水平位置已用完上升探索预算，须实际水平换位后恢复")
        return decision

    def record(self, observation, decision, feedback):
        end = np.asarray(feedback["position_ned_m"], dtype=float)
        rise = float(observation.position[2]-end[2])
        if (decision is not None and decision.action == "vertical_explore"
                and decision.vertical_direction == "up" and rise > 0.05):
            if self.up_count == 0:
                self.anchor = np.asarray(observation.position[:2], dtype=float).copy()
            self.up_count += 1
            self.up_gain_m += rise
        self.observe(end)

    def feedback(self):
        return {"up_explores_at_anchor": self.up_count,
                "up_explores_limit": self.options.max_up_explores,
                "up_gain_at_anchor_m": self.up_gain_m,
                "rise_from_spawn_m": self.start_z-float(self.position[2]),
                "horizontal_from_anchor_m": float(np.linalg.norm(self.position[:2]-self.anchor)),
                "reset_horizontal_m": self.options.reset_horizontal_m,
                "up_explore_allowed": not self.blocked,
                "guidance": ("同一水平位置反复上升没有证明搜索取得收益。上升探索预算已用完，"
                             "请选择其他合法动作；实际水平换位后恢复资格。"
                             if self.blocked else
                             "上升产生位移不等于搜索有收益。再次升高前检查新图是否改善目标证据或视野，"
                             "不要重复沿用上次升高理由。")}
