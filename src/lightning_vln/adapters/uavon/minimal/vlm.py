"""复用原两阶段视觉策略，仅补初始观察与重复上升约束。"""

import copy
import json

from lightning_vln.core.context import Context, image_content
from lightning_vln.core.evidence import NAVIGATION_PROMPT, navigation_schema
from lightning_vln.core.motion_options import NAVIGATION_PROMPT as MOTION_PROMPT, bind_motion
from lightning_vln.core.vlm import SCHEMA, SYSTEM_PROMPT, VLM, parse_decision
from lightning_vln.adapters.uavon.minimal import METHOD_VERSION


GUARD_PROMPT = """
ascent_guard是程序维护的实际运动记录，不是模型记忆。上升执行成功不表示搜索有收益。
检查已上升次数及高度；不要只复用“升高改善视野”的旧理由。
up_explore_allowed=false时禁止上升探索；必须改变可执行动作。下降、原地停留或微小
水平挪动不能刷新预算。此规则不提供目标位置，不替代当前目标证据和原安全保护。
"""


class NoPermittedAction(ValueError):
    """原恢复选项经上升保护过滤后为空，不借此退回自由上升。"""


class MinimalContext(Context):
    def __init__(self, instruction, history_turns, guard):
        super().__init__(instruction, history_turns)
        self.guard = guard
        self.initial_overview = None

    def messages(self, system_prompt, observation, iteration):
        if self.guard.options.ascent_guard:
            system_prompt += GUARD_PROMPT
        messages = super().messages(system_prompt, observation, iteration)
        current = messages[-1]["content"]
        if self.guard.options.ascent_guard:
            current.insert(1, {"type": "text", "text": json.dumps(
                {"ascent_guard": self.guard.feedback()}, ensure_ascii=False)})
        if self.initial_overview is not None and iteration == 1:
            previous = self.initial_overview
            overview = [{"type": "text", "text": "以下为开局旋转90度之前的历史观察，"
                         "不是当前视图；只能用作搜索线索，不能复用历史框或候选ID。"}]
            for name in ("front", "left", "right"):
                view = previous.views[name]
                overview.extend([{"type": "text", "text": json.dumps({
                    "initial_view": name, "position_ned_m": previous.position.tolist(),
                    "camera_quaternion_xyzw": view.quaternion.tolist()}, ensure_ascii=False)},
                    image_content(view.rgb)])
            # 当前四图仍放在最后，旧图有独立历史标签，不进入本轮定位器。
            current[:0] = overview
        return messages


def guard_schema(schema, blocked):
    if not blocked:
        return schema
    result = copy.deepcopy(schema)
    branches = []
    for branch in result["anyOf"]:
        if branch["properties"]["action"]["enum"] == ["vertical_explore"]:
            directions = branch["properties"]["vertical_direction"]["enum"]
            directions = [value for value in directions if value != "up"]
            if not directions:
                continue
            branch["properties"]["vertical_direction"]["enum"] = directions
        branches.append(branch)
    if not branches:
        raise NoPermittedAction("原可尝试选项只剩被预算禁止的上升探索")
    result["anyOf"] = branches
    return result


class MinimalVLM(VLM):
    def _query(self, messages, schema, parser, iteration, phase, prompt_version):
        options = getattr(self, "minimal_options", None)
        if phase == "navigation" and options and (options.initial_yaw90 or options.ascent_guard):
            prompt_version = METHOD_VERSION
        return super()._query(messages, schema, parser, iteration, phase, prompt_version)

    def decide(self, context, observation, iteration):
        guard = getattr(context, "guard", None)
        if guard is None or not guard.options.ascent_guard:
            return super().decide(context, observation, iteration)
        if not self.current_target_evidence or context.current_targets is None:
            raise ValueError("独立方法需要启用当前目标证据")
        distance = context.feedback["motion_policy"]["stop_distance_m"]
        options = context.current_motion_options
        messages = context.messages(SYSTEM_PROMPT + NAVIGATION_PROMPT +
                                    (MOTION_PROMPT if options is not None else ""), observation, iteration)
        schema = guard_schema(navigation_schema(SCHEMA, context.current_targets, distance, options),
                              context.guard.blocked)

        def parse(content):
            return context.guard.validate(bind_motion(
                parse_decision(content, context.current_targets, distance), options))

        return self._query(messages, schema, parse, iteration, "navigation", METHOD_VERSION)
