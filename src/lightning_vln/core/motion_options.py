"""连续无进展时的当前帧移动选项；使用原控制器的短段保护，不维护地图。"""

import numpy as np

from lightning_vln.core.geometry import destination, ground_bbox, horizontal_goal, segment_clear


PROMPT_VERSION = "visual_search_v7_current_motion_options"
from lightning_vln.core.prompts import MOTION_NAVIGATION_PROMPT as NAVIGATION_PROMPT


def repeated_direction(start, goal, avoided):
    delta = goal-start
    length = np.linalg.norm(delta)
    return bool(length > 1e-9 and any(
        np.linalg.norm(start-origin) < 1 and delta @ direction / length > np.cos(np.deg2rad(15))
        for origin, direction in avoided))


def recovery_options(observation, targets, limits, avoided, positions):
    start = observation.position

    def allowed(goal, blind_up=False):
        delta = goal-start
        length = np.linalg.norm(delta)
        if length <= limits.arrival or repeated_direction(start, goal, avoided):
            return False
        end = start+delta/length*min(limits.speed*limits.control_dt, length)
        return segment_clear(observation, end, limits.radius, limits.clearance, blind_up=blind_up)[0]

    moves = []
    directions = []
    for name in ("front", "left", "right"):
        for center in (100, 250, 400, 500, 600, 750, 900):
            bbox = [center-30, 450, center+30, 550]
            try:
                goal = horizontal_goal(observation.views[name], bbox, start, limits.max_step_len)
            except ValueError:
                continue
            direction = (goal-start)/limits.max_step_len
            if any(direction @ old > np.cos(np.deg2rad(5)) for old in directions) or not allowed(goal):
                continue
            directions.append(direction)
            moves.append(dict(action="horizontal_explore", view=name, bbox=bbox,
                              vertical_direction=None, goal_ned_m=goal.tolist(),
                              recent_visits=sum(bool(np.linalg.norm(goal-p) < 3) for p in positions[:-3]),
                              clearance="first_control_step_only"))
    for direction, sign in (("up", -1), ("down", 1)):
        goal = start+np.array([0., 0., sign*limits.max_step_len])
        if allowed(goal, blind_up=direction == "up"):
            moves.append(dict(action="vertical_explore", view="front" if direction == "up" else "down",
                              bbox=[400, 400, 600, 600], vertical_direction=direction,
                              goal_ned_m=goal.tolist(),
                              recent_visits=sum(bool(np.linalg.norm(goal-p) < 3) for p in positions[:-3]),
                              clearance="blind_up" if direction == "up" else "first_control_step_only"))
    target_actions = []
    for target in targets:
        if not target["groundable"]:
            continue
        actions = []
        point = ground_bbox(observation.views[target["view"]], target["bbox"])
        goal = destination(start, point, limits.target_standoff, limits.max_step_len)
        feasible = allowed(goal)
        # 原执行器允许受阻的斜向下降改为同高度靠近，预检保留这一能力。
        if not feasible and goal[2] > start[2] and not repeated_direction(start, goal, avoided):
            delta = goal-start
            length = np.linalg.norm(delta)
            if length > limits.arrival:
                end = start+delta/length*min(limits.speed*limits.control_dt, length)
                _, reason = segment_clear(observation, end, limits.radius, limits.clearance)
                if reason == "unobserved_or_near_obstacle":
                    surface = point.copy(); surface[2] = start[2]
                    feasible = allowed(destination(start, surface, limits.target_standoff, limits.max_step_len))
        if feasible:
            actions.append("approach")
        if target["match"] == "confirmed" and target["grounded_distance_m"] <= limits.stop_distance:
            actions.append("stop")
        if actions:
            target_actions.append(dict(target_id=target["target_id"], actions=actions))
    return dict(active=bool(moves or target_actions), moves=moves, targets=target_actions)


def bind_motion(decision, options):
    if options is None or not options["active"]:
        return decision
    if decision.action in ("approach", "stop"):
        valid = any(t["target_id"] == decision.target_id and decision.action in t["actions"]
                    for t in options["targets"])
    else:
        valid = any(decision.action == m["action"] and decision.view == m["view"]
                    and tuple(decision.bbox) == tuple(m["bbox"])
                    and decision.vertical_direction == m["vertical_direction"] for m in options["moves"])
    if not valid:
        raise ValueError("连续无进展时必须选择本轮可尝试的移动选项")
    return decision
