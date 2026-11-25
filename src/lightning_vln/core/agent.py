"""观察 → 深度定位 → 直线推进 → 新观察；只有一个 Agent 循环。"""

from lightning_vln.core.episode import run_decisions

import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TextIO
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from lightning_vln.core.context import Context
from lightning_vln.core.errors import SimulationControlFailure
from lightning_vln.core.evidence import bind_target, ground_candidates
from lightning_vln.core.geometry import destination, ground_bbox, horizontal_goal, segment_clear
from lightning_vln.core.observations import save_observation
from lightning_vln.core.motion_options import bind_motion, recovery_options, repeated_direction
from lightning_vln.core.step_timing import decision_timings


@dataclass
class Limits:
    decisions: int = 40
    history_turns: int = 4
    max_step_len: float = 6.0
    target_standoff: float = 2.0
    stop_distance: float = 10.0
    speed: float = 1.0
    control_dt: float = 0.2
    radius: float = 0.75
    clearance: float = 0.5
    arrival: float = 0.5
    motion_timeout: float = 30.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value < 0 or (key not in ("history_turns", "clearance") and value == 0):
                raise ValueError(f"{key} 必须为有限正数（history_turns、clearance 可以为 0）")
        if self.speed*self.control_dt > self.radius:
            raise ValueError("每个控制周期推进距离不能超过机体半径；降低 speed 或 control_dt")


def write_jsonl(target: Path | TextIO, record: dict):
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    if isinstance(target, Path):
        with target.open("a", encoding="utf-8") as stream:
            stream.write(line)
    else:
        target.write(line)
        target.flush()  # 运行中即可读取，异常退出前已完成的行也保留。


def movement_goal(start, surface, action: str, limits: Limits) -> np.ndarray:
    surface = np.array(surface, dtype=float, copy=True)
    if action == "horizontal_approach":
        surface[2] = start[2]
    return destination(start, surface, limits.target_standoff, limits.max_step_len)


def hold_after_motion(world):
    """碰撞后仍尝试制动；已确认碰撞的制动失败不能变成可重试的任务。"""
    try:
        world.hold()
    except SimulationControlFailure:
        if not world.collided():
            raise


def repeats_failed_direction(start, goal, avoided) -> bool:
    return repeated_direction(start, goal, avoided)


def move_straight(world, goal: np.ndarray, limits: Limits, log_path: Path | TextIO,
                  trajectory_path: Path | TextIO | None = None, *, blind_up: bool = False) -> dict:
    started = time.monotonic()
    distance = 0.0
    previous = world.position()
    cycles = 0
    def sample(event, position):
        if trajectory_path is not None:
            write_jsonl(trajectory_path, {"event": event, "position_ned_m": np.asarray(position).tolist()})
    try:
        yaw_started = time.monotonic()
        turned = world.face(goal)
        yaw_seconds = time.monotonic()-yaw_started
        sample("after_yaw", world.position())
        status = None if turned else "yaw_not_reached"
        while turned:
            cycle_started = time.monotonic()
            if world.collided():
                status = "collision"
                break
            observation = world.observe(with_rgb=False)
            sample("control_observation", observation.position)
            observed_at = time.monotonic()
            if getattr(world, "last_observe", {}).get("collision", False):
                status = "collision"
                break
            # 控制器惯性和取图期间也可能有位移，不能只累加命令返回瞬间的距离。
            distance += float(np.linalg.norm(observation.position-previous))
            previous = observation.position
            delta = goal - observation.position
            if blind_up:
                delta[:2] = 0  # 纯竖直命令；不把物理水平漂移变成斜向盲飞。
            remaining = float(np.linalg.norm(delta))
            if remaining <= limits.arrival:
                status = "reached"
                break
            if time.monotonic()-started >= limits.motion_timeout:
                status = "motion_timeout"
                break
            seconds = min(limits.control_dt, remaining/limits.speed)
            velocity = delta/remaining * limits.speed
            end = observation.position + velocity*seconds
            clear, status = segment_clear(observation, end, limits.radius, limits.clearance, blind_up=blind_up)
            checked_at = time.monotonic()
            record = {"position_ned_m": observation.position.tolist(),
                      "next_position_ned_m": end.tolist(), "depth_check": status,
                      "elapsed_s": checked_at-started,
                      "observe_s": observed_at-cycle_started,
                      "guard_s": checked_at-observed_at,
                      "sensor": getattr(world, "last_observe", {})}
            write_jsonl(log_path, record)
            if not clear:
                break
            world.advance(velocity, seconds)
            cycles += 1
            command_position = world.position()
            sample("command_completed", command_position)
            write_jsonl(log_path, {"event": "command_completed", "position_ned_m": command_position.tolist(),
                                  "requested_duration_s": seconds,
                                  "command_s": time.monotonic()-checked_at,
                                  "cycle_s": time.monotonic()-cycle_started,
                                  "rpc": getattr(getattr(world, "control", None), "last", {})})
    finally:
        hold_after_motion(world)
    if world.collided():
        status = "collision"
    final_position = world.position()
    sample("after_braking", final_position)
    distance += float(np.linalg.norm(final_position-previous))
    return {"status": status, "goal_ned_m": goal.tolist(),
            "position_ned_m": final_position.tolist(), "travel_m": distance,
            "elapsed_s": time.monotonic()-started, "control_commands": cycles,
            "yaw_s": yaw_seconds,
            "stop": getattr(world, "last_stop", {})}


def run_agent(world, model, instruction: str, output: Path, limits: Limits, *, record_details=True, progress=None, metric_trace=None, step_timing_path=None, hooks=None) -> dict:
    if not record_details:
        with open(os.devnull, 'w') as sink, decision_timings(sink, step_timing_path) as decisions:
            return _run_agent(world, model, instruction, output, limits, sink,
                              metric_trace if metric_trace is not None else sink, decisions, None,
                              record_details=False, on_progress=progress, hooks=hooks)
    # 网络存储上反复open/close有显著开销；一条任务内复用，成功或异常都关闭。
    with (
        (output / 'motion.jsonl').open('a', encoding='utf-8') as motion,
        (output / 'trajectory.jsonl').open('a', encoding='utf-8') as trajectory,
        (output / 'decisions.jsonl').open('a', encoding='utf-8') as decisions,
        # JPEG/NPZ 已压缩；单个 ZIP 避免逐轮创建小文件与目录。
        ZipFile(output / 'observations.zip', 'w', compression=ZIP_STORED, allowZip64=True) as observations,
    ):
        with decision_timings(decisions, step_timing_path) as logged_decisions:
            return _run_agent(world, model, instruction, output, limits, motion, trajectory, logged_decisions, observations, hooks=hooks)



@dataclass
class NavigationHooks:
    """适配器只能注入差异，不复制导航循环或运动控制。"""
    context_factory: object = Context
    before_decision: object = None
    initial_step: object = None
    initial_decision: dict = field(default_factory=dict)
    initial_success: str = ""
    validate_decision: object = None
    invalid_action_errors: tuple = ()
    invalid_action_status: str = "invalid_decision_or_depth"
    after_execution: object = None
    summary_fields: object = None
    terminal_statuses: tuple = ("stop", "collision")
    observation_writer: object = None
    collision_after_observation: bool = True
    progress_offset: int = 0


def _run_agent(world, model, instruction, output, limits, motion, trajectory_path, decisions, observations,
               *, record_details=True, on_progress=None, hooks=None):
    hooks = hooks or NavigationHooks()
    context = hooks.context_factory(instruction, limits.history_turns)
    policy = {"horizontal_explore": "direction_only_keep_altitude", "vertical_explore": "up_blind_down_observed", "target_standoff_m": limits.target_standoff,
              "stop_distance_m": limits.stop_distance, "max_step_len_m": limits.max_step_len}
    context.feedback["motion_policy"] = policy
    started = time.monotonic()
    travelled = 0.0
    start_position = world.position().tolist()
    write_jsonl(trajectory_path, {"event": "start", "position_ned_m": start_position})
    status = "decision_limit"
    count = 0
    avoided = deque(maxlen=4)
    no_progress = 0
    positions = []
    def decision_step(iteration):
        nonlocal status, count, travelled, no_progress
        iteration_started = time.monotonic()
        hold_after_motion(world)
        held_at = time.monotonic()
        if world.collided():
            status = "collision"
            return False
        observation = world.observe()
        if hooks.collision_after_observation and world.collided():
            status = "collision"
            return False
        observed_at = time.monotonic()
        sensor = getattr(world, "last_observe", {}).copy()
        write_jsonl(trajectory_path, {"event": "decision_observation", "iteration": iteration,
                                     "position_ned_m": observation.position.tolist()})
        # 保留可重放的原始米制深度与对应相机位姿；不把轨迹全量带进模型 context。
        if observations is not None:
            (hooks.observation_writer or save_observation)(observations, iteration, observation)
        request_started = time.monotonic()
        count += 1
        if hooks.before_decision is not None:
            hooks.before_decision(observation)
        initial_feedback = (hooks.initial_step(iteration, world, observation, output, trajectory_path, context, record_details)
                            if hooks.initial_step is not None else None)
        if initial_feedback is not None:
            feedback = initial_feedback
            travelled += feedback["travel_m"]
            feedback["motion_policy"] = policy
            context.feedback = feedback
            write_jsonl(decisions, {"iteration": iteration, "decision": hooks.initial_decision,
                "execution": feedback, "sensor": sensor,
                "current_target_evidence": None, "current_motion_options": None,
                "timing": {"iteration_s": time.monotonic()-iteration_started}})
            write_jsonl(trajectory_path, {"event": "decision_end", "iteration": iteration,
                "action": hooks.initial_decision["action"], "position_ned_m": feedback["position_ned_m"]})
            if record_details:
                print(f"iteration={iteration} status={feedback['status']} initial_yaw=90", flush=True)
            if feedback["status"] != hooks.initial_success:
                status = feedback["status"]
                return False
            return True
        decision = None
        point = None
        feedback = {}
        context.current_targets = None
        context.current_motion_options = None
        try:
            if getattr(model, "current_target_evidence", False):
                candidates = model.localize(instruction, observation, iteration)
                context.current_targets = ground_candidates(candidates, observation, iteration)
                if getattr(model, "motion_options", False) and no_progress >= 2:
                    context.current_motion_options = recovery_options(observation, context.current_targets, limits, avoided, positions)
            decision = model.decide(context, observation, iteration)
            if context.current_targets is not None:
                decision = bind_target(decision, context.current_targets, limits.stop_distance)
            decision = bind_motion(decision, context.current_motion_options)
            if hooks.validate_decision is not None:
                decision = hooks.validate_decision(decision)
            if decision.action in ("approach", "stop"):
                point = ground_bbox(observation.views[decision.view], decision.bbox)
            elif decision.action == "horizontal_explore":
                goal = horizontal_goal(observation.views[decision.view], decision.bbox,
                                       observation.position, limits.max_step_len)
            elif decision.action == "vertical_explore":
                if decision.vertical_direction not in ("up", "down"):
                    raise ValueError("垂直探索必须指定 up/down")
                goal = observation.position + np.array([0., 0.,
                    -limits.max_step_len if decision.vertical_direction == "up" else limits.max_step_len])
            else:
                raise ValueError("未知动作")
        except hooks.invalid_action_errors as error:
            feedback = {"status": hooks.invalid_action_status, "error": str(error),
                        "position_ned_m": world.position().tolist()}
        except ValueError as error:
            feedback = {"status": "invalid_decision_or_depth", "error": str(error),
                        "position_ned_m": world.position().tolist()}
        request_seconds = time.monotonic()-request_started
        if world.collided():
            feedback = {"status": "collision", "position_ned_m": world.position().tolist()}
        if not feedback:
            target_distance = float(np.linalg.norm(point-world.position())) if point is not None else None
            if decision.action == "stop":
                accepted = target_distance <= limits.stop_distance
                feedback = {"status": "stop" if accepted else "stop_rejected_too_far",
                            "grounded_distance_m": target_distance,
                            "position_ned_m": world.position().tolist()}
            else:
                if decision.action == "approach":
                    goal = movement_goal(observation.position, point, decision.action, limits)
                if np.linalg.norm(goal-observation.position) <= limits.arrival:
                    feedback = {"status": "goal_too_close", "goal_ned_m": goal.tolist(),
                                "position_ned_m": world.position().tolist()}
                elif repeats_failed_direction(observation.position, goal, avoided):
                    feedback = {"status": "repeated_blocked_direction", "goal_ned_m": goal.tolist(),
                                "position_ned_m": world.position().tolist()}
                else:
                    feedback = move_straight(world, goal, limits, motion, trajectory_path,
                                             blind_up=decision.action == "vertical_explore" and decision.vertical_direction == "up")
                    if (decision.action == "approach" and goal[2] > observation.position[2]
                            and feedback["status"] == "unobserved_or_near_obstacle"):
                        # 原三维路径受阻后，从真实停止点重算水平路径，仍检查整个机体通道。
                        first = feedback
                        origin = np.asarray(first["position_ned_m"])
                        candidate = movement_goal(origin, point, "horizontal_approach", limits)
                        remaining_budget = max(limits.max_step_len-first["travel_m"], 0)
                        candidate = destination(origin, candidate, 0, remaining_budget)
                        if (np.linalg.norm(candidate-origin) > limits.arrival
                                and not repeats_failed_direction(origin, candidate, avoided)):
                            goal = candidate
                            feedback = move_straight(world, goal, limits, motion, trajectory_path)
                            feedback["travel_m"] += first["travel_m"]
                            feedback["approach_fallback"] = {"mode": "horizontal", "initial_execution": first}
                    if feedback["status"] in ("blocked", "unobserved_or_near_obstacle", "motion_timeout", "yaw_not_reached"):
                        origin = np.asarray(feedback["position_ned_m"])
                        delta = goal-origin
                        length = np.linalg.norm(delta)
                        if length > limits.arrival:
                            avoided.append((origin, delta/length))
            travelled += feedback.get("travel_m", 0)
            feedback["grounded_distance_m"] = target_distance
        progress = float(np.linalg.norm(np.asarray(feedback["position_ned_m"])-observation.position))
        no_progress = no_progress+1 if progress < limits.arrival else 0
        positions.append(np.asarray(feedback["position_ned_m"]))
        feedback["displacement_m"] = progress
        feedback["motion_policy"] = policy
        feedback["no_progress_rounds"] = no_progress
        feedback["avoid_moves"] = [{"origin_ned_m": origin.tolist(), "direction_ned": direction.tolist()}
                                   for origin, direction in avoided
                                   if np.linalg.norm(observation.position-origin) < 1]
        if hooks.after_execution is not None:
            hooks.after_execution(observation, decision, feedback)
        context.record(observation, decision, feedback)
        record = {"iteration": iteration, "decision": asdict(decision) if decision else None,
                  "grounded_surface_ned_m": point.tolist() if point is not None else None,
                  "execution": feedback, "model_and_grounding_s": request_seconds,
                  "history_turns": len(context.turns),
                  "sensor": sensor,
                  "timing": {"hold_s": held_at-iteration_started,
                             "observe_s": observed_at-held_at,
                             "save_observation_s": request_started-observed_at,
                             "iteration_s": time.monotonic()-iteration_started}}
        if getattr(model, "current_target_evidence", False):
            record["current_target_evidence"] = context.current_targets
        if getattr(model, "motion_options", False):
            record["current_motion_options"] = context.current_motion_options
        write_jsonl(decisions, record)
        write_jsonl(trajectory_path, {"event": "decision_end", "iteration": iteration,
                                     "action": decision.action if decision else None,
                                     "position_ned_m": feedback["position_ned_m"]})
        if record_details:
            print(f"iteration={iteration} status={feedback['status']} model_s={request_seconds:.2f}", flush=True)
        if on_progress is not None:
            on_progress(count - hooks.progress_offset)
        if feedback["status"] in hooks.terminal_statuses:
            status = feedback["status"]
            return False
        return True
    run_decisions(limits.decisions, decision_step)
    hold_after_motion(world)
    if world.collided():
        status = "collision"
    final_position = world.position().tolist()
    write_jsonl(trajectory_path, {"event": "final", "position_ned_m": final_position})
    summary = {"status": status, "decisions": count,
               "start_position_ned_m": start_position,
               "position_ned_m": final_position, "travel_m": travelled,
               "elapsed_s": time.monotonic()-started}
    if hooks.summary_fields is not None:
        summary.update(hooks.summary_fields(count))
    if record_details:
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
