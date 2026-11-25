"""UAV-ON初始观察和上升保护；导航循环由共享核心执行。"""

import json
import time
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile
import numpy as np
from lightning_vln.core.agent import Limits, hold_after_motion, write_jsonl, run_agent as original_run_agent
from lightning_vln.core.geometry import rotation
from lightning_vln.core.observations import save_observation
from lightning_vln.adapters.uavon.minimal import METHOD_VERSION
from lightning_vln.adapters.uavon.minimal.policy import AscentGuard, Options
from lightning_vln.adapters.uavon.minimal.vlm import MinimalContext, NoPermittedAction


def heading(observation):
    forward = rotation(observation.views["front"].quaternion) @ np.array([1., 0., 0.])
    if np.linalg.norm(forward[:2]) < .1:
        raise ValueError("前视相机方向不足以确定水平航向")
    return float(np.arctan2(forward[1], forward[0]))


def initial_rotation(world, before, output, trajectory, *, record_details=True):
    started = time.monotonic()
    yaw = heading(before)
    target_yaw = yaw+np.pi/2
    goal = before.position+10*np.array([np.cos(target_yaw), np.sin(target_yaw), 0.])
    try:
        turned = world.face(goal)
    finally:
        hold_after_motion(world)
        braked_position = world.position()
        write_jsonl(trajectory, {"event": "initial_yaw_after_braking",
                               "position_ned_m": braked_position.tolist()})
    after = world.observe()
    error = float(np.rad2deg(abs(np.arctan2(np.sin(heading(after)-target_yaw),
                                          np.cos(heading(after)-target_yaw)))))
    if record_details:
        with ZipFile(output/"initial-scan.zip", "w", compression=ZIP_STORED) as archive:
            save_observation(archive, 0, before)
            save_observation(archive, 1, after)
    feedback = {"status": "initial_yaw_completed" if turned and error <= 7 else "initial_yaw_not_reached",
                "position_ned_m": after.position.tolist(), "yaw_requested_degrees": 90,
                "initial_heading_degrees": float(np.rad2deg(yaw)),
                "final_heading_degrees": float(np.rad2deg(heading(after))),
                "yaw_error_degrees": error,
                "displacement_m": float(np.linalg.norm(after.position-before.position)),
                "travel_m": float(np.linalg.norm(braked_position-before.position)
                                  +np.linalg.norm(after.position-braked_position)),
                "elapsed_s": time.monotonic()-started,
                "sensor_after_rotation": getattr(world, "last_observe", {}),
                "budget_cost": 1}
    if world.collided():
        feedback["status"] = "collision"
    if record_details:
        (output/"initial-scan.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2)+"\n")
    return feedback


def run_agent(world, model, instruction: str, output: Path, limits: Limits, options=None, *, record_details=True, progress=None, metric_trace=None, step_timing_path=None):
    from lightning_vln.core.agent import NavigationHooks
    options = options or Options()
    if not options.initial_yaw90 and not options.ascent_guard:
        return original_run_agent(world, model, instruction, output, limits, record_details=record_details,
                                  progress=progress, metric_trace=metric_trace, step_timing_path=step_timing_path)
    guard = AscentGuard(world.position(), options)
    def initial(iteration, world, observation, output, trajectory, context, detailed):
        if iteration != 0 or not options.initial_yaw90:
            return None
        feedback = initial_rotation(world, observation, output, trajectory, record_details=detailed)
        context.initial_overview = observation
        return feedback
    def after(observation, decision, feedback):
        guard.record(observation, decision, feedback)
        if options.ascent_guard:
            feedback["ascent_guard"] = guard.feedback()
    def summary(count):
        rotation_count = int(options.initial_yaw90 and count > 0)
        return dict(method_version=METHOD_VERSION, method_options=options.manifest(),
                    initial_rotation_actions=rotation_count, navigation_decisions=count-rotation_count,
                    ascent_guard_runtime_blocks=guard.blocks, ascent_guard_resets=guard.resets)
    hooks = NavigationHooks(
        context_factory=lambda instruction, turns: MinimalContext(instruction, turns, guard),
        before_decision=lambda observation: guard.observe(observation.position),
        initial_step=initial, initial_decision={"action":"initial_yaw", "yaw_degrees":90},
        initial_success="initial_yaw_completed", validate_decision=guard.validate,
        invalid_action_errors=(NoPermittedAction,), invalid_action_status="ascent_guard_exhausted",
        after_execution=after, summary_fields=summary, progress_offset=int(options.initial_yaw90),
        terminal_statuses=("stop", "collision", "ascent_guard_exhausted"),
        observation_writer=save_observation)
    return original_run_agent(world, model, instruction, output, limits, record_details=record_details,
                              progress=progress, metric_trace=metric_trace, step_timing_path=step_timing_path, hooks=hooks)
