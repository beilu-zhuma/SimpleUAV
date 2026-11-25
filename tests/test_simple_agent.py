import json
import math
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_STORED, ZipFile

import numpy as np
from PIL import Image

from lightning_vln.agent import Limits, move_straight, movement_goal, repeats_failed_direction, run_agent
from lightning_vln.context import Context, MEMORY_CHARS
from lightning_vln.geometry import destination, ground_bbox, horizontal_goal, rays, rotation, segment_clear
from lightning_vln.observations import load_observation, observation_steps
from lightning_vln.types import Decision, Observation, View, VIEWS
from lightning_vln.vlm import SYSTEM_PROMPT, parse_decision


def image_bytes():
    buffer = BytesIO()
    Image.new("RGB", (32, 32), "gray").save(buffer, "JPEG")
    return buffer.getvalue()


def view(depth=20, position=(0, 0, 0), quaternion=(0, 0, 0, 1)):
    return View(image_bytes(), np.full((32, 32), depth, dtype=float),
                np.array(position, dtype=float), np.array(quaternion, dtype=float), 90, 1)


def decision(action="horizontal_explore", memory="", name="front"):
    return Decision(action, name, (400, 400, 600, 600), "检查这个区域", memory)


class GeometryTest(unittest.TestCase):
    def test_near_guard_accounts_for_camera_offset_and_reuses_only_intrinsics(self):
        camera = view(5, position=(-5, 0, 0))
        observation = Observation(np.zeros(3), {"front": camera})
        self.assertEqual(segment_clear(observation, [.2, 0, 0], .75, .5), (False, "blocked"))
        first = rays(camera)
        self.assertFalse(first.flags.writeable)
        camera.depth[:] = 20
        self.assertIs(rays(camera), first)
        camera.fov = 110
        self.assertIsNot(rays(camera), first)

    def test_radial_depth_and_full_camera_pose(self):
        camera = view(10, (3, 4, 5))
        ray = rays(camera)[0, 0]
        self.assertAlmostEqual(np.linalg.norm(ray), 1)
        # 单像素角落框；径向深度在斜射线上不应被误当轴向深度。
        point = ground_bbox(camera, (0, 0, 1, 1))
        np.testing.assert_allclose(point, camera.position + 10*ray)
        self.assertAlmostEqual(np.linalg.norm(point-camera.position), 10)
        self.assertLess(point[0]-camera.position[0], 10)

    def test_yaw_and_down_camera_transform(self):
        yaw = (0, 0, math.sin(math.pi/4), math.cos(math.pi/4))
        np.testing.assert_allclose(rotation(yaw) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        down = (0, -math.sin(math.pi/4), 0, math.cos(math.pi/4))
        np.testing.assert_allclose(rotation(down) @ [1, 0, 0], [0, 0, 1], atol=1e-12)

    def test_foreground_sampling_and_invalid_depth(self):
        camera = view(20)
        camera.depth[12:20, 12:20] = 4
        point = ground_bbox(camera, (300, 300, 700, 700))
        self.assertLess(np.linalg.norm(point), 5)
        for value in (0, -1, math.nan, math.inf, 100, 1e38):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ground_bbox(view(value), (400, 400, 600, 600))

    def test_standoff_and_leg_cap(self):
        np.testing.assert_allclose(destination([0, 0, 0], [20, 0, 0], 2, 6), [6, 0, 0])
        np.testing.assert_allclose(destination([0, 0, 0], [1, 0, 0], 2, 6), [0, 0, 0])

    def test_depth_guard_blocks_body_width_obstacles(self):
        camera = view(20)
        observation = Observation(np.zeros(3), {"front": camera})
        self.assertTrue(segment_clear(observation, [0.2, 0, 0], .75, .5)[0])
        # 在中心线右侧的近障碍仍在机体通道内，不能只读中心深度。
        camera.depth[16, 20] = .9
        self.assertEqual(segment_clear(observation, [0.2, 0, 0], .75, .5), (False, "blocked"))

    def test_invalid_or_blind_corridor_cannot_move(self):
        for value in (0, -1, math.nan, math.inf):
            self.assertFalse(segment_clear(Observation(np.zeros(3), {"front": view(value)}),
                                           [0.2, 0, 0], .75, .5)[0])
        self.assertFalse(segment_clear(Observation(np.zeros(3), {"front": view()}),
                                       [-.2, 0, 0], .75, .5)[0])

    def test_distant_depth_can_clear_short_corridor_but_cannot_ground_target(self):
        camera = view(16400, position=(.5, 0, 0))  # Barnyard 天空的有限远平面深度。
        observation = Observation(np.zeros(3), {"front": camera})
        self.assertEqual(segment_clear(observation, [.2, 0, 0], .75, .5), (True, "clear"))
        with self.assertRaises(ValueError):
            ground_bbox(camera, (400, 400, 600, 600))
        # 远景不能掩盖胶囊体内的近障碍或缺失像素。
        camera.depth[16, 16] = .5
        self.assertEqual(segment_clear(observation, [.2, 0, 0], .75, .5), (False, "blocked"))
        camera.depth[16, 16] = math.nan
        self.assertFalse(segment_clear(observation, [.2, 0, 0], .75, .5)[0])

    def test_oblique_visible_corridor_with_offset_camera(self):
        camera = view(20, position=(.5, 0, 0))
        observation = Observation(np.zeros(3), {"front": camera})
        def end(degrees):
            angle = math.radians(degrees)
            return .2*np.array([math.cos(angle), 0, math.sin(angle)])
        self.assertEqual(segment_clear(observation, end(16), .75, .5), (True, "clear"))
        # 斜向过大时截面仍在 FOV 外，应停下而不是缩小安全半径。
        self.assertFalse(segment_clear(observation, end(40), .75, .5)[0])
        camera.depth[24, 16] = .6
        self.assertEqual(segment_clear(observation, end(16), .75, .5), (False, "blocked"))

    def test_two_views_jointly_cover_diagonal_corridor(self):
        camera = view(20, position=(.5, 0, 0))
        down = view(20, quaternion=(0, -math.sin(math.pi/4), 0, math.cos(math.pi/4)))
        camera.fov = down.fov = 110  # 外置相机需要重叠视野，90 度拼接会留缝。
        angle = math.radians(35)
        end = .2*np.array([math.cos(angle), 0, math.sin(angle)])
        observation = Observation(np.zeros(3), {"front": camera, "down": down})
        for name in observation.views:
            self.assertFalse(segment_clear(Observation(np.zeros(3), {name: observation.views[name]}),
                                           end, .75, .5)[0])
        self.assertEqual(segment_clear(observation, end, .75, .5), (True, "clear"))
        down.depth[:] = math.nan
        self.assertFalse(segment_clear(observation, end, .75, .5)[0])

    def test_direction_goal_ignores_depth_and_respects_pose_and_step(self):
        camera = view(math.nan, quaternion=(0, 0, math.sin(math.pi/4), math.cos(math.pi/4)))
        goal = horizontal_goal(camera, (400, 400, 600, 600), [1, 2, -8], 3)
        np.testing.assert_allclose(goal, [1, 5, -8], atol=1e-12)
        camera.quaternion = np.array([0, -math.sin(math.pi/4), 0, math.cos(math.pi/4)])
        with self.assertRaisesRegex(ValueError, "水平分量"):
            horizontal_goal(camera, (400, 400, 600, 600), [0, 0, 0], 6)

    def test_blind_up_still_checks_near_obstacles_and_never_exempts_down(self):
        camera = view(20)
        observation = Observation(np.zeros(3), {"front": camera})
        self.assertFalse(segment_clear(observation, [0, 0, -.2], .75, .5)[0])
        self.assertEqual(segment_clear(observation, [0, 0, -.2], .75, .5, blind_up=True),
                         (True, "clear_blind_up"))
        self.assertFalse(segment_clear(observation, [0, 0, .2], .75, .5, blind_up=True)[0])
        self.assertFalse(segment_clear(observation, [-.2, 0, 0], .75, .5, blind_up=True)[0])
        camera.depth[16, 16] = .5
        self.assertEqual(segment_clear(observation, [0, 0, -.2], .75, .5, blind_up=True),
                         (False, "blocked"))


class ContextTest(unittest.TestCase):
    def test_bounded_keyframes_and_latest_execution(self):
        context = Context("寻找红车", 2)
        observation = Observation(np.zeros(3), {name: view() for name in VIEWS})
        for index in range(6):
            context.record(observation, decision(memory=f"记忆{index}", name="left"),
                           {"status": "blocked", "position_ned_m": [index, 0, 0]})
        messages = context.messages(SYSTEM_PROMPT, observation, 6)
        self.assertEqual(len(context.turns), 2)
        self.assertEqual(len(messages), 7)
        images = [part for message in messages if isinstance(message["content"], list)
                  for part in message["content"] if part["type"] == "image_url"]
        self.assertEqual(len(images), 6)  # 2 张历史关键帧 + 当前 4 张。
        current = json.loads(messages[-1]["content"][0]["text"])
        self.assertEqual(current["search_memory_unverified"], "记忆5")
        self.assertEqual(current["last_execution"]["position_ned_m"], [5, 0, 0])
        self.assertNotIn("记忆0", json.dumps(messages, ensure_ascii=False))

    def test_no_history_preserves_current_feedback_and_memory(self):
        context = Context("寻找红车", 0)
        observation = Observation(np.zeros(3), {name: view() for name in VIEWS})
        context.record(observation, decision(memory="线索"), {"status": "blocked"})
        context.record(observation, None, {"status": "invalid_decision_or_depth"})
        self.assertEqual(len(context.turns), 0)
        self.assertEqual(context.memory, "线索")
        self.assertEqual(context.feedback["status"], "invalid_decision_or_depth")

    def test_strict_model_boundary(self):
        payload = dict(action="horizontal_explore", view="front", bbox=[10, 20, 100, 200], reason="查看", memory="", vertical_direction=None)
        self.assertEqual(parse_decision(json.dumps(payload)).action, "horizontal_explore")
        self.assertEqual(parse_decision(json.dumps(dict(payload, action="approach"))).action, "approach")
        for patch in ({"bbox": [0, 0, math.nan, 100]}, {"bbox": [True, 0, 100, 100]},
                      {"bbox": [100, 0, 0, 100]}, {"action": "teleport"},
                      {"memory": "a"*(MEMORY_CHARS+1)}, {"reason": ""}, {"extra": 1}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                parse_decision(json.dumps({**payload, **patch}))


class FakeWorld:
    def __init__(self, obstacle_after=None, failure_after=None):
        self.location = np.zeros(3)
        self.advances = 0
        self.holds = 0
        self.obstacle_after = obstacle_after
        self.failure_after = failure_after

    def position(self):
        return self.location.copy()

    def hold(self):
        self.holds += 1

    def face(self, goal):
        return True

    def collided(self):
        return False

    def observe(self, with_rgb=True):
        if self.failure_after is not None and self.advances >= self.failure_after:
            raise RuntimeError("取图失败")
        frames = {"front": view(20, self.location)}
        for name, yaw in (("left", -math.pi/2), ("right", math.pi/2)):
            frames[name] = view(20, self.location, (0, 0, math.sin(yaw/2), math.cos(yaw/2)))
        frames["down"] = view(20, self.location, (0, -math.sin(math.pi/4), 0, math.cos(math.pi/4)))
        # 固定世界 x=10 的表面，使下一轮观察反映真正的运动进展。
        frames["front"].depth = (10-self.location[0])/rays(frames["front"])[..., 0]
        if self.obstacle_after is not None and self.advances >= self.obstacle_after:
            frames["front"].depth[16, 16] = 1
        return Observation(self.position(), frames)

    def advance(self, velocity, seconds):
        self.location += velocity*seconds
        self.advances += 1


class MotionTest(unittest.TestCase):
    def test_unfinished_yaw_brakes_without_translation_and_reobserves(self):
        class UnfinishedYaw(FakeWorld):
            def face(self, goal):
                return False
        class Model:
            feedback = []
            def decide(self, context, *args):
                self.feedback.append(context.feedback.copy())
                return decision()
        world, model = UnfinishedYaw(), Model()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_agent(world, model, "寻找目标", Path(tmp), Limits(decisions=2))
            rows = [json.loads(s) for s in (Path(tmp)/"decisions.jsonl").read_text().splitlines()]
        self.assertEqual(result["status"], "decision_limit")
        self.assertEqual(rows[0]["execution"]["status"], "yaw_not_reached")
        self.assertEqual(model.feedback[1]["status"], "yaw_not_reached")
        self.assertEqual(rows[1]["execution"]["status"], "repeated_blocked_direction")
        self.assertGreaterEqual(world.holds, 3)
        self.assertEqual(world.advances, 0)

    def test_yaw_transport_timeout_brakes_and_still_fails(self):
        world = FakeWorld()
        with tempfile.TemporaryDirectory() as tmp, patch.object(world, "face", side_effect=TimeoutError("rotateToYaw")):
            with self.assertRaisesRegex(TimeoutError, "rotateToYaw"):
                move_straight(world, np.array([2., 0, 0]), Limits(), Path(tmp)/"motion.jsonl")
        self.assertEqual(world.holds, 1)
        self.assertEqual(world.advances, 0)

    def test_live_logs_are_readable_and_closed_after_model_failure(self):
        opened = []
        original = Path.open
        def track(path, *args, **kwargs):
            stream = original(path, *args, **kwargs)
            if path.suffix == '.jsonl' and args and args[0] == 'a':
                opened.append(stream)
            return stream
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            class Model:
                def decide(self, *args):
                    rows = [json.loads(line) for line in (output/'trajectory.jsonl').read_text().splitlines()]
                    assert rows[0]['event'] == 'start'
                    raise RuntimeError('model disconnected')
            with patch.object(Path, 'open', track), self.assertRaisesRegex(RuntimeError, 'model disconnected'):
                run_agent(FakeWorld(), Model(), '寻找目标', output, Limits())
            self.assertTrue(opened)
            self.assertTrue(all(stream.closed for stream in opened))
            self.assertEqual(json.loads((output/'trajectory.jsonl').read_text().splitlines()[-1])['event'],
                             'decision_observation')

    def test_exploration_of_ground_region_preserves_altitude(self):
        class Model:
            def decide(self, *args):
                return Decision("horizontal_explore", "front", (400, 700, 600, 900), "沿道路探索", "")
        world = FakeWorld()
        world.location[2] = -10
        with tempfile.TemporaryDirectory() as tmp:
            run_agent(world, Model(), "寻找公交站", Path(tmp), Limits(decisions=1))
            record = json.loads((Path(tmp)/"decisions.jsonl").read_text())
        self.assertIsNone(record["grounded_surface_ned_m"])
        self.assertEqual(record["execution"]["goal_ned_m"][2], -10)
        self.assertEqual(world.location[2], -10)
        self.assertGreater(world.advances, 0)

    def test_approach_can_change_height_with_standoff_and_leg_cap(self):
        start, surface = np.array([0., 0, -10]), np.array([10., 0, 0])
        goal = movement_goal(start, surface, "approach", Limits())
        self.assertGreater(goal[2], -10)
        self.assertAlmostEqual(np.linalg.norm(goal-start), 6)
        self.assertGreater(np.linalg.norm(surface-goal), 2)
        np.testing.assert_array_equal(surface, [10, 0, 0])

    def test_repeated_blocked_goals_do_not_issue_more_motion_commands(self):
        class Model:
            feedback = []
            def decide(self, context, *args):
                self.feedback.append(context.feedback)
                return decision()
        world, model = FakeWorld(obstacle_after=0), Model()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_agent(world, model, "寻找目标", Path(tmp), Limits(decisions=3, history_turns=0))
            records = [json.loads(line) for line in (Path(tmp)/"decisions.jsonl").read_text().splitlines()]
        self.assertEqual(result["status"], "decision_limit")
        self.assertEqual(records[0]["execution"]["status"], "blocked")
        self.assertEqual(records[1]["execution"]["status"], "repeated_blocked_direction")
        self.assertEqual(records[2]["execution"]["no_progress_rounds"], 3)
        self.assertTrue(model.feedback[-1]["avoid_moves"])
        self.assertEqual(world.advances, 0)

    def test_changed_direction_or_position_allows_retry(self):
        failed = [(np.zeros(3), np.array([1., 0, 0]))]
        self.assertTrue(repeats_failed_direction(np.zeros(3), np.array([6., .2, 0]), failed))
        self.assertFalse(repeats_failed_direction(np.zeros(3), np.array([0., 6, 0]), failed))
        self.assertFalse(repeats_failed_direction(np.array([0., 2, 0]), np.array([6., 2, 0]), failed))

    def test_close_move_is_not_reported_as_reached_or_stop(self):
        class Model:
            def decide(self, *args):
                return decision()
        world = FakeWorld()
        world.location[0] = 9
        with tempfile.TemporaryDirectory() as tmp:
            result = run_agent(world, Model(), "寻找目标", Path(tmp), Limits(decisions=1))
            record = json.loads((Path(tmp)/"decisions.jsonl").read_text())
        self.assertEqual(result["status"], "decision_limit")
        self.assertEqual(record["execution"]["status"], "blocked")
        self.assertEqual(world.advances, 0)

    def test_reached_with_short_velocity_commands(self):
        world = FakeWorld()
        with tempfile.TemporaryDirectory() as tmp:
            result = move_straight(world, np.array([2., 0, 0]), Limits(), Path(tmp)/"motion.jsonl")
        self.assertEqual(result["status"], "reached")
        self.assertGreater(world.advances, 1)
        self.assertGreater(world.holds, 0)

    def test_new_obstacle_stops_before_next_command(self):
        world = FakeWorld(obstacle_after=2)
        with tempfile.TemporaryDirectory() as tmp:
            result = move_straight(world, np.array([6., 0, 0]), Limits(), Path(tmp)/"motion.jsonl")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(world.advances, 2)
        np.testing.assert_allclose(world.location, [.4, 0, 0])

    def test_sensor_failure_holds_in_finally(self):
        world = FakeWorld(failure_after=1)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(RuntimeError, "取图失败"):
            move_straight(world, np.array([6., 0, 0]), Limits(), Path(tmp)/"motion.jsonl")
        self.assertEqual(world.advances, 1)
        self.assertEqual(world.holds, 1)

    def test_motion_distance_includes_sensor_interval_and_final_hover_drift(self):
        class DriftingWorld(FakeWorld):
            def observe(self, with_rgb=True):
                if self.advances:
                    self.location[0] += .05
                return super().observe(with_rgb)
            def hold(self):
                super().hold()
                self.location[0] += .1
        world = DriftingWorld()
        with tempfile.TemporaryDirectory() as tmp:
            result = move_straight(world, np.array([2., 0, 0]), Limits(), Path(tmp)/"motion.jsonl")
        self.assertAlmostEqual(result["travel_m"], world.location[0])

    def test_agent_reobserves_and_only_model_stop_ends_search(self):
        class Model:
            def decide(self, context, observation, iteration):
                if iteration:
                    self.last_feedback = context.feedback
                return decision("stop" if iteration else "horizontal_explore", memory="检查过当前方向")
        world, model = FakeWorld(), Model()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_agent(world, model, "寻找目标", Path(tmp), Limits(decisions=3))
            self.assertEqual(observation_steps(Path(tmp)), [0, 1])
            self.assertFalse((Path(tmp)/"observations").exists())
            saved = load_observation(Path(tmp), 1, world.location)
            with ZipFile(Path(tmp)/"observations.zip") as archive:
                self.assertIsNone(archive.testzip())
                self.assertTrue(all(info.compress_type == ZIP_STORED for info in archive.infolist()))
            for name, camera in world.observe().views.items():
                self.assertEqual(saved.views[name].rgb, camera.rgb)
                np.testing.assert_array_equal(saved.views[name].depth, camera.depth)
                np.testing.assert_array_equal(saved.views[name].position, camera.position)
                np.testing.assert_array_equal(saved.views[name].quaternion, camera.quaternion)
                self.assertEqual(saved.views[name].timestamp, camera.timestamp)
                self.assertEqual(saved.views[name].fov, camera.fov)
            records = [json.loads(line) for line in (Path(tmp)/"decisions.jsonl").read_text().splitlines()]
        self.assertEqual(result["status"], "stop")
        self.assertEqual(result["decisions"], 2)
        self.assertGreater(records[0]["execution"]["position_ned_m"][0], 0)
        self.assertEqual(model.last_feedback["status"], records[0]["execution"]["status"])

    def test_observation_archive_closes_on_model_error_or_interrupt(self):
        for error in (RuntimeError("模型故障"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp:
                class Model:
                    def decide(self, *args):
                        raise error
                with self.assertRaises(type(error)):
                    run_agent(FakeWorld(), Model(), "寻找目标", Path(tmp), Limits())
                self.assertEqual(observation_steps(Path(tmp)), [0])
                with ZipFile(Path(tmp)/"observations.zip") as archive:
                    self.assertIsNone(archive.testzip())

    def test_observation_replay_accepts_legacy_directory(self):
        class Model:
            def decide(self, *args):
                return decision("stop")
        world = FakeWorld()
        with tempfile.TemporaryDirectory() as tmp:
            ep = Path(tmp)
            run_agent(world, Model(), "寻找目标", ep, Limits(decisions=1))
            with ZipFile(ep/"observations.zip") as archive:
                archive.extractall(ep/"observations")
            (ep/"observations.zip").unlink()
            self.assertEqual(observation_steps(ep), [0])
            saved = load_observation(ep, 0, world.location)
            np.testing.assert_array_equal(saved.views["front"].depth, world.observe().views["front"].depth)

    def test_invalid_depth_never_executes_a_motion_or_accepts_stop(self):
        class InvalidWorld(FakeWorld):
            def observe(self, with_rgb=True):
                observation = super().observe(with_rgb=with_rgb)
                observation.views["front"].depth[:] = math.nan
                return observation
        class Model:
            def decide(self, *args):
                return decision("stop")
        world = InvalidWorld()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_agent(world, Model(), "寻找目标", Path(tmp), Limits(decisions=2))
        self.assertEqual(result["status"], "decision_limit")
        self.assertEqual(world.advances, 0)

    def test_control_step_limit(self):
        with self.assertRaises(ValueError):
            Limits(speed=5, control_dt=1)

    def test_zero_extra_clearance_preserves_finite_radius_and_control_step_bounds(self):
        limits = Limits(radius=.5, clearance=0)
        self.assertEqual(limits.radius+limits.clearance, .5)
        for change in ({'clearance': -.01}, {'clearance': math.nan},
                       {'clearance': math.inf}, {'radius': 0},
                       {'speed': 3, 'control_dt': .2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Limits(**(dict(radius=.5, clearance=0) | change))

    def test_stop_with_a_distant_bbox_is_rejected(self):
        class Model:
            def decide(self, *args):
                return decision("stop")
        world = FakeWorld()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_agent(world, Model(), "寻找目标", Path(tmp),
                               Limits(decisions=1, stop_distance=3))
            record = json.loads((Path(tmp)/"decisions.jsonl").read_text())
        self.assertEqual(result["status"], "decision_limit")
        self.assertEqual(record["execution"]["status"], "stop_rejected_too_far")
        self.assertEqual(world.advances, 0)

    def test_collision_during_model_call_cannot_be_accepted_as_stop(self):
        class World(FakeWorld):
            def collided(self):
                return getattr(self, "hit", False)
        world = World()
        class Model:
            def decide(self, *args):
                world.hit = True
                return decision("stop")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_agent(world, Model(), "寻找目标", Path(tmp), Limits(decisions=1))
        self.assertEqual(result["status"], "collision")
        self.assertEqual(world.advances, 0)

    def test_far_sky_explores_but_local_obstacle_blocks(self):
        class SkyWorld(FakeWorld):
            def observe(self, with_rgb=True):
                observation = super().observe(with_rgb)
                observation.views["front"].depth[:] = 25000
                if self.obstacle_after is not None:
                    observation.views["front"].depth[16,16] = .5
                return observation
        class Model:
            def decide(self, *args):
                return decision()
        for obstacle in (None, 0):
            world = SkyWorld(obstacle_after=obstacle)
            with self.subTest(obstacle=obstacle), tempfile.TemporaryDirectory() as tmp:
                run_agent(world, Model(), "探索远方", Path(tmp), Limits(decisions=1, max_step_len=2))
                row = json.loads((Path(tmp)/"decisions.jsonl").read_text())
            self.assertIsNone(row["grounded_surface_ned_m"])
            self.assertEqual(row["execution"]["status"], "reached" if obstacle is None else "blocked")
            self.assertLessEqual(np.linalg.norm(world.location), 2)

    def test_approach_eight_metre_surface_now_makes_progress(self):
        goal = movement_goal(np.zeros(3), np.array([8.42, 0, 0]), "approach", Limits())
        np.testing.assert_allclose(goal, [6, 0, 0])

    def test_downward_approach_falls_back_to_checked_horizontal_path(self):
        class Model:
            def decide(self, *args):
                return Decision("approach", "front", (450, 750, 550, 900), "检查车", "")
        world = FakeWorld()
        with tempfile.TemporaryDirectory() as tmp:
            run_agent(world, Model(), "找车", Path(tmp), Limits(decisions=1))
            row = json.loads((Path(tmp)/"decisions.jsonl").read_text())
        fallback = row["execution"]["approach_fallback"]
        self.assertEqual(fallback["initial_execution"]["status"], "unobserved_or_near_obstacle")
        self.assertGreater(world.location[0], 1)
        self.assertEqual(world.location[2], 0)
        self.assertLessEqual(row["execution"]["travel_m"], 6.001)

    def test_vertical_can_return_to_height_and_collision_is_terminal(self):
        from lightning_vln.errors import SimulationControlFailure
        class Model:
            def decide(self, context, obs, iteration):
                return Decision("vertical_explore", "front", (400,400,600,600), "调整视野", "",
                                "down" if iteration == 0 else "up")
        world = FakeWorld()
        with tempfile.TemporaryDirectory() as tmp:
            run_agent(world, Model(), "搜索", Path(tmp), Limits(decisions=2, max_step_len=2))
            rows = [json.loads(line) for line in (Path(tmp)/"decisions.jsonl").read_text().splitlines()]
        self.assertGreater(rows[0]["execution"]["position_ned_m"][2], 1)
        self.assertAlmostEqual(world.location[2], 0, places=6)
        class CrashWorld(FakeWorld):
            def collided(self):
                return self.advances > 0
            def hold(self):
                if self.collided():
                    raise SimulationControlFailure("collision prevents hold")
                super().hold()
        world = CrashWorld()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_agent(world, Model(), "搜索", Path(tmp), Limits(decisions=3))
        self.assertEqual(result["status"], "collision")
        self.assertEqual(result["decisions"], 1)
        self.assertEqual(world.advances, 1)

    def test_vertical_schema_direction_consistency(self):
        payload = dict(action="vertical_explore", view="front", bbox=[400,400,600,600],
                       reason="升高视野", memory="", vertical_direction="up")
        self.assertEqual(parse_decision(json.dumps(payload)).vertical_direction, "up")
        for patch in ({"vertical_direction": None}, {"vertical_direction": "side"},
                      {"action": "horizontal_explore"}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                parse_decision(json.dumps({**payload, **patch}))


if __name__ == "__main__":
    unittest.main()
