import json
from io import BytesIO
from PIL import Image
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lightning_vln.uavon_tuning import cli
from lightning_vln.context import Context
from lightning_vln.errors import SimulationUnavailable, SimulationControlFailure, SimulationInitializationFailure
from lightning_vln.scene import settings
from lightning_vln.simulator import AirSimWorld
from lightning_vln.vlm import VLM
from test_simple_agent import FakeWorld, image_bytes


class ModelHandler(BaseHTTPRequestHandler):
    requests = []
    finish_reason = "stop"
    invalid_bbox_on_first = False

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append(body)
        action = "horizontal_explore" if len(self.requests) == 1 else "stop"
        content = {"action": action, "vertical_direction": None, "view": "front", "bbox": [400, 400, 600, 600],
                   "reason": "检查当前目标", "memory": "当前方向已检查；继续核实"}
        if self.invalid_bbox_on_first and len(self.requests) == 1:
            content["bbox"] = [0, 0, 0, 0]
        response = {"choices": [{"finish_reason": self.finish_reason,
                                 "message": {"content": json.dumps(content)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        encoded = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args):
        pass


class HTTPAndCLITest(unittest.TestCase):
    def setUp(self):
        ModelHandler.requests = []
        ModelHandler.finish_reason = "stop"
        ModelHandler.invalid_bbox_on_first = False
        self.server = HTTPServer(("127.0.0.1", 0), ModelHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_cli_classifies_sensor_failure_and_preserves_primary_error_during_cleanup(self):
        for error in (SimulationUnavailable('empty frame'), TimeoutError('HTTP model timeout')):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)/'run'
                world = FakeWorld()
                world.prepare = lambda: None
                world.hold = lambda: (_ for _ in ()).throw(RuntimeError('cleanup failed'))
                arguments = ['simple-agent', '--instruction', 'test', '--model', 'fake',
                             '--model-hardware', 'CPU', '--sim-hardware', 'CPU', '--output', str(output)]
                with patch('sys.argv', arguments), patch.object(cli, 'AirSimWorld', return_value=world), \
                     patch.object(cli, 'run_agent', side_effect=error):
                    with self.assertRaisesRegex(type(error), str(error)):
                        cli.main()
                failure = json.loads((output/'failure.json').read_text())
                self.assertEqual(failure['recoverable_simulation'], isinstance(error, SimulationUnavailable))
                self.assertEqual(failure['error'], str(error))
                self.assertTrue((output/'cleanup_failure.json').exists())

    def test_cli_records_failure_before_world_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'run'
            arguments = ['simple-agent', '--instruction', 'test', '--model', 'fake',
                         '--model-hardware', 'CPU', '--sim-hardware', 'CPU', '--output', str(output)]
            with patch('sys.argv', arguments), patch.object(cli, 'AirSimWorld',
                    side_effect=SimulationUnavailable('connect failed')):
                with self.assertRaisesRegex(SimulationUnavailable, 'connect failed'):
                    cli.main()
            failure = json.loads((output/'failure.json').read_text())
            self.assertIsNone(failure['episode_index'])
            self.assertTrue(failure['recoverable_simulation'])

    def test_cli_full_pipeline_and_gt_is_only_used_for_offline_score(self):
        class World(FakeWorld):
            def prepare(self, position=None, quaternion=None):
                if position is not None:
                    self.location = position.copy()
        world = World()
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)/"tasks.json"
            dataset.write_text(json.dumps([{
                "episode_id": "001", "map_name": "Barnyard_test", "true_name": "SUV",
                "description": "red vehicle", "pose": [[12345, 67890, -10]],
                "start_pose": {"start_position": [0, 0, -10], "start_quaternionr": [0, 0, 0, 1]},
            }]))
            output = Path(tmp)/"run"
            arguments = ["simple-agent", "--dataset", str(dataset), "--scene", "Barnyard_test",
                         "--base-url", self.url, "--model", "fake-model", "--output", str(output),
                         "--model-hardware", "CPU fake HTTP service; no GPU",
                         "--sim-hardware", "CPU synthetic world; no GPU"]
            with patch("sys.argv", arguments), patch.object(cli, "AirSimWorld", return_value=world):
                cli.main()
            result = json.loads((output/"episode_0000/summary.json").read_text())
            self.assertEqual(result["status"], "stop")
            self.assertFalse(result["success"])  # 模型 STOP 不能冒充 GT 成功。
            self.assertFalse(result["comparable_official_metrics"])
            self.assertEqual(len(json.loads((output/"results.json").read_text())), 1)
            self.assertTrue((output/"manifest.json").exists())
            self.assertEqual(len((output/"episode_0000/model.jsonl").read_text().splitlines()), 2)
        self.assertEqual(len(ModelHandler.requests), 2)
        for request in ModelHandler.requests:
            self.assertNotIn("12345", json.dumps(request["messages"]))
            self.assertNotIn("67890", json.dumps(request["messages"]))
            self.assertEqual(request["response_format"]["type"], "json_schema")
            self.assertFalse(request["chat_template_kwargs"]["enable_thinking"])
        self.assertGreater(world.advances, 0)

    def test_truncated_response_is_logged_and_rejected(self):
        ModelHandler.finish_reason = "length"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.jsonl"
            model = VLM(self.url, "fake-model", path)
            with self.assertRaisesRegex(ValueError, "length"):
                model.decide(Context("寻找红车"), FakeWorld().observe(), 0)
            self.assertIn('"finish_reason": "length"', path.read_text())
            self.assertEqual(len(ModelHandler.requests), 2)  # 修复一次仍截断就失败。

    def test_invalid_bbox_is_repaired_once_without_adding_images(self):
        ModelHandler.invalid_bbox_on_first = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.jsonl"
            result = VLM(self.url, "fake-model", path).decide(Context("寻找红车"), FakeWorld().observe(), 0)
            attempts = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(result.action, "stop")
        self.assertEqual([a['attempt'] for a in attempts], [0, 1])
        self.assertEqual(len(ModelHandler.requests), 2)
        second = ModelHandler.requests[-1]['messages']
        self.assertIn('非空', second[-1]['content'])
        images = [p for m in second if isinstance(m['content'], list)
                  for p in m['content'] if p['type'] == 'image_url']
        self.assertEqual(len(images), 4)


def vector_object(values):
    return dict(zip(("x_val", "y_val", "z_val", "w_val"), values))


class FakeRPC:
    def __init__(self):
        self.timestamp = 100_000_000
        self.requests = []
        self.mismatch = False
        self.last = {}
        self.socket = SimpleNamespace(fileno=lambda: 1)
        self.pending_reads = None
        self.hover_position = None
        self.target = np.array([10., 20., -12.])

    def call(self, method, *args, **kwargs):
        if method == "simGetCollisionInfo":
            return dict(has_collided=False, time_stamp=0)
        if method == "simGetImages":
            self.requests = args[0]
            replies = []
            for request in self.requests:
                depth = request["image_type"] == 2
                position = (1, 0, 0) if self.mismatch and not depth else (0, 0, 0)
                replies.append(dict(
                    width=32, height=32, image_data_uint8=bytes([30, 80, 200])*(32*32),
                    image_data_float=[20.]*(32*32), camera_position=vector_object(position),
                    camera_orientation=vector_object((0, 0, 0, 1)), time_stamp=self.timestamp))
            return replies
        if method == "getMultirotorState":
            return dict(landed_state=1, collision=dict(has_collided=False), kinematics_estimated=dict(
                position=vector_object(self.target if self.pending_reads is not None else (0, 0, -2.5)),
                linear_velocity=vector_object((0, 0, 0))))
        if method == "simGetCameraInfo":
            return dict(fov=90)
        if method == "simSetVehiclePose":
            assert args[1] is True
            self.pending_reads = 0
        if method == "simGetVehiclePose":
            ready = self.pending_reads >= 1
            self.pending_reads += 1
            return dict(position=vector_object(self.target if ready else (0, 0, -2.5)))
        if method == "armDisarm":
            return True
        if method in ("moveByVelocity", "moveByVelocityZ"):
            if self.pending_reads is not None:
                assert self.pending_reads >= 2
                self.hover_position = self.target.copy()
            return True


class AirSimBoundaryTest(unittest.TestCase):
    def world(self):
        world = AirSimWorld.__new__(AirSimWorld)
        world.vehicle, world.client = "Drone_1", FakeRPC()
        world.control = world.client
        world._timestamps, world._fovs = {}, {name: 90 for name in ("front", "left", "right", "down")}
        return world

    def test_motion_requests_only_depth_and_uses_response_metadata(self):
        world = self.world()
        observation = world.observe(with_rgb=False)
        self.assertEqual(len(world.client.requests), 4)
        self.assertTrue(all(r["image_type"] == 2 for r in world.client.requests))
        self.assertEqual(observation.views["front"].rgb, b"")
        self.assertEqual(observation.views["front"].depth.dtype, np.float32)
        np.testing.assert_allclose(observation.views["front"].position, [0, 0, 0])
        world.client.timestamp += 1
        observation = world.observe()
        self.assertEqual(len(world.client.requests), 8)
        self.assertTrue(all(not r["compress"] for r in world.client.requests))
        image = Image.open(BytesIO(observation.views["front"].rgb))
        np.testing.assert_allclose(np.array(image)[0, 0], [200, 80, 30], atol=3)

    def test_empty_rgb_and_depth_are_recaptured_without_motion_or_timestamp_pollution(self):
        for index, reason in ((0, 'empty_rgb'), (1, 'empty_depth')):
            with self.subTest(reason=reason):
                world = self.world()
                original, methods = world.client.call, []
                def call(method, *args, **kwargs):
                    methods.append(method)
                    if method == 'simGetImages':
                        self.assertFalse(world._timestamps)
                    result = original(method, *args, **kwargs)
                    if methods == ['simGetImages']:
                        result[index]['width'] = 0
                    return result
                with patch.object(world.client, 'call', side_effect=call):
                    world.observe()
                self.assertEqual(methods, ['simGetImages', 'simGetImages', 'getMultirotorState', 'simGetCollisionInfo'])
                sensor = world.last_observe
                self.assertEqual(sensor['capture_attempts'], 2)
                self.assertEqual(sensor['rejected_rgb_depth_frames'], [])
                self.assertEqual(sensor['rejected_sensor_frames'], [{'view': 'front', 'reason': reason}])

    def test_persistent_empty_capture_is_bounded_and_motion_depth_is_not_retried(self):
        for rgb, attempts in ((True, 3), (False, 1)):
            with self.subTest(rgb=rgb):
                world = self.world()
                original, methods = world.client.call, []
                def call(method, *args, **kwargs):
                    methods.append(method)
                    result = original(method, *args, **kwargs)
                    result[0]['width'] = 0
                    return result
                with patch.object(world.client, 'call', side_effect=call):
                    with self.assertRaises(SimulationUnavailable):
                        world.observe(with_rgb=rgb)
                self.assertEqual(methods, ['simGetImages'] * attempts)
                self.assertFalse(world._timestamps)
                self.assertEqual(len(world.last_observe['rejected_sensor_frames']), attempts)

    def test_repeated_depth_timestamp_is_rejected(self):
        world = self.world()
        world.observe(with_rgb=False)
        with self.assertRaisesRegex(RuntimeError, "未更新"):
            world.observe(with_rgb=False)

    def test_rgb_depth_pose_mismatch_is_rejected(self):
        world = self.world()
        world.client.mismatch = True
        with self.assertRaisesRegex(RuntimeError, "位置错配"):
            world.observe()
        self.assertFalse(world._timestamps)  # 失败帧不能污染成功帧的时间戳。

    def test_rgb_depth_time_mismatch_discards_frame_before_bounded_recapture(self):
        world = self.world()
        original = world.client.call
        captures = []
        def call(method, *args, **kwargs):
            if method == "simGetImages":
                self.assertFalse(world._timestamps)
                world.client.timestamp += 1
                captures.append(method)
            result = original(method, *args, **kwargs)
            if method == "simGetImages" and len(captures) == 1:
                result[0]["time_stamp"] += 150_000_000
            return result
        with patch.object(world.client, "call", side_effect=call):
            observation = world.observe()
        self.assertEqual(len(captures), 2)
        self.assertEqual(observation.views["front"].timestamp, world.client.timestamp)
        self.assertEqual(world.last_observe["capture_attempts"], 2)
        self.assertEqual(world.last_observe["rejected_rgb_depth_frames"], [{"view": "front", "delta_ms": 150.0}])

    def test_persistent_rgb_depth_time_mismatch_still_fails_and_never_moves(self):
        world = self.world()
        original = world.client.call
        methods = []
        def call(method, *args, **kwargs):
            methods.append(method)
            result = original(method, *args, **kwargs)
            if method == "simGetImages":
                result[0]["time_stamp"] += 100_000_001
            return result
        with patch.object(world.client, "call", side_effect=call), self.assertRaisesRegex(RuntimeError, "100ms"):
            world.observe()
        self.assertEqual(methods, ["simGetImages"] * 3)
        self.assertFalse(world._timestamps)
        self.assertEqual(world.last_observe["capture_attempts"], 3)

    def test_rgb_capture_network_timeout_is_not_retried(self):
        world = self.world()
        with patch.object(world.client, "call", side_effect=TimeoutError("simGetImages")) as call:
            with self.assertRaises(TimeoutError):
                world.observe()
        self.assertEqual(call.call_count, 1)

    def test_rgb_depth_time_difference_at_threshold_is_accepted(self):
        world = self.world()
        original = world.client.call
        def call(method, *args, **kwargs):
            result = original(method, *args, **kwargs)
            if method == "simGetImages":
                result[0]["time_stamp"] += 100_000_000
            return result
        with patch.object(world.client, "call", side_effect=call):
            world.observe()
        self.assertEqual(world.last_observe["capture_attempts"], 1)
        self.assertEqual(world.last_observe["rejected_rgb_depth_frames"], [])
        self.assertEqual(world.last_observe["max_rgb_depth_delta_ms"], 100.0)

    def test_spawn_waits_for_render_pose_before_hover(self):
        world = self.world()
        with patch("lightning_vln.simulator.time.sleep"):
            world.prepare(world.client.target, [0, 0, 0, 1])
        np.testing.assert_allclose(world.client.hover_position, world.client.target)

    def test_cold_scene_spawn_sync_can_take_more_than_five_seconds(self):
        world = self.world()
        ticks = iter([0, 0, 6])
        with patch("lightning_vln.simulator.time.sleep"), patch("lightning_vln.simulator.time.monotonic",
                                                                  side_effect=lambda: next(ticks, 6)):
            world.prepare(world.client.target, [0, 0, 0, 1])
        np.testing.assert_allclose(world.client.hover_position, world.client.target)

    def test_spawn_accepts_synced_offset_within_fifty_centimeters(self):
        class DriftRPC(FakeRPC):
            corrected = False
            targets = []
            def call(self, method, *args):
                if method == "moveByVelocityZ":
                    self.targets.append(args[2])
                    self.corrected = abs(args[2]-self.target[2]) < .001
                result = super().call(method, *args)
                if self.pending_reads is not None and not self.corrected:
                    if method == "getMultirotorState":
                        result["kinematics_estimated"]["position"]["z_val"] += .134
                    elif method == "simGetVehiclePose" and self.pending_reads >= 2:
                        result["position"]["z_val"] += .134
                return result
        world = self.world()
        world.client = world.control = DriftRPC()
        requested = world.client.target.copy()
        with patch("lightning_vln.simulator.time.sleep"), patch("lightning_vln.simulator.time.monotonic", return_value=0):
            world.prepare(requested, [0, 0, 0, 1])
        self.assertFalse(world.client.corrected)
        self.assertTrue(all(abs(z-requested[2]-.134) < .001 for z in world.client.targets))
        self.assertAlmostEqual(world.last_prepare["before_hold_estimated_ned_m"][2], requested[2]+.134)
        self.assertAlmostEqual(world.last_prepare["error_m"], .134)
        self.assertEqual(world.last_prepare["position_tolerance_m"], .5)

    def test_spawn_does_not_control_before_render_and_estimate_agree(self):
        world = self.world()
        original = world.client.call
        def mismatched(method, *args):
            if method == "simGetVehiclePose":
                world.client.pending_reads += 1
                return {"position": vector_object(world.client.target + np.array([0, 0, 1]))}
            if method == "moveByVelocityZ":
                self.fail("两套位姿未同步，不得发出定高命令")
            return original(method, *args)
        world.client.call = mismatched
        with patch("lightning_vln.simulator.time.monotonic", side_effect=[0, 0, 16]), self.assertRaisesRegex(SimulationInitializationFailure, "未接近/同步"):
            world.prepare(world.client.target.copy(), [0, 0, 0, 1])

    def test_post_hold_spawn_drift_is_recoverable_without_relaxing_position_gate(self):
        for drift, fails in ((.5, False), (.6550448, True)):
            with self.subTest(drift=drift):
                world = self.world()
                requested = world.client.target.copy()
                def hold():
                    world.last_stop = {'mode': 'controlled'}
                    world.position = lambda: requested + np.array([drift, 0, 0])
                with patch.object(world, 'hold', side_effect=hold), patch('lightning_vln.simulator.time.sleep'):
                    if fails:
                        with self.assertRaises(SimulationInitializationFailure):
                            world.prepare(requested, [0, 0, 0, 1])
                    else:
                        world.prepare(requested, [0, 0, 0, 1])
                self.assertAlmostEqual(world.last_prepare['error_m'], drift)
                self.assertEqual(world.last_prepare['position_tolerance_m'], .5)

    def test_velocity_command_false_is_not_treated_as_success(self):
        world = self.world()
        original = world.control.call
        world.control.call = lambda method, *args: False if method == 'moveByVelocity' else original(method, *args)
        with self.assertRaisesRegex(RuntimeError, "速度命令未完成"):
            world.advance(np.ones(3), .2)

    def test_hold_timeout_is_recoverable_without_relaxing_speed_or_height_gate(self):
        for speed, target_z in ((.228, -2.5), (0., -3.5)):
            with self.subTest(speed=speed, target_z=target_z):
                world = self.world()
                world._hold_z = target_z
                original, methods = world.control.call, []
                def call(method, *args, **kwargs):
                    methods.append(method)
                    result = original(method, *args, **kwargs)
                    if method == 'getMultirotorState':
                        result['kinematics_estimated']['linear_velocity'] = vector_object((speed, 0, 0))
                    return result
                with patch.object(world.control, 'call', side_effect=call), \
                     patch('lightning_vln.simulator.time.monotonic', side_effect=[0, 5.1, 5.1]):
                    with self.assertRaises(SimulationControlFailure):
                        world.hold()
                method = 'moveByVelocity' if speed > .1 else 'moveByVelocityZ'
                self.assertEqual([m for m in methods if m != 'simGetCollisionInfo'], ['getMultirotorState', method, 'getMultirotorState'])
                self.assertEqual(world.last_stop['mode'], 'settle_timeout')
                self.assertEqual(world.last_stop['speed_m_s'], speed)
                self.assertEqual(world._hold_z, target_z)

    def test_hold_false_command_is_recoverable_and_keeps_stop_evidence(self):
        world = self.world()
        original = world.control.call
        def call(method, *args, **kwargs):
            return False if method == 'moveByVelocity' else original(method, *args, **kwargs)
        with patch.object(world.control, 'call', side_effect=call):
            with self.assertRaises(SimulationControlFailure):
                world.hold()
        self.assertEqual(world.last_stop['mode'], 'command_failed')

    def test_hold_accepts_velocity_braking_when_height_is_already_within_gate(self):
        world = self.world()
        control = world.control
        world.client = SimpleNamespace(call=lambda *args: self.fail("停止不应使用图像连接"))
        calls, speeds = [], iter([.5, .07, .04])
        original = control.call
        def call(method, *args):
            calls.append(method)
            result = original(method, *args)
            if method == "getMultirotorState" and len(calls) > 1:
                result["kinematics_estimated"]["linear_velocity"] = vector_object((next(speeds, 0), 0, 0))
            return result
        control.call = call
        world.hold()
        self.assertEqual(calls.count("moveByVelocity"), 2)
        self.assertEqual(calls.count("moveByVelocityZ"), 0)
        self.assertLessEqual(world.last_stop["speed_m_s"], .1)
        self.assertLessEqual(abs(world.last_stop["z_error_m"]), .5)
        self.assertEqual(world.last_stop["mode"], "velocity_braked_within_height")

    def test_moving_hold_rebrakes_without_resetting_existing_height(self):
        for height_error in (.12, .6):
            with self.subTest(height_error=height_error):
                world = self.world()
                world._hold_z = -2.5
                original = world.control.call
                methods = []
                state_reads = 0
                def call(method, *args, **kwargs):
                    nonlocal state_reads
                    methods.append(method)
                    result = original(method, *args, **kwargs)
                    if method == 'getMultirotorState':
                        state_reads += 1
                        speed = .22 if state_reads == 1 else .03
                        result['kinematics_estimated']['linear_velocity'] = vector_object((speed, 0, 0))
                        error = height_error if 'moveByVelocityZ' not in methods else 0
                        result['kinematics_estimated']['position']['z_val'] = -2.5+error
                    if method == 'moveByVelocityZ':
                        self.assertEqual(args[2], -2.5)
                    return result
                with patch.object(world.control, 'call', side_effect=call):
                    world.hold()
                self.assertEqual([m for m in methods if m != 'simGetCollisionInfo'][1], 'moveByVelocity')
                self.assertEqual(world._hold_z, -2.5)
                self.assertLessEqual(world.last_stop['speed_m_s'], .1)
                self.assertLessEqual(abs(world.last_stop['z_error_m']), .5)
                if height_error <= .5:
                    self.assertNotIn('moveByVelocityZ', methods)
                    self.assertEqual(world.last_stop['mode'], 'rebraked')
                else:
                    self.assertIn('moveByVelocityZ', methods)

    def test_stable_hold_only_reads_state_and_preserves_height_target(self):
        world = self.world()
        world._hold_z = -2.5
        original = world.control.call
        def call(method, *args):
            self.assertIn(method, ("getMultirotorState", "simGetCollisionInfo"))
            return original(method, *args)
        world.control.call = call
        world.hold()
        self.assertEqual(world.last_stop["mode"], "already_stable")
        self.assertEqual(world.last_stop["commands"], 0)
        self.assertEqual(world._hold_z, -2.5)

    def test_failed_turn_invalidates_stable_hold_cache(self):
        world = self.world()
        world.hold()
        original = world.control.call
        def call(method, *args, **kwargs):
            if method == "rotateToYaw":
                self.assertGreater(kwargs["timeout"], args[1])
                return False
            return original(method, *args, **kwargs)
        world.control.call = call
        self.assertFalse(world.face(np.array([1., 0, -2.5])))
        world.hold()
        self.assertEqual(world.last_stop["mode"], "velocity_braked_within_height")
        self.assertGreaterEqual(world.last_stop["commands"], 1)

    def test_repeated_holds_keep_original_altitude_until_next_motion(self):
        world = self.world()
        client = world.control
        original, methods, current_z = client.call, [], [-2.5]
        def call(method, *args):
            methods.append(method)
            result = original(method, *args)
            if method == "getMultirotorState":
                result["kinematics_estimated"]["position"]["z_val"] = current_z[0]
            return result
        client.call = call
        world.hold()
        self.assertEqual(world._hold_z, -2.5)
        world.hold()
        self.assertEqual(world._hold_z, -2.5)
        world.advance(np.array([1., 0, 0]), .2)
        current_z[0] = -2.49
        world.hold()
        self.assertEqual(world._hold_z, -2.49)
        self.assertNotIn("moveByVelocityZ", methods)

    def test_settings_keep_metric_depth_and_realtime_clock(self):
        config = settings(30100)
        self.assertEqual(config["ClockSpeed"], 1)
        self.assertEqual(config["ApiServerPort"], 30100)
        cameras = config["Vehicles"]["Drone_1"]["Cameras"]
        self.assertEqual(set(cameras), {"0", "1", "2", "3"})
        self.assertTrue(all(camera["CaptureSettings"][1]["ImageType"] == 2 for camera in cameras.values()))
        self.assertTrue(all(capture["FOV_Degrees"] == 110 for camera in cameras.values()
                            for capture in camera["CaptureSettings"]))
        # 三路水平相机应位于机体外，避免把机体自身当成近障碍。
        self.assertEqual([(cameras[n]["X"], cameras[n]["Y"]) for n in ("0", "1", "2")],
                         [(.5, 0), (0, -.5), (0, .5)])


class DiagnosticTest(unittest.TestCase):
    def test_only_actual_spawn_is_scored(self):
        task = {"pose": [[0, 0, 0]], "start_pose": {"start_position": [0, 0, 0]}}
        summary = {"status": "stop", "position_ned_m": [1, 0, -12],
                   "start_position_ned_m": [0, 0, -12]}
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"event": "start", "position_ned_m": summary["start_position_ned_m"]},
                    {"event": "final", "position_ned_m": summary["position_ned_m"]}]
            (Path(tmp)/"trajectory.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            result = cli.score_episode(task, Path(tmp), summary, 1)
        self.assertFalse(result["oracle_success"])
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()
