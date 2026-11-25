"""小型 AirSim 边界：原生 RPC、米制 RGB-D 和实际物理运动。"""

import math
import time
from io import BytesIO

import numpy as np
from PIL import Image

from lightning_vln.core.geometry import vector
from lightning_vln.core.errors import SimulationUnavailable, SimulationControlFailure, SimulationInitializationFailure
from lightning_vln.core.rpc import RPC
from lightning_vln.core.types import Observation, View, VIEWS

POSITION_TOLERANCE_M = 0.5
STOP_COMMAND_S = 0.05
DECISION_CAPTURE_ATTEMPTS = 3
COLLISION_PROTOCOL = "airsim_dedicated_collision_latched_v1"
SPAWN_COLLISION_POLICY = "terminal_collision_v1"


class SensorCaptureError(SimulationUnavailable):
    def __init__(self, message: str, view: str, reason: str):
        self.view, self.reason = view, reason
        super().__init__(message)


class RGBDepthTimeMismatch(SensorCaptureError):
    def __init__(self, view: str, delta_ns: int):
        self.view, self.delta_ms = view, delta_ns / 1_000_000
        super().__init__(f"RGB 与 depth 时间差超过 100ms：{view} ({self.delta_ms:.3f}ms)",
                         view, "rgb_depth_time_mismatch")


def xyz(value) -> np.ndarray:
    return np.array([value[k] for k in ("x_val", "y_val", "z_val")], dtype=float)


def xyzw(value) -> np.ndarray:
    return np.array([value[k] for k in ("x_val", "y_val", "z_val", "w_val")], dtype=float)


def fields(values) -> dict:
    return dict(zip(("x_val", "y_val", "z_val", "w_val"), map(float, values)))


class AirSimWorld:
    dedicated_collision = True

    def __init__(self, host: str = "127.0.0.1", port: int = 30100,
                 vehicle: str = "Drone_1", rpc_timeout: float = 10):
        self.vehicle = vehicle
        self.address = host, port, rpc_timeout
        self.client = RPC(*self.address)
        try:
            if not self.client.call("ping"):
                raise RuntimeError("AirSim ping 失败")
            if self.client.call("getMinRequiredClientVersion") > 1:
                raise RuntimeError("AirSim 需要更高版本的客户端")
            # 图像超时不占用停止通道。每条连接只允许一个同步调用。
            self.control = RPC(*self.address)
        except BaseException:
            self.client.close()
            raise
        self.last_observe = {}
        self._collision_baseline_stamp = -1
        self._collision_latched = None

    def collision_info(self, client=None):
        # 打包UE的getMultirotorState.collision会永久漏报，必须查询专用接口。
        info = (self.client if client is None else client).call("simGetCollisionInfo", self.vehicle)
        if not isinstance(info, dict) or not isinstance(info.get("has_collided"), bool):
            raise SimulationUnavailable("AirSim 专用碰撞接口返回无效信息")
        stamp = int(info.get("time_stamp", 0))
        if info["has_collided"] and stamp <= 0:
            raise SimulationUnavailable("AirSim 碰撞事件缺少有效时间戳")
        if info["has_collided"] and stamp > getattr(self, "_collision_baseline_stamp", -1):
            self._collision_latched = info.copy()
        latched = getattr(self, "_collision_latched", None)
        return latched.copy() if latched else dict(info, has_collided=False)

    def _reset_collision_baseline(self):
        if not self.dedicated_collision:
            return
        info = self.client.call("simGetCollisionInfo", self.vehicle)
        if not isinstance(info, dict) or not isinstance(info.get("has_collided"), bool):
            raise SimulationUnavailable("AirSim 专用碰撞接口返回无效信息")
        self._collision_baseline_stamp = int(info.get("time_stamp", 0))
        self._collision_latched = None

    def collision_record(self):
        info = getattr(self, "_collision_latched", None) or {}
        name = info.get("object_name")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        spawn = bool(getattr(self, 'last_prepare', {}).get('spawn_collision', False))
        return dict(protocol=COLLISION_PROTOCOL, has_collided=bool(info), spawn_collision=spawn,
                    collision_phase='spawn' if spawn else 'navigation' if info else None,
                    time_stamp=info.get("time_stamp"), object_name=name)

    def close(self):
        self.client.close()
        self.control.close()

    def prepare(self, position=None, quaternion=None):
        started = time.monotonic()
        self._hold_z = None
        self.last_prepare = {}
        self.control.call("reset")
        self._reset_collision_baseline()
        self.control.call("simPause", False)
        self.control.call("enableApiControl", True, self.vehicle)
        if not self.control.call("armDisarm", True, self.vehicle):
            raise RuntimeError("AirSim 解锁失败")
        self._timestamps = {}
        self._fovs = {name: float(self.client.call("simGetCameraInfo", camera, self.vehicle, False)["fov"])
                      for name, camera in VIEWS.items()}
        if position is None and self.client.call("getMultirotorState", self.vehicle)["landed_state"] == 0:
            if not self.control.call("takeoff", 10.0, self.vehicle, timeout=20.0):
                raise RuntimeError("AirSim 起飞未到达目标高度；请指定有效的空中出生点")
        if position is not None:
            position = vector(position, 3)
            quaternion = vector(quaternion, 4)
            if np.linalg.norm(quaternion) < 1e-12:
                raise ValueError("出生点四元数不能为零")
            quaternion = quaternion / np.linalg.norm(quaternion)
            self._set_spawn_pose(position, quaternion)
            # 位姿同步与到达请求位置是不同条件；出生点允许50cm误差。
            deadline = time.monotonic() + 15
            while True:
                actual = self.position()
                rendered = xyz(self.client.call("simGetVehiclePose", self.vehicle)["position"])
                synchronized = np.linalg.norm(actual-rendered) <= 0.1
                near_spawn = all(np.linalg.norm(p-position) <= POSITION_TOLERANCE_M for p in (actual, rendered))
                if self.collided() or (synchronized and near_spawn):
                    break
                if time.monotonic() >= deadline:
                    raise SimulationInitializationFailure(f"出生点15s内未接近/同步到物理状态：requested={position.tolist()}, "
                                       f"estimated={actual.tolist()}, ue={rendered.tolist()}")
                time.sleep(0.02)
            self.last_prepare = {"requested_ned_m": position.tolist(),
                                 "position_tolerance_m": POSITION_TOLERANCE_M,
                                 "before_hold_estimated_ned_m": actual.tolist(),
                                 "before_hold_ue_ned_m": rendered.tolist()}
        self.hold()
        spawn_collision = self.collided()
        if position is not None:
            actual = self.position()
            rendered = xyz(self.client.call("simGetVehiclePose", self.vehicle)["position"])
            self.last_prepare.update(actual_start_ned_m=actual.tolist(), ue_start_ned_m=rendered.tolist(),
                                     error_m=float(np.linalg.norm(actual-position)))
            if not spawn_collision and any(np.linalg.norm(p-position) > POSITION_TOLERANCE_M for p in (actual, rendered)):
                raise SimulationInitializationFailure(f"定高后出生点偏离请求位置：{self.last_prepare}")
        if spawn_collision and not self.dedicated_collision:
            raise RuntimeError("出生点发生碰撞")
        if spawn_collision:
            # 出生接触是当前任务的正常碰撞终态，不是批量/仿真故障。
            self.last_prepare.setdefault('position_tolerance_m', POSITION_TOLERANCE_M)
            self.last_prepare.setdefault('error_m', 0.0)
            self.last_prepare.update(spawn_collision=True, collision_phase='spawn')
        if self.last_prepare:
            self.last_prepare.update(elapsed_s=time.monotonic()-started, stop=self.last_stop.copy())

    def _set_spawn_pose(self, position, quaternion):
        # 清除起飞定高目标；仅出生点初始化允许跳过沿途碰撞扫掠。
        self.advance(np.zeros(3), 0.1)
        # 只忽略出生前事件；出生后的接触一律保留，直到下一任务初始化。
        self._reset_collision_baseline()
        self.control.call("simSetVehiclePose", {"position": fields(position),
                          "orientation": fields(quaternion)}, True, self.vehicle)

    def state(self):
        state = self.client.call("getMultirotorState", self.vehicle)
        if self.dedicated_collision:
            state["collision"] = self.collision_info()
        return state

    def position(self) -> np.ndarray:
        return xyz(self.state()["kinematics_estimated"]["position"])

    def collided(self) -> bool:
        return bool((self.collision_info() if self.dedicated_collision else self.state()["collision"])["has_collided"])

    def hold(self):
        # 只重建停止通道；从不重放失败的运动命令。
        if self.control.socket.fileno() < 0:
            self.control = RPC(*self.address)
        # 此 packaged 场景的 hover 返回 False 且仍有约 0.5m/s 残速。
        # 横向零速度 + 固定高度；反复请求三轴零速度会累计下沉。
        started = time.monotonic()
        state = self.control.call("getMultirotorState", self.vehicle)
        initial = vector(xyz(state["kinematics_estimated"]["position"]), 3)
        position = initial
        speed = float(np.linalg.norm(xyz(state["kinematics_estimated"]["linear_velocity"])))
        new_target = getattr(self, "_hold_z", None) is None
        rebraking = not new_target and speed > 0.1
        braking = new_target or rebraking
        commands = 0
        def record(mode):
            self.last_stop = {"elapsed_s": time.monotonic()-started, "speed_m_s": speed,
                              "drift_m": float(np.linalg.norm(position-initial)),
                              "target_z_ned_m": self._hold_z, "z_error_m": float(position[2]-self._hold_z),
                              "commands": commands, "command_duration_s": STOP_COMMAND_S,
                              "position_tolerance_m": POSITION_TOLERANCE_M, "mode": mode}
        if new_target:
            self._hold_z = float(initial[2])
        if self.dedicated_collision and self.collision_info(self.control)["has_collided"]:
            # 接触后仅制动一次，不为保持接触点高度持续推进或报成仿真故障。
            self.control.call("moveByVelocity", 0.0, 0.0, 0.0, STOP_COMMAND_S, 0,
                              {"is_rate": True, "yaw_or_rate": 0.0}, self.vehicle)
            commands += 1
            record("collision_brake")
            return
        if not new_target and speed <= 0.1 and abs(initial[2]-self._hold_z) <= POSITION_TOLERANCE_M:
            # 已完成制动、也没有新运动：一次状态查询即可，不重发控制命令。
            record("already_stable")
            return
        while True:
            if self.dedicated_collision and self.collision_info(self.control)["has_collided"]:
                record("collision_brake")
                return
            # 先三轴零速度制动，再锁定高度；每50ms确认一次，达标立即返回。
            method = "moveByVelocity" if braking else "moveByVelocityZ"
            z = 0.0 if braking else self._hold_z
            if not self.control.call(method, 0.0, 0.0, z, STOP_COMMAND_S, 0,
                                     {"is_rate": True, "yaw_or_rate": 0.0}, self.vehicle):
                record("command_failed")
                raise SimulationControlFailure("AirSim 定高制动未完成")
            commands += 1
            state = self.control.call("getMultirotorState", self.vehicle)
            speed = float(np.linalg.norm(xyz(state["kinematics_estimated"]["linear_velocity"])))
            position = xyz(state["kinematics_estimated"]["position"])
            if speed <= 0.1 and (braking or abs(position[2]-self._hold_z) <= POSITION_TOLERANCE_M):
                if braking:
                    # 零速度制动已经同时满足速度和高度门禁时直接接受。此时
                    # 强制切换位置定高会使 packaged 场景的高空控制重新失稳。
                    if abs(position[2]-self._hold_z) <= POSITION_TOLERANCE_M:
                        record("rebraked" if rebraking else "velocity_braked_within_height")
                        return
                    braking = False
                    continue
                record("controlled")
                return
            if time.monotonic()-started >= 5:
                record("settle_timeout")
                raise SimulationControlFailure(f"制动/定高未完成：{speed:.3f} m/s，高度误差 {position[2]-self._hold_z:.3f} m")

    def face(self, goal) -> bool:
        if self.dedicated_collision and self.collided():
            return False
        delta = np.asarray(goal) - self.position()
        if np.linalg.norm(delta[:2]) > 0.05:
            yaw = math.degrees(math.atan2(delta[1], delta[0]))
            # 转向也会接管控制器；即使调用失败，下一次hold仍须重新发送停止命令。
            self._hold_z = None
            # 客户端期限须长于控制器的10s执行期限；提前返回不额外等待。
            # False 是控制器未达标，交给 Agent 停稳后重观察；网络错误仍抛出。
            return bool(self.control.call("rotateToYaw", yaw, 10.0, 5.0, self.vehicle, timeout=20.0))
        return True

    def advance(self, velocity: np.ndarray, seconds: float):
        # 新的运动建立新的停止高度；重复 hold 不把微小残余误差累加进目标。
        self._hold_z = None
        if not self.control.call("moveByVelocity", *map(float, velocity), float(seconds), 0,
                                 {"is_rate": True, "yaw_or_rate": 0.0}, self.vehicle):
            if self.dedicated_collision and self.collided():
                return
            raise RuntimeError("AirSim 速度命令未完成")

    def observe(self, with_rgb: bool = True) -> Observation:
        # 决策观测在hold后采集；丢弃坏帧并有限重取，不重放任何运动。
        # 控制中的纯深度观测及RPC/位姿等其他错误仍立即失败。
        started = time.monotonic()
        rejected = []
        rejected_times = []
        attempts = DECISION_CAPTURE_ATTEMPTS if with_rgb else 1
        for attempt in range(attempts):
            try:
                observation = self._observe_once(with_rgb)
            except SensorCaptureError as error:
                record = {"view": error.view, "reason": error.reason}
                if isinstance(error, RGBDepthTimeMismatch):
                    record["delta_ms"] = error.delta_ms
                    rejected_times.append({"view": error.view, "delta_ms": error.delta_ms})
                rejected.append(record)
                self.last_observe = {"rgb": with_rgb, "capture_attempts": attempt + 1,
                                     "rejected_rgb_depth_frames": rejected_times,
                                     "rejected_sensor_frames": rejected,
                                     "total_s": time.monotonic() - started}
                if attempt + 1 == attempts:
                    raise
                time.sleep(.05)
            else:
                self.last_observe.update(capture_attempts=attempt + 1, rejected_rgb_depth_frames=rejected_times,
                                         rejected_sensor_frames=rejected,
                                         total_s=time.monotonic() - started)
                return observation

    def _observe_once(self, with_rgb: bool) -> Observation:
        started = time.monotonic()
        requests = []
        for camera in VIEWS.values():
            if with_rgb:
                requests.append(dict(camera_name=camera, image_type=0, pixels_as_float=False, compress=False))
            requests.append(dict(camera_name=camera, image_type=2, pixels_as_float=True, compress=False))
        responses = self.client.call("simGetImages", requests, self.vehicle, False)
        rpc = self.client.last.copy()
        convert_started = time.monotonic()
        stride = 2 if with_rgb else 1
        if len(responses) != stride*len(VIEWS):
            raise SensorCaptureError("AirSim 未返回完整四路图像", "all", "incomplete_images")
        views, timestamps, rgb_depth_deltas = {}, {}, []
        for index, (name, camera) in enumerate(VIEWS.items()):
            depth = responses[stride*index+stride-1]
            if min(depth["width"], depth["height"]) <= 0:
                raise SensorCaptureError("AirSim 返回空图像：" + name, name, "empty_depth")
            timestamp = int(depth["time_stamp"])
            if timestamp <= self._timestamps.get(name, -1):
                raise SensorCaptureError("AirSim 返回未更新的 depth：" + name, name, "stale_depth")
            timestamps[name] = timestamp
            qd = xyzw(depth["camera_orientation"])
            encoded = b""
            if with_rgb:
                rgb = responses[stride*index]
                if min(rgb["width"], rgb["height"]) <= 0:
                    raise SensorCaptureError("AirSim 返回空 RGB：" + name, name, "empty_rgb")
                if not np.isclose(rgb["width"]/rgb["height"], depth["width"]/depth["height"]):
                    raise RuntimeError("RGB 与 depth 宽高比不一致：" + name)
                if np.linalg.norm(xyz(rgb["camera_position"])-xyz(depth["camera_position"])) > 0.1:
                    raise RuntimeError("RGB 与 depth 相机位置错配：" + name)
                qr = xyzw(rgb["camera_orientation"])
                if abs(np.dot(qr/np.linalg.norm(qr), qd/np.linalg.norm(qd))) < 0.999:
                    raise RuntimeError("RGB 与 depth 相机朝向错配：" + name)
                delta_ns = abs(int(rgb["time_stamp"])-timestamp)
                if delta_ns > 100_000_000:
                    raise RGBDepthTimeMismatch(name, delta_ns)
                rgb_depth_deltas.append(delta_ns / 1_000_000)
                buffer = BytesIO()
                # AirSim 未压缩回读是 BGR，PNG 解码后才是 RGB。
                Image.frombytes("RGB", (rgb["width"], rgb["height"]), rgb["image_data_uint8"], "raw", "BGR").save(
                    buffer, "JPEG", quality=85)
                encoded = buffer.getvalue()
            meters = np.asarray(depth["image_data_float"], dtype=np.float32).reshape(depth["height"], depth["width"])
            views[name] = View(encoded, meters, xyz(depth["camera_position"]), qd,
                               self._fovs[name], timestamp)
        converted_at = time.monotonic()
        state = self.state()
        self._timestamps.update(timestamps)
        self.last_observe = {"rgb": with_rgb, "rpc": rpc, "convert_s": converted_at-convert_started,
                             "state_s": time.monotonic()-converted_at,
                             "total_s": time.monotonic()-started,
                             "max_rgb_depth_delta_ms": max(rgb_depth_deltas, default=0),
                             "collision": bool(state["collision"]["has_collided"])}
        return Observation(xyz(state["kinematics_estimated"]["position"]), views)
