"""AirSim 使用的同步 MessagePack RPC；单连接、无重试、无旧 SDK/Tornado。"""

import math
import socket
import time

import msgpack

from lightning_vln.core.errors import SimulationDisconnected, SimulationTimeout


class RPC:
    def __init__(self, host: str, port: int, timeout: float = 10):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("RPC timeout 必须为有限正数")
        self.timeout = timeout
        try:
            self.socket = socket.create_connection((host, port), timeout=timeout)
        except TimeoutError as error:
            raise SimulationTimeout(f"AirSim 连接超时：{host}:{port}") from error
        except ConnectionError as error:
            raise SimulationDisconnected(f"AirSim 连接失败：{host}:{port}") from error
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # AirSim 将 PNG/原始像素编码成 STR，不能按 UTF-8 解码值。
        # 仅 map 的字段名转为 str；大型浮点数组由 C 扩展直接解包。
        self.unpacker = msgpack.Unpacker(raw=True, max_buffer_size=64*1024*1024,
                                        object_hook=lambda d: {k.decode(): v for k, v in d.items()})
        self.sequence = 0
        self.last = {}

    def close(self):
        self.socket.close()

    def call(self, method: str, *arguments, timeout: float | None = None):
        timeout = self.timeout if timeout is None else timeout
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("RPC timeout 必须为有限正数")
        self.sequence += 1
        started = time.monotonic()
        deadline = started + timeout
        decode_seconds, received = 0.0, 0
        try:
            self.socket.settimeout(timeout)
            self.socket.sendall(msgpack.packb([0, self.sequence, method, arguments], use_bin_type=True))
            while True:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"RPC 超时：{method}")
                self.socket.settimeout(remaining)
                data = self.socket.recv(1024*1024)
                if not data:
                    raise ConnectionError(f"RPC 连接关闭：{method}")
                received += len(data)
                tick = time.monotonic()
                self.unpacker.feed(data)
                messages = list(self.unpacker)
                decode_seconds += time.monotonic()-tick
                if not messages:
                    continue
                if len(messages) != 1 or len(messages[0]) != 4 or messages[0][:2] != [1, self.sequence]:
                    raise RuntimeError(f"RPC 响应 ID/格式错误：{method}")
                _, _, error, result = messages[0]
                self.last = {"method": method, "total_s": time.monotonic()-started,
                             "decode_s": decode_seconds, "bytes": received}
                if error is not None:
                    raise RuntimeError(f"RPC {method}：{error!r}")
                return result
        except TimeoutError as error:
            self.close()
            raise SimulationTimeout(f"RPC 超时：{method}，期限 {timeout:g}s，"
                               f"已等待 {time.monotonic()-started:.3f}s") from error
        except ConnectionError as error:
            self.close()
            raise SimulationDisconnected(f"RPC 连接中断：{method}") from error
        except BaseException:
            # 超时后不能把迟到回复当作下一次响应，更不能自动重放运动命令。
            self.close()
            raise
