"""四视图 settings 和单场景启动；中断时回收自己拥有的 UE 进程组。"""

import argparse
import json
import os
import pwd
import signal
import subprocess
import threading
import time
from pathlib import Path

from lightning_vln.core.gpu_isolation import isolate_scene


def stop_process_group(process):
    # cleanup 期间避免第二次 Ctrl-C 打断回收。
    handlers = ({s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
                if threading.current_thread() is threading.main_thread() else {})
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        deadline = time.monotonic()+15
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(.2)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=15)
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def settings(port: int) -> dict:
    cameras = {}
    # 水平传感器在机体外；中心相机会把螺旋桨写入 RGB 和近距离 depth。
    for name, x, y, yaw, pitch in (("0", .5, 0, 0, 0), ("1", 0, -.5, -90, 0),
                                  ("2", 0, .5, 90, 0), ("3", 0, 0, 0, -90)):
        cameras[name] = {"X": x, "Y": y, "Z": 0, "Yaw": yaw, "Pitch": pitch,
                         "CaptureSettings": [
                             {"ImageType": 0, "Width": 512, "Height": 512, "FOV_Degrees": 110,
                              "AutoExposureMaxBrightness": 1, "AutoExposureMinBrightness": 0.03},
                             {"ImageType": 2, "Width": 256, "Height": 256, "FOV_Degrees": 110}]}
    return {"SettingsVersion": 1.2, "SimMode": "Multirotor", "ClockSpeed": 1,
            "ApiServerPort": port, "Vehicles": {"Drone_1": {
                "VehicleType": "SimpleFlight", "Cameras": cameras}}}


def main():
    parser = argparse.ArgumentParser(description="启动一个已有 UAV-ON packaged Linux 场景")
    parser.add_argument("--script", type=Path, required=True, help="场景 .sh 的绝对路径")
    parser.add_argument("--output", type=Path, required=True, help="全新 settings/log 目录")
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--isolate-gpu", action="store_true", help="仅向 UE 暴露指定物理 GPU；需要已有 bwrap")
    parser.add_argument("--unreal-user", default="uavonrunner", help="root 运行时使用的已有普通用户")
    parser.add_argument("--settings-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.gpu < 0:
        parser.error("无效的 port/gpu")
    if not args.settings_only and not args.script.is_file():
        parser.error("场景脚本不存在：" + str(args.script))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o755)
    config = output / "settings.json"
    configuration = settings(args.port)
    if args.isolate_gpu:
        configuration['LocalHostIp'] = '127.0.0.1'
    config.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    config.chmod(0o644)
    runtime = output / "runtime"
    runtime.mkdir(mode=0o700)
    arguments = ["-RenderOffscreen", "-NoSound", "-NoVSync",
                 f"-GraphicsAdapter={0 if args.isolate_gpu else args.gpu}",
                 f"-Abslog={runtime / 'unreal.log'}", f"-UserDir={runtime}/", f"--settings={config}"]
    environment, isolation = None, None
    if args.isolate_gpu:
        command, environment, isolation = isolate_scene(args.script, args.gpu, output, runtime, arguments)
    else:
        command = ["bash", str(args.script.resolve()), *arguments]
    if not args.settings_only and os.geteuid() == 0:
        try:
            account = pwd.getpwnam(args.unreal_user)
        except KeyError:
            parser.error("UE 拒绝 root；请提供已存在的 --unreal-user 或直接用普通用户启动")
        os.chown(runtime, account.pw_uid, account.pw_gid)
        command = ["runuser", "-u", account.pw_name, "--", "env",
                   f"HOME={runtime}", f"XDG_RUNTIME_DIR={runtime}", *command]
    print(json.dumps({"settings": str(config), "command": command, "device_isolation": isolation}, ensure_ascii=False), flush=True)
    if not args.settings_only:
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, interrupt)
        process = subprocess.Popen(command, start_new_session=True, env=environment)
        (output / "process.json").write_text(json.dumps({"pid": process.pid, "pgid": process.pid,
                                                       "command": command, "device_isolation": isolation}))
        try:
            code = process.wait()
        finally:
            stop_process_group(process)
        raise SystemExit(code)


if __name__ == "__main__":
    main()
