"""本轮独立批量适配：端口释放有界等待，导航与结算沿用冻结入口。"""
import errno
import time

import run_uavon_tuning_subset as batch


def wait_scene_port(port, *, timeout=30, interval=.5, check=None):
    check = check or original_check_scene_port
    deadline = time.monotonic() + timeout
    while True:
        try:
            check(port)
            return
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
            if time.monotonic() >= deadline:
                raise batch.SimulationUnavailable(
                    f'场景端口{port}等待{timeout:g}s仍被占用；归属未确认，不清理占用者：{error}') from error
            time.sleep(min(interval, max(0, deadline-time.monotonic())))


original_check_scene_port = batch.check_scene_port


def main():
    batch.check_scene_port = wait_scene_port
    batch.RECOVERY_POLICY = dict(batch.RECOVERY_POLICY, version=4,
                                port_release_wait_s=30, port_busy_recoverable=True)
    batch.main()


if __name__ == '__main__':
    main()
