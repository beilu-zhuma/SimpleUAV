"""仅在自有 UE 子进程中暴露指定物理 GPU；不修改主机设备或场景资产。"""

import os
from pathlib import Path
import re
import shlex
import shutil


PROJECTAIRSIM_GAME_MODE = '/Script/ProjectAirSim.ProjectAirSimGameMode'


def game_mode_arguments(arguments: list[str], game_mode: str) -> list[str]:
    """用URL选项覆盖GameMode；空地图部分保留场景包原默认地图。"""
    if not re.fullmatch(r'/Script/[A-Za-z0-9_]+\.[A-Za-z0-9_]+', game_mode):
        raise ValueError('无效的GameMode类路径')
    result = list(arguments)
    urls = [i for i, value in enumerate(result) if not value.startswith('-')]
    if len(urls) > 1 or (urls and urls[0] != 0):
        raise ValueError('地图URL必须唯一且位于启动参数首位')
    index = urls[0] if urls else 0
    url = result[index] if urls else ''
    modes = [v.split('=', 1)[1] for v in url.split('?')[1:] if v.lower().startswith('game=')]
    if len(modes) > 1 or (modes and modes[0] != game_mode):
        raise ValueError('重复或冲突的GameMode参数')
    if not modes:
        url += '?game=' + game_mode
    if urls:
        result[index] = url
    else:
        result.insert(0, url)
    return result


def confirm_game_mode(log: Path, game_mode: str, *, deadline=None, is_alive=None) -> dict:
    """缺少类日志为pending；共享原启动期限，明确错配仍立即拒绝。"""
    import time
    started = time.monotonic()
    deadline = started + 30 if deadline is None else deadline
    expected = game_mode.rsplit('.', 1)[-1]
    while True:
        if is_alive is not None and not is_alive():
            raise ConnectionError('GameMode确认前本轮UE已退出')
        try:
            content = log.read_text(errors='replace')
        except FileNotFoundError:
            content = ''
        classes = re.findall(r"Game class is '([^']+)'", content)
        actual = classes[-1] if classes else None
        if actual is not None:
            if actual != expected:
                raise RuntimeError(f'GameMode启动适配错误：请求{expected}，实际{actual}')
            maps = re.findall(r'LogLoad: LoadMap: ([^\r\n]+)', content)
            return dict(requested_game_mode=game_mode, actual_game_mode=actual,
                        loaded_map=maps[-1] if maps else None,
                        confirmation_wait_s=time.monotonic()-started,
                        confirmed_unix=time.time())
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'GameMode确认超时：请求{expected}，启动期限内未读到实际类日志')
        time.sleep(min(.2, remaining))


def packaged_command(script: Path, require_executable: bool = True) -> list[str]:
    """只读取 UE 打包脚本的标准启动行，避免执行其中的 chmod。"""
    root = script.resolve().parent
    for line in script.read_text().splitlines():
        prefixes = ('$UE4_PROJECT_ROOT/', '$UE_PROJECT_ROOT/')
        prefix = next((p for p in prefixes if line.strip().startswith('"' + p)), None)
        if prefix is None:
            continue
        fields = shlex.split(line)
        if len(fields) != 3 or fields[-1] != '$@' or not re.fullmatch(r'[\w-]+', fields[1]):
            continue
        binary = (root / fields[0].removeprefix(prefix)).resolve()
        if (not binary.is_relative_to(root) or not binary.is_file()
                or (require_executable and not os.access(binary, os.X_OK))):
            raise ValueError('打包场景二进制不存在、不可执行或超出场景目录：' + str(binary))
        return [str(binary), fields[1]]
    raise ValueError('隔离启动只支持标准 UE 打包脚本，拒绝执行未识别的 shell：' + str(script))


def gpu_device(gpu: int) -> tuple[str, list[Path]]:
    """按 NVIDIA minor 与 PCI 地址匹配 DRM，不能假定 cardN 等于 GPU N。"""
    for information in Path('/proc/driver/nvidia/gpus').glob('*/information'):
        content = information.read_text()
        minor = re.search(r'^Device Minor:\s*(\d+)', content, re.MULTILINE)
        if minor is None or int(minor[1]) != gpu:
            continue
        uuid = re.search(r'^GPU UUID:\s*(GPU-\S+)', content, re.MULTILINE)
        if uuid is None:
            raise ValueError('缺少 GPU UUID：' + str(information))
        nodes = [Path(f'/dev/nvidia{gpu}'), Path('/dev/nvidiactl'), Path('/dev/nvidia-uvm')]
        if Path('/dev/nvidia-modeset').exists():
            nodes.append(Path('/dev/nvidia-modeset'))
        for entry in sorted(Path('/sys/class/drm').iterdir()):
            if (re.fullmatch(r'(card\d+|renderD\d+)', entry.name)
                    and (entry / 'device').resolve().name == information.parent.name):
                nodes.append(Path('/dev/dri') / entry.name)
        if not all(p.exists() for p in nodes) or not any(p.name.startswith('renderD') for p in nodes):
            raise ValueError('指定 GPU 的 NVIDIA/DRM 设备节点不完整')
        return uuid[1], nodes
    raise ValueError('没有找到指定物理 GPU：' + str(gpu))


def isolate_scene(script: Path, gpu: int, output: Path, runtime: Path,
                  arguments: list[str], executable_copy: Path | None = None,
                  game_mode: str | None = None) -> tuple[list[str], dict, dict]:
    executable = shutil.which('bwrap')
    if executable is None:
        raise ValueError('GPU 隔离需要已安装的 bwrap，不自动安装或退回未隔离启动')
    if os.geteuid() == 0:
        raise ValueError('GPU 隔离入口须由普通用户运行')
    uuid, nodes = gpu_device(gpu)
    binary = packaged_command(script, require_executable=executable_copy is None)
    if executable_copy is not None:
        executable_copy = executable_copy.resolve()
        if (not executable_copy.is_relative_to(output.resolve())
                or not executable_copy.is_file() or not os.access(executable_copy, os.X_OK)):
            raise ValueError('执行副本必须位于本轮输出目录且可执行')
    temporary = runtime / 'tmp'
    temporary.mkdir()
    command = [executable, '--unshare-user', '--die-with-parent', '--ro-bind', '/', '/', '--dev', '/dev']
    if any(p.parent == Path('/dev/dri') for p in nodes):
        command += ['--dir', '/dev/dri']
    for node in nodes:
        command += ['--dev-bind', str(node), str(node)]
    command += ['--tmpfs', '/dev/shm', '--bind', str(output), str(output),
                '--bind', str(temporary), '/tmp']
    if executable_copy is not None:
        command += ['--ro-bind', str(executable_copy), binary[0]]
    if game_mode is not None:
        arguments = game_mode_arguments(arguments, game_mode)
    command += ['--', *binary, *arguments]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, SDL_HINT_EGL_DEVICE='0',
                       TMPDIR=str(temporary), XDG_RUNTIME_DIR=str(runtime),
                       XDG_CONFIG_HOME=str(runtime / 'config'), XDG_CACHE_HOME=str(runtime / 'cache'))
    metadata = {'backend': 'bwrap_gpu_device_namespace_v1', 'physical_gpu': gpu,
                'gpu_uuid': uuid, 'device_nodes': [str(n) for n in nodes],
                'isolated_vulkan_adapter': 0, 'isolated_egl_device': 0,
                'writable_root': str(output), 'binary': binary}
    if executable_copy is not None:
        metadata['private_executable_copy'] = str(executable_copy)
    if game_mode is not None:
        metadata.update(requested_game_mode=game_mode, startup_url=arguments[0])
    return command, environment, metadata
