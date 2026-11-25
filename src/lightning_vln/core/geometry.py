"""从旧 semantic_map_core 抽取的相机几何，以及无地图的短段深度检查。"""

import math
from functools import lru_cache

import numpy as np

from lightning_vln.core.types import Observation, View


def vector(value, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"需要 {size} 个有限数值")
    return result


def rotation(quaternion) -> np.ndarray:
    q = vector(quaternion, 4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("四元数不能为零")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def rays(view: View) -> np.ndarray:
    """DepthPerspective 是沿单位射线的径向距离，不是轴向 Z 深度。"""
    height, width = view.depth.shape
    return camera_rays(height, width, float(view.fov))


@lru_cache(maxsize=16)
def camera_rays(height: int, width: int, fov: float) -> np.ndarray:
    if not 0 < fov < 180 or not height or not width:
        raise ValueError("无效的相机内参")
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    v, u = np.indices((height, width))
    directions = np.stack((np.ones_like(u), (u-(width-1)/2)/focal,
                           (v-(height-1)/2)/focal), axis=-1)
    result = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    result.flags.writeable = False
    return result


@lru_cache(maxsize=1)
def guard_disk() -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.linspace(-1, 1, 33), np.linspace(-1, 1, 33))
    disk = u*u + v*v <= 1
    angles = np.linspace(0, 2*math.pi, 64, endpoint=False)
    result = np.concatenate((u[disk], np.cos(angles))), np.concatenate((v[disk], np.sin(angles)))
    for values in result:
        values.flags.writeable = False
    return result


def valid_depth(depth: np.ndarray, max_depth: float) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0) & (depth < max_depth)


def ground_bbox(view: View, bbox, max_depth: float = 100) -> np.ndarray:
    box = vector(bbox, 4)
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        raise ValueError("bbox 必须为 [0,1000] 范围内的非空 xyxy")
    height, width = view.depth.shape
    u0, v0 = int(x0*width/1000), int(y0*height/1000)
    u1, v1 = math.ceil(x1*width/1000), math.ceil(y1*height/1000)
    crop = view.depth[v0:v1, u0:u1]
    yy, xx = np.nonzero(valid_depth(crop, max_depth))
    if not len(xx):
        raise ValueError("目标区域没有有效的米制深度")
    depths = crop[yy, xx]
    # 沿用旧实现的 25% 分位前景采样；相同深度优先选框中心附近像素。
    desired = np.percentile(depths, 25)
    rank = np.lexsort(((xx-(u1-u0-1)/2)**2 + (yy-(v1-v0-1)/2)**2,
                      np.abs(depths-desired)))
    index = rank[0]
    ray = rays(view)[v0+yy[index], u0+xx[index]]
    return vector(view.position, 3) + rotation(view.quaternion) @ ray * depths[index]


def horizontal_goal(view: View, bbox, start, max_step_len: float) -> np.ndarray:
    """框中心只指定方向；天空/远景不需要可反投影的表面深度。"""
    x0, y0, x1, y1 = vector(bbox, 4)
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        raise ValueError("bbox 必须为 [0,1000] 范围内的非空 xyxy")
    height, width = view.depth.shape
    focal = width / (2 * math.tan(math.radians(view.fov) / 2))
    ray = np.array([1., ((x0+x1)*width/2000-width/2)/focal,
                    ((y0+y1)*height/2000-height/2)/focal])
    direction = rotation(view.quaternion) @ ray
    direction[2] = 0
    length = np.linalg.norm(direction)
    if length < 1e-6:
        raise ValueError("所选方向没有水平分量，请选择侧方区域")
    return vector(start, 3) + direction/length * max_step_len


def destination(start, surface, standoff: float, max_step_len: float) -> np.ndarray:
    delta = vector(surface, 3) - vector(start, 3)
    distance = np.linalg.norm(delta)
    if distance < 1e-9:
        return np.array(start, dtype=float)
    travel = min(max(distance-standoff, 0), max_step_len)
    return np.asarray(start) + delta * travel / distance


def segment_clear(observation: Observation, end, radius: float, clearance: float,
                  max_depth: float = 100, *, blind_up: bool = False) -> tuple[bool, str]:
    """检查短段胶囊体的可见障碍，并要求前方采样截面都有深度观测。

    不维护地图。采样深度无法保证检出细线、透明障碍或所有遮挡区域。
    """
    start = vector(observation.position, 3)
    delta = vector(end, 3) - start
    length = np.linalg.norm(delta)
    if length < 1e-9:
        return True, "reached"
    direction = delta / length
    protected_radius = radius + clearance
    # 只保留一个前瞻截面；允许多个相机共同覆盖跨视图的斜向通道。
    axis = np.eye(3)[np.argmin(np.abs(direction))]
    horizontal = np.cross(direction, axis)
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(direction, horizontal)
    u, v = (values * protected_radius for values in guard_disk())
    lookahead = max(length + protected_radius, 3*protected_radius)
    samples = start + direction*lookahead + u[:, None]*horizontal + v[:, None]*vertical
    observed = np.zeros(len(samples), dtype=bool)
    for view in observation.views.values():
        matrix = rotation(view.quaternion)
        valid = valid_depth(view.depth, max_depth)
        # 三角不等式：胶囊体内的点到相机最多这么远。先筛远点，再做原精确检查；
        # 上取一个float32步长，避免阈值转换时漏掉边界点。远景仍参与下方可见性检查。
        bound = np.linalg.norm(np.asarray(view.position)-start) + length + protected_radius
        valid &= view.depth <= np.nextafter(np.float32(bound), np.float32(np.inf))
        points = (rays(view)[valid] * view.depth[valid, None]) @ matrix.T + view.position
        relative = points - start
        along = np.clip(relative @ direction, 0, length)
        distances = np.linalg.norm(relative - along[:, None]*direction, axis=1)
        if np.any(distances <= protected_radius):
            return False, "blocked"

        # max_depth 是定位上限，不是 AirSim 的无效值标记。有限正远景
        # 可证明短段通行；NaN/Inf/非正值仍拒绝。取投影邻域最小值，不插值填洞。
        local = (samples-view.position) @ matrix
        height, width = view.depth.shape
        focal = width / (2*math.tan(math.radians(view.fov)/2))
        in_front = local[:, 0] > 0
        indices = np.flatnonzero(in_front)
        projected = local[in_front, 1:] / local[in_front, :1] * focal
        projected += np.array([(width-1)/2, (height-1)/2])
        # 近乎平行于成像平面的射线可投影到整数范围之外；先排除画外点再转换。
        inside = ((projected[:, 0] >= 0) & (projected[:, 0] < width-1)
                  & (projected[:, 1] >= 0) & (projected[:, 1] < height-1))
        indices, pixels = indices[inside], np.floor(projected[inside]).astype(int)
        if not len(indices):
            continue
        px, py = pixels.T
        patch = np.stack([view.depth[py+dy, px+dx] for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1))])
        required = np.linalg.norm(local[indices], axis=1) + protected_radius
        observed[indices] |= ((required < max_depth) & np.isfinite(patch).all(axis=0)
                              & (patch > required).all(axis=0))
    # 仅显式上升探索豁免上方覆盖；所有相机的近障碍检查已执行。
    # 不能借此放行下降或水平运动。
    if blind_up and delta[2] < 0 and np.linalg.norm(delta[:2]) < 1e-6:
        return True, "clear_blind_up"
    return (True, "clear") if observed.all() else (False, "unobserved_or_near_obstacle")
