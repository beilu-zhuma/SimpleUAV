"""数据集与模拟器之间的平移坐标边界；不向搜索策略暴露对象位置。"""

import json
from pathlib import Path

import numpy as np

from lightning_vln.core.geometry import vector
from lightning_vln.core.types import Observation, View


def load_scene_offset(path: Path, scene: str) -> np.ndarray:
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != 'uavon_train_scene_frame_v1' or manifest.get('split') != 'trainset':
        raise ValueError('不是训练场景坐标校准清单')
    if scene not in manifest['scenes']:
        raise ValueError('坐标校准清单未包含当前场景：'+str(scene))
    return vector(manifest['scenes'][scene]['offset_dataset_to_airsim_ned_m'],3).copy()


class DatasetFrameWorld:
    """速度和旋转不变；出生、观测、目标和轨迹统一使用数据集 NED。"""

    def __init__(self, world, offset):
        self.world = world
        self.offset = vector(offset,3).copy()

    def prepare(self, position=None, quaternion=None):
        self.world.prepare(None if position is None else vector(position,3)+self.offset,quaternion)

    def position(self):
        return self.world.position()-self.offset

    def observe(self, with_rgb=True):
        raw = self.world.observe(with_rgb=with_rgb)
        return self._observation(raw)

    def _observation(self, raw):
        return Observation(raw.position-self.offset,{name: View(v.rgb,v.depth,v.position-self.offset,
            v.quaternion,v.fov,v.timestamp) for name,v in raw.views.items()})

    def face(self, goal):
        return self.world.face(vector(goal,3)+self.offset)

    def advance(self, velocity, seconds):
        self.world.advance(velocity,seconds)

    def hold(self):
        self.world.hold()

    def collided(self):
        return self.world.collided()

    def close(self):
        self.world.close()

    @property
    def control(self):
        return self.world.control

    @property
    def last_observe(self):
        return self.world.last_observe

    def _stop(self, raw):
        result=raw.copy()
        if 'target_z_ned_m' in result:
            result['target_z_ned_m']=float(result['target_z_ned_m']-self.offset[2])
        return result

    @property
    def last_stop(self):
        return self._stop(self.world.last_stop)

    @property
    def last_prepare(self):
        result=self.world.last_prepare.copy()
        for key,value in list(result.items()):
            if key.endswith('_ned_m'):
                result[key]=(vector(value,3)-self.offset).tolist()
        if 'stop' in result:
            result['stop']=self._stop(result['stop'])
        result['dataset_to_airsim_offset_ned_m']=self.offset.tolist()
        return result
