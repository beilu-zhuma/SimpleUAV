"""仅冻结UAV-ON新实验；拒绝覆盖历史输出。"""
from pathlib import Path
from .core import REPO


def prepare(root, config, repo=REPO):
    root = Path(root).resolve()
    if config.get('benchmark') != 'uavon':
        raise ValueError('SimpleUAV只接受UAV-ON')
    if root.exists():
        raise ValueError('输出目录已存在，拒绝覆盖；接续请使用resume')
    from .uavon import prepare as prepare_uavon
    return prepare_uavon(root, config, repo)
