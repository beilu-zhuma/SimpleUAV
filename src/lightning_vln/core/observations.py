"""每条轨迹一个观测 ZIP；回放兼容旧的逐帧目录。"""

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from lightning_vln.core.types import Observation, View


def save_observation(archive: ZipFile, iteration: int, observation: Observation):
    prefix = f"{iteration:04d}/"
    for name, view in observation.views.items():
        archive.writestr(prefix + name + ".jpg", view.rgb)
    depth = BytesIO()
    np.savez_compressed(depth, **{n: v.depth for n, v in observation.views.items()})
    archive.writestr(prefix + "depth.npz", depth.getvalue())
    metadata = {n: {"position_ned_m": v.position.tolist(),
                    "quaternion_xyzw": v.quaternion.tolist(), "fov": v.fov,
                    "timestamp": v.timestamp} for n, v in observation.views.items()}
    archive.writestr(prefix + "cameras.json", json.dumps(metadata))


def observation_steps(episode: Path) -> list[int]:
    if (episode / "observations.zip").exists():
        with ZipFile(episode / "observations.zip") as archive:
            return sorted(int(n.split("/")[0]) for n in archive.namelist() if n.endswith("/cameras.json"))
    return sorted(int(p.parent.name) for p in (episode / "observations").glob("*/cameras.json"))


def load_observation(episode: Path, iteration: int, position) -> Observation:
    prefix = f"{iteration:04d}/"

    def load(read):
        metadata = json.loads(read(prefix + "cameras.json"))
        with np.load(BytesIO(read(prefix + "depth.npz")), allow_pickle=False) as depth:
            views = {n: View(read(prefix + n + ".jpg"), depth[n],
                             np.array(c["position_ned_m"]), np.array(c["quaternion_xyzw"]),
                             c["fov"], c["timestamp"]) for n, c in metadata.items()}
        return Observation(np.array(position), views)

    if (episode / "observations.zip").exists():
        with ZipFile(episode / "observations.zip") as archive:
            return load(archive.read)
    return load(lambda name: (episode / "observations" / name).read_bytes())
