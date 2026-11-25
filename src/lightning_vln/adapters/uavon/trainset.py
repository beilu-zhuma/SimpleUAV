"""官方 trainset 的可复现健康场景抽样；不改写旧 random50 身份。"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

TRAIN_SCENES = {
    'BrushifyUrban_TrainSets': 'BrushifyUrban/BrushifyUrban.sh',
    'CabinLake_TrainSets': 'CabinLake/CabinLake.sh',
    'ModularNeighborhood_TrainSets': 'Neighborhood/NewNeighborhood.sh',
    'UrbanJapan_TrainSets': 'UrbanJapan/UrbanJapan.sh',
    'Venice_train': 'Venice/vinice_new1.sh',
    'WesternTown_train': 'WesternTown/WesternTown1.sh',
    'WinterTown_train': 'WinterTown/WinterTown1.sh',
}
EXCLUDED_TRAIN_SCENES = {'CityPark_TrainSets', 'DownTown_train', 'Slum_train', 'BrushifyRoad_train'}


def encoded(rows):
    return (json.dumps(rows, ensure_ascii=False, indent=2) + '\n').encode()


def sample_trainset(source: Path, count: int, seed: int):
    pool, excluded, sources, seen = [], [], {}, set()
    for path in sorted(source.glob('*.json')):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in json.loads(path.read_text()):
            scene, original_id = row['map_name'], row['episode_id']
            identity = f'{scene}_{original_id}'
            if identity in seen:
                raise ValueError('trainset 重复身份：' + identity)
            seen.add(identity)
            if scene in EXCLUDED_TRAIN_SCENES:
                excluded.append(identity)
                continue
            if scene not in TRAIN_SCENES:
                raise ValueError('未知 train 场景：' + scene)
            if 'original_episode_id' in row:
                raise ValueError('输入已包装，必须使用官方 trainset 原件')
            pool.append(dict(row, episode_id=identity, original_episode_id=original_id))
    if not 1 <= count <= len(pool):
        raise ValueError('抽样数量超出健康 trainset 范围')
    rows = random.Random(seed).sample(pool, count)
    manifest = dict(schema='uavon_train_sample_v1', split='trainset', seed=seed, count=count,
                    sampling='uniform_without_replacement_after_scene_filter',
                    source_root=str(source.resolve()), source_files_sha256=sources,
                    source_total=len(seen), eligible_total=len(pool), excluded_ids=excluded,
                    excluded_scenes=sorted(EXCLUDED_TRAIN_SCENES),
                    population_by_scene=dict(Counter(t['map_name'] for t in pool)),
                    selected_by_scene=dict(Counter(t['map_name'] for t in rows)),
                    selected_ids=[t['episode_id'] for t in rows],
                    dataset_sha256=hashlib.sha256(encoded(rows)).hexdigest())
    return rows, manifest


def load_train_sample(dataset: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 'uavon_train_sample_v1' or manifest['split'] != 'trainset':
        raise ValueError('不是受支持的 trainset 抽样清单')
    rows, expected = sample_trainset(Path(manifest['source_root']), manifest['count'], manifest['seed'])
    if manifest != expected or dataset.read_bytes() != encoded(rows):
        raise ValueError('trainset 抽样身份、数据或来源与清单不一致')
    return rows, manifest
