"""官方 valset 配套 test 环境的完整健康集；不抽样、不修改原任务内容。"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from lightning_vln.adapters.uavon.evaluation import PROBLEM_SCENES
from lightning_vln.adapters.uavon.trainset import encoded

TEST_SCENES = {
    'NYC_test': 'NYC/NYC1950.sh', 'CityPark_test': 'CityPark/CityPark.sh',
    'UrbanJapan_test': 'UrbanJapan/UrbanJapan.sh', 'CabinLake_test': 'CabinLake/CabinLake.sh',
    'CityStreet_test': 'CityStreet/CleanCityStreet.sh', 'Barnyard_test': 'Barnyard/Barnyard_test1.sh',
    'Slum_test': 'Slum/Slum_test1.sh', 'WinterTown_test': 'WinterTown/WinterTown_test1.sh',
    'BrushifyRoad_test': 'BrushifyRoad/BrushifyRoad_test1.sh', 'DownTown_test': 'DownTown/DownTown_test1.sh',
    'Venice_test': 'Venice/Vinice_test1.sh', 'BrushifyUrban_test': 'BrushifyUrban/BrushifyUrban.sh',
    'ModularNeighborhood_test': 'Neighborhood/NewNeighborhood.sh',
    'WesternTown_test': 'WesternTown/WesternTown_test1.sh',
}


def build_full_testset(source: Path):
    rows, excluded, sources, seen, scenes = [], [], {}, set(), set()
    for path in sorted(source.glob('*.json')):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in json.loads(path.read_text()):
            scene, original_id = row['map_name'], row['episode_id']
            if scene not in TEST_SCENES:
                raise ValueError('未知 test 场景：' + scene)
            if 'original_episode_id' in row:
                raise ValueError('输入已包装，必须使用官方 valset 原件')
            identity = f'{scene}_{original_id}'
            if identity in seen:
                raise ValueError('test 重复身份：' + identity)
            seen.add(identity)
            scenes.add(scene)
            if scene in PROBLEM_SCENES:
                excluded.append(identity)
            else:
                rows.append(dict(row, episode_id=identity, original_episode_id=original_id))
    if scenes != set(TEST_SCENES):
        raise ValueError('官方 test 来源场景覆盖不完整：' + ','.join(sorted(set(TEST_SCENES)-scenes)))
    manifest = dict(schema='uavon_full_test_v1', split='valset', benchmark_split='test',
                    selection='all_tasks_after_problem_scene_filter',
                    source_root=str(source.resolve()), source_files_sha256=sources,
                    source_total=len(seen), excluded_ids=excluded,
                    excluded_scenes=sorted(set(TEST_SCENES) & PROBLEM_SCENES), count=len(rows),
                    selected_by_scene=dict(Counter(t['map_name'] for t in rows)),
                    selected_ids=[t['episode_id'] for t in rows],
                    dataset_sha256=hashlib.sha256(encoded(rows)).hexdigest())
    return rows, manifest


def load_full_testset(dataset: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'uavon_full_test_v1' or manifest.get('split') != 'valset':
        raise ValueError('不是受支持的完整 test 清单')
    rows, expected = build_full_testset(Path(manifest['source_root']))
    if manifest != expected or dataset.read_bytes() != encoded(rows):
        raise ValueError('完整 test 身份、数据或来源与清单不一致')
    return rows, manifest
