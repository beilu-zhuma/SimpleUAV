"""健康test集的固定种子抽样与任务身份校验。"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from lightning_vln.adapters.uavon.benchmark import load_full_testset
from lightning_vln.adapters.uavon.evaluation import PROBLEM_SCENES
from lightning_vln.adapters.uavon.trainset import encoded


def sample_testset(source_dataset, source_manifest, count, seed):
    source_dataset, source_manifest = Path(source_dataset), Path(source_manifest)
    population, original = load_full_testset(source_dataset, source_manifest)
    if any(row["map_name"] in PROBLEM_SCENES for row in population):
        raise ValueError("抽样总体含问题场景")
    if not 1 <= count <= len(population):
        raise ValueError("抽样数量超出健康test总体")
    rows = random.Random(seed).sample(population, count)
    manifest = {"schema": "uavon_minimal_test_sample_v1", "split": "valset",
                "selection": "uniform_without_replacement_after_problem_scene_filter",
                "source_dataset": str(source_dataset.resolve()), "source_manifest": str(source_manifest.resolve()),
                "source_dataset_sha256": hashlib.sha256(source_dataset.read_bytes()).hexdigest(),
                "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
                "source_total": original["source_total"], "eligible_total": len(population),
                "excluded_ids": original["excluded_ids"], "excluded_scenes": original["excluded_scenes"],
                "seed": seed, "count": count, "selected_ids": [row["episode_id"] for row in rows],
                "selected_by_scene": dict(Counter(row["map_name"] for row in rows)),
                "dataset_sha256": hashlib.sha256(encoded(rows)).hexdigest()}
    return rows, manifest


def load_test_sample(dataset, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('schema') == 'uavon_test_sample_partition_v1':
        rows, expected = partition_test_sample(manifest['parent_dataset'],manifest['parent_manifest'],
                                              manifest['start'],manifest['count'])
        if manifest != expected or Path(dataset).read_bytes() != encoded(rows):
            raise ValueError('分批抽样身份、内容或来源与固定父清单不一致')
        return rows, manifest
    if manifest.get("schema") == "uavon_minimal_test_sample_v2":
        rows, expected = sample_testset_excluding(manifest["source_dataset"], manifest["source_manifest"],
            manifest["count"], manifest["seed"], manifest["excluded_sample_dataset"],
            manifest["excluded_sample_manifest"])
        if manifest != expected or Path(dataset).read_bytes() != encoded(rows):
            raise ValueError("排除抽样身份、内容或来源与固定清单不一致")
        return rows, manifest
    if manifest.get("schema") != "uavon_minimal_test_sample_v1" or manifest.get("split") != "valset":
        raise ValueError("不是独立健康test抽样清单")
    rows, expected = sample_testset(manifest["source_dataset"], manifest["source_manifest"],
                                    manifest["count"], manifest["seed"])
    if manifest != expected or Path(dataset).read_bytes() != encoded(rows):
        raise ValueError("抽样身份、内容或来源与固定清单不一致")
    return rows, manifest


def partition_test_sample(parent_dataset,parent_manifest,start,count):
    parent_dataset,parent_manifest=Path(parent_dataset),Path(parent_manifest)
    metadata=json.loads(parent_manifest.read_text())
    if metadata.get('schema') not in ('uavon_minimal_test_sample_v1','uavon_minimal_test_sample_v2'):
        raise ValueError('分批父清单必须是已核验的固定随机样本')
    rows,_=load_test_sample(parent_dataset,parent_manifest)
    if start<0 or count<1 or start+count>len(rows):raise ValueError('分批范围错误')
    selected=rows[start:start+count]
    manifest=dict(schema='uavon_test_sample_partition_v1',split='valset',
        selection='fixed_contiguous_partition_of_uniform_parent_sample',parent_dataset=str(parent_dataset.resolve()),
        parent_manifest=str(parent_manifest.resolve()),parent_dataset_sha256=hashlib.sha256(parent_dataset.read_bytes()).hexdigest(),
        parent_manifest_sha256=hashlib.sha256(parent_manifest.read_bytes()).hexdigest(),start=start,count=count,
        excluded_ids=metadata['excluded_ids'],excluded_scenes=metadata['excluded_scenes'],
        selected_ids=[r['episode_id'] for r in selected],dataset_sha256=hashlib.sha256(encoded(selected)).hexdigest())
    return selected,manifest


def sample_testset_excluding(source_dataset, source_manifest, count, seed,
                            excluded_dataset, excluded_manifest):
    """v2：先核验原v1样本，再排除其身份；旧v1协议及随机序列保持不变。"""
    source_dataset, source_manifest = Path(source_dataset), Path(source_manifest)
    excluded_dataset, excluded_manifest = Path(excluded_dataset), Path(excluded_manifest)
    excluded_metadata = json.loads(excluded_manifest.read_text())
    if excluded_metadata.get('schema') != 'uavon_minimal_test_sample_v1':
        raise ValueError('排除来源必须为原v1固定样本')
    excluded, _ = load_test_sample(excluded_dataset, excluded_manifest)
    population, original = load_full_testset(source_dataset, source_manifest)
    by_id = {row['episode_id']: row for row in population}
    if len(by_id) != len(population) or any(by_id.get(r['episode_id']) != r for r in excluded):
        raise ValueError('排除任务不属于同一健康总体或内容不同')
    excluded_ids = {r['episode_id'] for r in excluded}
    eligible = [r for r in population if r['episode_id'] not in excluded_ids]
    if any(r['map_name'] in PROBLEM_SCENES for r in population) or not 1 <= count <= len(eligible):
        raise ValueError('抽样总体场景或数量错误')
    rows = random.Random(seed).sample(eligible, count)
    manifest = dict(schema='uavon_minimal_test_sample_v2', split='valset',
        selection='uniform_without_replacement_after_problem_scene_and_old_sample_filter',
        source_dataset=str(source_dataset.resolve()), source_manifest=str(source_manifest.resolve()),
        source_dataset_sha256=hashlib.sha256(source_dataset.read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        source_total=original['source_total'], healthy_total=len(population), eligible_total=len(eligible),
        excluded_ids=original['excluded_ids'], excluded_scenes=original['excluded_scenes'],
        excluded_sample_dataset=str(excluded_dataset.resolve()),
        excluded_sample_manifest=str(excluded_manifest.resolve()),
        excluded_sample_dataset_sha256=hashlib.sha256(excluded_dataset.read_bytes()).hexdigest(),
        excluded_sample_manifest_sha256=hashlib.sha256(excluded_manifest.read_bytes()).hexdigest(),
        excluded_sample_ids=[r['episode_id'] for r in excluded], seed=seed, count=count,
        selected_ids=[r['episode_id'] for r in rows],
        selected_by_scene=dict(Counter(r['map_name'] for r in rows)),
        dataset_sha256=hashlib.sha256(encoded(rows)).hexdigest())
    return rows, manifest
