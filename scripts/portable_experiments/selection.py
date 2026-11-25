"""从固定主任务清单抽取验收子集；不按历史结果筛选，不改变导航设置。"""
import copy
import hashlib
import json
import random


def select(tasks, config, identity='identity'):
    spec=config.get('selection')
    if spec is None:return copy.deepcopy(tasks),None
    if set(spec)-{'count','seed','scene'} or type(spec.get('count')) is not int or spec['count']<1 or type(spec.get('seed')) is not int:
        raise ValueError('selection需要正整数count及固定整数seed，可选scene')
    if spec.get('scene'):raise ValueError('UAV-ON验收不支持scene过滤')
    pool=[t for t in tasks if not spec.get('scene') or t['scene']==spec['scene']]
    if identity!='episode_id':pool.sort(key=lambda t:str(t[identity]))
    if not pool or len({t[identity] for t in pool})!=len(pool):raise ValueError('验收抽样总体为空或身份重复')
    chosen=random.Random(spec['seed']).sample(pool,min(spec['count'],len(pool)))
    proof=dict(schema=1,scope='independent_code_validation_subset',seed=spec['seed'],requested=spec['count'],
        source_count=len(tasks),eligible_count=len(pool),count=len(chosen),scene=spec.get('scene'),
        population_sha256=hashlib.sha256(json.dumps(tasks,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        selected_ids=[t[identity] for t in chosen],selection='uniform_without_replacement_no_result_filter')
    return copy.deepcopy(chosen),proof
