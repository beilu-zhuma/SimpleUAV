"""UAV-ON配置、文件摘要和冻结契约；不导入模型或分配GPU。"""
import hashlib
import json
import os
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]
SCHEMA = 1

def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.log')}


def clone(source, target):
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.log', '.git'))


def verify_bundle(root):
    manifest = read(Path(root)/'MANIFEST.json')
    for name, sha in manifest['files'].items():
        if digest(Path(root)/name) != sha:
            raise ValueError('方法源码版本改变：'+name)
    return manifest


def require(path, label, directory=False):
    if not (Path(path).is_dir() if directory else Path(path).is_file()):
        raise ValueError(label+'不存在：'+str(path))


def verify_output(root):
    root = Path(root).resolve(); receipt=read(root/'portable-contract.json')
    if receipt['root'] != str(root):
        raise ValueError('实验目录已迁移，不能静默修改包含绝对路径的接续契约')
    if receipt.get('benchmark') != 'uavon':
        raise ValueError('此分支只接续UAV-ON实验')
    for name, sha in receipt['files'].items():
        if digest(root/name) != sha:
            raise ValueError('冻结源码、配置或任务清单改变：'+name)
    for name, sha in receipt['inputs'].items():
        if digest(name) != sha:
            raise ValueError('外部任务输入改变：'+name)
    return receipt


def load_config(path):
    config = read(path)
    if config.get('schema') != SCHEMA or config.get('benchmark') != 'uavon':
        raise ValueError('uavon-only分支只接受schema=1、benchmark=uavon')
    config = json.loads(json.dumps(config))
    base = Path(path).resolve().parent
    for key, value in config['paths'].items():
        if value:
            config['paths'][key] = str((base/value).resolve()) if not Path(value).is_absolute() else value
    for model in config['models']:
        value = model['path']
        model['path'] = str((base/value).resolve()) if not Path(value).is_absolute() else value
    from .uavon import validate_config
    validate_config(config)
    return config


def check(config, repo=REPO):
    if config.get('benchmark') != 'uavon':
        raise ValueError('uavon-only分支只接受UAV-ON')
    from .uavon import check as check_uavon
    return check_uavon(config, repo)
