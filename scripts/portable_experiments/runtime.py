"""UAV-ON自有进程身份和接续入口；显式run/resume才分配GPU。"""
import os
from pathlib import Path
from .core import read, verify_output

def identity(pid):
    try:
        p=Path('/proc')/str(pid);fields=(p/'stat').read_bytes().rsplit(b')',1)[1].split()
        return dict(pid=pid,uid=p.stat().st_uid,start_ticks=fields[19].decode('ascii'),state=fields[0].decode('ascii'))
    except (OSError,IndexError,ValueError):
        return None


def alive(record):
    row=identity(record['pid'])
    return bool(row and row['state']!='Z' and row['uid']==record['uid']==os.getuid()
                and row['start_ticks']==str(record['start_ticks']))


def run(root, resume=False):
    root = Path(root).resolve()
    contract = verify_output(root)
    config = read(root/'machine-config.json')
    if contract['benchmark'] != 'uavon' or config.get('benchmark') != 'uavon':
        raise ValueError('SimpleUAV只接续UAV-ON运行')
    from .uavon import run as run_uavon
    return run_uavon(root, config, resume)
