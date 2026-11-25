"""UAV-ON统一入口；check和prepare只做CPU检查。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .core import load_config, check, read, verify_output


def request_stop(root):
    """源码损坏时仍可请求自有实验退出；不直接发送进程信号。"""
    root = Path(root).resolve()
    contract = read(root/'portable-contract.json')
    if contract['root'] != str(root) or root.stat().st_uid != os.getuid():
        raise ValueError('停止目标不属于本轮/当前用户')
    if contract.get('benchmark') != 'uavon':
        raise ValueError('此分支只停止UAV-ON实验')
    (root/'cancel.request').touch()
    (root/'stop.request').touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check','prepare','run','resume','stop'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.action in ('check','prepare'):
            if args.config is None:
                parser.error('check/prepare需要--config')
            config = load_config(args.config)
            if args.action == 'check':
                result = check(config)
            else:
                if args.output is None:
                    parser.error('prepare需要--output')
                from .prepare import prepare
                result = dict(prepared=str(prepare(args.output,config)), gpu_started=False)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return
        if args.output is None:
            parser.error('需要--output')
        root = args.output.resolve()
        if args.action == 'stop':
            request_stop(root)
            print('已写停止请求；等待本轮控制器保存证据并退出。')
            return
        verify_output(root)
        if Path(__file__).resolve().parent != root/'launcher/portable_experiments':
            config = read(root/'machine-config.json')
            os.execv(config['paths']['python'], [config['paths']['python'],'-B',str(root/'launcher/experiment.py'),
                     args.action,'--output',str(root)])
        from .runtime import run
        run(root,resume=args.action=='resume')
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
        print('UAV-ON入口拒绝继续：'+str(error),file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
