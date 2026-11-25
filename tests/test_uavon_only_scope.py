"""UAV-ON独立分支的范围、冻结、停止与身份保护。"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from portable_experiments import core
from portable_experiments.cli import request_stop
from portable_experiments.runtime import alive


class UavonOnlyScopeTests(unittest.TestCase):
    def test_only_uavon_config_and_relative_resource_paths(self):
        cfg=core.read(core.REPO/'configs/experiments/uavon.example.json')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'
            cfg['paths']['annotations']='../dataset';core.save(path,cfg)
            self.assertEqual(core.load_config(path)['paths']['annotations'],str((path.parent/'../dataset').resolve()))
            for name in ('a200k','travel'):
                core.save(path,dict(cfg,benchmark=name))
                with self.subTest(benchmark=name),self.assertRaisesRegex(ValueError,'uavon'):
                    core.load_config(path)

    def test_source_manifest_and_uavon_package_scope(self):
        core.verify_bundle(core.REPO/'src')
        self.assertFalse((core.REPO/'methods').exists())
        self.assertFalse((core.REPO/'variants').exists())
        self.assertEqual({p.name for p in (core.REPO/'src/lightning_vln/adapters').iterdir() if p.is_dir()},{'uavon'})
        self.assertEqual({p.name for p in (core.REPO/'third_party').iterdir() if p.is_dir()},{'UAV-ON'})

    def test_frozen_contract_and_external_inputs_remain_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'code.py';source.write_text('x=1\n')
            data=root/'external-task.json';data.write_text('{}')
            contract=dict(root=str(root),benchmark='uavon',files={'code.py':core.digest(source)},inputs={str(data):core.digest(data)})
            core.save(root/'portable-contract.json',contract);core.verify_output(root)
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'冻结源码'):core.verify_output(root)
            source.write_text('x=1\n');data.write_text('changed')
            with self.assertRaisesRegex(ValueError,'外部任务输入'):core.verify_output(root)

    def test_stop_is_available_for_corrupt_own_source_but_rejects_wrong_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            core.save(root/'portable-contract.json',dict(root=str(root),benchmark='uavon',files={'missing.py':'bad'}))
            request_stop(root)
            self.assertTrue((root/'stop.request').exists())
            for name in ('a200k','travel'):
                other=root/name;other.mkdir()
                core.save(other/'portable-contract.json',dict(root=str(other),benchmark=name))
                with self.assertRaisesRegex(ValueError,'UAV-ON'):request_stop(other)
                self.assertFalse((other/'stop.request').exists())

    def test_pid_reuse_and_foreign_uid_are_not_owned(self):
        row=dict(pid=123,uid=os.getuid(),start_ticks='10')
        for changed in (dict(row,start_ticks='11',state='S'),dict(row,uid=os.getuid()+1,state='S'),dict(row,state='Z')):
            with self.subTest(live=changed),patch('portable_experiments.runtime.identity',return_value=changed):
                self.assertFalse(alive(row))


if __name__=='__main__':unittest.main()
