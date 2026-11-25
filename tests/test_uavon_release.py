"""UAV-ON主方法、源码冻结与进程资源边界。"""
import copy
import ast
import json
import os
from pathlib import Path
import sys
import socket
import uuid
import tempfile
import unittest
from unittest.mock import patch
from portable_experiments import core, uavon
from portable_experiments.runtime import alive

class MainReleaseTests(unittest.TestCase):
    def config(self):
            return core.read(core.REPO/'configs/experiments/uavon.example.json')

    def test_vllm_runtime_supports_real_unix_socket_and_rejects_long_root(self):
            cfg=self.config()
            with tempfile.TemporaryDirectory() as directory:
                cfg['paths']['runtime_root']=directory
                runtime=uavon.runtime_directory(cfg);address=runtime/str(uuid.uuid4())
                with socket.socket(socket.AF_UNIX) as sock:sock.bind(str(address))
                address.unlink();runtime.rmdir()
                long=Path(directory)/('x'*80);long.mkdir();cfg['paths']['runtime_root']=str(long)
                with self.assertRaisesRegex(ValueError,'IPC'):uavon.runtime_directory(cfg)

    def test_main_uavon_rejects_silent_protocol_and_sr_only_changes(self):
            config=self.config();uavon.validate_config(config)
            for field,value in [('method',dict(uavon.METHOD,decisions=40)),('record_mode','results-only'),
                                ('gpus',[0,0]),('concurrency',0),('ports',{'model':30310,'scene':30310}),
                                ('serving',{'speculative_config':'mtp'})]:
                with self.subTest(field=field),self.assertRaises(ValueError):
                    uavon.validate_config(dict(config,**{field:value}))

    def test_commands_preserve_two_stage_policy_collision_and_complete_metrics(self):
            cfg=self.config()
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                core.save(root/'endpoints.json',{'0':dict(base_url='http://127.0.0.1:18630/v1',model='qwen36')})
                command=uavon.batch_command(root,cfg,root/'attempt-0001',root/'prior/full')
            self.assertNotIn('--initial-yaw90',command)
            for flag in ('--current-target-evidence','--motion-options','--ascent-guard',
                         '--isolate-gpu','--isolated-model-recovery'):
                self.assertIn(flag,command)
            for flag,value in [('--decisions','150'),('--history-turns','4'),('--text-memory-mode','on'),
                               ('--stop-distance','10'),('--success-distance','20'),('--record-mode','metrics-only')]:
                self.assertEqual(command[command.index(flag)+1],value)
            model=uavon.model_command(cfg,0)
            self.assertIn('--enable-prefix-caching',model)
            self.assertNotIn('--speculative-config',model)
            self.assertNotIn('--max-model-len',model)
            small=copy.deepcopy(cfg);small['models'][0]['name']='qwen3vl8b'
            self.assertNotIn('--reasoning-parser',uavon.model_command(small,0))

    def test_uavon_preparation_is_frozen_and_never_allocates_gpu(self):
            cfg=self.config()
            task={'episode_id':'example','map_name':'NYC_test'}
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'new'
                with patch.object(uavon,'check',return_value=dict(passed=True,inputs={},expected=733)),\
                     patch.object(uavon,'build_inputs',return_value=([task],{'schema':'fixture'})),\
                     patch.object(uavon,'check_resources',side_effect=AssertionError('prepare不能查询GPU')):
                    uavon.prepare(root,cfg,core.REPO)
                contract=core.verify_output(root)
                self.assertEqual(contract['benchmark'],'uavon')
                self.assertEqual(contract['validation']['gpu_smoke'],'pending')
                p=root/'source_snapshot/src/lightning_vln/simulator.py'
                self.assertEqual(p.read_bytes(),(core.REPO/'src/lightning_vln/simulator.py').read_bytes())
                p.write_text('changed')
                with self.assertRaisesRegex(ValueError,'冻结源码'):core.verify_output(root)

    def test_owned_cpu_child_is_registered_and_released(self):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                child=uavon.OwnedChild([sys.executable,'-B','-c','import time; time.sleep(10)'],
                    dict(os.environ,CUDA_VISIBLE_DEVICES=''),root/'child.log',root/'process.json')
                record=core.read(root/'process.json')
                self.assertEqual(record['uid'],os.getuid())
                try:self.assertTrue(alive(record))
                finally:child.close()
                self.assertFalse(alive(record))
