"""开局旋转配置、冻结与真实Agent预算语义；全部使用CPU替身。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from portable_experiments import core, uavon
from lightning_vln.core.agent import Limits
from lightning_vln.core.types import Decision
from lightning_vln.adapters.uavon.minimal.agent import run_agent
from lightning_vln.adapters.uavon.minimal.policy import Options
from uavon_fixtures import YawWorld


class InitialRotationSettingTests(unittest.TestCase):
    def config(self):
        return core.read(core.REPO/'configs/experiments/uavon.example.json')

    def test_default_off_explicit_on_and_boolean_validation(self):
        cfg=self.config();cfg.pop('method')
        self.assertFalse(uavon.method_config(cfg)['initial_yaw90'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            core.save(root/'endpoints.json',{'0':dict(base_url='http://127.0.0.1:18630/v1',model='qwen36')})
            off=uavon.batch_command(root,cfg,root/'attempt')
            self.assertNotIn('--initial-yaw90',off)
            enabled=copy.deepcopy(cfg);enabled['method']={'initial_yaw90':True}
            uavon.validate_config(enabled)
            on=uavon.batch_command(root,enabled,root/'attempt')
            self.assertEqual(on.count('--initial-yaw90'),1)
            self.assertEqual([x for x in on if x!='--initial-yaw90'],off)
            # 兼容过去显式保存的完整主方法配置，不把true静默改成false。
            enabled['method']=dict(uavon.METHOD,initial_yaw90=True)
            self.assertTrue(uavon.method_config(enabled)['initial_yaw90'])
        for value in (0,1,'true','false',None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                uavon.validate_config(dict(cfg,method={'initial_yaw90':value}))
        with self.assertRaises(ValueError):
            uavon.validate_config(dict(cfg,method={'initial_yaw90':True,'stop_distance_m':15}))
        with self.assertRaises(ValueError):
            uavon.validate_config(dict(cfg,method={'initial_yaw90':True,'unknown':True}))

    def test_prepare_freezes_effective_rotation_setting(self):
        for enabled in (False,True):
            with self.subTest(enabled=enabled),tempfile.TemporaryDirectory() as directory:
                cfg=self.config()
                if enabled:cfg['method']={'initial_yaw90':True}
                else:cfg.pop('method')  # 默认值也必须明确落盘。
                root=Path(directory)/'prepared'
                with patch.object(uavon,'check',return_value=dict(passed=True,inputs={},expected=733)),\
                     patch.object(uavon,'build_inputs',return_value=([{'episode_id':'one','map_name':'NYC_test'}],{'schema':'fixture'})):
                    uavon.prepare(root,cfg,core.REPO)
                core.verify_output(root)
                actual=core.read(root/'method.json')
                self.assertEqual(actual['initial_yaw90'],enabled)
                self.assertEqual(core.read(root/'machine-config.json')['method'],actual)
                self.assertEqual('--initial-yaw90' in uavon.batch_command(root,core.read(root/'machine-config.json'),root/'attempt'),enabled)
                changed=core.read(root/'machine-config.json');changed['method']['initial_yaw90']=not enabled
                core.save(root/'machine-config.json',changed)
                with self.assertRaisesRegex(ValueError,'冻结源码'):
                    core.verify_output(root)

    def test_agent_rotation_only_when_enabled_and_charged_once(self):
        class Model:
            def __init__(self):self.iterations=[];self.overviews=[]
            def decide(self,context,observation,iteration):
                self.iterations.append(iteration)
                request=context.messages('system',observation,iteration)
                self.overviews.append(any('开局旋转90度之前' in item.get('text','')
                    for message in request if isinstance(message['content'],list) for item in message['content']))
                # 估距大于1m，持续拒绝STOP，确保用尽指定预算而无额外平移动作。
                return Decision('stop','front',(400,400,600,600),'检查目标','线索')
        for enabled in (False,True):
            with self.subTest(enabled=enabled),tempfile.TemporaryDirectory() as directory:
                output=Path(directory);world=YawWorld();model=Model()
                summary=run_agent(world,model,'目标',output,Limits(decisions=3,stop_distance=1),
                                  Options(initial_yaw90=enabled,ascent_guard=True))
                self.assertEqual(summary['decisions'],3)
                self.assertEqual(summary['initial_rotation_actions'],int(enabled))
                self.assertEqual(summary['navigation_decisions'],3-int(enabled))
                self.assertEqual(model.iterations,list(range(int(enabled),3)))
                self.assertEqual(world.events,['face'] if enabled else [])
                self.assertEqual(model.overviews,[True,False] if enabled else [False,False,False])
                self.assertEqual((output/'initial-scan.json').exists(),enabled)
                self.assertEqual((output/'initial-scan.zip').exists(),enabled)
                rows=[json.loads(line) for line in (output/'decisions.jsonl').read_text().splitlines()]
                self.assertEqual(sum(row['decision']['action']=='initial_yaw' for row in rows),int(enabled))
                self.assertEqual(summary['method_options']['initial_yaw90'],enabled)


if __name__=='__main__':unittest.main()
