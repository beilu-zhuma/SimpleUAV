"""单一核心的结构约束、重构前行为回放与兼容入口验证。"""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from shared_core_replay import replay
from portable_experiments.core import REPO


class SharedCoreTests(unittest.TestCase):
    def test_uavon_replay_matches_pre_refactor_requests_motion_and_metrics(self):
        self.assert_replay('uavon')



    def assert_replay(self,name):
        original=json.loads((REPO/'tests/fixtures'/f'shared-core-{name}.json').read_text())
        self.assertEqual(replay(name),original)

    def test_compatibility_module_is_the_same_instance_as_core(self):
        for name in ('agent','context','vlm','geometry','evidence','motion_options','simulator','rpc','frames','observations'):
            self.assertIs(importlib.import_module('lightning_vln.'+name),importlib.import_module('lightning_vln.core.'+name))
        self.assertIs(importlib.import_module('lightning_vln.uavon_minimal.agent'),importlib.import_module('lightning_vln.adapters.uavon.minimal.agent'))

    def test_only_one_physical_core_package_is_shipped(self):
        for name in ('methods/a200k_original_invalid60/navigation/src','methods/a200k_original_invalid60/services/src','variants/travel-adaptation-v2/src'):
            self.assertFalse((REPO/name).exists())
        package=REPO/'src/lightning_vln'
        for function in ('move_straight','ground_candidates','bind_target','segment_clear','request_json'):
            definitions=[]
            for p in package.rglob('*.py'):
                definitions.extend(str(p.relative_to(REPO)) for node in ast.parse(p.read_text()).body
                                   if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==function)
            self.assertEqual(len(definitions),1,(function,definitions))
            self.assertTrue(definitions[0].startswith('src/lightning_vln/core/'))

    def test_core_has_no_benchmark_imports(self):
        for p in (REPO/'src/lightning_vln/core').glob('*.py'):
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node,ast.ImportFrom):self.assertNotIn('.adapters',node.module or '',str(p))
                if isinstance(node,ast.Import):
                    for item in node.names:self.assertNotIn('.adapters',item.name,str(p))


    def test_both_module_cli_names_execute_the_real_parser_without_gpu(self):
        for name in ('lightning_vln.adapters.uavon.tuning.cli','lightning_vln.uavon_tuning.cli'):
            result=subprocess.run([sys.executable,'-B','-m',name,'--help'],capture_output=True,text=True,timeout=20,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(REPO/'src')))
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--dataset',result.stdout)


    def test_prompt_literals_match_the_pre_refactor_export(self):
        from lightning_vln.core import prompts
        from lightning_vln.adapters.uavon.minimal.vlm import GUARD_PROMPT
        values=dict(system=prompts.SYSTEM_PROMPT,localize=prompts.LOCALIZATION_PROMPT,
                    evidence=prompts.EVIDENCE_NAVIGATION_PROMPT,motion=prompts.MOTION_NAVIGATION_PROMPT,
                    guard=GUARD_PROMPT)
        expected=json.loads((REPO/'tests/fixtures/prompt-hashes.json').read_text())['prompt_sha256']
        self.assertEqual({k:hashlib.sha256(v.encode()).hexdigest() for k,v in values.items()},expected)


if __name__=='__main__':unittest.main()
