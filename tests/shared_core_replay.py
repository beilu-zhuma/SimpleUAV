"""重构前后使用相同输入的CPU回放；不连接模型或仿真，不访问GPU。"""
from contextlib import redirect_stdout
from dataclasses import asdict
from io import BytesIO, StringIO
import importlib
import json
import hashlib
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image

def replay(benchmark, legacy=False):
    if benchmark != 'uavon':
        raise ValueError('只提供UAV-ON回放')

    def module(name):
        if legacy:
            return importlib.import_module('lightning_vln.' + name)
        if name.startswith('uavon_minimal'):
            name = 'adapters.uavon.minimal' + name[len('uavon_minimal'):]
        elif name in ('agent', 'vlm', 'context', 'types', 'evidence'):
            name = 'core.' + name
        return importlib.import_module('lightning_vln.' + name)
    types = module('types')
    data = BytesIO()
    Image.new('RGB', (32, 32), 'gray').save(data, 'JPEG')
    rgb = data.getvalue()

    class World:

        def __init__(self, scenario):
            self.p = np.zeros(3)
            self.yaw = 0.0
            self.events = []
            self.last_observe = {}
            self.last_stop = {}
            self.scenario = scenario
            self.commands = 0

        def position(self):
            return self.p.copy()

        def collided(self):
            return self.scenario == 'spawn' or (self.scenario == 'collision' and self.commands >= 2)

        def hold(self):
            self.events.append(['hold'])

        def face(self, goal):
            d = np.asarray(goal) - self.p
            if np.linalg.norm(d[:2]) > 0.05:
                self.yaw = math.atan2(d[1], d[0])
            self.events.append(['face', self.yaw])
            return True

        def advance(self, velocity, seconds):
            self.p += np.asarray(velocity) * seconds
            self.commands += 1
            self.events.append(['advance', np.asarray(velocity).tolist(), seconds])

        def observe(self, with_rgb=True):
            depth = 0.2 if self.scenario == 'blocked' else 8.0
            views = {}
            for name, angle in [('front', self.yaw), ('left', self.yaw - math.pi / 2), ('right', self.yaw + math.pi / 2), ('down', self.yaw)]:
                q = np.array([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])
                if name == 'down':
                    q = np.array([math.sin(angle / 2) * math.sin(math.pi / 4), -math.cos(angle / 2) * math.sin(math.pi / 4), math.sin(angle / 2) * math.cos(math.pi / 4), math.cos(angle / 2) * math.cos(math.pi / 4)])
                views[name] = types.View(rgb if with_rgb else b'', np.full((32, 32), depth), self.p.copy(), q, 90.0, 1)
            return types.Observation(self.p.copy(), views)
    Limits = module('agent').Limits

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k != 'timing' and (not k.endswith('_s'))}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, float):
            return round(value, 8)
        return value
    episodes = {}
    for scenario in ('stop', 'blocked', 'collision', 'spawn', 'up'):
        world = World(scenario)

        class Model:
            current_target_evidence = True
            motion_options = True

            def localize(self, *args):
                return [{'view': 'front', 'bbox': [400, 400, 600, 600], 'match': 'confirmed', 'evidence': '目标清晰', 'physical_object': True, 'class_match': True, 'attributes_match': True}] if scenario == 'stop' else []

            def decide(self, context, observation, iteration):
                action = 'stop' if scenario == 'stop' else 'vertical_explore' if scenario == 'up' else 'horizontal_explore'
                return types.Decision(action, 'front', (400, 400, 600, 600), '检查区域', '保持线索', 'up' if action == 'vertical_explore' else None, context.current_targets[0]['target_id'] if context.current_targets else None)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            out = Path(directory)
            result = module('uavon_minimal.agent').run_agent(world, Model(), '寻找黑色汽车', out, Limits(decisions=5), module('uavon_minimal.policy').Options(initial_yaw90=True, ascent_guard=True))
            records = [json.loads(line) for line in (out / 'decisions.jsonl').read_text().splitlines()]
            trajectory = [json.loads(line) for line in (out / 'trajectory.jsonl').read_text().splitlines()]
            metrics = None
            scorer = importlib.import_module('lightning_vln.evaluation' if legacy else 'lightning_vln.adapters.uavon.evaluation')
            metrics = scorer.score_episode(dict(pose=[[10, 0, -2]], info={'geodesic_distance': 10}), out, result)
            episodes[scenario] = clean(dict(summary=result, decisions=records, trajectory=trajectory, events=world.events, metrics=metrics))
    requests = []
    with tempfile.TemporaryDirectory() as directory:
        world = World('stop')
        observation = world.observe()
        log = Path(directory) / 'model.jsonl'
        guard = module('uavon_minimal.policy').AscentGuard(np.zeros(3), module('uavon_minimal.policy').Options(initial_yaw90=True, ascent_guard=True))
        context = module('uavon_minimal.vlm').MinimalContext('寻找黑色汽车', 4, guard)
        cls = module('uavon_minimal.vlm').MinimalVLM
        context.feedback['motion_policy'] = {'stop_distance_m': 10.0}
        context.current_targets = []
        vlm = cls('http://unit.invalid/v1', 'model', log, current_target_evidence=True, motion_options=True)
        value = dict(action='horizontal_explore', view='front', bbox=[400, 400, 600, 600], reason='检查', memory='线索', vertical_direction=None, target_id=None)
        valid = json.dumps(value)
        invalid = json.dumps(dict(value, bbox=[0, 0, 0, 0]))

        def navigation():
            return vlm.decide(context, observation, 1)
        spec = '寻找黑色汽车'
        responses = iter([json.dumps({'candidates': []}), invalid, valid, valid])

        def urlopen(request, *args, **kwargs):
            body = json.loads(request.data)
            requests.append(dict(sha256=hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), phase=body['response_format']['json_schema']['name']))
            return StringIO(json.dumps(dict(choices=[dict(finish_reason='stop', message=dict(content=next(responses)))])))
        with patch('urllib.request.urlopen', side_effect=urlopen):
            vlm.localize(spec, observation, 0)
            navigation()
            feedback = dict(status='blocked', position_ned_m=[0, 0, 0], motion_policy={'stop_distance_m': 10.0})
            context.record(observation, types.Decision('horizontal_explore', 'front', (400, 400, 600, 600), '检查区域', '保持线索'), feedback)
            context.current_targets = []
            navigation()
    return dict(benchmark=benchmark, episodes=episodes, requests=requests)
if __name__ == '__main__':
    import sys
    result = replay(sys.argv[1], legacy='--legacy' in sys.argv)
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
