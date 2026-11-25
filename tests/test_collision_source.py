"""专用碰撞接口、任务隔离及运动停止回归。"""
import copy
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from lightning_vln.simulator import AirSimWorld
from lightning_vln.errors import SimulationUnavailable
from lightning_vln.agent import Limits, move_straight
from lightning_vln.evaluation import score_episode
from lightning_vln.uavon_minimal.agent import run_agent
from lightning_vln.uavon_minimal.policy import Options
from test_interfaces import FakeRPC
from test_simple_agent import FakeWorld
from run_uavon_tuning_subset import routed_agent_args, reusable_results
import json

class CollisionSourceTest(unittest.TestCase):
    def world(self):
        world=AirSimWorld.__new__(AirSimWorld);world.vehicle='Drone_1'
        world.client=FakeRPC();world.control=world.client
        world._collision_baseline_stamp=0;world._collision_latched=None
        world._timestamps={};world._fovs={k:90 for k in ('front','left','right','down')}
        original=world.client.call
        world.event=dict(has_collided=False,time_stamp=0,object_name='')
        def call(method,*args,**kwargs):
            if method=='simGetCollisionInfo':return copy.deepcopy(world.event)
            return original(method,*args,**kwargs)
        world.client.call=call
        return world

    def test_dedicated_true_overrides_broken_state_and_observation(self):
        w=self.world();w.event=dict(has_collided=True,time_stamp=123,object_name='house')
        self.assertTrue(w.state()['collision']['has_collided'])
        w.observe(with_rgb=False);self.assertTrue(w.last_observe['collision'])
        self.assertEqual(w.collision_record()['object_name'],'house')

    def test_latch_survives_subsequent_false_and_reset_excludes_old_event(self):
        w=self.world();w.event=dict(has_collided=True,time_stamp=100,object_name='tree')
        self.assertTrue(w.collided());w.event['has_collided']=False;self.assertTrue(w.collided())
        w.event['has_collided']=True;w._reset_collision_baseline();self.assertFalse(w.collided())
        w.event['time_stamp']=101;self.assertTrue(w.collided())

    def test_invalid_dedicated_info_is_not_silently_free_flight(self):
        for event in (None,{},dict(has_collided=True,time_stamp=0)):
            w=self.world();w.event=event
            with self.assertRaises(SimulationUnavailable):w.collided()

    def test_collision_hold_only_brakes_and_face_does_not_rotate(self):
        w=self.world();w.event=dict(has_collided=True,time_stamp=1,object_name='house')
        calls=[];original=w.control.call
        def call(method,*args,**kwargs):calls.append(method);return original(method,*args,**kwargs)
        w.control.call=call;w.hold()
        self.assertEqual(w.last_stop['mode'],'collision_brake')
        self.assertEqual(calls.count('moveByVelocity'),1)
        self.assertFalse(w.face(np.array([10,0,0])));self.assertNotIn('rotateToYaw',calls)

    def test_collision_during_observation_prevents_model_call(self):
        class World(FakeWorld):
            contact=False
            def observe(self,*args,**kwargs):
                result=super().observe(*args,**kwargs);self.contact=True;return result
            def collided(self):return self.contact
        class Model:
            def decide(self,*args):raise AssertionError('碰撞后不应再请求模型')
        with tempfile.TemporaryDirectory() as tmp:
            result=run_agent(World(),Model(),'test',Path(tmp),Limits(),Options(ascent_guard=True),record_details=False)
        self.assertEqual(result['status'],'collision');self.assertEqual(result['decisions'],0)

    def test_four_gpu_routing_changes_only_resource_fields(self):
        args=['--base-url','old','--model','alias','--model-hardware','Qwen GPU2 TP1',
              '--sim-hardware','GPU2 C4','--decisions','150','--history-turns','4','--stop-distance','10']
        original=args.copy()
        endpoints={str(g):dict(base_url=f'http://localhost:{19000+g}/v1',model=f'gpu{g}') for g in range(4)}
        for g in range(4):
            got=routed_agent_args(args,endpoints,g)
            self.assertEqual(got[1],endpoints[str(g)]['base_url']);self.assertEqual(got[3],f'gpu{g}')
            self.assertEqual(got[5],f'Qwen GPU{g} TP1');self.assertEqual(got[7],f'GPU{g} C4')
            self.assertEqual(got[8:],args[8:])
        self.assertEqual(args,original);self.assertEqual(routed_agent_args(args,None,0),args)

    def test_collision_fix_cannot_resume_legacy_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'manifest.json').write_text(json.dumps({}))
            with self.assertRaisesRegex(ValueError,'collision_detection_protocol'):
                reusable_results(root,{'collision_detection_protocol':'airsim_dedicated_collision_latched_v1'},[])

    def test_birth_contact_is_terminal_not_initialization_error(self):
        w=self.world();original=w.control.call
        def call(method,*args,**kwargs):
            result=original(method,*args,**kwargs)
            if method=='simSetVehiclePose':
                w.event=dict(has_collided=True,time_stamp=123,object_name='spawn_house')
                w.client.pending_reads=2
            return result
        w.control.call=call
        with patch('lightning_vln.simulator.time.sleep'):
            w.prepare(w.client.target,[0,0,0,1])
        self.assertTrue(w.last_prepare['spawn_collision']);self.assertEqual(w.last_prepare['collision_phase'],'spawn')
        self.assertTrue(w.collided());self.assertTrue(w.collision_record()['spawn_collision'])
        w.event=dict(has_collided=False,time_stamp=0,object_name='');w.control.call=original
        with patch('lightning_vln.simulator.time.sleep'):
            w.prepare(w.client.target,[0,0,0,1])
        self.assertFalse(w.last_prepare.get('spawn_collision',False));self.assertFalse(w.collided())

    def test_birth_collision_cli_settles_then_continues_next_task_without_model_call(self):
        import sys
        from lightning_vln.uavon_tuning import cli
        from uavon_fixtures import YawWorld
        from lightning_vln.types import Decision
        class World(YawWorld):
            count=0
            def prepare(self):
                self.count+=1
                self.last_prepare=dict(position_tolerance_m=.5,error_m=1 if self.count==1 else 0,
                                       stop=dict(speed_m_s=0,z_error_m=0),spawn_collision=self.count==1,
                                       collision_phase='spawn' if self.count==1 else None)
            def collided(self):return self.count==1
        world=World();calls=[]
        class Model:
            def __init__(self,*args):pass
            def decide(self,*args):
                calls.append(world.count)
                return Decision('horizontal_explore','front',(400,400,600,600),'test','')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset=root/'tasks.json';output=root/'out'
            dataset.write_text(json.dumps([dict(episode_id=str(i),map_name='Barnyard_test',instruction='test',
                                              pose=[[0,10,0]],info=dict(geodesic_distance=10),size='small') for i in range(2)]))
            argv=['agent','--dataset',str(dataset),'--scene','Barnyard_test','--limit','0','--model','test',
                  '--model-hardware','fake','--sim-hardware','fake','--output',str(output),'--metrics-output',str(root/'metrics'),
                  '--decisions','2','--initial-yaw90','--ascent-guard','--current-target-evidence','--record-mode','metrics-only']
            with patch.object(sys,'argv',argv),patch.object(cli,'AirSimWorld',return_value=world),patch.object(cli,'VLM',Model):cli.main()
            rows=json.loads((output/'results.json').read_text());first=rows[0]
            self.assertEqual(len(rows),2);self.assertEqual(first['status'],'collision');self.assertEqual(first['decisions'],0)
            self.assertTrue(first['spawn_collision']);self.assertEqual(first['collision_phase'],'spawn')
            self.assertFalse(first['success']);self.assertFalse(first['oracle_success']);self.assertEqual(first['spl'],0)
            self.assertFalse(rows[1]['spawn_collision']);self.assertEqual(calls,[2])
            from run_uavon_tuning_subset import wait_for_initialization
            import threading
            agent=type('Process',(),{'poll':lambda self:None})()
            wait_for_initialization(agent,output/'episode_0000/initialization.json',threading.Event())

    def test_spawn_fix_resume_requires_explicit_upgrade_and_fixed_collision_protocol(self):
        from lightning_vln.simulator import COLLISION_PROTOCOL,SPAWN_COLLISION_POLICY
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=dict(collision_detection_protocol=COLLISION_PROTOCOL)
            (root/'manifest.json').write_text(json.dumps(old));new=dict(old,spawn_collision_policy=SPAWN_COLLISION_POLICY)
            with self.assertRaisesRegex(ValueError,'spawn_collision_policy'):reusable_results(root,new,[])
            self.assertEqual(reusable_results(root,new,[],allow_spawn_collision_upgrade=True),[])

if __name__=='__main__':unittest.main()
