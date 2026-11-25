"""模型网络重试、任务级延期以及组内后续任务不中断。"""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

from lightning_vln.model_transport import request_json,ModelTransportFailure
from lightning_vln.errors import SimulationUnavailable
from lightning_vln.uavon_tuning import cli
from lightning_vln.types import Decision
from uavon_fixtures import YawWorld
import run_uavon_tuning_subset as batch


class TransportTest(unittest.TestCase):
    def test_timeout_retry_is_bounded_and_resends_identical_request_without_motion(self):
        request=object();calls=[]
        with patch('urllib.request.urlopen',side_effect=TimeoutError('slow')) as http,patch('lightning_vln.model_transport.time.sleep') as sleep:
            with self.assertRaises(ModelTransportFailure) as caught:
                request_json(request,90,recovery=True,on_attempt=lambda:calls.append(1))
        self.assertEqual(http.call_count,3);self.assertEqual(len(calls),3);self.assertEqual(caught.exception.attempts,3)
        self.assertTrue(all(call.args==(request,) for call in http.call_args_list))
        self.assertEqual([call.kwargs['timeout'] for call in http.call_args_list],[90,180,180])
        self.assertEqual([c.args[0] for c in sleep.call_args_list],[5,15])
        self.assertNotIsInstance(caught.exception,SimulationUnavailable)

    def test_legacy_and_permanent_http_errors_are_not_retried(self):
        for recovery,error in [(False,TimeoutError()),(True,HTTPError('url',401,'auth',{},None))]:
            with patch('urllib.request.urlopen',side_effect=error) as http,patch('lightning_vln.model_transport.time.sleep') as sleep:
                with self.assertRaises(type(error)):request_json(object(),90,recovery=recovery)
            self.assertEqual(http.call_count,1);sleep.assert_not_called()

    def test_real_http503_recovers_without_restarting_service(self):
        class Handler(BaseHTTPRequestHandler):
            calls=0
            def do_POST(self):
                Handler.calls+=1;self.rfile.read(int(self.headers['Content-Length']))
                if Handler.calls==1:self.send_error(503);return
                body=b'{"ok":true}';self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            def log_message(self,*args):pass
        import urllib.request
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/test',b'{}')
            with patch('lightning_vln.model_transport.time.sleep'):
                self.assertEqual(request_json(request,2,recovery=True),{'ok':True})
            self.assertEqual(Handler.calls,2)
        finally:server.shutdown();server.server_close();thread.join()

    def test_cli_defers_first_task_then_scores_second_task_in_same_world(self):
        class World(YawWorld):
            current=0
            def prepare(self):self.current+=1
        world=World()
        class Model:
            def __init__(self,*args):self.http_attempts=0
            def decide(self,*args):
                if world.current==1:raise ModelTransportFailure(TimeoutError('slow'),3)
                return Decision('horizontal_explore','front',(400,400,600,600),'test','')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset=root/'tasks.json';output=root/'out'
            dataset.write_text(json.dumps([dict(episode_id=str(i),map_name='Barnyard_test',instruction='test',pose=[[0,10,0]],info=dict(geodesic_distance=10),size='small') for i in range(2)]))
            argv=['agent','--dataset',str(dataset),'--scene','Barnyard_test','--limit','0','--model','test','--model-hardware','fake','--sim-hardware','fake','--output',str(output),'--metrics-output',str(root/'metrics'),'--decisions','2','--initial-yaw90','--ascent-guard','--current-target-evidence','--record-mode','metrics-only','--isolated-model-recovery']
            with patch.object(sys,'argv',argv),patch.object(cli,'AirSimWorld',return_value=world),patch.object(cli,'VLM',Model):cli.main()
            deferred=json.loads((output/'deferred.json').read_text());rows=json.loads((output/'results.json').read_text())
            self.assertEqual([r['episode_id'] for r in deferred],['0']);self.assertTrue(deferred[0]['not_scored'])
            self.assertEqual([r['episode_id'] for r in rows],['1']);self.assertEqual(world.current,2)
            self.assertFalse((output/'episode_0000/summary.json').exists())

    def test_group_deferred_first_task_does_not_restart_or_hide_completed_peers(self):
        tasks=[dict(episode_id=str(i)) for i in range(5)];committed=[];calls=[]
        def run(root,pending):
            calls.append([t['episode_id'] for t in pending]);(root/'agent').mkdir()
            (root/'agent/deferred.json').write_text(json.dumps([dict(episode_id='0',not_scored=True)]))
        with tempfile.TemporaryDirectory() as tmp,patch.object(batch,'settled_results',return_value=tasks[1:]):
            with self.assertRaises(batch.DeferredModelTasks):batch.recover_scene(Path(tmp)/'group',tasks,run,lambda r:committed.extend(r),5)
        self.assertEqual(len(calls),1);self.assertEqual(committed,tasks[1:])

    def test_simulation_retry_excludes_already_deferred_model_task(self):
        tasks=[dict(episode_id=str(i)) for i in range(3)];calls=[];committed=[]
        def run(root,pending):
            calls.append([t['episode_id'] for t in pending]);(root/'agent').mkdir()
            if len(calls)==1:
                (root/'agent/deferred.json').write_text(json.dumps([dict(episode_id='0',not_scored=True)]));raise SimulationUnavailable('RPC down')
        with tempfile.TemporaryDirectory() as tmp,patch.object(batch,'settled_results',side_effect=[[tasks[1]],[tasks[2]]]):
            with self.assertRaises(batch.DeferredModelTasks):batch.recover_scene(Path(tmp)/'group',tasks,run,lambda r:committed.extend(r),5)
        self.assertEqual(calls,[['0','1','2'],['2']]);self.assertEqual(committed,tasks[1:])

    def test_recovery_upgrade_is_explicit_and_does_not_allow_policy_changes(self):
        old=dict(version=5,scene_restarts=5,model_transport_recoverable=True)
        new=dict(old,version=6,model_request_attempts=3,model_retry_timeout_s=180,model_retry_backoff_s=[5,15],model_fault_isolation='episode')
        self.assertTrue(batch.recovery_upgrade_allowed(old,new));self.assertFalse(batch.recovery_upgrade_allowed(old,dict(new,scene_restarts=100)))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=dict(recovery_policy=old,agent_args=['--model','same','--base-url','old']);b=dict(recovery_policy=new,agent_args=['--model','same','--base-url','new','--isolated-model-recovery']);(root/'manifest.json').write_text(json.dumps(a))
            with self.assertRaises(ValueError):batch.reusable_results(root,b,[],allow_resource_relocation=True)
            self.assertEqual(batch.reusable_results(root,b,[],allow_recovery_upgrade=True,allow_resource_relocation=True),[])
            with self.assertRaises(ValueError):batch.reusable_results(root,dict(b,agent_args=['--model','other','--isolated-model-recovery']),[],allow_recovery_upgrade=True,allow_resource_relocation=True)


if __name__=='__main__':unittest.main()
