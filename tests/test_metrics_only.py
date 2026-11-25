"""不留RGB-D或轨迹时，四指标与完整记录及原样官方脚本一致。"""
from io import StringIO
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np

from lightning_vln.agent import Limits
from lightning_vln.evaluation import score_episode,export_episode,export_metric_scalars,run_official_metrics
from lightning_vln.uavon_minimal.agent import run_agent
from lightning_vln.uavon_minimal.policy import Options
from lightning_vln.types import Decision
from uavon_fixtures import YawWorld,up
import run_uavon_tuning_subset as batch
from lightning_vln.uavon_tuning import cli


class MetricsOnlyTest(unittest.TestCase):
    def test_cli_end_to_end_persists_all_metrics_and_only_scalar_scoring_input(self):
        class World(YawWorld):
            def prepare(self):pass
        class Model:
            def __init__(self,*args):
                if args[2] is not None:raise AssertionError('禁止模型日志')
            def decide(self,*args):
                return Decision('horizontal_explore','front',(400,400,600,600),'探索','线索')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset=root/'tasks.json';out=root/'out'
            dataset.write_text(json.dumps([dict(episode_id='one',map_name='Barnyard_test',instruction='目标',
                pose=[[0,10,0]],info=dict(geodesic_distance=10),size='small')]))
            argv=['agent','--dataset',str(dataset),'--scene','Barnyard_test','--model','test','--model-hardware','fake',
                '--sim-hardware','fake','--output',str(out),'--decisions','2','--history-turns','4',
                '--initial-yaw90','--ascent-guard','--current-target-evidence','--record-mode','metrics-only']
            with patch.object(sys,'argv',argv),patch.object(cli,'AirSimWorld',return_value=World()),patch.object(cli,'VLM',Model),patch.object(cli.subprocess,'check_output',side_effect=['test\n','']):
                cli.main()
            row=json.loads((out/'results.json').read_text())[0]
            self.assertEqual(row['record_mode'],'metrics-only')
            for k in ('success','oracle_success','spl','ne_m','sampled_path_length_m'):self.assertIsNotNone(row[k])
            self.assertEqual(row['ne_m'],row['final_distance_m'])
            episode=out/'episode_0000'
            self.assertFalse((episode/'trajectory.jsonl').exists())
            self.assertFalse((episode/'observations.zip').exists())
            self.assertFalse((episode/'initial-scan.zip').exists())
            self.assertFalse((episode/'model.jsonl').exists())
            self.assertTrue((out/'uavon_metrics/official_metrics.json').exists())
            attempt=root/'attempt';attempt.mkdir();out.rename(attempt/'agent')
            self.assertEqual(batch.settled_results(attempt,json.loads(dataset.read_text())),[row])

    def test_identical_physics_four_metrics_rotation_and_climb_guard_without_files(self):
        class Model:
            def decide(self,context,observation,iteration):return up()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'full';b=root/'metrics';a.mkdir();b.mkdir()
            wa,wb=YawWorld(),YawWorld();limits=Limits(decisions=7,history_turns=4)
            options=Options(initial_yaw90=True,ascent_guard=True);trace=StringIO()
            full=run_agent(wa,Model(),'目标',a,limits,options)
            with patch('lightning_vln.uavon_minimal.agent.save_observation',side_effect=AssertionError('禁止图像')):
                thin=run_agent(wb,Model(),'目标',b,limits,options,record_details=False,metric_trace=trace)
            self.assertEqual(list(b.iterdir()),[])
            np.testing.assert_allclose(wa.position(),wb.position())
            self.assertEqual(wa.events,wb.events)
            self.assertEqual(full['ascent_guard_runtime_blocks'],thin['ascent_guard_runtime_blocks'])
            task=dict(episode_id='one',map_name='Barnyard_test',pose=[[0,0,-12]],info=dict(geodesic_distance=12),size='small')
            old=score_episode(task,a,full)
            new=score_episode(task,b,thin,trajectory_rows=[json.loads(s) for s in trace.getvalue().splitlines()])
            self.assertEqual(old,new)
            self.assertGreater(new['trajectory_samples'],7)
            self.assertEqual(new['ne_m'],new['final_distance_m'])
            self.assertEqual({p.name for p in b.iterdir()},{'evaluation.json'})

    def test_boundary_collision_zero_path_and_multiple_targets_match_official(self):
        for status,distance,positions in [('stop',20,[[0,0,0],[0,0,0]]),
            ('decision_limit',20,[[0,0,0],[3,0,0]]),('collision',1,[[0,0,0],[3,0,0]]),
            ('stop',19.999,[[0,0,0],[3,0,0]]),('decision_limit',20.001,[[0,0,0],[3,0,0]])]:
            with self.subTest(status=status,distance=distance),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'full').mkdir();(root/'metrics').mkdir()
                final=positions[-1];task=dict(episode_id='one',map_name='Barnyard_test',
                    pose=[[final[0]+distance,0,0],[100,100,100]],info=dict(geodesic_distance=10),size='small')
                rows=[dict(event='start',position_ned_m=positions[0]),dict(event='decision_end',position_ned_m=final),dict(event='final',position_ned_m=final)]
                summary=dict(status=status,decisions=1,start_position_ned_m=positions[0],position_ned_m=final)
                (root/'full/trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                a=score_episode(task,root/'full',summary);b=score_episode(task,root/'metrics',summary,trajectory_rows=rows)
                self.assertEqual(a,b)
                export_episode(task,root/'full',dict(summary,**a),root/'official-full')
                export_metric_scalars(task,dict(summary,**b),root/'official-scalars')
                one=run_official_metrics(root/'official-full',1);two=run_official_metrics(root/'official-scalars',1)
                for key in ('sr','osr','spl','dts_m'):self.assertEqual(one[key],two[key])
                for p in (root/'official-scalars').glob('*/*.json/task_*/log/trajectory.jsonl'):
                    lines=p.read_text().splitlines();self.assertEqual(len(lines),1)
                    self.assertNotIn('sensors',json.loads(lines[0]))

    def test_aggregate_requires_all_four_and_does_not_mix_sr_only(self):
        task=dict(episode_id='one');row=dict(episode_id='one',status='stop',success=True,oracle_success=True,
            spl=.5,ne_m=3.,final_distance_m=3.,sampled_path_length_m=20.,record_mode='metrics-only')
        result=batch.aggregate([task],[row]);self.assertEqual(result['record_mode'],'metrics-only')
        self.assertEqual([result[k] for k in ('sr','osr','spl','ne_m')],[1,1,.5,3])
        with self.assertRaises(ValueError):batch.aggregate([task],[dict(row,spl=None)])
        with self.assertRaises(ValueError):batch.aggregate([task,dict(episode_id='two')],[row,dict(row,episode_id='two',record_mode='results-only')])


if __name__=='__main__':unittest.main()
