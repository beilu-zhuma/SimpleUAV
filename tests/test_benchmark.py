import json
import tempfile
import unittest
from pathlib import Path

from lightning_vln.benchmark import TEST_SCENES, build_full_testset, load_full_testset
from lightning_vln.evaluation import PROBLEM_SCENES
from lightning_vln.trainset import encoded


class FullTestsetTest(unittest.TestCase):
    def source(self, root):
        originals = {}
        for scene in TEST_SCENES:
            rows = [dict(episode_id=i, map_name=scene, description=f'object{i}',
                         pose=[[1, 2, 3]], info={'geodesic_distance': i+1}) for i in range(2)]
            (root/f'{scene}.json').write_text(json.dumps(rows))
            originals[scene] = rows
        return originals

    def test_full_coverage_filter_and_original_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            originals = self.source(root)
            rows, manifest = build_full_testset(root)
            self.assertEqual(len(rows), 20)
            self.assertEqual(manifest['source_total'], 28)
            self.assertEqual(len(manifest['excluded_ids']), 8)
            self.assertEqual(set(manifest['selected_by_scene']), set(TEST_SCENES)-PROBLEM_SCENES)
            self.assertEqual(len({t['episode_id'] for t in rows}), 20)
            for row in rows:
                original = dict(row)
                original['episode_id'] = original.pop('original_episode_id')
                self.assertEqual(original, originals[row['map_name']][original['episode_id']])

    def test_missing_scene_duplicate_and_train_input_are_rejected(self):
        for mode in ('missing', 'duplicate', 'train'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                originals = self.source(root)
                path = root/'Venice_test.json'
                if mode == 'missing':
                    path.unlink()
                elif mode == 'duplicate':
                    path.write_text(json.dumps(originals['Venice_test']*2))
                else:
                    path.write_text(json.dumps([dict(originals['Venice_test'][0], map_name='Venice_train')]))
                with self.assertRaises(ValueError):
                    build_full_testset(root)

    def test_manifest_rejects_partial_data_and_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'
            source.mkdir()
            originals = self.source(source)
            rows, manifest = build_full_testset(source)
            data, meta = root/'tasks.json', root/'benchmark.json'
            data.write_bytes(encoded(rows))
            meta.write_bytes(encoded(manifest))
            self.assertEqual(load_full_testset(data, meta), (rows, manifest))
            data.write_bytes(encoded(rows[:-1]))
            with self.assertRaisesRegex(ValueError, '不一致'):
                load_full_testset(data, meta)
            data.write_bytes(encoded(rows))
            originals['Venice_test'][0]['description'] = 'changed'
            (source/'Venice_test.json').write_text(json.dumps(originals['Venice_test']))
            with self.assertRaisesRegex(ValueError, '不一致'):
                load_full_testset(data, meta)
