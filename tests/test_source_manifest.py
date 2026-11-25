"""源码开发清单更新与已准备运行的隔离保护。"""
import json
from pathlib import Path
import tempfile
import unittest

from portable_experiments import core


class SourceManifestTests(unittest.TestCase):
    def repository(self, root):
        source = root/'src'
        source.mkdir(parents=True)
        (source/'policy.py').write_text('action = "forward"\n')
        core.save(source/'MANIFEST.json', dict(schema=1, layout='test', files={}))
        core.refresh_source_manifest(root)
        return source

    def test_explicit_refresh_registers_changed_added_and_removed_python_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.repository(root)
            (source/'policy.py').write_text('action = "stop"\n')
            with self.assertRaisesRegex(ValueError, 'refresh-source'):
                core.verify_bundle(source)
            core.refresh_source_manifest(root)
            core.verify_bundle(source)
            extra = source/'controller.py'
            extra.write_text('speed = 1\n')
            with self.assertRaisesRegex(ValueError, 'refresh-source'):
                core.verify_bundle(source)
            core.refresh_source_manifest(root)
            self.assertIn('controller.py', core.verify_bundle(source)['files'])
            extra.unlink()
            with self.assertRaisesRegex(ValueError, 'refresh-source'):
                core.verify_bundle(source)
            core.refresh_source_manifest(root)
            self.assertNotIn('controller.py', core.verify_bundle(source)['files'])

    def test_refresh_does_not_change_prepared_snapshot_or_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.repository(root/'repository')
            prepared = root/'prepared'
            frozen = prepared/'source_snapshot/src'
            core.clone(source, frozen)
            core.save(prepared/'portable-contract.json', dict(
                root=str(prepared), benchmark='uavon', files=core.files(prepared), inputs={}))
            contract = (prepared/'portable-contract.json').read_bytes()
            original = (frozen/'policy.py').read_bytes()
            (source/'policy.py').write_text('action = "stop"\n')
            core.refresh_source_manifest(root/'repository')
            self.assertEqual((frozen/'policy.py').read_bytes(), original)
            self.assertEqual((prepared/'portable-contract.json').read_bytes(), contract)
            core.verify_output(prepared)
            for path in (prepared, prepared/'source_snapshot', prepared/'launcher'):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, '冻结输出目录'):
                    core.refresh_source_manifest(path)
            (frozen/'policy.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, '冻结源码'):
                core.verify_output(prepared)

    def test_symlink_source_is_rejected_without_updating_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.repository(root/'repository')
            original = (source/'MANIFEST.json').read_bytes()
            outside = root/'outside.py'
            outside.write_text('outside = True\n')
            (source/'external.py').symlink_to(outside)
            with self.assertRaisesRegex(ValueError, '符号链接'):
                core.refresh_source_manifest(root/'repository')
            self.assertEqual((source/'MANIFEST.json').read_bytes(), original)

    def test_empty_source_does_not_replace_existing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.repository(root)
            original = (source/'MANIFEST.json').read_bytes()
            (source/'policy.py').unlink()
            with self.assertRaisesRegex(ValueError, '空清单'):
                core.refresh_source_manifest(root)
            self.assertEqual((source/'MANIFEST.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
