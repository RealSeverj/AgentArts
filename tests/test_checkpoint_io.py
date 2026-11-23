"""Snapshots survive later last.pt replacement; failures preserve old best."""
import errno
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.checkpoint_io import publish_checkpoint_snapshot


class CheckpointSnapshotTests(unittest.TestCase):
    def test_hardlink_snapshot_survives_atomic_last_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            last, best = root / 'last.pt', root / 'best.pt'
            last.write_bytes(b'epoch one')
            self.assertEqual(publish_checkpoint_snapshot(last, best), 'hardlink')
            self.assertTrue(os.path.samefile(last, best))
            next_last = root / 'next.pt'
            next_last.write_bytes(b'epoch two')
            os.replace(next_last, last)
            self.assertEqual(best.read_bytes(), b'epoch one')
            self.assertEqual(last.read_bytes(), b'epoch two')
            self.assertEqual(publish_checkpoint_snapshot(last, best), 'hardlink')
            self.assertEqual(best.read_bytes(), b'epoch two')
            self.assertEqual(list(root.glob('.*.snapshot-*.tmp')), [])

    def test_unsupported_link_copies_then_atomically_replaces_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            last, best = root / 'last.pt', root / 'best.pt'
            last.write_bytes(b'new checkpoint')
            best.write_bytes(b'old checkpoint')
            real_replace = os.replace
            def check_before_replace(source, destination):
                self.assertEqual(best.read_bytes(), b'old checkpoint')
                self.assertEqual(Path(source).read_bytes(), last.read_bytes())
                real_replace(source, destination)
            with patch('training.checkpoint_io.os.link', side_effect=OSError(errno.EXDEV, 'cross device')):
                with patch('training.checkpoint_io.os.replace', side_effect=check_before_replace):
                    self.assertEqual(publish_checkpoint_snapshot(last, best), 'copy')
            self.assertEqual(best.read_bytes(), b'new checkpoint')
            self.assertFalse(os.path.samefile(last, best))
            self.assertEqual(list(root.glob('.*.snapshot-*.tmp')), [])

    def test_failures_preserve_old_best_cleanup_temp_and_propagate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            last, best = root / 'last.pt', root / 'best.pt'
            last.write_bytes(b'new')
            best.write_bytes(b'old')
            for failing_stage in ('link', 'copy', 'publish'):
                with self.subTest(stage=failing_stage):
                    if failing_stage == 'link':
                        first = patch('training.checkpoint_io.os.link', side_effect=OSError(errno.EIO, 'I/O failure'))
                        second = patch('training.checkpoint_io.shutil.copyfile')
                    elif failing_stage == 'copy':
                        first = patch('training.checkpoint_io.os.link', side_effect=OSError(errno.EXDEV, 'cross device'))
                        def broken_copy(source, destination):
                            Path(destination).write_bytes(b'incomplete')
                            raise OSError(errno.ENOSPC, 'disk full')
                        second = patch('training.checkpoint_io.shutil.copyfile', side_effect=broken_copy)
                    else:
                        first = patch('training.checkpoint_io.os.replace', side_effect=OSError(errno.EACCES, 'publication refused'))
                        second = patch('training.checkpoint_io.shutil.copyfile')
                    with first, second, self.assertRaises(OSError):
                        publish_checkpoint_snapshot(last, best)
                    self.assertEqual(last.read_bytes(), b'new')
                    self.assertEqual(best.read_bytes(), b'old')
                    self.assertEqual(list(root.glob('.*.snapshot-*.tmp')), [])

    def test_same_path_or_same_inode_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            last, alias = root / 'last.pt', root / 'alias.pt'
            last.write_bytes(b'checkpoint')
            with self.assertRaises(ValueError):
                publish_checkpoint_snapshot(last, last)
            os.link(last, alias)
            with self.assertRaises(ValueError):
                publish_checkpoint_snapshot(last, alias)
            self.assertEqual(last.read_bytes(), b'checkpoint')


if __name__ == '__main__':
    unittest.main()
