"""Tiny offline mirror fixtures: whole-container MD5 and original per-file SHA."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import requests

from scripts import fetch_rfmid_kaggle as fetch


class KaggleRFMiDTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rfmid_mirror_offline_')
        self.root = Path(self.temporary.name)
        self.data, self.stage, self.cache = (self.root / name for name in ('data', 'stage', 'cache'))
        self.archive = self.root / 'fixture.zip'
        self.files = {'Training_Set/Training/1.png': b'published original PNG one',
                      'Training_Set/Training/2.png': b'published original PNG two',
                      'Training_Set/RFMiD_Training_Labels.csv': b'original labels bytes'}
        self.original = {'files': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                                   for name, data in self.files.items()}}
        self.write_zip()

    def tearDown(self):
        self.temporary.cleanup()

    def write_zip(self, extra=None, omit=None):
        with zipfile.ZipFile(self.archive, 'w') as z:
            for name, data in self.files.items():
                if name != omit:
                    z.writestr(name.split('/', 1)[0] + '/' + name, data)
            if extra:
                z.writestr(extra, b'unknown bytes')

    def extract(self, original=None):
        with redirect_stdout(io.StringIO()):
            return fetch.extract_and_publish(self.archive, original or self.original, self.data, self.stage)

    def test_all_files_sha_before_publish_and_exact_original_paths_idempotent(self):
        first = self.extract()
        self.assertEqual(first['published_files'], 3)
        self.assertEqual(first['files_verified_against_original_sha256'], 3)
        for name, body in self.files.items():
            self.assertEqual((self.data / 'RFMiD' / name).read_bytes(), body)
        second = self.extract()
        self.assertEqual(second['published_files'], 0)
        self.assertEqual(second['reused_identical_files'], 3)
        self.assertFalse((self.data / 'RFMiD/Training_Set/Training_Set').exists())
        self.assertFalse(first['image_conversion_performed'])
        self.assertFalse(first['labels_rebuilt'])

    def test_one_file_bad_sha_prevents_every_new_destination_file(self):
        original = deepcopy(self.original)
        original['files']['Training_Set/Training/2.png']['sha256'] = '0' * 64
        with self.assertRaisesRegex(fetch.FetchError, 'SHA256 differs'):
            self.extract(original)
        self.assertFalse((self.data / 'RFMiD').exists())

    def test_unknown_missing_traversal_and_conflicting_existing_file_are_rejected(self):
        for extra in ('Training_Set/Training_Set/extra.png', '../outside.png', 'Unknown/Unknown/1.png'):
            self.write_zip(extra=extra)
            with self.subTest(extra=extra), self.assertRaises(fetch.FetchError):
                self.extract()
            self.assertFalse((self.data / 'RFMiD').exists())
        self.write_zip(omit='Training_Set/Training/2.png')
        with self.assertRaisesRegex(fetch.FetchError, 'complete original-file set'):
            self.extract()
        self.write_zip()
        conflict = self.data / 'RFMiD/Training_Set/Training/2.png'
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b'preserve a different existing image')
        with self.assertRaisesRegex(fetch.FetchError, 'conflicts'):
            self.extract()
        self.assertEqual(conflict.read_bytes(), b'preserve a different existing image')
        self.assertFalse((conflict.parent / '1.png').exists())

    def test_wrong_container_md5_preserves_cache_and_skips_network(self):
        self.cache.mkdir()
        archive = self.cache / 'rfmid_kaggle_v1.zip'
        body = self.archive.read_bytes()
        archive.write_bytes(body)
        with patch.object(fetch, 'resolve_container_url', side_effect=AssertionError('network forbidden')):
            with self.assertRaisesRegex(fetch.FetchError, 'size/MD5 differs'):
                fetch.download_container(self.cache, total=len(body), expected_md5='0' * 32)
        self.assertEqual(archive.read_bytes(), body)

    def test_completed_md5_range_journal_resumes_without_http_or_original_sha_claim(self):
        self.cache.mkdir()
        body = self.archive.read_bytes()
        digest = hashlib.md5(body, usedforsecurity=False).hexdigest()
        partial = self.cache / 'rfmid_kaggle_v1.md5.ranges.partial'
        journal = self.cache / 'rfmid_kaggle_v1.md5.ranges.json'
        partial.write_bytes(body)
        identity = {'format_version': 1, 'source': 'RFMiD_Kaggle_v1_container', 'bytes': len(body),
                    'container_md5': digest, 'chunk_bytes': 8 * 1024**2, 'container_is_original_release_zip': False,
                    'completed_chunks': [0]}
        journal.write_text(json.dumps(identity), encoding='utf-8')
        with patch.object(fetch, 'resolve_container_url', side_effect=AssertionError('network forbidden')):
            with redirect_stdout(io.StringIO()):
                path = fetch.download_container(self.cache, total=len(body), expected_md5=digest)
        self.assertEqual(path.read_bytes(), body)
        self.assertFalse(partial.exists())
        saved = json.loads(journal.read_text())
        self.assertNotIn('sha256', saved)
        self.assertNotIn('download_url', saved)
        self.assertFalse(saved['container_is_original_release_zip'])

    def test_manifest_schema_fixed_checksum_and_log_suppression(self):
        # Self-contained schema fixture; the production CLI independently uses
        # its hard-pinned real manifest. No public dataset files are needed.
        manifest = self.root / 'original_manifest.json'
        fixture = {'format_version': 1, 'dataset': 'RFMiD original release', 'relative_to': 'data/RFMiD',
                   'file_count': 3203, 'png_count': 3200, 'csv_count': 3,
                   'files': {name: {'bytes': 1, 'sha256': '0' * 64} for name in fetch.expected_paths()}}
        for folder, _, _, csv, expected, _ in fetch.SPLITS:
            fixture['files'][f'{folder}/{csv}']['sha256'] = expected
        manifest.write_text(json.dumps(fixture), encoding='utf-8')
        checksum = fetch.sha256_file(manifest)
        original = fetch.read_original_manifest(manifest, checksum)
        self.assertEqual(len(original['files']), 3203)
        self.assertEqual(sum(name.endswith('.png') for name in original['files']), 3200)
        altered = self.root / 'altered_manifest.json'
        altered.write_bytes(manifest.read_bytes() + b' ')
        with self.assertRaisesRegex(fetch.FetchError, 'checksum mismatch'):
            fetch.read_original_manifest(altered, checksum)
        argv = ['fetch_rfmid_kaggle.py', '--data-root', str(self.data), '--cache-root', str(self.cache),
                '--original-manifest', str(manifest)]
        output = io.StringIO()
        with patch('sys.argv', argv), patch.object(fetch, 'read_original_manifest', return_value=original), patch.object(fetch, 'download_container',
                side_effect=requests.ConnectionError('SIGNED_URL_TOKEN SECRET_COOKIE')):
            with redirect_stdout(output):
                self.assertEqual(fetch.main(), 1)
        self.assertNotIn('SIGNED_URL_TOKEN', output.getvalue())
        self.assertNotIn('SECRET_COOKIE', output.getvalue())
        self.assertIn('ConnectionError', output.getvalue())


if __name__ == '__main__':
    unittest.main()
