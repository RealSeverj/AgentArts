"""Offline RFMiD adapter checks using tiny ZIPs; no network or training."""
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

from scripts import fetch_rfmid_cloud as fetch


def metadata():
    keys = ('split', 'name', 'id', 'bytes', 'expected_images', 'sha256')
    return {'files': [{key: item[key] for key in keys} for item in fetch.PINNED]}


class RFMiDCloudFetchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rfmid_fetch_offline_')
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def tiny_archive(self):
        data, cache = self.root / 'data', self.root / 'cache'
        cache.mkdir()
        labels = ('ID,Disease_Risk,' + ','.join(f'label_{i}' for i in range(45))
                  + '\n1,' + ','.join(['0'] * 46) + '\n').encode()
        archive = cache / 'Training_Set.zip'
        # Bytes deliberately remain opaque: extraction must never convert PNGs.
        original_image = b'original-published-png-fixture-bytes'
        with zipfile.ZipFile(archive, 'w') as package:
            package.writestr('Training_Set/RFMiD_Training_Labels.csv', labels)
            package.writestr('Training_Set/Training/1.png', original_image)
        pinned = {**fetch.PINNED[0], 'bytes': archive.stat().st_size, 'expected_images': 1,
                  'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                  'csv_sha256': hashlib.sha256(labels).hexdigest()}
        source = {'name': 'RFMiD_training', 'archive_name': 'Training_Set.zip', 'bytes': pinned['bytes'],
                  'sha256': pinned['sha256'], 'extract_root': 'RFMiD',
                  'download_url': 'https://example.invalid/not-contacted'}
        return data, cache, archive, pinned, source, original_image

    def test_metadata_requires_fixed_hashes_sizes_and_public_ids(self):
        path = self.root / 'download_manifest.json'
        good = metadata()
        path.write_text(json.dumps(good), encoding='utf-8')
        sources = fetch.read_pinned_sources(path)
        self.assertEqual(sum(item['bytes'] for item in sources), 7988252748)
        self.assertEqual({item['extract_root'] for item in sources}, {'RFMiD'})
        self.assertEqual(sources[0]['archive_name'], 'Evaluation_Set.zip')
        for key, value in [('sha256', '0' * 64), ('id', 'other'), ('bytes', 12)]:
            changed = deepcopy(good)
            changed['files'][0][key] = value
            path.write_text(json.dumps(changed), encoding='utf-8')
            with self.subTest(key=key), self.assertRaises(fetch.legacy.FetchError):
                fetch.read_pinned_sources(path)

    def test_existing_zip_sha_gate_and_different_extracted_file_are_preserved(self):
        data, cache, archive, pinned, source, image = self.tiny_archive()
        before = archive.read_bytes()
        with patch('requests.sessions.Session.get', side_effect=AssertionError('network forbidden')):
            with redirect_stdout(io.StringIO()), patch.object(fetch, 'PINNED', (pinned,)):
                fetch.fetch_all([source], data, cache)
                fetch.fetch_all([source], data, cache)
                extracted = data / 'RFMiD/Training_Set/Training/1.png'
                self.assertEqual(extracted.read_bytes(), image)
                extracted.write_bytes(b'different existing bytes')
                with self.assertRaises(fetch.legacy.FetchError):
                    fetch.fetch_all([source], data, cache)
                self.assertEqual(extracted.read_bytes(), b'different existing bytes')
                wrong = {**source, 'sha256': '0' * 64}
                other = self.root / 'other_data'
                with self.assertRaisesRegex(fetch.legacy.FetchError, 'Existing final ZIP differs'):
                    fetch.fetch_all([wrong], other, cache)
                self.assertFalse((other / 'RFMiD').exists())
        self.assertEqual(archive.read_bytes(), before)

    def test_completed_range_journal_resumes_without_http_and_keeps_png_bytes(self):
        data, cache, archive, pinned, source, image = self.tiny_archive()
        partial = archive.with_name(archive.name + '.ranges.partial')
        archive.rename(partial)
        journal = archive.with_name(archive.name + '.ranges.json')
        journal.write_text(json.dumps({'format_version': 1, 'source': source['name'],
            'bytes': source['bytes'], 'sha256': source['sha256'], 'chunk_bytes': 32 * 1024**2,
            'completed_chunks': [0]}), encoding='utf-8')
        with patch('requests.sessions.Session.get', side_effect=AssertionError('network forbidden')):
            with redirect_stdout(io.StringIO()), patch.object(fetch, 'PINNED', (pinned,)):
                fetch.fetch_all([source], data, cache)
        self.assertTrue(archive.is_file())
        self.assertFalse(partial.exists())
        self.assertEqual((data / 'RFMiD/Training_Set/Training/1.png').read_bytes(), image)
        self.assertTrue(journal.is_file())

    def test_adapter_copies_source_and_logs_never_include_http_exception_text(self):
        source = {'bytes': 100, 'download_url': 'https://example.invalid/public?id=abc'}
        called = []
        original = lambda *args: called.append(args) or 7
        self.assertEqual(fetch.range_adapter(original)(source, 2, 10, None, None, 1, None), 7)
        self.assertTrue(called[0][0]['download_url'].endswith('&rfmid_range=20-29'))
        self.assertEqual(source['download_url'], 'https://example.invalid/public?id=abc')
        path = self.root / 'download_manifest.json'
        path.write_text(json.dumps(metadata()), encoding='utf-8')
        argv = ['fetch_rfmid_cloud.py', '--data-root', str(self.root / 'data'), '--cache-root', str(self.root / 'cache'),
                '--metadata', str(path)]
        output = io.StringIO()
        with patch('sys.argv', argv), patch.object(fetch, 'fetch_all', side_effect=requests.ConnectionError('SECRET_URL COOKIE_VALUE')):
            with redirect_stdout(output):
                self.assertEqual(fetch.main(), 1)
        self.assertNotIn('SECRET_URL', output.getvalue())
        self.assertNotIn('COOKIE_VALUE', output.getvalue())
        self.assertIn('ConnectionError', output.getvalue())
        with patch('sys.argv', argv + ['--plan']), redirect_stdout(io.StringIO()):
            self.assertEqual(fetch.main(), 0)
        self.assertFalse((self.root / 'cache').exists())


if __name__ == '__main__':
    unittest.main()
