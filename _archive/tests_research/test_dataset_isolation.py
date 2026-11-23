"""Single-source/task manifests preserve bytes and refuse leakage/overwrite."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.isolate_training_datasets import (PROTOCOLS, SPLITS, build_isolation,
                                               digest_file, protocol_root)


def row(source, split, name, task='dr_grading', **changes):
    path = source + '/' + name + '.png'
    result = {'sample_id': task + ':' + name, 'dataset': source, 'task': task, 'split': split,
              'image_paths': [path], 'image_path': path, 'patient_id': 'patient:' + name,
              'labels': {'dr_grade': 2}, 'label_mask': {'dr_grade': True}, 'annotations': {},
              'image_sha256': [hashlib.sha256(path.encode()).hexdigest()],
              'content_group_id': 'content:' + path, 'leakage_group_id': 'group:' + path}
    result.update(changes)
    return result


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='source_isolation_')
        self.root = Path(self.temp.name)
        self.data, self.output = self.root / 'data', self.root / 'isolated'
        self.data.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, values, task='dr_grading', protocol='strict_cross_domain', raw=False):
        folder = protocol_root(self.data, protocol) / 'tasks' / task
        folder.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            records = values.get(split, [])
            payload = b''.join((value if raw else (json.dumps(value, ensure_ascii=False) + '\n').encode()) for value in records)
            (folder / (split + '.jsonl')).write_bytes(payload)
        return folder

    def fixture(self):
        values = {split: [row('One', split, 'one_' + split), row('Two', split, 'two_' + split)]
                  for split in ('train', 'val', 'test')}
        values['external_test'] = [row('External', 'external_test', 'external')]
        self.write(values)
        return values

    def build(self, **kwargs):
        return build_isolation(self.data, self.output, protocols=('strict_cross_domain',), **kwargs)

    def test_single_source_splits_global_membership_and_original_bytes(self):
        values = self.fixture()
        originals = {p: p.read_bytes() for p in self.data.rglob('*.jsonl')}
        result = self.build()
        self.assertTrue(result['all_checks_passed'])
        self.assertEqual(result['publication_status'], 'published_and_verified')
        self.assertEqual(result['combination_count'], 3)
        self.assertEqual(result['trainable_combinations'], 2)
        self.assertTrue(all(p.read_bytes() == raw for p, raw in originals.items()))
        all_ids = {split: [] for split in SPLITS}
        for combo in result['combinations']:
            for split, entry in combo['splits'].items():
                path = self.output / entry['path']
                self.assertTrue(path.is_file())
                self.assertEqual(digest_file(path), entry['sha256'])
                for text in path.read_text().splitlines():
                    record = json.loads(text)
                    self.assertEqual(record['dataset'], combo['source'])
                    self.assertEqual(record['task'], combo['task'])
                    self.assertEqual(record['split'], split)
                    all_ids[split].append(record['sample_id'])
        for split in SPLITS:
            self.assertCountEqual(all_ids[split], [r['sample_id'] for r in values[split]])
        self.assertTrue(all(item['legacy'] for item in result['legacy_mixed_roots']))

    def test_raw_record_lines_preserve_spacing_crlf_and_missing_final_newline(self):
        train = row('One', 'train', 'train')
        validation = row('One', 'val', 'validation')
        raw_train = b'  ' + json.dumps(train, separators=(', ', ': ')).encode() + b'\r\n'
        raw_val = json.dumps(validation, separators=(',', ':')).encode()
        self.write({'train': [raw_train, b'\r\n'], 'val': [raw_val]}, raw=True)
        result = self.build()
        self.assertTrue(result['all_checks_passed'])
        folder = self.output / 'strict_cross_domain/tasks/dr_grading/One'
        self.assertEqual((folder / 'train.jsonl').read_bytes(), raw_train)
        self.assertEqual((folder / 'val.jsonl').read_bytes(), raw_val)

    def test_dry_run_creates_no_output(self):
        self.fixture()
        result = self.build(dry_run=True)
        self.assertTrue(result['all_checks_passed'])
        self.assertFalse(self.output.exists())

    def test_existing_identity_reuses_read_only_and_other_output_refused(self):
        self.fixture()
        self.build()
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.rglob('*') if p.is_file()}
        result = self.build()
        self.assertEqual(result['publication_status'], 'existing_identical_output_reused_read_only')
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
        (self.output / 'unexpected.json').write_text('{}')
        with self.assertRaises(ValueError):
            self.build()
        other = self.root / 'other'
        other.mkdir()
        with self.assertRaises(ValueError):
            build_isolation(self.data, other, ('strict_cross_domain',))

    def test_changed_input_or_changed_commit_manifest_refuses_overwrite(self):
        self.fixture()
        self.build()
        committed = self.output / 'isolation_manifest.json'
        before = committed.read_bytes()
        source = protocol_root(self.data, 'strict_cross_domain') / 'tasks/dr_grading/train.jsonl'
        source.write_bytes(source.read_bytes() + b'\n')
        with self.assertRaises(ValueError):
            self.build()
        self.assertEqual(committed.read_bytes(), before)

    def test_tampered_commit_metadata_cannot_reuse_valid_data_files(self):
        self.fixture()
        self.build()
        committed = self.output / 'isolation_manifest.json'
        value = json.loads(committed.read_text(encoding='utf-8'))
        value['combinations'][0]['trainable'] = not value['combinations'][0]['trainable']
        committed.write_text(json.dumps(value), encoding='utf-8')
        before = committed.read_bytes()
        with self.assertRaises(ValueError):
            self.build()
        self.assertEqual(committed.read_bytes(), before)

    def test_nontrainable_unknown_supervision_and_external_only_kept(self):
        values = self.fixture()
        values['train'][0]['label_mask'] = {'dr_grade': False}
        values['train'][0]['labels'] = {'dr_grade': None}
        self.write(values)
        result = self.build()
        combinations = {item['source']: item for item in result['combinations']}
        self.assertFalse(combinations['One']['trainable'])
        self.assertFalse(combinations['External']['trainable'])
        self.assertTrue(combinations['Two']['trainable'])
        self.assertEqual(combinations['External']['splits']['train']['records'], 0)

    def test_structural_annotation_supervision_is_effective(self):
        values = {}
        for split in ('train', 'val'):
            values[split] = [row('One', split, split, task='lesion_segmentation', labels={},
                                 label_mask={'MA': True}, annotations={'masks': {'MA': {'path': 'prepared/assets/nonexistent.png'}}})]
        self.write(values, task='lesion_segmentation')
        result = self.build()
        self.assertTrue(result['all_checks_passed'])
        self.assertTrue(result['combinations'][0]['trainable'])
        # No mask or image file was created: this builder is metadata-only.
        self.assertFalse((self.data / 'prepared/assets/nonexistent.png').exists())

    def test_duplicate_id_and_task_split_mismatch_fail_without_creating_output(self):
        first = row('One', 'train', 'same')
        self.write({'train': [first, deepcopy(first)], 'val': [row('One', 'wrong', 'v')]})
        result = self.build()
        self.assertFalse(result['all_checks_passed'])
        self.assertIn('duplicate_sample_id_within_file', result['error_counts'])
        self.assertIn('record_task_or_split_mismatch', result['error_counts'])
        self.assertFalse(self.output.exists())

    def test_same_image_across_tasks_same_split_legal_other_split_fails(self):
        shared = row('One', 'train', 'shared')
        self.write({'train': [shared], 'val': [row('One', 'val', 'v')]})
        quality = {**shared, 'task': 'image_quality', 'sample_id': 'quality:shared'}
        self.write({'train': [quality]}, task='image_quality')
        self.assertTrue(self.build(dry_run=True)['all_checks_passed'])
        quality['split'] = 'test'
        self.write({'test': [quality]}, task='image_quality')
        result = self.build(dry_run=True)
        self.assertFalse(result['all_checks_passed'])
        self.assertIn('same_source_image_path_cross_split_leakage', result['error_counts'])

    def test_patient_scoped_by_dataset_but_content_and_groups_global(self):
        train = row('One', 'train', 'a', patient_id='same-patient-name')
        test = row('Two', 'test', 'b', patient_id='same-patient-name')
        self.write({'train': [train], 'test': [test]})
        self.assertTrue(self.build(dry_run=True)['all_checks_passed'])
        for field, value, expected in [('image_sha256', train['image_sha256'], 'global_image_content_sha256_cross_split_leakage'),
                                        ('content_group_id', train['content_group_id'], 'global_content_group_cross_split_leakage'),
                                        ('leakage_group_id', train['leakage_group_id'], 'global_leakage_group_cross_split_leakage')]:
            changed = deepcopy(test)
            changed[field] = value
            self.write({'train': [train], 'test': [changed]})
            result = self.build(dry_run=True)
            self.assertIn(expected, result['error_counts'])

    def test_known_patient_within_source_cannot_cross_split(self):
        self.write({'train': [row('One', 'train', 'a', patient_id='p')],
                    'test': [row('One', 'test', 'b', patient_id='p')]})
        result = self.build(dry_run=True)
        self.assertIn('dataset_scoped_patient_cross_split_leakage', result['error_counts'])

    def test_protocols_are_alternatives_not_cross_protocol_leakage(self):
        a = row('One', 'train', 'same-image')
        self.write({'train': [a]})
        b = {**a, 'split': 'test'}
        self.write({'test': [b]}, protocol='within_source')
        result = build_isolation(self.data, self.output, PROTOCOLS)
        self.assertTrue(result['all_checks_passed'])
        self.assertEqual(result['base_protocols'], list(PROTOCOLS))

    def test_path_names_case_collisions_and_legacy_output_protected(self):
        self.write({'train': [row('../escape', 'train', 'a')]})
        result = self.build(dry_run=True)
        self.assertIn('unsafe_source_name', result['error_counts'])
        self.write({'train': [row('One', 'train', 'a'), row('one', 'train', 'b')]})
        result = self.build(dry_run=True)
        self.assertIn('portable_task_source_case_collision', result['error_counts'])
        for target in [self.data, protocol_root(self.data, 'strict_cross_domain'),
                       protocol_root(self.data, 'strict_cross_domain') / 'tasks/new_root']:
            with self.subTest(target=target), self.assertRaises(ValueError):
                build_isolation(self.data, target, ('strict_cross_domain',), dry_run=True)


if __name__ == '__main__':
    unittest.main()
