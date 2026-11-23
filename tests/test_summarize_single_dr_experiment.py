"""Pure-statistics fixtures for one complete DDR test export; no training/GPU."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.analyze_dr_evaluation import sha256_file, summarize_predictions
from scripts.summarize_single_dr_experiment import (DEFAULT_REPLICATES, EXPECTED_COUNT,
    build_report, supplementary_bootstrap)


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def dump_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


class SingleDRExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='single_dr_statistics_')
        self.root = Path(self.temporary.name)
        self.run, self.data = self.root / 'evaluation', self.root / 'data'
        self.index_path = self.data / 'prepared/source_isolated/isolation_manifest.json'
        relative = 'strict_cross_domain/tasks/dr_grading/DDR/test.jsonl'
        self.manifest_path = self.index_path.parent / relative
        rows, predictions = {}, []
        for i in range(EXPECTED_COUNT):
            sid = f'DDR:test:{i:05d}'
            grade = i % 5
            group = f'content_{i // 2}'
            rows[sid] = {'sample_id': sid, 'task': 'dr_grading', 'split': 'test', 'dataset': 'DDR',
                         'labels': {'dr_grade': grade}, 'label_mask': {'dr_grade': True},
                         'content_group_id': group, 'leakage_group_id': group, 'patient_id': None,
                         'image_sha256': [hashlib.sha256(sid.encode()).hexdigest()],
                         'content_group_evaluation_weight': 1.0}
            predicted = {1: 2, 3: 0, 4: 1}.get(i, grade)
            predictions.append({'sample_id': sid, 'dataset': 'DDR', 'target': grade,
                                'predicted_class': predicted, 'probabilities': np.eye(5)[predicted].tolist(),
                                'content_group_id': group, 'evaluation_weight': 1.0})
        self.rows = rows
        dump_lines(self.manifest_path, rows.values())
        dump(self.index_path, {'format_version': 1, 'all_checks_passed': True,
             'record_bytes_preserved': True, 'base_protocols': ['strict_cross_domain'],
             'combinations': [{'splits': {'test': {'path': relative, 'records': EXPECTED_COUNT,
                 'bytes': self.manifest_path.stat().st_size, 'sha256': sha256_file(self.manifest_path)}}}]})
        cfg = {'task': 'dr_grading', 'kind': 'multiclass', 'labels': ['dr_grade'], 'num_classes': 5,
               'model': {'name': 'fixture', 'img_size': 336}, 'training': {'epochs': 30},
               'data': {'datasets': ['DDR'], 'source_isolated': True, 'protocol': 'strict_cross_domain',
                        'isolation_manifest_sha256': sha256_file(self.index_path)}}
        self.pred_path = self.run / 'predictions_test.jsonl'
        self.meta_path = self.pred_path.with_suffix('.meta.json')
        self.eval_path = self.run / 'evaluation_test.json'
        dump_lines(self.pred_path, predictions)
        self.meta = {'format_version': 1, 'split': 'test', 'checkpoint': '/run/best.pt',
                     'checkpoint_completed_epoch': 24, 'sample_count': EXPECTED_COUNT,
                     'expected_dataset_count': EXPECTED_COUNT, 'unique_sample_ids': EXPECTED_COUNT,
                     'source_counts': {'DDR': EXPECTED_COUNT}, 'all_targets_known': True,
                     'complete_dataset_evaluated': True, 'target_label': 'dr_grade', 'class_order': list(range(5)),
                     'ordering': 'sample_id_lexicographic', 'manifest_sha256': sha256_file(self.manifest_path),
                     'jsonl_sha256': sha256_file(self.pred_path)}
        self.metrics = summarize_predictions(sorted(rows), rows, {row['sample_id']: row for row in predictions})
        self.evaluation = {'split': 'test', 'checkpoint': '/run/best.pt', 'config': cfg,
                           'evaluated_dataset_count': EXPECTED_COUNT,
                           'metrics': {**deepcopy(self.metrics), 'by_dataset': {'DDR': deepcopy(self.metrics)}}}
        dump(self.meta_path, self.meta)
        dump(self.eval_path, self.evaluation)

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, **kwargs):
        return build_report(self.run, self.data, replicates=21, seed=3, **kwargs)

    def test_one_model_full_ddr_metrics_intervals_and_unchanged_inputs(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        report = self.build()
        self.assertEqual(DEFAULT_REPLICATES, 5000)
        self.assertEqual(report['n'], 3898)
        self.assertEqual(report['metrics']['confusion_matrix'][3][0], 1)
        self.assertEqual(report['metrics']['confusion_matrix'][4][1], 1)
        self.assertAlmostEqual(report['metrics']['accuracy'], 3895 / 3898)
        self.assertEqual(report['metrics']['recall_by_grade'][3], 778 / 779)
        error = report['prioritized_errors']['true_3_or_4_to_0_or_1']
        self.assertEqual((error['count'], error['true_grade_denominator']), (2, 1558))
        self.assertEqual(error['sample_ids'], ['DDR:test:00003', 'DDR:test:00004'])
        self.assertEqual(report['bootstrap_design']['replicates'], 21)
        self.assertEqual(report['bootstrap_design']['grouping']['group_count'], 1949)
        self.assertEqual(report['qwk_bootstrap']['valid_replicates'], 21)
        self.assertEqual(report['uncertainty']['accuracy']['valid_replicates'], 21)
        self.assertNotIn('reference', report)
        self.assertNotIn('IDRiD', json.dumps(report))
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
        json.dumps(report, allow_nan=False)

    def test_metadata_sha_and_audited_manifest_changes_are_rejected(self):
        for changes in ({'jsonl_sha256': 'f' * 64}, {'manifest_sha256': 'f' * 64},
                        {'sample_count': 354}, {'complete_dataset_evaluated': False},
                        {'class_order': [4, 3, 2, 1, 0]}, {'checkpoint': '/another/best.pt'}):
            with self.subTest(changes=changes):
                dump(self.meta_path, {**self.meta, **changes})
                with self.assertRaises(ValueError):
                    self.build()
        dump(self.meta_path, self.meta)
        self.manifest_path.write_bytes(self.manifest_path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'audited isolation index'):
            self.build()

    def test_prediction_ids_targets_groups_and_argmax_are_strict_even_after_rehash(self):
        original = [json.loads(line) for line in self.pred_path.read_text().splitlines()]
        for changes in ({'sample_id': 'not_in_manifest'}, {'target': 4}, {'content_group_id': 'wrong'},
                        {'predicted_class': 1}, {'dataset': 'APTOS2019'}):
            rows = deepcopy(original)
            rows[0].update(changes)
            dump_lines(self.pred_path, rows)
            dump(self.meta_path, {**self.meta, 'jsonl_sha256': sha256_file(self.pred_path)})
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.build()

    def test_wrong_report_metrics_or_mixed_training_source_are_rejected(self):
        altered = deepcopy(self.evaluation)
        altered['metrics']['accuracy'] = 0.5
        dump(self.eval_path, altered)
        with self.assertRaisesRegex(ValueError, 'Reported metric differs'):
            self.build()
        altered = deepcopy(self.evaluation)
        altered['config']['data']['datasets'] = ['DDR', 'APTOS2019']
        dump(self.eval_path, altered)
        with self.assertRaisesRegex(ValueError, 'isolated DDR'):
            self.build()

    def test_optional_reference_is_paired_with_same_ids_and_zero_delta_if_identical(self):
        reference = self.root / 'reference'
        reference.mkdir()
        for path in (self.pred_path, self.meta_path, self.eval_path):
            (reference / path.name).write_bytes(path.read_bytes())
        report = self.build(reference_predictions=reference / self.pred_path.name)
        self.assertEqual(report['paired_qwk_delta']['point_delta_current_minus_reference'], 0.0)
        self.assertEqual(report['paired_qwk_delta']['ci95_percentile'], [0.0, 0.0])
        self.assertEqual(report['reference']['metrics']['qwk'], report['metrics']['qwk'])
        changed = deepcopy(self.meta)
        changed['manifest_sha256'] = 'e' * 64
        dump(reference / self.meta_path.name, changed)
        with self.assertRaises(ValueError):
            self.build(reference_predictions=reference / self.pred_path.name)

    def test_transitive_groups_zero_error_intervals_and_undefined_qwk_remain_explicit(self):
        rows = {'a': {'dataset': 'DDR', 'labels': {'dr_grade': 3}, 'leakage_group_id': 'one'},
                'b': {'dataset': 'DDR', 'labels': {'dr_grade': 3}, 'leakage_group_id': 'one', 'content_group_id': 'two'},
                'c': {'dataset': 'DDR', 'labels': {'dr_grade': 3}, 'content_group_id': 'two'}}
        predictions = {sid: {'predicted_class': 3} for sid in rows}
        extra = supplementary_bootstrap(rows, {'current': predictions, 'reference': predictions}, 5000, 3)
        self.assertEqual(extra['models']['current']['accuracy']['ci95_percentile'], [1.0, 1.0])
        self.assertEqual(extra['models']['current']['recall_grade_0']['valid_replicates'], 0)
        self.assertIsNone(extra['paired_qwk_delta']['point_delta_current_minus_reference'])
        self.assertIsNone(extra['paired_qwk_delta']['ci95_percentile'])
        self.assertEqual(extra['paired_qwk_delta']['undefined_replicates'], 5000)


if __name__ == '__main__':
    unittest.main()
