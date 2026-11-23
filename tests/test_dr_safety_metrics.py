import unittest
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from scripts.analyze_dr_evaluation import qwk
from scripts.report_dr_safety_metrics import (bootstrap_metrics, connected_groups,
                                              prioritized_errors, cohort_specification,
                                              fixed_cohort_manifest, actual_stage, select_summary_model)


class SafetyMetricTests(unittest.TestCase):
    def test_counts_use_true_grade_denominators(self):
        matrix = np.zeros((5, 5), dtype=int)
        matrix[3, 0], matrix[3, 1], matrix[3, 3] = 1, 2, 7
        matrix[4, 1], matrix[4, 4] = 2, 3
        result = prioritized_errors(matrix)
        self.assertEqual(result['true_3_to_0_or_1']['count'], 3)
        self.assertEqual(result['true_3_to_0_or_1']['true_grade_denominator'], 10)
        self.assertEqual(result['true_4_to_0_or_1']['rate'], .4)
        self.assertEqual(result['true_3_or_4_to_0_or_1']['rate'], 5 / 15)

    def test_no_severe_cases_means_undefined_rate_not_fake_zero(self):
        matrix = np.diag([5, 2, 3, 0, 0])
        for item in prioritized_errors(matrix).values():
            self.assertEqual(item['count'], 0)
            self.assertEqual(item['true_grade_denominator'], 0)
            self.assertIsNone(item['rate'])
        manifest = {str(i): {'sample_id': str(i), 'dataset': 'fixture', 'labels': {'dr_grade': i % 3}}
                    for i in range(10)}
        predictions = {s: {'predicted_class': r['labels']['dr_grade']} for s, r in manifest.items()}
        result = bootstrap_metrics(manifest, {'model': predictions}, 50, 7)
        rate = result['models']['model']['true_3_or_4_to_0_or_1']
        self.assertIsNone(rate['ci95_percentile'])
        self.assertEqual(rate['undefined_replicates'], 50)

    def test_qwk_perfect_prediction_and_bootstrap_reproducibility(self):
        self.assertEqual(qwk(np.diag([2, 3, 4, 5, 6])), 1.)
        self.assertIsNone(qwk(np.diag([5, 0, 0, 0, 0])))
        manifest = {str(i): {'sample_id': str(i), 'dataset': 'fixture', 'labels': {'dr_grade': i % 5}}
                    for i in range(20)}
        predictions = {s: {'predicted_class': r['labels']['dr_grade']} for s, r in manifest.items()}
        first = bootstrap_metrics(manifest, {'model': predictions}, 100, 19)
        self.assertEqual(first, bootstrap_metrics(manifest, {'model': predictions}, 100, 19))
        self.assertEqual(first['models']['model']['qwk']['ci95_percentile'], [1., 1.])

    def test_any_shared_known_group_link_stays_in_one_component(self):
        rows = {'a': {'sample_id': 'a', 'dataset': 'fixture', 'leakage_group_id': 'L'},
                'b': {'sample_id': 'b', 'dataset': 'fixture', 'leakage_group_id': 'L', 'content_group_id': 'C'},
                'c': {'sample_id': 'c', 'dataset': 'fixture', 'content_group_id': 'C'}}
        groups, info = connected_groups(rows)
        self.assertEqual(groups, [['a', 'b', 'c']])
        self.assertEqual(info['group_count'], 1)
        self.assertEqual(info['rows_without_patient_id'], 3)

    def test_explicit_ddr_count_does_not_reuse_aptos_count(self):
        self.assertEqual(cohort_specification()['test'], ('APTOS2019', 354))
        self.assertEqual(cohort_specification('DDR'), {'test': ('DDR', 3898), 'external_test': ('IDRiD', 506)})
        with self.assertRaises(ValueError):
            cohort_specification('EyeQ')

    def test_ddr_manifest_source_and_count_are_both_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            isolation = root / 'prepared/source_isolated'
            relative = 'strict_cross_domain/tasks/dr_grading/DDR/test.jsonl'
            path = isolation / relative
            path.parent.mkdir(parents=True)
            config = {'task': 'dr_grading', 'kind': 'multiclass', 'labels': ['dr_grade'], 'num_classes': 5,
                      'data': {'source_isolated': True, 'datasets': ['DDR'], 'external_dataset': 'IDRiD', 'protocol': 'strict_cross_domain'}}
            def write(source):
                rows = [{'sample_id': str(i), 'task': 'dr_grading', 'split': 'test', 'dataset': source,
                         'labels': {'dr_grade': i % 5}, 'label_mask': {'dr_grade': True}} for i in range(3898)]
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
                index = {'all_checks_passed': True, 'combinations': [{'splits': {'test': {'path': relative, 'records': len(rows),
                                                                                       'bytes': path.stat().st_size,
                                                                                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}}]}
                index_path = isolation / 'isolation_manifest.json'
                index_path.write_text(json.dumps(index), encoding='utf-8')
                config['data']['isolation_manifest_sha256'] = hashlib.sha256(index_path.read_bytes()).hexdigest()
            write('DDR')
            _, rows = fixed_cohort_manifest(root, config, 'test', 'DDR', 3898)
            self.assertEqual(len(rows), 3898)
            with self.assertRaises(ValueError):
                fixed_cohort_manifest(root, config, 'test', 'DDR', 354)
            with self.assertRaisesRegex(ValueError, 'another source'):
                fixed_cohort_manifest(root, config, 'test', 'APTOS2019', 354)
            write('APTOS2019')
            with self.assertRaisesRegex(ValueError, 'another dataset/source'):
                fixed_cohort_manifest(root, config, 'test', 'DDR', 3898)

    def test_model_routing_and_actual_stage_do_not_assume_epoch_eleven(self):
        cohort = {'models': {'ddr_transfer_best': {'evaluation_name': 'best', 'checkpoint_completed_epoch': 17},
                             'legacy_baseline': {'checkpoint_completed_epoch': 1}}}
        key, model = select_summary_model(cohort, 'best')
        self.assertEqual(key, 'ddr_transfer_best')
        self.assertEqual(model['checkpoint_completed_epoch'], 17)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / 'history.jsonl'
            history.write_text(json.dumps({'epoch': 17, 'stage': 'last_4_blocks'}) + '\n')
            config = {'training': {'freeze_backbone_epochs': 5, 'unfreeze_last_n_blocks': 4, 'full_finetune_epoch': 30}}
            stage, origin = actual_stage(root, 17, config, {'stage': 'last_4_blocks'})
            self.assertEqual(stage, 'last_4_blocks')
            self.assertEqual(origin, 'checkpoint_metadata/history')
            with self.assertRaises(ValueError):
                actual_stage(root, 17, config, {'stage': 'head_only'})


if __name__ == '__main__':
    unittest.main()
