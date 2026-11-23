"""Explicit evaluation input bypass and complete DR prediction exports."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from training.train import evaluation_runtime_config, save_dr_predictions

PROJECT = Path(__file__).resolve().parents[1]


def config(cache='/cache/train_val_only'):
    return {'task': 'dr_grading', 'kind': 'multiclass', 'labels': ['dr_grade'], 'num_classes': 5,
            'seed': 19, 'data': {'protocol': 'strict_cross_domain', 'image_cache_dir': cache},
            'model': {'name': 'tiny_cnn', 'pretrained': False, 'img_size': 16,
                      'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
            'augmentation': {'resize_before_augmentation': True, 'rotation_degrees': 10},
            'loss': {}, 'training': {'epochs': 30, 'batch_size': 2, 'grad_accum_steps': 4,
                                    'workers': 0, 'amp': 'none', 'lr': .0001}}


def outputs():
    return [{'sample_ids': ['b', 'a'], 'datasets': ['one', 'two'],
             'targets': np.asarray([2, 0]), 'mask': np.asarray([True, True]),
             'probabilities': np.asarray([[.05, .05, .8, .05, .05], [.8, .05, .05, .05, .05]]),
             'content_group_ids': ['group_b', 'group_a'], 'evaluation_weights': np.asarray([.5, 1.])}]


class SourceOverrideTests(unittest.TestCase):
    def test_override_is_deepcopy_and_only_removes_cache_path(self):
        saved = config()
        supplied = deepcopy(saved)
        runtime, recorded = evaluation_runtime_config(saved, supplied, True)
        expected = deepcopy(supplied)
        expected['data'].pop('image_cache_dir')
        self.assertEqual(runtime, expected)
        self.assertEqual(saved, config())
        self.assertEqual(supplied, saved)
        self.assertTrue(recorded['checkpoint_configuration_checked_before_override'])
        self.assertEqual(recorded['removed_image_cache_dir'], '/cache/train_val_only')
        unchanged, recorded = evaluation_runtime_config(saved, supplied, False)
        self.assertEqual(unchanged, supplied)
        self.assertFalse(recorded['eval_use_source_images'])

    def test_override_cannot_hide_protected_configuration_changes(self):
        for field, mutate in [
            ('data', lambda c: c['data'].pop('image_cache_dir')),
            ('data', lambda c: c['data'].update(image_cache_dir='/different/cache')),
            ('data', lambda c: c['data'].update(protocol='within_source')),
            ('task', lambda c: c.update(task='other')),
            ('kind', lambda c: c.update(kind='multilabel')),
            ('labels', lambda c: c.update(labels=['other'])),
            ('model', lambda c: c['model'].update(img_size=32)),
        ]:
            saved, supplied = config(), config()
            mutate(supplied)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                evaluation_runtime_config(saved, supplied, True)

    def test_evaluation_flags_rejected_for_training_before_config_is_read(self):
        for flag in ('--eval-use-source-images', '--save-predictions'):
            run = subprocess.run([sys.executable, '-m', 'training.train', '--config', 'not_a_real_config.json', flag],
                                 cwd=PROJECT, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 2)
            self.assertIn('require --eval-only', run.stderr)
            self.assertNotIn('Traceback', run.stderr)

    def test_prediction_export_rejects_partial_evaluation_at_cli(self):
        run = subprocess.run([sys.executable, '-m', 'training.train', '--config', 'not_a_real_config.json',
                              '--eval-only', '--save-predictions', '--max-val-steps', '1'],
                             cwd=PROJECT, capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 2)
        self.assertIn('complete evaluation dataset', run.stderr)


class PredictionExportTests(unittest.TestCase):
    def test_export_fields_count_probabilities_and_metadata_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions_test.jsonl'
            summary = save_dr_predictions(path, outputs(), config(), 2, {'split': 'test'})
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row['sample_id'] for row in rows], ['a', 'b'])
            self.assertEqual(rows[1]['target'], 2)
            self.assertEqual(rows[1]['predicted_class'], 2)
            self.assertEqual(rows[1]['evaluation_weight'], .5)
            self.assertEqual(set(rows[1]), {'sample_id', 'dataset', 'target', 'predicted_class', 'probabilities',
                                            'content_group_id', 'evaluation_weight'})
            meta = json.loads(Path(summary['metadata_path']).read_text())
            self.assertEqual(meta['sample_count'], 2)
            self.assertEqual(meta['unique_sample_ids'], 2)
            self.assertEqual(meta['jsonl_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(meta['complete_dataset_evaluated'])

    def test_duplicate_or_incomplete_export_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions.jsonl'
            repeated = outputs() + outputs()
            with self.assertRaisesRegex(ValueError, 'unique'):
                save_dr_predictions(path, repeated, config(), 4)
            with self.assertRaisesRegex(ValueError, 'count'):
                save_dr_predictions(path, outputs(), config(), 3)
            self.assertFalse(path.exists())

    def test_invalid_distributions_unknown_targets_and_shapes_rejected(self):
        mutations = [lambda b: b[0]['probabilities'].__setitem__((0, 0), np.nan),
                     lambda b: b[0]['probabilities'].__setitem__((0, 0), -.1),
                     lambda b: b[0]['probabilities'].__setitem__((0, 0), .2),
                     lambda b: b[0]['mask'].__setitem__(0, False),
                     lambda b: b[0]['targets'].__setitem__(0, 5),
                     lambda b: b[0].update(datasets=['one']),
                     lambda b: b[0]['evaluation_weights'].__setitem__(0, np.inf)]
        with tempfile.TemporaryDirectory() as tmp:
            for i, mutate in enumerate(mutations):
                batch = outputs()
                mutate(batch)
                with self.subTest(mutation=i), self.assertRaises(ValueError):
                    save_dr_predictions(Path(tmp) / 'p.jsonl', batch, config(), 2)

    def test_only_five_class_dr_export_is_supported(self):
        cfg = config()
        cfg['task'] = 'other'
        with self.assertRaises(ValueError):
            save_dr_predictions(Path('unused.jsonl'), outputs(), cfg, 2)


class RealEvaluationCliTests(unittest.TestCase):
    def test_source_image_evaluation_without_train_val_cache_and_full_predictions(self):
        from training.classification import build_model
        with tempfile.TemporaryDirectory(prefix='dr_eval_source_') as tmp:
            root = Path(tmp)
            data = root / 'data'
            data.mkdir()
            cfg = config(str(root / 'missing_train_val_only_cache'))
            task = data / 'prepared/tasks/dr_grading'
            task.mkdir(parents=True)
            rows = []
            for i in range(5):
                Image.new('RGB', (23 + i, 19), (i * 40, 70, 100)).save(data / f'{i}.png')
                rows.append({'sample_id': f'sample_{i}', 'task': 'dr_grading', 'dataset': 'fixture', 'split': 'test',
                             'image_paths': [f'{i}.png'], 'labels': {'dr_grade': i}, 'label_mask': {'dr_grade': True},
                             'annotations': {}, 'content_group_id': f'group_{i}', 'content_group_evaluation_weight': 1.})
            manifest = task / 'test.jsonl'
            manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(cfg), encoding='utf-8')
            checkpoint = root / 'best.pt'
            torch.save({'model': build_model(cfg, pretrained=False).state_dict(), 'config': deepcopy(cfg), 'epoch': 22}, checkpoint)
            output = root / 'evaluated'
            command = [sys.executable, '-m', 'training.train', '--config', str(config_path), '--data-root', str(data),
                       '--output', str(output), '--checkpoint', str(checkpoint), '--eval-only', '--split', 'test',
                       '--eval-use-source-images', '--save-predictions', '--device', 'cpu', '--workers', '0']
            run = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, msg=run.stderr)
            report = json.loads((output / 'evaluation_test.json').read_text())
            self.assertEqual(report['config']['data']['image_cache_dir'], cfg['data']['image_cache_dir'])
            self.assertTrue(report['runtime_overrides']['eval_use_source_images'])
            self.assertEqual(report['evaluated_dataset_count'], 5)
            exported = [json.loads(line) for line in (output / 'predictions_test.jsonl').read_text().splitlines()]
            self.assertEqual(len(exported), 5)
            self.assertEqual(len({row['sample_id'] for row in exported}), 5)
            self.assertEqual([row['target'] for row in exported], list(range(5)))
            meta = json.loads((output / 'predictions_test.meta.json').read_text())
            self.assertEqual(meta['checkpoint_completed_epoch'], 23)
            self.assertEqual(meta['manifest_sha256'], hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(json.loads(config_path.read_text()), cfg)


if __name__ == '__main__':
    unittest.main()
