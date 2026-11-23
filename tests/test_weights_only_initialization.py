"""Pinned model-only warm starts must never act like a training resume."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import json

import torch
from torch import nn

from training.train import (apply_weights_only_initialization, build_optimizer,
                            should_apply_weights_only_initialization, validate_initialization_config)
from training import train as runner


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, images):
        return self.head(self.backbone(images))


class WeightsOnlyInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix='weights_only_')
        self.path = Path(self.temporary.name) / 'parent.pt'
        self.config = {'task': 'dr_grading', 'kind': 'multiclass', 'labels': ['dr_grade'], 'num_classes': 2,
                       'model': {'name': 'fixture', 'img_size': 336, 'dropout': 0.1},
                       'data': {'datasets': ['DDR']},
                       'training': {'lr': 1e-4, 'head_lr': 1e-3, 'backbone_lr': 1e-5}}
        self.parent_config = deepcopy(self.config)
        self.parent_config['data']['datasets'] = ['APTOS2019']
        self.parent = Classifier()
        optimizer = build_optimizer(self.parent, self.config['training'])
        self.parent(torch.ones(2, 3)).square().mean().backward()
        optimizer.step()
        self.payload = {'epoch': 4, 'model': self.parent.state_dict(), 'config': self.parent_config,
                        'optimizer': optimizer.state_dict(), 'scheduler': {'last_epoch': 5},
                        'scaler': {'scale': 123}, 'best_metric': 0.9,
                        'rng_states': [{'torch': torch.zeros_like(torch.get_rng_state())}]}
        self.write_payload()

    def tearDown(self):
        self.temporary.cleanup()

    def write_payload(self):
        torch.save(self.payload, self.path)
        self.config['initialization'] = {'mode': 'weights_only', 'checkpoint': str(self.path),
                                         'checkpoint_sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(),
                                         'parent_training_sources': ['APTOS2019'], 'evaluate_before_training': True}

    def test_only_weights_imported_new_optimizer_epoch_and_rng(self):
        child = Classifier()
        before_rng = torch.get_rng_state().clone()
        lineage = apply_weights_only_initialization(child, self.config)
        for a, b in zip(child.parameters(), self.parent.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        optimizer = build_optimizer(child, self.config['training'])
        self.assertEqual(len(optimizer.state), 0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(lineage['starting_epoch_zero_based'], 0)
        self.assertEqual(lineage['parent_completed_epoch'], 5)
        self.assertFalse(lineage['optimizer_state_imported'])
        self.assertFalse(lineage['parent_rng_restored'])
        self.assertEqual(lineage['supervised_training_source_history'], ['APTOS2019', 'DDR'])
        self.assertFalse(lineage['independent_source_only_model'])

    def test_hash_architecture_and_declared_source_mismatch_are_rejected(self):
        bad = deepcopy(self.config)
        bad['initialization']['checkpoint_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            apply_weights_only_initialization(Classifier(), bad)
        bad = deepcopy(self.config)
        bad['model']['dropout'] = 0.2  # Shape-compatible still violates full model identity.
        with self.assertRaisesRegex(ValueError, 'parent model differs'):
            apply_weights_only_initialization(Classifier(), bad)
        bad = deepcopy(self.config)
        bad['initialization']['parent_training_sources'] = ['Other']
        with self.assertRaisesRegex(ValueError, 'supervision history'):
            apply_weights_only_initialization(Classifier(), bad)
        self.payload['model']['head.weight'] = torch.zeros(3, 4)
        self.write_payload()
        with self.assertRaisesRegex(ValueError, 'tensor shapes'):
            apply_weights_only_initialization(Classifier(), self.config)

    def test_eval_and_resume_never_reapply_parent_even_when_file_missing(self):
        self.path.unlink()
        for kwargs in ({'resume': True}, {'eval_only': True}):
            child = Classifier()
            original = deepcopy(child.state_dict())
            self.assertFalse(should_apply_weights_only_initialization(self.config, **kwargs))
            with patch('training.train.torch.load', side_effect=AssertionError('parent must not load')):
                self.assertIsNone(apply_weights_only_initialization(child, self.config, **kwargs))
            for key, value in child.state_dict().items():
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        self.assertIsNone(validate_initialization_config({'training': {}}))
        self.assertIsNone(apply_weights_only_initialization(Classifier(), {'training': {}}))

    def test_fresh_entrypoint_runs_complete_validation_before_first_update(self):
        self.config['training'].update(epochs=1, batch_size=2, grad_accum_steps=1, workers=0,
                                       amp='none', best_metric='loss', best_mode='min')
        config_path = Path(self.temporary.name) / 'config.json'
        config_path.write_text(json.dumps(self.config), encoding='utf-8')
        output = Path(self.temporary.name) / 'run'
        data_root = Path(self.temporary.name) / 'data'
        data_root.mkdir()
        for split in ('train', 'val'):
            (data_root / f'{split}.jsonl').write_text('{}\n', encoding='utf-8')
        events, pretrained_flags = [], []

        class Fixture(torch.utils.data.Dataset):
            def __init__(self, n):
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, index):
                return {'images': torch.tensor([float(index), 1., 0.]), 'targets': torch.tensor(index % 2),
                        'sample_id': str(index)}

        def build(config, pretrained=True):
            pretrained_flags.append(pretrained)
            return Classifier()

        def collate(samples):
            return {'images': torch.stack([s['images'] for s in samples]),
                    'targets': torch.stack([s['targets'] for s in samples]),
                    'sample_ids': [s['sample_id'] for s in samples]}

        def step(model, batch, device, config, training):
            events.append('train' if training else 'val')
            loss = torch.nn.functional.cross_entropy(model(batch['images']), batch['targets'])
            return loss, {'loss_weight': len(batch['sample_ids']), 'sample_ids': batch['sample_ids']}

        adapter = SimpleNamespace(build_model=build,
                                  build_dataset=lambda path, root, config, training: Fixture(3 if training else 5),
                                  collate=collate, step=step,
                                  metrics=lambda outputs, config: {'n': sum(len(o['sample_ids']) for o in outputs)})
        original_import = runner.importlib.import_module
        argv = ['training.train', '--config', str(config_path), '--data-root', str(data_root),
                '--output', str(output), '--device', 'cpu', '--workers', '0', '--max-val-steps', '1']
        with patch('sys.argv', argv), \
             patch('scripts.dataset_routing.validate_training_route'), \
             patch('training.train.manifest_path', side_effect=lambda config, root, split: data_root / f'{split}.jsonl'), \
             patch('training.train.importlib.import_module', side_effect=lambda name, *args, **kwargs:
                   adapter if name == 'training.classification' else original_import(name, *args, **kwargs)):
            runner.main()
        self.assertEqual(pretrained_flags, [False])
        self.assertEqual(events.index('train'), 3)  # All five validation samples, not the debug limit of two.
        initial = json.loads((output / 'initialization_validation.json').read_text())
        self.assertTrue(initial['complete_dataset_evaluated'])
        self.assertEqual(initial['metrics']['n'], 5)
        self.assertEqual(initial['optimizer_updates'], 0)
        saved = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        self.assertEqual(saved['epoch'], 0)
        self.assertEqual(saved['initialization_lineage']['parent_completed_epoch'], 5)
        self.assertEqual(saved['scheduler']['last_epoch'], 1)


if __name__ == '__main__':
    unittest.main()
