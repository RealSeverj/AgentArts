"""Optimizer coverage, scheduler ratios and legacy checkpoint compatibility."""
import copy
import math
import unittest

import torch
from torch import nn

from training.train import build_optimizer, differential_learning_rates, optimizer_learning_rates


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 4))
        self.head = nn.Linear(4, 2)
        self.extra_scale = nn.Parameter(torch.ones(()))

    def forward(self, x):
        return self.head(self.backbone(x)) * self.extra_scale


class DifferentialLearningRateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_partition_covers_frozen_backbone_and_all_other_parameters(self):
        model = Classifier()
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        optimizer = build_optimizer(model, {'lr': 1e-4, 'backbone_lr': 1e-5, 'head_lr': 1e-3})
        seen = [id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {id(p) for p in model.parameters()})
        self.assertEqual({id(p) for p in optimizer.param_groups[0]['params']}, {id(p) for p in model.backbone.parameters()})
        self.assertIn(id(model.extra_scale), {id(p) for p in optimizer.param_groups[1]['params']})
        before = model.backbone[0].weight.detach().clone()
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(True)
        model(torch.randn(2, 3)).square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.backbone[0].weight))

    def test_lambda_scheduler_preserves_ratio_and_named_history(self):
        optimizer = build_optimizer(Classifier(), {'backbone_lr': 1e-5, 'head_lr': 1e-3})
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 / (epoch + 1))
        for _ in range(3):
            optimizer.step()
            scheduler.step()
            rates = optimizer_learning_rates(optimizer)
            self.assertAlmostEqual(rates['head'] / rates['backbone'], 100.0)
        self.assertAlmostEqual(rates['backbone'], 2.5e-6)
        self.assertAlmostEqual(rates['head'], 2.5e-4)

    def test_no_overrides_match_legacy_update_and_optimizer_state(self):
        model = Classifier()
        legacy_model = copy.deepcopy(model)
        options = {'lr': 1e-4, 'weight_decay': 0.05}
        optimizer = build_optimizer(model, options)
        legacy = torch.optim.AdamW(legacy_model.parameters(), **options)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0].keys(), legacy.param_groups[0].keys())
        x = torch.randn(2, 3)
        for candidate, opt in ((model, optimizer), (legacy_model, legacy)):
            candidate(x).square().mean().backward()
            opt.step()
        for first, second in zip(model.parameters(), legacy_model.parameters()):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        resumed = build_optimizer(Classifier(), options)
        resumed.load_state_dict(legacy.state_dict())
        self.assertEqual(len(resumed.state), len(legacy.state))
        self.assertEqual(optimizer_learning_rates(resumed), {'all': 1e-4})

    def test_positive_finite_validation_and_single_override_fallback(self):
        for key in ('backbone_lr', 'head_lr'):
            for value in (0, -1, float('nan'), float('inf'), True, None, '1e-3'):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, 'positive finite'):
                        differential_learning_rates({key: value})
        self.assertEqual(differential_learning_rates({'lr': 2e-4, 'head_lr': 1e-3}),
                         {'backbone_lr': 2e-4, 'head_lr': 1e-3})
        self.assertIsNone(differential_learning_rates({'lr': 2e-4}))


if __name__ == '__main__':
    unittest.main()
