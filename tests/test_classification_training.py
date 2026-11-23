"""Regression tests for missing supervision, paired eyes and metric semantics."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from training import classification as adapter
from training.metrics import binary_metrics, classification_metrics, multiclass_metrics


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_unknown_targets_do_not_change_loss_or_gradient(self):
        config = {"kind": "multilabel", "labels": ["known", "missing"],
                  "model": {"name": "tiny_cnn"}, "loss": {"pos_weight": [2.0, 3.0]}}
        model = adapter.build_model(config)
        batch = {"images": torch.randn(2, 3, 16, 16),
                 "targets": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
                 "mask": torch.tensor([[True, False], [True, False]]),
                 "sample_ids": ["a", "b"], "datasets": ["fixture", "fixture"],
                 "content_group_ids": ["a", "b"], "evaluation_weights": np.ones(2)}
        first, output = adapter.step(model, batch, torch.device("cpu"), config, True)
        first.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        batch["targets"][:, 1] = 1.0
        second, _ = adapter.step(model, batch, torch.device("cpu"), config, True)
        second.backward()
        self.assertEqual(first.item(), second.item())
        self.assertEqual(output["loss_weight"], 2)
        for before, parameter in zip(gradients, model.parameters()):
            torch.testing.assert_close(before, parameter.grad, rtol=0, atol=0)

    def test_paired_view_prediction_is_one_patient_prediction(self):
        config = {"kind": "multilabel", "labels": ["a", "b"], "model": {"name": "tiny_cnn"}}
        model = adapter.build_model(config)
        eyes = torch.randn(3, 2, 3, 16, 16)
        paired = model(eyes)
        expected = (model(eyes[:, 0]) + model(eyes[:, 1])) / 2
        self.assertEqual(tuple(paired.shape), (3, 2))
        torch.testing.assert_close(paired, expected)

    def test_ddp_uses_global_known_count_only_during_training(self):
        config = {"kind": "multilabel", "labels": ["target"], "model": {"name": "tiny_cnn"}}
        model = adapter.build_model(config)
        batch = {"images": torch.randn(2, 3, 16, 16), "targets": torch.ones(2, 1),
                 "mask": torch.ones(2, 1, dtype=torch.bool), "sample_ids": ["a", "b"],
                 "datasets": ["fixture", "fixture"], "content_group_ids": ["a", "b"],
                 "evaluation_weights": np.ones(2)}
        reference, _ = adapter.step(model, batch, torch.device("cpu"), config, False)
        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.distributed.all_reduce", side_effect=lambda value, **kwargs: value.fill_(3)) as collective:
            distributed, _ = adapter.step(model, batch, torch.device("cpu"), config, True)
            torch.testing.assert_close(distributed, reference * 4 / 3)
            collective.assert_called_once()
            collective.reset_mock()
            validation, _ = adapter.step(model, batch, torch.device("cpu"), config, False)
            torch.testing.assert_close(validation, reference)
            collective.assert_not_called()

    def test_source_filter_and_unknown_supervision(self):
        with tempfile.TemporaryDirectory(prefix="classification_dataset_") as temporary:
            root = Path(temporary)
            Image.new("RGB", (8, 8), "red").save(root / "a.png")
            rows = []
            for identifier, source, value in [("a", "allowed", 1), ("b", "other", 0), ("c", "allowed", None)]:
                rows.append({"sample_id": identifier, "task": "fixture", "dataset": source, "split": "train",
                             "image_path": "a.png", "image_paths": ["a.png"], "labels": {"target": value},
                             "label_mask": {"target": value is not None}, "annotations": {}})
            manifest = root / "train.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            config = {"task": "fixture", "kind": "multilabel", "labels": ["target"],
                      "model": {"name": "tiny_cnn", "img_size": 16}, "data": {"datasets": ["allowed"]}}
            dataset = adapter.build_dataset(manifest, root, config, True)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["sample_id"], "a")
            config["labels"] = ["unavailable"]
            with self.assertRaisesRegex(ValueError, "No training supervision"):
                adapter.build_dataset(manifest, root, config, True)


class MetricTests(unittest.TestCase):
    def test_weighted_auc_and_average_precision_with_tied_scores(self):
        rng = np.random.default_rng(13)
        for size in (2, 10, 100):
            for _ in range(5):
                y, p = rng.integers(0, 2, size), np.round(rng.random(size), 1)
                weights = rng.uniform(0.1, 2, size)
                result = binary_metrics(y, p, weights)
                positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
                if len(positive) and len(negative):
                    # Independent pairwise definition, including half credit
                    # for tied positive/negative scores.
                    auc = sum(weights[i] * weights[j] * (float(p[i] > p[j]) + 0.5 * float(p[i] == p[j]))
                              for i in positive for j in negative) / (weights[positive].sum() * weights[negative].sum())
                    self.assertAlmostEqual(result["auroc"], auc)
                else:
                    self.assertIsNone(result["auroc"])
                if len(positive):
                    ap = sum(weights[i] / weights[positive].sum() *
                             weights[(p >= p[i]) & (y == 1)].sum() / weights[p >= p[i]].sum() for i in positive)
                    self.assertAlmostEqual(result["auprc"], ap)
                else:
                    self.assertIsNone(result["auprc"])

    def test_qwk_perfect_reversed_and_degenerate(self):
        self.assertEqual(multiclass_metrics([0, 1, 2], np.eye(3), 3)["qwk"], 1.0)
        reversed_score = np.array([[0, 0, 1], [1, 0, 0]])
        self.assertEqual(multiclass_metrics([0, 2], reversed_score, 3)["qwk"], -1.0)
        self.assertIsNone(multiclass_metrics([0, 0], np.array([[1, 0], [1, 0]]), 2)["qwk"])

    def test_ddp_padding_deduplicated_and_content_weights_preserved(self):
        output = {"targets": np.array([[1], [0]]), "mask": np.array([[True], [True]]),
                  "probabilities": np.array([[0.8], [0.9]]), "sample_ids": ["a", "b"],
                  "datasets": ["fixture", "fixture"], "content_group_ids": ["a", "b"],
                  "evaluation_weights": np.array([1.0, 0.25])}
        result = classification_metrics([output, output], {"kind": "multilabel", "labels": ["target"]})
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["per_label"]["target"]["weight_sum"], 2.0)
        self.assertEqual(result["content_weighted"]["per_label"]["target"]["weight_sum"], 1.25)
        self.assertEqual(result["unique_content_groups"], 2)


if __name__ == "__main__":
    unittest.main()
