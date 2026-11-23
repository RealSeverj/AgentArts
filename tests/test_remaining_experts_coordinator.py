"""Pure fixtures for the cloud suite gates; never load weights or use a GPU."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_remaining_experts_20261114 as suite


def configuration(spec):
    return {
        "task": spec[1], "kind": spec[2], "labels": spec[9],
        "data": {"source_isolated": True, "datasets": [spec[3]]},
        "model": {"name": spec[10], "img_size": spec[11]},
        "training": {"epochs": spec[4], "batch_size": 8 if spec[2] == "segmentation" else 16,
                     "grad_accum_steps": 1, "best_metric": spec[8], "best_mode": "max", "amp": "fp16",
                     "freeze_backbone_epochs": 0 if spec[2] == "segmentation" else 5,
                     "unfreeze_last_n_blocks": 0 if spec[2] == "segmentation" else 4,
                     "full_finetune_epoch": 0 if spec[2] == "segmentation" else 30},
    }


def history(config, spec, best_epoch=7):
    result, running_best = [], -float("inf")
    for epoch in range(1, spec[4] + 1):
        score = 0.9 if epoch == best_epoch else 0.5
        improved = score > running_best
        running_best = max(running_best, score)
        result.append({"epoch": epoch, "stage": suite.expected_stage(config, epoch - 1),
                       "val": {"samples" if spec[2] == "segmentation" else "n": spec[6], spec[8]: score},
                       "best_metric_name": spec[8], "best_metric": running_best, "improved": improved})
    return result


class SuiteGates(unittest.TestCase):
    def write_history(self, run, records):
        (run / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    def test_prepared_variants_and_wrong_source_or_schedule(self):
        for spec in suite.SPECS:
            config = configuration(spec)
            suite.validate_config(config, spec)
            bad = deepcopy(config)
            bad["data"]["datasets"] = ["DDR"]
            with self.assertRaisesRegex(ValueError, "isolated source"):
                suite.validate_config(bad, spec)
            if spec[2] != "segmentation":
                bad = deepcopy(config)
                bad["training"]["full_finetune_epoch"] = 10
                with self.assertRaisesRegex(ValueError, "schedule"):
                    suite.validate_config(bad, spec)

    def test_correct_best_from_complete_30_and_50_epochs(self):
        for spec in suite.SPECS:
            with tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary)
                config = configuration(spec)
                self.write_history(run, history(config, spec))
                records, best, score = suite.validate_history(run, config, spec)
                self.assertEqual(len(records), spec[4])
                self.assertEqual(best["epoch"], 7)
                self.assertEqual(score, 0.9)

    def test_reject_missing_or_duplicate_epochs(self):
        spec = suite.SPECS[0]
        config = configuration(spec)
        records = history(config, spec)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            self.write_history(run, records[:-1])
            with self.assertRaisesRegex(ValueError, "incomplete or duplicated"):
                suite.validate_history(run, config, spec)
            records[8]["epoch"] = records[7]["epoch"]
            self.write_history(run, records)
            with self.assertRaisesRegex(ValueError, "non-contiguous"):
                suite.validate_history(run, config, spec)

    def test_reject_incomplete_validation(self):
        spec = suite.SPECS[0]
        config = configuration(spec)
        records = history(config, spec)
        records[9]["val"]["n"] -= 1
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            self.write_history(run, records)
            with self.assertRaisesRegex(ValueError, "complete cohort"):
                suite.validate_history(run, config, spec)

    def test_reject_wrong_stage_and_best_tracking(self):
        spec = suite.SPECS[0]
        config = configuration(spec)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            records = history(config, spec)
            records[5]["stage"] = "full"
            self.write_history(run, records)
            with self.assertRaisesRegex(ValueError, "Wrong trainable stage"):
                suite.validate_history(run, config, spec)
            records = history(config, spec)
            records[-1]["best_metric"] = 0.4
            self.write_history(run, records)
            with self.assertRaisesRegex(ValueError, "running best"):
                suite.validate_history(run, config, spec)

    def test_prepare_rejects_wrong_source_before_route_and_binds_manifest_hashes(self):
        spec = list(suite.SPECS[0])
        spec[5:8] = [3, 2, 2]
        spec = tuple(spec)
        config = configuration(spec)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = root / "config.json"
            bad = deepcopy(config)
            bad["data"]["datasets"] = ["RFMiD"]
            cfg.write_text(json.dumps(bad), encoding="utf-8")
            with patch("scripts.dataset_routing.validate_training_route") as route:
                with self.assertRaisesRegex(ValueError, "isolated source"):
                    suite.prepare_config(cfg, spec, root)
                route.assert_not_called()
            cfg.write_text(json.dumps(config), encoding="utf-8")
            paths = {}
            for split, n in zip(("train", "val", "test"), spec[5:8]):
                paths[split] = root / (split + ".jsonl")
                paths[split].write_text("".join(json.dumps({"sample_id": f"{split}_{i}"}) + "\n"
                                                for i in range(n)), encoding="utf-8")
            with patch("scripts.dataset_routing.validate_training_route") as route, \
                 patch("scripts.dataset_routing.manifest_path", side_effect=lambda c, r, s: paths[s]), \
                 patch("scripts.dataset_routing.validate_row") as validate_row:
                observed_config, checks = suite.prepare_config(cfg, spec, root)
                self.assertEqual(observed_config, config)
                route.assert_called_once()
                self.assertEqual(validate_row.call_count, 7)
                self.assertEqual(checks["test"]["records"], 2)
                self.assertEqual(checks["test"]["sha256"], suite.sha256(paths["test"]))


if __name__ == "__main__":
    unittest.main()
