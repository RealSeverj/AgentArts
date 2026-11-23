"""Fresh training rejects source/task contamination and mixed-system holdouts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from scripts.dataset_routing import (config_for_split, expected_source, manifest_path,
                                    validate_isolated_config, validate_row,
                                    validate_suite_configs, validate_training_route)
from scripts.isolate_training_datasets import build_isolation, SPLITS

PROJECT = Path(__file__).resolve().parents[1]


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dataset_routing_")
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        old = self.data / "prepared/tasks/dr_grading"
        old.mkdir(parents=True)
        for split in SPLITS:
            rows = []
            source = "IDRiD" if split == "external_test" else "DDR"
            for grade in range(5):
                relative = f"{source}/{split}_{grade}.png"
                image = self.data / relative
                image.parent.mkdir(exist_ok=True)
                Image.new("RGB", (18, 20), (grade * 30, 70, 90)).save(image)
                identifier = f"{source}:{split}:{grade}"
                rows.append({"sample_id": identifier, "task": "dr_grading", "dataset": source, "split": split,
                             "image_path": relative, "image_paths": [relative], "labels": {"dr_grade": grade},
                             "label_mask": {"dr_grade": True}, "annotations": {}, "patient_id": identifier,
                             "content_group_id": identifier, "leakage_group_id": identifier,
                             "image_sha256": [hashlib.sha256(identifier.encode()).hexdigest()]})
            (old / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.isolated = self.data / "prepared/source_isolated"
        audit = build_isolation(self.data, self.isolated, ("strict_cross_domain",))
        self.assertTrue(audit["all_checks_passed"])
        self.config = {"seed": 7, "task": "dr_grading", "kind": "multiclass", "labels": ["dr_grade"], "num_classes": 5,
                       "data": {"protocol": "strict_cross_domain", "datasets": ["DDR"], "source_isolated": True,
                                "isolation_manifest_sha256": hashlib.sha256((self.isolated / "isolation_manifest.json").read_bytes()).hexdigest()},
                       "model": {"name": "tiny_cnn", "pretrained": False, "img_size": 16},
                       "training": {"epochs": 1, "batch_size": 2, "grad_accum_steps": 1, "workers": 0,
                                    "amp": "none", "best_metric": "loss", "best_mode": "min"}}

    def tearDown(self):
        self.temp.cleanup()

    def test_route_and_explicit_external_leave_training_source_unchanged(self):
        report = validate_training_route(self.config, self.data)
        self.assertEqual(report["splits"]["train"]["records"], 5)
        cfg = deepcopy(self.config)
        cfg["data"]["external_dataset"] = "IDRiD"
        self.assertEqual(expected_source(cfg, "train"), "DDR")
        self.assertEqual(expected_source(cfg, "external_test"), "IDRiD")
        self.assertEqual(config_for_split(cfg, "external_test")["data"]["datasets"], ["IDRiD"])
        self.assertEqual(cfg["data"]["datasets"], ["DDR"])
        self.assertIn("/IDRiD/external_test.jsonl", manifest_path(cfg, self.data, "external_test").as_posix())

    def test_legacy_multi_source_wrong_task_and_quality_label_are_rejected(self):
        legacy = deepcopy(self.config)
        legacy["data"].pop("source_isolated")
        with self.assertRaisesRegex(ValueError, "New training"):
            validate_training_route(legacy, self.data)
        cfg = deepcopy(self.config)
        cfg["data"]["datasets"] = ["DDR", "APTOS2019"]
        with self.assertRaises(ValueError):
            validate_isolated_config(cfg)
        cfg = deepcopy(self.config)
        cfg["labels"] = ["quality_grade"]
        with self.assertRaises(ValueError):
            validate_isolated_config(cfg)
        cfg = deepcopy(self.config)
        cfg["data"]["datasets"] = ["EyeQ"]
        with self.assertRaises(ValueError):
            validate_isolated_config(cfg)

    def test_changed_manifest_index_and_stale_mixed_weights_are_rejected(self):
        cfg = deepcopy(self.config)
        cfg["loss"] = {"training_label_counts": {"dr_grade": {"0": 19480}}}
        with self.assertRaisesRegex(ValueError, "another source"):
            validate_training_route(cfg, self.data)
        path = manifest_path(self.config, self.data, "train")
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "audited isolation index"):
            validate_training_route(self.config, self.data)

    def test_contamination_is_an_error_not_a_silent_filter(self):
        row = {"sample_id": "bad", "task": "dr_grading", "dataset": "APTOS2019", "split": "train"}
        with self.assertRaisesRegex(ValueError, "contamination"):
            validate_row(self.config, row, "train")
        row["dataset"], row["task"] = "DDR", "image_quality"
        with self.assertRaises(ValueError):
            validate_row(self.config, row, "train")

    def test_system_protocol_holdout_and_global_identity_mixing_are_rejected(self):
        cfg = deepcopy(self.config)
        cfg["data"]["protocol"] = "within_source"
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            validate_suite_configs([self.config, cfg])
        cfg = deepcopy(self.config)
        cfg["data"]["isolation_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "global split identity"):
            validate_suite_configs([self.config, cfg])
        dr = deepcopy(self.config)
        dr["data"]["external_dataset"] = "IDRiD"
        lesion = deepcopy(self.config)
        lesion.update(task="lesion_segmentation", kind="segmentation", labels=["EX", "HE", "MA", "SE"])
        lesion["data"]["datasets"] = ["IDRiD"]
        with self.assertRaisesRegex(ValueError, "already used to train"):
            validate_suite_configs([dr, lesion])

    def test_new_cache_can_be_built_before_its_configured_path_exists(self):
        from scripts.build_classification_cache import build_cache
        cfg = deepcopy(self.config)
        cfg["augmentation"] = {"resize_before_augmentation": True}
        cache = self.root / "source_cache"
        cfg["data"]["image_cache_dir"] = str(cache)
        result = build_cache(cfg, self.data, cache, workers=1)
        self.assertEqual(result["expected_unique_images"], 10)
        self.assertEqual(validate_training_route(cfg, self.data)["source"], "DDR")

    def test_real_cpu_training_and_explicit_external_evaluation(self):
        cfg = deepcopy(self.config)
        cfg["data"]["external_dataset"] = "IDRiD"
        path = self.root / "config.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        output = self.root / "run"
        base = [sys.executable, "-m", "training.train", "--config", str(path), "--data-root", str(self.data),
                "--output", str(output), "--device", "cpu", "--workers", "0"]
        train = subprocess.run(base, cwd=PROJECT, text=True, capture_output=True, timeout=60)
        self.assertEqual(train.returncode, 0, train.stdout + train.stderr)
        runtime = json.loads((output / "runtime.json").read_text())
        self.assertEqual((runtime["train_records"], runtime["val_records"]), (5, 5))
        evaluation = subprocess.run(base + ["--eval-only", "--split", "external_test", "--checkpoint", str(output / "best.pt"), "--save-predictions"],
                                    cwd=PROJECT, text=True, capture_output=True, timeout=60)
        self.assertEqual(evaluation.returncode, 0, evaluation.stdout + evaluation.stderr)
        report = json.loads((output / "evaluation_external_test.json").read_text())
        self.assertEqual(report["metrics"]["n"], 5)
        self.assertEqual(set(report["metrics"]["by_dataset"]), {"IDRiD"})


if __name__ == "__main__":
    unittest.main()
