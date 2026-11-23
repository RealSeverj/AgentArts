"""Source-only weighting, protocol separation and safe config publication."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.generate_source_configs import generate_configs


class SourceConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="source_configs_")
        self.root = Path(self.temporary.name)
        self.data, self.templates, self.output = self.root / "data", self.root / "templates", self.root / "generated"
        self.templates.mkdir()
        self.isolated = self.data / "prepared/source_isolated"
        self.isolated.mkdir(parents=True)
        self.combinations = []

    def tearDown(self):
        self.temporary.cleanup()

    def template(self, name="dr_grading", kind="multiclass", labels=None, classes=2, protocol="strict_cross_domain", sources=None):
        value = {"task": name, "kind": kind, "labels": labels or ["dr_grade"], "num_classes": classes,
                 "model": {"name": "unchanged_model"}, "augmentation": {"horizontal_flip": 0.5},
                 "data": {"protocol": protocol, "image_cache_dir": "old/mixed/cache"},
                 "loss": {"class_weights": [999, 999], "pos_weight": [999], "label_smoothing": 0.05},
                 "training": {"epochs": 30, "batch_size": 4, "lr": 0.0001}}
        if sources:
            value["data"]["datasets"] = sources
        (self.templates / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
        return value

    def combination(self, task, source, train_values, val_values, protocol="strict_cross_domain", external_values=(), label="dr_grade"):
        files = {}
        for split, values in (("train", train_values), ("val", val_values), ("test", ()), ("external_test", external_values)):
            rows = [{"task": task, "dataset": source, "split": split, "sample_id": f"{task}:{source}:{split}:{i}",
                     "labels": {label: value}, "label_mask": {label: True}, "annotations": {}}
                    for i, value in enumerate(values)]
            relative = f"{protocol}/tasks/{task}/{source}/{split}.jsonl"
            payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
            path = self.isolated / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            files[split] = {"path": relative, "records": len(rows), "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(), "supervised_records": len(rows)}
        self.combinations.append({"base_protocol": protocol, "task": task, "source": source,
                                  "trainable": bool(train_values and val_values), "splits": files})

    def publish_index(self):
        value = {"format_version": 1, "all_checks_passed": True, "record_bytes_preserved": True,
                 "dry_run": False, "identity_sha256": "test_identity", "combinations": self.combinations}
        (self.isolated / "isolation_manifest.json").write_text(json.dumps(value), encoding="utf-8")

    def test_source_only_weights_default_and_opt_in_external(self):
        template = self.template()
        self.combination("dr_grading", "DDR", [0, 0, 0, 1], [0, 1])
        self.combination("dr_grading", "APTOS2019", [0, 1, 1, 1], [0, 1])
        self.combination("dr_grading", "EyeQ", [0, 1], [0, 1])
        self.combination("dr_grading", "IDRiD", [], [], external_values=[0, 1])
        self.publish_index()
        result = generate_configs(self.templates, self.data, self.output, include_idrid_external=True)
        generated = json.loads((self.output / "strict_cross_domain/dr_grading_ddr.json").read_text())
        self.assertEqual(generated["data"]["datasets"], ["DDR"])
        self.assertEqual(generated["data"]["external_dataset"], "IDRiD")
        self.assertTrue(generated["data"]["source_isolated"])
        self.assertNotIn("image_cache_dir", generated["data"])
        self.assertEqual(generated["model"], template["model"])
        self.assertEqual(generated["training"], template["training"])
        expected = [math.sqrt(1 / 3), 1.0]
        expected = [x / (sum(expected) / 2) for x in expected]
        self.assertEqual(generated["loss"]["class_weights"], expected)
        self.assertNotIn("pos_weight", generated["loss"])
        self.assertFalse((self.output / "strict_cross_domain/dr_grading_eyeq.json").exists())
        self.assertEqual(result["default_entry"], "strict_cross_domain/dr_grading_ddr.json")
        again = generate_configs(self.templates, self.data, self.output, include_idrid_external=True)
        self.assertEqual(again["file_actions"], {"reused_identical": 3})

    def test_missing_class_and_binary_polarity_are_blocked(self):
        self.template()
        self.combination("dr_grading", "DDR", [0, 0], [0, 1])
        self.template("binary", kind="multilabel", labels=["positive"], classes=None)
        self.combination("binary", "OnlyPositive", [1, 1], [1, 0], label="positive")
        self.publish_index()
        result = generate_configs(self.templates, self.data, self.output)
        self.assertEqual(result["generated_config_count"], 0)
        reasons = [reason for row in result["registry"]["protocol_groups"]["strict_cross_domain"]["prohibited_or_unrouted"]
                   for reason in row["reasons"]]
        self.assertIn("missing_training_class:dr_grade:1", reasons)
        self.assertIn("no_negative_training_label:positive", reasons)

    def test_existing_source_filter_protocol_groups_and_conflict_refusal(self):
        self.template("binary", kind="multilabel", labels=["positive"], classes=None,
                      protocol="within_source", sources=["FIVES"])
        self.combination("binary", "FIVES", [0, 0, 0, 1], [0, 1], protocol="within_source", label="positive")
        self.combination("binary", "Other", [0, 1], [0, 1], protocol="within_source", label="positive")
        self.publish_index()
        result = generate_configs(self.templates, self.data, self.output)
        group = result["registry"]["protocol_groups"]["within_source"]
        self.assertTrue(group["independent_experiments_only"])
        self.assertFalse(group["default_system_eligible"])
        self.assertEqual(len(group["generated"]), 1)
        path = self.output / "within_source/binary_fives.json"
        config = json.loads(path.read_text())
        self.assertEqual(config["loss"]["pos_weight"], [math.sqrt(3)])
        original = path.read_bytes()
        changed = copy.deepcopy(config)
        changed["training"]["lr"] = 123
        path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            generate_configs(self.templates, self.data, self.output)
        self.assertNotEqual(path.read_bytes(), original)

    def test_large_backbone_remains_an_opt_in_candidate(self):
        template = self.template()
        candidate = copy.deepcopy(template)
        candidate["model"]["name"] = "vit_large_patch16_candidate"
        (self.templates / "dr_grading_dinov3_large.json").write_text(json.dumps(candidate), encoding="utf-8")
        self.combination("dr_grading", "DDR", [0, 1], [0, 1])
        self.publish_index()
        result = generate_configs(self.templates, self.data, self.output)
        entries = result["registry"]["protocol_groups"]["strict_cross_domain"]["generated"]
        standard = next(row for row in entries if row["template"] == "dr_grading.json")
        large = next(row for row in entries if row["template"] == "dr_grading_dinov3_large.json")
        self.assertTrue(standard["automatic_start_allowed"])
        self.assertTrue(large["candidate_only"])
        self.assertFalse(large["automatic_start_allowed"])


if __name__ == "__main__":
    unittest.main()
