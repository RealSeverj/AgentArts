"""Fair source-matched summaries and refusal of contaminated/incomplete inputs."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.analyze_dr_evaluation import confusion, matrix_summary, sha256_file
from scripts.summarize_aptos_vitl import checkpoint_relation, summarize


class APTOSSummaryTests(unittest.TestCase):
    def make_fixture(self, root):
        run, legacy, data = root / "run", root / "legacy", root / "data"
        config = {"task": "dr_grading", "kind": "multiclass", "labels": ["dr_grade"], "num_classes": 5,
                  "model": {"name": "vit_large_patch14_dinov2.fixture", "img_size": 392},
                  "data": {"source_isolated": True, "protocol": "strict_cross_domain", "datasets": ["APTOS2019"],
                           "external_dataset": "IDRiD"}}
        for split, source in (("test", "APTOS2019"), ("external_test", "IDRiD")):
            manifest_rows = [{"sample_id": f"{split}_{i}", "dataset": source, "task": "dr_grading", "split": split,
                              "labels": {"dr_grade": i}, "label_mask": {"dr_grade": True},
                              "content_group_id": f"{split}_content_{i}", "content_group_evaluation_weight": 1.0}
                             for i in range(5)]
            manifest = data / "prepared/source_isolated/strict_cross_domain/tasks/dr_grading" / source / f"{split}.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("".join(json.dumps(r) + "\n" for r in manifest_rows), encoding="utf-8")
            for name in ("best", "head_best", "baseline", "finetuned"):
                predicted = list(range(5)) if name in ("best", "finetuned") else [0, 0, 2, 2, 4]
                metrics = matrix_summary(confusion(range(5), predicted))
                metrics["content_weighted"] = deepcopy(metrics)
                source_metrics = deepcopy(metrics)
                metrics["by_dataset"] = {source: source_metrics}
                folder = run / "evaluation" / name if name in ("best", "head_best") else legacy / name
                folder.mkdir(parents=True, exist_ok=True)
                report = {"split": split, "checkpoint": f"/cloud/{name}.pt", "config": deepcopy(config),
                          "evaluated_dataset_count": 5, "metrics": metrics}
                if name in ("baseline", "finetuned") and split == "test":
                    # Deliberately unrelated historical aggregate. The correct
                    # comparator remains only its APTOS source-specific entry.
                    aggregate = matrix_summary(confusion([0] * 99, [0] * 99))
                    aggregate["by_dataset"] = {source: source_metrics, "other": {"n": 94}}
                    report["metrics"] = aggregate
                (folder / f"evaluation_{split}.json").write_text(json.dumps(report), encoding="utf-8")
                if name in ("best", "head_best"):
                    exports = [{"sample_id": row["sample_id"], "dataset": source, "target": i,
                                "predicted_class": predicted[i], "probabilities": np.eye(5)[predicted[i]].tolist(),
                                "content_group_id": row["content_group_id"], "evaluation_weight": 1.0}
                               for i, row in enumerate(manifest_rows)]
                    prediction = folder / f"predictions_{split}.jsonl"
                    prediction.write_text("".join(json.dumps(r) + "\n" for r in reversed(exports)), encoding="utf-8")
                    meta = {"sample_count": 5, "unique_sample_ids": 5, "source_counts": {source: 5},
                            "manifest_sha256": sha256_file(manifest), "jsonl_sha256": sha256_file(prediction),
                            "checkpoint_completed_epoch": 3 if name == "best" else 1}
                    (folder / f"predictions_{split}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return run, legacy, data

    def test_source_subset_comparison_recomputes_metrics_without_changing_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, legacy, data = self.make_fixture(root)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            result = summarize(run, legacy, data, expected_counts={"test": 5, "external_test": 5})
            internal = result["splits"]["test"]
            self.assertEqual(internal["models"]["legacy_baseline"]["metrics"]["n"], 5)
            self.assertLess(internal["models"]["legacy_baseline"]["metrics"]["qwk"], 1.0)
            self.assertEqual(internal["models"]["aptos_vitl_best"]["metrics"]["recall_by_grade"], [1.0] * 5)
            self.assertIsNone(result["checkpoint_relation"]["model_weights_identical"])
            self.assertTrue(all(path.read_bytes() == payload for path, payload in before.items()))

    def test_mixed_sources_partial_counts_and_changed_report_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, legacy, data = self.make_fixture(root)
            path = run / "evaluation/best/evaluation_test.json"
            original = json.loads(path.read_text())
            with self.assertRaisesRegex(ValueError, "supervised count"):
                summarize(run, legacy, data, expected_counts={"test": 6, "external_test": 5})
            wrong = deepcopy(original)
            wrong["metrics"]["by_dataset"]["DDR"] = wrong["metrics"]["by_dataset"]["APTOS2019"]
            path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, "mixed/incorrect"):
                summarize(run, legacy, data, expected_counts={"test": 5, "external_test": 5})
            wrong = deepcopy(original)
            wrong["metrics"]["qwk"] = 0.5
            path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, "Metric differs"):
                summarize(run, legacy, data, expected_counts={"test": 5, "external_test": 5})

    def test_checkpoint_identity_is_not_inferred_from_prediction_agreement(self):
        agreement = {"test": {"same_predicted_classes": True, "same_probability_vectors": True}}
        self.assertIsNone(checkpoint_relation(None, agreement)["model_weights_identical"])
        meta = {"checkpoints": {"best.pt": {"sha256": "a" * 64}, "head_best.pt": {"sha256": "b" * 64}}}
        result = checkpoint_relation(meta, agreement)
        self.assertFalse(result["checkpoint_files_identical"])
        self.assertIsNone(result["model_weights_identical"])
        meta["checkpoints"]["best.pt"]["model_state_sha256"] = "c" * 64
        meta["checkpoints"]["head_best.pt"]["model_state_sha256"] = "c" * 64
        self.assertTrue(checkpoint_relation(meta, agreement)["model_weights_identical"])
        meta["checkpoints"]["head_best.pt"]["sha256"] = "a" * 64
        self.assertTrue(checkpoint_relation(meta, agreement)["checkpoint_files_identical"])


if __name__ == "__main__":
    unittest.main()
