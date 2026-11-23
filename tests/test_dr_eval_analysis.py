"""Hand-checkable DR metrics and paired group-bootstrap analysis fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.analyze_dr_evaluation import (analyze, confusion, load_predictions, matrix_summary,
                                         paired_cluster_bootstrap)


class DREvaluationTests(unittest.TestCase):
    def test_binary_merge_and_row_recalls(self):
        result = matrix_summary(confusion([0, 1, 2, 4], [0, 2, 1, 4]))
        self.assertEqual(result["recall_by_grade"], [1.0, 0.0, 0.0, None, 1.0])
        self.assertEqual(result["predicted_counts"], [1, 1, 1, 0, 1])
        binary = result["grade_ge_2_binary"]
        self.assertEqual((binary["tp"], binary["fn"], binary["tn"], binary["fp"]), (1, 1, 1, 1))
        self.assertEqual(binary["sensitivity"], 0.5)
        self.assertEqual(binary["specificity"], 0.5)
        severe = matrix_summary(confusion([3, 4], [0, 1]))
        self.assertEqual(severe["severe_underestimation"]["rate"], 1.0)
        self.assertEqual(severe["grade4_recall"], 0.0)

    def test_identical_models_have_zero_paired_bootstrap_delta(self):
        manifest, predictions = {}, {}
        for i, (source, target) in enumerate((("a", 0), ("a", 4), ("b", 0), ("b", 4))):
            sid = str(i)
            manifest[sid] = {"sample_id": sid, "dataset": source, "labels": {"dr_grade": target},
                             "leakage_group_id": f"group{i}"}
            predictions[sid] = {"predicted_class": 0}
        result = paired_cluster_bootstrap(manifest, predictions, predictions, replicates=100, seed=7)
        self.assertEqual(result["all"]["point_delta_full_minus_head"], 0.0)
        self.assertEqual(result["all"]["ci95_percentile"], [0.0, 0.0])
        self.assertGreater(result["all"]["valid_replicates"], 0)
        self.assertEqual(result["row_group_identifier_counts"], {"leakage_group_id": 4})

    def test_strict_alignment_report_audit_and_unchanged_originals(self):
        with tempfile.TemporaryDirectory(prefix="dr_evaluation_") as temporary:
            root = Path(temporary)
            data, evaluation = root / "data", root / "evaluation"
            task = data / "prepared/tasks/dr_grading"
            task.mkdir(parents=True)
            rows_by_split = {}
            for split in ("train", "test", "external_test"):
                rows = []
                for i, target in enumerate((0, 0, 2, 4)):
                    rows.append({"sample_id": f"{split}{i}", "dataset": "fixture", "task": "dr_grading", "split": split,
                                 "labels": {"dr_grade": target}, "label_mask": {"dr_grade": True},
                                 "content_group_id": f"content{i}", "leakage_group_id": f"leak{i}",
                                 "content_group_evaluation_weight": 1.0})
                rows_by_split[split] = rows
                (task / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            for model in ("baseline", "finetuned"):
                directory = evaluation / model
                directory.mkdir(parents=True)
                for split in ("test", "external_test"):
                    exports = []
                    for row in rows_by_split[split]:
                        y = row["labels"]["dr_grade"]
                        p = 0 if model == "baseline" else y
                        exports.append({"sample_id": row["sample_id"], "dataset": row["dataset"], "target": y,
                                        "predicted_class": p, "probabilities": np.eye(5)[p].tolist(),
                                        "content_group_id": row["content_group_id"], "evaluation_weight": 1.0})
                    matrix = confusion([row["target"] for row in exports], [row["predicted_class"] for row in exports])
                    metrics = matrix_summary(matrix)
                    metrics["by_dataset"] = {"fixture": copy.deepcopy(metrics)}
                    (directory / f"evaluation_{split}.json").write_text(json.dumps({"split": split, "metrics": metrics}), encoding="utf-8")
                    (directory / f"predictions_{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in reversed(exports)), encoding="utf-8")
            before = {p: p.read_bytes() for p in evaluation.rglob("*") if p.is_file()}
            result = analyze(evaluation, data, bootstrap_replicates=20, expected_counts={"test": 4, "external_test": 4})
            self.assertEqual(result["training_majority_class"], 0)
            self.assertEqual(result["splits"]["test"]["audited_count"], 4)
            self.assertEqual(result["splits"]["test"]["models"]["finetuned"]["all"]["qwk"], 1.0)
            self.assertEqual(result["splits"]["test"]["models"]["baseline"]["all"]["qwk"], 0.0)
            self.assertTrue(all(path.read_bytes() == payload for path, payload in before.items()))
            # A duplicated export is a hard failure, rather than silently
            # deduplicating two predictions and reporting a plausible metric.
            path = evaluation / "baseline/predictions_test.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(path.read_text().splitlines()[0] + "\n")
            manifest = {row["sample_id"]: row for row in rows_by_split["test"]}
            with self.assertRaisesRegex(ValueError, "repeated prediction"):
                load_predictions(path, manifest)


if __name__ == "__main__":
    unittest.main()
