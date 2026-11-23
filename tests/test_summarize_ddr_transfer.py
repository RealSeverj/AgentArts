"""Verify fixed-source transfer provenance and refuse mismatched comparisons."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.analyze_dr_evaluation import confusion, matrix_summary, sha256_file
from scripts.summarize_ddr_transfer import PARENT_SHA256, summarize


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def dump_lines(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(p) + "\n" for p in payload), encoding="utf-8")


class DDRTransferSummaryTests(unittest.TestCase):
    def fixture(self, root):
        run, legacy, data, parent, oldpred = [root / p for p in ("run", "legacy", "data", "parent", "oldpred")]
        index = data / "prepared/source_isolated/isolation_manifest.json"
        dump(index, {"fixture": True})
        model = {"name": "vit_large_patch14_dinov2.fixture", "img_size": 336}
        config = {"task": "dr_grading", "kind": "multiclass", "labels": ["dr_grade"], "num_classes": 5,
                  "model": model, "training": {"epochs": 30},
                  "data": {"datasets": ["DDR"], "protocol": "strict_cross_domain", "source_isolated": True,
                           "external_dataset": "IDRiD", "isolation_manifest_sha256": sha256_file(index)},
                  "initialization": {"mode": "weights_only", "checkpoint": "/parent/head_best.pt", "checkpoint_sha256": PARENT_SHA256,
                                     "parent_training_sources": ["APTOS2019"], "evaluate_before_training": True}}
        parent_config = deepcopy(config)
        parent_config["data"]["datasets"] = ["APTOS2019"]
        parent_config.pop("initialization")
        dump(run / "config.resolved.json", config)
        dump(parent / "config.resolved.json", parent_config)
        dump(parent / "checkpoint_metadata.json", {"checkpoints": {"head_best.pt": {"sha256": PARENT_SHA256, "completed_epoch": 5}}})
        manifests = {}
        hashes = {}
        for split, source in [("train", "DDR"), ("val", "DDR"), ("test", "DDR"), ("external_test", "IDRiD")]:
            rows = [{"sample_id": f"{split}_{i}", "dataset": source, "task": "dr_grading", "split": split,
                     "labels": {"dr_grade": i}, "label_mask": {"dr_grade": True},
                     "content_group_id": f"{split}_content_{i}", "content_group_evaluation_weight": 1.0}
                    for i in range(5)]
            path = data / "prepared/source_isolated/strict_cross_domain/tasks/dr_grading" / source / f"{split}.jsonl"
            dump_lines(path, rows)
            manifests[split] = (rows, path)
            hashes[split] = sha256_file(path)
        trainval = {s: hashes[s] for s in ("train", "val")}
        dump(run / "data_identity.json", {"experiment": config, "manifests_sha256": trainval})
        dump(run / "runtime.json", {"train_records": 5, "val_records": 5})
        lineage = {"mode": "weights_only", "checkpoint": "/parent/head_best.pt", "checkpoint_sha256": PARENT_SHA256,
                   "parent_completed_epoch": 5, "parent_task": "dr_grading", "parent_training_sources": ["APTOS2019"],
                   "new_run_training_sources": ["DDR"], "supervised_training_source_history": ["APTOS2019", "DDR"],
                   "independent_source_only_model": False, "starting_epoch_zero_based": 0,
                   "reset_components": ["optimizer", "scheduler", "scaler", "epoch", "best_metric", "RNG"],
                   "optimizer_state_imported": False, "parent_rng_restored": False}
        dump(run / "initialization_lineage.json", lineage)
        initial = matrix_summary(confusion(range(5), [0, 0, 2, 2, 4]))
        initial["by_dataset"] = {"DDR": deepcopy(initial)}
        dump(run / "initialization_validation.json", {"split": "val", "completed_training_epochs": 0, "optimizer_updates": 0,
             "complete_dataset_evaluated": True, "evaluated_dataset_count": 5, "initialization": lineage, "metrics": initial})
        dump(run / "checkpoint_metadata.json", {"checkpoints": {
            "best.pt": {"sha256": "b" * 64, "model_state_sha256": "d" * 64, "completed_epoch": 7,
                        "manifests_sha256": trainval, "actual_stage": "last_4_blocks"},
            "head_best.pt": {"sha256": "c" * 64, "model_state_sha256": "e" * 64, "completed_epoch": 3,
                             "manifests_sha256": trainval, "actual_stage": "head_only"}}})
        for split, source in [("test", "DDR"), ("external_test", "IDRiD")]:
            rows, manifest = manifests[split]
            other = [{**rows[i], "sample_id": f"other_{split}_{i}", "dataset": "other", "content_group_id": f"other_{i}"}
                     for i in range(5)] if split == "test" else []
            global_path = data / "prepared/tasks/dr_grading" / f"{split}.jsonl"
            dump_lines(global_path, rows + other)
            for name in ("best", "head_best", "baseline", "finetuned", "parent"):
                predicted = list(range(5)) if name in ("best", "finetuned") else [0, 0, 2, 2, 4]
                metrics = matrix_summary(confusion(range(5), predicted))
                metrics["content_weighted"] = deepcopy(metrics)
                metrics["by_dataset"] = {source: deepcopy(metrics)}
                folder = parent / "evaluation/head_best" if name == "parent" else (run / "evaluation" / name if name in ("best", "head_best") else legacy / name)
                checkpoint = "/parent/head_best.pt" if name == "parent" else f"/run/{name}.pt"
                report = {"split": split, "checkpoint": checkpoint, "config": deepcopy(parent_config if name == "parent" else config),
                          "evaluated_dataset_count": 5, "metrics": metrics}
                if name in ("baseline", "finetuned"):
                    report["config"]["model"]["name"] = "vit_small_patch14_dinov2.fixture"
                dump(folder / f"evaluation_{split}.json", report)
                exports = [{"sample_id": row["sample_id"], "dataset": row["dataset"], "target": i,
                            "predicted_class": predicted[i], "probabilities": np.eye(5)[predicted[i]].tolist(),
                            "content_group_id": row["content_group_id"], "evaluation_weight": 1.0} for i, row in enumerate(rows)]
                if name in ("baseline", "finetuned"):
                    folder = oldpred / name
                    exports += [{**exports[i], "sample_id": row["sample_id"], "dataset": "other", "content_group_id": row["content_group_id"]} for i, row in enumerate(other)]
                prediction = folder / f"predictions_{split}.jsonl"
                dump_lines(prediction, reversed(exports))
                meta = {"split": split, "checkpoint": checkpoint, "sample_count": len(exports), "expected_dataset_count": len(exports),
                        "unique_sample_ids": len(exports), "source_counts": {source: 5, **({"other": 5} if other and name in ("baseline", "finetuned") else {})},
                        "manifest_sha256": sha256_file(global_path if name in ("baseline", "finetuned") else manifest),
                        "jsonl_sha256": sha256_file(prediction), "checkpoint_completed_epoch": 5 if name == "parent" else (7 if name == "best" else 3),
                        "complete_dataset_evaluated": True, "all_targets_known": True, "target_label": "dr_grade", "class_order": list(range(5))}
                dump(prediction.with_suffix(".meta.json"), meta)
        return run, legacy, data, parent, oldpred

    def summarize(self, paths):
        return summarize(*paths[:4], legacy_prediction_root=paths[4], include_parent_reference=True,
                         expected_counts={"train": 5, "val": 5, "test": 5, "external_test": 5})

    def test_same_source_predictions_and_transfer_history_are_verified_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.fixture(root)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            result = self.summarize(paths)
            model = result["splits"]["test"]["models"]["ddr_transfer_best"]
            self.assertEqual(result["splits"]["test"]["source"], "DDR")
            self.assertEqual(model["metrics"]["recall_by_grade"], [1.] * 5)
            self.assertEqual(model["current_run_training_sources"], ["DDR"])
            self.assertEqual(model["training_history_sources"], ["APTOS2019", "DDR"])
            self.assertEqual(model["checkpoint_completed_epoch"], 7)
            self.assertFalse(model["independent_source_only_model"])
            self.assertIn("Exact isolated source IDs", result["splits"]["test"]["models"]["legacy_baseline"]["comparison_scope"])
            self.assertEqual(result["references"]["aptos_parent_head_external_test"]["source"], "IDRiD")
            self.assertTrue(all(p.read_bytes() == payload for p, payload in before.items()))

    def test_incorrect_parent_or_single_source_history_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary))
            path = paths[0] / "initialization_lineage.json"
            original = json.loads(path.read_text())
            for change in ({"checkpoint_sha256": "f" * 64}, {"supervised_training_source_history": ["DDR"], "independent_source_only_model": True}):
                with self.subTest(change=change):
                    dump(path, {**original, **change})
                    with self.assertRaisesRegex(ValueError, "lineage"):
                        self.summarize(paths)

    def test_wrong_content_group_metadata_and_mixed_current_config_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary))
            pred = paths[0] / "evaluation/best/predictions_test.jsonl"
            original = pred.read_bytes()
            rows = [json.loads(x) for x in pred.read_text().splitlines()]
            rows[0]["content_group_id"] = "changed"
            dump_lines(pred, rows)
            with self.assertRaisesRegex(ValueError, "content_group"):
                self.summarize(paths)
            pred.write_bytes(original)
            meta = pred.with_suffix(".meta.json")
            obj = json.loads(meta.read_text())
            obj["manifest_sha256"] = "f" * 64
            dump(meta, obj)
            with self.assertRaisesRegex(ValueError, "manifest SHA"):
                self.summarize(paths)
            config = paths[0] / "config.resolved.json"
            obj = json.loads(config.read_text())
            obj["data"]["datasets"] = ["DDR", "APTOS2019"]
            dump(config, obj)
            with self.assertRaisesRegex(ValueError, "isolated DDR"):
                self.summarize(paths)


if __name__ == "__main__":
    unittest.main()
