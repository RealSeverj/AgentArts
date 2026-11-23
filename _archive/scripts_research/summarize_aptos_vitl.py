"""Audit fixed APTOS-only ViT-L evaluations against source-matched history.

New inputs: <run-root>/evaluation/{best,head_best}/evaluation_<split>.json
and predictions_<split>.jsonl. Only APTOS test354 and IDRiD external506 are
compared. No checkpoint, threshold, training parameter or original file changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.analyze_dr_evaluation import (
    assert_report_matches, load_manifest, load_predictions, matrix_summary,
    read_json, sha256_file, summarize_predictions,
)

SPLITS = {"test": ("APTOS2019", 354), "external_test": ("IDRiD", 506)}
NEW_CHECKPOINTS = ("best", "head_best")
LEGACY_CHECKPOINTS = ("baseline", "finetuned")


def checked_matrix(metrics, expected_n, context):
    required = {"n", "qwk", "accuracy", "macro_f1", "confusion_matrix"}
    if not isinstance(metrics, dict) or required - metrics.keys():
        raise ValueError(f"Required metrics missing: {context}")
    if type(metrics["n"]) is not int or metrics["n"] != expected_n:
        raise ValueError(f"Source-specific metrics.n differs from required count: {context}")
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    if (matrix.shape != (5, 5) or not np.isfinite(matrix).all() or
            (matrix < 0).any() or not np.equal(matrix, np.floor(matrix)).all() or
            int(matrix.sum()) != expected_n):
        raise ValueError(f"Invalid ordinary five-class confusion matrix: {context}")
    summary = matrix_summary(matrix, expected_n)
    for key in ("qwk", "accuracy", "macro_f1"):
        observed, expected = metrics[key], summary[key]
        if observed is None or expected is None:
            if observed != expected:
                raise ValueError(f"Undefined metric differs from matrix: {context}/{key}")
        elif not np.isclose(float(observed), expected, rtol=0, atol=1e-7):
            raise ValueError(f"Metric differs from matrix: {context}/{key}")
    return summary


def checkpoint_relation(metadata, prediction_agreement):
    """Serialization differences alone do not prove model weights differ."""
    result = {"checkpoint_files_identical": None, "model_weights_identical": None,
              "same_completed_epoch": None, "prediction_agreement": prediction_agreement,
              "note": "Equal predictions on these samples do not establish equal model weights."}
    if metadata is None:
        result["metadata_available"] = False
        return result
    entries = metadata.get("checkpoints", metadata)
    best = entries.get("best.pt", entries.get("best"))
    head = entries.get("head_best.pt", entries.get("head_best"))
    if not isinstance(best, dict) or not isinstance(head, dict):
        raise ValueError("Checkpoint metadata requires best.pt and head_best.pt records")
    result["metadata_available"] = True
    for field in ("sha256", "model_state_sha256"):
        values = (best.get(field), head.get(field))
        if any(v is not None and (not isinstance(v, str) or len(v) != 64 or
                                   any(c not in "0123456789abcdef" for c in v)) for v in values):
            raise ValueError(f"Invalid checkpoint metadata {field}")
        if all(v is not None for v in values):
            equal = values[0] == values[1]
            if field == "sha256":
                result["checkpoint_files_identical"] = equal
                if equal:
                    result["model_weights_identical"] = True
            else:
                if result["model_weights_identical"] is True and not equal:
                    raise ValueError("File and model-state identity metadata contradict")
                result["model_weights_identical"] = equal
    epochs = (best.get("completed_epoch"), head.get("completed_epoch"))
    if any(v is not None and (type(v) is not int or v < 1) for v in epochs):
        raise ValueError("Invalid checkpoint completed_epoch")
    if all(v is not None for v in epochs):
        result["same_completed_epoch"] = epochs[0] == epochs[1]
    result["records"] = {"best.pt": best, "head_best.pt": head}
    return result


def summarize(run_root, legacy_root, data_root, checkpoint_metadata=None, expected_counts=None):
    run_root, legacy_root, data_root = map(Path, (run_root, legacy_root, data_root))
    expected_counts = expected_counts or {key: value[1] for key, value in SPLITS.items()}
    inputs, results, agreements = set(), {}, {}
    model_names, configurations = {}, {}
    for split, (source, _) in SPLITS.items():
        count = expected_counts[split]
        manifest_path = (data_root / "prepared/source_isolated/strict_cross_domain/tasks/dr_grading" /
                         source / f"{split}.jsonl")
        manifest = load_manifest(manifest_path, split, count)
        if {r["dataset"] for r in manifest.values()} != {source}:
            raise ValueError(f"Isolated manifest contains an unexpected source: {split}")
        inputs.add(manifest_path)
        support = [sum(r["labels"]["dr_grade"] == grade for r in manifest.values()) for grade in range(5)]
        models, predictions = {}, {}
        for name in NEW_CHECKPOINTS:
            directory = run_root / "evaluation" / name
            report_path = directory / f"evaluation_{split}.json"
            prediction_path = directory / f"predictions_{split}.jsonl"
            report = read_json(report_path)
            config = report.get("config", {})
            data = config.get("data", {})
            model_name = config.get("model", {}).get("name")
            if (config.get("task") != "dr_grading" or config.get("kind") != "multiclass" or
                    config.get("labels") != ["dr_grade"] or config.get("num_classes") != 5 or
                    data.get("datasets") != ["APTOS2019"] or data.get("source_isolated") is not True or
                    data.get("protocol") != "strict_cross_domain"):
                raise ValueError(f"New checkpoint is not an isolated APTOS DR configuration: {name}/{split}")
            if not isinstance(model_name, str) or not model_name.startswith("vit_large"):
                raise ValueError(f"New checkpoint does not declare a ViT-L model: {name}/{split}")
            if split == "external_test" and data.get("external_dataset") != "IDRiD":
                raise ValueError("New external evaluation must explicitly route IDRiD")
            if report.get("evaluated_dataset_count", count) != count:
                raise ValueError("New report dataset count differs from the full isolated split")
            checked_matrix(report.get("metrics"), count, str(report_path))
            if set(report["metrics"].get("by_dataset", {})) != {source}:
                raise ValueError(f"New report contains mixed/incorrect evaluation sources: {name}/{split}")
            exported = load_predictions(prediction_path, manifest)
            computed = summarize_predictions(sorted(manifest), manifest, exported)
            assert_report_matches(report, split, computed, {source: computed}, str(report_path))
            inputs.update((report_path, prediction_path))
            meta_path = directory / f"predictions_{split}.meta.json"
            meta = None
            if meta_path.is_file():
                meta = read_json(meta_path)
                if (meta.get("sample_count") != count or meta.get("unique_sample_ids") != count or
                        meta.get("source_counts") != {source: count} or
                        meta.get("manifest_sha256") != sha256_file(manifest_path) or
                        meta.get("jsonl_sha256") != sha256_file(prediction_path)):
                    raise ValueError(f"Prediction metadata disagrees with actual inputs: {meta_path}")
                inputs.add(meta_path)
            models["aptos_vitl_" + name] = {
                "checkpoint": report.get("checkpoint"),
                "checkpoint_completed_epoch": meta.get("checkpoint_completed_epoch") if meta else None,
                "training_sources": ["APTOS2019"], "metrics": computed,
            }
            model_names[name] = config.get("model", {}).get("name")
            identity = {k: config.get(k) for k in ("task", "kind", "labels", "num_classes", "model", "data")}
            if configurations and identity != next(iter(configurations.values())):
                raise ValueError("New checkpoint evaluation configurations differ in model/data contracts")
            configurations[name] = identity
            predictions[name] = exported
        agreements[split] = {
            "same_predicted_classes": all(predictions["best"][sid]["predicted_class"] == predictions["head_best"][sid]["predicted_class"] for sid in manifest),
            "same_probability_vectors": all(predictions["best"][sid]["probabilities"] == predictions["head_best"][sid]["probabilities"] for sid in manifest),
        }
        for name in LEGACY_CHECKPOINTS:
            path = legacy_root / name / f"evaluation_{split}.json"
            report = read_json(path)
            if report.get("split") != split:
                raise ValueError("Legacy evaluation split differs")
            recorded = report.get("metrics", {}).get("by_dataset", {}).get(source)
            # Internal legacy reports may contain any number of other sources.
            # Their mixed aggregate score is deliberately never a comparator.
            computed = checked_matrix(recorded, count, str(path) + "/" + source)
            if computed["target_support"] != support:
                raise ValueError("Legacy source subset class support differs from the new held-out split")
            models["legacy_" + name] = {"checkpoint": report.get("checkpoint"),
                                         "training_sources": ["APTOS2019", "DDR", "EyeQ"],
                                         "selected_report_section": "metrics.by_dataset." + source,
                                         "metrics": computed}
            inputs.add(path)
        results[split] = {"source": source, "n": count, "models": models,
                          "manifest_sha256": sha256_file(manifest_path),
                          "new_sample_ids_and_targets_verified_against_manifest": True,
                          "legacy_comparison_scope": "Source-specific count/class support verified; legacy mixed aggregate metrics are excluded."}
    metadata = None
    if checkpoint_metadata is not None:
        metadata = read_json(checkpoint_metadata)
        inputs.add(Path(checkpoint_metadata))
    return {
        "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "all_checks_passed": True, "new_model_names": model_names,
        "splits": results, "checkpoint_relation": checkpoint_relation(metadata, agreements),
        "metric_selection": "Fixed five-class argmax; QWK and per-grade metrics only. No threshold fitting or checkpoint selection.",
        "comparison_limits": [
            "New ViT-L models use APTOS2019 training only; historical models trained on APTOS2019, DDR and EyeQ together.",
            "Training sample count, checkpoint-selection validation set, architecture/pretraining and other settings differ; this is a historical-system comparison, not a model-size-only causal ablation.",
            "Only the same APTOS354 and IDRiD506 source-specific held-out partitions are compared. Historical mixed validation/test aggregate scores are not compared.",
            "IDRiD and these historical test results were already observed in this project; they are fixed external/source benchmarks, not a newly unseen clinical cohort.",
            "Known image/group isolation does not establish patient independence where patient IDs are unavailable.",
        ],
        "input_files": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in sorted(inputs)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=PROJECT / "cloud_upload/aptos_vitl_20261113")
    parser.add_argument("--legacy-root", type=Path, default=PROJECT / "cloud_upload/evaluation")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--checkpoint-metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = args.checkpoint_metadata or args.run_root / "checkpoint_metadata.json"
    if args.checkpoint_metadata is None and not metadata.is_file():
        metadata = None
    result = summarize(args.run_root, args.legacy_root, args.data_root, metadata)
    destination = args.output.resolve()
    if destination in {Path(item["path"]) for item in result["input_files"]}:
        raise ValueError("Summary output cannot overwrite an original input")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps({"all_checks_passed": True, "output": str(destination),
                      "new_test_count": result["splits"]["test"]["n"],
                      "new_external_count": result["splits"]["external_test"]["n"]}))


if __name__ == "__main__":
    main()
