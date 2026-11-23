"""Audit fixed DDR-transfer predictions on DDR3898 / IDRiD506 without tuning.

This run's manifests contain DDR only, but supervised model history is
APTOS2019 -> DDR. Reads originals; writes only the requested new summary.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
    assert_report_matches, load_manifest, load_predictions, read_json,
    sha256_file, summarize_predictions,
)
from scripts.summarize_aptos_vitl import checked_matrix, checkpoint_relation

SPLITS = {"test": ("DDR", 3898), "external_test": ("IDRiD", 506)}
PARENT_SHA256 = "80c13752f5655290b2349ebc1dce951c83ef7aa7aa27d17572d09fbd0078fe05"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path, inputs):
    inputs.add(Path(path))
    return read_json(path)


def strict_manifest(path, split, source, count, inputs):
    rows = load_manifest(path, split, count)
    with Path(path).open(encoding="utf-8-sig") as handle:
        total = sum(bool(line.strip()) for line in handle)
    require(total == count and {r["dataset"] for r in rows.values()} == {source},
            f"Isolated manifest has unexpected sources or unknown/extra records: {path}")
    inputs.add(path)
    return rows


def validate_config(config, expected_data_source="DDR"):
    data = config.get("data", {})
    require(config.get("task") == "dr_grading" and config.get("kind") == "multiclass"
            and config.get("labels") == ["dr_grade"] and config.get("num_classes") == 5
            and data.get("datasets") == [expected_data_source]
            and data.get("source_isolated") is True and data.get("protocol") == "strict_cross_domain"
            and data.get("external_dataset") == "IDRiD",
            f"Configuration must use isolated {expected_data_source} DR manifests and IDRiD external_test")
    require(str(config.get("model", {}).get("name", "")).startswith("vit_large"),
            "Expected a declared ViT-L model")


def contract(config):
    return {k: config.get(k) for k in ("task", "kind", "labels", "num_classes", "model", "initialization")}


def checked_export(directory, split, source, manifest, inputs):
    count = len(manifest)
    report_path = directory / f"evaluation_{split}.json"
    prediction_path = directory / f"predictions_{split}.jsonl"
    report = load(report_path, inputs)
    meta = load(directory / f"predictions_{split}.meta.json", inputs)
    require(report.get("evaluated_dataset_count") == count,
            f"Evaluation must cover the complete source split: {report_path}")
    checked_matrix(report.get("metrics"), count, str(report_path))
    require(set(report["metrics"].get("by_dataset", {})) == {source},
            f"Mixed/incorrect evaluation sources: {report_path}")
    predictions = load_predictions(prediction_path, manifest)
    inputs.add(prediction_path)
    computed = summarize_predictions(sorted(manifest), manifest, predictions)
    assert_report_matches(report, split, computed, {source: computed}, str(report_path))
    require(meta.get("split") == split and meta.get("sample_count") == count
            and meta.get("expected_dataset_count") == count and meta.get("unique_sample_ids") == count
            and meta.get("source_counts") == {source: count}
            and meta.get("jsonl_sha256") == sha256_file(prediction_path)
            and meta.get("complete_dataset_evaluated") is True and meta.get("all_targets_known") is True
            and meta.get("target_label") == "dr_grade" and meta.get("class_order") == list(range(5))
            and meta.get("checkpoint") == report.get("checkpoint"),
            f"Prediction metadata does not describe the complete fixed export: {prediction_path}")
    require(type(meta.get("checkpoint_completed_epoch")) is int and meta["checkpoint_completed_epoch"] >= 1,
            f"Missing/invalid completed epoch: {prediction_path}")
    return report, meta, predictions, computed


def audit_lineage(run_root, parent_root, data_root, metadata_path, inputs, counts):
    config = load(run_root / "config.resolved.json", inputs)
    identity = load(run_root / "data_identity.json", inputs)
    runtime = load(run_root / "runtime.json", inputs)
    lineage = load(run_root / "initialization_lineage.json", inputs)
    initialization = load(run_root / "initialization_validation.json", inputs)
    metadata = load(metadata_path, inputs)
    parent_meta = load(parent_root / "checkpoint_metadata.json", inputs)
    parent_config = load(parent_root / "config.resolved.json", inputs)
    require(metadata.get("all_checks_passed", True) is True and parent_meta.get("all_checks_passed", True) is True,
            "Checkpoint metadata reports a failed audit")
    validate_config(config)
    validate_config(parent_config, "APTOS2019")
    parent = parent_meta.get("checkpoints", {}).get("head_best.pt", {})
    require(parent.get("sha256") == PARENT_SHA256 and parent.get("completed_epoch") == 5,
            "Parent metadata is not the pinned APTOS epoch5 head_best checkpoint")
    spec = config.get("initialization", {})
    require(spec.get("mode") == "weights_only" and spec.get("checkpoint_sha256") == PARENT_SHA256
            and spec.get("parent_training_sources") == ["APTOS2019"]
            and spec.get("evaluate_before_training") is True,
            "Configuration initialization differs from pinned APTOS epoch5 weights-only contract")
    require(all(config.get(k) == parent_config.get(k) for k in ("task", "kind", "labels", "num_classes", "model")),
            "Parent and transfer task/model contracts differ")
    require(lineage.get("mode") == "weights_only" and lineage.get("checkpoint_sha256") == PARENT_SHA256
            and lineage.get("checkpoint") == spec.get("checkpoint")
            and lineage.get("parent_completed_epoch") == 5 and lineage.get("parent_task") == "dr_grading"
            and lineage.get("parent_training_sources") == ["APTOS2019"]
            and lineage.get("new_run_training_sources") == ["DDR"]
            and lineage.get("supervised_training_source_history") == ["APTOS2019", "DDR"]
            and lineage.get("independent_source_only_model") is False
            and lineage.get("starting_epoch_zero_based") == 0
            and set(lineage.get("reset_components", [])) == {"optimizer", "scheduler", "scaler", "epoch", "best_metric", "RNG"}
            and lineage.get("optimizer_state_imported") is False and lineage.get("parent_rng_restored") is False,
            "Initialization lineage is incomplete or falsely claims independent DDR-only training")
    if "initialization_lineage" in metadata:
        require(metadata["initialization_lineage"] == lineage, "Checkpoint metadata initialization lineage differs")
    require(initialization.get("split") == "val" and initialization.get("completed_training_epochs") == 0
            and initialization.get("optimizer_updates") == 0 and initialization.get("complete_dataset_evaluated") is True
            and initialization.get("evaluated_dataset_count") == counts["val"]
            and initialization.get("initialization") == lineage,
            "Initialization validation must precede any update on the complete DDR val split")
    initial_metrics = checked_matrix(initialization.get("metrics"), counts["val"], "initialization_validation")
    require(set(initialization["metrics"].get("by_dataset", {})) == {"DDR"}, "Initialization validation has mixed sources")
    initial_source = checked_matrix(initialization["metrics"]["by_dataset"]["DDR"], counts["val"], "initialization_validation/DDR")
    require(initial_source["confusion_matrix"] == initial_metrics["confusion_matrix"], "Initialization source/overall matrices differ")
    require(identity.get("experiment") is not None and contract(identity["experiment"]) == contract(config)
            and identity["experiment"].get("data") == config["data"], "Run data identity differs from resolved config")
    require(runtime.get("train_records") == counts["train"] and runtime.get("val_records") == counts["val"],
            "Runtime training/validation counts differ from required DDR manifests")
    isolation = data_root / "prepared/source_isolated/isolation_manifest.json"
    require(config["data"].get("isolation_manifest_sha256") == sha256_file(isolation), "Isolation index hash differs")
    inputs.add(isolation)
    train_val = {}
    for split in ("train", "val"):
        path = data_root / "prepared/source_isolated/strict_cross_domain/tasks/dr_grading/DDR" / f"{split}.jsonl"
        manifest = strict_manifest(path, split, "DDR", counts[split], inputs)
        train_val[split] = sha256_file(path)
        require(identity.get("manifests_sha256", {}).get(split) == train_val[split], "Training data identity manifest hash differs")
        if split == "val":
            require(initial_metrics["target_support"] == [sum(r["labels"]["dr_grade"] == g for r in manifest.values()) for g in range(5)],
                    "Initialization validation support differs from DDR val manifest")
    for name in ("best", "head_best"):
        entry = metadata.get("checkpoints", {}).get(name + ".pt", {})
        digest = entry.get("sha256")
        require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Missing checkpoint file SHA256")
        require(type(entry.get("completed_epoch")) is int and 1 <= entry["completed_epoch"] <= config["training"]["epochs"], "Invalid current-run completed epoch")
        require(entry.get("manifests_sha256") == train_val, "Checkpoint train/val hashes differ from current DDR manifests")
    return config, metadata, lineage, {"parent": parent, "initialization_validation_metrics": initial_metrics,
                                     "parent_checkpoint_sha256": PARENT_SHA256, "parent_completed_epoch": 5,
                                     "current_run_manifest_sha256": train_val}


def summarize(run_root, legacy_root, data_root, parent_root, checkpoint_metadata=None,
              legacy_prediction_root=None, include_parent_reference=False, expected_counts=None):
    run_root, legacy_root, data_root, parent_root = map(Path, (run_root, legacy_root, data_root, parent_root))
    counts = expected_counts or {"train": 6041, "val": 2479, "test": 3898, "external_test": 506}
    inputs, results, agreements = set(), {}, {}
    config, metadata, lineage, lineage_evidence = audit_lineage(
        run_root, parent_root, data_root, Path(checkpoint_metadata or run_root / "checkpoint_metadata.json"), inputs, counts)
    history_path = run_root / "history.jsonl"
    history = {}
    if history_path.is_file():
        inputs.add(history_path)
        with history_path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    require(type(row.get("epoch")) is int and row["epoch"] not in history, "Duplicate/invalid history epoch")
                    history[row["epoch"]] = row
    for split, (source, _) in SPLITS.items():
        count = counts[split]
        path = data_root / "prepared/source_isolated/strict_cross_domain/tasks/dr_grading" / source / f"{split}.jsonl"
        manifest = strict_manifest(path, split, source, count, inputs)
        support = [sum(r["labels"]["dr_grade"] == g for r in manifest.values()) for g in range(5)]
        models, predictions = {}, {}
        for name in ("best", "head_best"):
            report, meta, exported, computed = checked_export(run_root / "evaluation" / name, split, source, manifest, inputs)
            evaluated_config = report.get("config", {})
            validate_config(evaluated_config)
            # Evaluation may remove only the image cache directory to read original images.
            run_data, eval_data = dict(config["data"]), dict(evaluated_config["data"])
            run_data.pop("image_cache_dir", None); eval_data.pop("image_cache_dir", None)
            require(contract(evaluated_config) == contract(config) and eval_data == run_data,
                    "Evaluation model/data/initialization contract differs from resolved training config")
            require(meta.get("manifest_sha256") == sha256_file(path), "Prediction manifest SHA differs")
            entry = metadata["checkpoints"][name + ".pt"]
            require(meta["checkpoint_completed_epoch"] == entry["completed_epoch"], "Prediction completed epoch differs from checkpoint metadata")
            require(str(report.get("checkpoint", "")).replace("\\", "/").endswith("/" + name + ".pt"), "Checkpoint export directory/name differs")
            epoch = entry["completed_epoch"]
            stage = entry.get("actual_stage", entry.get("stage"))
            observed_stage = history.get(epoch, {}).get("stage")
            require(not stage or not observed_stage or stage == observed_stage, "Checkpoint and history stages disagree")
            stage = stage or observed_stage
            models["ddr_transfer_" + name] = {"evaluation_name": name, "checkpoint": report["checkpoint"],
                "checkpoint_completed_epoch": epoch, "checkpoint_sha256": entry["sha256"], "actual_stage": stage,
                "current_run_training_sources": ["DDR"], "training_sources": ["APTOS2019", "DDR"],
                "training_history_sources": ["APTOS2019", "DDR"], "independent_source_only_model": False,
                "supervised_history": "APTOS2019 epoch5 head -> fresh DDR run (weights only; all optimizer state reset)",
                "prediction_file": str((run_root / "evaluation" / name / f"predictions_{split}.jsonl").resolve()), "metrics": computed}
            predictions[name] = exported
        agreements[split] = {"same_predicted_classes": all(predictions["best"][sid]["predicted_class"] == predictions["head_best"][sid]["predicted_class"] for sid in manifest),
                             "same_probability_vectors": all(predictions["best"][sid]["probabilities"] == predictions["head_best"][sid]["probabilities"] for sid in manifest)}
        for name in ("baseline", "finetuned"):
            report_path = legacy_root / name / f"evaluation_{split}.json"
            report = load(report_path, inputs)
            require(report.get("split") == split, "Legacy report split differs")
            legacy_model = report.get("config", {}).get("model", {}).get("name", "")
            require(isinstance(legacy_model, str) and legacy_model.startswith("vit_small"), "Legacy reference must declare the historical ViT-S model")
            computed = checked_matrix(report.get("metrics", {}).get("by_dataset", {}).get(source), count, str(report_path) + "/" + source)
            require(computed["target_support"] == support, "Legacy source-specific support differs from isolated held-out split")
            scope = "Source-specific count/class support verified; raw legacy prediction identity not checked."
            if legacy_prediction_root is not None:
                global_path = data_root / "prepared/tasks/dr_grading" / f"{split}.jsonl"
                global_manifest = load_manifest(global_path, split)
                inputs.add(global_path)
                prediction_path = Path(legacy_prediction_root) / name / f"predictions_{split}.jsonl"
                all_exported = load_predictions(prediction_path, global_manifest)
                inputs.add(prediction_path)
                old_meta = load(prediction_path.with_suffix(".meta.json"), inputs)
                require(old_meta.get("manifest_sha256") == sha256_file(global_path)
                        and old_meta.get("jsonl_sha256") == sha256_file(prediction_path)
                        and old_meta.get("sample_count") == len(global_manifest)
                        and old_meta.get("source_counts") == dict(Counter(r["dataset"] for r in global_manifest.values())),
                        "Legacy export metadata differs from original global manifest")
                require(set(manifest).issubset(global_manifest), "Legacy global manifest does not contain the complete source subset")
                require(all(manifest[sid] == global_manifest[sid] for sid in manifest), "Legacy and isolated manifest records differ")
                computed = summarize_predictions(sorted(manifest), manifest, all_exported)
                checked = checked_matrix(report["metrics"]["by_dataset"][source], count, str(report_path))
                require(np.array_equal(computed["confusion_matrix"], checked["confusion_matrix"]), "Legacy source confusion differs from fixed raw predictions")
                scope = "Exact isolated source IDs, targets, content groups and fixed predictions verified against original global manifest; mixed aggregate excluded."
            models["legacy_" + name] = {"checkpoint": report.get("checkpoint"), "model_name": legacy_model, "training_sources": ["APTOS2019", "DDR", "EyeQ"],
                "training_history_sources": ["APTOS2019", "DDR", "EyeQ"], "selected_report_section": "metrics.by_dataset." + source,
                "comparison_scope": scope, "metrics": computed}
        results[split] = {"source": source, "n": count, "models": models, "manifest_path": str(path.resolve()),
                         "manifest_sha256": sha256_file(path), "new_sample_ids_and_targets_verified_against_manifest": True}
    references = {}
    if include_parent_reference:
        path = data_root / "prepared/source_isolated/strict_cross_domain/tasks/dr_grading/IDRiD/external_test.jsonl"
        manifest = strict_manifest(path, "external_test", "IDRiD", counts["external_test"], inputs)
        report, meta, _, metrics = checked_export(parent_root / "evaluation/head_best", "external_test", "IDRiD", manifest, inputs)
        validate_config(report.get("config", {}), "APTOS2019")
        require(meta.get("manifest_sha256") == sha256_file(path) and meta["checkpoint_completed_epoch"] == 5, "Parent reference epoch/manifest differs")
        references["aptos_parent_head_external_test"] = {"source": "IDRiD", "n": counts["external_test"], "checkpoint_completed_epoch": 5,
            "checkpoint_sha256": PARENT_SHA256, "training_sources": ["APTOS2019"], "metrics": metrics,
            "scope": "Fixed parent reference on the same IDRiD506; no APTOS354-to-DDR3898 comparison."}
    return {"format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": True,
        "new_model_names": {name: config["model"]["name"] for name in ("best", "head_best")},
        "splits": results, "checkpoint_relation": checkpoint_relation(metadata, agreements),
        "initialization_lineage": lineage, "lineage_evidence": lineage_evidence, "references": references,
        "metric_selection": "Fixed five-class argmax. No threshold fitting, test-driven checkpoint selection or resplitting.",
        "comparison_limits": ["New DDR manifests are single-source, but model supervision is sequential APTOS2019 -> DDR; this is not an independent DDR-only experiment.",
            "Historical ViT-S models trained on mixed APTOS2019/DDR/EyeQ; architecture, supervision history, sample count and validation selection differ. This is a historical-system comparison, not a causal model-size or source-isolation ablation.",
            "Only DDR3898 and IDRiD506 source-specific scores are compared. Mixed validation/test aggregates and APTOS354 are excluded.",
            "IDRiD has already been observed in this project; it is a fixed external benchmark, not a newly unseen clinical cohort. No tuning on this report.",
            "Exact declared image/group checks and source separation do not prove clinical patient independence or absence of every transformed near duplicate."],
        "input_files": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in sorted(inputs)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=PROJECT / "cloud_upload/ddr_from_aptos_20261113")
    parser.add_argument("--legacy-root", type=Path, default=PROJECT / "cloud_upload/evaluation")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--parent-run-root", type=Path, default=PROJECT / "cloud_upload/aptos_vitl_20261113")
    parser.add_argument("--checkpoint-metadata", type=Path)
    parser.add_argument("--legacy-prediction-root", type=Path,
                        default=PROJECT / "cloud_backup/20261113_201850_DR_training/contents/evaluation")
    parser.add_argument("--report-only-legacy", action="store_true", help="Explicitly limit legacy checking to source counts/class support, without raw predictions")
    parser.add_argument("--include-parent-reference", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.run_root, args.legacy_root, args.data_root, args.parent_run_root,
                       args.checkpoint_metadata, None if args.report_only_legacy else args.legacy_prediction_root,
                       args.include_parent_reference)
    destination = (args.output or args.run_root / "comparison_summary.json").resolve()
    require(destination not in {Path(item["path"]) for item in result["input_files"]}, "Output cannot overwrite an original input")
    require(not destination.exists() or destination == (args.run_root / "comparison_summary.json").resolve(),
            "Refusing to overwrite an existing report outside this run's generated comparison summary")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps({"all_checks_passed": True, "output": str(destination), "test_n": result["splits"]["test"]["n"],
                      "external_n": result["splits"]["external_test"]["n"]}))


if __name__ == "__main__":
    main()
