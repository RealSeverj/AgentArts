"""Audit paired DR test exports and summarize fixed predictions without tuning.

Input layout: <evaluation-root>/{baseline,finetuned}/evaluation_<split>.json
and predictions_<split>.jsonl, for test and external_test. This script never
fits a threshold, selects a checkpoint, or modifies the original evaluations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

MODELS = ("baseline", "finetuned")
EXPECTED_COUNTS = {"test": 20501, "external_test": 506}
GRADES = 5


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grade(value, context):
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < GRADES:
        raise ValueError(f"Invalid DR grade in {context}: {value!r}")
    return value


def divide(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def confusion(targets, predictions, weights=None):
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    return np.bincount(targets * GRADES + predictions, weights=weights,
                       minlength=GRADES * GRADES).reshape(GRADES, GRADES).astype(np.float64)


def qwk(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    total = float(matrix.sum())
    if not total:
        return None
    distance = (np.arange(GRADES)[:, None] - np.arange(GRADES)[None, :]) ** 2
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / total
    denominator = float((expected * distance).sum())
    return 1.0 - float((matrix * distance).sum()) / denominator if denominator > 0 else None


def matrix_summary(matrix, n=None):
    """Rows are true grades and columns predicted grades; no threshold fitting."""
    matrix = np.asarray(matrix, dtype=np.float64)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    total = float(matrix.sum())
    diagonal = matrix.diagonal()
    denominator = support + predicted
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros(GRADES), where=denominator > 0)
    distance = np.abs(np.arange(GRADES)[:, None] - np.arange(GRADES)[None, :])
    binary_tp = float(matrix[2:, 2:].sum())
    binary_fn = float(matrix[2:, :2].sum())
    binary_tn = float(matrix[:2, :2].sum())
    binary_fp = float(matrix[:2, 2:].sum())
    severe_denominator = float(matrix[3:, :].sum())
    severe_numerator = float(matrix[3:, :2].sum())
    return {
        "n": int(n if n is not None else round(total)), "weight_sum": total,
        "qwk": qwk(matrix), "accuracy": divide(float(diagonal.sum()), total),
        "macro_f1": float(f1.mean()) if total else None,
        "mean_abs_grade_error": divide(float((matrix * distance).sum()), total),
        "confusion_matrix": matrix.tolist(), "target_support": support.tolist(),
        "recall_by_grade": [divide(diagonal[i], support[i]) for i in range(GRADES)],
        "predicted_counts": predicted.tolist(),
        "predicted_fraction": [divide(value, total) for value in predicted],
        "per_class_f1": f1.tolist(), "grade4_recall": divide(diagonal[4], support[4]),
        "absolute_error_counts": {str(i): float(matrix[distance == i].sum()) for i in range(GRADES)},
        "severe_underestimation": {"definition": "true grade 3 or 4, predicted grade 0 or 1",
                                   "numerator": severe_numerator, "denominator": severe_denominator,
                                   "rate": divide(severe_numerator, severe_denominator)},
        "grade_ge_2_binary": {"definition": "positive iff five-class argmax grade >= 2; fixed class merge, no fitted threshold",
                              "tp": binary_tp, "fn": binary_fn, "tn": binary_tn, "fp": binary_fp,
                              "sensitivity": divide(binary_tp, binary_tp + binary_fn),
                              "specificity": divide(binary_tn, binary_tn + binary_fp)},
    }


def load_manifest(path, expected_split, expected_count=None):
    rows = {}
    with Path(path).open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            context = f"{path}:{number}"
            if row.get("task") != "dr_grading" or row.get("split") != expected_split:
                raise ValueError(f"Manifest task/split mismatch: {context}")
            if not row.get("label_mask", {}).get("dr_grade", False):
                continue
            identifier = row.get("sample_id")
            if not isinstance(identifier, str) or not identifier or identifier in rows:
                raise ValueError(f"Invalid or repeated manifest sample_id: {context}")
            grade(row.get("labels", {}).get("dr_grade"), context)
            if not isinstance(row.get("dataset"), str) or not row["dataset"]:
                raise ValueError(f"Missing manifest dataset: {context}")
            rows[identifier] = row
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Manifest {expected_split} supervised count {len(rows)} differs from required {expected_count}")
    return rows


def load_predictions(path, manifest):
    predictions = {}
    required = {"sample_id", "dataset", "target", "predicted_class", "probabilities", "content_group_id", "evaluation_weight"}
    with Path(path).open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            context = f"{path}:{number}"
            if required - set(row):
                raise ValueError(f"Prediction fields missing in {context}: {sorted(required - set(row))}")
            identifier = row["sample_id"]
            if not isinstance(identifier, str) or identifier in predictions:
                raise ValueError(f"Invalid or repeated prediction sample_id: {context}")
            if identifier not in manifest:
                raise ValueError(f"Prediction sample_id is absent from manifest: {identifier}")
            authoritative = manifest[identifier]
            actual_target = grade(row["target"], context)
            actual_pred = grade(row["predicted_class"], context)
            if actual_target != authoritative["labels"]["dr_grade"] or row["dataset"] != authoritative["dataset"]:
                raise ValueError(f"Prediction target/dataset differs from manifest: {identifier}")
            probabilities = np.asarray(row["probabilities"], dtype=np.float64)
            if (probabilities.shape != (GRADES,) or not np.isfinite(probabilities).all() or
                    (probabilities < 0).any() or (probabilities > 1).any() or
                    not np.isclose(probabilities.sum(), 1.0, atol=1e-5, rtol=0)):
                raise ValueError(f"Invalid five-class probabilities: {context}")
            if actual_pred != int(probabilities.argmax()):
                raise ValueError(f"predicted_class differs from probability argmax: {context}")
            weight = row["evaluation_weight"]
            expected_weight = authoritative.get("content_group_evaluation_weight") or 1.0
            if (not isinstance(weight, (float, int)) or isinstance(weight, bool) or
                    not math.isfinite(weight) or weight <= 0 or not math.isclose(weight, expected_weight, rel_tol=0, abs_tol=1e-12)):
                raise ValueError(f"Prediction evaluation_weight differs from manifest: {identifier}")
            content_group = authoritative.get("content_group_id") or identifier
            if row["content_group_id"] != content_group:
                raise ValueError(f"Prediction content_group_id differs from manifest: {identifier}")
            predictions[identifier] = row
    if set(predictions) != set(manifest):
        missing = sorted(set(manifest) - set(predictions))
        raise ValueError(f"Prediction IDs do not exactly cover the manifest; missing {len(missing)}: {missing[:5]}")
    return predictions


def assert_report_matches(report, split, overall, by_source, context):
    if report.get("split") != split:
        raise ValueError(f"Evaluation report split mismatch: {context}")
    recorded = report.get("metrics")
    if not isinstance(recorded, dict) or recorded.get("n") != overall["n"]:
        raise ValueError(f"Evaluation report metrics.n differs from prediction/manifest count: {context}")

    def check(left, right, location):
        for source_key, calculated_key in (("qwk", "qwk"), ("accuracy", "accuracy"), ("macro_f1", "macro_f1"),
                                           ("mean_absolute_grade_error", "mean_abs_grade_error")):
            if source_key in left:
                observed, expected = left[source_key], right[calculated_key]
                if observed is None or expected is None:
                    if observed != expected:
                        raise ValueError(f"Undefined reported metric differs at {location}.{source_key}")
                elif not math.isclose(float(observed), expected, rel_tol=0, abs_tol=1e-7):
                    raise ValueError(f"Reported metric differs from predictions at {location}.{source_key}")
        if "confusion_matrix" in left:
            matrix = np.asarray(left["confusion_matrix"], dtype=np.float64)
            if matrix.shape != (GRADES, GRADES) or not np.allclose(matrix, right["confusion_matrix"], rtol=0, atol=1e-8):
                raise ValueError(f"Reported confusion matrix differs from predictions: {location}")
        if "content_weighted" in left:
            check(left["content_weighted"], right["content_weighted"], location + ".content_weighted")

    check(recorded, overall, context + ".metrics")
    recorded_sources = recorded.get("by_dataset", {})
    if set(recorded_sources) != set(by_source):
        raise ValueError(f"Reported source coverage differs from predictions: {context}")
    for source, calculated in by_source.items():
        if recorded_sources[source].get("n") != calculated["n"]:
            raise ValueError(f"Reported source count differs from predictions: {context}/{source}")
        check(recorded_sources[source], calculated, context + "/" + source)


def summarize_predictions(identifiers, manifest, predictions):
    targets = np.asarray([manifest[key]["labels"]["dr_grade"] for key in identifiers])
    predicted = np.asarray([predictions[key]["predicted_class"] for key in identifiers])
    weights = np.asarray([predictions[key]["evaluation_weight"] for key in identifiers], dtype=np.float64)
    result = matrix_summary(confusion(targets, predicted), len(identifiers))
    result["content_weighted"] = matrix_summary(confusion(targets, predicted, weights), len(identifiers))
    return result


def group_key(row):
    for field in ("leakage_group_id", "content_group_id"):
        if isinstance(row.get(field), str) and row[field]:
            return field, row[field]
    return "sample_id", row["sample_id"]


def paired_cluster_bootstrap(manifest, baseline, finetuned, replicates=1000, seed=20261113):
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    strata = defaultdict(dict)
    group_fields, cross_source_groups = Counter(), defaultdict(set)
    known_patients = 0
    for identifier, row in manifest.items():
        field, key = group_key(row)
        source = row["dataset"]
        group_fields[field] += 1
        cross_source_groups[(field, key)].add(source)
        known_patients += bool(row.get("patient_id"))
        identity = (field, key)
        if identity not in strata[source]:
            strata[source][identity] = np.zeros((2, GRADES, GRADES), dtype=np.int64)
        y = row["labels"]["dr_grade"]
        strata[source][identity][0, y, baseline[identifier]["predicted_class"]] += 1
        strata[source][identity][1, y, finetuned[identifier]["predicted_class"]] += 1
    matrices = {source: np.stack(list(groups.values())) for source, groups in sorted(strata.items())}
    deltas, source_deltas = [], {source: [] for source in matrices}
    rng = np.random.default_rng(seed)
    point_matrices = np.zeros((2, GRADES, GRADES), dtype=np.int64)
    source_points = {}
    for source, values in matrices.items():
        total = values.sum(axis=0)
        point_matrices += total
        first, second = qwk(total[0]), qwk(total[1])
        source_points[source] = second - first if first is not None and second is not None else None
    for _ in range(replicates):
        overall = np.zeros((2, GRADES, GRADES), dtype=np.int64)
        for source, values in matrices.items():
            # Draw the source's original number of groups, with replacement;
            # preserve every member of a drawn group and pair both models.
            selected = rng.integers(0, len(values), size=len(values))
            sampled = values[selected].sum(axis=0)
            overall += sampled
            first, second = qwk(sampled[0]), qwk(sampled[1])
            if first is not None and second is not None:
                source_deltas[source].append(second - first)
        first, second = qwk(overall[0]), qwk(overall[1])
        if first is not None and second is not None:
            deltas.append(second - first)

    def interval(values, point):
        return {"point_delta_full_minus_head": point,
                "ci95_percentile": [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else None,
                "bootstrap_mean_delta": float(np.mean(values)) if values else None,
                "valid_replicates": len(values), "undefined_replicates": replicates - len(values)}

    first, second = qwk(point_matrices[0]), qwk(point_matrices[1])
    return {"replicates": replicates, "seed": seed, "method": "paired source-stratified cluster percentile bootstrap",
            "metric": "unweighted sample-level QWK difference, finetuned minus baseline",
            "resampling_unit": "Within each dataset/source, sample leakage groups with replacement; use content groups or individual sample IDs only when the preceding identifier is unavailable",
            "row_group_identifier_counts": dict(group_fields),
            "groups_by_dataset": {source: len(values) for source, values in matrices.items()},
            "known_patient_id_rows": known_patients, "rows_without_patient_id": len(manifest) - known_patients,
            "groups_spanning_multiple_sources": sum(len(values) > 1 for values in cross_source_groups.values()),
            "limitations": ["A leakage/content group is not established to be an independent clinical patient.",
                            "Sources are resampled separately; groups spanning multiple sources are not coupled across source strata.",
                            "Intervals describe these fixed exported predictions and do not include retraining or checkpoint-selection uncertainty.",
                            "Undefined QWK replicates (zero expected disagreement) are excluded and explicitly counted."],
            "all": interval(deltas, second - first if first is not None and second is not None else None),
            "by_dataset": {source: interval(source_deltas[source], source_points[source]) for source in matrices}}


def analyze(evaluation_root, data_root, bootstrap_replicates=1000, seed=20261113, expected_counts=None):
    evaluation_root, data_root = Path(evaluation_root).resolve(), Path(data_root).resolve()
    expected_counts = EXPECTED_COUNTS if expected_counts is None else expected_counts
    task_root = data_root / "prepared/tasks/dr_grading"
    training = load_manifest(task_root / "train.jsonl", "train")
    training_counts = Counter(row["labels"]["dr_grade"] for row in training.values())
    if not training_counts:
        raise ValueError("Training manifest cannot establish a majority-class comparator")
    majority = min(training_counts, key=lambda value: (-training_counts[value], value))
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "purpose": "Descriptive comparison of already fixed model predictions; no test-based threshold fitting or model selection",
               "grade_order": list(range(GRADES)), "confusion_matrix_axes": {"rows": "true grade", "columns": "predicted grade"},
               "models": list(MODELS), "training_majority_class": majority,
               "training_class_counts": {str(i): training_counts[i] for i in range(GRADES)},
               "splits": {}, "input_files": []}
    input_paths = [task_root / "train.jsonl"]
    for split_index, (split, expected) in enumerate(expected_counts.items()):
        manifest_path = task_root / (split + ".jsonl")
        manifest = load_manifest(manifest_path, split, expected)
        identifiers = sorted(manifest)
        by_source_ids = defaultdict(list)
        for identifier in identifiers:
            by_source_ids[manifest[identifier]["dataset"]].append(identifier)
        models, reports, predictions = {}, {}, {}
        input_paths.append(manifest_path)
        for model in MODELS:
            report_path = evaluation_root / model / f"evaluation_{split}.json"
            prediction_path = evaluation_root / model / f"predictions_{split}.jsonl"
            reports[model] = read_json(report_path)
            predictions[model] = load_predictions(prediction_path, manifest)
            overall = summarize_predictions(identifiers, manifest, predictions[model])
            source_metrics = {source: summarize_predictions(ids, manifest, predictions[model])
                              for source, ids in sorted(by_source_ids.items())}
            assert_report_matches(reports[model], split, overall, source_metrics, str(report_path))
            models[model] = {"all": overall, "by_dataset": source_metrics}
            input_paths.extend((report_path, prediction_path))
        for identifier in identifiers:
            a, b = (predictions[model][identifier] for model in MODELS)
            for key in ("sample_id", "target", "dataset", "evaluation_weight", "content_group_id"):
                if a[key] != b[key]:
                    raise ValueError(f"Paired predictions disagree in {key}: {identifier}")
        constant = {identifier: {"predicted_class": majority,
                                  "evaluation_weight": predictions["baseline"][identifier]["evaluation_weight"]}
                    for identifier in identifiers}
        constant_metrics = {"predicted_class": majority,
                            "selection_source": "training-label majority only; no evaluation-label selection",
                            "all": summarize_predictions(identifiers, manifest, constant),
                            "by_dataset": {source: summarize_predictions(ids, manifest, constant)
                                           for source, ids in sorted(by_source_ids.items())}}
        summary["splits"][split] = {"audited_count": len(manifest), "required_manifest_count": expected,
                                     "dataset_counts": {key: len(value) for key, value in sorted(by_source_ids.items())},
                                     "models": models, "training_majority_baseline": constant_metrics,
                                     "paired_qwk_bootstrap": paired_cluster_bootstrap(manifest, predictions["baseline"], predictions["finetuned"],
                                                                                       bootstrap_replicates, seed + split_index),
                                     "original_evaluation_reports": reports}
        print(json.dumps({"event": "evaluation_split_analyzed", "split": split, "n": len(manifest),
                          "baseline_qwk": models["baseline"]["all"]["qwk"],
                          "finetuned_qwk": models["finetuned"]["all"]["qwk"]}, ensure_ascii=False), flush=True)
    for path in input_paths:
        summary["input_files"].append({"path": str(path), "sha256": sha256_file(path)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")
    report = analyze(args.evaluation_root, args.data_root, args.bootstrap_replicates, args.seed)
    destination = args.output.resolve()
    if destination in {Path(item["path"]) for item in report["input_files"]}:
        raise ValueError("Analysis output must not overwrite an original report, prediction export, or manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps({"success": True, "output": str(destination), "bootstrap_replicates": args.bootstrap_replicates}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
