"""NumPy classification metrics with missing-target and content-weight support.

No test-set threshold fitting is performed. Undefined metrics are returned as
None, so missing classes cannot silently look like a measured score of zero.
"""
from __future__ import annotations

import numpy as np


def _average(values):
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(valid)) if valid else None


def binary_metrics(target, score, weight=None, threshold=0.5):
    y = np.asarray(target, dtype=np.int64)
    p = np.asarray(score, dtype=np.float64)
    w = np.ones(len(y), dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    if not (len(y) == len(p) == len(w)):
        raise ValueError("Target, probability, and weight counts differ")
    positive = float(w[y == 1].sum())
    negative = float(w[y == 0].sum())
    auroc, ap = None, None
    if len(y):
        # Sum tied scores at once. AUC assigns half credit to every tie.
        order = np.argsort(-p, kind="stable")
        ys, ps, ws = y[order], p[order], w[order]
        last = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]
        tp = np.cumsum(ws * (ys == 1))[last]
        fp = np.cumsum(ws * (ys == 0))[last]
        if positive > 0 and negative > 0:
            tpr, fpr = np.r_[0.0, tp / positive], np.r_[0.0, fp / negative]
            auroc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
        if positive > 0:
            recall = tp / positive
            ap = float(np.sum(np.diff(np.r_[0.0, recall]) * tp / np.maximum(tp + fp, 1e-15)))
    pred = p >= float(threshold)
    tp = float(w[(y == 1) & pred].sum())
    fp = float(w[(y == 0) & pred].sum())
    fn = float(w[(y == 1) & ~pred].sum())
    tn = float(w[(y == 0) & ~pred].sum())
    return {
        "n": int(len(y)), "positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
        "weight_sum": float(w.sum()), "auroc": auroc, "auprc": ap,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "threshold": float(threshold),
    }


def multiclass_metrics(target, score, num_classes, weight=None):
    y = np.asarray(target, dtype=np.int64)
    p = np.asarray(score, dtype=np.float64)
    w = np.ones(len(y), dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    pred = p.argmax(axis=1) if len(y) else np.empty(0, dtype=np.int64)
    if len(y):
        np.add.at(matrix, (y, pred), w)
    total = float(matrix.sum())
    if not total:
        return {"n": 0, "accuracy": None, "macro_f1": None, "qwk": None,
                "confusion_matrix": matrix.tolist()}
    tp = matrix.diagonal()
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    denominators = support + predicted
    f1 = np.divide(2 * tp, denominators, out=np.zeros(num_classes), where=denominators > 0)
    distance = (np.arange(num_classes)[:, None] - np.arange(num_classes)[None, :]) ** 2
    expected = np.outer(support, predicted) / total
    observed_disagreement = float((distance * matrix).sum())
    expected_disagreement = float((distance * expected).sum())
    return {"n": int(len(y)), "weight_sum": total,
            "accuracy": float(tp.sum() / total), "macro_f1": float(f1.mean()),
            "qwk": 1 - observed_disagreement / expected_disagreement if expected_disagreement > 0 else None,
            "mean_absolute_grade_error": float((w * np.abs(pred - y)).sum() / total),
            "per_class_f1": f1.tolist(), "confusion_matrix": matrix.tolist()}


def _summarize(targets, probabilities, mask, weights, config):
    if config["kind"] == "multiclass":
        available = mask.astype(bool).reshape(-1)
        return multiclass_metrics(targets.reshape(-1)[available], probabilities[available],
                                  int(config["num_classes"]), weights[available])
    evaluation = config.get("evaluation", {})
    thresholds = evaluation.get("thresholds", [evaluation.get("threshold", 0.5)] * len(config["labels"]))
    if len(thresholds) != len(config["labels"]):
        raise ValueError("evaluation.thresholds must follow the configured label order")
    per_label = {}
    for index, label in enumerate(config["labels"]):
        available = mask[:, index].astype(bool)
        per_label[label] = binary_metrics(targets[available, index], probabilities[available, index],
                                          weights[available], thresholds[index])
    return {"n": int(len(targets)), "per_label": per_label,
            "macro_auroc": _average(v["auroc"] for v in per_label.values()),
            "macro_auprc": _average(v["auprc"] for v in per_label.values()),
            "macro_f1": _average(v["f1"] for v in per_label.values()),
            "auroc_labels": sum(v["auroc"] is not None for v in per_label.values()),
            "auprc_labels": sum(v["auprc"] is not None for v in per_label.values())}


def classification_metrics(outputs, config):
    """Aggregate detached step outputs; duplicates from DDP padding are removed.

    Both ordinary and content-weighted scores are reported. The configured
    monitor refers to the ordinary score unless it explicitly selects a nested
    content_weighted key.
    """
    if not outputs:
        return {"n": 0}
    targets = np.concatenate([np.asarray(o["targets"]) for o in outputs], axis=0)
    probabilities = np.concatenate([np.asarray(o["probabilities"]) for o in outputs], axis=0)
    mask = np.concatenate([np.asarray(o["mask"], dtype=bool) for o in outputs], axis=0)
    ids = [s for o in outputs for s in o["sample_ids"]]
    sources = np.asarray([s for o in outputs for s in o["datasets"]], dtype=object)
    groups = [s for o in outputs for s in o.get("content_group_ids", o["sample_ids"])]
    weights = np.concatenate([np.asarray(o.get("evaluation_weights", np.ones(len(o["sample_ids"]))),
                                          dtype=np.float64) for o in outputs])
    seen, keep = set(), []
    for index, sample_id in enumerate(ids):
        if sample_id not in seen:
            seen.add(sample_id)
            keep.append(index)
    keep = np.asarray(keep, dtype=np.int64)
    targets, probabilities, mask, sources, weights = (v[keep] for v in (targets, probabilities, mask, sources, weights))
    if not np.isfinite(probabilities).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Nonfinite prediction or invalid evaluation weight")
    ordinary = np.ones(len(keep), dtype=np.float64)
    result = _summarize(targets, probabilities, mask, ordinary, config)
    result["content_weighted"] = _summarize(targets, probabilities, mask, weights, config)
    result["unique_content_groups"] = len({groups[index] for index in keep})
    result["by_dataset"] = {}
    for source in sorted(set(sources)):
        selected = sources == source
        source_result = _summarize(targets[selected], probabilities[selected], mask[selected], ordinary[selected], config)
        source_result["content_weighted"] = _summarize(targets[selected], probabilities[selected], mask[selected], weights[selected], config)
        result["by_dataset"][str(source)] = source_result
    return result
