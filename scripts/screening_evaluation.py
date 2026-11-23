"""Lock screening thresholds on validation predictions and evaluate a fixed checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.metrics import binary_metrics


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_file(path, payload, *, lock=False):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp.write(payload.encode("utf-8"))
        name = temp.name
    try:
        if lock:
            # Atomic creation avoids changing a published threshold after viewing a test result.
            try:
                os.link(name, path)
            except FileExistsError:
                if Path(name).read_bytes() != path.read_bytes():
                    raise ValueError(f"Threshold lock differs from existing file: {path}")
        else:
            os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path, document, *, lock=False):
    write_file(path, json.dumps(document, indent=2, allow_nan=False) + "\n", lock=lock)


def load_predictions(run, split):
    path = run / f"{split}_predictions.jsonl"
    meta = read_json(run / f"{split}_predictions.meta.json")
    labels = meta["labels"]
    if (meta["split"] != split or not labels or len(labels) != len(set(labels))
            or not meta["checkpoint_sha256"] or not meta["manifest_sha256"]):
        raise ValueError(f"Invalid prediction metadata: {path}")
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    seen = set()
    for row in rows:
        key = (row["dataset"], row["sample_id"])
        lengths = [len(row[k]) for k in ("labels", "mask", "logits", "probabilities")]
        if (row["split"] != split or row["checkpoint_sha256"] != meta["checkpoint_sha256"]
                or key in seen or not all(key) or lengths != [len(labels)] * 4):
            raise ValueError(f"Prediction identity or dimensions differ: {key}")
        probs = np.asarray(row["probabilities"], dtype=float)
        if (not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1))
                or not np.isfinite(row["logits"]).all()
                or not all(type(m) is bool for m in row["mask"])
                or any(y not in (0, 1) for y, m in zip(row["labels"], row["mask"]) if m)):
            raise ValueError(f"Invalid scores, mask or targets: {key}")
        seen.add(key)
    if not rows:
        raise ValueError(f"No predictions: {path}")
    return rows, meta, hashlib.sha256(raw).hexdigest()


def known_arrays(rows, idx):
    known = [r for r in rows if r["mask"][idx]]
    identities = []
    for row in known:
        if row.get("patient_id") not in (None, ""):
            unit = ("patient", str(row["patient_id"]))
        elif row.get("content_group_id") not in (None, ""):
            unit = ("content", str(row["content_group_id"]))
        else:
            raise ValueError(f"No patient or content group for {row['sample_id']}")
        identities.append((row["dataset"], *unit))
    codes = {identity: i for i, identity in enumerate(sorted(set(identities)))}
    return (np.array([r["labels"][idx] for r in known], dtype=np.int64),
            np.array([r["probabilities"][idx] for r in known], dtype=float),
            np.array([codes[item] for item in identities], dtype=np.int64))


def validate_goal(goal):
    if not (0 < goal["min_recall"] <= 1 and 0 <= goal["max_fpr"] <= 1):
        raise ValueError("Policy requires 0 < min_recall <= 1 and 0 <= max_fpr <= 1")
    return {"min_recall": float(goal["min_recall"]), "max_fpr": float(goal["max_fpr"])}


def choose_threshold(y, p, goal):
    positive, negative = int(y.sum()), int((y == 0).sum())
    result = dict(goal, validation_positive=positive, validation_negative=negative)
    if not positive or not negative:
        return dict(result, threshold=None, status="insufficient_validation_classes")
    order = np.argsort(-p, kind="stable")
    last = np.r_[np.flatnonzero(np.diff(p[order])), len(y) - 1]
    tp = np.cumsum(y[order])[last]
    fp = last + 1 - tp
    candidates = np.flatnonzero(tp / positive >= goal["min_recall"])
    # Among thresholds meeting recall, minimize false alarms, then maximize
    # true positives, then use the highest threshold for deterministic ties.
    best = min(candidates, key=lambda i: (fp[i], -tp[i], -p[order[last[i]]]))
    return dict(result, threshold=float(p[order[last[best]]]), validation_tp=int(tp[best]),
                validation_fp=int(fp[best]), status=("met" if fp[best] / negative <= goal["max_fpr"]
                                                     else "false_positive_limit_exceeded"))


def lock_thresholds(args):
    rows, meta, digest = load_predictions(args.run, "val")
    labels = meta["labels"]
    if args.policy:
        if args.min_recall is not None or args.max_fpr is not None:
            raise ValueError("Provide either a per-label policy or shared targets")
        policy = read_json(args.policy)
        if set(policy) != set(labels):
            raise ValueError("Policy keys must match all label names in the prediction metadata")
    else:
        if args.min_recall is None or args.max_fpr is None:
            raise ValueError("Specify both --min-recall and --max-fpr, or --policy")
        policy = {name: {"min_recall": args.min_recall, "max_fpr": args.max_fpr} for name in labels}
    results = {}
    for i, name in enumerate(labels):
        y, p, _ = known_arrays(rows, i)
        results[name] = choose_threshold(y, p, validate_goal(policy[name]))
    lock = dict(labels=labels, checkpoint_sha256=meta["checkpoint_sha256"],
                val_manifest_sha256=meta["manifest_sha256"], val_predictions_sha256=digest,
                rule="min FP among recall-qualified thresholds, then max TP, then highest threshold; p >= t",
                per_label=results)
    write_json(args.run / "thresholds.json", lock, lock=True)


RATES = ("ap", "auroc", "sensitivity", "specificity", "precision", "f1")


def metrics(y, p, threshold, weights=None):
    # Reuse the same tied-score AP/AUC definitions as model selection.
    weighted = np.ones(len(y), dtype=int) if weights is None else weights
    rank = binary_metrics(y, p, weight=weighted, threshold=threshold if threshold is not None else 0.5)
    both = weighted[y == 0].sum() > 0 and weighted[y == 1].sum() > 0
    out = dict(ap=rank["auprc"] if both else None, auroc=rank["auroc"])
    if threshold is None:
        return dict(out, **{k: None for k in ("tp", "fp", "tn", "fn", *RATES[2:])})
    pred = p >= threshold
    tp = int(weighted[pred & (y == 1)].sum())
    fp = int(weighted[pred & (y == 0)].sum())
    fn = int(weighted[~pred & (y == 1)].sum())
    tn = int(weighted[~pred & (y == 0)].sum())
    return dict(out, tp=tp, fp=fp, fn=fn, tn=tn, sensitivity=rank["sensitivity"],
                specificity=rank["specificity"], precision=tp / (tp + fp) if tp + fp else None,
                f1=rank["f1"] if tp + fn else None)


def intervals(y, p, groups, threshold, repeats, rng):
    values = {name: [] for name in RATES}
    n_groups = len(set(groups))
    if n_groups > 1:
        for _ in range(repeats):
            selected = rng.integers(n_groups, size=n_groups)
            weights = np.bincount(selected, minlength=n_groups)[groups]
            draw = metrics(y, p, threshold, weights)
            for name in RATES:
                if draw[name] is not None:
                    values[name].append(draw[name])
    return {k: {"valid_draws": len(v), "interval": np.quantile(v, [0.025, 0.975]).tolist()
                if len(v) >= 2 else None} for k, v in values.items()}


def describe(rows, lock, repeats, seed):
    output = {}
    for idx, name in enumerate(lock["labels"]):
        y, p, groups = known_arrays(rows, idx)
        rule = lock["per_label"][name]
        t = rule["threshold"]
        output[name] = dict(positive=int(y.sum()), negative=int((y == 0).sum()),
                            unknown=len(rows) - len(y), groups=int(len(set(groups))),
                            positive_groups=int(len(set(groups[y == 1]))), threshold=t,
                            validation_status=rule["status"], locked=metrics(y, p, t),
                            at_0_5=metrics(y, p, 0.5),
                            ci95=intervals(y, p, groups, t, repeats, np.random.default_rng(seed)))
    defined = {k: item["locked"]["ap"] for k, item in output.items() if item["locked"]["ap"] is not None}
    return dict(images=len(rows), macro_ap_labels=list(defined),
                macro_ap=float(np.mean(list(defined.values()))) if defined else None, per_label=output)


def report(args):
    lock = read_json(args.run / "thresholds.json")
    _, _, val_hash = load_predictions(args.run, "val")
    if val_hash != lock["val_predictions_sha256"]:
        raise ValueError("Validation predictions have changed since thresholds were locked")
    if args.bootstrap < 2:
        raise ValueError("Specify at least two bootstrap draws")
    for split in args.split:
        rows, meta, digest = load_predictions(args.run, split)
        if lock["checkpoint_sha256"] != meta["checkpoint_sha256"] or lock["labels"] != meta["labels"]:
            raise ValueError("The threshold lock and prediction checkpoint or label order differ")
        summary = describe(rows, lock, args.bootstrap, args.seed)
        by_source = {}
        for source in sorted({r["dataset"] for r in rows}):
            subset = [r for r in rows if r["dataset"] == source]
            by_source[source] = summary if len(subset) == len(rows) else describe(subset, lock, args.bootstrap, args.seed)
        write_json(args.run / f"{split}_metrics.json",
                   dict(split=split, checkpoint_sha256=meta["checkpoint_sha256"],
                        manifest_sha256=meta["manifest_sha256"], predictions_sha256=digest,
                        val_predictions_sha256=val_hash, group_bootstrap_repeats=args.bootstrap,
                        group_bootstrap_seed=args.seed, resampling_unit="patient when known, else content group",
                        interval="percentile, conditional on fixed model and thresholds",
                        overall=summary, by_dataset=by_source))
        decisions = []
        for row in rows:
            calls = []
            for label, probability in zip(lock["labels"], row["probabilities"]):
                t = lock["per_label"][label]["threshold"]
                calls.append(None if t is None else int(probability >= t))
            decisions.append(json.dumps(dict(row, decisions=calls), ensure_ascii=False, allow_nan=False))
        write_file(args.run / f"{split}_decisions.jsonl", "\n".join(decisions) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("lock", help="Lock work points using validation predictions only")
    fit.add_argument("--run", required=True, type=Path)
    fit.add_argument("--policy", type=Path, help='JSON keyed by label, each with min_recall and max_fpr')
    fit.add_argument("--min-recall", type=float)
    fit.add_argument("--max-fpr", type=float)
    evaluate = commands.add_parser("report", help="Report operating metrics and grouped intervals")
    evaluate.add_argument("--run", required=True, type=Path)
    evaluate.add_argument("--split", choices=("val", "test", "external_test"), nargs="+", required=True)
    evaluate.add_argument("--bootstrap", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    if args.command == "lock":
        lock_thresholds(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
