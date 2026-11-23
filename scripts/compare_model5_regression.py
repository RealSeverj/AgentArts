"""Pair the validation-selected Model 5 with the historical run on identical held-out eyes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.screening_evaluation import load_predictions, metrics, read_json, write_json
from scripts.select_model5_candidate import BASELINE, LABELS, VALIDATION


MEASURES = ("ap", "sensitivity", "specificity", "tp", "fp")
REFERENCE = {
    "test": ("RFMiD", 615, "5ede17f3f694600ee8eb46688546c2806044f05929dd3e49209fc84f80abe1b4"),
    "external_test": ("RFMiD2", 848, "26b81c4be057558a6c590c35623c928da21f7a021ac08b441d50ec80d19bb103"),
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read_lock(run):
    rows, meta, val_digest = load_predictions(run, "val")
    lock = read_json(run / "thresholds.json")
    require(len(rows) == 634 and meta["labels"] == lock["labels"] == LABELS
            and meta["manifest_sha256"] == lock["val_manifest_sha256"] == VALIDATION
            and meta["checkpoint_sha256"] == lock["checkpoint_sha256"]
            and val_digest == lock["val_predictions_sha256"], "Validation threshold identity differs: " + str(run))
    return lock


def confirm_selection(receipt, selected, selected_dir):
    require(receipt["validation_manifest_sha256"] == VALIDATION, "Unrecognized selection cohort")
    if "winner" in receipt:
        require(receipt["selection_split"] == "val", "Winner must be chosen on validation")
        winner = receipt["winner"]
        if receipt["ensemble_selected"]:
            expected = receipt["ensemble_recipe_sha256"]
            require(hashlib.sha256((selected_dir / "thresholds.json").read_bytes()).hexdigest()
                    == receipt["ensemble_threshold_lock_sha256"], "Ensemble threshold file differs")
        else:
            provenance = receipt["validation_predictions"][winner]
            expected = provenance["checkpoint_sha256"]
            require(provenance["predictions_sha256"] == selected["val_predictions_sha256"],
                    "Winner validation scores differ from selection receipt")
    else:
        require(receipt["selection_metric"] == "validation_11_label_macro_ap", "Unrecognized selection metric")
        winner = receipt["selected"]
        winning = receipt["results"][winner]
        expected = winning["checkpoint_sha256"]
        if receipt.get("format") == "model5_final_validation_selection_v1":
            require(receipt["selection_split"] == "val"
                    and selected["val_predictions_sha256"] == winning["val_predictions_sha256"]
                    and hashlib.sha256((selected_dir / "thresholds.json").read_bytes()).hexdigest()
                    == winning["thresholds_sha256"], "Final winner validation predictions or thresholds differ")
    require(selected["checkpoint_sha256"] == expected, "Supplied scores are not from the validation winner")
    return winner


def interval(values):
    valid = np.asarray(values)
    valid = valid[np.isfinite(valid)]
    return {"valid_draws": len(valid), "interval": np.quantile(valid, [0.025, 0.975]).tolist()
            if len(valid) >= 2 else None}


def finite(value):
    return float(value) if np.isfinite(value) else None


def comparison(point, draws, metric):
    old, new = point
    values = [finite(v) for v in (old, new, new - old)]
    if metric in ("tp", "fp"):
        values = [int(v) if v is not None else None for v in values]
    return dict(zip(("baseline", "selected", "delta"), values), paired_delta_ci95=interval(draws))


def evaluate_split(paths, locks, split, repeats, seed):
    dataset, n, expected_manifest = REFERENCE[split]
    aligned, digests = [], []
    for path, lock in zip(paths, locks):
        data, meta, digest = load_predictions(path, split)
        require(len(data) == n and {r["dataset"] for r in data} == {dataset}
                and meta["manifest_sha256"] == expected_manifest and meta["labels"] == LABELS
                and meta["checkpoint_sha256"] == lock["checkpoint_sha256"], "Held-out cohort or model differs")
        aligned.append(sorted(data, key=lambda r: r["sample_id"]))
        digests.append(digest)
    rows = aligned[0]
    fields = ("sample_id", "image_path", "labels", "mask", "patient_id", "content_group_id")
    for left, right in zip(*aligned):
        require(all(left[k] == right[k] for k in fields), "Unpaired image/labels: " + left["sample_id"])
        if "image_sha256" in left and "image_sha256" in right:
            require(left["image_sha256"] == right["image_sha256"], "Different original image bytes")
    groups = []
    for r in rows:
        require(r["patient_id"] or r["content_group_id"], "Missing both patient and content-group identity")
        groups.append(("patient:" + r["patient_id"]) if r["patient_id"] else ("content:" + r["content_group_id"]))
    distinct, indexes = np.unique(groups, return_inverse=True)
    label_names = LABELS if split == "test" else LABELS[:-1]
    masks = np.asarray([r["mask"] for r in rows], dtype=bool)
    require(masks[:, :len(label_names)].all() and (split == "test" or not masks[:, -1].any()),
            "Unexpected supervision on the pinned release")
    truth = np.asarray([r["labels"] for r in rows], dtype=int)
    probs = np.asarray([[r["probabilities"] for r in predictions] for predictions in aligned], dtype=float)
    cuts = [[lock["per_label"][key]["threshold"] for key in label_names] for lock in locks]

    def calculate(weights):
        values = np.empty((2, len(label_names), len(MEASURES)))
        for side in range(2):
            for i, _ in enumerate(label_names):
                point = metrics(truth[:, i], probs[side, :, i], cuts[side][i], weights)
                values[side, i] = [point[m] for m in MEASURES]
        return values

    point = calculate(np.ones(n, dtype=int))
    deltas = np.empty((repeats, len(label_names), len(MEASURES)))
    rng = np.random.default_rng(seed)
    for draw in range(repeats):
        sampled = rng.integers(len(distinct), size=len(distinct))
        weights = np.bincount(sampled, minlength=len(distinct))[indexes]
        baseline, selected = calculate(weights)
        deltas[draw] = selected - baseline
    per_label = {}
    for i, name in enumerate(label_names):
        per_label[name] = {"positive": int(truth[:, i].sum()), "negative": int(n - truth[:, i].sum()),
                           "thresholds": {"baseline": cuts[0][i], "selected": cuts[1][i]},
                           **{m: comparison(point[:, i, j], deltas[:, i, j], m) for j, m in enumerate(MEASURES)}}
    scopes = {"comparable_10": len(LABELS) - 1}
    if split == "test":
        scopes["all_11"] = len(LABELS)
    macro = {}
    for scope, size in scopes.items():
        # Keep label membership fixed in every draw; a missing class invalidates that macro draw.
        macro[scope] = dict(labels=label_names[:size], **comparison(point[:, :size, 0].mean(axis=1),
                            deltas[:, :size, 0].mean(axis=1), "ap"))
    return dict(dataset=dataset, images=n, groups=len(distinct), manifest_sha256=expected_manifest,
                predictions_sha256=dict(zip(("baseline", "selected"), digests)), macro_ap=macro, per_label=per_label)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--selected", type=Path, required=True)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--selection", type=Path, help="Validation-only winner receipt")
    choice.add_argument("--identity-check", action="store_true", help="Only compare historical run to itself")
    p.add_argument("--output", type=Path, help="Write JSON to this path; otherwise print to stdout")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261114)
    args = p.parse_args()
    require(args.bootstrap >= 2, "At least two paired bootstrap draws are required")
    paths = (args.baseline, args.selected)
    locks = [read_lock(path) for path in paths]
    require(locks[0]["checkpoint_sha256"] == BASELINE, "Historical Model 5 checkpoint differs")
    if args.identity_check:
        require(paths[0].resolve() == paths[1].resolve(), "Identity verification requires the same baseline directory")
        winner = "historical_baseline_identity_check"
    else:
        winner = confirm_selection(read_json(args.selection), locks[1], paths[1])
    result = dict(winner=winner, selection_sha256=hashlib.sha256(args.selection.read_bytes()).hexdigest()
                  if args.selection else None, difference="selected minus baseline", paired_bootstrap_draws=args.bootstrap,
                  seed=args.seed, interval="Paired patient/content-group percentile 95%; fixed thresholds and scores.",
                  score_identities={role: {k:lock[k] for k in ("checkpoint_sha256", "val_predictions_sha256")}
                                    for role, lock in zip(("baseline", "selected"), locks)},
                  threshold_files_sha256={k: hashlib.sha256((path / "thresholds.json").read_bytes()).hexdigest()
                                          for k, path in zip(("baseline", "selected"), paths)},
                  splits={s: evaluate_split(paths, locks, s, args.bootstrap, args.seed) for s in REFERENCE})
    if args.output:
        source_names = {"thresholds.json", *(f"{s}_predictions{ext}" for s in ("val", *REFERENCE)
                                          for ext in (".jsonl", ".meta.json"))}
        inputs = {(path / name).resolve() for path in paths for name in source_names}
        if args.selection:
            inputs.add(args.selection.resolve())
        require(args.output.resolve() not in inputs, "Output must be separate from the input predictions and locks")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
