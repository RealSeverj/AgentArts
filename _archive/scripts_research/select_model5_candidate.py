"""Compare completed Model 5 candidates on the original validation cohort."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.screening_evaluation import load_predictions, read_json, write_json
from training.metrics import binary_metrics

import numpy as np

VALIDATION = "9f21c8e0cc337d6cf820f44329452fd3987fcf72487f1add227bbc8051fb379a"
BASELINE = "d518f2eac265cbab14763322c4b49a94eb9917a6880ef14ee256624a86f20a00"
LABELS = ["RVO", "LS", "DN", "AMD", "CSR", "OPTIC_DISC_CUPPING", "ODE", "ODP",
          "MYOPIC_FUNDUS", "TSLN", "ABNORMAL"]


def validation_summary(run):
    rows, meta, digest = load_predictions(run, "val")
    lock = read_json(run / "thresholds.json")
    complete = read_json(run / "complete.json")
    if (len(rows) != 634 or meta["labels"] != LABELS or meta["manifest_sha256"] != VALIDATION
            or lock["labels"] != LABELS or lock["val_manifest_sha256"] != VALIDATION
            or lock["val_predictions_sha256"] != digest or lock["checkpoint_sha256"] != meta["checkpoint_sha256"]
            or complete["checkpoint_sha256"] != meta["checkpoint_sha256"] or complete["epochs"] != 30
            or complete["best_epoch"] != meta["epoch"]):
        raise ValueError(f"Incompatible or incomplete validation artifacts in {run}")
    rows.sort(key=lambda r: r["sample_id"])
    cohort = [(r["sample_id"], r["dataset"], r["labels"], r["mask"]) for r in rows]
    per_label = {}
    for i, label in enumerate(LABELS):
        rule = lock["per_label"][label]
        if (rule["min_recall"], rule["max_fpr"]) != (0.9, 0.1):
            raise ValueError(f"Incompatible operating point for {label}")
        known = [r for r in rows if r["mask"][i]]
        y = np.asarray([r["labels"][i] for r in known])
        p = np.asarray([r["probabilities"][i] for r in known])
        result = binary_metrics(y, p, threshold=rule["threshold"])
        pred = p >= rule["threshold"]
        per_label[label] = dict(ap=result["auprc"], sensitivity=result["sensitivity"],
                                false_positives=int((pred & (y == 0)).sum()),
                                negatives=int((y == 0).sum()), meets_target=rule["status"] == "met")
    return cohort, dict(epoch=meta["epoch"], checkpoint_sha256=meta["checkpoint_sha256"],
                        macro_ap=float(np.mean([v["ap"] for v in per_label.values()])), per_label=per_label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--large-full", type=Path)
    parser.add_argument("--base-disc", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cohort, base = validation_summary(args.baseline)
    if base["checkpoint_sha256"] != BASELINE:
        parser.error("Unexpected baseline checkpoint")
    summary = {"baseline": base}
    for arm, path in (("large_full", args.large_full), ("base_disc", args.base_disc)):
        if path is None:
            continue
        run = read_json(path / "run.json")["identity"]
        wanted = ("vit_large_patch14_dinov2.lvd142m", "global") if arm == "large_full" else (
            "vit_base_patch14_dinov2.lvd142m", "global_disc_roi")
        if run["task"] != "grouped" or (run["backbone"], run["input_mode"]) != wanted:
            parser.error(f"Unexpected model or input views for {arm}")
        candidate_cohort, summary[arm] = validation_summary(path)
        if cohort != candidate_cohort:
            parser.error(f"Images or targets differ for {arm}")
    selected = max(summary, key=lambda arm: summary[arm]["macro_ap"])
    write_json(args.output, dict(selected=selected, selection_metric="validation_11_label_macro_ap",
               selection_order_for_ties=list(summary), validation_manifest_sha256=VALIDATION,
               results=summary), lock=True)
    print(f"{selected}  validation macro AP {summary[selected]['macro_ap']:.6f}")


if __name__ == "__main__":
    main()
