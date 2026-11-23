"""Run the archived RFMiD classifier on complete matched splits and compare four common labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.screening_evaluation import (choose_threshold, known_arrays, load_predictions, metrics,
                                          read_json, write_file, write_json)


LEGACY_SHA256 = "f5352ccb0834b41a2d9e6ed6d43c80b7b7b2e87dcfa1c0b3690f96eddafca640"
LEGACY_LABELS = ("DR", "AMD", "MYOPIC_FUNDUS", "OPTIC_DISC_CUPPING", "ABNORMAL")
COMMON = LEGACY_LABELS[1:]


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RetinalImages:
    """Only image pixels enter the archived classifier."""

    def __init__(self, rows, root, config):
        from training.classification import ClassificationTransform
        self.rows = rows
        self.root = root.resolve()
        self.transform = ClassificationTransform(config, training=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        source = (self.root / self.rows[index]["image_path"]).resolve()
        if not source.is_relative_to(self.root):
            raise ValueError(f"Image path escapes data root: {source}")
        with Image.open(source) as image:
            return self.transform(image.convert("RGB"))


def archived_predictions(args, records, indices, split, output, manifest_hash):
    import torch
    from torch.utils.data import DataLoader
    from training.classification import build_model

    if checksum(args.legacy_weights) != LEGACY_SHA256:
        raise ValueError("Legacy weights differ from the archived RFMiD checkpoint")
    # This historical checkpoint contains NumPy RNG state; its archive SHA was checked above.
    checkpoint = torch.load(args.legacy_weights, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if (tuple(cfg["labels"]) != LEGACY_LABELS or cfg.get("task") != "multidisease"
            or cfg["model"]["img_size"] != 392
            or cfg.get("preprocessing", {}).get("mode", "stretch") != "stretch"):
        raise ValueError("Archived model/preprocessing contract does not match this RFMiD comparison")
    net = build_model(cfg, pretrained=False)
    net.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    net.to(args.device).eval()
    loader = DataLoader(RetinalImages(records, args.data_root, cfg), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False, pin_memory=args.device.startswith("cuda"))
    rows = []
    with torch.inference_mode():
        for images in loader:
            logits = net(images.to(args.device, non_blocking=True)).float()
            if logits.ndim != 2 or logits.shape[1] != len(LEGACY_LABELS) or not torch.isfinite(logits).all():
                raise ValueError("Legacy checkpoint produced invalid logits")
            scores = logits.sigmoid().cpu().tolist()
            for logit, probability in zip(logits.cpu().tolist(), scores):
                record = records[len(rows)]
                row = {key: record[key] for key in
                       ("sample_id", "dataset", "split", "patient_id", "content_group_id")}
                row.update(labels=[record["labels"][i] for i in indices],
                           mask=[record["mask"][i] for i in indices],
                           logits=logit[1:], probabilities=probability[1:], checkpoint_sha256=LEGACY_SHA256)
                rows.append(row)
    write_file(output / f"{split}_predictions.jsonl",
               "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    write_json(output / f"{split}_predictions.meta.json",
               dict(labels=list(COMMON), split=split, checkpoint_sha256=LEGACY_SHA256,
                    manifest_sha256=manifest_hash, preprocessing=cfg.get("preprocessing", {"mode": "stretch"}),
                    input_size=cfg["model"]["img_size"], original_labels=list(LEGACY_LABELS)))


def paired_report(args, new_rows, old_rows, new_lock, old_lock):
    old_by_id = {row["sample_id"]: row for row in old_rows}
    if set(old_by_id) != {row["sample_id"] for row in new_rows}:
        raise ValueError("Legacy and current predictors cover different sample IDs")
    aligned = [old_by_id[row["sample_id"]] for row in new_rows]
    result = {}
    for old_idx, label in enumerate(COMMON):
        new_idx = new_lock["labels"].index(label)
        for new, old in zip(new_rows, aligned):
            if (any(new.get(k) != old.get(k) for k in ("dataset", "split", "patient_id", "content_group_id"))
                    or new["mask"][new_idx] != old["mask"][old_idx]
                    or (new["mask"][new_idx] and new["labels"][new_idx] != old["labels"][old_idx])):
                raise ValueError(f"Different reference labels or identity for {new['sample_id']}, {label}")
        y, new_p, groups = known_arrays(new_rows, new_idx)
        _, old_p, _ = known_arrays(aligned, old_idx)
        new_t = new_lock["per_label"][label]["threshold"]
        old_t = old_lock["per_label"][label]["threshold"]
        rng = np.random.default_rng(args.seed)
        n_groups = len(set(groups))
        old_point = metrics(y, old_p, old_t)
        new_point = metrics(y, new_p, new_t)
        differences = {"ap": [], "fp": [], "sensitivity": []}
        if n_groups > 1:
            for _ in range(args.bootstrap):
                weights = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)[groups]
                before = metrics(y, old_p, old_t, weights)
                after = metrics(y, new_p, new_t, weights)
                for key in differences:
                    if after[key] is not None and before[key] is not None:
                        differences[key].append(after[key] - before[key])
        result[label] = dict(known=len(y), positive=int(y.sum()), negative=int((y == 0).sum()),
                             legacy_validation=old_lock["per_label"][label],
                             new_validation=new_lock["per_label"][label],
                             legacy=old_point, new=new_point,
                             new_minus_legacy={k: {
                                 "estimate": new_point[k] - old_point[k] if new_point[k] is not None
                                 and old_point[k] is not None else None,
                                 "ci95": np.quantile(v, [0.025, 0.975]).tolist() if len(v) >= 2 else None,
                                 "valid_draws": len(v)} for k, v in differences.items()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Final new grouped model run with a val threshold lock")
    parser.add_argument("--release", type=Path, required=True, help="screening_v1 directory containing audit.json")
    parser.add_argument("--data-root", type=Path, required=True, help="Project root containing data/")
    parser.add_argument("--legacy-weights", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    if args.bootstrap < 2:
        raise ValueError("At least two bootstrap draws required")
    new_lock = read_json(args.run / "thresholds.json")
    val_rows, _, val_hash = load_predictions(args.run, "val")
    if val_hash != new_lock["val_predictions_sha256"]:
        raise ValueError("New model validation predictions differ from its threshold lock")
    del val_rows
    split = args.split
    new_rows, new_meta, new_digest = load_predictions(args.run, split)
    if new_meta["checkpoint_sha256"] != new_lock["checkpoint_sha256"] or new_meta["labels"] != new_lock["labels"]:
        raise ValueError("New predictions do not match the new model threshold lock")
    audit = read_json(args.release / "audit.json")
    if audit["label_order"]["grouped"] != new_meta["labels"]:
        raise ValueError("New label order differs from the data release")
    manifest = args.release / "grouped" / f"{split}.jsonl"
    manifest_hash = checksum(manifest)
    if new_meta["manifest_sha256"] != manifest_hash:
        raise ValueError("New predictor used a different data release for this split")
    records = [json.loads(s) for s in manifest.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if (any(r["split"] != split or r["dataset"] != "RFMiD" for r in records)
            or len({r["sample_id"] for r in records}) != len(records)):
        raise ValueError("Expected a complete single-source RFMiD split with unique IDs")
    output = args.run / "legacy_comparison"
    output.mkdir(exist_ok=True)
    old_lock_path = output / "thresholds.json"
    if split == "test" and not old_lock_path.exists():
        raise ValueError("Run the complete validation comparison to lock legacy thresholds first")
    indices = [new_meta["labels"].index(k) for k in COMMON]
    if not (output / f"{split}_predictions.meta.json").exists():
        archived_predictions(args, records, indices, split, output, manifest_hash)
    old_rows, old_meta, old_digest = load_predictions(output, split)
    if (old_meta["checkpoint_sha256"] != LEGACY_SHA256 or old_meta["labels"] != list(COMMON)
            or old_meta["manifest_sha256"] != manifest_hash):
        raise ValueError("Cached legacy prediction provenance mismatch")
    if split == "val":
        per_label = {}
        for i, name in enumerate(COMMON):
            y, p, _ = known_arrays(old_rows, i)
            goal = {key: new_lock["per_label"][name][key] for key in ("min_recall", "max_fpr")}
            per_label[name] = choose_threshold(y, p, goal)
        write_json(old_lock_path, dict(labels=list(COMMON), checkpoint_sha256=LEGACY_SHA256,
                                       new_lock_sha256=checksum(args.run / "thresholds.json"),
                                       val_predictions_sha256=old_digest, per_label=per_label), lock=True)
    old_lock = read_json(old_lock_path)
    if (old_lock["new_lock_sha256"] != checksum(args.run / "thresholds.json")
            or old_lock["val_predictions_sha256"] != checksum(output / "val_predictions.jsonl")):
        raise ValueError("Legacy thresholds differ from the frozen comparison")
    result = paired_report(args, new_rows, old_rows, new_lock, old_lock)
    write_json(output / f"{split}_paired.json",
               dict(split=split, matched_images=len(new_rows), manifest_sha256=manifest_hash,
                    legacy_checkpoint_sha256=LEGACY_SHA256, new_checkpoint_sha256=new_meta["checkpoint_sha256"],
                    new_predictions_sha256=new_digest, legacy_predictions_sha256=old_digest,
                    paired_bootstrap_draws=args.bootstrap, paired_bootstrap_seed=args.seed,
                    evaluation="paired on identical image IDs; both thresholds locked on validation",
                    per_label=result))


if __name__ == "__main__":
    main()
