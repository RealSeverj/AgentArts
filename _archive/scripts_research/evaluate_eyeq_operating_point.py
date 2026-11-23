"""Evaluate the archived EyeQ classifier at three retake operating points on its full held-out splits."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.screening_compare_legacy import RetinalImages, checksum
from scripts.screening_evaluation import read_json, write_json
from training.classification import build_model

WEIGHTS = "fa5ee22b483b09c8dec9050455b5e6010d51c2ecced17d0a55982657bdd66f8f"
EXPECTED = {"val": ("cdf5ce20022f6af08609313f7c38add1cb75b25808451e51c0276c111527de29", 1875),
            "test": ("f7e1436214957d925cc085c4992f09b7bf1ec05081cec232836a13fe124b93f3", 16249)}


def operating_points(target, score, argmax):
    bad = target == 2
    def count(reject):
        fn = int((bad & ~reject).sum())
        fp = int((~bad & reject).sum())
        return {"bad_images": int(bad.sum()), "usable_images": int((~bad).sum()),
                "bad_passed": fn, "usable_rejected": fp, "bad_passed_fraction": fn / int(bad.sum()),
                "usable_rejected_fraction": fp / int((~bad).sum())}
    return count, count(argmax == 2), count(score >= 0.5)


def fit_threshold(target, score):
    # Include reject-none so the false-reject bound is always attainable.
    bad = target == 2
    thresholds = np.r_[np.nextafter(1., 2.), np.unique(score)]
    allowed = math.floor(0.05 * int((~bad).sum()))
    choices = []
    for threshold in thresholds:
        rejected = score >= threshold
        fp = int((rejected & ~bad).sum())
        if fp <= allowed:
            choices.append((int((~rejected & bad).sum()), fp, -float(threshold)))
    return -min(choices)[2]


def load_records(manifest, split):
    if checksum(manifest) != EXPECTED[split][0]:
        raise ValueError(f"{split} manifest differs from archived EyeQ split")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (len(rows) != EXPECTED[split][1] or len({r["sample_id"] for r in rows}) != len(rows)
            or any(r["dataset"] != "EyeQ" or r["split"] != split or not r["label_mask"]["quality_grade"]
                   or r["labels"]["quality_grade"] not in (0, 1, 2) for r in rows)):
        raise ValueError("Invalid cohort coverage or labels")
    return rows


def score_split(split, model, cfg, args):
    rows = load_records(args.data_root / "prepared/source_isolated/strict_cross_domain/tasks/image_quality/EyeQ"
                        / f"{split}.jsonl", split)
    output = args.output / f"{split}_predictions.jsonl"
    temp = output.with_suffix(".partial")
    dataset = RetinalImages(rows, args.data_root, cfg)
    if output.exists():
        meta = read_json(args.output / f"{split}_predictions.meta.json")
        if meta["checkpoint_sha256"] != WEIGHTS or meta["predictions_sha256"] != checksum(output):
            raise ValueError(f"Cached {split} predictions differ")
    else:
        with temp.open("w", encoding="utf-8") as stream, torch.inference_mode(), ThreadPoolExecutor(4) as pool:
            for start in range(0, len(rows), args.batch):
                stop = min(len(rows), start + args.batch)
                images = torch.stack(list(pool.map(dataset.__getitem__, range(start, stop)))).to("cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    probabilities = model(images).float().softmax(dim=1).cpu().numpy()
                if probabilities.shape != (stop - start, 3) or not np.isfinite(probabilities).all():
                    raise ValueError("Invalid EyeQ probability outputs")
                for row, scores in zip(rows[start:stop], probabilities):
                    item = {k: row[k] for k in ("sample_id", "patient_id", "content_group_id", "image_path")}
                    item.update(split=split, grade=row["labels"]["quality_grade"], probabilities=scores.tolist(),
                                predicted_grade=int(np.argmax(scores)))
                    stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                if start == 0 or stop % (args.batch * 64) == 0 or stop == len(rows):
                    print(f"{split}: {stop}/{len(rows)}", flush=True)
        os.replace(temp, output)
        write_json(args.output / f"{split}_predictions.meta.json",
                   dict(checkpoint_sha256=WEIGHTS, manifest_sha256=EXPECTED[split][0], samples=len(rows),
                        predictions_sha256=checksum(output), grade_order=["good", "usable", "reject"],
                        precision="fp16 autocast, fp32 softmax", batch_size=args.batch, device=torch.cuda.get_device_name()))
    results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    if len(results) != len(rows) or any(a["sample_id"] != b["sample_id"] for a,b in zip(results,rows)):
        raise ValueError("Prediction IDs differ from registered cohort")
    return (np.array([r["grade"] for r in results]), np.array([r["probabilities"][2] for r in results]),
            np.array([r["predicted_grade"] for r in results]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or checksum(args.checkpoint) != WEIGHTS:
        raise ValueError("Expected CUDA and the pinned EyeQ checkpoint")
    weights = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    cfg = weights["config"]
    if cfg["task"] != "image_quality" or cfg["num_classes"] != 3 or cfg["data"]["datasets"] != ["EyeQ"]:
        raise ValueError("Incompatible EyeQ checkpoint architecture or label mapping")
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(weights["model"], strict=True)
    del weights
    gc.collect()
    model.to("cuda").eval()
    args.output.mkdir(parents=True, exist_ok=True)
    policy = dict(max_usable_reject_fraction=0.05, policy="Minimize bad images passed, then usable images "
                  "rejected, then use highest threshold; P(reject) >= threshold", checkpoint_sha256=WEIGHTS)
    write_json(args.output / "policy.json", policy, lock=True)
    for split in ("val", "test"):
        y, score, argmax = score_split(split, model, cfg, args)
        if split == "val":
            lock = dict(policy, validation_predictions_sha256=checksum(args.output / "val_predictions.jsonl"),
                        threshold=fit_threshold(y, score))
            write_json(args.output / "threshold.json", lock, lock=True)
        lock = read_json(args.output / "threshold.json")
        count, historic, at_half = operating_points(y, score, argmax)
        matrix = np.zeros((3,3), dtype=int)
        np.add.at(matrix, (y, argmax), 1)
        report = dict(split=split, samples=len(y), checkpoint_sha256=WEIGHTS, historical_test=(split == "test"),
                      threshold=lock["threshold"], val_predictions_sha256=lock["validation_predictions_sha256"],
                      confusion_argmax=matrix.tolist(), argmax=historic, threshold_0_5=at_half,
                      validation_locked=count(score >= lock["threshold"]))
        write_json(args.output / f"{split}_metrics.json", report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
