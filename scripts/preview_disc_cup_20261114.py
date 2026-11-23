"""Preview three predetermined REFUGE validation cases from pinned own weights.

No training, test-set reads, threshold fitting, result-based case selection, or
cup/disc containment correction. Uses the existing structural validation adapter
for the actual 512-pixel image/mask geometry. Pillow draws scientific contours;
Matplotlib and additional packages are not required.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps, __version__ as pillow_version
import torch

from scripts import dataset_routing as routing
from scripts.manifest_dataset import resolve_data_path
from training import structural as adapter
from training.fundus_preprocessing import preprocessing_spec


LABELS = ["optic_cup", "optic_disc"]
SOURCE = "REFUGE"
COUNT = 400
SIZE = 512
THRESHOLD = 0.5
COLORS = {"optic_cup": (255, 183, 65), "optic_disc": (63, 220, 229)}
CHUNK = 8 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def reject_constant(value):
    raise ValueError(f"Non-standard JSON constant: {value}")


def read_json(payload):
    return json.loads(payload.decode("utf-8-sig"), parse_constant=reject_constant)


def stat_identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def hash_file(path):
    with Path(path).open("rb") as handle:
        before = os.fstat(handle.fileno())
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
        require(stat_identity(before) == stat_identity(os.fstat(handle.fileno())),
                f"Input changed while hashing: {path}")
    return {"path": str(Path(path).resolve()), "bytes": before.st_size, "sha256": digest.hexdigest()}


def load_checkpoint(path, expected_sha256):
    require(is_sha256(expected_sha256), "--expected-sha256 must be a lowercase SHA256")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        require(before.st_size > 0, "Checkpoint is empty")
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
        require(digest.hexdigest() == expected_sha256, "Checkpoint SHA mismatch; no weights were deserialized")
        require(stat_identity(before) == stat_identity(os.fstat(handle.fileno())),
                "Checkpoint changed while hashing")
        handle.seek(0)
        # Only the user's own, externally SHA-pinned training artifact is accepted.
        checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
        require(stat_identity(before) == stat_identity(os.fstat(handle.fileno()))
                and stat_identity(before) == stat_identity(path.stat()), "Checkpoint changed while loading")
    require(isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict),
            "Expected an owned training checkpoint with model state")
    config = checkpoint.get("config")
    require(isinstance(config, dict) and config.get("task") == "disc_cup_segmentation"
            and config.get("kind") == "segmentation" and config.get("labels") == LABELS,
            "Only the reviewed cup/disc segmentation task and channel order are accepted")
    data, model = config.get("data", {}), config.get("model", {})
    require(data.get("source_isolated") is True and data.get("datasets") == [SOURCE]
            and data.get("protocol") == "strict_cross_domain" and not data.get("external_dataset"),
            "Only the isolated REFUGE model without an external source is accepted")
    require(model.get("name") == "unet" and model.get("img_size") == SIZE
            and model.get("base_channels") == 32 and model.get("pretrained") is False,
            "Expected the randomly initialized base32 U-Net at 512 pixels")
    require(preprocessing_spec(config)["mode"] == "stretch", "DR crop/letterbox is not a segmentation geometry")
    threshold = config.get("evaluation", {}).get("threshold", THRESHOLD)
    require(isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
            and math.isfinite(threshold) and float(threshold) == THRESHOLD, "Expected the fixed threshold 0.5")
    require(config.get("training", {}).get("epochs") == 50
            and config.get("training", {}).get("amp") == "fp16", "Expected the reviewed 50-epoch FP16 recipe")
    require(type(checkpoint.get("epoch")) is int and 0 <= checkpoint["epoch"] < 50,
            "Checkpoint completed epoch is invalid")
    normalized = deepcopy(config)
    normalized.pop("notes", None)
    normalized["training"] = {key: value for key, value in normalized["training"].items()
                              if key not in ("workers", "log_every")}
    identity = checkpoint.get("data_identity")
    require(isinstance(identity, dict) and identity.get("experiment") == normalized,
            "Checkpoint configuration contradicts its saved experiment identity")
    return checkpoint, {"path": str(path), "bytes": before.st_size, "sha256": expected_sha256,
                        "mtime_ns": before.st_mtime_ns, "ctime_ns": before.st_ctime_ns,
                        "device_id": before.st_dev, "inode": before.st_ino,
                        "sha_checked_before_cpu_deserialization": True, "stat_stable_during_load": True}


def validation_dataset(config, identity, data_root):
    """Read only the global index and REFUGE val leaf, never a test leaf."""
    routing.validate_isolated_config(config)
    root = routing.isolation_root(config, data_root)
    index_path = root / "isolation_manifest.json"
    index_bytes = index_path.read_bytes()
    index_sha = hashlib.sha256(index_bytes).hexdigest()
    require(config["data"].get("isolation_manifest_sha256") == index_sha,
            "Isolation index differs from the checkpoint's pin")
    index = read_json(index_bytes)
    require(index.get("format_version") == 1 and index.get("all_checks_passed") is True
            and index.get("record_bytes_preserved") is True
            and "strict_cross_domain" in index.get("base_protocols", []), "Isolation index was not successfully audited")
    matches = [entry for entry in index["combinations"]
               if (entry.get("base_protocol"), entry.get("task"), entry.get("source")) ==
               ("strict_cross_domain", "disc_cup_segmentation", SOURCE)]
    require(len(matches) == 1, "REFUGE segmentation has no unique isolation-index entry")
    info = matches[0]["splits"]["val"]
    manifest = routing.manifest_path(config, data_root, "val")
    require(info.get("path") == manifest.relative_to(root).as_posix() and info.get("records") == COUNT,
            "Unexpected REFUGE validation leaf/count")
    payload = manifest.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    require(info.get("sha256") == digest and info.get("bytes") == len(payload)
            and identity.get("manifests_sha256", {}).get("val") == digest,
            "Validation manifest differs from audited index or saved training identity")
    rows = [read_json(line) for line in payload.splitlines() if line.strip()]
    identifiers = set()
    for row in rows:
        routing.validate_row(config, row, "val")
        identifier = row.get("sample_id")
        require(isinstance(identifier, str) and identifier and identifier not in identifiers,
                "Validation IDs must be nonempty and unique")
        require(all(row.get("label_mask", {}).get(label) is True for label in LABELS)
                and isinstance(row.get("image_paths"), list) and len(row["image_paths"]) == 1,
                "Validation must have aligned known cup/disc targets and one source image")
        identifiers.add(identifier)
    require(len(rows) == COUNT, "REFUGE validation must contain exactly 400 records")
    dataset = adapter.build_dataset(manifest, data_root, config, training=False)
    require(len(dataset) == COUNT and {row["sample_id"] for row in dataset.rows} == identifiers,
            "Structural adapter omitted/changed validation records")
    require(manifest.read_bytes() == payload, "Validation manifest changed while constructing dataset")
    selected = sorted(range(len(dataset.rows)), key=lambda i: dataset.rows[i]["sample_id"])[:3]
    return dataset, selected, {"path": str(manifest), "bytes": len(payload), "sha256": digest,
                               "source": SOURCE, "split": "val", "total_records": COUNT,
                               "isolation_manifest_sha256": index_sha}


def metrics(prediction, truth):
    result = {}
    for i, label in enumerate(LABELS):
        positive, target = prediction[i].bool(), truth[i].bool()
        tp = int((positive & target).sum())
        fp = int((positive & ~target).sum())
        fn = int((~positive & target).sum())
        denominator = 2 * tp + fp + fn
        union = tp + fp + fn
        result[label] = {"dice": 2 * tp / denominator if denominator else 1.0,
                         "iou": tp / union if union else 1.0, "tp_pixels": tp,
                         "fp_pixels": fp, "fn_pixels": fn,
                         "predicted_pixels": int(positive.sum()), "gt_pixels": int(target.sum())}
    result["macro_dice"] = sum(result[label]["dice"] for label in LABELS) / len(LABELS)
    return result


def containment(mask):
    cup, disc = mask[0].bool(), mask[1].bool()
    outside, total = int((cup & ~disc).sum()), int(cup.sum())
    return {"cup_outside_disc": outside > 0, "cup_outside_disc_pixels": outside,
            "cup_pixels": total, "outside_fraction_of_cup": outside / total if total else None,
            "containment_correction_applied": False}


def display_image(tensor, config):
    model = config["model"]
    mean = torch.tensor(model.get("mean") or [0.0] * 3).view(3, 1, 1)
    std = torch.tensor(model.get("std") or [1.0] * 3).view(3, 1, 1)
    require(mean.shape == std.shape == (3, 1, 1) and bool((std > 0).all()), "Invalid display normalization")
    rgb = ((tensor.detach().cpu().float() * std + mean).clamp(0, 1) * 255).round().to(torch.uint8)
    return Image.fromarray(rgb.permute(1, 2, 0).numpy(), mode="RGB")


def contour_overlay(image, masks):
    result = image.copy()
    # Zero padding makes inner boundaries at image edges explicit. Dilation
    # affects the visual stroke only; predictions and quantitative masks stay intact.
    for i, label in enumerate(LABELS):
        binary = Image.fromarray(masks[i].to(torch.uint8).numpy() * 255, mode="L")
        padded = ImageOps.expand(binary, border=1, fill=0)
        eroded = padded.filter(ImageFilter.MinFilter(3)).crop((1, 1, SIZE + 1, SIZE + 1))
        boundary = ImageChops.subtract(binary, eroded).filter(ImageFilter.MaxFilter(3))
        result.paste(COLORS[label], (0, 0), boundary)
    return result


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for candidate in (f"/usr/share/fonts/truetype/dejavu/{name}", name,
                      "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    raise RuntimeError("A readable DejaVu Sans/Arial font is required for a legible figure")


def render(cases, completed_epoch, digest):
    margin, gap, row_height, header = 36, 18, 610, 166
    width = 2 * margin + 3 * SIZE + 2 * gap
    canvas = Image.new("RGB", (width, 2120), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    title_font, text_font, small_font = font(32, True), font(20), font(17)
    draw.text((margin, 20), "REFUGE: fixed validation cup/disc examples", fill="#17283d", font=title_font)
    draw.text((margin, 67), f"U-Net at 512 px | checkpoint epoch {completed_epoch} | sigmoid threshold 0.50", fill="#475569", font=text_font)
    legend_y = 105
    for x, label, text in ((margin, "optic_cup", "Cup"), (190, "optic_disc", "Disc")):
        draw.rectangle((x, legend_y + 4, x + 28, legend_y + 16), fill=COLORS[label])
        draw.text((x + 39, legend_y - 3), text, fill="#17283d", font=text_font)
    draw.text((370, legend_y - 3), "Contours only; no mask correction or result-based case selection", fill="#475569", font=text_font)
    headings = ("Input (whole image, resized)", "Ground-truth contours", "Predicted contours")
    x_positions = [margin + i * (SIZE + gap) for i in range(3)]
    for x, heading in zip(x_positions, headings):
        draw.text((x, 139), heading, fill="#17283d", font=font(21, True))
    for i, case in enumerate(cases):
        y = header + i * row_height
        record = case["record"]
        source_id = record.get("source_sample_id") or record["sample_id"]
        draw.text((margin, y + 10), f"Case {i + 1} | {source_id} | {record['sample_id']}", fill="#334155", font=text_font)
        image_y = y + 40
        panels = (case["image"], contour_overlay(case["image"], case["truth"]),
                  contour_overlay(case["image"], case["prediction"]))
        for x, panel in zip(x_positions, panels):
            canvas.paste(panel, (x, image_y))
            draw.rectangle((x, image_y, x + SIZE - 1, image_y + SIZE - 1), outline="#cbd5e1", width=1)
        label_y = image_y + SIZE + 7
        native = record["original_size"]
        draw.text((x_positions[0], label_y), f"Native {native[0]} x {native[1]} -> 512 x 512", fill="#475569", font=text_font)
        ground = record["ground_truth_containment"]
        status = "YES" if ground["cup_outside_disc"] else "no"
        draw.text((x_positions[1], label_y), f"GT cup outside disc: {status} ({ground['cup_outside_disc_pixels']} px)",
                  fill="#475569", font=text_font)
        m = record["metrics"]
        draw.text((x_positions[2], label_y), f"Cup Dice {m['optic_cup']['dice']:.4f} | Disc Dice {m['optic_disc']['dice']:.4f}",
                  fill="#17283d", font=text_font)
        pred = record["prediction_containment"]
        status = "YES" if pred["cup_outside_disc"] else "no"
        draw.text((x_positions[2], label_y + 25), f"Cup outside disc: {status} ({pred['cup_outside_disc_pixels']} px)",
                  fill="#ae293b" if pred["cup_outside_disc"] else "#475569", font=text_font)
    footer = header + 3 * row_height + 16
    notes = (
        "Fixed selection: first 3 sample IDs in sorted REFUGE validation manifest (N=400); no test records are read.",
        "Whole-image aligned resize: bilinear RGB, nearest-neighbor masks. Contours are drawn 3 px wide for display only.",
        "Per-case Dice is descriptive; these examples are not the full-cohort score. Cup/disc containment is not enforced.",
        "Checkpoint SHA256: " + digest,
    )
    for j, note in enumerate(notes):
        draw.text((margin, footer + 23 * j), note, fill="#475569", font=small_font)
    return canvas


def preview(checkpoint_path, expected_sha256, data_root, prefix, device):
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    checkpoint_path, data_root, prefix = (Path(value).expanduser().resolve() for value in (checkpoint_path, data_root, prefix))
    png, json_path = Path(str(prefix) + ".png"), Path(str(prefix) + ".json")
    require(not any(path.is_relative_to(data_root) or path == checkpoint_path for path in (png, json_path)),
            "Preview outputs must be outside immutable data and checkpoint paths")
    if png.exists() or json_path.exists():
        raise FileExistsError("Preview outputs exist; choose a new --output-prefix")
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "Run this preview as one process, not torchrun/DDP")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device cpu explicitly")
    if device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    target_device = torch.device(device)
    checkpoint, checkpoint_record = load_checkpoint(checkpoint_path, expected_sha256)
    config = deepcopy(checkpoint["config"])
    original_config = json.dumps(config, sort_keys=True)
    completed_epoch = checkpoint["epoch"] + 1
    dataset, selected, cohort = validation_dataset(config, checkpoint["data_identity"], data_root)
    model = adapter.build_model(config, pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False).to(target_device)
    del checkpoint
    gc.collect()
    cases = []
    for index in selected:
        row = dataset.rows[index]
        image_path = resolve_data_path(row["image_paths"][0], data_root)
        image_file = hash_file(image_path)
        declared_sha = row.get("image_sha256")
        require(isinstance(declared_sha, list) and len(declared_sha) == 1
                and declared_sha[0] == image_file["sha256"], "Selected image differs from its declared original SHA")
        mask_files = {}
        for label in LABELS:
            mask = adapter._mask_spec(row, label, config)
            require(isinstance(mask, dict) and isinstance(mask.get("path"), str), "Selected known mask is missing")
            mask_files[label] = hash_file(resolve_data_path(mask["path"], data_root)) | {"manifest_path": mask["path"]}
        sample = dataset[index]
        # StructuralDataset returns singular 'image'; 'images' is introduced
        # only by collate(). Using this sample preserves its exact validation transform.
        image, truth, known = sample.get("image"), sample.get("targets"), sample.get("known")
        require(isinstance(image, torch.Tensor) and tuple(image.shape) == (3, SIZE, SIZE)
                and bool(torch.isfinite(image).all()), "Unexpected adapter image tensor")
        require(isinstance(truth, torch.Tensor) and tuple(truth.shape) == (2, SIZE, SIZE)
                and bool(((truth == 0) | (truth == 1)).all()), "Unexpected aligned binary cup/disc tensors")
        require(isinstance(known, torch.Tensor) and tuple(known.shape) == (2,) and bool(known.all())
                and sample.get("sample_id") == row["sample_id"] and sample.get("dataset") == SOURCE,
                "Adapter sample ID/source/known masks differ")
        require(hash_file(image_path)["sha256"] == image_file["sha256"]
                and all(hash_file(item["path"])["sha256"] == item["sha256"] for item in mask_files.values()),
                "Selected image/mask files changed while decoding")
        if device == "cuda":
            torch.cuda.synchronize(target_device)
        inference_start = time.monotonic()
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext():
                logits = model(image.unsqueeze(0).to(target_device))
            require(tuple(logits.shape) == (1, 2, SIZE, SIZE) and bool(torch.isfinite(logits).all()),
                    "Model output is nonfinite or has the wrong shape")
            # Matches training.structural.step: FP16 forward, then FP32 sigmoid.
            probabilities = logits[0].float().sigmoid().cpu()
        if device == "cuda":
            torch.cuda.synchronize(target_device)
        elapsed = time.monotonic() - inference_start
        predicted, truth = probabilities >= THRESHOLD, truth.bool().cpu()
        record = {"sample_id": row["sample_id"], "source_sample_id": row.get("source_sample_id"),
                  "dataset": SOURCE, "split": "val", "manifest_row_index": index,
                  "image_paths": row["image_paths"], "source_image": image_file,
                  "target_masks": mask_files, "original_size": list(sample["original_size"]),
                  "metrics": metrics(predicted, truth), "ground_truth_containment": containment(truth),
                  "prediction_containment": containment(predicted), "inference_seconds": round(elapsed, 6)}
        cases.append({"record": record, "image": display_image(image, config), "truth": truth, "prediction": predicted})
    require(json.dumps(config, sort_keys=True) == original_config, "Original checkpoint configuration was modified")
    canvas = render(cases, completed_epoch, expected_sha256)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with png.open("xb") as handle:
        canvas.save(handle, format="PNG")
    result = {"format_version": 1, "all_checks_passed": True,
              "created_utc": datetime.now(timezone.utc).isoformat(), "started_utc": started_utc,
              "elapsed_seconds": round(time.monotonic() - started, 6),
              "mode": "fixed_three_validation_cases_scientific_preview", "checkpoint": checkpoint_record,
              "parent_weights_sha256": expected_sha256, "checkpoint_completed_epoch": completed_epoch,
              "model_config": config["model"], "original_config": config, "cohort": cohort,
              "selection_policy": "First three sample_id values in lexicographic order over the complete fixed REFUGE val manifest; selection occurs before inference.",
              "selected_sample_ids": [case["record"]["sample_id"] for case in cases],
              "cases": [case["record"] for case in cases], "channel_order": LABELS, "threshold": THRESHOLD,
              "geometry": {"evaluation_space": "whole_image_resized_to_model_img_size", "size_hw": [SIZE, SIZE],
                           "image_resize": "PIL_BILINEAR", "mask_resize": "PIL_NEAREST",
                           "transform": "training.structural.StructuralDataset.__getitem__(training=False)",
                           "display_rgb": "Inverse normalization of the actual adapter tensor; rounded to uint8 RGB",
                           "containment_correction_applied": False},
              "metric_convention": "Per-case cup/disc binary Dice and IoU; empty-empty=1, no smoothing. Not a full-cohort metric.",
              "figure": hash_file(png) | {"size_wh": list(canvas.size), "cup_color_rgb": list(COLORS["optic_cup"]),
                                        "disc_color_rgb": list(COLORS["optic_disc"]),
                                        "contour_display": "8-neighbor inner binary boundary dilated to a 3-pixel stroke for display only"},
              "runtime": {"hostname": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
                          "torch": torch.__version__, "torch_cuda": torch.version.cuda, "numpy": np.__version__,
                          "Pillow": pillow_version, "device": device,
                          "gpu": torch.cuda.get_device_name(target_device) if device == "cuda" else None,
                          "inference_batch_size": 1, "forward_precision": "fp16" if device == "cuda" else "fp32",
                          "sigmoid_precision": "fp32", "model_eval": True, "inference_mode": True},
              "training_performed": False, "optimizer_restored": False, "pretrained_download": False,
              "model_weights_updated": False, "test_records_read": False, "threshold_fitted": False,
              "limitations": ["Three predetermined validation illustrations do not summarize all 400 validation cases or any test cohort.",
                              "Validation was used for checkpoint selection; these examples are descriptive.",
                              "Batch-one FP16 predictions may differ near the threshold from other batch layouts.",
                              "REFUGE is the preserved audited partial REFUGE/REFUGE2 public mirror, not asserted to be a pure official release."]}
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {"all_checks_passed": True, "png": str(png), "json": str(json_path),
            "checkpoint_sha256": expected_sha256, "checkpoint_completed_epoch": completed_epoch,
            "selected_sample_ids": result["selected_sample_ids"], "output_sha256":
            {png.name: result["figure"]["sha256"], json_path.name: hash_file(json_path)["sha256"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    try:
        print(json.dumps(preview(args.checkpoint, args.expected_sha256, args.data_root, args.output_prefix, args.device), ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Fixed cup/disc preview failed: {error}\n")


if __name__ == "__main__":
    main()
