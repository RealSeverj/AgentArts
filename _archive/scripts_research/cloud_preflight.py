"""Read-only cloud training checks for a selected task/configuration.

Unlike the full protocol validator this accepts an intentionally partial upload.
It checks runtime image/mask/XML references, not provenance-only source paths.
No dependencies, model weights, or source files are installed/modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys

try:
    from scripts.package_cloud_training import load_config, protocol_root, resolve_reference, runtime_annotations, selected_rows, validate_packaged_leaves
except ImportError:
    from package_cloud_training import load_config, protocol_root, resolve_reference, runtime_annotations, selected_rows, validate_packaged_leaves
from scripts import dataset_routing as routing

BASE = Path(__file__).resolve().parents[1]


def collect_supervision(row, labels, kind, config):
    known = row.get("label_mask", {})
    if kind in {"multiclass", "multilabel"}:
        return {label: bool(known.get(label, False)) and row.get("labels", {}).get(label) is not None for label in labels}
    annotations = row.get("annotations", {})
    if kind == "segmentation":
        masks = annotations.get("masks", {})
        experts = annotations.get("experts", {})
        selected = config.get("data", {}).get("expert", "expert_1")
        if experts:
            # REFUGE and IDRiD provide one reference expert; PAPILA has two.
            if selected in experts:
                masks = experts[selected]
            elif len(experts) == 1:
                masks = next(iter(experts.values()))
            else:
                masks = {}
        return {label: bool(known.get(label, False)) and bool(masks.get(label, {}).get("path")) for label in labels}
    if kind == "landmark":
        return {label: bool(known.get(label, False)) and bool(annotations.get("points", {}).get(label, {}).get("xy")) for label in labels}
    if kind == "detection":
        return {"objects": bool(known.get("objects", False)) and bool(annotations.get("xml_path"))}
    raise ValueError(f"Unsupported task kind: {kind!r}")


def run_checks(config, data_root, sample_count, allow_cpu):
    report = {"task": config["task"], "kind": config["kind"], "protocol": config.get("data", {}).get("protocol", "strict_cross_domain"),
              "errors": [], "warnings": [], "splits": {}, "packages": {}, "cuda": {}}
    errors, warnings = report["errors"], report["warnings"]
    try:
        from training.fundus_preprocessing import validate_preprocessing_config
        validate_preprocessing_config(config)
        report["source_isolated"] = routing.is_source_isolated(config)
        route = routing.validate_training_route(config, data_root)
        report["training_route"] = json.loads(json.dumps(route, default=str))
        report["leaf_integrity"] = validate_packaged_leaves(config, data_root)
    except (OSError, ValueError, KeyError) as error:
        errors.append(f"Source-isolated training route rejected: {error}")
        report["all_checks_passed"] = False
        return report
    for package in ["torch", "torchvision", "timm", "numpy", "Pillow", "pycocotools"]:
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    try:
        import torch
        cuda = torch.cuda.is_available()
        report["cuda"] = {"available": cuda, "torch_cuda_version": torch.version.cuda,
                          "device_count": torch.cuda.device_count(), "devices": []}
        if cuda:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                report["cuda"]["devices"].append({"name": properties.name, "memory_gib": round(properties.total_memory / 1024**3, 2),
                                                      "capability": list(torch.cuda.get_device_capability(index))})
        elif not allow_cpu:
            errors.append("CUDA is unavailable. Install a matching CUDA PyTorch build in the cloud environment; --allow-cpu is for smoke/debug only.")
    except Exception as error:
        errors.append(f"PyTorch import failed: {type(error).__name__}: {error}")
    kind = config["kind"]
    if config.get("model", {}).get("name") not in {"tiny_cnn", "unet"} and kind != "detection" and not report["packages"]["timm"]:
        errors.append("timm is not installed for the configured backbone")
    if kind == "detection" and not report["packages"]["torchvision"]:
        errors.append("torchvision is not installed for detection")
    if kind == "detection" and not report["packages"]["pycocotools"]:
        errors.append("pycocotools is not installed for COCO-style detection evaluation")
    if kind == "detection" and config.get("data", {}).get("box_coordinate_mode") not in {
        "source_xyxy_no_offset", "voc_1based_inclusive", "zero_based_inclusive"
    }:
        errors.append("Detection requires an explicit supported data.box_coordinate_mode; source_xyxy_no_offset preserves the source numbers without claiming an official pixel origin")
    labels = config.get("labels", [])
    if kind != "detection" and not labels:
        errors.append("Configuration has no output labels")
    rng = random.Random(20261113)
    paths_checked, decoded, mask_decoded = set(), set(), set()
    identities = defaultdict(dict)
    counts_by_source = defaultdict(Counter)
    maximum_details = 30
    missing_total = 0
    for split in ("train", "val", "test", "external_test"):
        manifest = routing.manifest_path(config, data_root, split)
        if not manifest.is_file():
            errors.append(f"Missing {split} manifest: {manifest}")
            continue
        try:
            rows = list(selected_rows(manifest, config, split))
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"Invalid routed {split} manifest: {error}")
            continue
        known = Counter()
        row_ids = set()
        split_sources = Counter()
        for row in rows:
            sample_id = row.get("sample_id")
            if sample_id in row_ids:
                errors.append(f"Repeated sample_id in {split}: {sample_id}")
            row_ids.add(sample_id)
            if row.get("split") != split or row.get("task") != config["task"]:
                errors.append(f"Task/split differs from manifest: {sample_id}")
            for label, available in collect_supervision(row, labels, kind, config).items():
                known[label] += int(available)
                if available and kind in {"multiclass", "multilabel"}:
                    value = row["labels"][label]
                    valid = isinstance(value, (int, float, bool)) and math.isfinite(value)
                    if kind == "multilabel":
                        valid = valid and value in (0, 1)
                    elif config.get("num_classes") is not None:
                        valid = valid and int(value) == value and 0 <= value < config["num_classes"]
                    if not valid and len(errors) < maximum_details:
                        errors.append(f"Invalid {label} label for {sample_id}: {value!r}")
                    counts_by_source[(row["dataset"], split, label)][str(value)] += 1
            split_sources[row["dataset"]] += 1
            references = (row.get("image_paths") or ([row["image_path"]] if row.get("image_path") else []))
            if not references:
                errors.append(f"No image path: {sample_id}")
            for name, values in (("sample_id", [sample_id]), ("image_path", references), ("sha256", row.get("image_sha256", [])),
                                 ("patient", [(row["dataset"], row["patient_id"])] if row.get("patient_id") else []),
                                 ("leakage_group", [row["leakage_group_id"]] if row.get("leakage_group_id") else [])):
                for value in values:
                    seen_split = identities[name].get(value)
                    if seen_split is not None and seen_split != split:
                        if len(errors) < maximum_details:
                            errors.append(f"{name} crosses {seen_split}/{split}: {value}")
                    identities[name][value] = split
            for reference in [*references, *runtime_annotations(row.get("annotations", {}))]:
                path = resolve_reference(data_root, reference)
                if path not in paths_checked:
                    paths_checked.add(path)
                    if not path.is_file():
                        missing_total += 1
                        if missing_total <= maximum_details:
                            errors.append(f"Required runtime file is missing: {reference}")
        report["splits"][split] = {"records": len(rows), "known_supervision": dict(known), "sources": dict(split_sources)}
        if split in {"train", "val"}:
            if not rows:
                errors.append(f"{split} is empty; use a protocol/source filter with training and validation samples")
            if rows and not sum(known.values()):
                errors.append(f"{split} has no known supervision for the configured labels")
            # Detection has image-level objects supervision. An annotated
            # negative image or a split without one lesion type is still a
            # valid detection target; class names are not label_mask keys.
            supervision_labels = ["objects"] if kind == "detection" else labels
            for label in supervision_labels:
                if known[label] == 0:
                    errors.append(f"{split} label {label!r} has no known supervision")
        # Stratify the decode samples by source, so one large dataset cannot
        # hide a broken smaller source from the preflight sample.
        groups = defaultdict(list)
        for row in rows:
            groups[row["dataset"]].append(row)
        for source_rows in groups.values():
            for row in rng.sample(source_rows, min(sample_count, len(source_rows))):
                try:
                    from PIL import Image
                    for value in row.get("image_paths", []):
                        path = resolve_reference(data_root, value)
                        if path not in decoded:
                            with Image.open(path) as image:
                                image.convert("RGB").load()
                            decoded.add(path)
                    for value in runtime_annotations(row.get("annotations", {})):
                        path = resolve_reference(data_root, value)
                        if path.suffix.lower() == ".png" and path not in mask_decoded:
                            with Image.open(path) as mask:
                                mask.load()
                                if mask.mode != "L" or not set(mask.getdata()).issubset({0, 255}):
                                    raise ValueError(f"Derived mask is not single-channel binary: {value}")
                            mask_decoded.add(path)
                        elif path.suffix.lower() == ".xml":
                            import xml.etree.ElementTree as ET
                            ET.parse(path)
                except Exception as error:
                    errors.append(f"Sample decode failed for {row.get('sample_id')}: {type(error).__name__}: {error}")
    if missing_total > maximum_details:
        errors.append(f"Additional missing files omitted: {missing_total - maximum_details}")
    for label in labels:
        train_counts, val_counts = Counter(), Counter()
        for (_, split, item), counts in counts_by_source.items():
            if item == label:
                if split == "train":
                    train_counts.update(counts)
                if split == "val":
                    val_counts.update(counts)
        if train_counts:
            if len(train_counts) < 2:
                warnings.append(f"Training label {label!r} contains one observed class: {dict(train_counts)}")
            elif min(train_counts.values()) / max(train_counts.values()) < .1:
                warnings.append(f"Training label {label!r} is imbalanced: {dict(train_counts)}")
    report["runtime_paths_checked"] = len(paths_checked)
    report["missing_runtime_files"] = missing_total
    report["images_decoded"] = len(decoded)
    report["binary_masks_decoded"] = len(mask_decoded)
    report["class_counts"] = [{"dataset": source, "split": split, "label": label, "counts": dict(counts)}
                              for (source, split, label), counts in sorted(counts_by_source.items())]
    report["limitations"] = ["Decoded images/masks are sampled; path existence and configured supervision are checked for all selected records.",
                             "File hashes are not recomputed. A passed preflight is not a completed training/accuracy evaluation.",
                             "Unknown patient identities prevent a claim of complete clinical patient independence."]
    report["all_checks_passed"] = not errors
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE / "configs/source_isolated/strict_cross_domain/dr_grading_ddr.json")
    parser.add_argument("--data-root", type=Path, default=BASE / "data")
    parser.add_argument("--samples-per-source", type=int, default=2)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.samples_per_source < 1:
        parser.error("--samples-per-source must be positive")
    try:
        config = load_config(args.config.resolve())
        report = run_checks(config, args.data_root.resolve(), args.samples_per_source, args.allow_cpu)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(0 if report["all_checks_passed"] else 1)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Preflight error: {error}\n")


if __name__ == "__main__":
    main()
