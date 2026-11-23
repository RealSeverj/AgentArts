"""Explicit task/source routing for new experiments; legacy reads are audit-only.

Source images and original protocol manifests stay immutable. Each new training
run names one source and reads physically separated, checksum-bound manifests.
External evaluation may name ONE separate source in external_dataset.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re

SPLITS = ("train", "val", "test", "external_test")
PROTOCOLS = ("strict_cross_domain", "within_source")
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SIGNATURES = {
    "dr_grading": ("multiclass", ["dr_grade"], 5, {"DDR", "APTOS2019"}, {"IDRiD"}),
    "disc_cup_segmentation": ("segmentation", ["optic_cup", "optic_disc"], None, {"PAPILA", "REFUGE"}, {"IDRiD"}),
    "glaucoma_classification": ("multilabel", ["glaucoma"], None, {"PAPILA", "REFUGE"}, set()),
    "lesion_segmentation": ("segmentation", ["EX", "HE", "MA", "SE"], None, {"DDR", "IDRiD"}, set()),
    "lesion_detection": ("detection", ["EX", "HE", "MA", "SE"], None, {"DDR"}, set()),
    "landmark_localization": ("landmark", ["fovea_center", "optic_disc_center"], None, {"IDRiD"}, set()),
    "vessel_segmentation": ("segmentation", ["vessel"], None, {"FIVES"}, set()),
    "multidisease": ("multilabel", ["DR", "AMD", "MYOPIC_FUNDUS", "OPTIC_DISC_CUPPING", "ABNORMAL"], None, {"RFMiD", "RFMiD2", "FIVES"}, set()),
    "odir_patient_multilabel": ("multilabel", ["N", "D", "G", "C", "A", "H", "M", "O"], None, {"ODIR-5K"}, set()),
}
QUALITY_SIGNATURES = {
    "EyeQ": ("multiclass", ["quality_grade"], 3),
    "DDR": ("multilabel", ["ungradable"], None),
    "FIVES": ("multilabel", ["fives_IC", "fives_Blur", "fives_LC"], None),
}


def safe_component(value, label):
    if not isinstance(value, str) or not _SAFE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def is_source_isolated(config):
    value = config.get("data", {}).get("source_isolated", False)
    if not isinstance(value, bool):
        raise ValueError("data.source_isolated must be boolean")
    return value


def expected_source(config, split):
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split!r}")
    if not is_source_isolated(config):
        return None
    data = config.get("data", {})
    sources = data.get("datasets")
    if not isinstance(sources, list) or len(sources) != 1:
        raise ValueError("An isolated run requires exactly one data.datasets source")
    source = safe_component(sources[0], "source")
    external = data.get("external_dataset")
    if external is not None:
        safe_component(external, "external source")
    return external if split == "external_test" and external is not None else source


def isolation_root(config, data_root):
    root = Path(data_root).resolve()
    value = config.get("data", {}).get("isolation_root", "prepared/source_isolated")
    if not isinstance(value, str) or not value or PureWindowsPath(value).drive or Path(value).is_absolute() or ".." in value.replace("\\", "/").split("/"):
        raise ValueError("data.isolation_root must be a contained relative data path")
    target = (root / value).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("Isolation root must stay inside data_root")
    return target


def manifest_path(config, data_root, split):
    if split not in SPLITS:
        raise ValueError("Unknown dataset split")
    task = safe_component(config["task"], "task")
    protocol = config.get("data", {}).get("protocol", "strict_cross_domain")
    if protocol not in PROTOCOLS:
        raise ValueError("Unknown data.protocol: " + str(protocol))
    if is_source_isolated(config):
        return isolation_root(config, data_root) / protocol / "tasks" / task / expected_source(config, split) / (split + ".jsonl")
    root = Path(data_root) / "prepared"
    if protocol == "within_source":
        root = root / "protocols/within_source"
    return root / "tasks" / task / (split + ".jsonl")


def validate_isolated_config(config):
    if not is_source_isolated(config):
        raise ValueError("New training requires data.source_isolated=true and a configs/source_isolated configuration; legacy mixed runs are retained for evaluation only")
    source = expected_source(config, "train")
    task = config["task"]
    if task == "image_quality":
        if source not in QUALITY_SIGNATURES:
            raise ValueError("Unsupported quality-label source")
        kind, labels, classes = QUALITY_SIGNATURES[source]
        allowed = {source}
    else:
        if task not in SIGNATURES:
            raise ValueError("Task has no reviewed source/label contract")
        kind, labels, classes, train_sources, evaluation_sources = SIGNATURES[task]
        if source not in train_sources:
            raise ValueError(f"Source {source!r} is not enabled for training task {task!r}")
        allowed = train_sources | evaluation_sources
    if config.get("kind") != kind or config.get("labels") != labels or (classes is not None and config.get("num_classes") != classes):
        raise ValueError("Configured task, kind and output labels violate the source-specific label contract")
    if expected_source(config, "external_test") not in allowed:
        raise ValueError("External source has no compatible label contract for this task")
    return source


def validate_row(config, row, split):
    if not is_source_isolated(config):
        return
    if row.get("dataset") != expected_source(config, split) or row.get("task") != config["task"] or row.get("split") != split:
        raise ValueError(f"Task/source/split contamination at {row.get('sample_id')!r}: expected {config['task']}/{expected_source(config, split)}/{split}")
    if config.get("kind") in {"multiclass", "multilabel"}:
        for label in config["labels"]:
            known = row.get("label_mask", {}).get(label, False)
            if not isinstance(known, bool):
                raise ValueError("Label masks must remain boolean")
            if not known:
                continue
            value = row.get("labels", {}).get(label)
            if config["kind"] == "multiclass":
                if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < config["num_classes"]:
                    raise ValueError("Invalid known multiclass label")
            elif value not in (0, 1, False, True):
                raise ValueError("Invalid known binary label")


def config_for_split(config, split):
    result = deepcopy(config)
    if is_source_isolated(config):
        result["data"]["datasets"] = [expected_source(config, split)]
    return result


def validate_training_route(config, data_root):
    source = validate_isolated_config(config)
    root = isolation_root(config, data_root)
    index_path = root / "isolation_manifest.json"
    payload = index_path.read_bytes()
    index = json.loads(payload.decode("utf-8-sig"))
    digest = hashlib.sha256(payload).hexdigest()
    protocol = config.get("data", {}).get("protocol", "strict_cross_domain")
    if (config.get("data", {}).get("isolation_manifest_sha256") != digest or index.get("all_checks_passed") is not True
            or index.get("format_version") != 1 or index.get("record_bytes_preserved") is not True
            or protocol not in index.get("base_protocols", [])):
        raise ValueError("Isolation index is missing, changed or not successfully audited")
    entries = {entry["path"]: entry for combination in index["combinations"] for entry in combination["splits"].values()}
    counts, fingerprints, seen, train_labels = {}, {}, {}, {label: Counter() for label in config["labels"]}
    required = ["objects"] if config["kind"] == "detection" else config["labels"]
    for split in ("train", "val"):
        path = manifest_path(config, data_root, split)
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        info = entries.get(relative)
        if not info or info["sha256"] != checksum or info["bytes"] != len(data):
            raise ValueError("Selected manifest differs from its audited isolation index")
        known, n, effective, identifiers = Counter(), 0, 0, set()
        for line in data.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            validate_row(config, row, split)
            if not row.get("sample_id") or row["sample_id"] in identifiers:
                raise ValueError("Sample IDs must be nonempty and unique in each isolated manifest")
            identifiers.add(row["sample_id"])
            n += 1
            for label in required:
                known[label] += bool(row.get("label_mask", {}).get(label, False))
            effective += any(row.get("label_mask", {}).get(label, False) for label in required)
            if split == "train" and config["kind"] in {"multiclass", "multilabel"}:
                for label in config["labels"]:
                    if row.get("label_mask", {}).get(label, False):
                        train_labels[label][str(int(row["labels"][label]))] += 1
            keys = [("sample_id", row.get("sample_id")), ("patient", row.get("patient_id")),
                    ("leakage_group", row.get("leakage_group_id")), ("content_group", row.get("content_group_id"))]
            keys += [("image", x.replace("\\", "/").casefold()) for x in row.get("image_paths", [])]
            keys += [("sha256", x) for x in row.get("image_sha256", [])]
            for kind, value in keys:
                if value is None or value == "":
                    continue
                key = (kind, source if kind == "patient" else "", str(value))
                if key in seen and seen[key] != split:
                    raise ValueError(f"{kind} crosses train/val in isolated source")
                seen[key] = split
        if n != info["records"]:
            raise ValueError("Selected manifest row count differs from its audited index")
        if not n or any(known[k] == 0 for k in required):
            raise ValueError(f"Isolated {split} lacks configured supervision: {dict(known)}")
        counts[split], fingerprints[split] = {"records": n, "effective_records": effective, "known_supervision": dict(known)}, checksum
    loss = config.get("loss", {})
    recorded_counts = loss.get("training_label_counts")
    if recorded_counts is not None and recorded_counts != {label: dict(values) for label, values in train_labels.items()}:
        raise ValueError("Loss label counts belong to another source; recompute source-only training weights")
    weight_source = loss.get("weighting_source")
    if weight_source is not None and weight_source != {"dataset": source, "base_protocol": protocol, "split": "train", "manifest_sha256": fingerprints["train"]}:
        raise ValueError("Loss weights are not bound to this source's training manifest")
    if config.get("data", {}).get("image_cache_dir"):
        if config.get("augmentation", {}).get("resize_before_augmentation") is not True:
            raise ValueError("A resized training cache requires explicit resize_before_augmentation=true")
        cache = json.loads((Path(config["data"]["image_cache_dir"]) / "cache_manifest.json").read_text(encoding="utf-8-sig"))
        from training.fundus_preprocessing import preprocessing_spec
        from training.image_cache import validate_manifest
        image_size = config.get("model", {}).get("img_size", 384)
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            image_size = (image_size[1], image_size[0])
        validate_manifest(cache, image_size,
                          expected_preprocessing=preprocessing_spec(config))
        if (cache.get("task") != config["task"] or cache.get("manifest_sha256") != fingerprints
                or cache.get("datasets") != [source] or cache.get("protocol") != protocol
                or cache.get("source_isolated") is not True):
            raise ValueError("Image cache belongs to another source/task manifest; rebuild a separate cache")
    return {"source": source, "protocol": config.get("data", {}).get("protocol", "strict_cross_domain"),
            "splits": counts, "isolation_manifest_sha256": digest}


def validate_suite_configs(configs):
    if not configs:
        raise ValueError("System expert list is empty")
    for config in configs:
        validate_isolated_config(config)
    protocols = {c.get("data", {}).get("protocol", "strict_cross_domain") for c in configs}
    if len(protocols) != 1:
        raise ValueError("A system evaluation cannot mix strict_cross_domain and within_source experts; IDRiD/FIVES holdouts would overlap expert training")
    identities = {c.get("data", {}).get("isolation_manifest_sha256") for c in configs}
    roots = {c.get("data", {}).get("isolation_root", "prepared/source_isolated") for c in configs}
    if len(identities) != 1 or None in identities or len(roots) != 1:
        raise ValueError("All system experts must share one audited global split identity")
    training_sources = {expected_source(c, "train") for c in configs}
    external_sources = {c.get("data", {}).get("external_dataset") for c in configs} - {None}
    if training_sources & external_sources:
        raise ValueError("A declared system external source is already used to train another expert")
    return {"protocol": next(iter(protocols)), "training_sources": sorted(training_sources), "external_sources": sorted(external_sources)}
