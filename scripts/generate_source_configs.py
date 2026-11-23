"""Generate audited single-source configs from unchanged legacy templates.

Training weights use only the chosen source's training supervision. Evaluation
labels never enter weighting or automatic source selection. Existing outputs
are reused only when identical; conflicting files are never silently replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

try:
    from scripts.cloud_preflight import collect_supervision
except ModuleNotFoundError:
    from cloud_preflight import collect_supervision

PROJECT = Path(__file__).resolve().parents[1]
PROTOCOLS = ("strict_cross_domain", "within_source")
SPLITS = ("train", "val", "test", "external_test")
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_slug(source):
    if not isinstance(source, str) or not SAFE.fullmatch(source):
        raise ValueError(f"Unsafe source identifier: {source!r}")
    return re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")


def safe_member(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Manifest member must be a relative path string")
    normalized = relative.replace("\\", "/")
    member = PurePosixPath(normalized)
    if member.is_absolute() or PureWindowsPath(normalized).drive or ".." in member.parts or ":" in normalized:
        raise ValueError(f"Unsafe isolated manifest member: {relative!r}")
    result = (root / member.as_posix()).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Isolated manifest member leaves its root")
    return result


def read_combination(combo, root, template):
    protocol, task, source = combo["base_protocol"], combo["task"], combo["source"]
    if protocol not in PROTOCOLS or not SAFE.fullmatch(task) or not SAFE.fullmatch(source):
        raise ValueError("Unsafe isolated combination identity")
    keys = ["objects"] if template["kind"] == "detection" else template["labels"]
    stats = {}
    for split in SPLITS:
        info = combo["splits"][split]
        expected = f"{protocol}/tasks/{task}/{source}/{split}.jsonl"
        if info.get("path") != expected:
            raise ValueError(f"Isolated manifest path differs from its identity: {expected}")
        path = safe_member(root, info["path"])
        if not path.is_file() or digest_file(path) != info["sha256"] or path.stat().st_size != info["bytes"]:
            raise ValueError(f"Isolated source manifest fails bytes/SHA256 verification: {path}")
        known = {label: 0 for label in keys}
        counts = {label: Counter() for label in template["labels"]}
        object_presence = {label: 0 for label in template["labels"]}
        records = effective = 0
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("task") != task or row.get("dataset") != source or row.get("split") != split:
                    raise ValueError(f"Cross-source/task/split record in isolated manifest: {path}")
                records += 1
                available = collect_supervision(row, template["labels"], template["kind"], template)
                effective += int(any(available.values()))
                for label, flag in available.items():
                    known[label] += int(flag)
                    if flag and template["kind"] in ("multiclass", "multilabel"):
                        value = row["labels"][label]
                        if template["kind"] == "multiclass":
                            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < int(template["num_classes"]):
                                raise ValueError(f"Invalid class in {path}: {label}={value!r}")
                        elif value not in (False, True, 0, 1):
                            raise ValueError(f"Invalid binary label in {path}: {label}={value!r}")
                        counts[label][int(value)] += 1
                if template["kind"] == "detection" and available.get("objects"):
                    for label, count in row.get("annotations", {}).get("object_counts", {}).items():
                        if label in object_presence and count > 0:
                            object_presence[label] += 1
        if records != info["records"]:
            raise ValueError(f"Isolated record count differs from index: {path}")
        stats[split] = {"records": records, "effective_records": effective,
                        "known_supervision_by_label": known,
                        "classification_label_counts": {label: {str(k): v for k, v in sorted(values.items())}
                                                         for label, values in counts.items()},
                        "detection_class_positive_records_declared": object_presence,
                        "manifest_sha256": info["sha256"]}
    return stats


def training_weights(template, stats, source, protocol):
    """Return a fresh loss configuration and blocking training-data issues."""
    loss = copy.deepcopy(template.get("loss", {}))
    for key in ("class_weights", "pos_weight", "weighting_note", "weighting_source", "training_label_counts"):
        loss.pop(key, None)
    issues = []
    train, val = stats["train"], stats["val"]
    for split in ("train", "val"):
        if not stats[split]["effective_records"]:
            issues.append(f"no_effective_{split}_supervision")
        for label, count in stats[split]["known_supervision_by_label"].items():
            if not count:
                issues.append(f"no_{split}_supervision_for_output:{label}")
    if template["kind"] == "multiclass":
        label = template["labels"][0]
        counts = train["classification_label_counts"][label]
        missing = [i for i in range(int(template["num_classes"])) if not counts.get(str(i), 0)]
        issues.extend(f"missing_training_class:{label}:{i}" for i in missing)
        if not missing:
            raw = [math.sqrt(1.0 / counts[str(i)]) for i in range(int(template["num_classes"]))]
            mean = sum(raw) / len(raw)
            loss["class_weights"] = [value / mean for value in raw]
    elif template["kind"] == "multilabel":
        weights = []
        for label in template["labels"]:
            counts = train["classification_label_counts"][label]
            negative, positive = counts.get("0", 0), counts.get("1", 0)
            if not positive:
                issues.append(f"no_positive_training_label:{label}")
            if not negative:
                issues.append(f"no_negative_training_label:{label}")
            if positive and negative:
                weights.append(min(10.0, math.sqrt(negative / positive)))
        if len(weights) == len(template["labels"]):
            loss["pos_weight"] = weights
    elif template["kind"] == "detection":
        for label, count in train["detection_class_positive_records_declared"].items():
            if not count:
                issues.append(f"no_declared_positive_training_objects:{label}")
    if template["kind"] in ("multiclass", "multilabel"):
        loss["training_label_counts"] = train["classification_label_counts"]
        loss["weighting_source"] = {"dataset": source, "base_protocol": protocol, "split": "train",
                                    "manifest_sha256": train["manifest_sha256"]}
        loss["weighting_note"] = ("Recomputed from this source's known training labels only: multiclass sqrt inverse frequency normalized to mean 1; "
                                  "multilabel sqrt(negative/positive) capped at 10. No validation/test labels used for weighting.")
    return loss, sorted(set(issues))


def emit_identical_or_new(path, payload):
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8-sig")) != payload:
            raise ValueError(f"Existing generated output differs; refusing overwrite: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8-sig")) != payload:
                raise ValueError(f"Generated output appeared with different contents: {path}")
        else:
            os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "created"


def generate_configs(template_dir, data_root, output_root=None, isolation_root=None, include_idrid_external=False):
    template_dir, data_root = Path(template_dir).resolve(), Path(data_root).resolve()
    output_root = Path(output_root).resolve() if output_root else template_dir / "source_isolated"
    if output_root == template_dir:
        raise ValueError("Generated output root must not be the legacy template directory")
    isolation_root = Path(isolation_root).resolve() if isolation_root else data_root / "prepared/source_isolated"
    if not isolation_root.is_relative_to(data_root):
        raise ValueError("Isolation root must be data-relative for portable configs")
    index_path = isolation_root / "isolation_manifest.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Publish and verify isolated manifests before config generation: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    if (index.get("format_version") != 1 or index.get("all_checks_passed") is not True or
            index.get("record_bytes_preserved") is not True or index.get("dry_run", False)):
        raise ValueError("Isolation index is not a verified, published version-1 record-preserving manifest")
    index_hash = digest_file(index_path)
    combinations = {}
    for combo in index["combinations"]:
        key = (combo["base_protocol"], combo["task"], combo["source"])
        if key in combinations:
            raise ValueError(f"Duplicate isolated source combination: {key}")
        combinations[key] = combo
    planned = {}
    registry = {"format_version": 1, "generator_version": 1,
                "isolation_manifest_sha256": index_hash,
                "isolation_identity_sha256": index.get("identity_sha256"),
                "default_system_base_protocol": "strict_cross_domain", "mixing_base_protocols_allowed": False,
                "default_entry": None, "include_idrid_external": bool(include_idrid_external),
                "protocol_groups": {p: {"default_system_eligible": p == "strict_cross_domain",
                                         "independent_experiments_only": p == "within_source",
                                         "generated": [], "prohibited_or_unrouted": []} for p in PROTOCOLS},
                "templates": [],
                "rules": ["Each config has exactly one training/validation/test source in data.datasets.",
                          "An explicit external_dataset is used only for external_test, never concatenated into train/val/test.",
                          "EyeQ/EyePACS DR records remain audit-only by default; DR training candidates are DDR and APTOS2019.",
                          "Weights use the isolated source's known training labels only; missing training classes/positives/negatives prohibit default training.",
                          "Models, augmentation and training hyperparameters are preserved from legacy templates; no automatic B/L architecture replacement occurs.",
                          "Within-source FIVES/IDRiD experts are independent experiments and must not be combined with the strict system while claiming untouched source evaluation.",
                          "One image can support multiple tasks, but task label routing remains independent.",
                          "Only the named default is eligible for automatic start; DINOv3-L configs are opt-in candidates."]}
    routed = set()
    templates = sorted(template_dir.glob("*.json"))
    if not templates:
        raise ValueError("No legacy JSON templates found")
    for path in templates:
        if not SAFE.fullmatch(path.stem):
            raise ValueError(f"Unsafe template filename: {path.name}")
        template = json.loads(path.read_text(encoding="utf-8-sig"))
        protocol = template.get("data", {}).get("protocol", "strict_cross_domain")
        if protocol not in PROTOCOLS:
            raise ValueError(f"Unsupported template protocol: {protocol}")
        allowed = set(template.get("data", {}).get("datasets") or [])
        registry["templates"].append({"filename": path.name, "sha256": digest_file(path), "base_protocol": protocol})
        matches = [combo for key, combo in sorted(combinations.items()) if key[:2] == (protocol, template["task"])]
        if not matches:
            registry["protocol_groups"][protocol]["prohibited_or_unrouted"].append({"template": path.name, "task": template["task"],
                                                                                     "reasons": ["no_isolated_source_combination_for_template_protocol"]})
        for combo in matches:
            source = combo["source"]
            routed.add((protocol, template["task"], source))
            stats = read_combination(combo, isolation_root, template)
            loss, issues = training_weights(template, stats, source, protocol)
            if allowed and source not in allowed:
                issues.append("excluded_by_existing_template_source_filter")
            if template["task"] == "dr_grading" and source not in ("DDR", "APTOS2019"):
                issues.append("dr_source_retained_for_audit_not_default_training")
            info = {"template": path.name, "base_protocol": protocol, "task": template["task"],
                    "kind": template["kind"], "source": source,
                    "effective_records": {split: stats[split]["effective_records"] for split in SPLITS},
                    "known_supervision_by_split": {split: stats[split]["known_supervision_by_label"] for split in SPLITS},
                    "training_label_counts": stats["train"]["classification_label_counts"]}
            if issues:
                info.update(training_start_allowed=False, reasons=sorted(set(issues)))
                registry["protocol_groups"][protocol]["prohibited_or_unrouted"].append(info)
                continue
            generated = copy.deepcopy(template)
            data = generated.setdefault("data", {})
            data.pop("image_cache_dir", None)
            data.pop("image_cache_root", None)
            data.pop("external_dataset", None)
            data.update(datasets=[source], source_isolated=True,
                        isolation_root=isolation_root.relative_to(data_root).as_posix(),
                        isolation_manifest_sha256=index_hash)
            generated["loss"] = loss
            external = combinations.get((protocol, template["task"], "IDRiD"))
            external_info = None
            if (include_idrid_external and protocol == "strict_cross_domain" and source == "DDR" and
                    template["task"] in ("dr_grading", "lesion_segmentation") and external is not None):
                ext_stats = read_combination(external, isolation_root, template)
                if ext_stats["external_test"]["effective_records"]:
                    data["external_dataset"] = "IDRiD"
                    external_info = {"dataset": "IDRiD", "split": "external_test",
                                     "effective_records": ext_stats["external_test"]["effective_records"],
                                     "training_records_added": 0}
            relative = f"{protocol}/{path.stem}_{source_slug(source)}.json"
            if relative in planned:
                raise ValueError(f"Portable source slug/config path collision: {relative}")
            planned[relative] = generated
            large_candidate = "large" in path.stem.lower() or "large" in generated.get("model", {}).get("name", "").lower()
            is_default = relative == "strict_cross_domain/dr_grading_ddr.json"
            info.update(config=relative, training_start_allowed=True,
                        eligible_for_default_system=protocol == "strict_cross_domain",
                        candidate_only=large_candidate,
                        automatic_start_allowed=is_default and not large_candidate,
                        selected_external_evaluation=external_info)
            registry["protocol_groups"][protocol]["generated"].append(info)
            if is_default:
                registry["default_entry"] = relative
    for key, combo in sorted(combinations.items()):
        if key not in routed:
            registry["protocol_groups"][key[0]]["prohibited_or_unrouted"].append({"base_protocol": key[0], "task": key[1], "source": key[2],
                                                                                "training_start_allowed": False,
                                                                                "reasons": ["no_template_with_matching_task_and_base_protocol"],
                                                                                "manifest_declared_trainable": combo.get("trainable", False)})
    registry["generated_config_count"] = len(planned)
    registry["blocked_or_unrouted_count"] = sum(len(g["prohibited_or_unrouted"]) for g in registry["protocol_groups"].values())
    planned["registry.json"] = registry
    # Verify the entire planned set before writing any file, so a conflicting
    # rerun cannot partially replace an existing set of experiment definitions.
    if output_root.exists():
        extras = {p.relative_to(output_root).as_posix() for p in output_root.rglob("*") if p.is_file()} - set(planned)
        if extras:
            raise ValueError(f"Generated config root has unexpected files; refusing overwrite/reuse: {sorted(extras)[:5]}")
    for relative, payload in planned.items():
        target = output_root / relative
        if target.exists() and json.loads(target.read_text(encoding="utf-8-sig")) != payload:
            raise ValueError(f"Existing generated output differs; refusing overwrite: {target}")
    statuses = Counter(emit_identical_or_new(output_root / relative, payload) for relative, payload in planned.items())
    return {"output_root": str(output_root), "generated_config_count": registry["generated_config_count"],
            "blocked_or_unrouted_count": registry["blocked_or_unrouted_count"],
            "default_entry": registry["default_entry"], "file_actions": dict(statuses), "registry": registry}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-dir", type=Path, default=PROJECT / "configs")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--isolation-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--include-idrid-external", action="store_true")
    args = parser.parse_args()
    result = generate_configs(args.template_dir, args.data_root, args.output_root, args.isolation_root, args.include_idrid_external)
    print(json.dumps({key: value for key, value in result.items() if key != "registry"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
