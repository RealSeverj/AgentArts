"""Build task manifests with globally consistent, leakage-aware partitions.

Run source preparation scripts and image_fingerprints.py first. Originals stay
read-only. Default protocol reserves FIVES/IDRiD/RFMiD2 for external evaluation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import os
import shutil
import time
import numpy as np

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = DATA / "prepared"
SEED = 20261113
SPLITS = ["train", "val", "test", "external_test"]
PRIORITY = {"train": 0, "val": 1, "test": 2, "external_test": 3}


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def jsonl_read(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def jsonl_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def image_paths(row):
    values = row.get("image_paths") or ([row["image_path"]] if row.get("image_path") else [])
    return [str(value).replace("\\", "/") for value in values]


def load_candidates():
    records, quarantine = [], []
    for module in ["multidisease", "grading_quality", "structural_tasks", "odir"]:
        path = OUT / "staging" / module / "candidates.jsonl"
        inputs = [path] if path.is_file() else sorted(path.parent.glob("*.jsonl")) if module == "structural_tasks" else []
        inputs = [item for item in inputs if item.name != "quarantine.jsonl"]
        if not inputs:
            raise RuntimeError(f"Missing completed preparation module: {module}")
        for row in [row for item in inputs for row in jsonl_read(item)]:
            row["preparation_module"] = module
            paths = image_paths(row)
            canonical = []
            for value in paths:
                if value.startswith("IDRiD/extracted/C. Localization/1. Original Images/"):
                    replacement = value.replace("C. Localization/1. Original Images/", "B. Disease Grading/1. Original Images/", 1)
                    if hashlib.sha256((DATA / value).read_bytes()).digest() != hashlib.sha256((DATA / replacement).read_bytes()).digest():
                        raise ValueError("IDRiD localization copy differs from canonical grading image")
                    row["source_image_path"] = value
                    value = replacement
                canonical.append(value)
            paths = canonical
            if row["task"] == "multidisease" and "abnormality" in row["labels"]:
                row["labels"]["ABNORMAL"] = row["labels"].pop("abnormality")
                row["label_mask"]["ABNORMAL"] = row["label_mask"].pop("abnormality")
            if not paths:
                raise ValueError("Candidate without input images")
            row["image_paths"] = paths
            row["image_path"] = paths[0] if len(paths) == 1 else None
            record_key = row["task"] + "|" + row["dataset"] + "|" + "|".join(paths)
            row["source_sample_id"] = row.get("sample_id")
            row["sample_id"] = row["task"] + ":" + hashlib.sha256(record_key.encode()).hexdigest()[:20]
            row["image_uid"] = hashlib.sha256("|".join(paths).encode()).hexdigest()[:24]
            row.setdefault("annotations", {})
            row.setdefault("patient_id", None)
            row.setdefault("eye", None)
            row.setdefault("input_unit", "image" if len(paths) == 1 else "patient_pair")
            records.append(row)
        qpath = path.parent / "quarantine.jsonl"
        if qpath.is_file():
            for row in jsonl_read(qpath):
                row["preparation_module"] = module
                row["reason"] = row.get("reason") or row.get("quarantine_reason", "source_preparation_exclusion")
                quarantine.append(row)
    ids = [row["sample_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task records produced by source modules")
    return records, quarantine


def materialize_annotation_assets(records):
    prefix = "prepared/staging/structural_tasks/derived_masks/"
    linked = set()
    def visit(value):
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            if normalized.startswith(prefix):
                source = DATA / normalized
                relative = normalized[len(prefix):]
                target = OUT / "assets" / "masks" / relative
                if not source.is_file():
                    raise ValueError(f"Missing derived mask: {normalized}")
                if not target.resolve().is_relative_to((OUT / "assets").resolve()):
                    raise ValueError("Derived asset leaves prepared assets directory")
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)
                elif target.stat().st_size != source.stat().st_size:
                    raise ValueError("Existing prepared asset differs; review before replacing")
                linked.add(target)
                return target.relative_to(DATA).as_posix()
        return value
    for row in records:
        row["annotations"] = visit(row.get("annotations", {}))
    print(json.dumps({"event": "annotation_assets_materialized", "files": len(linked)}), flush=True)


def load_fingerprints():
    database = sqlite3.connect(OUT / "audit" / "image_fingerprints.sqlite")
    database.row_factory = sqlite3.Row
    fingerprints = [dict(row) for row in database.execute("SELECT * FROM images ORDER BY path")]
    database.close()
    return fingerprints


def build_image_groups(fingerprints):
    union = UnionFind(len(fingerprints))
    paths = {row["path"]: index for index, row in enumerate(fingerprints)}
    signature = hashlib.sha256(("identity-policy-v1|" + "|".join(str(row["path"]) + str(row["sha256"]) + str(row["phash"]) for row in fingerprints)).encode()).hexdigest()
    cache = OUT / "audit" / "image_identity_cache.json"
    if cache.is_file():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored.get("signature") == signature and len(stored.get("groups", [])) == len(fingerprints):
            for index, group in enumerate(stored["groups"]):
                union.union(index, group)
            print(json.dumps({"event": "identity_groups_loaded_from_cache", "images": len(fingerprints)}), flush=True)
            return union, paths
    first_hash, ddr_ids = {}, {}
    for index, row in enumerate(fingerprints):
        if row["error"]:
            continue
        checksum = row["sha256"]
        if checksum in first_hash:
            union.union(index, first_hash[checksum])
        else:
            first_hash[checksum] = index
        # DDR intentionally reuses original image IDs across its grading and
        # lesion annotation directories. Other datasets may reset their IDs.
        if row["path"].startswith("DDR/"):
            key = Path(row["path"]).stem
            if key in ddr_ids:
                union.union(index, ddr_ids[key])
            else:
                ddr_ids[key] = index
    buckets = defaultdict(list)
    near_pairs = []
    for index, row in enumerate(fingerprints):
        if row["error"] or not row["phash"] or row["gray_std"] < 5:
            continue
        phash = int(row["phash"], 16)
        dhash = int(row["dhash"], 16)
        candidates = set()
        for band in range(4):
            candidates.update(buckets[(band, (phash >> (16 * band)) & 65535)])
        for other in candidates:
            reference = fingerprints[other]
            if reference["sha256"] == row["sha256"]:
                continue
            pdistance = (int(reference["phash"], 16) ^ phash).bit_count()
            if pdistance > 2 or (int(reference["dhash"], 16) ^ dhash).bit_count() > 4:
                continue
            ratio = row["width"] / row["height"]
            rratio = reference["width"] / reference["height"]
            if abs(ratio / rratio - 1) > 0.025:
                continue
            a = np.frombuffer(row["thumb"], dtype=np.uint8).astype(np.float32)
            b = np.frombuffer(reference["thumb"], dtype=np.uint8).astype(np.float32)
            rmse = float(np.sqrt(np.mean((a - b) ** 2)) / 255)
            if rmse > 0.02:
                continue
            ag = a.reshape(-1, 3).mean(1); bg = b.reshape(-1, 3).mean(1)
            ag -= ag.mean(); bg -= bg.mean()
            denominator = float(np.linalg.norm(ag) * np.linalg.norm(bg))
            correlation = float(ag.dot(bg) / denominator) if denominator else 0
            if correlation < 0.9997:
                continue
            union.union(index, other)
            near_pairs.append({"image_a": reference["path"], "image_b": row["path"], "phash_distance": pdistance,
                               "thumbnail_rgb_rmse_0_1": rmse, "thumbnail_gray_correlation": correlation,
                               "status": "conservative_visual_match_candidate_grouped_for_split_only"})
        for band in range(4):
            buckets[(band, (phash >> (16 * band)) & 65535)].append(index)
    jsonl_write(OUT / "audit" / "near_duplicate_candidates.jsonl", near_pairs)
    exact = defaultdict(list)
    for row in fingerprints:
        if not row["error"]:
            exact[row["sha256"]].append(row["path"])
    exact_groups = [{"sha256": key, "image_paths": value} for key, value in exact.items() if len(value) > 1]
    jsonl_write(OUT / "audit" / "exact_duplicate_groups.jsonl", exact_groups)
    json_write(cache, {"signature": signature, "groups": [union.find(index) for index in range(len(fingerprints))]})
    print(json.dumps({"event": "image_identity_groups", "exact_duplicate_groups": len(exact_groups), "visual_match_pairs": len(near_pairs)}), flush=True)
    return union, paths


def build_leakage_groups(records, image_union, paths):
    union = UnionFind(len(image_union.parent))
    for index in range(len(union.parent)):
        union.union(index, image_union.find(index))
    patients = {}
    for row in records:
        ids = [paths[path] for path in row["image_paths"]]
        for index in ids[1:]:
            union.union(ids[0], index)
        patient = row.get("patient_id")
        if patient:
            if patient in patients:
                union.union(ids[0], patients[patient])
            else:
                patients[patient] = ids[0]
    return union


def exclusions(records, source_quarantine, fingerprints, image_union, paths):
    bad_image_groups, findings = set(), []
    for index, fingerprint in enumerate(fingerprints):
        if fingerprint["error"]:
            bad_image_groups.add(image_union.find(index))
            findings.append({"reason": "image_decode_error", "image_path": fingerprint["path"], "error": fingerprint["error"]})
    critical = ["conflicting_duplicate_labels", "chinese_english_label_disagreement", "anterior_segment_photo"]
    for row in source_quarantine:
        if any(reason in row["reason"] for reason in critical):
            for path in image_paths(row):
                if path in paths:
                    bad_image_groups.add(image_union.find(paths[path]))
    labels_by_group = defaultdict(lambda: defaultdict(set))
    records_by_group = defaultdict(list)
    for row in records:
        if len(row["image_paths"]) != 1:
            continue
        group = image_union.find(paths[row["image_paths"][0]])
        records_by_group[group].append(row["sample_id"])
        if row["task"] not in {"dr_grading", "multidisease"}:
            continue
        for label, value in row["labels"].items():
            if row["label_mask"].get(label) and value is not None:
                labels_by_group[group][(row["task"], label)].add(value)
    for group, annotations in labels_by_group.items():
        conflicts = {task + ":" + label: sorted(values) for (task, label), values in annotations.items() if len(values) > 1}
        if conflicts:
            bad_image_groups.add(group)
            findings.append({"reason": "global_duplicate_target_conflict", "image_group": group, "conflicting_labels": conflicts,
                             "sample_ids": records_by_group[group]})
    jsonl_write(OUT / "audit" / "global_label_conflicts.jsonl", findings)
    return bad_image_groups


def original_split(row):
    split = str(row["source_split"]).lower()
    if "valid" in split or split in {"val", "validation"}:
        return "val"
    if "test" in split:
        return "test"
    if "train" in split:
        return "train"
    raise ValueError(f"Unknown original source split: {row['dataset']} {split}")


def assign_splits(records, leakage, paths, protocol):
    assignments = {}
    grouped_records = defaultdict(list)
    for row in records:
        group = leakage.find(paths[row["image_paths"][0]])
        row["_group"] = group
        split = row["proposed_split"]
        if protocol == "within_source" and row["dataset"] in {"FIVES", "IDRiD", "RFMiD2"}:
            split = original_split(row)
        grouped_records[group].append(row)
        if PRIORITY[split] > PRIORITY.get(assignments.get(group), -1):
            assignments[group] = split
    reserve_sources = {"EyeQ", "ODIR-5K"}
    if protocol == "within_source":
        reserve_sources.update({"FIVES", "IDRiD"})
    # Stratify train-only groups, never individual eyes. Histograms and original
    # splits remain available so a changed benchmark is not misrepresented.
    for dataset in sorted(reserve_sources):
        strata = defaultdict(list)
        for group, rows in grouped_records.items():
            own = [row for row in rows if row["dataset"] == dataset]
            if not own or assignments[group] != "train":
                continue
            if dataset == "ODIR-5K":
                key = tuple(max(row["labels"].get(code, 0) for row in own) for code in ["N", "D", "G", "C", "A", "H", "M", "O"])
            else:
                grades = [row["labels"].get("dr_grade") for row in own if row["label_mask"].get("dr_grade")]
                qualities = [row["labels"].get("quality_grade") for row in own if row["label_mask"].get("quality_grade")]
                key = (max(grades) if grades else -1, max(qualities) if qualities else -1)
            strata[str(key)].append(group)
        for stratum, groups in strata.items():
            groups.sort(key=lambda group: hashlib.sha256((str(SEED) + dataset + stratum + min(p for row in grouped_records[group] for p in row["image_paths"])).encode()).hexdigest())
            count = min(len(groups) - 1, max(1, round(len(groups) * 0.15))) if len(groups) >= 2 else 0
            for group in groups[:count]:
                assignments[group] = "val"
    return assignments


def build_protocol(records, source_quarantine, fingerprints, image_union, leakage, paths, bad_groups, protocol):
    root = OUT if protocol == "strict_cross_domain" else OUT / "protocols" / protocol
    kept, quarantined = [], []
    # Propagate critically uncertain image labels through each patient group too.
    bad_leakage = {leakage.find(index) for index in range(len(fingerprints)) if image_union.find(index) in bad_groups}
    for original in records:
        row = copy.deepcopy(original)
        group = leakage.find(paths[row["image_paths"][0]])
        if group in bad_leakage:
            quarantined.append({**row, "reason": "critical_image_or_patient_group_quarantined"})
            continue
        if not any(row.get("label_mask", {}).values()) and not row.get("annotations"):
            quarantined.append({**row, "reason": "no_known_supervised_target"})
            continue
        kept.append(row)
    assignments = assign_splits(kept, leakage, paths, protocol)
    # Same SHA image appears once per task. Distinct tasks retain their own
    # targets, while patient-related images stay in a single global partition.
    dedup = {}
    fingerprint_by_path = {row["path"]: row for row in fingerprints}
    for row in sorted(kept, key=lambda r: (-PRIORITY[assignments[r["_group"]]], -PRIORITY[r["proposed_split"]], r["dataset"], r["sample_id"])):
        if protocol == "strict_cross_domain" and assignments[row["_group"]] == "external_test" and row["dataset"] not in {"FIVES", "IDRiD", "RFMiD2"}:
            quarantined.append({**row, "reason": "overlap_with_reserved_external_source"})
            continue
        hashes = tuple(fingerprint_by_path[path]["sha256"] for path in row["image_paths"])
        key = (row["task"], hashes)
        if key in dedup:
            quarantined.append({**row, "reason": "same_task_exact_duplicate", "representative_sample_id": dedup[key]["sample_id"]})
            continue
        dedup[key] = row
    final = []
    for row in dedup.values():
        group = row.pop("_group")
        row["split"] = assignments[group]
        row["protocol"] = protocol
        row["leakage_group_id"] = f"group_{group:06d}"
        content_groups = [image_union.find(paths[path]) for path in row["image_paths"]]
        row["content_group_id"] = "content_" + hashlib.sha256(json.dumps(content_groups).encode()).hexdigest()[:20]
        row["image_sha256"] = [fingerprint_by_path[path]["sha256"] for path in row["image_paths"]]
        row["image_sizes"] = [[fingerprint_by_path[path]["width"], fingerprint_by_path[path]["height"]] for path in row["image_paths"]]
        row["split_changed_from_proposal"] = row["split"] != row["proposed_split"]
        final.append(row)
    final.sort(key=lambda r: (r["task"], r["split"], r["dataset"], r["sample_id"]))
    group_sizes = Counter((row["task"], row["split"], row["content_group_id"]) for row in final)
    for row in final:
        row["content_group_evaluation_weight"] = 1.0 / group_sizes[(row["task"], row["split"], row["content_group_id"])]
    tasks = sorted({row["task"] for row in final})
    scalar_keys = {task: {key for row in final if row["task"] == task for key in row["labels"]} for task in tasks}
    for row in final:
        for key in scalar_keys[row["task"]]:
            if key not in row["labels"]:
                row["labels"][key] = None
                row["label_mask"][key] = False
    stats = {}
    for task in tasks:
        stats[task] = {}
        for split in SPLITS:
            rows = [row for row in final if row["task"] == task and row["split"] == split]
            jsonl_write(root / "tasks" / task / (split + ".jsonl"), rows)
            stats[task][split] = len(rows)
    for split in SPLITS:
        jsonl_write(root / "all_tasks" / (split + ".jsonl"), [row for row in final if row["split"] == split])
    by_source = Counter((row["task"], row["dataset"], row["split"]) for row in final)
    with (root / "split_counts.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["task", "dataset", *SPLITS, "total_records"])
        for task, dataset in sorted({(row["task"], row["dataset"]) for row in final}):
            values = [by_source[(task, dataset, split)] for split in SPLITS]
            writer.writerow([task, dataset, *values, sum(values)])
    class_counts = Counter()
    for row in final:
        for key, value in row["labels"].items():
            known = bool(row["label_mask"].get(key))
            class_counts[(row["task"], row["dataset"], row["split"], key, str(value) if known else "__UNKNOWN__")] += 1
    with (root / "class_counts.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["task", "dataset", "split", "label", "value", "records"])
        for key, count in sorted(class_counts.items()):
            writer.writerow([*key, count])
    jsonl_write(root / "quarantine" / "records.jsonl", source_quarantine + quarantined)
    summary = {"protocol": protocol, "seed": SEED, "task_records": len(final),
               "unique_image_paths": len({path for row in final for path in row["image_paths"]}),
               "unique_image_sha256": len({value for row in final for value in row["image_sha256"]}),
               "known_patients": len({row["patient_id"] for row in final if row.get("patient_id")}),
               "tasks": stats, "source_preparation_quarantine_records": len(source_quarantine),
               "central_excluded_task_records": len(quarantined), "split_reassigned_task_records": sum(row["split_changed_from_proposal"] for row in final)}
    json_write(root / "summary.json", summary)
    schema = {}
    for task in tasks:
        own = [row for row in final if row["task"] == task]
        available = {key for row in own for key in set(row["labels"]) | set(row["label_mask"])}
        preferred = {"multidisease": ["DR", "AMD", "MYOPIC_FUNDUS", "OPTIC_DISC_CUPPING", "ABNORMAL"],
                     "odir_patient_multilabel": ["N", "D", "G", "C", "A", "H", "M", "O"]}.get(task, [])
        keys = [key for key in preferred if key in available] + sorted(available - set(preferred))
        schema[task] = {"label_order": keys, "observed_values": {key: sorted({str(row["labels"][key]) for row in own if row["label_mask"].get(key) and key in row["labels"]}) for key in keys},
                        "input_units": sorted({row["input_unit"] for row in own}), "known_label_mask_required": True}
    json_write(root / "label_schema.json", schema)
    print(json.dumps({"event": "protocol_built", **summary}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["both", "strict_cross_domain", "within_source"], default="both")
    args = parser.parse_args()
    records, quarantine = load_candidates()
    materialize_annotation_assets(records)
    fingerprints = load_fingerprints()
    indexed = {row["path"] for row in fingerprints}
    missing = {path for row in records for path in row["image_paths"] if path not in indexed}
    if missing:
        raise ValueError(f"Candidate paths missing fingerprints: {list(missing)[:10]}")
    image_union, paths = build_image_groups(fingerprints)
    leakage = build_leakage_groups(records, image_union, paths)
    bad_groups = exclusions(records, quarantine, fingerprints, image_union, paths)
    json_write(OUT / "split_policy.json", {"default": "strict_cross_domain", "seed": SEED, "held_out_sources": ["FIVES", "IDRiD", "RFMiD2"],
               "holdout_priority": SPLITS[::-1], "validation_fraction_for_missing_val": 0.15,
               "grouping": ["file_SHA256", "conservative_visual_match_candidates", "known_patient_ID", "ODIR_pair", "DDR_original_image_ID"],
               "limitations": ["Unknown patient IDs cannot be invented; image deduplication does not prove patient independence.",
                              "Perceptual screening is conservative and does not establish absence of every crop, rotation or near duplicate.",
                              "within_source is a separate benchmark; never use its training records and report the strict external test as untouched."]})
    protocols = ["strict_cross_domain", "within_source"] if args.protocol == "both" else [args.protocol]
    for protocol in protocols:
        build_protocol(records, quarantine, fingerprints, image_union, leakage, paths, bad_groups, protocol)


if __name__ == "__main__":
    main()
