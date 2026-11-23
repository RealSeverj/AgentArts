"""Create read-only, task-specific candidate manifests for central split integration.

No images or original labels are moved, copied, resized or modified. Proposed
splits preserve source splits except that all clean IDRiD grading records are
reserved as external_test. The central preparation script owns final splits,
cross-source duplicate resolution, and EyeQ patient-grouped validation selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any


@lru_cache(maxsize=120000)
def relative(path: Path, data_root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(data_root.resolve()):
        raise ValueError(f"Path leaves data root: {path}")
    return resolved.relative_to(data_root.resolve()).as_posix()


def table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        result = []
        for source in reader:
            cleaned = {str(k).strip(): str(v).strip() for k, v in source.items() if k is not None and str(k).strip() and v is not None}
            if any(cleaned.values()):
                result.append(cleaned)
    return result


def integer(value: str, allowed: set[int], field: str) -> int:
    result = int(value)
    if str(result) != str(value).strip() or result not in allowed:
        raise ValueError(f"Invalid {field}: {value!r}")
    return result


def record(data_root: Path, dataset: str, image_id: str, source_split: str,
           proposed_split: str, image: Path, annotation: Path, task: str,
           labels: dict[str, Any], label_mask: dict[str, bool], representation: str,
           patient_id: str | None = None, eye: str | None = None,
           **extra: Any) -> dict[str, Any]:
    if not image.is_file() or not annotation.is_file():
        raise FileNotFoundError(f"Missing input image or annotation: {image}, {annotation}")
    if labels.keys() != label_mask.keys():
        raise ValueError("Each label must have an explicit availability mask")
    for key, available in label_mask.items():
        if available != (labels[key] is not None):
            raise ValueError(f"Availability differs from label presence: {key}")
    return {
        "sample_id": f"{task}:{dataset}:{source_split}:{image_id}",
        "image_id": f"{dataset}:{source_split}:{image_id}",
        "dataset": dataset,
        "source_split": source_split,
        "proposed_split": proposed_split,
        "image_path": relative(image, data_root),
        "patient_id": patient_id,
        "eye": eye,
        "task": task,
        "labels": labels,
        "label_mask": label_mask,
        "annotation_path": relative(annotation, data_root),
        "representation": representation,
        **extra,
    }


def ddr_candidates(data_root: Path) -> tuple[list[dict], dict]:
    base = data_root / "DDR" / "DDR-dataset" / "DR_grading"
    records = []
    stats = {}
    for source_split, proposed, expected in [("train", "train", 6835), ("valid", "val", 2733), ("test", "test", 4105)]:
        annotation = base / f"{source_split}.txt"
        lines = [line.strip() for line in annotation.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(lines) != expected:
            raise ValueError(f"DDR {source_split} count changed: {len(lines)}")
        ids = set()
        grades = Counter()
        for line in lines:
            filename, raw_grade = line.rsplit(None, 1)
            if filename in ids:
                raise ValueError("Duplicate filename in a DDR label split")
            ids.add(filename)
            grade = integer(raw_grade, set(range(6)), "DDR grade")
            grades[grade] += 1
            image = base / source_split / filename
            identity = Path(filename).stem
            common = dict(data_root=data_root, dataset="DDR", image_id=identity,
                          source_split=source_split, proposed_split=proposed, image=image,
                          annotation=annotation, representation="native_resolution_public_mirror",
                          source_grade_label=grade)
            # Class 5 denotes ungradable. It is never emitted as a DR grade.
            if grade < 5:
                records.append(record(**common, task="dr_grading", labels={"dr_grade": grade},
                                      label_mask={"dr_grade": True}))
            records.append(record(**common, task="image_quality",
                                  labels={"quality_grade": None, "ungradable": grade == 5},
                                  label_mask={"quality_grade": False, "ungradable": True},
                                  quality_label_definition="DDR source class5 vs source classes0-4; not EyeQ Reject"))
        actual_images = {p.name for p in (base / source_split).iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        if actual_images != ids:
            raise ValueError(f"DDR image-label mismatch in {source_split}")
        stats[source_split] = {"images": len(lines), "source_grade_counts": dict(sorted(grades.items()))}
    return records, stats


def aptos_candidates(data_root: Path) -> tuple[list[dict], dict]:
    base = data_root / "APTOS2019"
    index = base / "all_labeled_images.csv"
    rows = table(index)
    if len(rows) != 3662:
        raise ValueError("APTOS expected exactly 3662 labeled source records")
    csv_by_split = {"train_1": base / "public_mirror" / "train_1.csv",
                    "valid": base / "public_mirror" / "valid.csv",
                    "test": base / "public_mirror" / "test.csv"}
    original = {split: {r["id_code"]: integer(r["diagnosis"], set(range(5)), "APTOS grade")
                        for r in table(path)} for split, path in csv_by_split.items()}
    seen = set()
    records = []
    counts = Counter()
    for row in rows:
        identity, source_split = row["id_code"], row["mirror_split"]
        if identity in seen or source_split not in original:
            raise ValueError("APTOS duplicate image ID or unexpected mirror split")
        seen.add(identity)
        grade = integer(row["diagnosis"], set(range(5)), "APTOS grade")
        if original[source_split].get(identity) != grade:
            raise ValueError("APTOS combined index differs from downloaded source label")
        image = base / row["image_path"]
        records.append(record(data_root, "APTOS2019", identity, source_split,
                              {"train_1": "train", "valid": "val", "test": "test"}[source_split],
                              image, csv_by_split[source_split], "dr_grading", {"dr_grade": grade},
                              {"dr_grade": True}, "native_resolution_public_resplit_mirror",
                              split_definition="Mirror repartition of original labeled3662; not original competition public test"))
        counts[source_split] += 1
    if counts != Counter({"train_1": 2930, "valid": 366, "test": 366}):
        raise ValueError("APTOS source split counts changed")
    return records, {"mirror_split_counts": dict(counts), "patients_unknown": True}


def eyeq_candidates(data_root: Path) -> tuple[list[dict], dict]:
    base = data_root / "EyeQ"
    records = []
    stats = {}
    patients_by_split = defaultdict(set)
    for source_split, expected in [("train", 12543), ("test", 16249)]:
        annotation = base / "labels" / f"Label_EyeQ_{source_split}.csv"
        rows = table(annotation)
        if len(rows) != expected:
            raise ValueError(f"EyeQ {source_split} count changed")
        ids = set()
        quality_counts = Counter()
        grade_counts = Counter()
        for row in rows:
            filename = row["image"]
            if filename in ids:
                raise ValueError("Duplicate EyeQ filename within source split")
            ids.add(filename)
            match = re.fullmatch(r"(\d+)_(left|right)\.(?:jpeg|jpg|png)", filename, re.I)
            if not match:
                raise ValueError(f"Unexpected EyePACS filename format: {filename}")
            patient = "EyePACS:" + match.group(1)
            eye = match.group(2).lower()
            patients_by_split[source_split].add(patient)
            grade = integer(row["DR_grade"], set(range(5)), "EyeQ DR grade")
            quality = integer(row["quality"], {0, 1, 2}, "EyeQ quality")
            image = base / "images_480px_public_mirror" / f"resized_{source_split}" / filename
            common = dict(data_root=data_root, dataset="EyeQ", image_id=Path(filename).stem,
                          source_split=source_split, proposed_split=source_split, image=image,
                          annotation=annotation, representation="public_resized_480x480_jpeg",
                          patient_id=patient, eye=eye, source_quality_grade=quality,
                          source_data_family="EyePACS")
            records.append(record(**common, task="dr_grading", labels={"dr_grade": grade},
                                  label_mask={"dr_grade": True}))
            records.append(record(**common, task="image_quality",
                                  labels={"quality_grade": quality, "ungradable": None},
                                  label_mask={"quality_grade": True, "ungradable": False},
                                  quality_label_definition="EyeQ:0 Good,1 Usable,2 Reject; not raw-resolution EyePACS"))
            quality_counts[quality] += 1
            grade_counts[grade] += 1
        actual_images = {p.name for p in (base / "images_480px_public_mirror" / f"resized_{source_split}").iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        if actual_images != ids:
            raise ValueError("EyeQ image-label filenames differ")
        stats[source_split] = {"images": len(rows), "patients": len(patients_by_split[source_split]),
                               "quality_counts": dict(sorted(quality_counts.items())),
                               "dr_grade_counts": dict(sorted(grade_counts.items()))}
    overlap = sorted(patients_by_split["train"] & patients_by_split["test"])
    stats["patients_in_both_source_splits"] = overlap
    stats["validation_policy"] = "No local reassignment; central integrator must select val from source train by patient group"
    return records, stats


def idrid_candidates(data_root: Path) -> tuple[list[dict], list[dict], dict]:
    base = data_root / "IDRiD"
    grading = base / "extracted" / "B. Disease Grading"
    all_records = []
    byte_groups = defaultdict(list)
    for source_split, folder, filename, expected in [
        ("train", "a. Training Set", "a. IDRiD_Disease Grading_Training Labels.csv", 413),
        ("test", "b. Testing Set", "b. IDRiD_Disease Grading_Testing Labels.csv", 103),
    ]:
        annotation = grading / "2. Groundtruths" / filename
        rows = [r for r in table(annotation) if r.get("Image name")]
        if len(rows) != expected:
            raise ValueError("IDRiD source grading counts changed")
        seen = set()
        for row in rows:
            identity = row["Image name"]
            if identity in seen:
                raise ValueError("Duplicate IDRiD name within source split")
            seen.add(identity)
            grade = integer(row["Retinopathy grade"], set(range(5)), "IDRiD DR grade")
            dme = integer(row["Risk of macular edema"], {0, 1, 2}, "IDRiD DME risk")
            image = grading / "1. Original Images" / folder / (identity + ".jpg")
            item = record(data_root, "IDRiD", identity, source_split, "external_test", image, annotation,
                          "dr_grading", {"dr_grade": grade}, {"dr_grade": True},
                          "native_4288x2848_public_mirror", auxiliary_labels={"dme_risk_grade": dme},
                          split_definition="Whole IDRiD source reserved for external evaluation after quarantine")
            item["source_label"] = {"dr_grade": grade, "dme_risk_grade": dme}
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            item["file_sha256"] = digest
            byte_groups[digest].append(item)
            all_records.append(item)
        actual_images = {p.stem for p in (grading / "1. Original Images" / folder).glob("*.jpg")}
        if actual_images != seen:
            raise ValueError("IDRiD image-label IDs differ")
    kept, quarantine = [], []
    for digest, group in sorted(byte_groups.items()):
        label_combinations = {(r["labels"]["dr_grade"], r["auxiliary_labels"]["dme_risk_grade"]) for r in group}
        if len(label_combinations) > 1:
            for item in group:
                quarantine.append({**item, "reason": "conflicting_duplicate_labels", "duplicate_group_sha256": digest})
        else:
            # Prefer original test record where an identical, consistently labeled
            # image spans source splits; otherwise select the lexicographic ID.
            ordered = sorted(group, key=lambda r: (0 if r["source_split"] == "test" else 1, r["image_id"]))
            kept.append(ordered[0])
            for item in ordered[1:]:
                quarantine.append({**item, "reason": "same_label_exact_duplicate", "duplicate_group_sha256": digest,
                                   "representative_sample_id": ordered[0]["sample_id"]})
    stats = {"source_records": len(all_records), "distinct_file_sha256": len(byte_groups),
             "retained_external_records": len(kept), "quarantined": len(quarantine),
             "quarantine_reasons": dict(Counter(r["reason"] for r in quarantine)),
             "decision": "Conflicts in either DR or DME quarantine the whole exact-byte group; consistent exact duplicates retain one."}
    return kept, quarantine, stats


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project / "data")
    parser.add_argument("--output-root", type=Path,
                        default=project / "data" / "prepared" / "staging" / "grading_quality")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, quarantine, datasets = [], [], {}
    for name, producer in [("DDR", ddr_candidates), ("APTOS2019", aptos_candidates), ("EyeQ", eyeq_candidates)]:
        subset, stats = producer(data_root)
        records.extend(subset)
        datasets[name] = stats
    subset, rejected, stats = idrid_candidates(data_root)
    records.extend(subset)
    quarantine.extend(rejected)
    datasets["IDRiD"] = stats
    record_ids = [row["sample_id"] for row in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Candidate sample IDs are not unique")
    if set(record_ids) & {row["sample_id"] for row in quarantine}:
        raise ValueError("A quarantined record was retained")
    image_assignments = defaultdict(set)
    for row in records:
        image_assignments[row["image_path"]].add(row["proposed_split"])
    if any(len(splits) != 1 for splits in image_assignments.values()):
        raise ValueError("Same image assigned inconsistent proposed splits across tasks")
    counts = Counter((row["task"], row["dataset"], row["proposed_split"]) for row in records)
    report = {"candidate_records": len(records), "distinct_image_paths": len(image_assignments),
              "quarantine_records": len(quarantine), "all_checks_passed": True,
              "counts": [{"task": t, "dataset": d, "proposed_split": s, "records": n}
                         for (t, d, s), n in sorted(counts.items())], "datasets": datasets,
              "notes": ["Candidate splits are not the final centrally deduplicated/grouped splits.",
                        "FIVES quality is prepared by structural_tasks; it is not duplicated here.",
                        "EyeQ480 labels and image usage scopes remain as documented in source README.",
                        "This script does not inspect ODIR, resample pixels, copy images, or overwrite original labels."]}
    for filename, rows in [("candidates.jsonl", records), ("quarantine.jsonl", quarantine)]:
        with (output / filename).open("w", encoding="utf-8", newline="\n") as handle:
            for row in sorted(rows, key=lambda r: r["sample_id"]):
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    (output / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# DR grading and image-quality candidate manifests\n\n"
        "These are staging records for central integration, not the final split release. Images and source labels remain unchanged.\n\n"
        "- `dr_grading`: DDR grades0-4, APTOS labeled mirror, EyeQ source DR grades, and cleaned IDRiD reserved as external_test.\n"
        "- `image_quality`: EyeQ quality_grade0=Good/1=Usable/2=Reject, or DDR ungradable=(source class5). The two quality ontologies are not equated.\n"
        "- `label_mask` is true only for the corresponding known source label. Other targets remain null.\n"
        "- EyeQ patient_id/eye derive solely from EyePACS filename prefixes/suffixes. No patient IDs are guessed for other sources.\n"
        "- All same-image task records share their proposed split. Final central splitting must keep patients/duplicate groups together.\n"
        "- IDRiD exact-byte groups with conflicting DR or DME labels are entirely quarantined; consistent duplicates keep one record.\n"
        "- FIVES quality and other structural tasks are handled separately.\n\n"
        "See audit_report.json for exact candidate counts and quarantine.jsonl for per-record reasons.\n",
        encoding="utf-8")
    print(json.dumps({"output_root": str(output), "candidate_records": len(records),
                      "quarantine_records": len(quarantine), "datasets": {k: v for k, v in datasets.items() if k == "IDRiD"},
                      "all_checks_passed": True}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
