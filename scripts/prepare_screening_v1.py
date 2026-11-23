"""Prepare the two screening manifest contracts and verify their source bytes."""
import argparse
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test", "external_test")
LABELS = {
    "grouped": ["RVO", "LS", "DN", "AMD", "CSR", "OPTIC_DISC_CUPPING",
                "ODE", "ODP", "MYOPIC_FUNDUS", "TSLN", "ABNORMAL"],
    "glaucoma": ["glaucoma"],
}
DIRECT = {"LS": "LS", "DN": "DN", "AMD": "ARMD", "CSR": "CSR",
          "OPTIC_DISC_CUPPING": "ODC", "ODE": "ODE", "ODP": "ODP",
          "MYOPIC_FUNDUS": "MYA", "TSLN": "TSLN"}
SOURCES = {
    "grouped": {
        "train": ["RFMiD"], "val": ["RFMiD"], "test": ["RFMiD"],
        "external_test": ["RFMiD2"],
    },
    "glaucoma": {
        "train": ["REFUGE", "PAPILA"], "val": ["PAPILA"], "test": ["PAPILA"],
        "external_test": ["FIVES"],
    },
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def encode_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def binary(value):
    check(value is None or (isinstance(value, int) and value in (0, 1)),
          f"Invalid binary target: {value!r}")
    return int(value) if value is not None else None


def targets(row, task):
    if task == "glaucoma":
        label = binary(row["labels"]["glaucoma"])
        check(row["label_mask"]["glaucoma"] == (label is not None), "Invalid glaucoma mask")
        return [label]
    raw = row["raw_labels"]
    components = [binary(raw.get(key)) for key in ("BRVO", "CRVO")]
    values = {label: binary(raw.get(key)) for label, key in DIRECT.items()}
    values["RVO"] = 1 if 1 in components else (0 if components == [0, 0] else None)
    values["ABNORMAL"] = binary(raw["Disease_Risk"]) if row["dataset"] == "RFMiD" else None
    return [values[k] for k in LABELS[task]]


def check_file(job):
    path, expected, dims = job
    payload = path.read_bytes()
    check(digest(payload) == expected, f"Image hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as img:
        check(tuple(dims) == img.size, f"Image dimensions mismatch: {path}")
        img.convert("RGB").load()


def prepare(root, workers):
    data = root / "data"
    inputs = data / "prepared/source_isolated/strict_cross_domain/tasks"
    out = data / "prepared/screening_v1"
    baseline = root / "02_模型训练与评估/02_待训练方案/data_audit.json"
    expected = json.loads(baseline.read_text(encoding="utf-8-sig"))
    expected_hashes = {item["sha256"] for item in expected["source_files"].values()}
    identities = defaultdict(dict)
    jobs, file_info, sources, patients = {}, {}, {}, {}
    refuge_prefixes, refuge_examples = Counter(), defaultdict(list)
    payloads = {}

    for task, task_splits in SOURCES.items():
        src_task = "multidisease" if task == "grouped" else "glaucoma_classification"
        baseline_task = "grouped_screening" if task == "grouped" else "glaucoma"
        for split, datasets in task_splits.items():
            rows, support = [], [Counter() for _ in LABELS[task]]
            known_patients, positive_patients, all_patients = set(), set(), set()
            for dataset in datasets:
                source = inputs / src_task / dataset / f"{split}.jsonl"
                contents = source.read_bytes()
                sha = digest(contents)
                check(sha in expected_hashes, f"Unexpected source version: {source}")
                sources[source.relative_to(root).as_posix()] = sha
                for line in contents.decode("utf-8-sig").splitlines():
                    row = json.loads(line)
                    sid = row["sample_id"]
                    check(row["dataset"] == dataset and row["split"] == split, f"Split/source mismatch: {sid}")
                    relative = Path(row["image_path"])
                    check(not relative.is_absolute() and ".." not in relative.parts, f"Bad source path: {sid}")
                    path = data / relative
                    hashes = row["image_sha256"]
                    check(len(hashes) == 1 and row["image_paths"] == [row["image_path"]], f"Expected one image: {sid}")
                    image_sha = hashes[0]
                    image_rel = path.relative_to(root).as_posix()
                    job = (path, image_sha, row["image_sizes"][0])
                    previous_job = jobs.setdefault(image_rel, job)
                    check(job == previous_job, f"Inconsistent image identity: {sid}")
                    values = targets(row, task)
                    for counter, value in zip(support, values):
                        counter["unknown" if value is None else ("positive" if value else "negative")] += 1
                    pid = row["patient_id"]
                    record = {
                        "sample_id": sid, "image_path": image_rel, "dataset": dataset, "split": split,
                        "labels": [v if v is not None else 0 for v in values],
                        "mask": [v is not None for v in values],
                        "patient_id": pid, "content_group_id": row["content_group_id"],
                        "source_sample_id": row["source_sample_id"], "image_sha256": image_sha,
                        "leakage_group_id": row["leakage_group_id"],
                        "annotation_path": ("data/" + row["annotation_path"]) if row.get("annotation_path") else None,
                        "eye": row.get("eye"),
                    }
                    if task == "glaucoma":
                        record["raw_glaucoma_status"] = row["labels"]["glaucoma_status"]
                        if dataset == "PAPILA":
                            check(pid and (values[0] is None) == (record["raw_glaucoma_status"] == 2),
                                  f"Invalid PAPILA patient/status: {sid}")
                            all_patients.add(pid)
                            if values[0] is not None:
                                known_patients.add(pid)
                            if values[0] == 1:
                                positive_patients.add(pid)
                        if dataset == "REFUGE":
                            match = re.fullmatch(r"([gn])\d+", path.stem.lower())
                            check(match and int(match[1] == "g") == values[0], f"REFUGE prefix mismatch: {sid}")
                            prefix = match[1]
                            record["label_source"] = "mirror_training_filename_prefix"
                            refuge_prefixes[prefix] += 1
                            if len(refuge_examples[prefix]) < 3:
                                refuge_examples[prefix].append(path.name)
                    else:
                        record["annotation_row_1based"] = row.get("annotation_row_1based")
                    for key, value in (("sample_id", sid), ("image_path", image_rel),
                                       ("image_sha256", image_sha), ("image_uid", row["image_uid"]),
                                       ("content_group_id", row["content_group_id"]),
                                       ("leakage_group_id", row["leakage_group_id"]), ("patient_id", pid)):
                        if value:
                            prev = identities[key].setdefault(value, split)
                            check(prev == split, f"Cross-split overlap: {key}/{value}/{prev}/{split}")
                    rows.append(record)
            check(len({r["sample_id"] for r in rows}) == len(rows), f"Repeated sample ID: {task}/{split}")
            counts = {label: {term: ctr[term] for term in ("positive", "negative", "unknown")}
                      for label, ctr in zip(LABELS[task], support)}
            reference = expected[baseline_task]["split_support"][split]
            check(len(rows) == reference["records"] and counts == reference["labels"],
                  f"Support differs from plan: {task}/{split}")
            if task == "glaucoma":
                patients[split] = {"all": len(all_patients), "known": len(known_patients),
                                   "positive": len(positive_patients)}
            manifest = out / task / f"{split}.jsonl"
            payload = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows).encode()
            key = f"{task}/{split}"
            file_info[key] = {"path": manifest.relative_to(root).as_posix(), "sha256": digest(payload),
                              "records": len(rows), "support": counts}
            payloads[manifest] = payload

    check(set(sources.values()) == expected_hashes and len(jobs) == 5263, "Unexpected manifest or image count")
    verification = data / "REFUGE/verification_report.json"
    refuge_evidence = json.loads(verification.read_text(encoding="utf-8"))
    check(refuge_prefixes == {"g": 40, "n": 360} == refuge_evidence["splits"][0]["image_filename_prefix_counts"],
          "REFUGE prefix distribution differs from mirror verification")
    metadata = {
        "version": "screening_v1", "label_order": LABELS, "manifests": file_info,
        "source_manifests_sha256": sources, "plan_audit_sha256": digest(baseline.read_bytes()),
        "papila_distinct_patients": patients, "cross_split_identity_overlaps": {k: 0 for k in identities},
        "refuge": {"label_basis": "mirror training filenames; g=1, n=0", "counts": dict(refuge_prefixes),
                   "examples": dict(refuge_examples), "independent_local_disease_label_table": False,
                   "mirror_report": verification.relative_to(root).as_posix(),
                   "mirror_report_sha256": digest(verification.read_bytes())},
        "limitations": ["Patient IDs missing for RFMiD, RFMiD2, REFUGE, FIVES.",
                        "Image and group checks cannot exclude all transformed near duplicates."],
    }
    for name, payload in payloads.items():
        write_atomic(name, payload)
    audit = out / "audit.json"
    write_atomic(audit, encode_json({**metadata, "status": "image_verification_in_progress"}))
    print(f"Prepared {len(file_info)} manifests. Verifying {len(jobs)} unique source images.", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for n, _ in enumerate(executor.map(check_file, jobs.values()), start=1):
            if n % 500 == 0:
                print(f"Verified {n}/{len(jobs)}", flush=True)
    write_atomic(audit, encode_json({**metadata, "status": "verified", "images_rgb_decoded_sha256_verified": len(jobs)}))
    print(f"Verified all {len(jobs)} image hashes, dimensions, and RGB decodes. Receipt: {audit}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.workers)
