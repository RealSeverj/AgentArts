"""Prepare RFMiD multi-disease candidate manifests without changing source data.

These manifests are staging inputs. The top-level preparation process must still
apply cross-dataset duplicate/patient grouping and choose final task splits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
LABELS = ["DR", "AMD", "MYOPIC_FUNDUS", "OPTIC_DISC_CUPPING", "ABNORMAL"]
DIRECT_MAPPING = {
    "DR": "DR",
    "AMD": "ARMD",
    "MYOPIC_FUNDUS": "MYA",
    "OPTIC_DISC_CUPPING": "ODC",
}
SOURCES = [
    {"dataset": "RFMiD", "source_split": "train", "proposed_split": "train", "rows": 1920,
     "csv": "RFMiD/Training_Set/RFMiD_Training_Labels.csv", "images": "RFMiD/Training_Set/Training", "extension": ".png"},
    {"dataset": "RFMiD", "source_split": "val", "proposed_split": "val", "rows": 640,
     "csv": "RFMiD/Evaluation_Set/RFMiD_Validation_Labels.csv", "images": "RFMiD/Evaluation_Set/Validation", "extension": ".png"},
    {"dataset": "RFMiD", "source_split": "test", "proposed_split": "test", "rows": 640,
     "csv": "RFMiD/Test_Set/RFMiD_Testing_Labels.csv", "images": "RFMiD/Test_Set/Test", "extension": ".png"},
    {"dataset": "RFMiD2", "source_split": "train", "proposed_split": "external_test", "rows": 509,
     "csv": "RFMiD2/extracted/Training_set/RFMiD_2_Training_labels.csv", "images": "RFMiD2/extracted/Training_set", "extension": ".jpg"},
    {"dataset": "RFMiD2", "source_split": "val", "proposed_split": "external_test", "rows": 177,
     "csv": "RFMiD2/extracted/Validation_set/RFMiD_2_Validation_labels.csv", "images": "RFMiD2/extracted/Validation_set", "extension": ".jpg"},
    {"dataset": "RFMiD2", "source_split": "test", "proposed_split": "external_test", "rows": 174,
     "csv": "RFMiD2/extracted/Test_set/RFMiD_2_Testing_labels.csv", "images": "RFMiD2/extracted/Test_set", "extension": ".jpg"},
]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def save_json(path: Path, obj: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def normalize_header(header: str) -> str:
    return header.replace("\u00a0", " ").strip()


def read_table(path: Path, dataset: str) -> tuple[list[str], list[list[str]], str]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        if dataset != "RFMiD2":
            raise
        text = raw.decode("cp1252")
        encoding = "cp1252"
    table = list(csv.reader(io.StringIO(text, newline="")))
    if not table:
        raise ValueError(f"Empty source table: {path}")
    headers = table[0]
    named = [normalize_header(header) for header in headers if normalize_header(header)]
    if len(set(named)) != len(named) or named[0] != "ID":
        raise ValueError(f"Unexpected duplicate or missing named headers in {path}")
    rows = [row for row in table[1:] if any(value.strip() for value in row)]
    if any(len(row) != len(headers) for row in rows):
        raise ValueError(f"CSV row/header width mismatch: {path}")
    return headers, rows, encoding


def parse_labels(headers: list[str], row: list[str]) -> tuple[str, dict, dict, list[dict]]:
    normal = {}
    raw_labels = {}
    unnamed = []
    source_id = row[0].strip()
    if not source_id.isdigit() or str(int(source_id)) != source_id:
        raise ValueError(f"Unexpected source ID: {source_id!r}")
    for index, (header, value) in enumerate(zip(headers, row), 1):
        key = normalize_header(header)
        value = value.strip()
        if key == "ID":
            continue
        if not key:
            # Unnamed source columns are retained, never interpreted as a class.
            unnamed.append({"column_number_1based": index, "header": header, "value": value})
            raw_labels[header] = value
            continue
        if value not in {"0", "1"}:
            raise ValueError(f"Nonbinary source label {header!r}={value!r} at ID {source_id}")
        raw_labels[header] = int(value)
        normal[key] = int(value)
    return source_id, raw_labels, normal, unnamed


def build(data_root: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    accepted_before_duplicates = []
    quarantine = []
    source_reports = []
    schema_reports = {}

    for source in SOURCES:
        dataset = source["dataset"]
        table_path = data_root / source["csv"]
        image_root = data_root / source["images"]
        headers, rows, encoding = read_table(table_path, dataset)
        if len(rows) != source["rows"]:
            raise ValueError(f"Source row count changed: {source['csv']}: {len(rows)}")
        image_paths = sorted(p for p in image_root.iterdir() if p.is_file() and p.suffix.lower() == source["extension"])
        image_by_id = {path.stem: path for path in image_paths}
        if len(image_by_id) != len(image_paths):
            raise ValueError("Duplicate same-stem images within source split")
        row_ids = []
        missing_images = []
        schema_reports[source["csv"]] = {
            "encoding_used": encoding,
            "original_headers": headers,
            "header_normalization": [{"original": header, "normalized": normalize_header(header) or None} for header in headers],
            "csv_sha256": digest(table_path),
        }
        for line_number, row in enumerate(rows, 2):
            source_id, raw_labels, named_labels, unnamed = parse_labels(headers, row)
            if source_id in row_ids:
                raise ValueError(f"Duplicate label ID within split: {source['csv']} {source_id}")
            row_ids.append(source_id)
            labels = {key: named_labels.get(column) for key, column in DIRECT_MAPPING.items()}
            # RFMiD2 WNL is retained as a source-normal label; no cross-dataset
            # abnormality equivalence is silently assumed in this staging task.
            labels["ABNORMAL"] = named_labels.get("Disease_Risk") if dataset == "RFMiD" else None
            mask = {key: labels[key] is not None for key in LABELS}
            path = image_by_id.get(source_id)
            record = {
                "sample_id": f"{dataset}:{source['source_split']}:{source_id}",
                "dataset": dataset,
                "source_id": source_id,
                "source_split": source["source_split"],
                "proposed_split": source["proposed_split"],
                "image_path": path.relative_to(data_root).as_posix() if path else None,
                "patient_id": None,
                "eye": None,
                "task": "multidisease",
                "labels": labels,
                "label_mask": mask,
                "annotation_path": source["csv"],
                "annotation_row_1based": line_number,
                "representation": "published_png" if dataset == "RFMiD" else "published_jpeg",
                "modality": "color_fundus",
                "raw_labels": raw_labels,
                "raw_unmapped_columns": unnamed,
                "source_normal_label": {"WNL": named_labels["WNL"]} if "WNL" in named_labels else None,
                "normalization_policy": "source_binary_labels_only; unknown_labels_are_null_and_masked",
                "license": "CC BY 4.0",
            }
            if path is None:
                missing_images.append(source_id)
                quarantine.append({**record, "quarantine_reason": "label_id_has_no_same_split_image", "expected_image_path": (Path(source["images"]) / (source_id + source["extension"])).as_posix()})
                continue
            if dataset == "RFMiD":
                disease_labels = {key: value for key, value in named_labels.items() if key != "Disease_Risk"}
                if bool(named_labels["Disease_Risk"]) != any(disease_labels.values()):
                    quarantine.append({**record, "quarantine_reason": "source_abnormality_and_disease_labels_inconsistent"})
                    continue
            else:
                record["image_sha256"] = digest(path)
            accepted_before_duplicates.append(record)
        orphan_ids = sorted(set(image_by_id) - set(row_ids))
        for source_id in orphan_ids:
            path = image_by_id[source_id]
            quarantine.append({
                "sample_id": f"{dataset}:{source['source_split']}:{source_id}:unlabelled",
                "dataset": dataset, "source_id": source_id, "source_split": source["source_split"],
                "proposed_split": source["proposed_split"], "image_path": path.relative_to(data_root).as_posix(),
                "patient_id": None, "eye": None, "task": "multidisease",
                "labels": {label: None for label in LABELS}, "label_mask": {label: False for label in LABELS},
                "annotation_path": None, "representation": "published_jpeg", "raw_labels": {},
                "image_sha256": digest(path), "quarantine_reason": "image_id_has_no_same_split_annotation",
            })
        source_reports.append({
            "dataset": dataset, "source_split": source["source_split"], "proposed_split": source["proposed_split"],
            "annotation_rows": len(rows), "image_files": len(image_paths), "matched_ids": len(set(row_ids) & set(image_by_id)),
            "missing_image_ids": missing_images, "orphan_image_ids": orphan_ids,
        })

    # RFMiD2 has known cross-split exact copies. Isolate every member rather than
    # choosing a label, a source split or a duplicate representative arbitrarily.
    hashes = defaultdict(list)
    for record in accepted_before_duplicates:
        if record["dataset"] == "RFMiD2":
            hashes[record["image_sha256"]].append(record)
    duplicate_groups = []
    duplicate_ids = set()
    for sha, group in sorted(hashes.items()):
        if len(group) < 2:
            continue
        duplicate_groups.append({"sha256": sha, "sample_ids": [record["sample_id"] for record in group],
                                 "primary_label_conflict": len({json.dumps(record["labels"], sort_keys=True) for record in group}) > 1})
        for record in group:
            duplicate_ids.add(record["sample_id"])
            quarantine.append({**record, "quarantine_reason": "exact_duplicate_image_group_all_members_isolated", "duplicate_group_sha256": sha})
    candidates = [record for record in accepted_before_duplicates if record["sample_id"] not in duplicate_ids]
    candidates.sort(key=lambda record: (record["dataset"], record["source_split"], int(record["source_id"])))
    quarantine.sort(key=lambda record: (record["dataset"], record["source_split"], int(record["source_id"]), record["quarantine_reason"]))

    # Candidate-manifest contract checks, without decoding or rewriting the images.
    ids = [record["sample_id"] for record in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Nonunique candidate sample IDs")
    for record in candidates:
        if not (data_root / record["image_path"]).is_file():
            raise ValueError("Candidate points to a missing image")
        if set(record["labels"]) != set(LABELS) or set(record["label_mask"]) != set(LABELS):
            raise ValueError("Canonical target schema mismatch")
        for label in LABELS:
            if record["label_mask"][label] != (record["labels"][label] is not None):
                raise ValueError("Label mask disagrees with missingness")
            if record["label_mask"][label] and record["labels"][label] not in {0, 1}:
                raise ValueError("Candidate contains a nonbinary known label")
    if any(record["dataset"] == "RFMiD2" and record["proposed_split"] != "external_test" for record in candidates):
        raise ValueError("RFMiD2 accidentally assigned to development splits")

    save_jsonl(output_root / "candidates.jsonl", candidates)
    save_jsonl(output_root / "quarantine.jsonl", quarantine)
    for split in ["train", "val", "test", "external_test"]:
        save_jsonl(output_root / f"{split}.jsonl", [record for record in candidates if record["proposed_split"] == split])
    by_split = {}
    for split in ["train", "val", "test", "external_test"]:
        records = [record for record in candidates if record["proposed_split"] == split]
        by_split[split] = {"samples": len(records),
                           "known_labels": {label: sum(record["label_mask"][label] for record in records) for label in LABELS},
                           "positive_labels": {label: sum(record["labels"][label] == 1 for record in records if record["label_mask"][label]) for label in LABELS}}
    summary = {"status": "staging_only_requires_global_leakage_audit", "task": "multidisease",
               "candidate_samples": len(candidates), "candidate_datasets": dict(Counter(record["dataset"] for record in candidates)),
               "by_proposed_split": by_split, "quarantine_records": len(quarantine),
               "quarantine_reasons": dict(Counter(record["quarantine_reason"] for record in quarantine)),
               "rfmid2_exact_duplicate_groups": duplicate_groups, "source_reports": source_reports,
               "checks": {"unique_sample_ids": True, "all_image_paths_exist": True, "canonical_labels_and_masks_valid": True,
                          "rfmid2_no_duplicate_sha_in_candidates": len([record for record in candidates if record["dataset"] == "RFMiD2"]) == len({record["image_sha256"] for record in candidates if record["dataset"] == "RFMiD2"}),
                          "original_files_modified": False, "image_files_copied": False},
               "known_limits": ["Patient IDs and laterality are not provided; both remain null.",
                                "OPTIC_DISC_CUPPING is a structural-abnormality label, not clinical GLAUCOMA.",
                                "RFMiD2 WNL is retained without automatically mapping it to RFMiD Disease_Risk.",
                                "Cross-dataset exact/perceptual duplicates and final split assignment belong to the top-level audit."]}
    save_json(output_root / "summary.json", summary)
    save_json(output_root / "label_schema.json", {"task": "multidisease", "canonical_labels": LABELS,
              "direct_mapping": DIRECT_MAPPING, "ABNORMAL": {"RFMiD": "Disease_Risk", "RFMiD2": None},
              "source_headers": schema_reports, "unknown_label": {"value": None, "label_mask": False},
              "negative_label": {"value": 0, "label_mask": True},
              "all_raw_named_labels_preserved": True, "unnamed_source_columns": "Retained but never treated as a disease target."})
    (output_root / "README.md").write_text(
        "# RFMiD 多病种候选清单\n\n"
        "这些是暂存候选，必须由顶层流程完成跨来源图像/患者重复检查和最终划分后使用。原始图像和标注只读，未复制大图。\n\n"
        "RFMiD 保留原始 train/val/test；RFMiD2 所有可用记录保留原来源集合名，但作为 external_test 候选，不能用于训练或调阈值。\n\n"
        "标准标签：DR、AMD、MYOPIC_FUNDUS、OPTIC_DISC_CUPPING、ABNORMAL。ODC 仅映射视盘凹陷，不等同青光眼确诊。ABNORMAL 只使用 RFMiD Disease_Risk；RFMiD2 WNL 保留在 raw_labels/source_normal_label，标准异常标签为 null、mask=false。\n\n"
        "raw_labels 保留来源标签字段（包括 ON 后的不换行空格及未命名列）；规范化对应见 label_schema.json。未命名列无法解释为某个疾病，因此不作为监督目标。未知标签绝不当作阴性。\n\n"
        "image_path 和 annotation_path 均相对项目 data/；source_id 只能在 dataset+source_split 范围解释。patient_id、eye 未由来源提供，均为 null。\n\n"
        "错配标注、无标注图像及 RFMiD2 完全重复图像的全部成员保存在 quarantine.jsonl，未猜测纠正配对。候选与隔离计数、阳性标签统计及核验结果见 summary.json。\n",
        encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else data_root / "prepared/staging/multidisease"
    summary = build(data_root, output_root)
    print(json.dumps({"output": str(output_root), "candidates": summary["candidate_samples"],
                      "datasets": summary["candidate_datasets"], "quarantine_records": summary["quarantine_records"],
                      "split_counts": {split: value["samples"] for split, value in summary["by_proposed_split"].items()}}, ensure_ascii=True))


if __name__ == "__main__":
    main()
