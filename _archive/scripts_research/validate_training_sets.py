"""Independently validate prepared ophthalmology manifest protocols.

Only task manifests are counted as records. all_tasks/ is checked as a redundant
index, not counted twice. Original files are read-only. PNG/JPEG/BMP dimensions
are read from headers; raw TIFF masks are never opened with Pillow.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import struct
import time

BASE = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test", "external_test")
REQUIRED = {"sample_id", "image_uid", "image_path", "image_paths", "split", "task", "dataset",
            "patient_id", "labels", "label_mask", "annotations", "image_sha256", "leakage_group_id"}
FILE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".xml", ".txt", ".csv",
                 ".tsv", ".json", ".jsonl", ".xlsx", ".xls", ".py", ".pdf", ".md", ".yaml", ".yml"}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def read_image_dimensions(path: Path) -> tuple[int, int]:
    """Read dimensions only; never decompress pixels or invoke libtiff."""
    with path.open("rb") as handle:
        header = handle.read(32)
        if header.startswith(PNG_MAGIC):
            if len(header) < 24 or header[12:16] != b"IHDR":
                raise ValueError("Malformed PNG IHDR")
            width, height = struct.unpack(">II", header[16:24])
        elif header[:2] == b"BM":
            if len(header) < 26:
                raise ValueError("Truncated BMP header")
            dib_size = struct.unpack("<I", header[14:18])[0]
            if dib_size == 12:
                width, height = struct.unpack("<HH", header[18:22])
            elif dib_size >= 40:
                width, signed_height = struct.unpack("<ii", header[18:26])
                height = abs(signed_height)
            else:
                raise ValueError(f"Unsupported BMP DIB header: {dib_size}")
        elif header[:2] == b"\xff\xd8":
            handle.seek(2)
            sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
            while True:
                value = handle.read(1)
                if not value:
                    raise ValueError("JPEG has no SOF dimensions")
                if value != b"\xff":
                    continue
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if not marker:
                    raise ValueError("Truncated JPEG marker")
                code = marker[0]
                if code in {0x00, 0x01, 0xD8} or 0xD0 <= code <= 0xD7:
                    continue
                if code in {0xD9, 0xDA}:
                    raise ValueError("JPEG ended before SOF dimensions")
                length_bytes = handle.read(2)
                if len(length_bytes) != 2:
                    raise ValueError("Truncated JPEG segment")
                length = struct.unpack(">H", length_bytes)[0]
                if length < 2:
                    raise ValueError("Invalid JPEG segment length")
                if code in sof_markers:
                    dimensions = handle.read(5)
                    if len(dimensions) != 5:
                        raise ValueError("Truncated JPEG SOF")
                    height, width = struct.unpack(">HH", dimensions[1:5])
                    break
                handle.seek(length - 2, os.SEEK_CUR)
        elif header[:4] in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}:
            raise ValueError("TIFF image input is unsupported by this header-only validator; raw annotation TIFF is not decoded")
        else:
            raise ValueError("Unrecognized PNG/JPEG/BMP image signature")
    if width <= 0 or height <= 0:
        raise ValueError("Nonpositive image dimensions")
    return width, height


def bool_mask(value) -> bool:
    return isinstance(value, (bool, int)) and value in (False, True)


def looks_like_path(text: str) -> bool:
    if not text or "\n" in text or "://" in text or text.startswith("data:"):
        return False
    return Path(text.replace("\\", "/")).suffix.lower() in FILE_SUFFIXES


def annotation_references(value, location="annotations", key=None):
    """Yield local file references, preserving nested expert/point dictionaries."""
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from annotation_references(child, f"{location}.{child_key}", str(child_key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from annotation_references(child, f"{location}[{index}]", key)
    elif isinstance(value, str):
        explicit = key in {"path", "source_path", "annotation_path", "xml_path"} or bool(key and key.endswith("_path"))
        if explicit or looks_like_path(value):
            yield location, value


def annotation_supports(label: str, annotations: dict) -> bool:
    masks = annotations.get("masks")
    if isinstance(masks, dict) and masks.get(label) is not None:
        return True
    experts = annotations.get("experts")
    if isinstance(experts, dict):
        if any(isinstance(expert, dict) and expert.get(label) is not None for expert in experts.values()):
            return True
    points = annotations.get("points")
    if isinstance(points, dict) and points.get(label) is not None:
        return True
    if label == "objects" and annotations.get("xml_path"):
        return True
    return False


class FileChecks:
    def __init__(self, data_root: Path, rehash: bool):
        self.data_root = data_root.resolve()
        self.rehash = rehash
        self.path_cache = {}
        self.dimension_cache = {}
        self.sha_cache = {}

    def resolve(self, text):
        if not isinstance(text, str) or not text:
            raise ValueError("Path must be a nonempty string")
        normalized = text.replace("\\", "/")
        if Path(normalized).is_absolute() or PureWindowsPath(normalized).drive:
            raise ValueError("Manifest path must be relative to data/")
        if normalized in self.path_cache:
            result = self.path_cache[normalized]
            if isinstance(result, Exception):
                raise result
            return result
        try:
            result = (self.data_root / normalized).resolve()
            if not result.is_relative_to(self.data_root):
                raise ValueError("Path leaves data/")
            if not result.is_file():
                raise FileNotFoundError(f"Referenced file is missing: {normalized}")
            self.path_cache[normalized] = result
            return result
        except (OSError, ValueError) as error:
            self.path_cache[normalized] = error
            raise

    def dimensions(self, path: Path):
        key = str(path).casefold()
        if key not in self.dimension_cache:
            try:
                self.dimension_cache[key] = read_image_dimensions(path)
            except (OSError, ValueError) as error:
                self.dimension_cache[key] = error
        result = self.dimension_cache[key]
        if isinstance(result, Exception):
            raise result
        return result

    def sha256(self, path: Path):
        key = str(path).casefold()
        if key not in self.sha_cache:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    hasher.update(block)
            self.sha_cache[key] = hasher.hexdigest()
        return self.sha_cache[key]


class ProtocolAudit:
    def __init__(self, root: Path, files: FileChecks, max_details: int):
        self.root, self.files, self.max_details = root.resolve(), files, max_details
        self.errors, self.warnings = [], []
        self.error_counts, self.warning_counts = Counter(), Counter()
        self.counts = defaultdict(Counter)
        self.unique_paths, self.known_patients = set(), set()
        self.sample_ids = {}
        self.path_splits, self.sha_splits, self.group_splits, self.patient_splits = {}, {}, {}, {}
        self.uid_paths, self.path_uids, self.path_hashes, self.path_groups = {}, {}, {}, {}
        self.row_hashes = {}
        self.annotation_files = set()
        self.mask_png_files = set()
        self.input_dimensions_checked = set()
        self.annot_png_dimensions_checked = set()
        self.records = 0
        self.summary_comparison = {}

    def add(self, code, message, context=None, warning=False):
        counter = self.warning_counts if warning else self.error_counts
        details = self.warnings if warning else self.errors
        counter[code] += 1
        if len(details) < self.max_details:
            details.append({"code": code, "message": message, **(context or {})})

    def consistent_split(self, mapping, key, split, code, context):
        if key in mapping and mapping[key] != split:
            self.add(code, f"Identity appears in both {mapping[key]} and {split}", context)
        else:
            mapping[key] = split

    def path(self, text, context, kind):
        try:
            return self.files.resolve(text)
        except (OSError, ValueError) as error:
            self.add("invalid_" + kind + "_path", str(error), {**context, "path": text})
            return None

    def check_record(self, row, task, split, manifest, line_number):
        context = {"manifest": str(manifest.relative_to(self.root)), "line": line_number,
                   "sample_id": row.get("sample_id") if isinstance(row, dict) else None}
        if not isinstance(row, dict):
            self.add("record_not_object", "JSONL record must be an object", context)
            return
        self.records += 1
        self.counts[task][split] += 1
        missing = REQUIRED - set(row)
        if missing:
            self.add("missing_required_fields", "Missing fields: " + ", ".join(sorted(missing)), context)
        if row.get("task") != task or row.get("split") != split:
            self.add("task_or_split_mismatch", "Record task/split differs from its manifest location", context)
        for key in ["sample_id", "image_uid", "dataset", "leakage_group_id"]:
            if not isinstance(row.get(key), str) or not row[key]:
                self.add("invalid_identity_field", f"{key} must be a nonempty string", context)
        sample_id = row.get("sample_id")
        if isinstance(sample_id, str):
            if sample_id in self.sample_ids:
                self.add("duplicate_sample_id", "sample_id repeats across task manifests", {**context, "first_location": self.sample_ids[sample_id]})
            else:
                self.sample_ids[sample_id] = context["manifest"] + ":" + str(line_number)
                self.row_hashes[sample_id] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        patient = row.get("patient_id")
        if patient is not None:
            if not isinstance(patient, str) or not patient:
                self.add("invalid_patient_id", "Known patient_id must be a nonempty namespaced string", context)
            else:
                self.known_patients.add(patient)
                self.consistent_split(self.patient_splits, patient, split, "patient_split_leakage", context)
        group = row.get("leakage_group_id")
        if isinstance(group, str) and group:
            self.consistent_split(self.group_splits, group, split, "leakage_group_split_leakage", context)

        labels, masks, annotations = row.get("labels"), row.get("label_mask"), row.get("annotations")
        if not isinstance(labels, dict) or not isinstance(masks, dict) or not isinstance(annotations, dict):
            self.add("invalid_supervision_containers", "labels, label_mask and annotations must be dictionaries", context)
            labels = labels if isinstance(labels, dict) else {}
            masks = masks if isinstance(masks, dict) else {}
            annotations = annotations if isinstance(annotations, dict) else {}
        for label, value in labels.items():
            if label not in masks:
                self.add("missing_label_mask", f"No mask for label {label}", context)
            elif not bool_mask(masks[label]):
                self.add("invalid_label_mask", f"Mask for {label} must be boolean or integer 0/1", context)
            elif masks[label] and value is None:
                self.add("known_label_is_null", f"Mask is true but label {label} is null", context)
            if isinstance(value, float) and not math.isfinite(value):
                self.add("nonfinite_label", f"Nonfinite label {label}", context)
        for label, mask in masks.items():
            if not bool_mask(mask):
                self.add("invalid_label_mask", f"Mask for {label} must be boolean or integer 0/1", context)
            elif mask and label not in labels and not annotation_supports(label, annotations):
                self.add("known_structural_target_missing", f"No label or annotation supervision for known target {label}", context)

        paths = row.get("image_paths")
        if not isinstance(paths, list) or not paths or any(not isinstance(path, str) or not path for path in paths):
            self.add("invalid_image_paths", "image_paths must be a nonempty array of path strings", context)
            paths = []
        if len(set(paths)) != len(paths):
            self.add("duplicate_image_in_record", "image_paths contains the same path twice", context)
        if paths and row.get("image_path") != (paths[0] if len(paths) == 1 else None):
            self.add("single_pair_image_path_mismatch", "Single-image records need matching image_path; paired records need null", context)
        uid = row.get("image_uid")
        normalized_paths = tuple(path.replace("\\", "/") for path in paths)
        if isinstance(uid, str) and uid and normalized_paths:
            if uid in self.uid_paths and self.uid_paths[uid] != normalized_paths:
                self.add("image_uid_collision", "image_uid is associated with different input paths", context)
            self.uid_paths[uid] = normalized_paths
            if normalized_paths in self.path_uids and self.path_uids[normalized_paths] != uid:
                self.add("image_uid_inconsistent", "Same input path tuple has different image_uid values", context)
            self.path_uids[normalized_paths] = uid
        hashes = row.get("image_sha256")
        if not isinstance(hashes, list) or len(hashes) != len(paths):
            self.add("image_hash_count_mismatch", "image_sha256 must have one hash per input image", context)
            hashes = [None] * len(paths)
        sizes = row.get("image_sizes")
        if sizes is not None and (not isinstance(sizes, list) or len(sizes) != len(paths)):
            self.add("image_size_count_mismatch", "image_sizes must have one pair per input image", context)
            sizes = None
        actual_sizes = []
        for index, text in enumerate(paths):
            normalized = text.replace("\\", "/")
            self.unique_paths.add(normalized)
            image = self.path(text, context, "image")
            if image is None:
                actual_sizes.append(None)
                continue
            identity = str(image).casefold()
            self.consistent_split(self.path_splits, identity, split, "image_path_split_leakage", context)
            if identity in self.path_groups and self.path_groups[identity] != group:
                self.add("image_path_group_inconsistent", "Same input file has different leakage_group_id values", context)
            self.path_groups[identity] = group
            checksum = hashes[index]
            if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdefABCDEF" for c in checksum):
                self.add("invalid_image_sha256", "Input image SHA256 must contain 64 hex digits", context)
            else:
                checksum = checksum.lower()
                self.consistent_split(self.sha_splits, checksum, split, "image_sha256_split_leakage", context)
                if identity in self.path_hashes and self.path_hashes[identity] != checksum:
                    self.add("same_path_different_sha256", "Same file has inconsistent declared SHA256", context)
                self.path_hashes[identity] = checksum
                if self.files.rehash and self.files.sha256(image) != checksum:
                    self.add("image_sha256_content_mismatch", "File bytes do not match declared SHA256", {**context, "image_path": text})
            try:
                dimensions = self.files.dimensions(image)
                actual_sizes.append(dimensions)
                self.input_dimensions_checked.add(identity)
                if sizes is not None and list(dimensions) != sizes[index]:
                    self.add("source_image_size_mismatch", "image_sizes differs from the image header", {**context, "image_path": text, "header_size": dimensions, "declared_size": sizes[index]})
            except (OSError, ValueError) as error:
                actual_sizes.append(None)
                self.add("input_image_header_error", str(error), {**context, "image_path": text})

        refs = list(annotation_references(annotations))
        if row.get("annotation_path") is not None:
            refs.append(("annotation_path", row["annotation_path"]))
        for location, text in refs:
            annotation = self.path(text, {**context, "annotation_field": location}, "annotation")
            if annotation is None:
                continue
            identity = str(annotation).casefold()
            self.annotation_files.add(identity)
            if annotation.suffix.lower() == ".png":
                self.mask_png_files.add(identity)
                try:
                    dimensions = self.files.dimensions(annotation)
                    self.annot_png_dimensions_checked.add(identity)
                    if len(actual_sizes) == 1 and actual_sizes[0] and dimensions != actual_sizes[0]:
                        self.add("annotation_png_size_mismatch", "Annotation PNG dimensions differ from source image", {**context, "annotation_field": location, "path": text, "mask_size": dimensions, "source_size": actual_sizes[0]})
                    elif len(actual_sizes) > 1:
                        self.add("paired_annotation_size_not_assigned", "Paired input annotation has no unambiguous image index for dimension matching", context, warning=True)
                except (OSError, ValueError) as error:
                    self.add("annotation_png_header_error", str(error), {**context, "annotation_field": location, "path": text})
        points = annotations.get("points")
        if isinstance(points, dict) and len(actual_sizes) == 1 and actual_sizes[0]:
            width, height = actual_sizes[0]
            for name, specification in points.items():
                xy = specification.get("xy") if isinstance(specification, dict) else None
                if not (isinstance(xy, list) and len(xy) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in xy)
                        and 0 <= xy[0] <= width and 0 <= xy[1] <= height):
                    self.add("invalid_landmark_coordinates", f"Landmark {name} is invalid or outside image bounds", context)

    def read_task_manifests(self):
        task_root = self.root / "tasks"
        if not task_root.is_dir():
            self.add("missing_tasks_directory", "Protocol tasks/ directory does not exist")
            return
        tasks = sorted(path.name for path in task_root.iterdir() if path.is_dir())
        if not tasks:
            self.add("no_task_manifests", "No task directories found")
        for task in tasks:
            for split in SPLITS:
                path = task_root / task / (split + ".jsonl")
                self.counts[task][split] += 0
                if not path.is_file():
                    self.add("missing_task_manifest", "Expected task/split JSONL file does not exist", {"manifest": str(path.relative_to(self.root))})
                    continue
                try:
                    with path.open(encoding="utf-8-sig") as handle:
                        for line_number, line in enumerate(handle, 1):
                            if not line.strip():
                                continue
                            try:
                                row = json.loads(line)
                            except (ValueError, TypeError) as error:
                                self.add("invalid_jsonl", str(error), {"manifest": str(path.relative_to(self.root)), "line": line_number})
                                continue
                            self.check_record(row, task, split, path, line_number)
                except (OSError, UnicodeError) as error:
                    self.add("manifest_read_error", str(error), {"manifest": str(path.relative_to(self.root))})

    def compare_summary(self):
        path = self.root / "summary.json"
        try:
            summary = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            self.add("missing_or_invalid_summary", str(error))
            return
        observed = {"task_records": self.records, "unique_image_paths": len(self.unique_paths), "known_patients": len(self.known_patients),
                    "tasks": {task: {split: counts[split] for split in SPLITS} for task, counts in sorted(self.counts.items())}}
        for key, value in observed.items():
            expected = summary.get(key)
            matched = expected == value
            self.summary_comparison[key] = {"matches": matched, "declared": expected, "observed": value}
            if not matched:
                self.add("summary_count_mismatch", f"summary.json {key} differs from task manifests", {"field": key, "declared": expected, "observed": value})

    def check_all_tasks_index(self):
        index_root = self.root / "all_tasks"
        if not index_root.is_dir():
            self.add("all_tasks_index_absent", "Optional redundant all_tasks/ index is absent", warning=True)
            return
        seen = set()
        for split in SPLITS:
            path = index_root / (split + ".jsonl")
            if not path.is_file():
                self.add("missing_all_tasks_index", "all_tasks split file is missing", {"manifest": str(path.relative_to(self.root))})
                continue
            try:
                with path.open(encoding="utf-8-sig") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                            sample_id = row.get("sample_id")
                            checksum = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
                            if sample_id in seen:
                                self.add("duplicate_all_tasks_index_record", "sample_id repeats in redundant index", {"sample_id": sample_id})
                            seen.add(sample_id)
                            if row.get("split") != split or self.row_hashes.get(sample_id) != checksum:
                                self.add("all_tasks_index_mismatch", "Redundant index record differs from its task manifest", {"sample_id": sample_id, "manifest": str(path.relative_to(self.root)), "line": line_number})
                        except (ValueError, TypeError, AttributeError) as error:
                            self.add("invalid_all_tasks_index", str(error), {"manifest": str(path.relative_to(self.root)), "line": line_number})
            except (OSError, UnicodeError) as error:
                self.add("all_tasks_index_read_error", str(error), {"manifest": str(path.relative_to(self.root))})
        missing = set(self.row_hashes) - seen
        if missing:
            self.add("all_tasks_index_missing_records", f"Redundant index is missing {len(missing)} task records", {"sample_ids_sample": sorted(missing)[:10]})

    def run(self):
        started = time.monotonic()
        self.read_task_manifests()
        self.compare_summary()
        self.check_all_tasks_index()
        report = {
            "protocol_root": str(self.root), "validated_utc": datetime.now(timezone.utc).isoformat(),
            "all_checks_passed": not self.error_counts, "errors_total": sum(self.error_counts.values()),
            "error_counts": dict(self.error_counts), "errors": self.errors,
            "warnings_total": sum(self.warning_counts.values()), "warning_counts": dict(self.warning_counts), "warnings": self.warnings,
            "record_count": self.records, "unique_sample_ids": len(self.sample_ids),
            "unique_image_paths": len(self.unique_paths), "known_patients": len(self.known_patients),
            "unique_image_sha256": len(self.sha_splits), "leakage_groups": len(self.group_splits),
            "tasks": {task: dict(counts) for task, counts in sorted(self.counts.items())},
            "referenced_annotation_files": len(self.annotation_files),
            "source_image_headers_checked": len(self.input_dimensions_checked),
            "annotation_png_files": len(self.mask_png_files), "annotation_png_headers_checked": len(self.annot_png_dimensions_checked),
            "summary_comparison": self.summary_comparison,
            "file_sha256_recomputed": self.files.rehash,
            "validation_scope": ["Records counted only from tasks/<task>/<split>.jsonl; all_tasks is cross-checked without double counting.",
                                 "Patient, image-path, SHA256 and leakage-group split constraints apply across every task inside each protocol, not between alternative protocols.",
                                 "Known scalar labels cannot be null; mask=false may retain a source value. Structural targets may be supervised by annotations instead of labels.",
                                 "Annotation paths are found recursively, including both PAPILA experts. Raw TIFF annotations are checked for existence only, never decoded.",
                                 "All input images and annotation PNGs use read-only header dimension checks. SHA256 is checked for manifest consistency; file bytes are rehashed only with --rehash-images."],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        output = self.root / "validation_report.json"
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        print(json.dumps({key: report[key] for key in ["protocol_root", "all_checks_passed", "record_count", "unique_image_paths", "known_patients", "errors_total", "warnings_total", "elapsed_seconds"]}, ensure_ascii=True), flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=BASE / "data")
    parser.add_argument("--prepared-root", type=Path, default=None)
    parser.add_argument("--protocol", choices=["both", "strict_cross_domain", "within_source"], default="both")
    parser.add_argument("--rehash-images", action="store_true", help="Also recompute every unique source-image SHA256; default checks declared hashes for consistency.")
    parser.add_argument("--max-error-details", type=int, default=200)
    args = parser.parse_args()
    if args.max_error_details < 1:
        parser.error("--max-error-details must be positive")
    data_root = args.data_root.resolve()
    prepared = args.prepared_root.resolve() if args.prepared_root else data_root / "prepared"
    files = FileChecks(data_root, args.rehash_images)
    protocols = ["strict_cross_domain", "within_source"] if args.protocol == "both" else [args.protocol]
    passed = True
    for protocol in protocols:
        root = prepared if protocol == "strict_cross_domain" else prepared / "protocols/within_source"
        report = ProtocolAudit(root, files, args.max_error_details).run()
        passed = passed and report["all_checks_passed"]
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
