"""Safely extract the complete DDR public mirror and verify task annotations."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import collections
import hashlib
import json
import os
import stat
import zipfile
import zlib
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archives" / "DDR_full_kaggle.zip"


def crc32(path):
    checksum = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xffffffff


def checked_target(name):
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or any(":" in p for p in path.parts):
        raise ValueError(f"Unsafe member: {name}")
    target = ROOT.joinpath(*path.parts).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("Archive member escapes dataset directory")
    return target


def main():
    expected_index = json.loads((ROOT / "mirror_archive_index.json").read_text(encoding="utf-8"))
    expected = {i["name"]: (i["bytes"], i["crc32"]) for i in expected_index}
    report = {"dataset": "DDR", "source_type": "public Kaggle mirror of original data directory",
              "mirror": "https://www.kaggle.com/datasets/nishantbharti007/ddr-dataset-full",
              "original_repository": "https://github.com/nkicsl/DDR-dataset",
              "license": "CC BY-NC-SA 4.0 per original author",
              "verified_utc": datetime.now(timezone.utc).isoformat()}
    extracted = 0
    with zipfile.ZipFile(ARCHIVE) as package:
        members = package.infolist()
        actual = {i.filename: (i.file_size, f"{i.CRC:08x}") for i in members}
        if actual != expected:
            raise ValueError("Local archive differs from separately read remote index")
        for number, item in enumerate(members, 1):
            target = checked_target(item.filename)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Symlink in data archive")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != item.file_size or crc32(target) != item.CRC:
                    raise ValueError(f"Existing file differs from archive: {item.filename}")
            else:
                temporary = target.with_name(target.name + ".extracting")
                copied = 0
                with package.open(item) as source, temporary.open("wb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        copied += len(block)
                if copied != item.file_size:
                    raise ValueError("Extracted member length mismatch")
                os.replace(temporary, target)
            extracted += 1
            if number % 1000 == 0:
                print(json.dumps({"event": "extraction_progress", "files": number, "total": len(members)}), flush=True)
    dataset = ROOT / "DDR-dataset"
    grading = []
    all_grading_ids = set()
    for split, expected_count in [("train", 6835), ("valid", 2733), ("test", 4105)]:
        folder = dataset / "DR_grading" / split
        images = {p.name for p in folder.glob("*.jpg")}
        lines = (dataset / "DR_grading" / (split + ".txt")).read_text(encoding="utf-8-sig").splitlines()
        rows = [line.rsplit(None, 1) for line in lines if line.strip()]
        labels = {name: int(value) for name, value in rows}
        if len(images) != expected_count or len(labels) != expected_count or set(labels) != images:
            raise ValueError(f"Image/label mismatch in grading {split}")
        if not set(labels.values()).issubset(set(range(6))):
            raise ValueError("Unexpected grading class")
        if all_grading_ids.intersection(images):
            raise ValueError("Overlapping grading image filenames across splits")
        all_grading_ids.update(images)
        grading.append({"split": split, "images": len(images), "label_rows": len(rows),
                        "grade_counts": dict(sorted(collections.Counter(labels.values()).items())), "ids_match": True})
    segmentation = []
    detection = []
    for split, expected_count in [("train", 383), ("valid", 149), ("test", 225)]:
        folder = dataset / "lesion_segmentation" / split
        images = {p.stem for p in (folder / "image").glob("*.jpg")}
        if len(images) != expected_count:
            raise ValueError("Unexpected segmentation image count")
        mask_folder = folder / ("segmentation label" if split == "valid" else "label")
        counts = {}
        for label in ["EX", "HE", "MA", "SE"]:
            masks = {p.stem for p in (mask_folder / label).glob("*.tif")}
            if masks != images:
                raise ValueError(f"Segmentation masks do not match: {split}/{label}")
            counts[label] = len(masks)
        segmentation.append({"split": split, "images": len(images), "masks": counts, "ids_match": True})
        xmls = list((dataset / "lesion_detection" / split).glob("*.xml"))
        if {p.stem for p in xmls} != images:
            raise ValueError("Detection annotations do not match lesion images")
        object_names = collections.Counter()
        for xml_path in xmls:
            annotation = ET.parse(xml_path).getroot()
            object_names.update(obj.findtext("name", "") for obj in annotation.findall("object"))
        detection.append({"split": split, "xml_files": len(xmls), "xml_parse_valid": True,
                          "object_class_counts": dict(object_names), "ids_match": True})
    report.update({"archive_members": extracted, "zip_crc_checked_during_extraction": True,
                   "uncompressed_bytes": sum(i["bytes"] for i in expected_index),
                   "grading": grading, "segmentation": segmentation, "detection": detection,
                   "grading_images": sum(x["images"] for x in grading),
                   "segmentation_images": sum(x["images"] for x in segmentation),
                   "mask_files": sum(sum(x["masks"].values()) for x in segmentation),
                   "detection_xml_files": sum(x["xml_files"] for x in detection), "all_checks_passed": True})
    (ROOT / "verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "verified", "grading_images": report["grading_images"],
                      "segmentation_images": report["segmentation_images"], "all_checks_passed": True}), flush=True)


if __name__ == "__main__":
    main()
