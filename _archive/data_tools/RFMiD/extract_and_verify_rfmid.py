"""Extract RFMiD with ZIP CRC verification and validate image/label matches."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile
import zlib

ROOT = Path(__file__).resolve().parent
SPECS = [
    {"archive": "Training_Set.zip", "split": "training", "folder": "Training_Set/Training", "csv": "Training_Set/RFMiD_Training_Labels.csv", "count": 1920,
     "csv_sha256": "03f3d0afcbe6829eef111ceb73ee6fe1e1c1b18531f6a789502279bfda371878"},
    {"archive": "Evaluation_Set.zip", "split": "validation", "folder": "Evaluation_Set/Validation", "csv": "Evaluation_Set/RFMiD_Validation_Labels.csv", "count": 640,
     "csv_sha256": "465ed8cbea77f269d17bf4055a90b2bd5b75edacb82e0519c876f28d98b24fff"},
    {"archive": "Test_Set.zip", "split": "test", "folder": "Test_Set/Test", "csv": "Test_Set/RFMiD_Testing_Labels.csv", "count": 640,
     "csv_sha256": "4ebc844ddd280777cb18112f60b4890692efcb6e5121672b888d2d5a3fdf5e05"},
]


def checked_target(name: str) -> Path:
    member = PurePosixPath(name.replace("\\", "/"))
    if member.is_absolute() or ".." in member.parts or any(":" in part for part in member.parts):
        raise ValueError(f"Unsafe ZIP member: {name}")
    target = ROOT.joinpath(*member.parts).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError(f"ZIP member escapes dataset directory: {name}")
    return target


def file_crc32(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xffffffff


def main() -> None:
    report = {"dataset": "RFMiD original release", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "zip_crc_checked_during_extraction": True, "splits": []}
    for spec in SPECS:
        archive = ROOT / "archives" / spec["archive"]
        print(json.dumps({"event": "extracting", "archive": spec["archive"]}), flush=True)
        with zipfile.ZipFile(archive) as package:
            members = package.infolist()
            images_in_zip = [m for m in members if not m.is_dir() and m.filename.lower().endswith(".png")]
            if len(images_in_zip) != spec["count"]:
                raise ValueError(f"Unexpected image count in {spec['archive']}")
            for number, member in enumerate(members, 1):
                target = checked_target(member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("Symbolic links are not permitted in data archives")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.stat().st_size == member.file_size and file_crc32(target) == member.CRC:
                        continue
                    raise ValueError(f"Existing extracted file differs from source: {member.filename}")
                temporary = target.with_name(target.name + ".extracting")
                copied = 0
                # Reading each member to EOF also checks its stored ZIP CRC.
                with package.open(member) as source, temporary.open("wb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        copied += len(block)
                if copied != member.file_size:
                    raise ValueError("Extracted file size differs from ZIP metadata")
                os.replace(temporary, target)
                if number % 250 == 0:
                    print(json.dumps({"event": "extraction_progress", "split": spec["split"],
                                      "members_done": number, "members_total": len(members)}), flush=True)
        label_path = ROOT / spec["csv"]
        label_hash = hashlib.sha256(label_path.read_bytes()).hexdigest()
        if label_hash != spec["csv_sha256"]:
            raise ValueError("Label hash differs from independently fetched original CSV")
        with label_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        if len(rows) != spec["count"] or not fields or len(fields) != 47:
            raise ValueError("Unexpected label rows or columns")
        if fields[:2] != ["ID", "Disease_Risk"]:
            raise ValueError("Unexpected original label schema")
        ids = [str(int(row["ID"])) for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate IDs within a split")
        for row in rows:
            if any(row[field] not in {"0", "1"} for field in fields[1:]):
                raise ValueError("Unexpected nonbinary disease label")
        folder = ROOT / spec["folder"]
        images = list(folder.glob("*.png"))
        image_ids = {image.stem for image in images}
        if len(images) != spec["count"] or image_ids != set(ids):
            raise ValueError("Images and label IDs do not match")
        split_report = {"split": spec["split"], "image_directory": spec["folder"], "image_count": len(images),
                        "label_csv": spec["csv"], "label_rows": len(rows), "label_columns": len(fields),
                        "label_sha256": label_hash, "image_label_ids_match": True, "zip_members_crc_valid": True}
        report["splits"].append(split_report)
        print(json.dumps({"event": "split_verified", **split_report}), flush=True)
    report["total_images"] = sum(item["image_count"] for item in report["splits"])
    report["all_checks_passed"] = True
    path = ROOT / "verification_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "verified", "total_images": report["total_images"], "all_checks_passed": True}), flush=True)


if __name__ == "__main__":
    main()
