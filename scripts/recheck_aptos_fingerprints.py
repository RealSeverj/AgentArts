"""Read-only recomputation of file/pixel/perceptual fingerprints for APTOS splits.

Writes audit artifacts only. Does not edit images, fingerprint DB, manifests,
checkpoints, labels or split assignments; it does not prove absence of all
possible transformed/clinical-patient duplicates.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts.image_fingerprints import COS, pack_bits


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def recompute(item):
    path, relative, split = item
    stat = path.stat()
    result = {"image_path": relative, "split": split, "bytes": stat.st_size,
              "mtime_ns": stat.st_mtime_ns, "file_sha256": file_sha(path)}
    with Image.open(path) as source:
        result["original_size"] = list(source.size)
        result["exif_orientation"] = source.getexif().get(274)
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        result["decoded_rgb_size"] = list(rgb.size)
        prefix = (f"RGB|{rgb.width}|{rgb.height}|").encode("ascii")
        pixel_digest = hashlib.sha256(prefix)
        pixel_digest.update(rgb.tobytes())
        result["decoded_rgb_sha256"] = pixel_digest.hexdigest()
        thumbnail = rgb.resize((32, 32), Image.Resampling.LANCZOS)
        gray = np.asarray(thumbnail.convert("L"), dtype=np.float32)
        coeff = (COS @ gray @ COS.T).flatten()
        result["phash"] = pack_bits(coeff > np.median(coeff[1:]))
        tiny = np.asarray(rgb.resize((9, 8), Image.Resampling.LANCZOS).convert("L"))
        result["dhash"] = pack_bits((tiny[:, 1:] > tiny[:, :-1]).flatten())
        result["gray_std"] = float(gray.std())
        result["rgb32_sha256"] = hashlib.sha256(thumbnail.tobytes()).hexdigest()
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Source changed while fingerprinting")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    data = args.data_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    leaf = data / "prepared/source_isolated/strict_cross_domain/tasks/dr_grading/APTOS2019"
    db_path = data / "prepared/audit/image_fingerprints.sqlite"
    wal = db_path.with_name(db_path.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError("Fingerprint DB has a nonempty WAL; obtain a consistent read-only snapshot first")
    db = sqlite3.connect("file:" + db_path.as_posix() + "?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    original = {r["path"]: dict(r) for r in db.execute("SELECT * FROM images WHERE path LIKE 'APTOS2019/%'")}
    global_signatures = list(db.execute("SELECT path,sha256,phash FROM images ORDER BY path"))
    db.close()
    cache_path = data / "prepared/audit/image_identity_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    signature = hashlib.sha256(("identity-policy-v1|" + "|".join(str(r[0]) + str(r[1]) + str(r[2]) for r in global_signatures)).encode()).hexdigest()
    rows = {}
    for split, count in (("train", 2775), ("val", 359), ("test", 354)):
        selected = [json.loads(x) for x in (leaf / f"{split}.jsonl").read_text(encoding="utf-8-sig").splitlines() if x.strip()]
        if len(selected) != count:
            raise ValueError("Actual split count differs from the frozen APTOS protocol")
        for row in selected:
            if row["dataset"] != "APTOS2019" or row["task"] != "dr_grading" or row["split"] != split or len(row["image_paths"]) != 1:
                raise ValueError("Task/source/split mismatch")
            relative = row["image_paths"][0]
            if relative in rows or relative not in original:
                raise ValueError("Duplicate path or source fingerprint missing")
            rows[relative] = (split, row)
    before_inputs = {str(p): file_sha(p) for p in (db_path, cache_path, *(leaf / f"{s}.jsonl" for s in ("train", "val", "test")))}
    fresh, errors, mismatches = [], [], []
    started = last = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(recompute, (data / p, p, s)): p for p, (s, _) in rows.items()}
        for future in as_completed(pending):
            path = pending[future]
            try:
                result = future.result()
                stored = original[path]
                expected = {"bytes": stored["bytes"], "mtime_ns": stored["mtime_ns"],
                            "file_sha256": stored["sha256"], "phash": stored["phash"],
                            "dhash": stored["dhash"], "rgb32_sha256": hashlib.sha256(stored["thumb"]).hexdigest()}
                for key, value in expected.items():
                    if result[key] != value:
                        mismatches.append({"image_path": path, "field": key, "stored": value, "fresh": result[key]})
                if result["file_sha256"] != rows[path][1]["image_sha256"][0]:
                    mismatches.append({"image_path": path, "field": "manifest_image_sha256"})
                fresh.append(result)
            except Exception as exc:
                errors.append({"image_path": path, "error_type": type(exc).__name__, "error": str(exc)})
            now = time.monotonic()
            if now - last >= 15 or len(fresh) + len(errors) == len(rows):
                print(json.dumps({"event": "fresh_aptos_fingerprints", "completed": len(fresh), "errors": len(errors),
                                  "total": len(rows), "elapsed_seconds": round(now - started, 1)}), flush=True)
                last = now
    fresh.sort(key=lambda r: r["image_path"])
    with (output / "fresh_fingerprints.jsonl").open("w", encoding="utf-8") as handle:
        for row in fresh:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    duplicate_groups = {}
    for field in ("file_sha256", "decoded_rgb_sha256"):
        groups = {}
        for row in fresh:
            groups.setdefault(row[field], []).append({"image_path": row["image_path"], "split": row["split"]})
        duplicate_groups[field] = [dict(digest=digest, members=members) for digest, members in groups.items()
                                   if len({x["split"] for x in members}) > 1]
    unchanged = all(file_sha(Path(p)) == digest for p, digest in before_inputs.items())
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "image_integrity_checks_passed": not errors and not mismatches and len(fresh) == len(rows) and unchanged,
              "phase": "fresh_fingerprints_complete; near-candidate visual review is separate",
              "images_expected": len(rows), "images_recomputed": len(fresh), "errors": errors,
              "cached_and_manifest_mismatches": mismatches, "source_audit_inputs_unchanged": unchanged,
              "original_global_identity_cache_signature_matches_db": signature == cache.get("signature"),
              "global_fingerprint_rows": len(global_signatures), "cached_source_rows": len(original),
              "cross_split_duplicate_groups": duplicate_groups,
              "pixel_hash_definition": "SHA256 of ASCII RGB|width|height| followed by full-resolution EXIF-transposed 8-bit RGB decoded pixels; no resize/crop/ICC conversion",
              "fresh_software": {"numpy": np.__version__, "pillow": pillow_version},
              "fingerprint_output": str(output / "fresh_fingerprints.jsonl"),
              "fingerprint_output_sha256": file_sha(output / "fresh_fingerprints.jsonl"),
              "input_files": [{"path": p, "sha256": digest} for p, digest in before_inputs.items()],
              "elapsed_seconds": round(time.monotonic() - started, 2),
              "limitations": ["Perceptual hash screening is not invariant to arbitrary crops, rotations or photometric changes.",
                              "Patient IDs are missing; this does not establish independent clinical patients.",
                              "This checks downstream fine-tuning splits, not undisclosed public pretraining image membership."]}
    (output / "fingerprint_recheck.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "fresh_recheck_complete", "passed": report["image_integrity_checks_passed"],
                      "file_SHA_cross_split_groups": len(duplicate_groups["file_sha256"]),
                      "decoded_pixel_cross_split_groups": len(duplicate_groups["decoded_rgb_sha256"]),
                      "report": str(output / "fingerprint_recheck.json")}), flush=True)


if __name__ == "__main__":
    main()
