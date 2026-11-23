"""Cache source-image fingerprints without modifying any original data."""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
AUDIT = DATA / "prepared" / "audit"
COS = np.cos(np.pi / 32 * (np.arange(32)[None, :] + 0.5) * np.arange(8)[:, None]).astype(np.float32)


def source_images():
    patterns = {
        "RFMiD": ["Training_Set/Training/*.png", "Evaluation_Set/Validation/*.png", "Test_Set/Test/*.png"],
        "RFMiD2": ["extracted/**/*.jpg"],
        "APTOS2019": ["public_mirror/*_images/*_images/*.png"],
        "DDR": ["DDR-dataset/DR_grading/*/*.jpg", "DDR-dataset/lesion_segmentation/*/image/*.jpg"],
        "EyeQ": ["images_480px_public_mirror/*/*.jpeg"],
        "IDRiD": ["extracted/A. Segmentation/1. Original Images/*/*.jpg", "extracted/B. Disease Grading/1. Original Images/*/*.jpg"],
        "PAPILA": ["extracted/*/FundusImages/*.jpg"],
        "FIVES": ["extracted/*/*/Original/*.png"],
        "REFUGE": ["extracted/REFUGE2/*/images/*.jpg"],
        "ODIR-5K": ["*/Images/*.jpg"],
    }
    paths = set()
    for dataset, globs in patterns.items():
        for pattern in globs:
            paths.update((DATA / dataset).glob(pattern))
    return sorted(paths)


def pack_bits(bits):
    number = 0
    for bit in bits:
        number = (number << 1) | int(bit)
    return f"{number:016x}"


def fingerprint(path):
    rel = path.relative_to(DATA).as_posix()
    st = path.stat()
    checksum = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                checksum.update(block)
        with Image.open(path) as original:
            width, height = original.size
            original.draft("RGB", (128, 128))
            rgb = ImageOps.exif_transpose(original).convert("RGB")
            thumbnail = rgb.resize((32, 32), Image.Resampling.LANCZOS)
            gray = np.asarray(thumbnail.convert("L"), dtype=np.float32)
            coefficients = COS @ gray @ COS.T
            flattened = coefficients.flatten()
            phash = pack_bits(flattened > np.median(flattened[1:]))
            tiny = np.asarray(rgb.resize((9, 8), Image.Resampling.LANCZOS).convert("L"))
            dhash = pack_bits((tiny[:, 1:] > tiny[:, :-1]).flatten())
            return (rel, st.st_size, st.st_mtime_ns, checksum.hexdigest(), width, height, phash, dhash,
                    float(gray.std()), thumbnail.tobytes(), None)
    except Exception as error:
        return (rel, st.st_size, st.st_mtime_ns, checksum.hexdigest(), None, None, None, None, None, None,
                f"{type(error).__name__}: {error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    AUDIT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(AUDIT / "image_fingerprints.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS images(path TEXT PRIMARY KEY, bytes INTEGER, mtime_ns INTEGER, sha256 TEXT, width INTEGER, height INTEGER, phash TEXT, dhash TEXT, gray_std REAL, thumb BLOB, error TEXT)")
    cached = {p: (size, mtime) for p, size, mtime in db.execute("SELECT path,bytes,mtime_ns FROM images WHERE error IS NULL")}
    paths = source_images()
    pending = []
    counts = {}
    for path in paths:
        rel = path.relative_to(DATA).as_posix()
        counts[rel.split("/")[0]] = counts.get(rel.split("/")[0], 0) + 1
        st = path.stat()
        if cached.get(rel) != (st.st_size, st.st_mtime_ns):
            pending.append(path)
    print(json.dumps({"event": "fingerprint_inventory", "images": len(paths), "cached": len(paths) - len(pending), "datasets": counts}), flush=True)
    if len(paths) < 60000:
        raise RuntimeError("Source inventory unexpectedly incomplete")
    started = time.monotonic()
    last_report = started
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fingerprint, path) for path in pending]
        for future in cf.as_completed(futures):
            row = future.result()
            db.execute("INSERT OR REPLACE INTO images VALUES(?,?,?,?,?,?,?,?,?,?,?)", row)
            done += 1
            if done % 100 == 0:
                db.commit()
            now = time.monotonic()
            if now - last_report >= 15 or done == len(pending):
                print(json.dumps({"event": "fingerprint_progress", "new_completed": done, "new_total": len(pending),
                                  "elapsed_seconds": round(now - started, 1)}), flush=True)
                last_report = now
    db.commit()
    errors = [{"image_path": p, "error": error} for p, error in db.execute("SELECT path,error FROM images WHERE error IS NOT NULL")]
    (AUDIT / "fingerprint_inventory.json").write_text(json.dumps({"image_files": len(paths), "dataset_counts": counts, "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    print(json.dumps({"event": "fingerprints_complete", "images": len(paths), "errors": len(errors)}), flush=True)


if __name__ == "__main__":
    main()
