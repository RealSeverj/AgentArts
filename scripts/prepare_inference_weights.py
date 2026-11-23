"""Stage the four existing backups under portable names; verify before/after copy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "quality": "20261114_remaining_experts/quality_best.pt",
    "multidisease": "20261114_remaining_experts/multidisease_best.pt",
    "dr_grading": "20261114_DDR_crop_letterbox/best.pt",
    "disc_cup_segmentation": "20261114_remaining_experts/disc_refuge_best.pt",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, default=ROOT / "cloud_backup")
    parser.add_argument("--output-root", type=Path, default=ROOT / "weights")
    args = parser.parse_args()
    models = json.loads((ROOT / "configs/inference/models.json").read_text(encoding="utf-8"))["models"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    for stage, relative in SOURCES.items():
        source, item = args.backup_root / relative, models[stage]
        destination = args.output_root / item["checkpoint"]
        if sha256(source) != item["sha256"]:
            raise RuntimeError(f"Source hash mismatch: {stage}")
        if destination.exists():
            if not destination.is_file() or sha256(destination) != item["sha256"]:
                raise RuntimeError(f"Refusing to overwrite a differing destination: {stage}")
        else:
            # Exclusive create prevents replacing an artifact introduced concurrently.
            with source.open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024**2)
        if sha256(destination) != item["sha256"]:
            raise RuntimeError(f"Copied weight hash mismatch: {stage}")
        print(f"{stage}: verified ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
