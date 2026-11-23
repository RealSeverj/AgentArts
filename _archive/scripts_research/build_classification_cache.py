"""Build a separately stored, verified resized-RGB cache for train/val only.

Original images and manifests remain read-only. Each PNG and the final index
are written atomically. Interrupted builds reuse only stat- and hash-verified
files, using cache_progress.json as a resumable partial index.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from PIL import Image, __version__ as pillow_version
from scripts import dataset_routing as routing
from training.fundus_preprocessing import apply_preprocessing, preprocessing_spec
from training.image_cache import (FORMAT_VERSION, REPRESENTATION, RESIZE, canonical_relative_path,
                                   normalize_image_size, pixel_sha256, read_verified_png,
                                   resolve_relative, validate_manifest, validate_roots)


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def cache_path_for(source_relative):
    digest = hashlib.sha256(source_relative.encode("utf-8")).hexdigest()
    return f"images/{digest[:2]}/{digest}.png"


def process_image(payload):
    # Preserve the original six-item payload for callers/tests using stretch.
    if len(payload) == 6:
        relative, data_root, cache_root, size, prior, compress_level = payload
        spec = {"mode": "stretch"}
    else:
        relative, data_root, cache_root, size, prior, compress_level, spec = payload
    source = resolve_relative(data_root, relative)
    if not source.is_file():
        raise FileNotFoundError(f"Original image is missing: {relative}")
    stat = source.stat()
    cache_relative = cache_path_for(relative)
    output = resolve_relative(cache_root, cache_relative)
    if (prior and prior.get("cache_path") == cache_relative and
            prior.get("source_size") == stat.st_size and prior.get("source_mtime_ns") == stat.st_mtime_ns):
        try:
            read_verified_png(output, size, prior["pixel_sha256"])
            return relative, prior, "reused"
        except (OSError, ValueError, KeyError):
            pass  # This build explicitly repairs a stale/missing/corrupt cache.
    with Image.open(source) as original:
        source_dimensions = list(original.size)
        direct, preprocessing_metadata = apply_preprocessing(original, size, spec, return_metadata=True)
    expected = pixel_sha256(direct)
    result = {"cache_path": cache_relative, "source_size": stat.st_size,
              "source_mtime_ns": stat.st_mtime_ns, "pixel_sha256": expected,
              "source_image_size": source_dimensions}
    if spec["mode"] != "stretch":
        result["preprocessing_metadata"] = preprocessing_metadata
    status = "built"
    try:
        read_verified_png(output, size, expected)
        status = "reused_verified_against_source"
    except (OSError, ValueError):
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            direct.save(temporary, format="PNG", compress_level=compress_level)
            read_verified_png(temporary, size, expected)
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
    # The source might have changed while it was being decoded or written.
    after = source.stat()
    if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError(f"Original image changed during cache construction: {relative}")
    return relative, result, status


def selected_paths(config, data_root):
    if routing.is_source_isolated(config):
        source_config = deepcopy(config)
        # Validate the source before constructing the not-yet-existing cache.
        source_config.get("data", {}).pop("image_cache_dir", None)
        routing.validate_training_route(source_config, data_root)
    allowed = config.get("data", {}).get("datasets")
    selected, manifest_hashes, split_counts = set(), {}, {}
    task = canonical_relative_path(config["task"])
    if "/" in task:
        raise ValueError("Task must be one directory name")
    labels = config.get("labels", [])
    if not labels:
        raise ValueError("Classification cache requires output labels")
    for split in ("train", "val"):
        manifest = routing.manifest_path(config, data_root, split)
        manifest_hashes[split] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        count = 0
        with manifest.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if routing.is_source_isolated(config):
                    routing.validate_row(config, row, split)
                elif allowed and row["dataset"] not in allowed:
                    continue
                if row.get("task") != task or row.get("split") != split:
                    raise ValueError(f"Task/split mismatch in {manifest}")
                if not any(row.get("label_mask", {}).get(label, False) and row.get("labels", {}).get(label) is not None for label in labels):
                    continue
                for path in row["image_paths"]:
                    selected.add(canonical_relative_path(path))
                count += 1
        split_counts[split] = count
    return sorted(selected), manifest_hashes, split_counts


def build_cache(config, data_root, cache_root, workers=8, image_size=None, compress_level=1, progress_seconds=30):
    if config.get("kind") not in ("multiclass", "multilabel"):
        raise ValueError("This cache builder supports classification image inputs")
    if workers < 1:
        raise ValueError("workers must be positive")
    cache_root, data_root = validate_roots(cache_root, data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Original data root is missing: {data_root}")
    requested = image_size or config.get("model", {}).get("img_size", 392)
    # Model config follows [height,width]; the explicit cache API/PIL uses [width,height].
    if image_size is None and isinstance(requested, (list, tuple)) and len(requested) == 2:
        requested = (requested[1], requested[0])
    size = normalize_image_size(requested)
    spec = preprocessing_spec(config)
    paths, manifest_hashes, split_counts = selected_paths(config, data_root)
    if not paths:
        raise ValueError("Selected train/val manifests contain no supervised image inputs")
    if (cache_root.exists() and any(cache_root.iterdir())
            and not any((cache_root / name).is_file() for name in ("cache_manifest.json", "cache_progress.json"))):
        raise ValueError("Nonempty cache root has no recognized metadata; use a separate cache root")
    cache_root.mkdir(parents=True, exist_ok=True)
    prior = {}
    for name in ("cache_manifest.json", "cache_progress.json"):
        candidate = cache_root / name
        if candidate.is_file():
            old = json.loads(candidate.read_text(encoding="utf-8-sig"))
            validate_manifest(old, size, require_complete=False, expected_preprocessing=spec)
            expected_binding = {"task": config["task"],
                                "protocol": config.get("data", {}).get("protocol", "strict_cross_domain"),
                                "datasets": config.get("data", {}).get("datasets"),
                                "manifest_sha256": manifest_hashes}
            if any(old.get(key) != value for key, value in expected_binding.items()):
                raise ValueError("Image cache belongs to another source/task manifest; use a separate cache root")
            prior.update(old["entries"])
    metadata = {"format_version": FORMAT_VERSION, "representation": REPRESENTATION,
                "image_size": list(size), "resize": RESIZE, "preprocessing": spec,
                "pixel_equivalence_verified": False, "pillow_version": pillow_version,
                "source_size_unit": "encoded_bytes", "png_compress_level": compress_level,
                "task": config["task"], "protocol": config.get("data", {}).get("protocol", "strict_cross_domain"),
                "source_isolated": routing.is_source_isolated(config),
                "datasets": config.get("data", {}).get("datasets"),
                "source_data_root_at_build": str(data_root), "manifest_sha256": manifest_hashes,
                "split_supervised_records": split_counts, "expected_unique_images": len(paths), "entries": {}}
    completed = metadata["entries"]
    counts, start, last_progress = {}, time.monotonic(), time.monotonic()
    atomic_json(cache_root / "cache_progress.json", metadata)
    print(json.dumps({"event": "cache_build_started", "images": len(paths), "workers": workers,
                      "image_size": list(size), "cache_root": str(cache_root)}, ensure_ascii=False), flush=True)

    def record(result):
        nonlocal last_progress
        relative, entry, status = result
        completed[relative] = entry
        counts[status] = counts.get(status, 0) + 1
        now = time.monotonic()
        if now - last_progress >= progress_seconds or len(completed) == len(paths):
            atomic_json(cache_root / "cache_progress.json", metadata)
            print(json.dumps({"event": "cache_build_progress", "completed": len(completed), "total": len(paths),
                              "counts": counts, "elapsed_seconds": round(now - start, 2)}, ensure_ascii=False), flush=True)
            last_progress = now

    payloads = ((path, str(data_root), str(cache_root), size, prior.get(path), compress_level, spec) for path in paths)
    if workers == 1:
        try:
            for payload in payloads:
                record(process_image(payload))
        except BaseException:
            atomic_json(cache_root / "cache_progress.json", metadata)
            raise
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = [pool.submit(process_image, payload) for payload in payloads]
        try:
            for future in as_completed(futures):
                record(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            atomic_json(cache_root / "cache_progress.json", metadata)
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
    for relative, entry in completed.items():
        stat = resolve_relative(data_root, relative).stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry["source_size"], entry["source_mtime_ns"]):
            raise ValueError(f"Original source changed before final cache verification: {relative}")
    if len(completed) != len(paths):
        raise RuntimeError("Incomplete cache build cannot be marked pixel-equivalent")
    metadata.update(pixel_equivalence_verified=True, completed_utc=datetime.now(timezone.utc).isoformat(),
                    build_counts=counts, elapsed_seconds=time.monotonic() - start)
    atomic_json(cache_root / "cache_manifest.json", metadata)
    atomic_json(cache_root / "cache_progress.json", metadata)
    print(json.dumps({"event": "cache_build_complete", "images": len(completed), "counts": counts,
                      "elapsed_seconds": round(time.monotonic() - start, 2),
                      "manifest": str(cache_root / "cache_manifest.json")}, ensure_ascii=False), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/source_isolated/strict_cross_domain/dr_grading_ddr.json")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--png-compress-level", type=int, choices=range(10), default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    build_cache(config, args.data_root, args.cache_root, workers=args.workers,
                image_size=args.image_size, compress_level=args.png_compress_level)


if __name__ == "__main__":
    main()
