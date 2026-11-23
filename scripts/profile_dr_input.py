"""Profile actual DR image decoding and alternative CPU preprocessing orders.

This script does not change training code, manifests, or source images. The
resize-before-augmentation comparison intentionally changes preprocessing
semantics; its speed is not presented as a pixel-equivalent replacement.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import random
import statistics
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from PIL import Image, ImageEnhance, ImageOps, ImageStat
import torch

from scripts.manifest_dataset import ResizeToTensor, resolve_data_path


RAMP = Image.frombytes("L", (256, 1), bytes(range(256)))


def rgb_lut_blend(image, constant, factor):
    # Use Pillow's own blend kernel to preserve its rounding/clamping exactly.
    base = Image.new("L", (256, 1), constant)
    lut = list(Image.blend(base, RAMP, factor).getdata())
    return image.point(lut * 3)


def augmentation(image, settings, seed, lut=False):
    rng = random.Random(seed)
    if rng.random() < float(settings.get("horizontal_flip", 0.5)):
        image = ImageOps.mirror(image)
    degrees = float(settings.get("rotation_degrees", 0))
    if degrees:
        image = image.rotate(rng.uniform(-degrees, degrees), resample=Image.Resampling.BILINEAR)
    jitter = float(settings.get("color_jitter", 0))
    if jitter:
        brightness = rng.uniform(max(0.0, 1 - jitter), 1 + jitter)
        contrast = rng.uniform(max(0.0, 1 - jitter), 1 + jitter)
        if lut:
            image = rgb_lut_blend(image, 0, brightness)
            mean = int(ImageStat.Stat(image.convert("L")).mean[0] + 0.5)
            image = rgb_lut_blend(image, mean, contrast)
        else:
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)
    return image


def milliseconds(function):
    start = time.perf_counter()
    value = function()
    return value, (time.perf_counter() - start) * 1000


def decode_file(path):
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def decode_memory(payload):
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB").copy()


def summary(values):
    ordered = sorted(values)
    return {"mean": statistics.mean(ordered), "median": statistics.median(ordered),
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "min": ordered[0], "max": ordered[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/dr_grading.json")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--samples-per-source", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--png-compress-level", type=int, choices=range(10), default=1)
    parser.add_argument("--skip-lut", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples_per_source < 1 or args.repeats < 1:
        parser.error("Sample and repeat counts must be positive")
    torch.set_num_threads(1)  # DataLoader worker processes also use one thread.
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if config["task"] != "dr_grading":
        parser.error("This profiler is scoped to the single-image DR grading manifests")
    model = config["model"]
    size = args.image_size or model.get("img_size", 392)
    height, width = (size, size) if isinstance(size, int) else size
    if min(height, width) < 1:
        parser.error("Image dimensions must be positive")
    tensor = ResizeToTensor((height, width), mean=model.get("mean"), std=model.get("std"))
    settings = config.get("augmentation", {}) if args.split == "train" else {"horizontal_flip": 0}
    data_root = args.data_root.resolve()
    prepared = data_root / "prepared"
    if config.get("data", {}).get("protocol", "strict_cross_domain") == "within_source":
        prepared = prepared / "protocols/within_source"
    manifest = prepared / "tasks/dr_grading" / (args.split + ".jsonl")
    sources = defaultdict(list)
    native_bytes = defaultdict(int)
    allowed = config.get("data", {}).get("datasets")
    with manifest.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if allowed and row["dataset"] not in allowed:
                continue
            if not row.get("label_mask", {}).get("dr_grade", False):
                continue
            sources[row["dataset"]].append(row)
            for image_size in row.get("image_sizes", []):
                native_bytes[row["dataset"]] += image_size[0] * image_size[1] * 3
    rng = random.Random(args.seed)
    details, aggregate = [], {}
    for source, rows in sorted(sources.items()):
        samples = rng.sample(rows, min(args.samples_per_source, len(rows)))
        measurements = defaultdict(list)
        compressed_bytes, matches = [], []
        for sample_index, row in enumerate(samples):
            path = resolve_data_path(row["image_paths"][0], data_root)
            for repeat in range(args.repeats):
                seed = args.seed + sample_index * 100003 + repeat
                native, decode_ms = milliseconds(lambda: decode_file(path))
                augmented, augment_ms = milliseconds(lambda: augmentation(native, settings, seed))
                original_tensor, tensor_ms = milliseconds(lambda: tensor(augmented))
                resized, initial_resize_ms = milliseconds(lambda: native.resize((width, height), Image.Resampling.BILINEAR))
                small_augmented, small_augment_ms = milliseconds(lambda: augmentation(resized, settings, seed))
                small_tensor, small_tensor_ms = milliseconds(lambda: tensor(small_augmented))
                png = io.BytesIO()
                _, encode_ms = milliseconds(lambda: resized.save(png, format="PNG", compress_level=args.png_compress_level))
                payload = png.getvalue()
                cached, cache_decode_ms = milliseconds(lambda: decode_memory(payload))
                cache_roundtrip_equal = cached.tobytes() == resized.tobytes()
                item = {"dataset": source, "path": row["image_paths"][0], "repeat": repeat,
                        "native_size": list(native.size), "encoded_source_bytes": path.stat().st_size,
                        "resized_png_bytes": len(payload), "decode_native_ms": decode_ms,
                        "native_augmentation_ms": augment_ms, "native_final_resize_tensor_ms": tensor_ms,
                        "resize_to_cache_size_ms": initial_resize_ms, "small_augmentation_ms": small_augment_ms,
                        "small_tensor_ms": small_tensor_ms, "cache_png_encode_ms": encode_ms,
                        "cache_png_memory_decode_ms": cache_decode_ms,
                        "original_total_ms": decode_ms + augment_ms + tensor_ms,
                        "resize_before_total_no_cache_ms": decode_ms + initial_resize_ms + small_augment_ms + small_tensor_ms,
                        "resize_before_png_cached_cpu_ms": cache_decode_ms + small_augment_ms + small_tensor_ms,
                        "cache_roundtrip_pixel_equal": cache_roundtrip_equal,
                        "resized_augmentation_tensor_equal_to_original": torch.equal(original_tensor, small_tensor)}
                if not args.skip_lut:
                    exact_augmented, lut_ms = milliseconds(lambda: augmentation(native, settings, seed, lut=True))
                    pixel_equal = augmented.tobytes() == exact_augmented.tobytes()
                    item.update(native_lut_augmentation_ms=lut_ms, native_lut_pixel_equal=pixel_equal,
                                native_lut_total_estimate_ms=decode_ms + lut_ms + tensor_ms)
                    matches.append(pixel_equal)
                for key, value in item.items():
                    if key.endswith("_ms"):
                        measurements[key].append(value)
                compressed_bytes.append(len(payload))
                details.append(item)
            print(json.dumps({"event": "source_sample_profiled", "dataset": source,
                              "completed": sample_index + 1, "selected": len(samples)}, ensure_ascii=False), flush=True)
        aggregate[source] = {"total_manifest_records": len(rows), "sampled_images": len(samples),
                             "measurements": len(samples) * args.repeats,
                             "native_decoded_rgb_gib": native_bytes[source] / 1024**3,
                             "resized_decoded_rgb_gib": len(rows) * width * height * 3 / 1024**3,
                             "estimated_resized_png_gib_from_sample": len(rows) * statistics.mean(compressed_bytes) / 1024**3,
                             "lut_all_sampled_pixels_equal": all(matches) if matches else None,
                             "timing_ms": {key: summary(values) for key, values in measurements.items()}}
    report = {"scope": "CPU-only sampled input pipeline, not model training or accuracy evaluation",
              "config": str(args.config.resolve()), "manifest": str(manifest), "split": args.split,
              "target_size_hw": [height, width], "seed": args.seed, "png_compress_level": args.png_compress_level,
              "source_counts": {key: len(value) for key, value in sources.items()},
              "notes": ["Original and alternatives use the same randomly drawn augmentation parameters.",
                        "Resize-before-augmentation changes the training pixel transformation; it is not mathematically equivalent.",
                        "Validation does not augment; resizing then lossless PNG caching is pixel-equivalent to its original resize.",
                        "The native LUT alternative retains original enhancement order and checks byte equality on every sampled image.",
                        "Native file decode includes file access, potentially from the OS page cache. Cached PNG decode uses in-memory encoded bytes and excludes disk access.",
                        "Repeated and source execution order may affect OS caches. Use cloud measurements and complete-epoch timing to judge actual throughput.",
                        "Compressed cache size is a sample estimate, not a guaranteed final disk footprint."],
              "by_source": aggregate, "samples": details}
    if args.output:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"by_source": aggregate, "output": str(args.output) if args.output else None}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
