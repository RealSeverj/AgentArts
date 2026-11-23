"""Strict loader for a separately stored, lossless resized-RGB image cache.

The cache is an explicit preprocessing representation, not a replacement for
source files or the immutable training manifests. Paths are relative to their
respective roots, and every load checks source metadata and cached pixel bytes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image

from .fundus_preprocessing import preprocessing_spec

FORMAT_VERSION = 1
REPRESENTATION = "rgb_resized_lossless_png"
RESIZE = "PIL_BILINEAR"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def normalize_image_size(value):
    """Return (width, height), matching cache_manifest.json's image_size."""
    if isinstance(value, int) and not isinstance(value, bool):
        value = (value, value)
    if (not isinstance(value, (tuple, list)) or len(value) != 2 or
            any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in value)):
        raise ValueError("Cache image_size must be a positive int or (width, height)")
    return tuple(value)


def canonical_relative_path(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Cache references must be nonempty relative path strings")
    normalized = value.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if (candidate.is_absolute() or PureWindowsPath(normalized).drive or
            any(part in ("..", ".") for part in normalized.split("/")) or
            ":" in normalized or candidate.as_posix() != normalized):
        raise ValueError(f"Unsafe or noncanonical cache reference: {value!r}")
    return normalized


def resolve_relative(root, value):
    normalized = canonical_relative_path(value)
    root = Path(root).resolve()
    target = (root / normalized).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"Cache reference leaves its declared root: {value!r}")
    return target


def validate_roots(cache_root, data_root):
    cache_root, data_root = Path(cache_root).resolve(), Path(data_root).resolve()
    # Neither root may contain the other. This deliberately requires a separate
    # cache location rather than writing into any original data subtree.
    if cache_root.is_relative_to(data_root) or data_root.is_relative_to(cache_root):
        raise ValueError("Cache root must be separate from, and outside, the original data root")
    return cache_root, data_root


def pixel_sha256(image):
    if image.mode != "RGB":
        raise ValueError("Pixel hashing requires an RGB image")
    return hashlib.sha256(image.tobytes()).hexdigest()


def read_verified_png(path, image_size, expected_hash):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cached PNG is missing: {path}")
    with path.open("rb") as handle:
        if handle.read(8) != PNG_SIGNATURE:
            raise ValueError(f"Cached file is not a genuine PNG: {path}")
    with Image.open(path) as source:
        if source.format != "PNG" or source.mode != "RGB" or source.size != tuple(image_size):
            raise ValueError(f"Cached PNG mode/size differs from the manifest: {path} ({source.mode}, {source.size})")
        image = source.copy()
    if pixel_sha256(image) != expected_hash:
        raise ValueError(f"Cached PNG pixel SHA256 mismatch: {path}")
    return image


def validate_manifest(manifest, expected_size=None, require_complete=True, expected_preprocessing=None):
    """Validate a cache, including its exact deterministic image preprocessing.

    Missing preprocessing metadata is the original stretch-resize contract.
    A caller that omits expected_preprocessing also requests this legacy
    contract; it cannot silently load a crop/letterbox cache.
    """
    if not isinstance(manifest, dict):
        raise ValueError("Image cache manifest must be a JSON object")
    if (type(manifest.get("format_version")) is not int or manifest.get("format_version") != FORMAT_VERSION or
            manifest.get("representation") != REPRESENTATION):
        raise ValueError("Unsupported resized image cache format or representation")
    size = normalize_image_size(manifest.get("image_size"))
    if expected_size is not None and size != normalize_image_size(expected_size):
        raise ValueError(f"Image cache dimensions {size} differ from requested {normalize_image_size(expected_size)}")
    if manifest.get("resize") != RESIZE:
        raise ValueError("Image cache was not declared as PIL_BILINEAR")
    declared = manifest.get("preprocessing", {"mode": "stretch"})
    actual_spec = preprocessing_spec({"preprocessing": declared})
    if "preprocessing" in manifest and actual_spec != declared:
        raise ValueError("Image cache preprocessing metadata must be canonical")
    expected_spec = preprocessing_spec({"preprocessing": ({"mode": "stretch"} if expected_preprocessing is None
                                                          else expected_preprocessing)})
    if actual_spec != expected_spec:
        raise ValueError("Image cache preprocessing differs from the configured preprocessing; use a separate cache root")
    if require_complete and manifest.get("pixel_equivalence_verified") is not True:
        raise ValueError("Image cache build is incomplete or pixel equivalence was not verified")
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("Image cache manifest requires an entries mapping")
    for source_path, entry in entries.items():
        canonical_relative_path(source_path)
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid image cache entry: {source_path}")
        canonical_relative_path(entry.get("cache_path"))
        for key in ("source_size", "source_mtime_ns"):
            if not isinstance(entry.get(key), int) or isinstance(entry[key], bool) or entry[key] < 0:
                raise ValueError(f"Invalid {key} for image cache entry {source_path}")
        digest = entry.get("pixel_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(v not in "0123456789abcdef" for v in digest):
            raise ValueError(f"Invalid pixel_sha256 for image cache entry {source_path}")
    return size


class CachedImageLoader:
    """Callable compatible with ManifestDataset's optional image_loader hook.

    image_size is an int or (width, height). Source size means encoded file
    bytes, not image dimensions. Missing/stale files raise; no fallback occurs.
    """
    is_resized_image_cache = True

    def __init__(self, cache_root, data_root, image_size=None, expected_preprocessing=None):
        self.cache_root, self.data_root = validate_roots(cache_root, data_root)
        path = self.cache_root / "cache_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Image cache manifest is missing: {path}")
        self.manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        self.image_size = validate_manifest(self.manifest, image_size,
                                            expected_preprocessing=expected_preprocessing)
        self.entries = self.manifest["entries"]

    def __call__(self, absolute_source):
        source = Path(absolute_source)
        if not source.is_absolute():
            raise ValueError("CachedImageLoader requires an absolute source-image path")
        source = source.resolve()
        if not source.is_relative_to(self.data_root):
            raise ValueError(f"Original image leaves the declared data root: {source}")
        relative = source.relative_to(self.data_root).as_posix()
        if relative not in self.entries:
            raise KeyError(f"Image is absent from the configured cache: {relative}")
        entry = self.entries[relative]
        if not source.is_file():
            raise FileNotFoundError(f"Original source image is missing: {source}")
        stat = source.stat()
        if stat.st_size != entry["source_size"] or stat.st_mtime_ns != entry["source_mtime_ns"]:
            raise ValueError(f"Original source changed since the cache was built: {relative}")
        cached = resolve_relative(self.cache_root, entry["cache_path"])
        return read_verified_png(cached, self.image_size, entry["pixel_sha256"])


ResizedImageCache = CachedImageLoader
