"""Deterministic fundus crop and letterbox, shared by source and cache paths.

This is image preprocessing, not label generation. The crop is deliberately
conservative: detect a substantial connected foreground on a <=512px thumbnail,
expand to its lower-threshold connected region, then add an explicit margin.
An unreliable/small foreground leaves the entire image intact. The heuristics
do not establish that every retained image is diagnostically readable.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import numpy as np
from PIL import Image


MODE = "fundus_crop_letterbox_v1"
DEFAULT_SPEC = {
    "mode": MODE,
    "black_threshold": 8,
    "analysis_size": 512,
    "crop_margin": 0.02,
    "min_component_fraction": 0.01,
    "min_axis_fraction": 0.25,
    "padding_color": [0, 0, 0],
}


def _canonical_spec(value):
    if not isinstance(value, Mapping):
        raise ValueError("preprocessing must be an object")
    mode = value.get("mode")
    if mode == "stretch":
        if set(value) != {"mode"}:
            raise ValueError("stretch preprocessing accepts only mode")
        return {"mode": "stretch"}
    if mode != MODE:
        raise ValueError(f"Unsupported preprocessing mode: {mode!r}")
    unknown = set(value) - set(DEFAULT_SPEC)
    if unknown:
        raise ValueError(f"Unknown preprocessing fields: {sorted(unknown)}")
    spec = {**DEFAULT_SPEC, **value}
    if type(spec["black_threshold"]) is not int or not 0 <= spec["black_threshold"] <= 254:
        raise ValueError("preprocessing.black_threshold must be an integer in [0,254]")
    if type(spec["analysis_size"]) is not int or not 32 <= spec["analysis_size"] <= 1024:
        raise ValueError("preprocessing.analysis_size must be an integer in [32,1024]")
    for key, lower, upper in (("crop_margin", 0.0, 0.25),
                              ("min_component_fraction", 0.0, 1.0),
                              ("min_axis_fraction", 0.0, 1.0)):
        number = spec[key]
        if isinstance(number, bool) or not isinstance(number, Real) or not math.isfinite(float(number)):
            raise ValueError(f"preprocessing.{key} must be finite numeric")
        if not lower <= number <= upper or (key != "crop_margin" and number == 0):
            raise ValueError(f"preprocessing.{key} is outside its valid range")
        spec[key] = float(number)
    color = spec["padding_color"]
    # This version's background is fixed black so training and deployment cannot
    # silently disagree about the normalization of padded pixels.
    if not isinstance(color, (list, tuple)) or len(color) != 3 or any(type(v) is not int or v != 0 for v in color):
        raise ValueError("preprocessing.padding_color must be [0,0,0]")
    spec["padding_color"] = [0, 0, 0]
    return spec


def preprocessing_spec(config):
    """Return an independent, JSON-serializable canonical preprocessing contract.

    Missing top-level ``preprocessing`` preserves the legacy stretch exactly.
    Explicit malformed/unknown settings fail rather than fall back silently.
    """
    if not isinstance(config, Mapping):
        raise ValueError("config must be an object")
    if "preprocessing" not in config:
        return {"mode": "stretch"}
    return _canonical_spec(config["preprocessing"])


def validate_preprocessing_config(config):
    """Shared early contract for preflight and training; no model is created."""
    spec = preprocessing_spec(config)
    if spec['mode'] != 'stretch':
        if config.get('kind') not in ('multiclass', 'multilabel'):
            raise ValueError('Fundus crop/letterbox is restricted to classification; masks need matching geometry')
        if config.get('augmentation', {}).get('resize_before_augmentation') is not True:
            raise ValueError('Fundus crop/letterbox requires augmentation.resize_before_augmentation=true')
    return spec


def _target_size(size):
    if type(size) is int:
        size = (size, size)
    if not isinstance(size, (tuple, list)) or len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
        raise ValueError("preprocessing size must be a positive int or (width,height)")
    return tuple(size)


def _components(mask, seed=None):
    """8-connected components using horizontal runs, avoiding per-pixel Python BFS.

    Each result has area, exclusive-right/bottom bbox, a foreground seed, and
    whether it contains the caller's seed. This needs only NumPy and scales with
    foreground runs rather than the whole-resolution source pixel count.
    """
    parent, runs, previous = [], [], []
    seed_label = None

    def find(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    for y, row in enumerate(mask):
        padded = np.pad(row.astype(np.int8, copy=False), (1, 1))
        edges = np.flatnonzero(np.diff(padded))
        current, j = [], 0
        for left, right in zip(edges[::2].tolist(), edges[1::2].tolist()):
            label = len(parent)
            parent.append(label)
            # Intervals [left,right) and a preceding interval are connected
            # when they overlap or their endpoint pixels touch diagonally.
            while j < len(previous) and previous[j][1] < left:
                j += 1
            k = j
            while k < len(previous) and previous[k][0] <= right:
                prior = find(previous[k][2])
                own = find(label)
                if prior != own:
                    parent[max(prior, own)] = min(prior, own)
                k += 1
            current.append((left, right, label))
            runs.append((y, left, right, label))
            if seed is not None and seed[1] == y and left <= seed[0] < right:
                seed_label = label
        previous = current
    seed_root = find(seed_label) if seed_label is not None else None
    merged = {}
    for y, left, right, label in runs:
        root = find(label)
        if root not in merged:
            merged[root] = {"area": 0, "bbox": [left, y, right, y + 1],
                            "seed": (left, y), "contains_seed": root == seed_root}
        info = merged[root]
        info["area"] += right - left
        box = info["bbox"]
        box[0], box[1] = min(box[0], left), min(box[1], y)
        box[2], box[3] = max(box[2], right), max(box[3], y + 1)
    return list(merged.values())


def _crop_box(rgb, spec):
    width, height = rgb.size
    thumb = rgb.copy()
    thumb.thumbnail((spec["analysis_size"], spec["analysis_size"]), Image.Resampling.BILINEAR)
    aw, ah = thumb.size
    brightness = np.asarray(thumb).max(axis=2)
    threshold = spec["black_threshold"]
    components = _components(brightness > threshold)
    if not components:
        return (0, 0, width, height), "no_reliable_foreground"
    # Largest substantial area normally picks the retina instead of detached
    # corner text or a thin bright frame; ties have deterministic scan order.
    main = max(components, key=lambda item: item["area"])
    left, top, right, bottom = main["bbox"]
    if (main["area"] < aw * ah * spec["min_component_fraction"] or
            right - left < aw * spec["min_axis_fraction"] or
            bottom - top < ah * spec["min_axis_fraction"]):
        return (0, 0, width, height), "foreground_too_small"
    # An expanded, connected lower-threshold region protects dark peripheral
    # retina. Background noise may make this retain more border, which is safer
    # than treating a bright central patch as the entire fundus.
    weak_threshold = threshold // 2
    if weak_threshold != threshold:
        expanded = _components(brightness > weak_threshold, seed=main["seed"])
        connected = next((item for item in expanded if item["contains_seed"]), None)
        if connected is not None:
            left, top, right, bottom = connected["bbox"]
    x0 = math.floor(left * width / aw)
    y0 = math.floor(top * height / ah)
    x1 = math.ceil(right * width / aw)
    y1 = math.ceil(bottom * height / ah)
    margin_x = math.ceil((x1 - x0) * spec["crop_margin"])
    margin_y = math.ceil((y1 - y0) * spec["crop_margin"])
    return (max(0, x0 - margin_x), max(0, y0 - margin_y),
            min(width, x1 + margin_x), min(height, y1 + margin_y)), None


def apply_preprocessing(image, size, spec=None, return_metadata=False):
    """Return RGB pixels for a declared contract; never mutate the input image.

    ``size`` is an int or (width,height), matching PIL/cache conventions.
    Set ``return_metadata`` for ``(image, dict)`` including source-space crop,
    fitted size and (left,top,right,bottom) padding. Dark/uncertain fallbacks keep
    all source pixels and still use aspect-preserving fit with black padding.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("preprocessing input must be a PIL image")
    target_width, target_height = _target_size(size)
    spec = _canonical_spec({"mode": "stretch"} if spec is None else spec)
    rgb = image.convert("RGB")
    width, height = rgb.size
    if min(width, height) < 1:
        raise ValueError("preprocessing input dimensions must be positive")
    metadata = {"mode": spec["mode"], "original_size": [width, height],
                "crop_bbox": [0, 0, width, height], "resized_size": [target_width, target_height],
                "padding": [0, 0, 0, 0], "fallback": False, "fallback_reason": None,
                "cropped": False}
    if spec["mode"] == "stretch":
        result = rgb.resize((target_width, target_height), Image.Resampling.BILINEAR)
    else:
        bbox, reason = _crop_box(rgb, spec)
        metadata.update(crop_bbox=list(bbox), fallback=reason is not None,
                        fallback_reason=reason, cropped=bbox != (0, 0, width, height))
        cropped = rgb.crop(bbox)
        ratio = min(target_width / cropped.width, target_height / cropped.height)
        fitted_width = min(target_width, max(1, round(cropped.width * ratio)))
        fitted_height = min(target_height, max(1, round(cropped.height * ratio)))
        fitted = cropped.resize((fitted_width, fitted_height), Image.Resampling.BILINEAR)
        left, top = (target_width - fitted_width) // 2, (target_height - fitted_height) // 2
        result = Image.new("RGB", (target_width, target_height), tuple(spec["padding_color"]))
        result.paste(fitted, (left, top))
        metadata.update(resized_size=[fitted_width, fitted_height],
                        padding=[left, top, target_width - fitted_width - left, target_height - fitted_height - top])
    return (result, metadata) if return_metadata else result
