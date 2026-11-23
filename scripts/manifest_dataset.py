"""Lazy image reader for the prepared ophthalmology JSONL manifests.

Pillow is the only required dependency. PyTorch is imported only by the optional
ResizeToTensor/classification_collate helpers. No pandas, NumPy, skimage or
torchvision are used. Source CSVs, raw_labels and diagnostic keywords are never
read into model inputs.

Example::

    dataset = ManifestDataset(
        'data/prepared/tasks/dr_grading/train.jsonl',
        image_transform=ResizeToTensor(384),
    )
    # torch.utils.data.DataLoader(dataset, collate_fn=classification_collate)

image_transform changes images only. Segmentation callers must apply matching
spatial transforms to masks themselves; load_png_mask uses nearest-neighbor when
an explicit mask size is requested.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
import json
from pathlib import Path, PureWindowsPath
from typing import Any

try:
    from PIL import Image
except ImportError as exc:
    raise ImportError("ManifestDataset requires Pillow in the current Python environment; no packages were changed.") from exc

_FILE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".xml", ".json", ".jsonl", ".csv", ".txt"}
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def infer_data_root(manifest_path: str | Path) -> Path:
    """Find the enclosing data/ directory without depending on the current cwd."""
    manifest = Path(manifest_path).expanduser().resolve()
    for parent in manifest.parents:
        if parent.name.casefold() == "data":
            return parent
    raise ValueError("Cannot infer the data root: place the manifest under data/ or pass data_root explicitly.")


def resolve_data_path(value: str | Path, data_root: str | Path) -> Path:
    """Resolve a data-relative manifest path and reject directory traversal."""
    root = Path(data_root).expanduser().resolve()
    text = str(value).replace("\\", "/")
    if not text or Path(text).is_absolute() or PureWindowsPath(text).drive:
        raise ValueError(f"Manifest paths must be nonempty and relative to data/: {value!r}")
    target = (root / text).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Manifest path leaves data_root: {value!r}")
    return target


def load_png_mask(path: str | Path, size: int | tuple[int, int] | None = None) -> Image.Image:
    """Load only a genuine PNG mask; preserve mode/values, never open raw TIFF.

    size is (height, width), matching tensor image conventions. No normalization
    or label-value remapping is performed.
    """
    source = Path(path)
    if source.suffix.lower() != ".png":
        raise ValueError("Mask loading is restricted to derived PNG files; raw TIFF masks must be converted first.")
    with source.open("rb") as handle:
        if handle.read(8) != _PNG_MAGIC:
            raise ValueError(f"Mask file does not have a PNG signature: {source}")
    with Image.open(source) as mask:
        result = mask.copy()
    if size is not None:
        height, width = _size_pair(size)
        result = result.resize((width, height), resample=Image.Resampling.NEAREST)
    return result


def _size_pair(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        height, width = size, size
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        height, width = size
    else:
        raise ValueError("size must be a positive int or a (height, width) pair")
    if not isinstance(height, int) or not isinstance(width, int) or min(height, width) <= 0:
        raise ValueError("Image dimensions must be positive integers")
    return height, width


@lru_cache(maxsize=1)
def _torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "Optional PyTorch support could not be imported in this Python environment. "
            "Use ManifestDataset in default PIL mode or repair the environment separately. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return torch


class ResizeToTensor:
    """Explicit RGB resize -> float32 CHW tensor in [0,1], without NumPy.

    Optional mean/std must both be supplied as three-channel sequences. This
    helper does not choose model-specific normalization or alter target masks.
    """

    def __init__(self, size: int | tuple[int, int] = 384,
                 mean: Sequence[float] | None = None, std: Sequence[float] | None = None):
        self.height, self.width = _size_pair(size)
        if (mean is None) != (std is None):
            raise ValueError("Supply mean and std together, or neither")
        if mean is not None and (len(mean) != 3 or len(std) != 3 or any(float(s) <= 0 for s in std)):
            raise ValueError("mean/std must contain three channels; std values must be positive")
        self.mean = tuple(float(x) for x in mean) if mean is not None else None
        self.std = tuple(float(x) for x in std) if std is not None else None

    def __call__(self, image: Image.Image):
        torch = _torch()
        rgb = image.convert("RGB").resize((self.width, self.height), resample=Image.Resampling.BILINEAR)
        # bytearray owns a writable buffer. No Tensor.numpy()/from_numpy path is
        # used, which avoids dependency on the installed NumPy binary interface.
        tensor = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
        tensor = tensor.reshape(self.height, self.width, 3).permute(2, 0, 1).contiguous()
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
        if self.mean is not None:
            mean = tensor.new_tensor(self.mean).view(3, 1, 1)
            std = tensor.new_tensor(self.std).view(3, 1, 1)
            tensor = (tensor - mean) / std
        return tensor


class ManifestDataset:
    """Map-style Dataset compatible with PyTorch DataLoader by protocol.

    Initialization indexes byte offsets only. Each __getitem__ reads one JSONL
    record and its image(s). A safe subset is returned, never the complete source
    record. Images are detached RGB PIL copies unless image_transform is given.
    """

    def __init__(self, manifest_path: str | Path, data_root: str | Path | None = None,
                 image_transform: Callable[[Image.Image], Any] | None = None,
                 image_loader: Callable[[Path], Image.Image] | None = None):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"JSONL manifest not found: {self.manifest_path}")
        self.data_root = Path(data_root).expanduser().resolve() if data_root is not None else infer_data_root(self.manifest_path)
        if not self.data_root.is_dir():
            raise NotADirectoryError(f"data_root is not a directory: {self.data_root}")
        self.image_transform = image_transform
        self.image_loader = image_loader
        self._offsets: list[tuple[int, int]] = []
        with self.manifest_path.open("rb") as handle:
            number = 0
            while True:
                position = handle.tell()
                line = handle.readline()
                if not line:
                    break
                number += 1
                if line.strip():
                    self._offsets.append((position, number))

    def __len__(self) -> int:
        return len(self._offsets)

    def _read_record(self, index: int) -> dict[str, Any]:
        if not isinstance(index, int):
            raise TypeError("ManifestDataset indices must be integers")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        position, line_number = self._offsets[index]
        with self.manifest_path.open("rb") as handle:
            handle.seek(position)
            raw = handle.readline()
        try:
            result = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSONL at {self.manifest_path}:{line_number}: {exc}") from exc
        if not isinstance(result, dict):
            raise ValueError(f"Manifest record must be an object at line {line_number}")
        return result

    def _annotations(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: self._annotations(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._annotations(item) for item in value]
        if isinstance(value, str) and Path(value).suffix.lower() in _FILE_SUFFIXES and not "://" in value:
            return str(resolve_data_path(value, self.data_root))
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._read_record(index)
        paths = record.get("image_paths")
        if paths is None:
            single = record.get("image_path")
            paths = [single] if single is not None else []
        if not isinstance(paths, list) or not paths or not all(isinstance(path, str) and path for path in paths):
            raise ValueError(f"Record {record.get('sample_id')} needs a nonempty image_paths list or image_path")
        if record.get("image_path") is not None and len(paths) == 1 and record["image_path"] != paths[0]:
            raise ValueError("image_path differs from the single canonical image_paths entry")
        images, absolute_paths = [], []
        for path in paths:
            absolute = resolve_data_path(path, self.data_root)
            if absolute.suffix.lower() in {".tif", ".tiff"}:
                raise ValueError("Raw TIFF image loading is disabled; reference a converted PNG image")
            try:
                if self.image_loader is None:
                    with Image.open(absolute) as source:
                        image = source.convert("RGB").copy()
                else:
                    image = self.image_loader(absolute)
                    if not isinstance(image, Image.Image):
                        raise TypeError("image_loader must return a detached PIL image")
            except Exception as exc:
                raise RuntimeError(f"Cannot load image for sample {record.get('sample_id')}: {absolute}: {exc}") from exc
            images.append(self.image_transform(image) if self.image_transform is not None else image)
            absolute_paths.append(str(absolute))
        supplied_labels = record.get("labels", {})
        supplied_mask = record.get("label_mask", {})
        if not isinstance(supplied_labels, dict) or not isinstance(supplied_mask, dict):
            raise ValueError("labels and label_mask must be dictionaries")
        labels, masks = {}, {}
        for key in sorted(supplied_labels.keys() | supplied_mask.keys()):
            available = supplied_mask.get(key, False)
            if not isinstance(available, bool):
                raise ValueError(f"label_mask[{key!r}] must be boolean")
            target = supplied_labels.get(key)
            if key not in supplied_labels:
                annotations = record.get("annotations", {})
                structured = annotations.get("masks", {}).get(key) or annotations.get("points", {}).get(key)
                structured = structured or any(expert.get(key) for expert in annotations.get("experts", {}).values())
                structured = structured or (key == "objects" and annotations.get("xml_path"))
                if available and not structured:
                    raise ValueError(f"Known structural supervision {key!r} has no annotation")
                masks[key] = available
                continue
            if available and target is None:
                raise ValueError(f"Known target {key!r} cannot be None")
            masks[key] = available
            labels[key] = target if available else None
        # Only explicit public training fields are returned. Do not expose raw
        # labels, diagnostic keywords, auxiliary textual answers or source rows.
        result = {key: record.get(key) for key in ["sample_id", "dataset", "task", "patient_id", "representation",
                                                  "content_group_id", "leakage_group_id", "content_group_evaluation_weight"]}
        result.update({
            "split": record.get("split", record.get("proposed_split")),
            "image": images[0] if len(images) == 1 else tuple(images),
            "images": images,
            "image_paths": absolute_paths,
            "labels": labels,
            "label_mask": masks,
            "annotations": self._annotations(record.get("annotations", {})),
        })
        return result

    def load_mask(self, annotation_path: str | Path, size: int | tuple[int, int] | None = None) -> Image.Image:
        """Load a derived PNG annotation while enforcing the same data root."""
        path = Path(annotation_path)
        if path.is_absolute():
            path = path.resolve()
            if not path.is_relative_to(self.data_root):
                raise ValueError("Mask path leaves data_root")
        else:
            path = resolve_data_path(path, self.data_root)
        return load_png_mask(path, size=size)


def classification_collate(batch: Sequence[dict[str, Any]], label_keys: Sequence[str] | None = None,
                           label_dtypes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Batch transformed classification samples without inventing negative labels.

    Requires tensor images of identical shape and equal view count. images is
    BCHW for single-eye batches and BVCHW for paired-view batches. labels are
    float32 by default; pass label_dtypes={'dr_grade':'long'} for CrossEntropy.
    Unknown targets are filled with zero only in the tensor, with mask=False;
    the loss must honor label_mask. Segmentation annotations remain path lists.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    torch = _torch()
    view_count = len(batch[0]["images"])
    if any(len(sample["images"]) != view_count for sample in batch):
        raise ValueError("Batch samples have different view counts; batch single and paired tasks separately")
    if any(not torch.is_tensor(image) or image.ndim != 3 for sample in batch for image in sample["images"]):
        raise TypeError("classification_collate requires CHW tensor images; use image_transform=ResizeToTensor(...)")
    try:
        image_tensor = torch.stack([torch.stack(sample["images"], dim=0) for sample in batch], dim=0)
    except RuntimeError as exc:
        raise ValueError("Image shapes differ; apply a shared resize transform before classification_collate") from exc
    if view_count == 1:
        image_tensor = image_tensor[:, 0]
    keys = list(label_keys) if label_keys is not None else sorted({key for sample in batch for key in sample["labels"]})
    dtype_map = label_dtypes or {}
    labels, masks = {}, {}
    for key in keys:
        values, available = [], []
        for sample in batch:
            known = sample["label_mask"].get(key, False)
            value = sample["labels"].get(key)
            if known and not isinstance(value, (int, float, bool)):
                raise TypeError(f"Classification label {key!r} must be a scalar number")
            values.append(value if known else 0)
            available.append(known)
        dtype = dtype_map.get(key, torch.float32)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, None)
            if dtype is None:
                raise ValueError(f"Unknown torch dtype for {key!r}")
        labels[key] = torch.tensor(values, dtype=dtype)
        masks[key] = torch.tensor(available, dtype=torch.bool)
    return {"images": image_tensor, "labels": labels, "label_mask": masks,
            "sample_ids": [sample["sample_id"] for sample in batch],
            "datasets": [sample["dataset"] for sample in batch],
            "patient_ids": [sample["patient_id"] for sample in batch],
            "splits": [sample["split"] for sample in batch], "view_count": view_count,
            "annotations": [sample["annotations"] for sample in batch],
            "annotation_supervision": [sample["label_mask"] for sample in batch],
            "image_paths": [sample["image_paths"] for sample in batch]}
