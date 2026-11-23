"""Load the four owned, SHA-pinned experts without training or downloads.

Training checkpoints are Python pickle containers. Only the exact audited
artifacts listed in SPECS are accepted; never expose checkpoint selection to
an HTTP caller. Future model versions require an explicit registry revision.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch

from training.fundus_preprocessing import DEFAULT_SPEC, preprocessing_spec


STAGES = ("quality", "multidisease", "dr_grading", "disc_cup_segmentation")
DISEASE_LABELS = ("DR", "AMD", "MYOPIC_FUNDUS", "OPTIC_DISC_CUPPING", "ABNORMAL")
QUALITY_LABELS = ("Good", "Usable", "Reject")
MASK_LABELS = ("optic_cup", "optic_disc")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SPECS = {
    "quality": {"task": "image_quality", "kind": "multiclass", "labels": ["quality_grade"],
                "num_classes": 3, "source": "EyeQ", "size": 392, "epoch": 11,
                "architecture": "vit_base_patch14_dinov2.lvd142m",
                "sha256": "fa5ee22b483b09c8dec9050455b5e6010d51c2ecced17d0a55982657bdd66f8f",
                "model_id": "quality_eyeq_vitb14_392_best11_20261114"},
    "multidisease": {"task": "multidisease", "kind": "multilabel", "labels": list(DISEASE_LABELS),
                     "source": "RFMiD", "size": 392, "epoch": 21,
                     "architecture": "vit_base_patch14_dinov2.lvd142m",
                     "sha256": "f5352ccb0834b41a2d9e6ed6d43c80b7b7b2e87dcfa1c0b3690f96eddafca640",
                     "model_id": "multidisease_rfmid_vitb14_392_best21_20261114"},
    "dr_grading": {"task": "dr_grading", "kind": "multiclass", "labels": ["dr_grade"],
                   "num_classes": 5, "source": "DDR", "size": 336, "epoch": 24,
                   "architecture": "vit_large_patch14_dinov2.lvd142m",
                   "sha256": "a85acf367a34361986739e93030bc5c51a390ddd33c63d49da883abf1562d4ab",
                   "model_id": "dr_ddr_vitl14_crop336_best24_20261114"},
    "disc_cup_segmentation": {"task": "disc_cup_segmentation", "kind": "segmentation", "labels": list(MASK_LABELS),
                              "source": "REFUGE", "size": 512, "epoch": 33,
                              "architecture": "unet",
                              "sha256": "bb25840047eedca5b115c3a81c9f7b8e3e73a8a0729cc6c67b9f55b1c38ef5e7",
                              "model_id": "disc_cup_refuge_unet32_512_best33_20261114"},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _stat(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def load_verified_checkpoint(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Hash the opened artifact BEFORE deserialization; protect stat identity."""
    require(re.fullmatch(r"[a-f0-9]{64}", expected_sha256) is not None, "Invalid checkpoint SHA-256")
    require(path.is_file() and not path.is_symlink(), "Checkpoint must be an ordinary local file")
    before = _stat(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        import os
        opened = os.fstat(handle.fileno())
        # Windows Python 3.11 stat/fstat report different ctime semantics
        # (creation vs change time). Compare identity/size/mtime across APIs;
        # retain ctime for the path-to-path checks before and after loading.
        require((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) == before[:4],
                "Checkpoint changed before hashing")
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    require(digest.hexdigest() == expected_sha256, "Checkpoint SHA-256 mismatch; weights were not deserialized")
    require(_stat(path) == before, "Checkpoint changed while hashing")
    # mmap avoids materializing the unused optimizer tensors. load_state_dict
    # later copies only model tensors; optimizers and training state are never restored.
    result = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    require(_stat(path) == before, "Checkpoint changed while loading")
    require(isinstance(result, dict), "Checkpoint payload must be a mapping")
    return result


def validate_checkpoint_config(stage: str, checkpoint: dict) -> dict:
    """Validate task, source, channel order and the exact recorded geometry."""
    spec = SPECS[stage]
    config = checkpoint.get("config")
    require(isinstance(config, dict), f"{stage}: checkpoint configuration missing")
    require(config.get("task") == spec["task"] and config.get("kind") == spec["kind"], f"{stage}: task/kind differs")
    require(config.get("labels") == spec["labels"], f"{stage}: output label order differs")
    if "num_classes" in spec:
        require(config.get("num_classes") == spec["num_classes"], f"{stage}: class count differs")
    model, data = config.get("model", {}), config.get("data", {})
    require(model.get("name") == spec["architecture"] and model.get("img_size") == spec["size"], f"{stage}: architecture/size differs")
    require(model.get("mean") == MEAN and model.get("std") == STD, f"{stage}: normalization differs")
    require(not model.get("checkpoint"), f"{stage}: external backbone checkpoint not supported")
    require(data.get("datasets") == [spec["source"]] and data.get("source_isolated") is True,
            f"{stage}: training source differs")
    require(data.get("protocol") == "strict_cross_domain", f"{stage}: source protocol differs")
    expected_preprocessing = dict(DEFAULT_SPEC) if stage == "dr_grading" else {"mode": "stretch"}
    require(preprocessing_spec(config) == expected_preprocessing, f"{stage}: preprocessing differs")
    if stage == "disc_cup_segmentation":
        require(model.get("base_channels") == 32 and model.get("pretrained") is False, "U-Net channel configuration differs")
        require(config.get("evaluation", {}).get("threshold") == 0.5, "U-Net mask threshold differs")
    if stage == "dr_grading":
        parent = config.get("initialization", {})
        require(parent.get("parent_training_sources") == ["APTOS2019"], "DR parent history differs")
        require(config.get("augmentation", {}).get("resize_before_augmentation") is True, "DR crop ordering differs")
    else:
        require(config.get("initialization") is None, f"{stage}: supervised weight transfer differs")
    require(type(checkpoint.get("epoch")) is int and checkpoint["epoch"] + 1 == spec["epoch"], f"{stage}: selected epoch differs")
    identity = checkpoint.get("data_identity")
    expected_identity = {k: v for k, v in config.items() if k != "notes"}
    expected_identity["training"] = {k: v for k, v in config["training"].items() if k not in ("workers", "log_every")}
    require(isinstance(identity, dict) and identity.get("experiment") == expected_identity, f"{stage}: saved configuration identity differs")
    tensors = checkpoint.get("model")
    require(isinstance(tensors, dict) and bool(tensors), f"{stage}: model state missing")
    for tensor in tensors.values():
        require(torch.is_tensor(tensor) and tensor.device.type == "cpu", f"{stage}: invalid model tensor")
        require(not torch.is_floating_point(tensor) or bool(torch.isfinite(tensor).all()), f"{stage}: non-finite model tensor")
    return deepcopy(config)


@dataclass(frozen=True)
class RegistryEntry:
    stage: str
    checkpoint: Path
    sha256: str
    model_id: str


@dataclass
class LoadedExpert:
    stage: str
    model_id: str
    sha256: str
    config: dict
    model: torch.nn.Module
    completed_epoch: int


class ModelRegistry:
    def __init__(self, entries: dict[str, RegistryEntry]):
        require(set(entries) == set(STAGES), "Registry must declare exactly the four experts")
        self.entries = dict(entries)

    @classmethod
    def from_file(cls, path, weights_root=None):
        path = Path(path).expanduser().resolve()
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        models = document.get("models")
        require(isinstance(models, dict) and set(models) == set(STAGES), "models must declare exactly four experts")
        root = Path(weights_root).expanduser().resolve() if weights_root is not None else path.parent
        entries = {}
        for stage in STAGES:
            item = models[stage]
            require(isinstance(item, dict) and not (set(item) - {"checkpoint", "sha256", "model_id"}), f"{stage}: unknown registry fields")
            require(isinstance(item.get("checkpoint"), str) and bool(item["checkpoint"]), f"{stage}: checkpoint path missing")
            require(item.get("sha256") == SPECS[stage]["sha256"], f"{stage}: checkpoint is not the audited owned model version")
            model_id = item.get("model_id", SPECS[stage]["model_id"])
            require(model_id == SPECS[stage]["model_id"], f"{stage}: model ID differs from the audited version")
            checkpoint = Path(item["checkpoint"]).expanduser()
            if weights_root is not None:
                require(not checkpoint.is_absolute() and not checkpoint.drive and ".." not in checkpoint.parts,
                        f"{stage}: paths under weights_root must be relative")
            resolved = (checkpoint if checkpoint.is_absolute() else root / checkpoint).resolve()
            if weights_root is not None:
                require(resolved.is_relative_to(root), f"{stage}: checkpoint leaves weights_root")
            entries[stage] = RegistryEntry(stage, resolved, item["sha256"], model_id)
        return cls(entries)

    def load_models(self):
        """No manifest, cache, parent checkpoint, pretrained URL or optimizer reads."""
        result = {}
        for stage in STAGES:
            entry = self.entries[stage]
            require(entry.sha256 == SPECS[stage]["sha256"], f"{stage}: untrusted checkpoint digest")
            checkpoint = load_verified_checkpoint(entry.checkpoint, entry.sha256)
            config = validate_checkpoint_config(stage, checkpoint)
            if stage == "disc_cup_segmentation":
                from training.structural import build_model
            else:
                from training.classification import build_model
            model = build_model(config, pretrained=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval().requires_grad_(False)
            result[stage] = LoadedExpert(stage, entry.model_id, entry.sha256, config, model, checkpoint["epoch"] + 1)
            del checkpoint
        return result
