"""Classification adapters for the immutable prepared ophthalmology manifests.

An ODIR record stays one supervised patient: both eyes pass through the same
backbone and their feature vectors are averaged before the prediction head.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from scripts.manifest_dataset import ManifestDataset, ResizeToTensor
from .metrics import classification_metrics
from .fundus_preprocessing import apply_preprocessing, preprocessing_spec


def _model_config(config):
    return config.get("model", {})


def _normalization(config):
    model = _model_config(config)
    mean, std = model.get("mean"), model.get("std")
    if (mean is None) != (std is None):
        raise ValueError("model.mean and model.std must be supplied together")
    if mean is None:
        if model.get("name", "tiny_cnn") == "tiny_cnn":
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            import timm
            pretrained = timm.models.get_pretrained_cfg(model["name"])
            if pretrained is None:
                raise ValueError("No model preprocessing metadata; configure model.mean and model.std explicitly")
            mean, std = pretrained.mean, pretrained.std
    return mean, std


class ClassificationTransform:
    def __init__(self, config, training, input_preprocessed=False):
        model = _model_config(config)
        mean, std = _normalization(config)
        self.tensor = ResizeToTensor(model.get("img_size", 384), mean=mean, std=std)
        self.training = training
        self.preprocessing = preprocessing_spec(config)
        self.input_preprocessed = input_preprocessed
        self.augmentation = config.get("augmentation", {})
        self.resize_before_augmentation = self.augmentation.get("resize_before_augmentation", False)
        if not isinstance(self.resize_before_augmentation, bool):
            raise ValueError("augmentation.resize_before_augmentation must be boolean")

    def __call__(self, image):
        # Cached images already contain this exact deterministic geometry.
        # Applying the crop again would trim the deliberate letterbox padding.
        if self.preprocessing['mode'] != 'stretch' and not self.input_preprocessed:
            image = apply_preprocessing(image, (self.tensor.width, self.tensor.height), self.preprocessing)
        if self.training:
            if self.resize_before_augmentation:
                image = image.convert("RGB").resize((self.tensor.width, self.tensor.height),
                                                     resample=Image.Resampling.BILINEAR)
            augmentation = self.augmentation
            if random.random() < float(augmentation.get("horizontal_flip", 0.5)):
                image = ImageOps.mirror(image)
            degrees = float(augmentation.get("rotation_degrees", 0))
            if degrees:
                image = image.rotate(random.uniform(-degrees, degrees), resample=Image.Resampling.BILINEAR)
            # Quality assessment should disable jitter: it changes the target.
            jitter = float(augmentation.get("color_jitter", 0))
            if jitter:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(max(0.0, 1 - jitter), 1 + jitter))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(max(0.0, 1 - jitter), 1 + jitter))
        return self.tensor(image)


class ClassificationDataset(Dataset):
    def __init__(self, manifest, data_root, config, training):
        self.config = config
        self.labels = list(config.get("labels", []))
        self.kind = config.get("kind")
        if not self.labels or self.kind not in {"multiclass", "multilabel"}:
            raise ValueError("Classification requires labels and kind=multiclass/multilabel")
        if self.kind == "multiclass" and (len(self.labels) != 1 or int(config.get("num_classes", 0)) < 2):
            raise ValueError("Multiclass requires one target label and num_classes >= 2")
        image_loader = None
        cache_root = config.get("data", {}).get("image_cache_dir")
        transform = ClassificationTransform(config, training, input_preprocessed=bool(cache_root))
        if cache_root:
            if training and not transform.resize_before_augmentation:
                raise ValueError("A resized training cache requires augmentation.resize_before_augmentation=true")
            from .image_cache import CachedImageLoader
            image_loader = CachedImageLoader(cache_root, data_root, (transform.tensor.width, transform.tensor.height),
                                             expected_preprocessing=transform.preprocessing)
        self.reader = ManifestDataset(manifest, data_root=data_root, image_transform=transform,
                                     image_loader=image_loader)
        sources = config.get("data", {}).get("datasets")
        allowed = set(sources) if sources else None
        self.indices = []
        self.source_counts = Counter()
        self.known_label_counts = Counter()
        self.positive_label_counts = Counter()
        self.class_counts = Counter()
        self.view_counts = Counter()
        for index in range(len(self.reader)):
            record = self.reader._read_record(index)
            if allowed is not None and record["dataset"] not in allowed:
                continue
            if record.get("task") != config["task"]:
                raise ValueError(f"Manifest task {record.get('task')!r} differs from configured task {config['task']!r}")
            masks, labels = record.get("label_mask", {}), record.get("labels", {})
            if not any(masks.get(label, False) for label in self.labels):
                continue
            for label in self.labels:
                if not masks.get(label, False):
                    continue
                value = labels.get(label)
                if self.kind == "multilabel" and value not in (0, 1, False, True):
                    raise ValueError(f"Non-binary known target {label} in {record['sample_id']}")
                if self.kind == "multiclass" and (not isinstance(value, int) or not 0 <= value < int(config["num_classes"])):
                    raise ValueError(f"Invalid class {value!r} in {record['sample_id']}")
                self.known_label_counts[label] += 1
                if self.kind == "multilabel":
                    self.positive_label_counts[label] += int(value)
                else:
                    self.class_counts[int(value)] += 1
            self.indices.append(index)
            self.source_counts[record["dataset"]] += 1
            self.view_counts[len(record["image_paths"])] += 1
        if len(self.view_counts) > 1:
            raise ValueError("Single-eye and paired-view records must use separate classification configurations")
        if training:
            absent = [label for label in self.labels if self.known_label_counts[label] == 0]
            if absent:
                raise ValueError(f"No training supervision for output labels {absent}; choose the correct protocol/source")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        record = self.reader[self.indices[index]]
        targets = [record["labels"].get(label) if record["label_mask"].get(label, False) else 0 for label in self.labels]
        mask = [bool(record["label_mask"].get(label, False)) for label in self.labels]
        if self.kind == "multiclass":
            targets, mask = int(targets[0]), mask[0]
        return {"images": torch.stack(record["images"]),
                "targets": torch.tensor(targets, dtype=torch.long if self.kind == "multiclass" else torch.float32),
                "mask": torch.tensor(mask, dtype=torch.bool),
                "sample_id": record["sample_id"], "dataset": record["dataset"],
                "content_group_id": record.get("content_group_id") or record["sample_id"],
                "evaluation_weight": float(record.get("content_group_evaluation_weight") or 1.0)}


def build_dataset(manifest: Path, data_root: Path, config: dict, training: bool):
    return ClassificationDataset(manifest, data_root, config, training)


def collate(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    images = torch.stack([s["images"] for s in samples])
    if images.shape[1] == 1:
        images = images[:, 0]
    return {"images": images, "targets": torch.stack([s["targets"] for s in samples]),
            "mask": torch.stack([s["mask"] for s in samples]),
            "sample_ids": [s["sample_id"] for s in samples],
            "datasets": [s["dataset"] for s in samples],
            "content_group_ids": [s["content_group_id"] for s in samples],
            "evaluation_weights": np.asarray([s["evaluation_weight"] for s in samples], dtype=np.float64)}


class TinyBackbone(nn.Module):
    """Offline smoke-test backbone, not a substitute for the configured model."""
    num_features = 64

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GELU(),
                                      nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
                                      nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, images):
        return self.features(images)


class SharedViewClassifier(nn.Module):
    def __init__(self, backbone, num_outputs, dropout=0.0):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.num_features, num_outputs))

    def forward(self, images):
        if images.ndim == 5:
            batch, views = images.shape[:2]
            features = self.backbone(images.flatten(0, 1))
            features = features.reshape(batch, views, -1).mean(dim=1)
        elif images.ndim == 4:
            features = self.backbone(images)
        else:
            raise ValueError("Expected BCHW or BVCHW image tensor")
        return self.head(features)


def build_model(config: dict, pretrained: bool = True):
    model = _model_config(config)
    local_checkpoint = model.get("checkpoint")
    if model.get("name", "tiny_cnn") == "tiny_cnn":
        backbone = TinyBackbone()
    else:
        import timm
        kwargs = dict(model.get("kwargs", {}))
        if "img_size" not in kwargs and "img_size" in model:
            # CNN constructors do not accept img_size. Only ViT families need it.
            if any(token in model["name"].lower() for token in ("vit", "deit", "beit", "eva")):
                kwargs["img_size"] = model["img_size"]
        backbone = timm.create_model(model["name"], pretrained=bool(pretrained and model.get("pretrained", True) and not local_checkpoint),
                                     num_classes=0, **kwargs)
        if model.get("gradient_checkpointing", model.get("grad_checkpointing", False)):
            if not hasattr(backbone, "set_grad_checkpointing"):
                raise ValueError("Configured backbone does not support gradient checkpointing")
            backbone.set_grad_checkpointing(True)
    if local_checkpoint and pretrained:
        _load_backbone_checkpoint(backbone, local_checkpoint)
    outputs = int(config["num_classes"]) if config["kind"] == "multiclass" else len(config["labels"])
    result = SharedViewClassifier(backbone, outputs, float(model.get("dropout", 0.0)))
    if model.get("freeze_backbone", False):
        for parameter in result.backbone.parameters():
            parameter.requires_grad_(False)
    return result


def _load_backbone_checkpoint(backbone, checkpoint):
    """Load a trusted local checkpoint for this exact backbone architecture.

    This intentionally does not claim compatibility with arbitrary RETFound/
    MAE checkpoints. Every backbone tensor must match, including position
    embeddings. Training resume is handled by the runner instead.
    """
    source = Path(checkpoint).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Local backbone checkpoint does not exist: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Backbone checkpoint must contain a state dictionary")
    for key in ("state_dict", "model", "backbone"):
        if isinstance(payload.get(key), dict):
            payload = payload[key]
            break
    expected = backbone.state_dict()
    state = {}
    for key, value in payload.items():
        normalized = str(key).removeprefix("module.").removeprefix("backbone.")
        if normalized in expected:
            state[normalized] = value
    missing = sorted(set(expected) - set(state))
    mismatched = [key for key in state if not torch.is_tensor(state[key]) or state[key].shape != expected[key].shape]
    if missing or mismatched:
        raise ValueError(f"Checkpoint is not an exact match for the configured backbone; missing={missing[:8]}, shape_mismatch={mismatched[:8]}")
    backbone.load_state_dict(state, strict=True)


def step(model, batch, device, config, training: bool):
    images = batch["images"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    logits = model(images)
    loss_config = config.get("loss", {})
    if config["kind"] == "multiclass":
        class_weights = loss_config.get("class_weights")
        weight = torch.as_tensor(class_weights, dtype=torch.float32, device=logits.device) if class_weights is not None else None
        if weight is not None and len(weight) != int(config["num_classes"]):
            raise ValueError("loss.class_weights must follow the class order")
        known = mask.bool()
        losses = F.cross_entropy(logits.float(), targets, weight=weight,
                                 label_smoothing=float(loss_config.get("label_smoothing", 0.0)), reduction="none")
        probabilities = logits.float().softmax(dim=-1)
    else:
        positive_weights = loss_config.get("pos_weight")
        pos_weight = torch.as_tensor(positive_weights, dtype=torch.float32, device=logits.device) if positive_weights is not None else None
        if pos_weight is not None and len(pos_weight) != len(config["labels"]):
            raise ValueError("loss.pos_weight must follow the configured label order")
        known = mask.bool()
        losses = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), pos_weight=pos_weight, reduction="none")
        probabilities = logits.float().sigmoid()
    # Keeping the zero-loss branch connected to logits also keeps DDP ranks in
    # sync if a partial-supervision batch has no known labels on one rank.
    denominator = known.sum().to(dtype=torch.float32)
    scale = 1
    if training and torch.distributed.is_available() and torch.distributed.is_initialized():
        # DDP averages gradients across ranks. Multiplying the local numerator
        # by world size makes that average equal to the global known-target
        # mean, including ranks with different supervision coverage.
        denominator = denominator.detach().clone()
        torch.distributed.all_reduce(denominator, op=torch.distributed.ReduceOp.SUM)
        scale = torch.distributed.get_world_size()
    loss = (losses * known).sum() * scale / denominator.clamp_min(1)
    output = {"targets": targets.detach().cpu().numpy(), "mask": mask.detach().cpu().numpy(),
              "probabilities": probabilities.detach().cpu().numpy(),
              "sample_ids": list(batch["sample_ids"]), "datasets": list(batch["datasets"]),
              "content_group_ids": list(batch["content_group_ids"]),
              "evaluation_weights": np.asarray(batch["evaluation_weights"]),
              "loss_weight": int(known.sum().item())}
    return loss, output


def metrics(outputs: list[dict], config: dict):
    return classification_metrics(outputs, config)
