"""Task adapters for prepared segmentation, landmarks and DDR detection.

Only derived PNG mask ``path`` fields are training targets. ``source_path``
fields remain audit references and are never decoded here. Validation of a
patch-trained segmentation model uses a resized WHOLE image, not a center crop;
reported Dice/IoU therefore describe the configured resized-image resolution.
"""
from __future__ import annotations

import json
import contextlib
import io
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from scripts.manifest_dataset import resolve_data_path, load_png_mask


def _size(config):
    size = config.get("model", {}).get("img_size", 512)
    if isinstance(size, int):
        size = (size, size)
    if len(size) != 2 or min(size) < 16:
        raise ValueError("model.img_size must be >=16 or [height,width]")
    return tuple(int(v) for v in size)


def _tensor(image, config, normalize=True):
    array = np.asarray(image, dtype=np.uint8).copy()
    result = torch.from_numpy(array).permute(2, 0, 1).float().div_(255)
    if normalize:
        model = config.get("model", {})
        mean, std = model.get("mean") or [0., 0., 0.], model.get("std") or [1., 1., 1.]
        if len(mean) != 3 or len(std) != 3 or min(std) <= 0:
            raise ValueError("model.mean/std must be three channels with positive std")
        result = (result - result.new_tensor(mean)[:, None, None]) / result.new_tensor(std)[:, None, None]
    return result


def _mask_spec(row, label, config):
    annotations = row["annotations"]
    if "masks" in annotations:
        return annotations["masks"].get(label)
    experts = annotations.get("experts", {})
    if "reference" in experts:
        return experts["reference"].get(label)
    name = config.get("data", {}).get("expert", "expert_1")
    if name not in experts:
        raise ValueError(f"Expert {name!r} is unavailable for {row['sample_id']}; have {list(experts)}")
    return experts[name].get(label)


def parse_detection_xml(path, labels, image_size, coordinate_mode):
    """Parse source coordinates with an EXPLICIT conversion policy.

    source_xyxy_no_offset preserves source numbers as continuous xyxy; this
    policy does not claim that the source's unpublished origin is known.
    voc_1based_inclusive converts inclusive one-based VOC to zero-based
    half-open xyxy. Choose it only with source evidence for that convention.
    """
    modes = {"source_xyxy_no_offset", "voc_1based_inclusive", "zero_based_inclusive"}
    if coordinate_mode not in modes:
        raise ValueError("data.box_coordinate_mode must explicitly be one of " + ", ".join(sorted(modes)))
    root = ET.parse(path).getroot()
    width, height = image_size
    declared = (int(root.findtext("size/width", "0")), int(root.findtext("size/height", "0")))
    if declared != image_size:
        raise ValueError(f"Detection XML size {declared} differs from image {image_size}: {path}")
    lookup = {name.lower(): i + 1 for i, name in enumerate(labels)}
    boxes, classes, difficult = [], [], []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in lookup:
            raise ValueError(f"Unknown detection class {name!r}: {path}")
        box = [float(obj.findtext("bndbox/" + k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        if coordinate_mode == "voc_1based_inclusive":
            box[0] -= 1
            box[1] -= 1
        elif coordinate_mode == "zero_based_inclusive":
            box[2] += 1
            box[3] += 1
        if not all(np.isfinite(box)) or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError(f"Invalid/outside detection box {box}: {path}; check coordinate convention")
        # Source 'difficult' is retained separately. It is not COCO 'iscrowd'.
        boxes.append(box)
        classes.append(lookup[name])
        difficult.append(int(obj.findtext("difficult", "0")))
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(classes, dtype=torch.int64),
            "difficult": torch.tensor(difficult, dtype=torch.int64)}


class StructuralDataset(Dataset):
    def __init__(self, manifest, data_root, config, training):
        self.root, self.config, self.training = Path(data_root).resolve(), config, training
        self.kind, self.labels = config["kind"], list(config["labels"])
        self.size = _size(config)
        selected_sources = set(config.get("data", {}).get("datasets") or [])
        self.rows = []
        with Path(manifest).open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if selected_sources and row["dataset"] not in selected_sources:
                    continue
                keys = ["objects"] if self.kind == "detection" else self.labels
                if not any(row["label_mask"].get(label, False) for label in keys):
                    continue
                self.rows.append(row)
        if self.kind == "detection":
            # Fail early, including before an expensive detector download.
            mode = config.get("data", {}).get("box_coordinate_mode")
            if mode not in {"source_xyxy_no_offset", "voc_1based_inclusive", "zero_based_inclusive"}:
                raise ValueError("Detection requires explicit data.box_coordinate_mode; origin is not declared by source")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        paths = row["image_paths"]
        if len(paths) != 1:
            raise ValueError("Structural supervision requires one image per record")
        with Image.open(resolve_data_path(paths[0], self.root)) as source:
            image = source.convert("RGB")
        width, height = image.size
        out_h, out_w = self.size
        aug = self.config.get("augmentation", {})
        hflip = self.training and random.random() < float(aug.get("horizontal_flip", 0.5))
        vflip = self.training and random.random() < float(aug.get("vertical_flip", 0.0))
        known = torch.tensor([bool(row["label_mask"].get(label, False)) for label in self.labels])
        result = {"sample_id": row["sample_id"], "dataset": row["dataset"], "original_size": (width, height),
                  "content_group_id": row.get("content_group_id"),
                  "content_group_evaluation_weight": row.get("content_group_evaluation_weight", 1.0)}
        if self.kind == "segmentation":
            masks = []
            for channel, label in enumerate(self.labels):
                spec = _mask_spec(row, label, self.config)
                if known[channel]:
                    if not spec or not spec.get("path"):
                        raise ValueError(f"Known mask {label} missing in {row['sample_id']}")
                    mask = load_png_mask(resolve_data_path(spec["path"], self.root))
                    if mask.mode != "L" or mask.size != image.size:
                        raise ValueError(f"Derived mask must be native-size L PNG: {spec['path']}")
                    # Fail on unexpected soft-label/intermediate encodings.
                    values = np.unique(np.asarray(mask))
                    if not np.isin(values, [0, 255]).all():
                        raise ValueError(f"Derived mask is not binary 0/255: {spec['path']}")
                else:
                    mask = Image.new("L", image.size, 0)
                masks.append(mask)
            patch = int(aug.get("patch_size") or 0) if self.training else 0
            if patch:
                if patch < 16:
                    raise ValueError("augmentation.patch_size must be at least 16")
                # Padding and cropping use identical geometry for every target.
                if width < patch or height < patch:
                    border = (0, 0, max(0, patch - width), max(0, patch - height))
                    image = ImageOps.expand(image, border=border, fill=0)
                    masks = [ImageOps.expand(m, border=border, fill=0) for m in masks]
                width2, height2 = image.size
                left, top = random.randint(0, width2 - patch), random.randint(0, height2 - patch)
                if random.random() < float(aug.get("positive_crop_probability", 0.0)):
                    candidates = [m for m, k in zip(masks, known) if k and m.getbbox() is not None]
                    if candidates:
                        chosen = np.asarray(random.choice(candidates))
                        ys, xs = np.nonzero(chosen)
                        position = random.randrange(len(xs))
                        left = min(max(0, int(xs[position]) - random.randrange(patch)), width2 - patch)
                        top = min(max(0, int(ys[position]) - random.randrange(patch)), height2 - patch)
                box = (left, top, left + patch, top + patch)
                image = image.crop(box)
                masks = [m.crop(box) for m in masks]
            if hflip:
                image, masks = ImageOps.mirror(image), [ImageOps.mirror(m) for m in masks]
            if vflip:
                image, masks = ImageOps.flip(image), [ImageOps.flip(m) for m in masks]
            image = image.resize((out_w, out_h), Image.Resampling.BILINEAR)
            targets = [torch.from_numpy(np.asarray(m.resize((out_w, out_h), Image.Resampling.NEAREST)).copy()).float().div_(255) for m in masks]
            result.update(image=_tensor(image, self.config), targets=torch.stack(targets), known=known)
        elif self.kind == "landmark":
            points = torch.zeros((len(self.labels), 2), dtype=torch.float32)
            for channel, label in enumerate(self.labels):
                if known[channel]:
                    xy = row["annotations"]["points"][label]["xy"]
                    if not (0 <= xy[0] <= width and 0 <= xy[1] <= height):
                        raise ValueError(f"Point outside image: {row['sample_id']} {label}")
                    # Preserve source numeric convention. No unknown-origin offset.
                    points[channel] = torch.tensor([xy[0] / width, xy[1] / height])
            if hflip:
                image = ImageOps.mirror(image)
                points[:, 0] = 1 - points[:, 0]
            if vflip:
                image = ImageOps.flip(image)
                points[:, 1] = 1 - points[:, 1]
            result.update(image=_tensor(image.resize((out_w, out_h), Image.Resampling.BILINEAR), self.config),
                          targets=points, known=known)
        elif self.kind == "detection":
            target = parse_detection_xml(resolve_data_path(row["annotations"]["xml_path"], self.root), self.labels,
                                         image.size, self.config["data"]["box_coordinate_mode"])
            target["boxes"] *= torch.tensor([out_w / width, out_h / height, out_w / width, out_h / height])
            if hflip:
                image = ImageOps.mirror(image)
                target["boxes"][:, [0, 2]] = out_w - target["boxes"][:, [2, 0]]
            if vflip:
                image = ImageOps.flip(image)
                target["boxes"][:, [1, 3]] = out_h - target["boxes"][:, [3, 1]]
            target["area"] = (target["boxes"][:, 2:] - target["boxes"][:, :2]).prod(1)
            target["iscrowd"] = torch.zeros(len(target["boxes"]), dtype=torch.int64)
            target["image_id"] = torch.tensor(index, dtype=torch.int64)
            result.update(image=_tensor(image.resize((out_w, out_h), Image.Resampling.BILINEAR), self.config, normalize=False),
                          targets=target, known=torch.tensor(True))
        else:
            raise ValueError(f"Unknown structural kind {self.kind}")
        return result


def build_dataset(manifest: Path, data_root: Path, config: dict, training: bool):
    return StructuralDataset(manifest, data_root, config, training)


def collate(samples):
    batch = {"images": torch.stack([s["image"] for s in samples]),
             "sample_ids": [s["sample_id"] for s in samples], "datasets": [s["dataset"] for s in samples],
             "original_sizes": torch.tensor([s["original_size"] for s in samples], dtype=torch.float32),
             "content_group_ids": [s.get("content_group_id") for s in samples],
             "content_group_evaluation_weights": [s.get("content_group_evaluation_weight", 1.) for s in samples],
             "known": torch.stack([s["known"] for s in samples])}
    batch["targets"] = ([s["targets"] for s in samples] if isinstance(samples[0]["targets"], dict)
                        else torch.stack([s["targets"] for s in samples]))
    return batch


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels, output_channels):
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        super().__init__(nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(groups, output_channels), nn.SiLU(inplace=True),
                         nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(groups, output_channels), nn.SiLU(inplace=True))


class UNet(nn.Module):
    """Four-level U-Net with GroupNorm for small per-GPU microbatches."""
    def __init__(self, classes, base=32):
        super().__init__()
        widths = [base * (2 ** i) for i in range(5)]
        self.encoders = nn.ModuleList([ConvBlock(3 if i == 0 else widths[i-1], c) for i, c in enumerate(widths)])
        self.decoders = nn.ModuleList([ConvBlock(widths[i+1] + widths[i], widths[i]) for i in range(3, -1, -1)])
        self.head = nn.Conv2d(base, classes, 1)

    def forward(self, image):
        skips = []
        for i, encoder in enumerate(self.encoders):
            image = encoder(image if i == 0 else F.max_pool2d(image, 2))
            skips.append(image)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            image = decoder(torch.cat([F.interpolate(image, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        return self.head(image)


class LandmarkModel(nn.Module):
    def __init__(self, encoder, landmarks):
        super().__init__()
        self.encoder, self.landmarks = encoder, landmarks

    def forward(self, images):
        return self.encoder(images).reshape(-1, self.landmarks, 2).sigmoid()


def build_model(config, pretrained=True):
    kind, model_cfg, classes = config["kind"], config.get("model", {}), len(config["labels"])
    if kind == "segmentation":
        if model_cfg.get("name", "unet") != "unet":
            raise ValueError("Segmentation currently supports model.name=unet")
        return UNet(classes, int(model_cfg.get("base_channels", 32)))
    if kind == "landmark":
        name = model_cfg.get("name", "resnet18")
        if name == "tiny_cnn":
            encoder = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, classes * 2))
        else:
            import timm
            kwargs = dict(pretrained=bool(pretrained and model_cfg.get("pretrained", True)), num_classes=classes * 2)
            if any(token in name for token in ("vit", "deit", "beit", "eva", "dinov")):
                h, w = _size(config)
                kwargs["img_size"] = (h, w)
            encoder = timm.create_model(name, **kwargs)
            if model_cfg.get("grad_checkpointing", False):
                if not hasattr(encoder, "set_grad_checkpointing"):
                    raise ValueError(f"Model {name!r} does not expose gradient checkpointing")
                encoder.set_grad_checkpointing(True)
        return LandmarkModel(encoder, classes)
    if kind == "detection":
        from torchvision.models.detection import (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights)
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        builders = {
            "fasterrcnn_resnet50_fpn": (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights.DEFAULT),
            "fasterrcnn_mobilenet_v3_large_fpn": (fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
        }
        name = model_cfg.get("name", "fasterrcnn_resnet50_fpn")
        if name not in builders:
            raise ValueError(f"Unsupported detector {name!r}; choose {list(builders)}")
        height, width = _size(config)
        builder, default_weights = builders[name]
        weights = default_weights if pretrained and model_cfg.get("pretrained", True) else None
        model = builder(weights=weights, weights_backbone=None, min_size=min(height, width),
                        max_size=max(height, width), box_detections_per_img=int(model_cfg.get("detections_per_image", 300)))
        features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(features, classes + 1)
        return model
    raise ValueError(f"Unknown kind: {kind}")


def step(model, batch, device, config, training):
    images = batch["images"].to(device, non_blocking=True)
    output = {"sample_ids": batch["sample_ids"], "datasets": batch["datasets"]}
    kind = config["kind"]
    if kind == "detection":
        targets = [{key: value.to(device) for key, value in target.items() if key != "difficult"}
                   for target in batch["targets"]]
        if training:
            losses = model(list(images), targets)
            loss = sum(losses.values())
            output["loss_components"] = {key: float(value.detach()) for key, value in losses.items()}
        else:
            predictions = model(list(images))
            # torchvision eval returns predictions only, not validation loss.
            loss = images.new_zeros(())
            output["predictions"] = [{k: v.detach().cpu() for k, v in p.items()} for p in predictions]
            output["targets"] = [{k: v.detach().cpu() for k, v in t.items()} for t in batch["targets"]]
            output["validation_loss_available"] = False
        return loss, output
    targets, known = batch["targets"].to(device), batch["known"].to(device)
    prediction = model(images)
    if kind == "segmentation":
        # Explicit float32 reductions stay stable under FP16 autocast.
        prediction = prediction.float()
        bce = F.binary_cross_entropy_with_logits(prediction, targets.float(), reduction="none").mean((-2, -1))
        probabilities = prediction.sigmoid()
        intersection = (probabilities * targets).sum((-2, -1))
        dice_loss = 1 - (2 * intersection + 1.) / (probabilities.sum((-2, -1)) + targets.sum((-2, -1)) + 1.)
        options = config.get("loss", {})
        per_channel = float(options.get("bce_weight", 1.0)) * bce + float(options.get("dice_weight", 1.0)) * dice_loss
        loss = _known_mean((per_channel * known).sum(), known, training)
        threshold = float(config.get("evaluation", {}).get("threshold", 0.5))
        positive, truth = probabilities.detach() >= threshold, targets.bool()
        output.update(tp=(positive & truth).sum((-2, -1)).cpu(), fp=(positive & ~truth).sum((-2, -1)).cpu(),
                      fn=(~positive & truth).sum((-2, -1)).cpu(), known=known.cpu())
    elif kind == "landmark":
        per_point = F.smooth_l1_loss(prediction.float(), targets.float(), reduction="none").mean(-1)
        loss = _known_mean((per_point * known).sum(), known, training)
        delta = prediction.detach().float() - targets
        source_sizes = batch["original_sizes"].to(device)[:, None, :]
        output.update(pixel_error=torch.linalg.vector_norm(delta * source_sizes, dim=-1).cpu(),
                      normalized_error=torch.linalg.vector_norm(delta, dim=-1).cpu(), known=known.cpu(),
                      predictions=prediction.detach().float().cpu(), targets=targets.cpu())
    else:
        raise ValueError(f"Unknown structural kind {kind}")
    output["loss_weight"] = int(known.sum())
    return loss, output


def _known_mean(numerator, known, training):
    """Known-target mean across DDP ranks, preserving DDP gradient averaging.

    Validation deliberately has no collectives: disjoint eval shards may have
    different numbers of batches. Training uses equal-length rank samplers.
    """
    denominator = known.sum().detach().to(dtype=torch.float32).clone()
    scale = 1
    if training and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(denominator, op=torch.distributed.ReduceOp.SUM)
        scale = torch.distributed.get_world_size()
    return numerator * scale / denominator.clamp_min(1)


def _tensor_cat(outputs, key):
    return torch.cat([torch.as_tensor(o[key]) for o in outputs], 0)


def metrics(outputs, config):
    if not outputs:
        return {"samples": 0}
    labels, kind = config["labels"], config["kind"]
    if kind == "detection":
        predictions = [p for o in outputs for p in o.get("predictions", [])]
        targets = [t for o in outputs for t in o.get("targets", [])]
        if not predictions:
            return {"samples": sum(len(o["sample_ids"]) for o in outputs)}
        return _coco_metrics(predictions, targets, config)
    known = _tensor_cat(outputs, "known").bool()
    result = {"samples": int(known.shape[0]), "per_label": {}}
    if kind == "segmentation":
        tp, fp, fn = (_tensor_cat(outputs, k).double() for k in ("tp", "fp", "fn"))
        values = []
        for i, label in enumerate(labels):
            mask = known[:, i]
            if not mask.any():
                result["per_label"][label] = {"known": 0, "dice": None, "iou": None}
                continue
            a, b, c = tp[mask, i].sum().item(), fp[mask, i].sum().item(), fn[mask, i].sum().item()
            # Empty-empty channel scores 1; state convention in output.
            dice = 2*a / (2*a+b+c) if 2*a+b+c else 1.
            iou = a / (a+b+c) if a+b+c else 1.
            values.append(dice)
            result["per_label"][label] = {"known": int(mask.sum()), "dice": dice, "iou": iou,
                                            "tp_pixels": a, "fp_pixels": b, "fn_pixels": c}
        result.update(macro_dice=float(np.mean(values)) if values else None,
                      evaluation_space="whole_image_resized_to_model_img_size", empty_empty_score=1.0)
    else:
        pixels, normalized = (_tensor_cat(outputs, k).float() for k in ("pixel_error", "normalized_error"))
        for i, label in enumerate(labels):
            mask = known[:, i]
            result["per_label"][label] = {"known": int(mask.sum()),
                "mean_pixel_error": float(pixels[mask, i].mean()) if mask.any() else None,
                "mean_normalized_error": float(normalized[mask, i].mean()) if mask.any() else None}
        result.update(mean_pixel_error=float(pixels[known].mean()) if known.any() else None,
                      mean_normalized_error=float(normalized[known].mean()) if known.any() else None,
                      coordinate_policy="source_xy_no_origin_offset_normalized_by_width_height")
    return result


def _coco_metrics(predictions, targets, config):
    """Official COCOeval AP; no TorchMetrics/SciPy dependency.

    Source ``difficult`` objects are retained as ordinary ground truth. Their
    semantics are not assumed to equal COCO crowd/ignore; this policy and their
    count are recorded in the result. Empty predictions have AP=0 when ground
    truth exists. Undefined scores for categories without truth are None.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    labels = config["labels"]
    annotations, detections = [], []
    difficult_count = 0
    for image_id, (prediction, target) in enumerate(zip(predictions, targets)):
        difficult_count += int(torch.as_tensor(target.get("difficult", [])).sum())
        for box, category in zip(target["boxes"], target["labels"]):
            xmin, ymin, xmax, ymax = [float(v) for v in box]
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": int(category), "bbox": [xmin, ymin, xmax-xmin, ymax-ymin],
                                "area": (xmax-xmin)*(ymax-ymin), "iscrowd": 0})
        for box, category, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
            xmin, ymin, xmax, ymax = [float(v) for v in box]
            detections.append({"image_id": image_id, "category_id": int(category),
                               "bbox": [xmin, ymin, xmax-xmin, ymax-ymin], "score": float(score)})
    truth = COCO()
    truth.dataset = {"info": {}, "images": [{"id": i} for i in range(len(targets))],
                     "annotations": annotations,
                     "categories": [{"id": i+1, "name": label} for i, label in enumerate(labels)]}
    # COCOeval is verbose; keep epoch logs and JSON output readable.
    with contextlib.redirect_stdout(io.StringIO()):
        truth.createIndex()
        if detections:
            predicted = truth.loadRes(detections)
        else:
            predicted = COCO()
            predicted.dataset = {**truth.dataset, "annotations": []}
            predicted.createIndex()
        evaluator = COCOeval(truth, predicted, "bbox")
        evaluator.params.imgIds = list(range(len(targets)))
        evaluator.params.catIds = list(range(1, len(labels)+1))
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    def score(value):
        return float(value) if np.isfinite(value) and value >= 0 else None
    result = {"samples": len(targets), "map": score(evaluator.stats[0]),
              "map_50": score(evaluator.stats[1]), "map_75": score(evaluator.stats[2]),
              "mar_100": score(evaluator.stats[8]), "validation_loss_available": False,
              "coordinate_mode": config["data"]["box_coordinate_mode"], "evaluation_space": "resized_image",
              "max_detections_for_ap": 100, "source_difficult_objects": difficult_count,
              "source_difficult_policy": "included_as_ordinary_groundtruth_not_assumed_crowd", "per_label": {}}
    precision = evaluator.eval["precision"]
    for i, label in enumerate(labels):
        values = precision[:, :, i, 0, -1]
        values = values[values >= 0]
        result["per_label"][label] = {"ap": score(values.mean()) if len(values) else None}
    return result
