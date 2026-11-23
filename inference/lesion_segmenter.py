"""Image-only research lesion masks from Clément Playout's five-class U-Net.

Model: ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d (MIT),
revision a2cfcb594e483d8baa17f24e1152009641cb9847. The model card reports
training on DDR, FGADR, IDRiD, MESSIDOR and RETLES. No annotation database is
read during inference. Source preprocessing, labels and model definition:
https://github.com/ClementPla/fundus-lesions-toolkit
https://github.com/ClementPla/fundus-data-toolkit
"""
from __future__ import annotations

import hashlib
import logging
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1] / "inference_runtime" / "lesion_localization"
_SHA256 = "a7a7cb45b92328f7c9a8e581eec3944fd435d37fae8cbf340b1319c5f987c6d2"
_CLASS_NAMES = {1: "SE", 2: "EX", 3: "HE", 4: "MA"}
_LOCK = Lock()
_LOG = logging.getLogger(__name__)
SOURCE_LABEL = "Clément Playout 四类眼底病灶分割模型（研究性候选区）"


@lru_cache(maxsize=1)
def _verified_weights():
    path = _ROOT / "model.safetensors"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != _SHA256:
        raise ValueError("Lesion model fingerprint mismatch")
    return path


@lru_cache(maxsize=2)
def _model(device):
    path = _verified_weights()
    # The optional wheels live beside these weights. Existing packages keep
    # precedence, in particular the six core models' pinned torch/timm stack.
    vendor = str(_ROOT / "vendor")
    if vendor not in sys.path:
        sys.path.append(vendor)
    import torchseg
    from safetensors.torch import load_file

    network = torchseg.create_model("unet", encoder_name="seresnext50_32x4d",
                                    encoder_weights=None, in_channels=3, classes=5)
    state = load_file(str(path))
    network.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=True)
    return network.eval().requires_grad_(False).to(device)


def _prepare(image):
    import cv2
    import torch
    from torchvision.transforms.functional import normalize

    pixels = np.asarray(image.convert("RGB"))
    brightest = pixels.max(axis=2)
    visible = brightest > 0.05 * int(brightest.max())
    ys = np.flatnonzero(visible.any(axis=1))
    xs = np.flatnonzero(visible.any(axis=0))
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("Insufficient fundus field")
    # Upper bound is exclusive in the author's autocrop implementation.
    left, top, right, bottom = int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])
    patch = pixels[top:bottom, left:right]
    scale = 1024 / max(patch.shape[:2])
    if min(patch.shape[:2]) * scale < 1:
        raise ValueError("Degenerate fundus field")
    resized = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    h, w = resized.shape[:2]
    y, x = (1024 - h) // 2, (1024 - w) // 2
    square = np.zeros((1024, 1024, 3), dtype=np.uint8)
    square[y:y + h, x:x + w] = resized
    image_tensor = torch.from_numpy(square).permute(2, 0, 1).float().div_(255)
    normalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
    return image_tensor[None], (left, top, right, bottom), (x, y, w, h)


def _restore(probabilities, image_size, source_box, tensor_box):
    import torch
    import torch.nn.functional as F

    left, top, right, bottom = source_box
    x, y, w, h = tensor_box
    # Resize class probabilities before argmax, as in the author's pipeline.
    # Use the exact source extent so subpixel rounding cannot shift annotations.
    area = probabilities[:, y:y + h, x:x + w]
    height, width = bottom - top, right - left
    best = torch.full((height, width), -1.0)
    classes = torch.zeros((height, width), dtype=torch.uint8)
    for index, channel in enumerate(area):
        restored = F.interpolate(channel[None, None], size=(height, width),
                                 mode="bilinear", align_corners=True)[0, 0]
        better = restored > best
        classes[better] = index
        torch.maximum(best, restored, out=best)
    labels = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    labels[top:bottom, left:right] = classes.numpy()
    return {code: Image.fromarray((labels == index).astype(np.uint8) * 255)
            for index, code in _CLASS_NAMES.items()}


def _predict(image, device):
    import torch

    network = _model(device)
    tensor, source_box, tensor_box = _prepare(image)
    with torch.inference_mode():
        logits = network(tensor.to(device))
        probabilities = logits.softmax(dim=1)[0].cpu()
        return _restore(probabilities, image.size, source_box, tensor_box)


def get_lesion_masks(image: Image.Image) -> dict | None:
    """Return original-resolution PIL binary masks, including empty classes."""
    if not (_ROOT / "model.safetensors").is_file():
        return None
    try:
        import torch
        with _LOCK:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                masks = _predict(image, device)
            except torch.cuda.OutOfMemoryError:
                masks = _predict(image, "cpu")
        return {"masks": masks, "source_label": SOURCE_LABEL}
    except Exception as exc:
        _LOG.warning("lesion_localization_unavailable reason=%s", type(exc).__name__)
        return None
