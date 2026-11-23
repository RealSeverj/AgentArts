"""Validated checkpoint loaders and score-only inference for the two research models."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import logging
from pathlib import Path
import time

from PIL import Image
import torch

from scripts.predict_model5_ensemble_original import Ensemble, FINAL, LABELS, optic, pinned
from training.classification import ClassificationTransform
from training import screening_training as glaucoma_training


logger = logging.getLogger(__name__)
GLAUCOMA_FINAL = "1cf18c04f4581536da035f1667f2b2f616b22c2c3b2e2c2ced403da10f5fe3c9"


@contextmanager
def _replay_precision():
    """Keep the frozen locator's FP32 settings local to serialized research work."""
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


class ResearchExperts:
    """Load once under the engine's lock and reuse both models on accepted images."""

    def __init__(self, project_root: Path, device):
        self.root = Path(project_root)
        self.device = torch.device(device)
        self.ensemble = None
        self.glaucoma = None
        self.glaucoma_transform = None
        self._info = {}

    def load(self):
        if self._info:
            return self
        archive = self.root / "02_模型训练与评估"
        model5 = archive / "07_模型5云端优化结果_20261122"
        locator = archive / "01_在用模型/04_视盘视杯分割/disc_refuge_best.pt"
        with _replay_precision():
            ensemble = Ensemble(model5, locator, self.device)

        selection = pinned(
            archive / "08_其余五模型优化_20261122/model6_glaucoma/vitl392/final_selection.json",
            GLAUCOMA_FINAL,
        )
        if selection["selected"] != "historical":
            raise ValueError("Unexpected glaucoma selection")
        weight_hash = selection["validation"]["historical"]["checkpoint_sha256"]
        historical = archive / "05_本轮训练结果_20261122/glaucoma_partial"
        weights = historical / "best.pt"
        pretrained = self.root / "cloud_upload/pretrained/dinov2_vitb14_pretrain.pth"
        if (glaucoma_training.sha256(weights) != weight_hash or
                glaucoma_training.sha256(pretrained) != glaucoma_training.OFFICIAL_SHA):
            raise ValueError("Glaucoma checkpoint fingerprint differs from the selected model")
        glaucoma = glaucoma_training.ScreeningModel("glaucoma", pretrained)
        state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        if state["identity"]["task"] != "glaucoma":
            raise ValueError("Unexpected task in selected glaucoma checkpoint")
        glaucoma.load_state_dict(state["model"], strict=True)
        glaucoma = glaucoma.eval().requires_grad_(False).to(self.device)
        transform = ClassificationTransform(glaucoma_training.TRANSFORM_CONFIG, training=False)

        member_hashes = {member["role"]: member["checkpoint_sha256"]
                         for member in ensemble.recipe["members"]}
        member_hashes["disc_locator"] = optic.LOCATOR
        self._info = {
            "multilabel_11": {
                "model_id": "rfmid11_vitl_vitb_disc_mean_20261122",
                "weights_sha256": ensemble.recipe_sha,
                "member_sha256": member_hashes,
                "selection_sha256": FINAL,
                "preprocessing_version": "rgb392_global_and_disc_" + optic.POLICY_SHA[:16],
                "labels": list(LABELS),
                "research_only": True,
            },
            "glaucoma": {
                "model_id": "glaucoma_historical_vitb392_selected_20261122",
                "weights_sha256": weight_hash,
                "member_sha256": {"historical_vitb": weight_hash},
                "selection_sha256": GLAUCOMA_FINAL,
                "preprocessing_version": "rgb392_stretch_imagenet_single_view",
                "labels": list(glaucoma_training.LABELS["glaucoma"]),
                "research_only": True,
            },
        }
        self.ensemble = ensemble
        self.glaucoma = glaucoma
        self.glaucoma_transform = transform
        return self

    def model_info(self):
        return deepcopy(self._info)

    @torch.inference_mode()
    def execute(self, stage: str, image: Image.Image):
        info = self._info[stage]
        started = time.monotonic()
        result = {key: info[key] for key in ("model_id", "weights_sha256", "preprocessing_version")}
        result.update(status="error", reason="research_inference_failed", output=None)
        try:
            rgb = image if image.mode == "RGB" else image.convert("RGB")
            view_details = {}
            if stage == "multilabel_11":
                with _replay_precision():
                    replay = self.ensemble.score_image(rgb)
                scores = dict(zip(info["labels"], replay["probabilities"], strict=True))
                # The two-view branch substitutes the whole frame when localization fails.
                # Only return what actually ran, without exposing crop coordinates.
                view_details = {
                    "disc_roi_used": replay["bbox_xyxy"] is not None,
                    "disc_roi_fallback_reason": replay["locator_fallback"],
                }
            else:
                tensor = self.glaucoma_transform(rgb).unsqueeze(0).to(self.device)
                with _replay_precision():
                    logit = self.glaucoma(tensor)
                if logit.shape != (1, 1) or not torch.isfinite(logit).all().item():
                    raise ValueError("Invalid glaucoma classifier output")
                scores = {"glaucoma": float(logit.float().sigmoid().item())}
            output = {
                "label_order": list(info["labels"]), "scores": scores, "research_only": True,
                "calibration_state": "uncalibrated_research_model_scores",
                "clinical_decision": None, "decision_thresholds": None, "binary_decisions": None,
                "severity_grade": None,
                **view_details,
            }
            json.dumps(output, allow_nan=False)
            result.update(status="completed", reason=None, output=output)
        except Exception:
            logger.exception("Research inference failed for %s", stage)
        result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
        return result
