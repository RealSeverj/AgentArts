"""Serialized retinal inference with optional separately reported research models.

The HTTP boundary owns fetching, size limits and the single RGB decode. This
module only accepts that already-decoded image. It never trains, tunes a
threshold, retrieves an image URL or manufactures a successful prediction.
"""
from __future__ import annotations

import base64
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import re
import threading
import time

from PIL import Image
import torch

from inference.registry import DISEASE_LABELS, MASK_LABELS, QUALITY_LABELS, STAGES, require
from training.classification import ClassificationTransform
from training.fundus_preprocessing import preprocessing_spec
from training.structural import _tensor as structural_tensor


CONTRACT_VERSION = "draft-20261114.1"
CALIBRATION_STATE = "not_validated_for_production"


def empty_stage(reason="not_requested"):
    return {"status": "not_run", "reason": reason, "model_id": None, "weights_sha256": None,
            "preprocessing_version": None, "elapsed_ms": None, "output": None}


def empty_response(*, status="tool_unavailable", message="当前推理服务尚未连接，未完成图像筛查。",
                   request_id=None, image_sha256=None):
    return {"contract_version": CONTRACT_VERSION, "request_id": request_id, "status": status,
            "message": message, "image_sha256": image_sha256,
            "created_utc": datetime.now(timezone.utc).isoformat(), "routing": None,
            "results": {stage: empty_stage(status) for stage in STAGES},
            "warnings": ["未执行的项目不能解释为正常、疾病阴性或DR等级0。"]}


def validate_routing(value):
    if value is None:
        value = {"mode": "all", "policy_id": "development-all-v1", "validation_passed": False}
    require(isinstance(value, dict), "routing must be an object")
    require(not (set(value) - {"mode", "policy_id", "validation_passed", "validation_reference", "thresholds", "allow_unvalidated"}), "Unknown routing fields")
    result = deepcopy(value)
    require(result.get("mode") in ("all", "threshold"), "Routing mode must be all or threshold")
    require(isinstance(result.get("policy_id"), str) and bool(result["policy_id"].strip()), "Routing policy ID is required")
    result.setdefault("validation_passed", False)
    result.setdefault("validation_reference", None)
    result.setdefault("allow_unvalidated", False)
    require(type(result["validation_passed"]) is bool and type(result["allow_unvalidated"]) is bool, "Routing validation flags must be boolean")
    if result["validation_passed"]:
        require(isinstance(result["validation_reference"], str) and bool(result["validation_reference"].strip()), "Validated policy requires its validation reference")
    if result["mode"] == "threshold":
        thresholds = result.get("thresholds")
        require(isinstance(thresholds, dict) and set(thresholds) == {"dr_grading", "disc_cup_segmentation"}, "Explicit thresholds for both routed experts are required")
        for threshold in thresholds.values():
            require(type(threshold) in (int, float) and math.isfinite(threshold) and 0 <= threshold <= 1, "Routing thresholds must be finite in [0,1]")
        require(result["validation_passed"] or result["allow_unvalidated"], "Unvalidated threshold routing requires explicit allow_unvalidated=true")
    else:
        require(result.get("thresholds") is None, "all routing does not use score thresholds")
        result["thresholds"] = None
    return result


def _preprocessing_version(config):
    contract = {"preprocessing": preprocessing_spec(config), "image_size": config["model"]["img_size"],
                "mean": config["model"]["mean"], "std": config["model"]["std"],
                "resize": "PIL_BILINEAR", "augmentation": False, "exif_transpose": False}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return contract["preprocessing"]["mode"] + ":" + digest


class FundusEngine:
    def __init__(self, registry, *, device="auto", precision="auto", routing=None, research=None):
        self.registry = registry
        self.research = research
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        require(self.device.type in ("cpu", "cuda"), "Inference supports cpu or cuda")
        require(self.device.type != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
        if precision == "auto":
            precision = "fp16" if self.device.type == "cuda" else "fp32"
        require(precision in ("fp32", "fp16"), "Inference precision must be fp32 or fp16")
        require(self.device.type == "cuda" or precision == "fp32", "CPU inference requires fp32")
        self.precision = precision
        self.routing = validate_routing(routing)
        self._lock = threading.Lock()
        self._experts = {}
        self._ready = False
        self.last_load_error = None

    @property
    def is_ready(self):
        return self._ready

    def load(self):
        with self._lock:
            if self._ready:
                return self
            self.last_load_error = None
            try:
                experts = self.registry.load_models()
                require(set(experts) == set(STAGES), "All four experts must load before readiness")
                for expert in experts.values():
                    expert.model.eval().requires_grad_(False).to(self.device)
                if self.research is not None:
                    self.research.load()
                self._experts = experts
                self._ready = True
            except Exception as error:
                self._experts = {}
                self._ready = False
                self.last_load_error = type(error).__name__
                raise
        return self

    def model_info(self):
        with self._lock:
            if not self._ready:
                return {}
            models = {key: {"model_id": expert.model_id, "weights_sha256": expert.sha256,
                          "completed_epoch": expert.completed_epoch, "task": expert.config["task"],
                          "training_sources": expert.config["data"]["datasets"],
                          "input_size": expert.config["model"]["img_size"],
                          "labels": expert.config["labels"],
                          "preprocessing_version": _preprocessing_version(expert.config)}
                    for key, expert in self._experts.items()}
            if self.research is not None:
                models.update(self.research.model_info())
            return models

    def _logits(self, expert, image):
        config = expert.config
        if expert.stage == "disc_cup_segmentation":
            size = config["model"]["img_size"]
            tensor = structural_tensor(image.resize((size, size), Image.Resampling.BILINEAR), config)
        else:
            tensor = ClassificationTransform(config, training=False, input_preprocessed=False)(image)
        autocast = torch.autocast("cuda", dtype=torch.float16) if self.precision == "fp16" else nullcontext()
        with torch.inference_mode(), autocast:
            logits = expert.model(tensor.unsqueeze(0).to(self.device))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        require(torch.is_tensor(logits) and bool(torch.isfinite(logits).all()), "Non-finite model logits")
        return logits.float().detach().cpu()

    def _output(self, stage, logits, image):
        if stage in ("quality", "dr_grading"):
            labels = QUALITY_LABELS if stage == "quality" else tuple(range(5))
            require(tuple(logits.shape) == (1, len(labels)), "Classifier output shape differs")
            scores = torch.softmax(logits, dim=-1)[0].tolist()
            prediction = max(range(len(scores)), key=scores.__getitem__)
            if stage == "quality":
                return {"class_order": list(labels), "scores": dict(zip(labels, scores)),
                        "predicted_label": labels[prediction], "calibration_state": CALIBRATION_STATE}
            return {"grade_order": list(labels), "scores": scores, "predicted_grade": prediction,
                    "calibration_state": CALIBRATION_STATE}
        if stage == "multidisease":
            require(tuple(logits.shape) == (1, len(DISEASE_LABELS)), "Multidisease output shape differs")
            return {"label_order": list(DISEASE_LABELS), "scores": dict(zip(DISEASE_LABELS, logits.sigmoid()[0].tolist())),
                    "decision_thresholds": None, "binary_decisions": None, "calibration_state": CALIBRATION_STATE,
                    "label_semantics": {"OPTIC_DISC_CUPPING": "视盘凹陷表型，不等于青光眼诊断。",
                                        "ABNORMAL": "独立总体异常标签，不由其他标签合并计算。"}}
        require(tuple(logits.shape) == (1, 2, 512, 512), "Segmentation output shape differs")
        masks = logits.sigmoid()[0] >= 0.5
        artifacts = {}
        for index, label in enumerate(MASK_LABELS):
            buffer = io.BytesIO()
            Image.fromarray(masks[index].to(torch.uint8).numpy() * 255).save(buffer, format="PNG")
            payload = buffer.getvalue()
            artifacts[label] = {"encoding": "base64", "media_type": "image/png", "width": 512, "height": 512,
                                "foreground_value": 255, "background_value": 0,
                                "sha256": hashlib.sha256(payload).hexdigest(),
                                "data": base64.b64encode(payload).decode("ascii")}
        outside = int((masks[0] & ~masks[1]).sum())
        return {"channel_order": list(MASK_LABELS), "mask_artifacts": artifacts, "mask_threshold": 0.5,
                "coordinate_space": "whole_image_resized_to_model_img_size", "size_hw": [512, 512],
                "source_size_wh": list(image.size),
                "mapping_to_source": {"kind": "stretch", "scale_x": image.width / 512, "scale_y": image.height / 512,
                                      "source_pixel_center_formula": "(model_index + 0.5) * scale - 0.5",
                                      "native_mask_resize": "nearest", "native_mask_artifact_supplied": False},
                "cup_outside_disc_pixels": outside, "containment_correction_applied": False,
                "derived_measurements": None}

    def _execute(self, stage, image):
        expert = self._experts[stage]
        started = time.monotonic()
        result = {"status": "error", "reason": None, "model_id": expert.model_id,
                  "weights_sha256": expert.sha256, "preprocessing_version": _preprocessing_version(expert.config),
                  "elapsed_ms": None, "output": None}
        try:
            result["output"] = self._output(stage, self._logits(expert, image), image)
            # Prevent accidental non-standard JSON floats anywhere in a completed result.
            json.dumps(result["output"], allow_nan=False)
            result["status"] = "completed"
        except Exception:
            # No raw exception/path/URL or arbitrary model error text goes to clients.
            result["output"] = None
            result["reason"] = "inference_failed"
        result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
        return result

    def screen(self, image, *, request_id, image_sha256):
        require(isinstance(image, Image.Image) and image.mode == "RGB" and min(image.size) > 0,
                "screen requires an already decoded RGB PIL image")
        require(isinstance(request_id, str) and bool(request_id), "A request ID is required")
        require(isinstance(image_sha256, str) and re.fullmatch(r"[a-f0-9]{64}", image_sha256) is not None,
                "A lowercase image SHA-256 is required")
        with self._lock:
            if self._ready and self.research is not None:
                from inference.image_io import validate_screening_pixels
                validate_screening_pixels(image)
            response = self._screen_core(image, request_id=request_id, image_sha256=image_sha256)
            if self.research is None or not self._ready:
                return response
            from inference.report_summary import build_report_summary
            results = response["results"]
            quality = results["quality"]
            accepted = quality["status"] == "completed" and quality["output"]["predicted_label"] in ("Good", "Usable")
            research_results = {}
            for stage in ("multilabel_11", "glaucoma"):
                if accepted:
                    research_results[stage] = self.research.execute(stage, image)
                else:
                    reason = "quality_rejected" if response["status"] == "quality_rejected" else "quality_failed"
                    research_results[stage] = empty_stage(reason)
            response["contract_version"] = "research-20261122.1"
            response["research_results"] = research_results
            response["report_summary"] = build_report_summary(results, research_results)
            if response["status"] == "ok" and any(r["status"] != "completed" for r in research_results.values()):
                response.update(status="partial", message="部分研究项目未完成，已完成项目的结果仍可查看。")
            return response

    def _screen_core(self, image, *, request_id, image_sha256):
        if not self._ready:
            return empty_response(request_id=request_id, image_sha256=image_sha256)
        response = empty_response(status="ok", message="图像分析完成；结果供人工复核。",
                                  request_id=request_id, image_sha256=image_sha256)
        response["routing"] = {**deepcopy(self.routing), "stage_reasons": {}}
        if not self.routing["validation_passed"]:
            response["warnings"].append("当前路由策略尚未经独立验证；原始模型分数未经概率校准。")
        results = response["results"]
        results["quality"] = self._execute("quality", image)
        response["routing"]["stage_reasons"]["quality"] = "required"
        if results["quality"]["status"] != "completed":
            for stage in STAGES[1:]:
                results[stage] = empty_stage("quality_failed")
                response["routing"]["stage_reasons"][stage] = "quality_failed"
            response.update(status="error", message="图像质控执行失败，未完成下游筛查。")
            return response
        quality = results["quality"]["output"]["predicted_label"]
        if quality == "Reject":
            for stage in STAGES[1:]:
                results[stage] = empty_stage("quality_rejected")
                response["routing"]["stage_reasons"][stage] = "quality_rejected"
            response.update(status="quality_rejected", message="图像质量不足，请重拍或人工复核；未执行下游筛查。")
            return response
        if quality == "Usable":
            response["warnings"].append("质控为Usable，图像质量有限，不能等同Good。")
        results["multidisease"] = self._execute("multidisease", image)
        response["routing"]["stage_reasons"]["multidisease"] = "quality_accepted"
        requires_gate = self.routing["mode"] == "threshold" or self.research is None
        if requires_gate and results["multidisease"]["status"] != "completed":
            for stage in STAGES[2:]:
                results[stage] = empty_stage("multidisease_failed")
                response["routing"]["stage_reasons"][stage] = "multidisease_failed"
            response.update(status="partial", message="已完成质控，多病种分析失败，未执行分级和分割。")
            return response
        for stage, score_key in (("dr_grading", "DR"), ("disc_cup_segmentation", "OPTIC_DISC_CUPPING")):
            reason = "all_policy"
            if self.routing["mode"] == "threshold":
                score = results["multidisease"]["output"]["scores"][score_key]
                triggered = score >= self.routing["thresholds"][stage]
                reason = "score_at_or_above_gate" if triggered else "score_below_gate"
                if not triggered:
                    results[stage] = empty_stage(reason)
                    response["routing"]["stage_reasons"][stage] = reason
                    continue
            response["routing"]["stage_reasons"][stage] = reason
            results[stage] = self._execute(stage, image)
        if any(result["status"] == "error" for result in results.values()):
            response.update(status="partial", message="部分分析执行失败，请查看各项目状态并人工复核。")
        if results["disc_cup_segmentation"]["status"] == "completed" and results["disc_cup_segmentation"]["output"]["cup_outside_disc_pixels"]:
            response["warnings"].append("预测视杯存在超出视盘的像素，原始结果保留，需复核。")
        return response
