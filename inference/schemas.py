"""Public HTTP contract. Unexecuted stages never acquire invented outputs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

STAGES = ("quality", "multidisease", "dr_grading", "disc_cup_segmentation")
CONTRACT_VERSION = "draft-20261114.1"


class ScreenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    image_url: str = Field(min_length=1, max_length=8192, description="可访问的 HTTPS JPEG、PNG 或 WEBP 图片地址")
    eye: str | None = Field(default=None, max_length=32, description="可选眼别（left/right/unknown）")
    symptoms: str | list[str] | None = Field(default=None, description="可选随图症状描述")
    history: str | list[str] | None = Field(default=None, description="可选随图既往病史")


class StageResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    status: Literal["completed", "not_run", "error"]
    reason: str | None
    model_id: str | None
    weights_sha256: str | None = Field(pattern="^[a-f0-9]{64}$")
    preprocessing_version: str | None
    elapsed_ms: float | None = Field(ge=0)
    output: dict[str, Any] | None

    @model_validator(mode="after")
    def validate_execution(self):
        if self.status == "not_run" and (self.output is not None or self.elapsed_ms is not None):
            raise ValueError("Unexecuted stage has an output or elapsed time")
        if self.status == "completed" and any(value is None for value in (
                self.model_id, self.weights_sha256, self.preprocessing_version, self.elapsed_ms, self.output)):
            raise ValueError("Completed stage lacks metadata")
        if self.status == "error" and self.output is not None:
            raise ValueError("Failed stage cannot supply a successful output")
        return self


class StageResults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quality: StageResult
    multidisease: StageResult
    dr_grading: StageResult
    disc_cup_segmentation: StageResult


class ResearchResults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    multilabel_11: StageResult
    glaucoma: StageResult


class FundusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    contract_version: Literal["draft-20261114.1", "research-20261122.1"]
    request_id: str | None
    status: Literal["ok", "partial", "quality_rejected", "tool_unavailable", "input_error", "error"]
    message: str
    image_sha256: str | None = Field(pattern="^[a-f0-9]{64}$")
    created_utc: str | None
    routing: dict[str, Any] | None
    results: StageResults
    research_results: ResearchResults | None = None
    report_summary: dict[str, Any] | None = None
    visual_report: dict[str, Any] | None = None
    warnings: list[str]

    @model_validator(mode="after")
    def validate_status(self):
        stages = [getattr(self.results, name) for name in STAGES]
        if self.contract_version == "research-20261122.1":
            if self.research_results is None or self.report_summary is None:
                raise ValueError("Six-model response is missing its research stages or report")
            research = [self.research_results.multilabel_11, self.research_results.glaucoma]
            if self.status == "ok" and any(stage.status != "completed" for stage in research):
                raise ValueError("Successful six-model response requires both research stages")
            stages.extend(research)
        elif self.research_results is not None or self.report_summary is not None:
            raise ValueError("Research results require the six-model contract")
        if self.status in {"tool_unavailable", "input_error"} and any(x.status != "not_run" for x in stages):
            raise ValueError("Unavailable or invalid input must not contain predictions")
        quality = self.results.quality
        label = (quality.output or {}).get("predicted_label") if quality.status == "completed" else None
        if self.status == "quality_rejected" and label != "Reject":
            raise ValueError("Rejected image requires a Reject quality output")
        if label not in ("Good", "Usable") and (
                self.status in {"ok", "partial"} or any(x.status != "not_run" for x in stages[1:])):
            raise ValueError("Clinical and research stages require accepted image quality")
        if self.status == "ok" and any(x.status == "error" for x in stages):
            raise ValueError("Invalid successful result")
        return self


def failure_response(status: str, message: str, request_id: str, *, reason=None, image_sha256=None):
    return {
        "contract_version": CONTRACT_VERSION, "request_id": request_id, "status": status,
        "message": message, "image_sha256": image_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat(), "routing": None,
        "results": {name: {"status": "not_run", "reason": reason or status, "model_id": None,
                           "weights_sha256": None, "preprocessing_version": None,
                           "elapsed_ms": None, "output": None} for name in STAGES},
        "warnings": ["未执行的项目不能解释为正常、疾病阴性或DR等级0。"],
    }
