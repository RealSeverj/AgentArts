"""SQLite-backed clinical evidence store, deterministic symptom/follow-up rules, and eye comparisons."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator

from inference.report_summary import ELEVEN_LABELS, FIVE_LABELS, LABELS

URGENT_SYMPTOM_RULES = (
    ("突发大量飞蚊或黑影飘动", ("飞蚊", "黑影", "飘动", "絮状物")),
    ("眼部闪光感", ("闪光", "闪电感")),
    ("视野幕帘样遮挡或视野缺损", ("幕帘", "黑幕", "遮挡", "视野缺损", "视野变暗", "看东西缺一块")),
    ("突发视力骤降或失明", ("突发视力", "视力骤降", "突然看不清", "突然失明", "视力突然下降")),
    ("剧烈眼痛伴头痛恶心", ("剧烈眼痛", "眼痛伴头痛", "眼胀头痛", "恶心呕吐")),
)

NEGATION_PREFIXES = ("无", "没有", "未见", "未出现", "否认", "非", "不伴")

LABEL_SOURCES = {
    "DR": "SRC_DR_GUIDELINE_2024",
    "AMD": "SRC_DICT_DN_AMD",
    "DN": "SRC_DICT_DN_AMD",
    "CSR": "SRC_DICT_MACULA",
    "RVO": "SRC_RVO_ODE_CONSENSUS",
    "ODE": "SRC_RVO_ODE_CONSENSUS",
    "ODP": "SRC_DICT_DISC",
    "OPTIC_DISC_CUPPING": "SRC_DICT_ODC_GLAUCOMA",
    "glaucoma": "SRC_GLAUCOMA_SCREEN_2024",
    "LS": "SRC_DICT_LS",
    "MYOPIC_FUNDUS": "SRC_DICT_MYOPIA",
    "TSLN": "SRC_DICT_MYOPIA",
    "ABNORMAL": "SRC_SCREEN_SCOPE",
}


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[，,。;；\n]+", value) if part.strip()]
        return parts[:20]
    return value


class IntakeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assessment_id: str | None = Field(default=None, min_length=1, max_length=128)
    session_id: str = Field(default="default", min_length=1, max_length=128)
    eye: Literal["left", "right", "unknown"] | None = None
    symptoms: list[str] = Field(default_factory=list, max_length=20)
    symptom_onset: str | None = Field(default=None, max_length=256)
    history: list[str] = Field(default_factory=list, max_length=20)
    confirmed_plan: str | None = Field(default=None, max_length=512)

    @field_validator("assessment_id", "eye", "symptom_onset", "confirmed_plan", mode="before")
    @classmethod
    def normalize_optional_strings(cls, value: Any) -> Any:
        return _coerce_optional_str(value)

    @field_validator("session_id", mode="before")
    @classmethod
    def normalize_session_id(cls, value: Any) -> Any:
        cleaned = _coerce_optional_str(value)
        return "default" if cleaned is None else cleaned

    @field_validator("symptoms", "history", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: Any) -> Any:
        return _coerce_str_list(value)


class CareEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assessment_id: str | None = Field(default=None, min_length=1, max_length=128)
    session_id: str = Field(default="default", min_length=1, max_length=128)

    @field_validator("assessment_id", mode="before")
    @classmethod
    def normalize_assessment_id(cls, value: Any) -> Any:
        return _coerce_optional_str(value)

    @field_validator("session_id", mode="before")
    @classmethod
    def normalize_session_id(cls, value: Any) -> Any:
        cleaned = _coerce_optional_str(value)
        return "default" if cleaned is None else cleaned


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    previous_assessment_id: str = Field(min_length=1, max_length=128)
    current_assessment_id: str = Field(min_length=1, max_length=128)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text_list(items: list[str]) -> list[str]:
    cleaned: list[str] = []
    for raw in items:
        text = " ".join(raw.split())
        if text and text not in cleaned:
            cleaned.append(text[:256])
    return cleaned


def _detect_urgent_symptoms(symptoms: list[str]) -> list[str]:
    matched: list[str] = []
    for entry in symptoms:
        clauses = [c.strip() for c in re.split(r"[，,。;；、\n]+", entry) if c.strip()]
        for clause in clauses:
            if any(clause.startswith(prefix) for prefix in NEGATION_PREFIXES):
                continue
            for label, keywords in URGENT_SYMPTOM_RULES:
                if any(kw in clause for kw in keywords):
                    desc = f"{clause}（命中警示：{label}）"
                    if desc not in matched:
                        matched.append(desc)
    return matched


def _extract_predictions(screening: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str], int | None]:
    summary = screening.get("report_summary") or {}
    dr_sev = summary.get("dr_severity") or {}
    dr_grade = dr_sev.get("grade") if dr_sev.get("status") == "completed" else None
    if type(dr_grade) is not int or not 0 <= dr_grade <= 4:
        dr_grade = None

    results = screening.get("results") or {}
    research = screening.get("research_results") or {}
    predictions: list[dict[str, Any]] = []
    reached_codes: set[str] = set()

    groups = (
        ("multidisease", results.get("multidisease") or {}, summary.get("five_label_screening") or {}, FIVE_LABELS),
        ("multilabel_11", research.get("multilabel_11") or {}, (summary.get("research") or {}).get("multilabel_11") or {}, ELEVEN_LABELS),
        ("glaucoma", research.get("glaucoma") or {}, (summary.get("research") or {}).get("glaucoma") or {}, ("glaucoma",)),
    )

    for source_model, stage, report_group, expected_codes in groups:
        item_map = {item.get("code"): item for item in (report_group.get("items") or []) if item.get("code")}
        scores = ((stage.get("output") or {}).get("scores") or {}) if stage.get("status") == "completed" else {}
        for code in expected_codes:
            item = item_map.get(code) or {}
            name, kind = LABELS.get(code, (code, "未知"))
            raw_score = item.get("score") if "score" in item else scores.get(code)
            threshold = item.get("research_threshold")
            met = item.get("research_threshold_met")
            if met is True:
                flag = "positive"
                reached_codes.add(code)
            elif met is False:
                flag = "negative"
            else:
                flag = "undetermined"
            predictions.append({
                "source_model": source_model,
                "model_id": stage.get("model_id"),
                "weights_sha256": stage.get("weights_sha256"),
                "label_code": code,
                "label_name": name,
                "concept_type": kind,
                "raw_score": raw_score,
                "calibrated_score": None,
                "threshold": threshold,
                "threshold_applies_to": "raw_score",
                "screening_flag": flag,
            })

    return predictions, reached_codes, dr_grade


def evaluate_care_rules(screening: dict[str, Any], intake: dict[str, Any]) -> dict[str, Any]:
    """Deterministically compute care priority, missing facts, and follow-up actions."""
    symptoms = list(intake.get("symptoms") or [])
    history = list(intake.get("history") or [])
    eye = intake.get("eye") or "unknown"
    confirmed_plan = intake.get("confirmed_plan")

    urgent_hits = _detect_urgent_symptoms(symptoms)
    predictions, reached_codes, dr_grade = _extract_predictions(screening)
    quality_stage = (screening.get("results") or {}).get("quality") or {}
    quality_label = (quality_stage.get("output") or {}).get("predicted_label")
    quality_ok = (
        screening.get("status") in ("ok", "partial")
        and quality_stage.get("status") == "completed"
        and quality_label in ("Good", "Usable")
    )

    missing_facts: list[str] = []
    candidate_questions: list[str] = []
    if eye == "unknown":
        missing_facts.append("eye")
        candidate_questions.append("请问本次拍摄的眼底照片是左眼还是右眼？")
    if not symptoms:
        missing_facts.append("symptoms")
        candidate_questions.append("请问近期是否有视物模糊、飞蚊闪光、视野遮挡或突发视力变化？若均无不适也可以直接说明。")
    if "LS" in reached_codes and not any(k in " ".join(history) for k in ("激光", "光凝")):
        missing_facts.append("laser_history")
        candidate_questions.append("筛查提示存在疑似激光瘢痕特征，请问既往是否接受过眼底激光光凝治疗？")
    if ((dr_grade is not None and dr_grade >= 1) or "DR" in reached_codes) and not any(
        k in " ".join(history) for k in ("糖尿", "血糖", "糖化")
    ):
        missing_facts.append("diabetes_history")
        candidate_questions.append("请问是否有糖尿病史，以及近期血糖或糖化血红蛋白控制情况如何？")

    questions_to_confirm = candidate_questions[:2]

    actions: list[str] = []
    monitoring_items: list[str] = []
    questions_for_clinician: list[str] = []
    source_ids: list[str] = []

    def _add_sources(*codes: str):
        for c in codes:
            if c and c not in source_ids:
                source_ids.append(c)

    if urgent_hits:
        care_priority = "urgent_ophthalmology"
        priority_label = "立即眼科急诊评估"
        followup_time = "立即（当日尽快前往眼科急诊或眼底专科就诊）"
        rule_ids = ["RULE_URGENT_SYMPTOM_01"]
        _add_sources("SRC_NEI_RD_URGENT")
        priority_basis = (
            "已触发紧急症状优先规则：" + "；".join(urgent_hits) + "。"
            "无论当前眼底照片是否检出异常或是否完成分析，警示症状均需优先急诊排查视网膜裂孔、视网膜脱离或急性眼底出血。"
        )
        actions = [
            "请立即前往具备眼底检查条件的眼科急诊，避免剧烈运动与揉眼。",
            "就诊时携带本次眼底原图及完整症状起始时间，告知医生上述警示症状。",
        ]
        monitoring_items = ["视野遮挡范围是否扩大", "飞蚊或闪光感是否持续加重", "单眼视力是否进一步下降"]
        questions_for_clinician = [
            "是否需要散瞳详细检查周边视网膜及眼底 B 超以排除视网膜裂孔或脱离？",
            "目前症状是否需要急诊干预或限制日常活动？",
        ]
    elif not quality_ok:
        care_priority = "retake_image"
        priority_label = "图像质量不足，建议重拍"
        followup_time = "重新采集清晰眼底照片后再次分析；若有不适随时就诊"
        rule_ids = ["RULE_QC_RETAKE_01"]
        _add_sources("SRC_QC_RETAKE")
        priority_basis = (
            f"当前图像状态为 {screening.get('status')}（质控判定：{quality_label or '未完成'}），"
            "未达到可靠判读条件，未对下游疾病做阴性或正常结论。"
        )
        actions = [
            "请在暗室或充分对焦条件下重新拍摄眼底彩色照，避免反光、严重模糊或视野遮挡。",
            "如重拍仍受限或伴有视物不适，请直接由眼科医生裂隙灯或散瞳检查确认。",
        ]
        monitoring_items = ["是否新发视力下降、黑影飘动或视物变形"]
        questions_for_clinician = ["是否需要散瞳拍摄眼底照或行 OCT 检查评估眼底情况？"]
    elif (dr_grade is not None and dr_grade >= 3) or bool({"RVO", "ODE"} & reached_codes):
        care_priority = "prompt_ophthalmology"
        priority_label = "尽快眼科眼底专科就诊"
        followup_time = "建议 1~2 周内眼科眼底专科就诊（若有既有医嘱则以医嘱为准）"
        rule_ids = ["RULE_DR_SEVERE_OR_VASC_01"]
        _add_sources("SRC_DR_GUIDELINE_2024", "SRC_RVO_ODE_CONSENSUS")
        triggers = []
        if dr_grade is not None and dr_grade >= 3:
            triggers.append(f"糖网专病模型预测为 {dr_grade} 级（{'重度非增殖期' if dr_grade == 3 else '增殖期'}）")
        if "RVO" in reached_codes:
            triggers.append("十一标签模型提示视网膜静脉阻塞（RVO）达到研究筛查线")
        if "ODE" in reached_codes:
            triggers.append("十一标签模型提示视盘水肿（ODE）达到研究筛查线")
        priority_basis = "；".join(triggers) + "。高风险或血管/视盘急性体征线索需由专科医生尽快面诊复核。"
        actions = [
            "预约 1~2 周内眼科眼底专科门诊，核对原图与病灶候选区。",
            "就诊时建议携带既往血糖记录、血压记录及既往眼底检查/光凝病历。",
        ]
        monitoring_items = ["单眼视力是否突然变模糊", "视物是否出现直线弯曲变形", "是否伴随眼胀、头痛或突发黑影"]
        questions_for_clinician = [
            "是否需要进一步行黄斑 OCT 或眼底荧光素血管造影（FFA）以明确病变范围？",
            "目前是否需要激光光凝、抗 VEGF 或其他专科干预，以及后续复查间隔？",
        ]
    elif dr_grade == 2 or bool({"AMD", "CSR", "glaucoma", "ODP", "OPTIC_DISC_CUPPING"} & reached_codes):
        care_priority = "scheduled_ophthalmology"
        priority_label = "限期眼科专科复核"
        followup_time = "建议 1 个月内前往眼科门诊复核"
        rule_ids = ["RULE_MODERATE_OR_SPECIALIST_02"]
        triggers = []
        if dr_grade == 2:
            triggers.append("糖网专病模型预测为 2 级（中度非增殖期）")
            _add_sources("SRC_DR_GUIDELINE_2024")
        if {"AMD", "CSR"} & reached_codes:
            triggers.append("黄斑区相关筛查项（" + "、".join(sorted({"AMD", "CSR"} & reached_codes)) + "）达到研究筛查线")
            _add_sources("SRC_DICT_DN_AMD", "SRC_DICT_MACULA")
        if {"glaucoma", "ODP", "OPTIC_DISC_CUPPING"} & reached_codes:
            triggers.append("视盘与青光眼线索（" + "、".join(sorted({"glaucoma", "ODP", "OPTIC_DISC_CUPPING"} & reached_codes)) + "）达到研究筛查线")
            _add_sources("SRC_DICT_ODC_GLAUCOMA", "SRC_GLAUCOMA_SCREEN_2024")
        priority_basis = "；".join(triggers) + "。需由眼科医生结合临床体征及专科检查进行明确诊断。"
        actions = ["建议 1 个月内至眼科门诊就诊，由医生结合原图复核相应眼底区域。"]
        if {"glaucoma", "ODP", "OPTIC_DISC_CUPPING"} & reached_codes:
            actions.append("若提示视盘凹陷/苍白或青光眼风险线索，建议完善眼压测量、视野检查及视神经纤维层 OCT。")
        if {"AMD", "CSR"} & reached_codes:
            actions.append("若提示黄斑异常线索，建议完善黄斑 OCT 排查积液或结构改变。")
        monitoring_items = ["中心视力是否下降或看字变模糊", "单眼交替遮盖看门框直线是否扭曲", "视野周边是否出现缺损"]
        questions_for_clinician = [
            "结合本次眼底照片，是否需要补充黄斑 OCT、眼压或视野检查？",
            "下一步建议多久复查一次？",
        ]
    elif dr_grade == 1 or bool({"DN", "MYOPIC_FUNDUS", "TSLN", "LS", "ABNORMAL", "DR"} & reached_codes):
        care_priority = "routine_followup"
        priority_label = "常规眼科随访与病史核对"
        followup_time = "建议 3~6 个月常规眼科复查（若存在明确临床诊断请遵医嘱）"
        rule_ids = ["RULE_MILD_OR_PHENOTYPE_03"]
        _add_sources("SRC_DR_GUIDELINE_2024", "SRC_DICT_LS", "SRC_DICT_MYOPIA")
        triggers = []
        if dr_grade == 1 or "DR" in reached_codes:
            triggers.append("存在轻度糖网分级或 DR 筛查提示")
        if "LS" in reached_codes:
            triggers.append("提示激光瘢痕（LS）治疗痕迹特征")
        if {"DN", "MYOPIC_FUNDUS", "TSLN"} & reached_codes:
            triggers.append("提示眼底影像表型（" + "、".join(sorted({"DN", "MYOPIC_FUNDUS", "TSLN"} & reached_codes)) + "）")
        if not triggers and "ABNORMAL" in reached_codes:
            triggers.append("总体异常标签（ABNORMAL）达到研究筛查线")
        priority_basis = "；".join(triggers) + "。建议结合既往诊治与病史常规随访。"
        actions = ["建议 3~6 个月到眼科常规随访，同时控制血糖、血压与血脂等全身危险因素。"]
        if "LS" in reached_codes:
            actions.append("激光瘢痕（LS）通常对应既往光凝治疗痕迹而非新发疾病，请核对既往眼底激光治疗病历。")
        monitoring_items = ["常规关注视力清晰度变化", "高度近视或既往光凝患者注意有无新发黑影或闪光感"]
        questions_for_clinician = ["本次发现属于既往稳定改变还是需要补充检查？建议多久复查？"]
    else:
        care_priority = "annual_screening"
        priority_label = "定期常规眼健康随访"
        followup_time = "一般人群建议每 1 年检查一次眼底；糖尿病或高度近视人群建议每 6~12 个月复查"
        rule_ids = ["RULE_ROUTINE_ANNUAL_04"]
        _add_sources("SRC_SCREEN_SCOPE")
        priority_basis = (
            "本模型覆盖范围内未检出糖网 1~4 级或达到研究筛查线的异常特征。"
            "需注意模型未覆盖周边视网膜裂孔及全部眼科疾病，不能替代完整临床眼科检查。"
        )
        actions = [
            "保持健康用眼与全身代谢指标管理，按建议周期定期复查眼底。",
            "如后续出现视力变化或新发不适症状，请及时线下就医。",
        ]
        monitoring_items = ["日常双眼中远近视力变化", "有无新发飞蚊、闪光或视物变形"]
        questions_for_clinician = ["结合本人基础病史，下一次常规眼科随访建议安排在什么时候？"]

    if confirmed_plan:
        actions.insert(0, f"已记录医生既往安排：{confirmed_plan}（如无新发急症表现，复查节奏请优先遵照既定医嘱）。")
        if not urgent_hits:
            followup_time = f"优先沿用已确认医生安排：{confirmed_plan}；参考周期：{followup_time}"

    plan_status = "urgent_action_required" if urgent_hits else ("needs_confirmation" if missing_facts else "ready")

    return {
        "care_priority": care_priority,
        "priority_label": priority_label,
        "priority_basis": priority_basis,
        "actions": actions,
        "followup_time": followup_time,
        "monitoring_items": monitoring_items,
        "questions_for_clinician": questions_for_clinician,
        "questions_to_confirm": questions_to_confirm,
        "missing_facts": missing_facts,
        "rule_ids": rule_ids,
        "references": source_ids,
        "plan_status": plan_status,
        "urgent_symptoms_detected": urgent_hits,
    }


def build_evidence_packet(
    screening: dict[str, Any],
    intake: dict[str, Any],
    care: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a versioned EvidencePacket aligned with the system design specification."""
    request_id = screening.get("request_id") or uuid.uuid4().hex
    image_sha = screening.get("image_sha256")
    results = screening.get("results") or {}
    quality = results.get("quality") or {}
    summary = screening.get("report_summary") or {}
    dr_sev = summary.get("dr_severity") or {}
    visual = screening.get("visual_report") or {}
    predictions, _, _ = _extract_predictions(screening)

    return {
        "schema_version": "evidence.v1",
        "exam_id": f"exam_{request_id[:12]}",
        "assessment_id": request_id,
        "image_id": f"img_{image_sha[:12]}" if image_sha else None,
        "image_sha256": image_sha,
        "eye": intake.get("eye") or "unknown",
        "revision": int(intake.get("revision") or 1),
        "created_utc": screening.get("created_utc") or _utc_now(),
        "status": screening.get("status"),
        "quality": {
            "status": quality.get("status"),
            "predicted_label": (quality.get("output") or {}).get("predicted_label"),
            "scope": "technical_checks_and_eyeq_qc",
            "model_id": quality.get("model_id"),
            "clinical_gradability": "not_evaluated",
        },
        "ood": {"status": "not_evaluated", "score": None},
        "uncertainty": {"method": None, "value": None},
        "dr_grading": {
            "status": dr_sev.get("status"),
            "grade": dr_sev.get("grade"),
            "level": dr_sev.get("level"),
            "clinical_stage": dr_sev.get("clinical_stage"),
            "score": dr_sev.get("score"),
            "score_display": dr_sev.get("score_display"),
            "model_id": (results.get("dr_grading") or {}).get("model_id"),
            "weights_sha256": (results.get("dr_grading") or {}).get("weights_sha256"),
            "preprocessing_version": (results.get("dr_grading") or {}).get("preprocessing_version"),
        },
        "predictions": predictions,
        "lesion_localization": {
            "localization_status": visual.get("localization_status"),
            "candidate_status": visual.get("candidate_status"),
            "source_label": visual.get("source_label"),
            "legend": visual.get("legend") or [],
            "reference_grade": visual.get("reference_grade"),
        },
        "visual_report": {
            "original_url": visual.get("original_url"),
            "overlay_url": visual.get("overlay_url"),
            "report_url": visual.get("report_url"),
        },
        "intake": deepcopy(intake),
        "care_priority": care["care_priority"],
        "priority_label": care["priority_label"],
        "rule_ids": care["rule_ids"],
        "source_ids": care["references"],
        "missing_facts": care["missing_facts"],
        "clinical_review": deepcopy(summary.get("clinical_review") or {}),
        "warnings": list(screening.get("warnings") or []),
        "trace_id": request_id,
    }


class ClinicalWorkflowStore:
    """Thread-safe SQLite store for assessments, intake snapshots, care plans, and comparisons."""

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            self._uri = ":memory:"
            self._path: Path | None = None
        else:
            self._path = Path(db_path).resolve()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._uri = str(self._path)
        self._lock = threading.RLock()
        self._conn_obj: sqlite3.Connection | None = None
        self._conn()

    def _conn(self) -> sqlite3.Connection:
        if self._conn_obj is None:
            conn = sqlite3.connect(self._uri, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS assessments (
                    assessment_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    image_sha256 TEXT,
                    eye TEXT NOT NULL DEFAULT 'unknown',
                    intake_revision INTEGER NOT NULL DEFAULT 1,
                    intake_json TEXT NOT NULL,
                    screening_json TEXT NOT NULL,
                    evidence_packet_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS care_plans (
                    plan_id TEXT PRIMARY KEY,
                    assessment_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_utc TEXT NOT NULL,
                    care_priority TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS session_links (
                    session_id TEXT PRIMARY KEY,
                    active_assessment_id TEXT,
                    active_plan_id TEXT,
                    updated_utc TEXT NOT NULL
                );
                """
            )
            conn.commit()
            self._conn_obj = conn
        return self._conn_obj

    def close(self) -> None:
        with self._lock:
            if self._conn_obj is not None:
                self._conn_obj.close()
                self._conn_obj = None

    def record_screening(self, screening: dict[str, Any], *, session_id: str = "default") -> str | None:
        assessment_id = screening.get("request_id")
        if not assessment_id:
            return None
        now = _utc_now()
        intake = {
            "eye": "unknown",
            "symptoms": [],
            "symptom_onset": None,
            "history": [],
            "confirmed_plan": None,
            "revision": 1,
            "updated_utc": now,
        }
        care = evaluate_care_rules(screening, intake)
        evidence = build_evidence_packet(screening, intake, care)
        with self._lock:
            conn = self._conn()
            conn.execute(
                """
                INSERT OR REPLACE INTO assessments (
                    assessment_id, session_id, created_utc, image_sha256, eye,
                    intake_revision, intake_json, screening_json, evidence_packet_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assessment_id,
                    session_id,
                    evidence["created_utc"],
                    screening.get("image_sha256"),
                    "unknown",
                    1,
                    json.dumps(intake, ensure_ascii=False),
                    json.dumps(screening, ensure_ascii=False),
                    json.dumps(evidence, ensure_ascii=False),
                ),
            )
            conn.execute(
                """
                INSERT INTO session_links (session_id, active_assessment_id, active_plan_id, updated_utc)
                VALUES (?, ?, NULL, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    active_assessment_id = excluded.active_assessment_id,
                    active_plan_id = NULL,
                    updated_utc = excluded.updated_utc
                """,
                (session_id, assessment_id, now),
            )
            conn.commit()
        return assessment_id

    def _resolve_assessment_row(
        self, conn: sqlite3.Connection, assessment_id: str | None, session_id: str
    ) -> sqlite3.Row | None:
        if assessment_id:
            return conn.execute(
                "SELECT * FROM assessments WHERE assessment_id = ?", (assessment_id,)
            ).fetchone()
        link = conn.execute(
            "SELECT active_assessment_id FROM session_links WHERE session_id = ?", (session_id,)
        ).fetchone()
        if link and link["active_assessment_id"]:
            row = conn.execute(
                "SELECT * FROM assessments WHERE assessment_id = ?", (link["active_assessment_id"],)
            ).fetchone()
            if row is not None:
                return row
        return conn.execute(
            "SELECT * FROM assessments ORDER BY created_utc DESC LIMIT 1"
        ).fetchone()

    def get_context(self, *, session_id: str = "default") -> dict[str, Any]:
        with self._lock:
            conn = self._conn()
            row = self._resolve_assessment_row(conn, None, session_id)
            if row is None:
                return {
                    "has_active_assessment": False,
                    "session_id": session_id,
                    "active_assessment_id": None,
                    "message": "当前暂无已保存的筛查检查记录，请先上传或提供眼底照片进行筛查。",
                }
            assessment_id = row["assessment_id"]
            evidence = json.loads(row["evidence_packet_json"])
            plan_row = conn.execute(
                "SELECT * FROM care_plans WHERE assessment_id = ? ORDER BY version DESC LIMIT 1",
                (assessment_id,),
            ).fetchone()
            latest_plan = json.loads(plan_row["plan_json"]) if plan_row else None
            recent_rows = conn.execute(
                "SELECT assessment_id, created_utc, eye, image_sha256 FROM assessments ORDER BY created_utc DESC LIMIT 5"
            ).fetchall()
            return {
                "has_active_assessment": True,
                "session_id": session_id,
                "active_assessment_id": assessment_id,
                "exam_id": evidence.get("exam_id"),
                "created_utc": row["created_utc"],
                "eye": row["eye"],
                "intake_revision": row["intake_revision"],
                "status": evidence.get("status"),
                "dr_grading": evidence.get("dr_grading"),
                "care_priority": evidence.get("care_priority"),
                "missing_facts": evidence.get("missing_facts"),
                "visual_report": evidence.get("visual_report"),
                "clinical_review": evidence.get("clinical_review"),
                "latest_care_plan": latest_plan,
                "recent_assessments": [dict(r) for r in recent_rows],
            }

    def get_assessment(self, assessment_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._conn()
            row = conn.execute(
                "SELECT * FROM assessments WHERE assessment_id = ?", (assessment_id,)
            ).fetchone()
            if row is None:
                return None
            plan_row = conn.execute(
                "SELECT * FROM care_plans WHERE assessment_id = ? ORDER BY version DESC LIMIT 1",
                (assessment_id,),
            ).fetchone()
            return {
                "assessment_id": row["assessment_id"],
                "evidence_packet": json.loads(row["evidence_packet_json"]),
                "latest_care_plan": json.loads(plan_row["plan_json"]) if plan_row else None,
            }

    def record_intake(self, payload: IntakeRequest) -> dict[str, Any]:
        with self._lock:
            conn = self._conn()
            row = self._resolve_assessment_row(conn, payload.assessment_id, payload.session_id)
            if row is None:
                raise KeyError("assessment_not_found")
            assessment_id = row["assessment_id"]
            intake = json.loads(row["intake_json"])
            screening = json.loads(row["screening_json"])

            if payload.eye is not None:
                intake["eye"] = payload.eye
            if payload.symptoms:
                existing_symptoms = list(intake.get("symptoms") or [])
                intake["symptoms"] = _clean_text_list(existing_symptoms + payload.symptoms)
            if payload.symptom_onset is not None:
                intake["symptom_onset"] = " ".join(payload.symptom_onset.split()) or None
            if payload.history:
                existing_history = list(intake.get("history") or [])
                intake["history"] = _clean_text_list(existing_history + payload.history)
            if payload.confirmed_plan is not None:
                intake["confirmed_plan"] = " ".join(payload.confirmed_plan.split()) or None

            now = _utc_now()
            revision = int(row["intake_revision"]) + 1
            intake["revision"] = revision
            intake["updated_utc"] = now
            intake["source"] = "user_reported"

            care = evaluate_care_rules(screening, intake)
            evidence = build_evidence_packet(screening, intake, care)

            conn.execute(
                """
                UPDATE assessments
                SET eye = ?, intake_revision = ?, intake_json = ?, evidence_packet_json = ?
                WHERE assessment_id = ?
                """,
                (
                    intake["eye"],
                    revision,
                    json.dumps(intake, ensure_ascii=False),
                    json.dumps(evidence, ensure_ascii=False),
                    assessment_id,
                ),
            )
            conn.commit()
            visual = evidence.get("visual_report") or {}
            return {
                "assessment_id": assessment_id,
                "intake_revision": revision,
                "intake": intake,
                "visual_report": visual,
                "report_url": visual.get("report_url"),
                "care_preview": {
                    "care_priority": care["care_priority"],
                    "priority_label": care["priority_label"],
                    "urgent_symptoms_detected": care["urgent_symptoms_detected"],
                    "missing_facts": care["missing_facts"],
                },
            }

    def evaluate_care(self, payload: CareEvaluateRequest) -> dict[str, Any]:
        with self._lock:
            conn = self._conn()
            row = self._resolve_assessment_row(conn, payload.assessment_id, payload.session_id)
            if row is None:
                raise KeyError("assessment_not_found")
            assessment_id = row["assessment_id"]
            intake = json.loads(row["intake_json"])
            screening = json.loads(row["screening_json"])
            care = evaluate_care_rules(screening, intake)
            evidence = build_evidence_packet(screening, intake, care)

            ver_row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS max_ver FROM care_plans WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchone()
            version = int(ver_row["max_ver"]) + 1
            plan_id = f"plan_{uuid.uuid4().hex[:12]}"
            now = _utc_now()

            visual = evidence.get("visual_report") or {}
            plan_record = {
                "plan_id": plan_id,
                "assessment_id": assessment_id,
                "version": version,
                "intake_revision": row["intake_revision"],
                "eye": intake["eye"],
                "created_utc": now,
                "visual_report": visual,
                "report_url": visual.get("report_url"),
                **care,
            }
            conn.execute(
                """
                INSERT INTO care_plans (plan_id, assessment_id, version, created_utc, care_priority, plan_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (plan_id, assessment_id, version, now, care["care_priority"], json.dumps(plan_record, ensure_ascii=False)),
            )
            conn.execute(
                "UPDATE assessments SET evidence_packet_json = ? WHERE assessment_id = ?",
                (json.dumps(evidence, ensure_ascii=False), assessment_id),
            )
            conn.execute(
                """
                INSERT INTO session_links (session_id, active_assessment_id, active_plan_id, updated_utc)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    active_assessment_id = excluded.active_assessment_id,
                    active_plan_id = excluded.active_plan_id,
                    updated_utc = excluded.updated_utc
                """,
                (payload.session_id, assessment_id, plan_id, now),
            )
            conn.commit()
            return plan_record

    def compare_assessments(self, payload: CompareRequest) -> dict[str, Any]:
        with self._lock:
            conn = self._conn()
            prev_row = conn.execute(
                "SELECT * FROM assessments WHERE assessment_id = ?", (payload.previous_assessment_id,)
            ).fetchone()
            curr_row = conn.execute(
                "SELECT * FROM assessments WHERE assessment_id = ?", (payload.current_assessment_id,)
            ).fetchone()
            if prev_row is None or curr_row is None:
                missing_id = payload.previous_assessment_id if prev_row is None else payload.current_assessment_id
                raise KeyError(f"assessment_not_found:{missing_id}")

        prev_ev = json.loads(prev_row["evidence_packet_json"])
        curr_ev = json.loads(curr_row["evidence_packet_json"])
        prev_scr = json.loads(prev_row["screening_json"])
        curr_scr = json.loads(curr_row["screening_json"])

        prev_eye, curr_eye = prev_row["eye"], curr_row["eye"]
        base = {
            "previous_assessment_id": payload.previous_assessment_id,
            "current_assessment_id": payload.current_assessment_id,
            "previous_created_utc": prev_row["created_utc"],
            "current_created_utc": curr_row["created_utc"],
            "previous_eye": prev_eye,
            "current_eye": curr_eye,
        }

        if prev_eye == "unknown" or curr_eye == "unknown":
            return {
                **base,
                "comparable": False,
                "reason": "eye_unconfirmed",
                "message": "存在眼别为 unknown 的记录，请先确认左右眼别后再进行同眼对比（两个 unknown 不代表同一只眼）。",
            }
        if prev_eye != curr_eye:
            return {
                **base,
                "comparable": False,
                "reason": "different_eye",
                "message": f"两次检查眼别不一致（分别为 {prev_eye} 与 {curr_eye}），不能作为同眼纵向比较。",
            }

        prev_q = (prev_ev.get("quality") or {}).get("predicted_label")
        curr_q = (curr_ev.get("quality") or {}).get("predicted_label")
        if prev_ev.get("status") not in ("ok", "partial") or curr_ev.get("status") not in ("ok", "partial") or (
            prev_q not in ("Good", "Usable") or curr_q not in ("Good", "Usable")
        ):
            return {
                **base,
                "comparable": False,
                "reason": "quality_not_comparable",
                "message": "至少一次检查未完成分析或图像质量不足，无法进行可靠比较。",
            }

        version_differences = []
        for container in ("results", "research_results"):
            prev_group = prev_scr.get(container) or {}
            curr_group = curr_scr.get(container) or {}
            for stage_name in sorted(set(prev_group) | set(curr_group)):
                p_stage = prev_group.get(stage_name) or {}
                c_stage = curr_group.get(stage_name) or {}
                if (p_stage.get("model_id"), p_stage.get("preprocessing_version")) != (
                    c_stage.get("model_id"), c_stage.get("preprocessing_version")
                ):
                    version_differences.append({
                        "stage": stage_name,
                        "previous_model_id": p_stage.get("model_id"),
                        "current_model_id": c_stage.get("model_id"),
                    })

        prev_grade = (prev_ev.get("dr_grading") or {}).get("grade")
        curr_grade = (curr_ev.get("dr_grading") or {}).get("grade")
        grade_delta = (curr_grade - prev_grade) if (type(prev_grade) is int and type(curr_grade) is int) else None

        prev_preds = {(p["source_model"], p["label_code"]): p for p in prev_ev.get("predictions") or []}
        curr_preds = {(p["source_model"], p["label_code"]): p for p in curr_ev.get("predictions") or []}
        prev_reached = {k[1] for k, v in prev_preds.items() if v["screening_flag"] == "positive"}
        curr_reached = {k[1] for k, v in curr_preds.items() if v["screening_flag"] == "positive"}

        score_diffs = []
        for key in sorted(set(prev_preds) & set(curr_preds)):
            p, c = prev_preds[key], curr_preds[key]
            if p.get("model_id") != c.get("model_id"):
                continue
            ps, cs = p.get("raw_score"), c.get("raw_score")
            delta = round(cs - ps, 6) if (type(ps) in (int, float) and type(cs) in (int, float)) else None
            score_diffs.append({
                "source_model": key[0],
                "label_code": key[1],
                "label_name": c.get("label_name"),
                "previous_score": ps,
                "current_score": cs,
                "score_delta": delta,
                "previous_flag": p.get("screening_flag"),
                "current_flag": c.get("screening_flag"),
            })

        return {
            **base,
            "comparable": True,
            "reason": None,
            "version_consistent": len(version_differences) == 0,
            "version_differences": version_differences,
            "dr_grade_comparison": {
                "previous_grade": prev_grade,
                "current_grade": curr_grade,
                "grade_delta": grade_delta,
            },
            "screening_flag_changes": {
                "previous_reached": sorted(prev_reached),
                "current_reached": sorted(curr_reached),
                "newly_triggered": sorted(curr_reached - prev_reached),
                "resolved_below_threshold": sorted(prev_reached - curr_reached),
                "persistently_triggered": sorted(prev_reached & curr_reached),
            },
            "score_diffs": score_diffs,
            "interpretation_note": (
                "模型分类分数不具备严重度量纲，得分波动仅反映同一方法下两张图像的输出差异，"
                "不能直接等同于临床病情进展或好转；请结合采集时间及临床医生记录综合判断。"
            ),
        }
