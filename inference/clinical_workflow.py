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

from inference.clinical_referral import build_clinical_decision
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


def _coerce_eye(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    text = value.strip().lower()
    if text in ("left", "os", "l") or "左" in text:
        return "left"
    if text in ("right", "od", "r") or "右" in text:
        return "right"
    return "unknown"


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
        comparable_items = report_group.get("items") if report_group.get("status") == "completed" else []
        item_map = {item.get("code"): item for item in (comparable_items or []) if item.get("code")}
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
    symptoms = _coerce_str_list(intake.get("symptoms"))
    history = _coerce_str_list(intake.get("history"))
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
    decision = build_clinical_decision(screening, predictions, dr_grade, quality_ok=quality_ok, urgent_hits=urgent_hits)
    followup_time = decision["timeframe"]
    priority_basis = " ".join(
        f"{item['name']}：{item['evidence']}{item['basis']}" for item in decision["trigger_items"]
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

    source_ids: list[str] = []

    def _add_sources(*codes: str):
        for c in codes:
            if c and c not in source_ids:
                source_ids.append(c)

    if decision["status"] == "emergency":
        care_priority = "urgent_ophthalmology"
        priority_label = "立即眼科急诊评估"
        rule_ids = ["RULE_URGENT_SYMPTOM_01"]
        _add_sources("SRC_NEI_RD_URGENT")
        actions = [
            "请立即前往具备眼底检查条件的眼科急诊。",
            "就诊时请告知医生症状起始时间、眼别及近期变化，随身携带可取得的眼底资料。",
        ]
        monitoring_items = ["视野遮挡范围是否扩大", "飞蚊或闪光感是否加重", "单眼视力是否进一步下降"]
        questions_for_clinician = ["是否需要散瞳检查周边视网膜或行眼部 B 超？"]
    elif decision["status"] == "image_ungradable":
        care_priority = "retake_image"
        priority_label = "图像可判读性不足，建议重拍或人工复核"
        rule_ids = ["RULE_QC_RETAKE_01"]
        _add_sources("SRC_QC_RETAKE")
        priority_basis = f"图像质控判定为 {quality_label or '未完成'}，当前图像无法用于转诊分级。"
        actions = ["重新采集清晰眼底彩照，或由眼科医生通过散瞳等检查评估。"]
        monitoring_items = ["是否新发视力下降、黑影飘动或视物变形"]
        questions_for_clinician = ["是否需要散瞳拍照或补充 OCT 检查？"]
    else:
        level = decision["level"]
        care_priority, priority_label, rule_id = {
            "L4": ("prompt_ophthalmology", "尽快安排专科评估", "RULE_DR_SEVERE_OR_VASC_01"),
            "L3": ("scheduled_ophthalmology", "限期眼科专科复核", "RULE_MODERATE_OR_SPECIALIST_02"),
            "L2": ("routine_followup", "常规眼科随访与病史核对", "RULE_MILD_OR_PHENOTYPE_03"),
            "L1": ("annual_screening", "定期常规眼健康随访", "RULE_ROUTINE_ANNUAL_04"),
            None: ("scheduled_ophthalmology", "完成专科复核后再分级", "RULE_SCREEN_REVIEW_REQUIRED_01"),
        }[level]
        rule_ids = [rule_id]
        actions = [f"{decision['destination']} · {followup_time}。"]
        for item in decision["trigger_items"]:
            code = item["code"]
            if code.startswith("DR"):
                _add_sources(LABEL_SOURCES["DR"])
            else:
                _add_sources(LABEL_SOURCES.get(code))
            if item["level"] == level and item["tests"]:
                actions.append(f"{item['name']}建议由医生结合情况安排" + "、".join(item["tests"]) + "。")
            if code in {"ODE", "AMD"}:
                actions.append(item["basis"])
        if level is None or level == "L1":
            _add_sources("SRC_SCREEN_SCOPE")
        monitoring_items = ["视力变化", "新发飞蚊、闪光、视物变形或遮挡"]
        questions_for_clinician = ["结合原图和临床资料，是否需要补充检查或调整复查时间？"]
    if confirmed_plan and decision["level"] in {"L1", "L2", "L3"}:
        actions.insert(0, f"已记录医生既往安排：{confirmed_plan}（如无新发急症表现，复查节奏请优先遵照既定医嘱）。")
        followup_time = f"优先沿用已确认医生安排：{confirmed_plan}；参考周期：{followup_time}"

    if urgent_hits:
        plan_status = "urgent_action_required"
    else:
        plan_status = "needs_confirmation" if missing_facts or decision["level"] is None else "ready"

    return {
        "clinical_decision": decision,
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
        "clinical_decision": deepcopy(care["clinical_decision"]),
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

    def record_screening(
        self,
        screening: dict[str, Any],
        *,
        session_id: str = "default",
        eye: str | None = None,
        symptoms: str | list[str] | None = None,
        history: str | list[str] | None = None,
    ) -> str | None:
        assessment_id = screening.get("request_id")
        if not assessment_id:
            return None
        now = _utc_now()
        norm_eye = _coerce_eye(eye)
        norm_symptoms = _clean_text_list(_coerce_str_list(symptoms))
        norm_history = _clean_text_list(_coerce_str_list(history))
        has_user_facts = norm_eye != "unknown" or bool(norm_symptoms) or bool(norm_history)
        intake = {
            "eye": norm_eye,
            "symptoms": norm_symptoms,
            "symptom_onset": None,
            "history": norm_history,
            "confirmed_plan": None,
            "revision": 1,
            "updated_utc": now,
        }
        if has_user_facts:
            intake["source"] = "user_reported"
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
                    norm_eye,
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
            "SELECT * FROM assessments WHERE session_id = ? ORDER BY created_utc DESC LIMIT 1",
            (session_id,),
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
                "SELECT assessment_id, created_utc, eye, image_sha256 FROM assessments "
                "WHERE session_id = ? ORDER BY created_utc DESC LIMIT 5",
                (session_id,),
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
                "clinical_decision": evidence.get("clinical_decision"),
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
                    "clinical_decision": care["clinical_decision"],
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
