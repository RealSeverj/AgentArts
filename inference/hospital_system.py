"""Hospital PACS imaging sync, phone-keyed patient records, dual-audience persistent reports, and agent query helpers."""
from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import re
import shutil
import sqlite3
from tempfile import TemporaryDirectory
import threading
from typing import Any, Literal
import uuid

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from inference.clinical_review import build_clinical_review
from inference.clinical_workflow import (
    _coerce_str_list,
    _detect_urgent_symptoms,
    _extract_predictions,
    evaluate_care_rules,
)
from inference.image_io import InputImageError, URLFetchPolicy, decode_image_bytes
from inference.report_summary import DR_LEVELS, LABELS
from inference.schemas import FundusResponse
from inference.visual_report import _comparison, _image_readable, _quality_blocked_info, _report_html

ROOT = Path(__file__).resolve().parents[1]
PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
EYE_CN = {"left": "左眼", "right": "右眼"}
EYE_MEDICAL = {"left": "左眼（OS）", "right": "右眼（OD）"}
CARE_LEVEL_STYLE = """
  .care-level{display:inline-block;border-radius:6px;padding:1px 8px;font-weight:700;white-space:nowrap;background:#e2e8f0;color:#334155}
  .care-level-l4{background:#fee2e2;color:#991b1b}
  .care-level-l3{background:#ffedd5;color:#9a3412}
  .care-level-l2{background:#fef3c7;color:#854d0e}
  .care-level-l1{background:#dcfce7;color:#166534}
"""


class PatientUpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phone: str = Field(pattern=r"^1[3-9]\d{9}$")
    name: str = Field(min_length=1, max_length=64)
    age: int = Field(default=50, ge=1, le=120)
    gender: Literal["男", "女", "未知"] = "未知"
    chronic_history: str = Field(default="", max_length=512)
    eye_history: str = Field(default="", max_length=512)


class HospitalImageUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phone: str = Field(pattern=r"^1[3-9]\d{9}$")
    eye: Literal["left", "right"]
    source: Literal["upload", "sample"] = "upload"
    sample_id: str | None = Field(default=None, max_length=64)
    image_base64: str | None = Field(default=None, max_length=16 * 1024 * 1024)
    note: str = Field(default="眼科拍片室常规眼底照相", max_length=256)
    name: str | None = Field(default=None, max_length=64)
    age: int | None = Field(default=None, ge=1, le=120)
    gender: Literal["男", "女", "未知"] | None = None
    chronic_history: str | None = Field(default=None, max_length=512)
    eye_history: str | None = Field(default=None, max_length=512)


class AgentQueryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str = Field(min_length=1, max_length=4000)


class PortalChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["patient", "doctor"]
    phone: str | None = Field(default=None, max_length=32)
    action: Literal["auto", "bind", "report_or_diagnose", "symptom", "education"] = "auto"
    eye: Literal["left", "right", "auto"] = "auto"
    query: str = Field(default="", max_length=2000)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_phone(text: str) -> str | None:
    m = PHONE_RE.search(text or "")
    return m.group(1) if m else None


def extract_eye(text: str) -> str | None:
    t = (text or "").lower()
    has_left = ("左" in t) or ("left" in t) or bool(re.search(r"\bos\b", t))
    has_right = ("右" in t) or ("right" in t) or bool(re.search(r"\bod\b", t))
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    return None


def _friendly_labels(codes: list[str]) -> list[str]:
    friendly = {
        "DR": "糖尿病相关的眼底微血管变化",
        "AMD": "黄斑区老年性改变线索",
        "RVO": "视网膜静脉血管循环阻滞线索",
        "CSR": "黄斑区积液相关变化线索",
        "glaucoma": "青光眼相关的视神经筛查线索",
        "DN": "黄斑区微小的玻璃膜疣沉着",
        "OPTIC_DISC_CUPPING": "视盘中心生理或结构凹陷偏大",
        "ODE": "视盘边缘水肿信号",
        "ODP": "视神经颜色偏淡信号",
        "MYOPIC_FUNDUS": "近视相关的眼底底色表现",
        "TSLN": "常见的豹纹状眼底纹理（多见于近视或中老年）",
        "LS": "可能是既往眼底激光治疗留下的痕迹",
        "ABNORMAL": "眼底整体影像信号提示需要医生复核",
    }
    return [friendly.get(c, LABELS.get(c, (c,))[0]) for c in codes]


def _evaluate_report_care(screening: dict[str, Any], patient: dict[str, Any], eye: str) -> dict[str, Any]:
    history = _coerce_str_list(f"{patient.get('chronic_history', '')} {patient.get('eye_history', '')}")
    return evaluate_care_rules(screening, {"eye": eye, "symptoms": [], "history": history})


def _read_report_care(row: sqlite3.Row, patient: dict[str, Any], screening: dict[str, Any] | None = None) -> dict[str, Any]:
    care = json.loads(row["care_plan_json"])
    if "clinical_decision" in care:
        return care
    if screening is None:
        screening = json.loads(row["screening_json"])
    return _evaluate_report_care(screening, patient, row["eye"])


def _decision_badge(level: str | None) -> str:
    if level in {"L1", "L2", "L3", "L4"}:
        return f'<span class="care-level care-level-{level.lower()}">{level}</span>'
    return ""


def _seed_baseline_screening(dr_grade: int, *, status_label: str = "Good") -> dict[str, Any]:
    """Create a realistic baseline screening snapshot for pre-seeded historical patient records."""
    dr_info = DR_LEVELS[dr_grade]
    dr_score = 0.851 if dr_grade == 2 else 0.962
    is_pos = dr_grade >= 1
    return {
        "contract_version": "research-20261122.1",
        "status": "ok",
        "message": "历史基线检查已归档。",
        "report_summary": {
            "dr_severity": {
                "status": "completed",
                "grade": dr_grade,
                "level": dr_info[0],
                "clinical_stage": dr_info[1],
                "score": dr_score,
                "score_display": f"{dr_score * 100:.3f}分",
            },
            "five_label_screening": {
                "status": "completed",
                "model_id": "multidisease_rfmid_vitb14_392_best21_20261114",
                "items": [
                    {
                        "code": "DR",
                        "name": "糖尿病视网膜病变",
                        "kind": "疾病",
                        "score": 0.78 if is_pos else 0.008,
                        "research_threshold": 0.043763,
                        "research_threshold_met": is_pos,
                    },
                    {
                        "code": "ABNORMAL",
                        "name": "总体异常",
                        "kind": "综合标签",
                        "score": 0.89 if is_pos else 0.12,
                        "research_threshold": 0.841816,
                        "research_threshold_met": is_pos,
                    },
                ],
            },
            "research": {
                "multilabel_11": {"status": "completed", "model_id": "rfmid11_vitl_vitb_disc_mean_20261122", "items": []},
                "glaucoma": {
                    "status": "completed",
                    "model_id": "glaucoma_historical_vitb392_selected_20261122",
                    "items": [
                        {
                            "code": "glaucoma",
                            "name": "青光眼",
                            "kind": "疾病",
                            "score": 0.03,
                            "research_threshold": 0.149283,
                            "research_threshold_met": False,
                        }
                    ],
                },
            },
            "clinical_review": {
                "quality": f"图像质量{'良好，可判读。' if status_label == 'Good' else '可判读，质量受限。'}",
                "primary": f"糖网专病模型预测为{dr_grade}级，{dr_info[1]}。",
                "review_items": [],
                "other_results": [],
                "scope": "研究筛查线用于选择复阅项目。病变性质和分期由阅片医生结合原图及临床资料确认。",
            },
        },
        "results": {
            "quality": {
                "status": "completed",
                "model_id": "quality_eyeq_vitb14_392_best11_20261114",
                "output": {"predicted_label": status_label},
            }
        },
        "research_results": {},
    }


def _compute_eye_comparison(prev_report: dict[str, Any] | None, curr_screening: dict[str, Any]) -> dict[str, Any] | None:
    if prev_report is None:
        return None
    prev_scr = json.loads(prev_report["screening_json"])
    prev_preds, prev_reached, prev_grade = _extract_predictions(prev_scr)
    curr_preds, curr_reached, curr_grade = _extract_predictions(curr_screening)
    del prev_preds, curr_preds
    grade_delta = (curr_grade - prev_grade) if (type(prev_grade) is int and type(curr_grade) is int) else None
    resolved = sorted(prev_reached - curr_reached)
    newly = sorted(curr_reached - prev_reached)
    if (grade_delta is not None and grade_delta < 0) or (len(resolved) > len(newly) and not newly):
        trend = "improved"
        patient_note = "跟上一次检查相比，这次照片里提示需要留意的变化有所减少，整体情况在往好的方向发展（具体恢复程度下次门诊让主治医生再帮您最终确认一下）。"
        doctor_note = f"与前次同眼记录（{prev_report['analyzed_utc'][:10]}）相比，糖网分级由 {prev_grade} 级降至 {curr_grade} 级，阳性提示项减少，呈改善趋势。"
    elif (grade_delta is not None and grade_delta > 0) or newly:
        trend = "worsened_attention"
        patient_note = "和上次检查比起来，这次照片新出现了一点值得留意的信号，为了安全起见，建议您带好这份报告去眼科请医生重点看一看。"
        doctor_note = f"与前次同眼记录（{prev_report['analyzed_utc'][:10]}）相比，糖网分级（{prev_grade} -> {curr_grade}）或新触发筛查项（{', '.join(newly) or '无'}）提示关注度上升，建议尽快专科复核。"
    else:
        trend = "stable"
        patient_note = "和上一次检查相比，眼底整体表现基本保持稳定，没有出现明显的新增波动，按医生建议的节奏定期复查就好。"
        doctor_note = f"与前次同眼记录（{prev_report['analyzed_utc'][:10]}）相比，糖网分级保持 {curr_grade} 级，筛查特征整体平稳。"
    return {
        "previous_report_id": prev_report["report_id"],
        "previous_analyzed_utc": prev_report["analyzed_utc"],
        "previous_dr_grade": prev_grade,
        "current_dr_grade": curr_grade,
        "grade_delta": grade_delta,
        "resolved_codes": resolved,
        "newly_triggered_codes": newly,
        "trend": trend,
        "patient_note": patient_note,
        "doctor_note": doctor_note,
    }


class HospitalSystemStore:
    """Thread-safe SQLite hospital patient + PACS image + dual-report store."""

    def __init__(self, root_dir: Path | str | None = None, *, public_origin: str | None = None):
        self._temp_dir: TemporaryDirectory[str] | None = None
        if root_dir is None:
            self._temp_dir = TemporaryDirectory(prefix="retina-his-", ignore_cleanup_errors=True)
            self.root = Path(self._temp_dir.name).resolve()
        else:
            self.root = Path(root_dir).resolve()
            self.root.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.root / "images"
        self.reports_dir = self.root / "reports"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.public_origin = (public_origin or "").rstrip("/")
        self.db_path = self.root / "hospital_pacs.sqlite3"
        self._lock = threading.RLock()
        self._conn_obj: sqlite3.Connection | None = None
        self._conn()
        self._seed_demo_patients()

    def _conn(self) -> sqlite3.Connection:
        if self._conn_obj is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS his_patients (
                    phone TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL DEFAULT 50,
                    gender TEXT NOT NULL DEFAULT '未知',
                    chronic_history TEXT NOT NULL DEFAULT '',
                    eye_history TEXT NOT NULL DEFAULT '',
                    created_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS his_images (
                    image_id TEXT PRIMARY KEY,
                    phone TEXT NOT NULL,
                    eye TEXT NOT NULL,
                    captured_utc TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    image_relpath TEXT NOT NULL,
                    image_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    latest_report_id TEXT,
                    FOREIGN KEY (phone) REFERENCES his_patients(phone) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS his_reports (
                    report_id TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    phone TEXT NOT NULL,
                    eye TEXT NOT NULL,
                    analyzed_utc TEXT NOT NULL,
                    dr_grade INTEGER,
                    care_priority TEXT NOT NULL,
                    screening_json TEXT NOT NULL,
                    care_plan_json TEXT NOT NULL,
                    comparison_json TEXT,
                    has_overlay INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (image_id) REFERENCES his_images(image_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS his_sessions (
                    role_channel TEXT PRIMARY KEY,
                    active_phone TEXT NOT NULL,
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
            if self._temp_dir is not None:
                self._temp_dir.cleanup()
                self._temp_dir = None

    def _report_urls(self, report_id: str, has_overlay: bool) -> dict[str, str | None]:
        prefix = self.public_origin if self.public_origin else ""
        base = f"{prefix}/v1/his/reports/{report_id}"
        return {
            "patient_report_url": f"{base}/patient.html",
            "doctor_report_url": f"{base}/doctor.html",
            "original_url": f"{base}/original.jpg",
            "overlay_url": f"{base}/overlay.jpg" if has_overlay else None,
        }

    def _save_image_file(self, image_id: str, raw: bytes) -> tuple[str, str]:
        policy = URLFetchPolicy()
        decoded = decode_image_bytes(raw, policy=policy)
        try:
            rel = f"{image_id}.jpg"
            dest = self.images_dir / rel
            decoded.image.convert("RGB").save(dest, format="JPEG", quality=95, subsampling=0)
            return rel, decoded.sha256
        finally:
            decoded.image.close()

    def _seed_demo_patients(self) -> None:
        with self._lock:
            conn = self._conn()
            count = conn.execute("SELECT COUNT(*) FROM his_patients").fetchone()[0]
            if count > 0:
                return
            sample_dr = ROOT / "examples" / "IDRiD_03.jpg"
            sample_norm = ROOT / "inference_runtime" / "local_gpu" / "retina_public_sample_proxy.jpg"
            if not (sample_dr.is_file() and sample_norm.is_file()):
                return

            now = _utc_now()
            patients = [
                ("13800138001", "张建国", 62, "男", "2型糖尿病8年，高血压5年", "半年前左眼曾提示中度糖网改变，近期一直按医嘱控糖复诊"),
                ("13800138002", "林晓雯", 29, "女", "无基础慢病", "双眼近视550度，年度常规眼底体检"),
                ("13800138003", "李秀英", 68, "女", "2型糖尿病12年，偶感视物微蒙", "近2年未查眼底"),
            ]
            for row in patients:
                conn.execute(
                    "INSERT INTO his_patients (phone, name, age, gender, chronic_history, eye_history, created_utc) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (*row, now),
                )

            # Seed Patient 13800138001: historical left DR2 + historical right DR0 + TODAY pending left improved!
            dr_bytes = sample_dr.read_bytes()
            norm_bytes = sample_norm.read_bytes()
            dr_sha = hashlib.sha256(dr_bytes).hexdigest()
            norm_sha = hashlib.sha256(norm_bytes).hexdigest()

            def _write_seed_jpg(img_id: str, raw: bytes, sha: str) -> tuple[str, str]:
                rel = f"{img_id}.jpg"
                (self.images_dir / rel).write_bytes(raw)
                return rel, sha

            def _create_seed_analyzed(img_id: str, rep_id: str, phone: str, eye: str, ts: str, raw: bytes, sha_val: str, dr_grade: int, note: str):
                rel, sha = _write_seed_jpg(img_id, raw, sha_val)
                scr = _seed_baseline_screening(dr_grade)
                scr["request_id"] = rep_id
                scr["image_sha256"] = sha
                scr["created_utc"] = ts
                care = evaluate_care_rules(
                    scr,
                    {"eye": eye, "symptoms": [], "history": ["2型糖尿病8年", "高血压5年"]},
                )
                rep_dir = self.reports_dir / rep_id
                rep_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.images_dir / rel, rep_dir / "original.jpg")

                conn.execute(
                    "INSERT INTO his_images (image_id, phone, eye, captured_utc, note, image_relpath, image_sha256, status, latest_report_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'analyzed', ?)",
                    (img_id, phone, eye, ts, note, rel, sha, rep_id),
                )
                conn.execute(
                    "INSERT INTO his_reports (report_id, image_id, phone, eye, analyzed_utc, dr_grade, care_priority, "
                    "screening_json, care_plan_json, comparison_json, has_overlay) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0)",
                    (rep_id, img_id, phone, eye, ts, dr_grade, care["care_priority"],
                     json.dumps(scr, ensure_ascii=False), json.dumps(care, ensure_ascii=False)),
                )

            _create_seed_analyzed(
                "img_seed_138001_l0", "rep_seed_138001_l0", "13800138001", "left",
                "2026-05-20T09:15:00+00:00", dr_bytes, dr_sha, 2, "半年前门诊眼底基线复查（左眼）",
            )
            _create_seed_analyzed(
                "img_seed_138001_r0", "rep_seed_138001_r0", "13800138001", "right",
                "2026-10-12T10:20:00+00:00", norm_bytes, norm_sha, 0, "上月右眼常规眼底复查",
            )

            # Pending today's left image for 13800138001, right for 13800138002, left for 13800138003
            for img_id, phone, eye, raw, sha_val, note in (
                ("img_seed_138001_l1", "13800138001", "left", norm_bytes, norm_sha, "今日拍片室同步左眼最新复查照（待分析）"),
                ("img_seed_138002_r1", "13800138002", "right", norm_bytes, norm_sha, "今日体检中心右眼眼底照（待分析）"),
                ("img_seed_138003_l1", "13800138003", "left", dr_bytes, dr_sha, "今日内分泌科转诊左眼眼底照（待分析）"),
            ):
                rel, sha = _write_seed_jpg(img_id, raw, sha_val)
                conn.execute(
                    "INSERT INTO his_images (image_id, phone, eye, captured_utc, note, image_relpath, image_sha256, status, latest_report_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL)",
                    (img_id, phone, eye, now, note, rel, sha),
                )
            conn.commit()

    def set_active_phone(self, role: str, phone: str) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                """
                INSERT INTO his_sessions (role_channel, active_phone, updated_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(role_channel) DO UPDATE SET
                    active_phone = excluded.active_phone,
                    updated_utc = excluded.updated_utc
                """,
                (role, phone, _utc_now()),
            )
            conn.commit()

    def get_active_phone(self, role: str) -> str | None:
        with self._lock:
            row = self._conn().execute(
                "SELECT active_phone FROM his_sessions WHERE role_channel = ?", (role,)
            ).fetchone()
            return row["active_phone"] if row else None

    def upsert_patient(self, req: PatientUpsertRequest) -> dict[str, Any]:
        now = _utc_now()
        with self._lock:
            conn = self._conn()
            conn.execute(
                """
                INSERT INTO his_patients (phone, name, age, gender, chronic_history, eye_history, created_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(phone) DO UPDATE SET
                    name = excluded.name,
                    age = excluded.age,
                    gender = excluded.gender,
                    chronic_history = excluded.chronic_history,
                    eye_history = excluded.eye_history
                """,
                (req.phone, req.name.strip(), req.age, req.gender, req.chronic_history.strip(), req.eye_history.strip(), now),
            )
            conn.commit()
        return self.get_patient_overview(req.phone)  # type: ignore[return-value]

    def get_patient_overview(self, phone: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._conn()
            p = conn.execute("SELECT * FROM his_patients WHERE phone = ?", (phone,)).fetchone()
            if p is None:
                return None
            imgs = conn.execute(
                "SELECT * FROM his_images WHERE phone = ? ORDER BY captured_utc DESC", (phone,)
            ).fetchall()
            reps = conn.execute(
                "SELECT report_id, image_id, eye, analyzed_utc, dr_grade, care_priority, has_overlay, care_plan_json, "
                "screening_json FROM his_reports WHERE phone = ? ORDER BY analyzed_utc DESC",
                (phone,),
            ).fetchall()

        patient = dict(p)
        image_list = []
        for im in imgs:
            item = dict(im)
            if im["latest_report_id"]:
                item["report_urls"] = self._report_urls(im["latest_report_id"], True)
            image_list.append(item)

        report_list = []
        for rp in reps:
            d = dict(rp)
            care = _read_report_care(rp, patient)
            del d["screening_json"], d["care_plan_json"]
            decision = care.get("clinical_decision") or {}
            d.update(care_priority=care["care_priority"], clinical_level=decision.get("level"),
                     clinical_title=decision.get("title"))
            d.update(self._report_urls(rp["report_id"], bool(rp["has_overlay"])))
            report_list.append(d)

        return {
            "patient": patient,
            "images": image_list,
            "reports": report_list,
            "pending_eyes": sorted({im["eye"] for im in imgs if im["status"] == "pending"}),
            "available_eyes": sorted({im["eye"] for im in imgs}),
        }

    def list_patients_brief(self) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._conn()
            rows = conn.execute("SELECT phone, name, age, gender, chronic_history FROM his_patients ORDER BY created_utc ASC").fetchall()
            res = []
            for r in rows:
                pending_cnt = conn.execute(
                    "SELECT COUNT(*) FROM his_images WHERE phone = ? AND status = 'pending'", (r["phone"],)
                ).fetchone()[0]
                res.append({**dict(r), "pending_count": pending_cnt})
            return res

    def upload_hospital_image(self, req: HospitalImageUploadRequest) -> dict[str, Any]:
        with self._lock:
            conn = self._conn()
            existing = conn.execute("SELECT * FROM his_patients WHERE phone = ?", (req.phone,)).fetchone()
            if existing is None:
                if not (req.name and req.name.strip()):
                    raise ValueError("该手机号首次建档，请同时填写患者姓名。")
                conn.execute(
                    "INSERT INTO his_patients (phone, name, age, gender, chronic_history, eye_history, created_utc) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        req.phone,
                        req.name.strip(),
                        req.age or 50,
                        req.gender or "未知",
                        (req.chronic_history or "").strip(),
                        (req.eye_history or "").strip(),
                        _utc_now(),
                    ),
                )
            elif any(v is not None for v in (req.name, req.age, req.gender, req.chronic_history, req.eye_history)):
                conn.execute(
                    """
                    UPDATE his_patients
                    SET name = COALESCE(?, name),
                        age = COALESCE(?, age),
                        gender = COALESCE(?, gender),
                        chronic_history = COALESCE(?, chronic_history),
                        eye_history = COALESCE(?, eye_history)
                    WHERE phone = ?
                    """,
                    (
                        req.name.strip() if req.name else None,
                        req.age,
                        req.gender,
                        req.chronic_history.strip() if req.chronic_history is not None else None,
                        req.eye_history.strip() if req.eye_history is not None else None,
                        req.phone,
                    ),
                )

            if req.source == "sample":
                sample_map = {
                    "idrid_03": ROOT / "examples" / "IDRiD_03.jpg",
                    "public_proxy": ROOT / "inference_runtime" / "local_gpu" / "retina_public_sample_proxy.jpg",
                }
                path = sample_map.get(req.sample_id or "")
                if not path or not path.is_file():
                    raise InputImageError("invalid_image", "未找到对应的拍片室演示样片。")
                raw = path.read_bytes()
            else:
                text = (req.image_base64 or "").strip()
                if "," in text and text.lower().startswith("data:"):
                    text = text.split(",", 1)[1]
                if not text:
                    raise InputImageError("invalid_image", "请选择需要上传的眼底彩色照片。")
                try:
                    raw = base64.b64decode(text, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise InputImageError("invalid_image", "图片数据格式无效。") from exc

            image_id = f"img_{uuid.uuid4().hex[:12]}"
            rel, sha = self._save_image_file(image_id, raw)
            now = _utc_now()
            conn.execute(
                "INSERT INTO his_images (image_id, phone, eye, captured_utc, note, image_relpath, image_sha256, status, latest_report_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL)",
                (image_id, req.phone, req.eye, now, req.note.strip() or "眼科拍片室同步影像", rel, sha),
            )
            conn.commit()
        return {
            "ok": True,
            "image_id": image_id,
            "phone": req.phone,
            "eye": req.eye,
            "status": "pending",
            "overview": self.get_patient_overview(req.phone),
        }

    def analyze_or_fetch_eye(self, *, phone: str, eye: str | None, engine: Any) -> dict[str, Any]:
        """Auto-pick target eye, run 6-model screening if a new pending photo exists, else return latest report."""
        overview = self.get_patient_overview(phone)
        if overview is None:
            raise KeyError("patient_not_found")
        images: list[dict[str, Any]] = overview["images"]
        if not images:
            return {"has_image": False, "has_new_image": False, "patient": overview["patient"], "eye": eye}

        chosen_eye = eye
        if not chosen_eye:
            pending = [im for im in images if im["status"] == "pending"]
            chosen_eye = pending[0]["eye"] if pending else images[0]["eye"]

        eye_imgs = [im for im in images if im["eye"] == chosen_eye]
        if not eye_imgs:
            return {"has_image": False, "has_new_image": False, "patient": overview["patient"], "eye": chosen_eye}

        latest_img = eye_imgs[0]
        if latest_img["status"] == "analyzed" and latest_img["latest_report_id"]:
            report_row = self.get_report(latest_img["latest_report_id"])
            return {
                "has_image": True,
                "has_new_image": False,
                "patient": overview["patient"],
                "eye": chosen_eye,
                "report": report_row,
            }

        # Execute real 6-model + lesion localization inference for pending hospital image
        if engine is None or not engine.is_ready:
            raise RuntimeError("推理模型尚未就绪，无法对院内待分析影像执行自动推理。")
        img_path = self.images_dir / latest_img["image_relpath"]
        report_id = f"rep_{uuid.uuid4().hex[:12]}"
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            screening = engine.screen(rgb, request_id=report_id, image_sha256=latest_img["image_sha256"])
            screening = FundusResponse.model_validate(screening).model_dump(exclude_unset=True)
            rep_dir = self.reports_dir / report_id
            rep_dir.mkdir(parents=True, exist_ok=True)
            rgb.save(rep_dir / "original.jpg", format="JPEG", quality=95, subsampling=0)
            if _image_readable(screening):
                overlay, comp_info = _comparison(rgb, screening, latest_img["image_sha256"])
            else:
                overlay, comp_info = None, _quality_blocked_info()
            has_overlay = overlay is not None
            if overlay is not None:
                overlay.save(rep_dir / "overlay.jpg", format="JPEG", quality=95, subsampling=0)
            screening["visual_report"] = {
                **comp_info,
                **self._report_urls(report_id, has_overlay),
            }
            if screening.get("report_summary") is not None:
                screening["report_summary"]["clinical_review"] = build_clinical_review(
                    screening, screening.get("visual_report")
                )

        care_plan = _evaluate_report_care(screening, overview["patient"], chosen_eye)
        dr_grade = ((screening.get("report_summary") or {}).get("dr_severity") or {}).get("grade")

        with self._lock:
            conn = self._conn()
            prev_row = conn.execute(
                "SELECT * FROM his_reports WHERE phone = ? AND eye = ? ORDER BY analyzed_utc DESC LIMIT 1",
                (phone, chosen_eye),
            ).fetchone()
            comparison = _compute_eye_comparison(dict(prev_row) if prev_row else None, screening)
            now = _utc_now()
            conn.execute(
                "INSERT INTO his_reports (report_id, image_id, phone, eye, analyzed_utc, dr_grade, care_priority, "
                "screening_json, care_plan_json, comparison_json, has_overlay) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    report_id,
                    latest_img["image_id"],
                    phone,
                    chosen_eye,
                    now,
                    dr_grade if type(dr_grade) is int else None,
                    care_plan["care_priority"],
                    json.dumps(screening, ensure_ascii=False),
                    json.dumps(care_plan, ensure_ascii=False),
                    json.dumps(comparison, ensure_ascii=False) if comparison else None,
                    1 if has_overlay else 0,
                ),
            )
            conn.execute(
                "UPDATE his_images SET status = 'analyzed', latest_report_id = ? WHERE image_id = ?",
                (report_id, latest_img["image_id"]),
            )
            conn.commit()

        return {
            "has_image": True,
            "has_new_image": True,
            "patient": overview["patient"],
            "eye": chosen_eye,
            "report": self.get_report(report_id),
        }

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", report_id):
            return None
        with self._lock:
            conn = self._conn()
            row = conn.execute("SELECT * FROM his_reports WHERE report_id = ?", (report_id,)).fetchone()
            if row is None:
                return None
            p = conn.execute("SELECT * FROM his_patients WHERE phone = ?", (row["phone"],)).fetchone()
        data = dict(row)
        data["patient"] = dict(p) if p else {}
        data["screening"] = json.loads(data.pop("screening_json"))
        data["care_plan"] = _read_report_care(row, data["patient"], data["screening"])
        del data["care_plan_json"]
        data["care_priority"] = data["care_plan"]["care_priority"]
        comp_raw = data.pop("comparison_json")
        data["comparison"] = json.loads(comp_raw) if comp_raw else None
        data["urls"] = self._report_urls(report_id, bool(data["has_overlay"]))
        return data

    def read_report_media(self, report_id: str, filename: str) -> bytes | None:
        if filename not in {"original.jpg", "overlay.jpg"} or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", report_id):
            return None
        path = self.reports_dir / report_id / filename
        if not path.is_file():
            return None
        return path.read_bytes()

    def build_patient_reply(self, result: dict[str, Any]) -> str:
        patient = result["patient"]
        name = patient["name"]
        eye_cn = EYE_CN.get(result.get("eye") or "", "眼睛")
        if not result.get("has_image"):
            return (
                f"{name}您好，医院系统里暂时还没有同步您的{eye_cn}眼底检查照片。"
                "您拍完眼底照后，医院系统会自动上传到这里，到时候再来问我就好！"
            )

        rep = result["report"]
        scr = rep["screening"]
        care = rep["care_plan"]
        comp = rep.get("comparison")
        url = rep["urls"]["patient_report_url"]
        _, reached, dr_grade = _extract_predictions(scr)
        decision = care.get("clinical_decision") or {}
        level = decision.get("level")
        plan = " · ".join(filter(None, (decision.get("destination"), care["followup_time"])))

        if result.get("has_new_image"):
            opener = f"{name}您好，发现医院刚刚上传了您{eye_cn}的最新检查照片，我已经帮您仔细分析好啦！"
        else:
            opener = f"{name}您好，医院暂时还没有上传新的{eye_cn}照片，我帮您调出了最近一次的检查报告："

        if care["care_priority"] == "urgent_ophthalmology":
            body = "您描述的症状需要立即评估，请现在前往眼科急诊。"
        elif care["care_priority"] == "retake_image":
            body = (
                f"这张{eye_cn}照片暂时无法可靠判读，建议请拍片技师帮您重新拍一张清晰的眼底照。"
            )
        elif level == "L1":
            body = (
                f"这次{eye_cn}筛查没有发现明显的糖尿病视网膜病变，各项筛查指标均未达线。"
                f"建议安排 **{plan}**。"
            )
        elif level in {"L3", "L4"}:
            next_step = "请尽快联系专科安排评估" if level == "L4" else "请预约眼科专科复核"
            body = (
                f"这次{eye_cn}检查有需要眼科医生复核的线索。"
                f"{next_step}。建议安排 **{plan}**，就诊时请带上这份解读单。"
            )
        elif level == "L2":
            findings = _friendly_labels(sorted(reached - {"ABNORMAL"}))
            if dr_grade == 1 and "DR" not in reached:
                findings.insert(0, "轻度糖尿病相关的眼底变化线索")
            detail = "、".join(findings) or "轻度眼底变化线索"
            body = f"这次{eye_cn}检查提示{detail}。建议安排 **{plan}**。"
        else:
            detail = "已完成的糖网分级没有发现明显的糖尿病视网膜病变。" if dr_grade == 0 and "DR" not in reached else ""
            body = f"{detail}本次{eye_cn}筛查仍需补充评估。建议安排 **{plan}**。"

        comp_text = f"\n\n{comp['patient_note']}" if comp else ""
        safety_note = decision.get("safety_note") if level in {"L3", "L4"} else None
        safety_text = f"\n\n{safety_note}" if safety_note else ""
        return f"{opener}\n\n{body}{safety_text}{comp_text}\n\n您的个人解读报告在这里（页面支持一键保存 PDF）：\n[打开您的{eye_cn}专属健康报告]({url})"

    def build_doctor_reply(self, result: dict[str, Any]) -> str:
        patient = result["patient"]
        eye_cn = EYE_MEDICAL.get(result.get("eye") or "", "眼别待确认")
        if not result.get("has_image"):
            return f"当前接诊患者：**{patient['name']}**（{patient['phone']}）名下暂无该眼别的院内眼底照片，请先通过拍片室系统同步影像。"
        rep = result["report"]
        scr = rep["screening"]
        care = rep["care_plan"]
        comp = rep.get("comparison")
        url = rep["urls"]["doctor_report_url"]
        _, reached, dr_grade = _extract_predictions(scr)
        review = (scr.get("report_summary") or {}).get("clinical_review") or {}
        decision = care.get("clinical_decision") or {}
        decision_title = " · ".join(filter(None, (decision.get("level"), decision.get("title") or care["priority_label"])))
        triggers = "、".join(item["name"] for item in decision.get("trigger_items", []))
        lesion_legend = ((scr.get("visual_report") or {}).get("legend") or [])
        lesion_str = "、".join(item["name"] for item in lesion_legend if item.get("name")) or "未见显著四类病灶候选区"
        reached_str = "、".join(f"{LABELS.get(c, (c,))[0]}({c})" for c in sorted(reached)) or "暂无达到研究筛查线的记录"
        prefix = "已完成院内新同步影像实时分析" if result.get("has_new_image") else "已为您调阅最新归档临床辅助报告"
        comp_line = f"\n- **同眼纵向对照**：{comp['doctor_note']}" if comp else ""
        trigger_line = f"\n- **转诊触发项**：{triggers}" if triggers else ""
        references = "、".join(care.get("references") or [])
        reference_line = f"\n- **参考依据**：{references}" if references else ""
        return (
            f"{prefix}：**{patient['name']}**（{patient['age']}岁，{patient['chronic_history'] or '未记特殊慢病'}）— {eye_cn}\n"
            f"- **辅助分期与筛查结论**：DR {dr_grade if dr_grade is not None else '--'} 级；{review.get('primary', '')}\n"
            f"- **达到研究筛查线项目**：{reached_str}\n"
            f"- **四类病灶定位**：{lesion_str}\n"
            f"- **专科决策与复核建议**：{decision_title}（{care['followup_time']}）。{trigger_line}{reference_line}{comp_line}\n\n"
            f"临床辅助诊断与病灶分析报告（支持导出 A4 PDF）：\n[打开{patient['name']}{eye_cn}临床辅助报告]({url})"
        )

    def render_patient_report_html(self, report_id: str) -> str | None:
        rep = self.get_report(report_id)
        if rep is None:
            return None
        patient = rep["patient"]
        scr = rep["screening"]
        care = rep["care_plan"]
        comp = rep.get("comparison")
        eye_cn = EYE_CN.get(rep["eye"], "眼底")
        _, reached, _ = _extract_predictions(scr)
        decision = care.get("clinical_decision") or {}
        level = decision.get("level")
        destination = decision.get("destination") or ""
        destination_html = f"<p><strong>建议去向</strong> {escape(destination)}</p>" if destination else ""
        safety_note = decision.get("safety_note") or ""

        if care["care_priority"] == "urgent_ophthalmology":
            headline = "请立即前往眼科急诊"
            tone_cls = "tone-urgent"
            comfort = "请告知医生症状开始的时间和受影响的眼睛。"
        elif care["care_priority"] == "retake_image":
            headline = "照片需要重拍或由医生确认"
            tone_cls = "tone-info"
            comfort = "目前这张照片无法可靠判读。请按建议补充清晰的眼底照。"
        elif level == "L1":
            headline = "目前这只眼未见明显糖网病变，整体状态平稳"
            tone_cls = "tone-ok"
            comfort = "本次各项筛查指标均未达线，请按建议安排年度眼科检查。"
        elif level in {"L3", "L4"}:
            headline = "请尽快联系专科安排评估" if level == "L4" else "请预约眼科医生复核"
            tone_cls = "tone-urgent" if level == "L4" else "tone-warn"
            comfort = "请携带这份解读单和既往眼科检查资料，由医生结合面诊评估。"
        elif level == "L2":
            headline = "眼底变化建议按期复查"
            tone_cls = "tone-mild"
            comfort = "本次筛查有轻度眼底变化线索，请按建议时间复查。"
        else:
            headline = "本次筛查仍需补充评估"
            tone_cls = "tone-info"
            comfort = "请按下方建议补充检查，由眼科医生结合完整结果评估。"

        friendly_list = _friendly_labels(sorted(reached - {"ABNORMAL"}))
        if care["care_priority"] == "retake_image":
            findings_html = "<p>请先补充可判读的眼底照片。</p>"
        elif friendly_list:
            findings_html = "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in friendly_list) + "</ul>"
        elif level == "L1":
            findings_html = ""
        else:
            findings_html = "<p>具体发现请结合眼科复核结果。</p>"
        findings_section = f'<section class="card"><h2>眼底筛查发现</h2>{findings_html}</section>' if findings_html else ""
        comp_html = f"<section class='card'><h2>和上次检查对比</h2><p>{escape(comp['patient_note'])}</p></section>" if comp else ""
        overlay_fig = (
            f"<figure><img src='overlay.jpg' alt='重点观察示意图'><figcaption>机器重点帮您标出来的局部细节位置</figcaption></figure>"
            if rep["has_overlay"] else ""
        )

        return f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(patient.get('name', '患者'))}的个人眼健康解读单</title>
<style>
  @page{{size:A4 portrait;margin:14mm}}
  *{{box-sizing:border-box}}
  body{{margin:0;padding:24px;background:#f7f9fc;color:#1e293b;font:15px/1.75 "PingFang SC","Microsoft YaHei",sans-serif}}
  .wrap{{max-width:860px;margin:0 auto}}
  .toolbar{{display:flex;justify-content:space-between;align-items:center;background:#fff;border:1px solid #dbe4f0;border-radius:10px;padding:12px 18px;margin-bottom:16px}}
  .btn{{border:none;background:#0f4c81;color:#fff;padding:9px 18px;border-radius:7px;font-weight:700;cursor:pointer}}
  .hero{{background:#fff;border:1px solid #dbe4f0;border-radius:12px;padding:22px 24px;margin-bottom:14px}}
  .hero.tone-ok{{border-left:6px solid #16a34a;background:#f0fdf4}}
  .hero.tone-urgent{{border-left:6px solid #b91c1c;background:#fef2f2}}
  .hero.tone-warn{{border-left:6px solid #c2410c;background:#fff7ed}}
  .hero.tone-mild{{border-left:6px solid #ca8a04;background:#fffbeb}}
  .hero.tone-info{{border-left:6px solid #0284c7;background:#f0f9ff}}
{CARE_LEVEL_STYLE}
  h1{{margin:0 0 8px;font-size:22px;color:#0f172a}}
  .meta{{font-size:13.5px;color:#475569;display:flex;flex-wrap:wrap;gap:14px}}
  .card{{background:#fff;border:1px solid #dbe4f0;border-radius:12px;padding:18px 22px;margin-bottom:14px}}
  .card h2{{margin:0 0 10px;font-size:16.5px;color:#0f4c81}}
  .imgs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}}
  figure{{margin:0;background:#0f172a;border-radius:8px;padding:8px;color:#e2e8f0}}
  figure img{{width:100%;border-radius:4px;display:block}}
  figcaption{{font-size:12.5px;margin-top:6px;text-align:center}}
  @media print{{body{{padding:0;background:#fff}}.no-print{{display:none}}}}
</style>
<div class="wrap">
  <div class="toolbar no-print">
    <span><strong>患者版通俗解读单</strong> · 永久关联档案（可随时回来打印）</span>
    <button class="btn" type="button" onclick="window.print()">导出 / 打印标准 A4 PDF</button>
  </div>
  <header class="hero {tone_cls}">
    <h1>{escape(patient.get('name', ''))}先生/女士的{eye_cn}眼健康解读单</h1>
    <div class="meta">
      <span>档案手机号：{escape(patient.get('phone', ''))}</span>
      <span>检查眼别：{eye_cn}</span>
      <span>分析时间：{escape(rep['analyzed_utc'][:10])}</span>
      <span>报告号：{escape(rep['report_id'])}</span>
    </div>
    <p style="margin:12px 0 0;font-weight:700;font-size:16px;">{_decision_badge(level)} {escape(headline)}</p>
    <p style="margin:6px 0 0;">{escape(comfort)}</p>
  </header>
  {findings_section}
  {comp_html}
  <section class="card">
    <h2>给您的贴心行动建议</h2>
    <p><strong>建议时限</strong> {escape(care['followup_time'])}</p>
    {destination_html}
    <ul>{"".join(f"<li>{escape(a)}</li>" for a in care.get("actions", []))}</ul>
    <p style="font-size:13px;color:#64748b;margin-top:10px;">{escape(safety_note)}</p>
  </section>
  <section class="card">
    <h2>医院同步的眼底照片</h2>
    <div class="imgs">
      <figure><img src="original.jpg" alt="本次眼底原图"><figcaption>您的{eye_cn}清晰眼底照片</figcaption></figure>
      {overlay_fig}
    </div>
  </section>
</div>
</html>"""

    def render_doctor_report_html(self, report_id: str, *, report_explanation_label: str | None = None) -> str | None:
        report = self.get_report(report_id)
        if report is None:
            return None
        screening = report["screening"]
        folder = self.reports_dir / report_id
        has_overlay = bool(report["has_overlay"] and (folder / "overlay.jpg").is_file())
        return _report_html(
            screening.get("visual_report") if has_overlay or not report["has_overlay"] else None,
            screening,
            has_overlay=has_overlay,
            hospital_record={**report, "has_original": (folder / "original.jpg").is_file()},
            report_explanation_label=report_explanation_label,
        )
