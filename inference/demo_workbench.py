"""Integrated web demo workbench (/v1/demo) showcasing multi-agent routing, audience adaptation, and A4 PDF reports."""
from __future__ import annotations

import base64
import binascii
from html import escape
from pathlib import Path
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from inference.clinical_review import build_clinical_review
from inference.clinical_workflow import (
    CompareRequest,
    CareEvaluateRequest,
    _coerce_str_list,
    _detect_urgent_symptoms,
)
from inference.image_io import InputImageError, URLFetchPolicy, decode_image_bytes, fetch_and_decode_image
from inference.schemas import FundusResponse
from inference.visual_report import EYE_LABELS

ROOT = Path(__file__).resolve().parents[1]

DEMO_SAMPLES: dict[str, dict[str, Any]] = {
    "idrid_03": {
        "id": "idrid_03",
        "title": "样例一：糖尿病视网膜病变示例（左眼）",
        "subtitle": "典型微动脉瘤、视网膜出血与渗出病灶特征，适合体验四类病灶定位与专科随访建议",
        "path": ROOT / "examples" / "IDRiD_03.jpg",
        "default_eye": "left",
        "default_symptoms": "近期偶觉视物轻度模糊",
        "default_history": "2型糖尿病8年",
    },
    "public_proxy": {
        "id": "public_proxy",
        "title": "样例二：常规眼健康体检对照图（右眼）",
        "subtitle": "适合与左眼配合演示双图左右眼防错位建档、常规年度随访及症状优先冲突测试",
        "path": ROOT / "inference_runtime" / "local_gpu" / "retina_public_sample_proxy.jpg",
        "default_eye": "right",
        "default_symptoms": "无明显视力下降，无飞蚊闪光",
        "default_history": "年度常规体检",
    },
}


class DemoRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["sample", "url", "upload", "no_image"] = "sample"
    sample_id: str | None = Field(default="idrid_03", max_length=64)
    image_url: str | None = Field(default=None, max_length=8192)
    image_base64: str | None = Field(default=None, max_length=16 * 1024 * 1024)
    audience: Literal["patient", "caregiver", "clinician"] = "patient"
    eye: Literal["left", "right", "unknown"] = "unknown"
    symptoms: str = Field(default="", max_length=512)
    history: str = Field(default="", max_length=512)
    query_text: str = Field(default="", max_length=1000)
    session_id: str = Field(default="demo_workbench", min_length=1, max_length=64)


def get_sample_image_bytes(sample_id: str) -> bytes | None:
    meta = DEMO_SAMPLES.get(sample_id)
    if not meta:
        return None
    path: Path = meta["path"]
    if not path.is_file():
        return None
    return path.read_bytes()


def _decode_base64_image(data_uri_or_b64: str, policy: URLFetchPolicy):
    text = (data_uri_or_b64 or "").strip()
    if "," in text and text.lower().startswith("data:"):
        text = text.split(",", 1)[1]
    if not text:
        raise InputImageError("invalid_image", "未检测到有效的上传图像内容。")
    try:
        raw = base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise InputImageError("invalid_image", "上传的图像数据格式无效，请重试。") from exc
    if len(raw) > policy.max_download_bytes:
        raise InputImageError("image_too_large", "上传图像超过允许大小限制。")
    return decode_image_bytes(raw, policy=policy)


def _audience_narratives(evidence: dict[str, Any], care: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build deterministic three-audience explanations strictly anchored to verified evidence."""
    quality = (evidence.get("quality") or {}).get("predicted_label") or "未知"
    dr = evidence.get("dr_grading") or {}
    dr_grade = dr.get("grade")
    dr_stage = dr.get("clinical_stage") or "未取得有效糖网分期"
    eye_cn = EYE_LABELS.get(evidence.get("eye") or "unknown", "眼别待确认")
    preds = evidence.get("predictions") or []
    reached = [p for p in preds if p.get("screening_flag") == "positive"]
    reached_names = [f"{p['label_name']}（{p['label_code']}）" for p in reached]
    reached_codes = {p.get("label_code") for p in reached}
    lesions = (evidence.get("lesion_localization") or {}).get("legend") or []
    lesion_names = [item.get("name") for item in lesions if item.get("name")]

    benign_notes = []
    if "TSLN" in reached_codes:
        benign_notes.append(
            "豹纹状眼底（TSLN）在近视及中老年人群中十分常见，主要反映眼底血管纹理透见，并不直接代表严重病变或推导具体近视度数 [SRC_EDU_CHECKUP_BENIGN]。"
        )
    if "DN" in reached_codes:
        benign_notes.append(
            "玻璃膜疣（DN）属于视网膜深层沉着体征，不直接等同于晚期年龄相关性黄斑变性（AMD），建议通过黄斑 OCT 确认稳定状态 [SRC_DICT_DN_AMD]。"
        )
    if "LS" in reached_codes:
        benign_notes.append(
            "激光瘢痕（LS）通常对应既往眼底光凝治疗留下的陈旧斑痕，不属于新发眼病，请重点核对既往治疗史 [SRC_DICT_LS]。"
        )
    if "OPTIC_DISC_CUPPING" in reached_codes or "glaucoma" in reached_codes:
        benign_notes.append(
            "视盘凹陷或青光眼筛查线索属于形态与研究提示，部分人群可为生理性大视杯，最终需眼科完善眼压、视野及视神经纤维层 OCT 确认 [SRC_DICT_ODC_GLAUCOMA]。"
        )

    patient_points = [
        f"本次核对眼别为「{eye_cn}」，图像质控判定为「{quality}」。",
        f"糖尿病视网膜病变专项模型评估结果为：{dr_grade} 级（{dr_stage}）。" if dr_grade is not None else "本次未得出有效的糖网分级结论。",
        (
            "模型覆盖范围内达到研究筛查线的项目有：" + "、".join(reached_names) + "。"
            if reached_names
            else "在当前多标签研究模型覆盖范围内，未见达到筛查线的额外异常提示（模型分数仅为特征提示强度，不代表患病概率百分比）[SRC_DICT_OVERVIEW]。"
        ),
    ]
    if lesion_names:
        patient_points.append("辅助定位模型在图上标出了可能值得医生对照原图核查的候选区域：" + "、".join(lesion_names) + "。")
    patient_points.extend(benign_notes)

    caregiver_points = [
        f"就医紧迫度：{care['priority_label']}（建议时间窗：{care['followup_time']}）。",
        "建议陪诊时携带：本次导出的标准 A4 临床报告（含眼底原图与病灶定位图）、近 3 个月血糖血压监测记录、既往眼科光凝或手术病历及日常用药清单 [SRC_EDU_CAREGIVER_GUIDE]。",
        "居家重点观察以下警示红线：若突然出现大量飞蚊黑影、持续闪光感、黑幕样视野遮挡、突发视力骤降或剧烈眼痛，须立即前往眼科急诊 [SRC_NEI_RD_URGENT]。",
    ]
    if care.get("questions_to_confirm"):
        caregiver_points.append("待补充确认事项：" + "；".join(care["questions_to_confirm"]))

    clinician_rows = []
    if dr_grade is not None:
        clinician_rows.append(
            {
                "item": "糖尿病视网膜病变分期",
                "model": "糖网五级专病模型",
                "score": dr.get("score_display") or "--",
                "threshold": "0~4 级分类",
                "status": f"{dr_grade} 级 · {dr_stage}",
            }
        )
    for pred in preds:
        raw = pred.get("raw_score")
        thr = pred.get("threshold")
        clinician_rows.append(
            {
                "item": f"{pred['label_name']} ({pred['label_code']}) [{pred['concept_type']}]",
                "model": pred["source_model"],
                "score": f"{raw * 100:.3f} 分" if isinstance(raw, (int, float)) else "--",
                "threshold": f"{thr * 100:.4f} 分" if isinstance(thr, (int, float)) else "--",
                "status": "达到研究筛查线" if pred["screening_flag"] == "positive" else ("未达线" if pred["screening_flag"] == "negative" else "未评估"),
            }
        )

    clinician_summary = (evidence.get("clinical_review") or {}).get("primary") or ""
    if lesion_names:
        clinician_summary += " 四类病灶分割提示关注区域：" + "、".join(lesion_names) + "。"
    clinician_summary += " 触发随访规则：" + "、".join(care.get("rule_ids") or []) + "。"

    return {
        "patient": {
            "title": "受检者通俗定心解读（大众体检与初筛视角）",
            "lead": f"{care['priority_label']} — {care['followup_time']}",
            "points": patient_points,
            "sources": care.get("references") or ["SRC_DICT_OVERVIEW"],
        },
        "caregiver": {
            "title": "家属陪诊与慢病照护清单（子女/家属视角）",
            "lead": f"优先安排：{care['priority_label']}（规则：{', '.join(care.get('rule_ids') or [])}）",
            "points": caregiver_points + [f"诊前向医生重点提问：{q}" for q in (care.get("questions_for_clinician") or [])],
            "sources": sorted(set((care.get("references") or []) + ["SRC_EDU_CAREGIVER_GUIDE"])),
        },
        "clinician": {
            "title": "基层全科与体检医生阅片明细（专业核对视角）",
            "lead": clinician_summary,
            "points": [item["topic"] + "：" + item["basis"] + " 建议核查：" + item["action"]
                       for item in ((evidence.get("clinical_review") or {}).get("review_items") or [])]
                      or ["当前模型未触发额外高危单项复核条目，仍请结合裂隙灯与眼底原图综合阅片。"],
            "table_rows": clinician_rows,
            "sources": care.get("references") or ["SRC_SCREEN_SCOPE"],
        },
    }


def _no_image_consultation(req: DemoRunRequest) -> dict[str, Any]:
    """Handle symptom-first urgent triage and paper checkup Q&A without an eye photo."""
    t0 = time.monotonic()
    symptom_list = _coerce_str_list(f"{req.symptoms} {req.query_text}")
    urgent_hits = _detect_urgent_symptoms(symptom_list)
    combined_text = f"{req.symptoms} {req.history} {req.query_text}"

    trace = [
        {
            "agent": "A1 主控导诊智能体",
            "stage": "入口识别与分流",
            "status": "completed",
            "elapsed_ms": 2.0,
            "detail": f"识别为「无眼底照片纯文本咨询」，受众模式={req.audience}；抢先执行症状红线扫描。",
        }
    ]

    if urgent_hits:
        priority_label = "立即眼科急诊评估（无图优先拦截）"
        care_priority = "urgent_ophthalmology"
        followup = "立即（当日尽快前往具备眼底检查条件的眼科急诊）"
        basis = (
            "已触发紧急症状优先规则 [RULE_URGENT_SYMPTOM_01]：" + "；".join(urgent_hits) + "。"
            "在尚未拍摄或上传眼底照的情况下，出现急性视网膜警示表现仍需第一时间线下急诊排查视网膜裂孔、脱离或眼底急性出血 [SRC_NEI_RD_URGENT]。"
        )
        actions = [
            "请立即停止剧烈活动、避免揉眼或低头憋气，当日尽快到眼科急诊散瞳查眼底与眼部 B 超。",
            "就诊时告知医生具体起始时间及症状出现在左眼、右眼或双眼。",
        ]
        rule_ids = ["RULE_URGENT_SYMPTOM_01"]
        sources = ["SRC_NEI_RD_URGENT", "SRC_EDU_NO_IMAGE_TRIAGE"]
        trace.append({
            "agent": "A3 随访与急症分诊 (W30)",
            "stage": "紧急症状抢先拦截",
            "status": "urgent",
            "elapsed_ms": round((time.monotonic() - t0) * 1000, 2),
            "detail": f"命中紧急红线：{'；'.join(urgent_hits)}。阻断普通科普流程，直接置顶急诊建议 [RULE_URGENT_SYMPTOM_01]。",
        })
    else:
        priority_label = "无图科普解答与规范补检引导"
        care_priority = "no_image_guidance"
        followup = "无急性症状时建议上传清晰眼底照片以开启六模型筛查，或按体检建议到眼科完善检查"
        basis = "当前未输入眼底图像，系统依据标签字典与大众科普知识库回答体检名词疑问；严禁在未见图片的情况下作出“眼底正常/未患病”结论 [SRC_EDU_NO_IMAGE_TRIAGE]。"
        actions = [
            "如手头有单位体检或医院拍摄的眼底彩色照，可切换至“样例/上传模式”获取六模型量化与病灶定位分析。",
            "常见纸质体检名词提示：豹纹状眼底（TSLN）多为良性结构纹理；玻璃膜疣（DN）与杯盘比偏大建议分别结合黄斑 OCT 或眼压/视野进一步核实。",
        ]
        rule_ids = ["RULE_NO_IMAGE_BOUNDARY_00"]
        sources = ["SRC_EDU_NO_IMAGE_TRIAGE", "SRC_EDU_CHECKUP_BENIGN"]
        if any(k in combined_text for k in ("糖尿", "高血压", "血糖", "胰岛素")):
            sources.append("SRC_EDU_CHRONIC_CARE")
            actions.append("糖尿病/高血压慢病患者建议每 6~12 个月规律检查眼底，并携带近 3 月血糖血压记录就诊 [SRC_EDU_CHRONIC_CARE]。")
        trace.append({
            "agent": "A2 医学解读 + A4 大众科普 (W20/W40)",
            "stage": "知识库检索与科普应答",
            "status": "completed",
            "elapsed_ms": round((time.monotonic() - t0) * 1000, 2),
            "detail": f"已检索知识来源 {', '.join(sources)}，严格执行无图不下阴性保证的医学边界 [SRC_EDU_NO_IMAGE_TRIAGE]。",
        })

    narratives = {
        "patient": {
            "title": "受检者无图咨询与体检词答疑",
            "lead": f"{priority_label} — {followup}",
            "points": [basis, *actions],
            "sources": sources,
        },
        "caregiver": {
            "title": "家属陪诊与无图红线筛查指引",
            "lead": f"{priority_label}",
            "points": [basis, *actions, "如代父母咨询，建议提前拍照留存原检查单、眼底照与近 3 月血糖血压记录，避免到院重复检查 [SRC_EDU_CAREGIVER_GUIDE]。"],
            "sources": sorted(set(sources + ["SRC_EDU_CAREGIVER_GUIDE"])),
        },
        "clinician": {
            "title": "基层预分诊与转诊指征参考（未见图像）",
            "lead": basis,
            "points": actions,
            "table_rows": [],
            "sources": sources,
        },
    }

    return {
        "ok": True,
        "mode": "no_image",
        "audience": req.audience,
        "assessment_id": None,
        "care_priority": care_priority,
        "priority_label": priority_label,
        "followup_time": followup,
        "rule_ids": rule_ids,
        "source_ids": sources,
        "narratives": narratives,
        "trace": trace,
    }


def run_demo_workflow(
    req: DemoRunRequest,
    *,
    engine: Any,
    workflow: Any,
    delivery: Any,
    policy: URLFetchPolicy,
    fetcher: Any = None,
) -> dict[str, Any]:
    """Execute the end-to-end multi-agent simulation pipeline using live engine & SQLite store."""
    if req.mode == "no_image":
        return _no_image_consultation(req)

    trace: list[dict[str, Any]] = []
    t_start = time.monotonic()
    trace.append({
        "agent": "A1 主控导诊智能体",
        "stage": "入口识别与原子建档准备",
        "status": "completed",
        "elapsed_ms": 1.5,
        "detail": (
            f"识别模式={req.mode}，受众视角={req.audience}，原子绑定参数 eye={req.eye}，"
            "按防错位规范直接随图像一次性写入，绝不并发竞态写入。"
        ),
    })

    decoded = None
    try:
        if req.mode == "sample":
            raw = get_sample_image_bytes(req.sample_id or "idrid_03")
            if raw is None:
                raise InputImageError("invalid_image", f"未找到预置示例图像：{req.sample_id}")
            decoded = decode_image_bytes(raw, policy=policy)
        elif req.mode == "upload":
            decoded = _decode_base64_image(req.image_base64 or "", policy=policy)
        elif req.mode == "url":
            if not (req.image_url or "").strip():
                raise InputImageError("invalid_image", "请输入有效的 HTTPS 眼底照片地址。")
            decoded = (fetcher or fetch_and_decode_image)(req.image_url.strip(), policy=policy)
        else:
            raise InputImageError("invalid_image", "不支持的演示模式。")

        import uuid
        rid = uuid.uuid4().hex
        t_infer = time.monotonic()
        response = engine.screen(decoded.image, request_id=rid, image_sha256=decoded.sha256)
        response["warnings"].extend(decoded.warnings)
        response = FundusResponse.model_validate(response).model_dump(exclude_unset=True)
        infer_ms = round((time.monotonic() - t_infer) * 1000, 1)

        if delivery is not None and response["status"] in {"ok", "partial"}:
            response["visual_report"] = delivery.create(decoded.image, response, decoded.sha256, rid)
        if response.get("report_summary") is not None:
            response["report_summary"]["clinical_review"] = build_clinical_review(
                response, response.get("visual_report")
            )

        trace.append({
            "agent": "W10 视觉筛选工作流",
            "stage": "六模型推理与四类病灶定位",
            "status": response["status"],
            "elapsed_ms": infer_ms,
            "detail": (
                f"完成图像质控与专家模型推理，request_id / assessment_id = {rid}，"
                f"病灶状态={(response.get('visual_report') or {}).get('localization_status', 'unavailable')}。"
            ),
        })

        t_wf = time.monotonic()
        combined_symptoms = ", ".join(p for p in (req.symptoms.strip(), req.query_text.strip()) if p)
        if response["status"] in {"ok", "partial", "quality_rejected"}:
            workflow.record_screening(
                response,
                session_id=req.session_id,
                eye=req.eye,
                symptoms=combined_symptoms,
                history=req.history,
            )
            care_plan = workflow.evaluate_care(
                CareEvaluateRequest(assessment_id=rid, session_id=req.session_id)
            )
            assessment = workflow.get_assessment(rid)
            evidence = (assessment or {}).get("evidence_packet") or {}
        else:
            raise InputImageError("inference_rejected", response.get("message") or "分析未完成")

        narratives = _audience_narratives(evidence, care_plan)
        trace.append({
            "agent": "A2 医学解读问答 (W20)",
            "stage": "冻结证据包与三层受众解读",
            "status": "completed",
            "elapsed_ms": round((time.monotonic() - t_wf) * 1000, 2),
            "detail": (
                f"读取 evidence.v1 ({rid})，引用来源 {', '.join(care_plan.get('references') or [])}，"
                "零改动模型原值，完成患者/家属/医生三视角自适应文本装配。"
            ),
        })

        trace.append({
            "agent": "A3 随访与急症分诊 (W30)",
            "stage": "确定性规则引擎与优先级裁决",
            "status": "urgent" if care_plan["care_priority"] == "urgent_ophthalmology" else "completed",
            "elapsed_ms": 1.8,
            "detail": (
                f"命中规则 {', '.join(care_plan.get('rule_ids') or [])} -> {care_plan['priority_label']}；"
                f"缺失问询项：{', '.join(care_plan.get('missing_facts') or ['无'])}。"
            ),
        })

        context = workflow.get_context(session_id=req.session_id)
        recent = context.get("recent_assessments") or []
        comparison_preview = None
        same_eye_prev = next(
            (r for r in recent if r["assessment_id"] != rid and r["eye"] == req.eye and req.eye in ("left", "right")),
            None,
        )
        if same_eye_prev is not None:
            comparison_preview = workflow.compare_assessments(
                CompareRequest(previous_assessment_id=same_eye_prev["assessment_id"], current_assessment_id=rid)
            )
            trace.append({
                "agent": "A4 历史对比与大众科普 (W40/W60)",
                "stage": "同眼可比性自动校验与对照",
                "status": "completed",
                "elapsed_ms": 1.2,
                "detail": f"检测到既往同眼记录 {same_eye_prev['assessment_id'][:10]}...，已完成可比性与模型差异汇总。",
            })
        else:
            trace.append({
                "agent": "A4 历史对比与大众科普 (W40/W60)",
                "stage": "科普边界与双眼建档索引",
                "status": "completed",
                "elapsed_ms": 0.9,
                "detail": f"当前会话已入库 {len(recent)} 条检查记录；支持与同眼复查对比或左右眼对照展示。",
            })

        visual = evidence.get("visual_report") or {}
        # On local HTTP (when delivery paths are https://... or absent), relative /v1/view/... keeps previews working everywhere.
        def _relative_view_url(url: str | None) -> str | None:
            if not url:
                return None
            marker = "/v1/view/"
            idx = url.find(marker)
            return url[idx:] if idx >= 0 else url

        return {
            "ok": True,
            "mode": req.mode,
            "audience": req.audience,
            "total_elapsed_ms": round((time.monotonic() - t_start) * 1000, 1),
            "assessment_id": rid,
            "eye": evidence.get("eye"),
            "eye_label": EYE_LABELS.get(evidence.get("eye") or "unknown", "眼别待确认"),
            "quality_label": (evidence.get("quality") or {}).get("predicted_label"),
            "dr_grading": evidence.get("dr_grading"),
            "clinical_decision": care_plan["clinical_decision"],
            "care_priority": care_plan["care_priority"],
            "priority_label": care_plan["priority_label"],
            "followup_time": care_plan["followup_time"],
            "priority_basis": care_plan["priority_basis"],
            "rule_ids": care_plan.get("rule_ids") or [],
            "source_ids": care_plan.get("references") or [],
            "missing_facts": care_plan.get("missing_facts") or [],
            "questions_to_confirm": care_plan.get("questions_to_confirm") or [],
            "visual_urls": {
                "original": _relative_view_url(visual.get("original_url")),
                "overlay": _relative_view_url(visual.get("overlay_url")),
                "report": _relative_view_url(visual.get("report_url")),
                "report_external": visual.get("report_url"),
            },
            "localization": evidence.get("lesion_localization") or {},
            "narratives": narratives,
            "trace": trace,
            "recent_assessments": recent,
            "comparison_preview": comparison_preview,
        }
    finally:
        if decoded is not None:
            decoded.image.close()


def render_demo_workbench_html(*, ready: bool) -> str:
    """Render the zero-emoji integrated web demo workbench page."""
    status_badge = "六模型与病灶引擎在线就绪" if ready else "推理引擎尚未就绪（可先体验无图分诊流程）"
    status_cls = "status-ready" if ready else "status-warn"

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>眼底筛查与随访多智能体演示工作台</title>
<style>
  :root {{
    --bg: #f3f6fa;
    --card: #ffffff;
    --ink: #0f172a;
    --muted: #475569;
    --line: #dbe3ee;
    --primary: #0f4c81;
    --primary-dark: #0a365c;
    --accent: #0284c7;
    --urgent: #b91c1c;
    --urgent-bg: #fef2f2;
    --warn: #b45309;
    --warn-bg: #fffbeb;
    --ok: #15803d;
    --ok-bg: #f0fdf4;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font: 14px/1.6 "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
  }}
  .topbar {{
    background: linear-gradient(120deg, #0a365c 0%, #0f4c81 65%, #155e75 100%);
    color: #fff;
    padding: 16px 24px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.14);
  }}
  .brand h1 {{ margin: 0; font-size: 19px; font-weight: 700; letter-spacing: 0.3px; }}
  .brand p {{ margin: 2px 0 0; font-size: 12.5px; color: #cbd5e1; }}
  .top-right {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }}
  .status-pill {{
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.28);
    background: rgba(255,255,255,0.12);
  }}
  .status-pill.status-ready {{ background: rgba(21,128,61,0.35); border-color: #86efac; }}
  .status-pill.status-warn {{ background: rgba(180,83,9,0.45); border-color: #fde68a; }}
  .audience-switch {{
    display: inline-flex;
    background: rgba(15, 23, 42, 0.28);
    border-radius: 8px;
    padding: 3px;
    gap: 3px;
  }}
  .audience-btn {{
    border: none;
    background: transparent;
    color: #e2e8f0;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .audience-btn.active {{
    background: #ffffff;
    color: var(--primary-dark);
  }}
  .shell {{
    max-width: 1380px;
    margin: 18px auto 32px;
    padding: 0 18px;
    display: grid;
    grid-template-columns: 380px 1fr;
    gap: 18px;
    align-items: start;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
  }}
  .card + .card {{ margin-top: 14px; }}
  .card-title {{
    margin: 0 0 12px;
    font-size: 15px;
    font-weight: 700;
    color: var(--primary-dark);
    padding-bottom: 8px;
    border-bottom: 1px solid var(--line);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .tabs {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    margin-bottom: 12px;
  }}
  .tab-btn {{
    border: 1px solid var(--line);
    background: #f8fafc;
    color: var(--muted);
    padding: 7px 4px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }}
  .tab-btn.active {{
    background: var(--primary);
    color: #fff;
    border-color: var(--primary);
  }}
  .sample-card {{
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 8px;
    cursor: pointer;
    display: flex;
    gap: 10px;
    align-items: center;
    background: #fafcff;
    transition: border-color 0.15s ease;
  }}
  .sample-card.active {{
    border-color: var(--primary);
    background: #eff6ff;
    box-shadow: inset 0 0 0 1px var(--primary);
  }}
  .sample-thumb {{
    width: 64px;
    height: 48px;
    object-fit: cover;
    border-radius: 5px;
    background: #0f172a;
    flex-shrink: 0;
  }}
  .sample-info strong {{ display: block; font-size: 13px; color: var(--ink); }}
  .sample-info span {{ font-size: 11.5px; color: var(--muted); display: block; line-height: 1.4; }}
  label.field {{
    display: block;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--ink);
    margin: 10px 0 4px;
  }}
  input[type="text"], select, textarea {{
    width: 100%;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px 10px;
    font: inherit;
    font-size: 13px;
    background: #fff;
  }}
  textarea {{ min-height: 64px; resize: vertical; }}
  .quick-scenarios {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
  }}
  .chip-btn {{
    border: 1px solid #cbd5e1;
    background: #f8fafc;
    color: #334155;
    padding: 4px 9px;
    border-radius: 999px;
    font-size: 11.5px;
    cursor: pointer;
  }}
  .chip-btn:hover {{ background: #e2e8f0; }}
  .run-btn {{
    margin-top: 14px;
    width: 100%;
    border: none;
    background: var(--primary);
    color: #fff;
    padding: 11px 16px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
  }}
  .run-btn:hover {{ background: var(--primary-dark); }}
  .run-btn:disabled {{ opacity: 0.6; cursor: wait; }}
  .hero-banner {{
    border-radius: 10px;
    padding: 16px 18px;
    border: 1px solid var(--line);
    background: #ffffff;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }}
  .hero-banner.urgent {{ border-color: #fca5a5; background: var(--urgent-bg); }}
  .hero-banner.warn {{ border-color: #fcd34d; background: var(--warn-bg); }}
  .hero-banner.ok {{ border-color: #86efac; background: var(--ok-bg); }}
  .hero-title {{ font-size: 17px; font-weight: 700; margin: 0 0 4px; }}
  .hero-sub {{ margin: 0; font-size: 13px; color: var(--muted); }}
  .pdf-link {{
    display: inline-block;
    text-decoration: none;
    background: var(--primary);
    color: #fff;
    padding: 9px 16px;
    border-radius: 7px;
    font-weight: 700;
    font-size: 13px;
  }}
  .pdf-link:hover {{ background: var(--primary-dark); }}
  .metrics-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 12px;
  }}
  .metric-box {{
    background: #f8fafc;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 10px 12px;
  }}
  .metric-box .k {{ font-size: 11.5px; color: var(--muted); }}
  .metric-box .v {{ font-size: 14.5px; font-weight: 700; color: var(--ink); margin-top: 2px; }}
  .dual-imgs {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }}
  .img-pane {{
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 8px;
    background: #0f172a;
    color: #e2e8f0;
  }}
  .img-pane img {{ width: 100%; border-radius: 4px; display: block; max-height: 310px; object-fit: contain; }}
  .img-cap {{ font-size: 12px; margin-top: 6px; color: #cbd5e1; }}
  .legend-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; font-size: 12px; color: var(--muted); }}
  .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }}
  .trace-list {{ display: grid; gap: 8px; }}
  .trace-item {{
    border: 1px solid var(--line);
    border-left: 4px solid var(--accent);
    border-radius: 6px;
    padding: 9px 12px;
    background: #f8fafc;
  }}
  .trace-item.urgent {{ border-left-color: var(--urgent); background: #fff5f5; }}
  .trace-head {{ display: flex; justify-content: space-between; font-size: 12.5px; font-weight: 700; color: var(--primary-dark); }}
  .trace-meta {{ font-size: 11.5px; color: var(--muted); }}
  .trace-body {{ font-size: 12.5px; margin-top: 4px; color: var(--ink); }}
  .narrative-list {{ margin: 8px 0 0; padding-left: 18px; }}
  .narrative-list li {{ margin-bottom: 6px; }}
  .src-tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
  .src-tag {{
    font-size: 11.5px;
    background: #eff6ff;
    color: #1e40af;
    border: 1px solid #bfdbfe;
    border-radius: 4px;
    padding: 2px 7px;
    font-family: ui-monospace, Consolas, monospace;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 10px; }}
  th, td {{ border: 1px solid var(--line); padding: 6px 8px; text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  .empty-state {{
    background: #fff;
    border: 1px dashed #cbd5e1;
    border-radius: 10px;
    padding: 40px 24px;
    text-align: center;
    color: var(--muted);
  }}
  .hidden {{ display: none !important; }}
  @media (max-width: 980px) {{
    .shell {{ grid-template-columns: 1fr; }}
    .metrics-strip {{ grid-template-columns: repeat(2, 1fr); }}
    .dual-imgs {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<header class="topbar">
  <div class="brand">
    <h1>眼底筛查与随访多智能体演示工作台</h1>
    <p>六模型专家推理 · 四类病灶定位 · 症状优先规则引擎 · 三层受众自适应解读与 A4 PDF 导出</p>
  </div>
  <div class="top-right">
    <span class="status-pill {status_cls}">{escape(status_badge)}</span>
    <div class="audience-switch" id="audienceSwitch">
      <button type="button" class="audience-btn active" data-audience="patient">大众体检版</button>
      <button type="button" class="audience-btn" data-audience="caregiver">家属陪诊版</button>
      <button type="button" class="audience-btn" data-audience="clinician">基层医生版</button>
    </div>
  </div>
</header>

<main class="shell">
  <section>
    <div class="card">
      <h2 class="card-title">1. 检查输入与场景演练配置</h2>
      <div class="tabs" id="modeTabs">
        <button type="button" class="tab-btn active" data-mode="sample">预置样例</button>
        <button type="button" class="tab-btn" data-mode="upload">本地上传</button>
        <button type="button" class="tab-btn" data-mode="url">HTTPS 链接</button>
        <button type="button" class="tab-btn" data-mode="no_image">无图咨询</button>
      </div>

      <div id="pane-sample">
        <div class="sample-card active" data-sample="idrid_03">
          <img class="sample-thumb" src="/v1/demo/sample/idrid_03" alt="IDRiD_03">
          <div class="sample-info">
            <strong>{escape(DEMO_SAMPLES['idrid_03']['title'])}</strong>
            <span>{escape(DEMO_SAMPLES['idrid_03']['subtitle'])}</span>
          </div>
        </div>
        <div class="sample-card" data-sample="public_proxy">
          <img class="sample-thumb" src="/v1/demo/sample/public_proxy" alt="public_proxy">
          <div class="sample-info">
            <strong>{escape(DEMO_SAMPLES['public_proxy']['title'])}</strong>
            <span>{escape(DEMO_SAMPLES['public_proxy']['subtitle'])}</span>
          </div>
        </div>
      </div>

      <div id="pane-upload" class="hidden">
        <label class="field" for="localFileInput">选择本地眼底照片（JPEG / PNG / WEBP）</label>
        <input type="file" id="localFileInput" accept="image/jpeg,image/png,image/webp">
      </div>

      <div id="pane-url" class="hidden">
        <label class="field" for="imageUrlInput">公网 HTTPS 眼底照片地址</label>
        <input type="text" id="imageUrlInput" placeholder="https://...">
      </div>

      <div id="pane-no_image" class="hidden">
        <p style="font-size:12.5px;color:#475569;margin:4px 0 8px;">
          无图咨询模式：支持体验“纸质体检报告名词解释（如豹纹状眼底/玻璃膜疣）”及“急性飞蚊闪光感症状优先急诊拦截”，严禁在无图时作出正常结论。
        </p>
      </div>

      <label class="field" for="eyeSelect">原子绑定眼别（同步入库防错位）</label>
      <select id="eyeSelect">
        <option value="left">左眼（OS）</option>
        <option value="right">右眼（OD）</option>
        <option value="unknown">暂不明确（unknown）</option>
      </select>

      <label class="field" for="symptomsInput">眼部当前症状（触发症状优先规则检查）</label>
      <input type="text" id="symptomsInput" value="近期偶觉视物轻度模糊" placeholder="例如：突发大量飞蚊和闪光感 / 无明显不适">

      <label class="field" for="historyInput">慢病与既往眼科治疗史</label>
      <input type="text" id="historyInput" value="2型糖尿病8年" placeholder="例如：糖尿病10年, 高血压, 既往激光光凝">

      <label class="field" for="queryInput">补充提问内容（可选）</label>
      <textarea id="queryInput" placeholder="例如：报告里的激光瘢痕或玻璃膜疣是什么意思？带家里老人就诊要准备什么？"></textarea>

      <div class="quick-scenarios">
        <button type="button" class="chip-btn" data-preset="dr_left">预设1：左眼糖网+糖尿病</button>
        <button type="button" class="chip-btn" data-preset="normal_right">预设2：右眼体检对照</button>
        <button type="button" class="chip-btn" data-preset="urgent_conflict">预设3：急症症状冲突</button>
        <button type="button" class="chip-btn" data-preset="no_img_checkup">预设4：无图体检科普</button>
      </div>

      <button type="button" class="run-btn" id="runBtn">运行多智能体协作全流程</button>
      <div id="errBox" style="color:#b91c1c;font-size:12.5px;margin-top:8px;"></div>
    </div>

    <div class="card">
      <h2 class="card-title">2. 本演示会话检查记录与同眼对比</h2>
      <div id="historyBox" style="font-size:12.5px;color:#475569;">暂无已执行的演示检查记录。依次运行左眼与右眼（或同眼两次）后将自动汇总于此。</div>
    </div>
  </section>

  <section id="resultArea">
    <div class="empty-state" id="emptyState">
      <h2 style="margin:0 0 8px;color:#0f4c81;">请在左侧选择预置样例或输入场景后点击“运行多智能体协作全流程”</h2>
      <p style="margin:0;">系统将串联执行：A1 主控分诊 → W10 六模型专家与四类病灶推理 → A2 医学证据解读 → A3 症状优先随访决策 → A4 同眼对比与科普输出。</p>
    </div>

    <div id="liveResults" class="hidden">
      <div class="hero-banner" id="heroBanner">
        <div>
          <div class="hero-title" id="heroTitle">--</div>
          <p class="hero-sub" id="heroSub">--</p>
        </div>
        <div id="reportBtnWrap"></div>
      </div>

      <div class="metrics-strip">
        <div class="metric-box"><div class="k">绑定眼别</div><div class="v" id="mEye">--</div></div>
        <div class="metric-box"><div class="k">图像质控</div><div class="v" id="mQuality">--</div></div>
        <div class="metric-box"><div class="k">糖网五级模型</div><div class="v" id="mDr">--</div></div>
        <div class="metric-box"><div class="k">端到端总耗时</div><div class="v" id="mElapsed">--</div></div>
      </div>

      <div class="card" style="margin-top:14px;">
        <h2 class="card-title">
          <span id="narrativeTitle">受众自适应医学解读</span>
          <span style="font-size:12px;font-weight:400;color:#64748b;">切换顶部受众视角可实时刷新解读话术，数值与规则优先级别保持恒定</span>
        </h2>
        <div id="narrativeBody" style="font-size:13.5px;"></div>
        <div class="src-tags" id="sourceTags"></div>
        <div id="clinicianTableWrap"></div>
      </div>

      <div class="card" id="visualCard">
        <h2 class="card-title">原图与病灶/结构定位双图对照</h2>
        <div class="dual-imgs">
          <div class="img-pane">
            <img id="imgOrig" alt="原图">
            <div class="img-cap">眼底原图（经图像标准化解码）</div>
          </div>
          <div class="img-pane">
            <img id="imgOverlay" alt="标注图">
            <div class="img-cap" id="overlayCap">病灶候选或解剖轮廓图</div>
          </div>
        </div>
        <div class="legend-row" id="legendRow"></div>
      </div>

      <div class="card hidden" id="compareCard">
        <h2 class="card-title">同眼纵向历史对比结果（A4 / W60）</h2>
        <div id="compareBody" style="font-size:13px;"></div>
      </div>

      <div class="card">
        <h2 class="card-title">多智能体协同路由与规则 Trace 面板</h2>
        <div class="trace-list" id="traceList"></div>
      </div>
    </div>
  </section>
</main>

<script>
(() => {{
  const state = {{
    mode: "sample",
    sampleId: "idrid_03",
    audience: "patient",
    uploadB64: null,
    lastData: null,
  }};

  const el = (id) => document.getElementById(id);

  function setMode(mode) {{
    state.mode = mode;
    document.querySelectorAll("#modeTabs .tab-btn").forEach((b) => {{
      b.classList.toggle("active", b.dataset.mode === mode);
    }});
    ["sample", "upload", "url", "no_image"].forEach((m) => {{
      el("pane-" + m).classList.toggle("hidden", m !== mode);
    }});
  }}

  function selectSample(sampleId) {{
    state.sampleId = sampleId;
    document.querySelectorAll(".sample-card").forEach((c) => {{
      c.classList.toggle("active", c.dataset.sample === sampleId);
    }});
    if (sampleId === "idrid_03") {{
      el("eyeSelect").value = "left";
      el("symptomsInput").value = "近期偶觉视物轻度模糊";
      el("historyInput").value = "2型糖尿病8年";
    }} else if (sampleId === "public_proxy") {{
      el("eyeSelect").value = "right";
      el("symptomsInput").value = "无明显视力下降，无飞蚊闪光";
      el("historyInput").value = "年度常规体检";
    }}
  }}

  function renderNarrative() {{
    const data = state.lastData;
    if (!data || !data.narratives) return;
    const n = data.narratives[state.audience] || data.narratives.patient;
    el("narrativeTitle").textContent = n.title;
    let html = `<p style="margin:0 0 8px;font-weight:600;color:#0f4c81;">${{esc(n.lead)}}</p><ul class="narrative-list">`;
    for (const p of n.points || []) {{
      html += `<li>${{esc(p)}}</li>`;
    }}
    html += "</ul>";
    el("narrativeBody").innerHTML = html;

    const tags = el("sourceTags");
    tags.innerHTML = "";
    for (const r of (data.rule_ids || [])) {{
      tags.innerHTML += `<span class="src-tag">${{esc(r)}}</span>`;
    }}
    for (const s of (n.sources || data.source_ids || [])) {{
      tags.innerHTML += `<span class="src-tag">${{esc(s)}}</span>`;
    }}

    const tbl = el("clinicianTableWrap");
    if (state.audience === "clinician" && n.table_rows && n.table_rows.length) {{
      let rows = `<table><thead><tr><th>评估项目</th><th>来源模型</th><th>模型分数</th><th>冻结筛查线</th><th>结果判定</th></tr></thead><tbody>`;
      for (const r of n.table_rows) {{
        rows += `<tr><td>${{esc(r.item)}}</td><td>${{esc(r.model)}}</td><td>${{esc(r.score)}}</td><td>${{esc(r.threshold)}}</td><td>${{esc(r.status)}}</td></tr>`;
      }}
      rows += "</tbody></table>";
      tbl.innerHTML = rows;
    }} else {{
      tbl.innerHTML = "";
    }}
  }}

  function esc(str) {{
    return String(str ?? "").replace(/[&<>"']/g, (c) => ({{
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }}[c]));
  }}

  document.querySelectorAll("#modeTabs .tab-btn").forEach((btn) => {{
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  }});

  document.querySelectorAll(".sample-card").forEach((card) => {{
    card.addEventListener("click", () => selectSample(card.dataset.sample));
  }});

  document.querySelectorAll("#audienceSwitch .audience-btn").forEach((btn) => {{
    btn.addEventListener("click", () => {{
      state.audience = btn.dataset.audience;
      document.querySelectorAll("#audienceSwitch .audience-btn").forEach((b) => {{
        b.classList.toggle("active", b === btn);
      }});
      renderNarrative();
    }});
  }});

  el("localFileInput").addEventListener("change", (ev) => {{
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {{ state.uploadB64 = reader.result; }};
    reader.readAsDataURL(file);
  }});

  document.querySelectorAll(".chip-btn").forEach((btn) => {{
    btn.addEventListener("click", () => {{
      const p = btn.dataset.preset;
      if (p === "dr_left") {{
        setMode("sample"); selectSample("idrid_03"); el("queryInput").value = "";
      }} else if (p === "normal_right") {{
        setMode("sample"); selectSample("public_proxy"); el("queryInput").value = "";
      }} else if (p === "urgent_conflict") {{
        setMode("sample"); selectSample("public_proxy");
        el("symptomsInput").value = "右眼今天突然出现大量飞蚊和闪光感";
        el("queryInput").value = "照片看起来没事，为什么眼前一直闪？";
      }} else if (p === "no_img_checkup") {{
        setMode("no_image");
        el("eyeSelect").value = "unknown";
        el("symptomsInput").value = "无明显不适";
        el("historyInput").value = "单位年度体检";
        el("queryInput").value = "体检单写着豹纹状眼底(TSLN)和少量玻璃膜疣(DN)，严重吗？需要手术吗？";
      }}
    }});
  }});

  el("runBtn").addEventListener("click", async () => {{
    const btn = el("runBtn");
    el("errBox").textContent = "";
    btn.disabled = true;
    btn.textContent = "多智能体协同分析执行中...";
    try {{
      const payload = {{
        mode: state.mode,
        sample_id: state.sampleId,
        image_url: el("imageUrlInput").value.trim() || null,
        image_base64: state.mode === "upload" ? state.uploadB64 : null,
        audience: state.audience,
        eye: el("eyeSelect").value,
        symptoms: el("symptomsInput").value.trim(),
        history: el("historyInput").value.trim(),
        query_text: el("queryInput").value.trim(),
        session_id: "demo_workbench",
      }};
      const resp = await fetch("/v1/demo/run", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload),
      }});
      const data = await resp.json();
      if (!resp.ok || !data.ok) {{
        throw new Error(data.message || (data.error && data.error.message) || data.detail || "执行失败，请检查输入或服务状态。");
      }}
      state.lastData = data;
      el("emptyState").classList.add("hidden");
      el("liveResults").classList.remove("hidden");

      const banner = el("heroBanner");
      banner.className = "hero-banner " + (
        data.care_priority === "urgent_ophthalmology" ? "urgent" :
        (data.care_priority === "annual_screening" ? "ok" : "warn")
      );
      const level = data.clinical_decision && data.clinical_decision.level;
      el("heroTitle").textContent = level ? level + " · " + data.priority_label : data.priority_label;
      el("heroSub").textContent = "建议时间窗：" + data.followup_time;

      const repWrap = el("reportBtnWrap");
      const urls = data.visual_urls || {{}};
      if (urls.report) {{
        repWrap.innerHTML = `<a class="pdf-link" href="${{esc(urls.report)}}" target="_blank" rel="noopener">打开完整临床报告 / 导出 A4 PDF</a>`;
      }} else {{
        repWrap.innerHTML = "";
      }}

      if (data.mode === "no_image") {{
        el("mEye").textContent = "未传图像";
        el("mQuality").textContent = "未评估";
        el("mDr").textContent = "未评估";
        el("mElapsed").textContent = "实时响应";
        el("visualCard").classList.add("hidden");
      }} else {{
        el("mEye").textContent = data.eye_label || "--";
        el("mQuality").textContent = data.quality_label || "--";
        const dr = data.dr_grading || {{}};
        el("mDr").textContent = dr.grade !== null && dr.grade !== undefined ? `${{dr.grade}} 级 (${{dr.level || ''}})` : "未完成";
        el("mElapsed").textContent = data.total_elapsed_ms ? `${{data.total_elapsed_ms}} ms` : "--";

        if (urls.original) {{
          el("visualCard").classList.remove("hidden");
          el("imgOrig").src = urls.original;
          el("imgOverlay").src = urls.overlay || urls.original;
          const loc = data.localization || {{}};
          el("overlayCap").textContent = loc.source_label || "模型辅助标注图";
          const leg = el("legendRow");
          leg.innerHTML = "";
          for (const item of (loc.legend || [])) {{
            leg.innerHTML += `<span><span class="swatch" style="background:${{esc(item.color)}}"></span>${{esc(item.name)}}</span>`;
          }}
        }} else {{
          el("visualCard").classList.add("hidden");
        }}
      }}

      renderNarrative();

      const cmpCard = el("compareCard");
      if (data.comparison_preview && data.comparison_preview.comparable) {{
        cmpCard.classList.remove("hidden");
        const c = data.comparison_preview;
        const drC = c.dr_grade_comparison || {{}};
        el("compareBody").innerHTML =
          `<p style="margin:0 0 6px;">已自动对比同眼检查（${{esc(c.previous_assessment_id.slice(0,10))}}... → ${{esc(c.current_assessment_id.slice(0,10))}}...），版本一致性：${{c.version_consistent ? '完全一致' : '存在模型版本差异'}}；糖网分期变化：${{drC.previous_grade ?? '--'}} 级 → ${{drC.current_grade ?? '--'}} 级。</p>` +
          `<p style="margin:0;color:#475569;">${{esc(c.interpretation_note)}}</p>`;
      }} else {{
        cmpCard.classList.add("hidden");
      }}

      const tr = el("traceList");
      tr.innerHTML = "";
      for (const step of (data.trace || [])) {{
        tr.innerHTML += `
          <div class="trace-item ${{step.status === 'urgent' ? 'urgent' : ''}}">
            <div class="trace-head"><span>${{esc(step.agent)}} · ${{esc(step.stage)}}</span><span>状态: ${{esc(step.status)}} · ${{esc(step.elapsed_ms)}} ms</span></div>
            <div class="trace-body">${{esc(step.detail)}}</div>
          </div>`;
      }}

      if (data.recent_assessments && data.recent_assessments.length) {{
        let h = "<table><thead><tr><th>眼别</th><th>评估 ID (前10位)</th><th>时间</th></tr></thead><tbody>";
        for (const r of data.recent_assessments) {{
          const ec = r.eye === "left" ? "左眼" : (r.eye === "right" ? "右眼" : "待确认");
          h += `<tr><td>${{ec}}</td><td><code>${{esc(r.assessment_id.slice(0, 10))}}</code></td><td>${{esc((r.created_utc || '').slice(11, 19))}}</td></tr>`;
        }}
        h += "</tbody></table>";
        el("historyBox").innerHTML = h;
      }}
    }} catch (err) {{
      el("errBox").textContent = err.message || "执行失败。";
    }} finally {{
      btn.disabled = false;
      btn.textContent = "运行多智能体协作全流程";
    }}
  }});
}})();
</script>
</body>
</html>
"""
