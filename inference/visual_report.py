"""Render an interactive, printable A4 report and same-view image pair."""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import re
import shutil
import tempfile
from html import escape
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter

from inference.clinical_referral import SAFETY_NOTE
from inference.clinical_review import REFERENCE_SHA, build_clinical_review, screening_groups
from inference.report_language import decision_evidence, patient_followup
from inference.report_levels import display_report_level, report_level_css
from inference.report_metadata import format_report_metadata
from inference.report_summary import DR_LEVELS


REFERENCE_FOLDER = Path(__file__).resolve().parent / "reference_cases" / "idrid_03"
# These four codes follow IDRiD's segmentation task, independent of its DR grades.
LESIONS = {"MA": ("微动脉瘤", "#53E8FF", 1), "HE": ("视网膜出血", "#FF7397", 2),
           "EX": ("硬性渗出", "#FFD64D", 4), "SE": ("软性渗出（棉絮斑）", "#74F49A", 8)}
ANATOMY = {"optic_disc": ("视盘", "#8edbd8"), "optic_cup": ("视杯", "#d5d9f2")}

EYE_LABELS = {"left": "左眼（OS）", "right": "右眼（OD）", "unknown": "眼别待确认"}
PRIORITY_BADGE = {
    "urgent_ophthalmology": ("badge-urgent", "建议立即眼科急诊评估"),
    "prompt_ophthalmology": ("badge-prompt", "建议尽快眼底专科就诊"),
    "scheduled_ophthalmology": ("badge-warn", "建议限期眼科专科复核"),
    "routine_followup": ("badge-info", "建议常规眼科随访"),
    "annual_screening": ("badge-ok", "建议定期常规复查"),
    "retake_image": ("badge-warn", "图像质量不足，建议重拍"),
}
DR_STAGE_SHORT = ("0级 无明显病变", "1级 轻度", "2级 中度", "3级 重度", "4级 增殖期")


def _reference_masks(sha, size):
    if sha != REFERENCE_SHA or size != (4288, 2848):
        return None
    try:
        info = json.loads((REFERENCE_FOLDER / "manifest.json").read_text(encoding="utf-8-sig"))
        blob = (REFERENCE_FOLDER / "masks.png").read_bytes()
        if (info["image_sha256"] != sha or info["mask_sha256"] != hashlib.sha256(blob).hexdigest()
                or info["dr_grade"] != 3 or info["source_image_id"] != "IDRiD_03"
                or info["grading_image_id"] != "IDRiD_369"):
            return None
        with Image.open(io.BytesIO(blob)) as bits:
            if bits.format != "PNG" or bits.mode != "L" or bits.size != size or bits.getextrema()[1] > 15:
                return None
            return {key: bits.point([255 if n & bit else 0 for n in range(256)])
                    for key, (_, _, bit) in LESIONS.items()}
    except (OSError, KeyError, ValueError):
        return None


def _anatomy_masks(response, size):
    stage = (response.get("results") or {}).get("disc_cup_segmentation") or {}
    out = stage.get("output") or {}
    if (stage.get("status") != "completed" or out.get("source_size_wh") != list(size)
            or out.get("size_hw") != [512, 512]
            or out.get("coordinate_space") != "whole_image_resized_to_model_img_size"
            or (out.get("mapping_to_source") or {}).get("kind") != "stretch"):
        return {}
    masks = {}
    try:
        for code in ANATOMY:
            artifact = out["mask_artifacts"][code]
            if artifact.get("encoding") != "base64" or artifact.get("media_type") != "image/png":
                return {}
            if len(artifact["data"]) > 1000000:
                return {}
            data = base64.b64decode(artifact["data"], validate=True)
            if hashlib.sha256(data).hexdigest() != artifact.get("sha256"):
                return {}
            with Image.open(io.BytesIO(data)) as mask:
                if mask.format != "PNG" or mask.size != (512, 512) or mask.mode != "L":
                    return {}
                hist = mask.histogram()
                if sum(hist[1:255]) or not hist[0] or not hist[255]:
                    return {}
                masks[code] = mask.resize(size, Image.Resampling.NEAREST)
    except (OSError, KeyError, ValueError, TypeError, binascii.Error):
        return {}
    return masks


def _predict_masks(image):
    try:
        from inference.lesion_segmenter import get_lesion_masks
    except ImportError:
        return None
    return get_lesion_masks(image)


def _draw_masks(image, masks, colors, anatomy=False):
    # Work in source coordinates. Subtle transparent interiors preserve retinal detail.
    painted = image.copy()
    legend = []
    for code, mask in masks.items():
        if not mask.getbbox():
            continue
        name, color = colors[code][:2]
        if not anatomy:
            painted.paste(color, (0, 0, *painted.size), mask.point([0] + [77] * 255))
        # Scale the thin outer border to keep tiny components legible at report size.
        edge_width = 3 if max(image.size) < 2500 else 5
        edge = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(edge_width)), mask)
        painted.paste(color, (0, 0, *painted.size), edge)
        legend.append({"code": code, "name": name, "color": color})
    return painted, legend


def _comparison(image, response, sha):
    reference_grade = 3 if sha == REFERENCE_SHA and image.size == (4288, 2848) else None
    pred = _predict_masks(image)
    if isinstance(pred, dict):
        pm = pred.get("masks")
        if (isinstance(pm, dict) and set(pm) == set(LESIONS)
                and all(isinstance(m, Image.Image) and m.mode == "L" and m.size == image.size
                        and sum(m.histogram()[1:255]) == 0 for m in pm.values())):
            overlay, legend = _draw_masks(image, pm, LESIONS)
            if legend:
                return overlay, dict(kind="model_candidates", localization_status="localized",
                                     source_label=str(pred["source_label"]), legend=legend, reference_grade=reference_grade,
                                     candidate_status="candidates_present",
                                     note="着色范围为分割模型提示的候选病灶，供医生与原图对照复核。覆盖范围限于微动脉瘤、出血、硬性渗出和软性渗出。")
            return image, dict(kind="no_candidates", localization_status="unavailable",
                               source_label=str(pred["source_label"]), legend=[], reference_grade=reference_grade,
                               candidate_status="no_candidates",
                               note="本次病灶定位已运行，四类范围内未产生可绘制的候选区。仍需医生阅片。")
    masks = _reference_masks(sha, image.size)
    if masks:
        overlay, legend = _draw_masks(image, masks, LESIONS)
        return overlay, dict(kind="expert_reference", localization_status="lesion_reference",
                             source_label="IDRiD 公开病例 · 数据集专家标注", legend=legend,
                             reference_grade=reference_grade, candidate_status="reference_only",
                             note="本次未获得分割模型的定位结果。右图供教学对照，着色范围来自该病例的公开专家标注。")
    anatomy = _anatomy_masks(response, image.size)
    if anatomy:
        overlay, legend = _draw_masks(image, anatomy, ANATOMY, anatomy=True)
        return overlay, dict(kind="anatomy", localization_status="anatomical_only",
                             source_label="视盘视杯模型 · 解剖结构轮廓", legend=legend,
                             reference_grade=None, candidate_status="unavailable",
                             note="当前只显示模型提取的视盘和视杯边界。本图缺少可用的疾病定位结果。")
    return None, dict(kind="unavailable", localization_status="unavailable",
                      source_label="本图无可用空间定位结果", legend=[], reference_grade=None,
                      candidate_status="unavailable", note="本次未取得有效的定位掩膜，请结合原图阅片。")


def _score_rows(response):
    summary = response.get("report_summary") or {}
    rows = []
    dr = summary.get("dr_severity") or {}
    if dr.get("status") == "completed":
        rows.append("<tr><td>糖尿病视网膜病变</td><td>糖网专病分级</td><td>" +
                    escape(str(dr.get("score_display") or "未提供")) + "</td><td>不适用</td><td>分期模型</td></tr>")
    for source, group, _ in screening_groups(summary):
        if group.get("status") != "completed":
            continue
        for item in group.get("items", []):
            name = str(item.get("name") or "未提供")
            values = (name, source, item.get("score_display"), item.get("research_threshold_display"),
                      item.get("research_threshold_status"))
            rows.append("<tr>" + "".join("<td>" + escape(str(v or "未提供")) + "</td>" for v in values) + "</tr>")
    return "".join(rows)


def _dr_scale_html(response):
    summary = response.get("report_summary") or {}
    dr = summary.get("dr_severity") or {}
    grade = dr.get("grade") if dr.get("status") == "completed" else None
    if type(grade) is not int or not 0 <= grade <= 4:
        return ""
    cells = []
    for idx, label in enumerate(DR_STAGE_SHORT):
        cls = "scale-step active" if idx == grade else "scale-step"
        cells.append(f'<div class="{cls}"><span class="step-num">{idx}</span><span>{escape(label)}</span></div>')
    return '<div class="dr-scale" aria-label="糖网五级分期位置">' + "".join(cells) + "</div>"


def _image_readable(response):
    quality = (response.get("results") or {}).get("quality") or {}
    return (quality.get("status") == "completed"
            and (quality.get("output") or {}).get("predicted_label") in {"Good", "Usable"})


def _quality_blocked_info():
    return {"kind": "unavailable", "localization_status": "unavailable", "source_label": "图像可判读性不足",
            "candidate_status": "quality_blocked", "reference_grade": None, "legend": [],
            "note": "请核对原图，必要时重新拍摄。", "original_url": None, "overlay_url": None, "report_url": None}


def _primary_reading(summary, decision, *, readable):
    if not readable:
        return "这张照片暂无法可靠判读，眼底情况需由医生当面评估。"
    dr = summary.get("dr_severity") or {}
    grade = dr.get("grade") if dr.get("status") == "completed" else None
    if type(grade) is not int or not 0 <= grade < len(DR_LEVELS):
        primary = "本次筛查暂无法完成糖网分期，建议结合原图复核。"
    elif grade == 0:
        primary = "本次糖网分期暂评为 0 级，当前照片未见明确的典型糖网病变。"
    else:
        primary = f"眼底筛查提示{DR_LEVELS[grade][1]}（暂评 {grade} 级）。"
    if any(item.get("code") == "DR_SEVERE_REVIEW" for item in decision.get("trigger_items") or []):
        primary += "存在重度糖网低估风险，建议眼底专科复核分期。"
    return primary


def _resolve_care_and_intake(response, info, assessment_record):
    from inference.clinical_workflow import _coerce_eye, _coerce_str_list, evaluate_care_rules

    record = assessment_record or {}
    evidence = record.get("evidence_packet") or {}
    intake = evidence.get("intake") or {}
    if assessment_record is None and response.get("_report_intake"):
        initial = response["_report_intake"]
        intake = {
            "eye": _coerce_eye(initial.get("eye")),
            "symptoms": _coerce_str_list(initial.get("symptoms")),
            "history": _coerce_str_list(initial.get("history")),
        }
    plan = record.get("latest_care_plan")
    is_current = plan is not None and plan.get("intake_revision") == intake.get("revision")
    if is_current and plan.get("clinical_decision"):
        return intake, plan, True

    evaluated = evaluate_care_rules({**response, "visual_report": info}, intake)
    if is_current:
        return intake, {**plan, **evaluated}, True
    return intake, evaluated, False


def _referral_time(value):
    return str(value or "").replace("筛查流程参考", "建议")


def _priority_chip(decision, *, label="", fallback="badge-info"):
    level = display_report_level(decision)
    if level:
        text = f"{level} · {label}" if label else level
        return f'<span class="pill level-chip" data-level="{level}">{escape(text)}</span>'
    return f'<span class="pill {fallback}">{escape(label)}</span>' if label else ""


def _decision_section_html(decision):
    if not decision:
        return ""

    def field(item, key):
        return escape(str(item.get(key) or ""))

    title = field(decision, "title")
    marker = _priority_chip(decision)
    next_step = " · ".join(value for value in
                           (field(decision, "destination"), escape(_referral_time(decision.get("timeframe")))) if value)
    timing = f'<p class="decision-next"><strong>建议就诊安排</strong> {next_step}</p>' if next_step else ""

    rows = []
    for item in decision.get("trigger_items") or []:
        tests = item.get("tests") or []
        if isinstance(tests, str):
            tests = [tests]
        item_name = field(item, "name")
        item_label = f'{_priority_chip(item)} {item_name}'
        basis = field(item, "basis")
        basis_html = f'<div class="basis">{basis}</div>' if basis else ""
        destination = field(item, "destination")
        destination_html = f'<div class="basis">{destination}</div>' if destination else ""
        rows.append(
            f'<tr><th scope="row">{item_label}</th>'
            f'<td>{escape(decision_evidence(item))}{basis_html}</td>'
            f'<td>{escape(_referral_time(item.get("timeframe")))}{destination_html}</td>'
            f'<td>{escape("、".join(map(str, tests))) or "由医生评估"}</td></tr>'
        )
    details = ""
    if rows:
        details = ('<div class="scroll"><table class="decision-table"><thead><tr>'
                   '<th scope="col">触发项目</th><th scope="col">判定依据</th>'
                   '<th scope="col">时限与去向</th><th scope="col">建议检查</th>'
                   '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>')

    safety = f'<p class="note-line decision-safety"><strong>急症提示</strong> {escape(SAFETY_NOTE)}</p>'
    scope = field(decision, "scope")
    scope_note = f'<p class="scope-note">{scope}</p>' if scope else ""
    return (
        '<section class="card decision-card" aria-labelledby="clinical-decision-title">'
        '<div class="section-head"><h2 id="clinical-decision-title">临床决策与转诊紧急度</h2>'
        f'{marker}</div><p class="primary">{title}</p>{timing}{details}{safety}{scope_note}</section>'
    )


def _care_section_html(intake, care, decision, has_saved_plan):
    priority_code = care.get("care_priority") or "annual_screening"
    badge_cls, default_label = PRIORITY_BADGE.get(priority_code, ("badge-info", "常规随访"))
    priority_label = care.get("priority_label") or default_label
    eye_text = EYE_LABELS.get(intake.get("eye") or "unknown", "眼别待确认")
    symptoms_text = "、".join(intake.get("symptoms") or []) or "还未记录近期的眼部感受"
    history_text = "、".join(intake.get("history") or []) or "还未填写既往病史"
    onset_text = intake.get("symptom_onset")
    confirmed_plan = intake.get("confirmed_plan")
    plan_tag = "已生成随访计划" if has_saved_plan else "可在就诊后补充安排"
    guide = patient_followup(decision, intake)

    def _li(items):
        return "".join("<li>" + escape(str(x)) + "</li>" for x in items)

    def _fact(label, value):
        return f'<div><span class="meta-k">{label}</span><span class="meta-v">{escape(value)}</span></div>' if value else ""

    window = _referral_time(decision.get("timeframe")) or "请按医生建议复查"
    destination = str(decision.get("destination") or "")
    next_step = escape(" · ".join(part for part in (destination, window) if part))
    card_cls = "card care-card urgent-border" if priority_code == "urgent_ophthalmology" else "card care-card"

    return (
        f'<section class="{card_cls}">'
        '<div class="section-head">'
        '<h2>受检者信息与随访计划</h2>'
        f'{_priority_chip(decision, label=priority_label, fallback=badge_cls)}'
        "</div>"
        '<div class="meta-grid">'
        f'{_fact("检查眼别", eye_text)}{_fact("随访记录", plan_tag)}'
        f'{_fact("近期眼部感受", symptoms_text)}{_fact("症状开始时间", onset_text)}'
        f'{_fact("既往病史", history_text)}{_fact("医生原有安排", confirmed_plan)}'
        "</div>"
        '<div class="care-columns">'
        f'<div class="subcard"><h3>下一步怎么做</h3><p class="care-next">{next_step}</p><ul>{_li(guide["actions"])}</ul></div>'
        f'<div class="subcard"><h3>在家留意什么</h3><ul>{_li(guide["monitoring"])}</ul></div>'
        f'<div class="subcard"><h3>就诊时可以问</h3><ul>{_li(guide["questions"])}</ul></div>'
        f'<div class="subcard"><h3>就诊前可以准备</h3><ul>{_li(guide["preparation"])}</ul></div>'
        "</div>"
        "</section>"
    )


_REPORT_EXPLANATION_CARD = """<section class="card no-print" aria-labelledby="report-explanation-title">
    <div class="section-head">
      <h2 id="report-explanation-title">{heading}</h2>
      <div class="toolbar-actions">
        <span class="pill badge-info">{provider}</span>
        <button class="btn btn-secondary" type="button" id="report-explanation-button">生成解读</button>
      </div>
    </div>
    <p class="report-explanation-scope">根据本次眼底筛查与分诊结果梳理主要发现和复核重点，供医生结合原图核对。</p>
    <p class="report-explanation-status" id="report-explanation-status" role="status"></p>
    <div class="report-explanation-result" id="report-explanation-result" hidden>
      <p class="report-explanation-text" id="report-explanation-text"></p>
      <p class="report-explanation-model" id="report-explanation-model" hidden></p>
    </div>
  </section>"""

_REPORT_EXPLANATION_SCRIPT = """
const explanationButton = document.getElementById('report-explanation-button');
const explanationStatus = document.getElementById('report-explanation-status');
const explanationResult = document.getElementById('report-explanation-result');
explanationButton.addEventListener('click', async () => {
  explanationButton.disabled = true;
  explanationResult.hidden = true;
  explanationStatus.textContent = '正在生成报告解读，请稍候。';
  try {
    const response = await fetch('./report-explanation', {method: 'POST'});
    const data = await response.json();
    if (!response.ok || data?.ok !== true || typeof data.explanation !== 'string' || !data.explanation.trim()) {
      explanationStatus.textContent = typeof data?.message === 'string' && data.message
        ? data.message : '未能生成解读，请重试。';
      return;
    }
    const model = document.getElementById('report-explanation-model');
    model.textContent = typeof data.model === 'string' && data.model ? '生成模型 ' + data.model : '';
    model.hidden = !model.textContent;
    document.getElementById('report-explanation-text').textContent = data.explanation;
    explanationResult.hidden = false;
    explanationStatus.textContent = '解读生成完成。';
    explanationButton.textContent = '重新生成';
  } catch {
    explanationStatus.textContent = '解读服务暂不可用，请重试。';
  } finally {
    explanationButton.disabled = false;
  }
});
"""


def _report_html(info, response, has_overlay, *, assessment_record=None, hospital_record=None,
                 report_explanation_label: str | None = None):
    readable = _image_readable(response)
    has_overlay = has_overlay and readable
    info = (info or {}) if readable else _quality_blocked_info()
    review = build_clinical_review(response, info)
    hospital_record = hospital_record or {}
    patient = hospital_record.get("patient") or {}
    if hospital_record:
        title = "眼科临床辅助诊断与病灶分析报告"
        intake = {
            "eye": hospital_record.get("eye"),
            "history": [patient[key] for key in ("chronic_history", "eye_history") if patient.get(key)],
        }
        care = hospital_record["care_plan"]
        has_saved_plan = True
    else:
        title = "眼底影像智能筛查与随访报告"
        intake, care, has_saved_plan = _resolve_care_and_intake(response, info, assessment_record)
    summary = response.get("report_summary") or {}
    initial_decision = summary.get("clinical_decision") or {}
    care_decision = care.get("clinical_decision") or {}
    initial_emergency = (not hospital_record and assessment_record is None
                         and initial_decision.get("status") == "emergency")
    decision = initial_decision if initial_emergency else care_decision or initial_decision
    decision_section = _decision_section_html(decision)
    rows = _score_rows(response)
    primary = _primary_reading(summary, decision, readable=readable)
    tasks = "".join("<li><strong>" + escape(item["topic"]) + "</strong>　" + escape(item["action"]) +
                    "</li>" for item in review["review_items"])
    review_block = "<section class='card'><h2>阅片意见</h2><ol>" + tasks + "</ol></section>" if tasks else ""
    technical_notes = list(review["other_results"])
    technical_notes.extend(item["basis"] for item in review["review_items"] if item["basis"])
    other = "".join("<p>" + escape(line) + "</p>" for line in technical_notes)
    kind = info.get("kind")
    caption = {"expert_reference": "病灶标注图（专家参考）", "model_candidates": "病灶候选区（模型预测）",
               "anatomy": "视盘视杯结构图", "no_candidates": "本次未检出可标注候选区",
               "unavailable": "本次无法定位"}.get(kind, "眼底标注图")
    keys = "".join('<span><i style="background:' + x["color"] + '"></i>' + escape(x["name"]) + '</span>'
                   for x in info.get("legend") or [])
    right = (
        f'<figure><figcaption>{caption}</figcaption><a href="overlay.jpg" target="_blank" rel="noreferrer">'
        f'<img src="overlay.jpg" alt="{caption}"></a><div class="key">{keys}</div></figure>'
        if has_overlay else ""
    )
    original = (
        '<figure><figcaption>原图（完整视野）</figcaption><a href="original.jpg" target="_blank" rel="noreferrer">'
        '<img src="original.jpg" alt="本次眼底原图"></a></figure>'
        if hospital_record.get("has_original", True) else ""
    )
    image_note = " · ".join(info[key] for key in ("source_label", "note") if info.get(key))
    if not original:
        image_note = "本次报告未附眼底原图。" + image_note
    elif not image_note and not has_overlay:
        image_note = "本次报告未附病灶标注图。"
    credit = ""
    if type(info.get("reference_grade")) is int and info["reference_grade"] == 3 and response.get("image_sha256") == REFERENCE_SHA:
        origin = "右图叠加了公开像素标注。" if kind == "expert_reference" else "右图显示本次模型运行结果。"
        reference_grade = escape(str(info["reference_grade"]))
        credit = (f'<p class="sample-credit">演示样例对照：IDRiD_03 专家标注 {reference_grade} 级</p>'
                  '<p class="credit">案例来源：<a rel="noreferrer" href="https://doi.org/10.21227/H25W98">IDRiD</a>，分割图 IDRiD_03 与分级图 IDRiD_369 同源同图。'
                  '© Prasanna Porwal、Samiksha Pachade、Manesh Kokare，<a rel="noreferrer" href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>。'
                  + origin + '</p>')

    check_time, report_number = format_report_metadata(
        hospital_record.get("analyzed_utc") or response.get("created_utc"),
        hospital_record.get("report_id") or response.get("request_id"),
    )
    assessment_id = escape(report_number)
    check_time = escape(check_time)
    eye_badge = escape(EYE_LABELS.get(intake.get("eye") or "unknown", "眼别待确认"))
    priority_code = care.get("care_priority") or "annual_screening"
    priority_cls, priority_default = PRIORITY_BADGE.get(priority_code, ("badge-info", "常规随访"))
    priority_text = str(care.get("priority_label") or priority_default)
    dr_scale = _dr_scale_html(response) if readable else ""
    care_section = _care_section_html(intake, care, decision, has_saved_plan)
    if initial_emergency and care_decision.get("status") != "emergency":
        priority_cls, priority_default = PRIORITY_BADGE["urgent_ophthalmology"]
        priority_text = priority_default
        care_section = ""
    header_chip = _priority_chip(decision, label=priority_text, fallback=priority_cls)
    patient_facts = ""
    if patient:
        identity = "、".join(str(value) for value in (
            patient.get("gender"), f"{patient['age']}岁" if patient.get("age") is not None else None
        ) if value)
        name = str(patient.get("name") or "姓名待核对")
        patient_facts = f'<span>患者：{escape(f"{name}（{identity}）" if identity else name)}</span>'
        if patient.get("phone"):
            patient_facts += f'<span>手机档案：{escape(str(patient["phone"]))}</span>'
    comparison = (hospital_record.get("comparison") or {}).get("doctor_note") if readable else None
    comparison_block = (
        f'<section class="card"><h2>同眼纵向历史对照</h2><p>{escape(comparison)}</p></section>'
        if comparison else ""
    )
    image_grid_class = "image-grid" if original and right else "image-grid single-col"
    image_toggle = (
        '<button type="button" class="btn btn-secondary" onclick="toggleImageLayout()">切换单图大图 / 双图对照</button>'
        if original and right else ""
    )
    explanation_card = (
        _REPORT_EXPLANATION_CARD.format(
            heading="DeepSeek 辅助分析" if report_explanation_label == "华为 MaaS · DeepSeek" else "模型辅助解读",
            provider=escape(report_explanation_label),
        )
        if hospital_record and report_explanation_label else ""
    )

    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>{title}</title><style>
@page{{size:A4 portrait;margin:12mm 11mm}}
*{{box-sizing:border-box}}
body{{font:15px/1.65 system-ui,-apple-system,"Microsoft YaHei","PingFang SC",sans-serif;color:#14282e;background:#f4f7f8;margin:0;padding:20px 14px;-webkit-print-color-adjust:exact;print-color-adjust:exact}}
.page{{max-width:1100px;margin:0 auto}}
.toolbar{{display:flex;justify-content:flex-end;background:#ffffff;border:1px solid #d5e0e2;border-radius:10px;padding:12px 16px;margin-bottom:16px;box-shadow:0 1px 3px rgba(15,41,48,.05)}}
.toolbar-actions{{display:flex;flex-wrap:wrap;gap:8px}}
button.btn{{border:1px solid #155e75;background:#155e75;color:#ffffff;padding:7px 14px;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer}}
button.btn-secondary{{background:#ffffff;color:#155e75}}
button.btn:hover{{opacity:.92}}
button.btn:disabled{{cursor:wait;opacity:.65}}
.report-explanation-scope,.report-explanation-status{{font-size:13.5px;color:#4a6369;margin:0}}
.report-explanation-status:not(:empty){{margin-top:8px}}
.report-explanation-result{{margin-top:10px;padding:12px 14px;background:#f5fafc;border:1px solid #dbe7eb;border-radius:8px}}
.report-explanation-text{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0}}
.report-explanation-model{{font-size:12.5px;color:#4a6369;margin:8px 0 0}}
.report-header{{background:#ffffff;border:1px solid #d5e0e2;border-top:4px solid #155e75;border-radius:10px;padding:18px 22px;margin-bottom:16px}}
.header-top{{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-start;gap:10px}}
h1{{font-size:24px;margin:0 0 6px;color:#0f2930;letter-spacing:.3px}}
h2{{font-size:17px;margin:0 0 10px;color:#0f2930;border-left:3px solid #155e75;padding-left:9px}}
h3{{font-size:14px;margin:0 0 6px;color:#155e75}}
.header-sub{{font-size:13px;color:#4a6369;display:flex;flex-wrap:wrap;gap:14px}}
.pill{{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid transparent}}
.badge-urgent{{background:#fef2f2;color:#991b1b;border-color:#fca5a5}}
.badge-prompt{{background:#fff7ed;color:#9a3412;border-color:#fdba74}}
.badge-warn{{background:#fefce8;color:#854d0e;border-color:#fde047}}
.badge-info{{background:#ecfeff;color:#155e75;border-color:#a5f3fc}}
.badge-ok{{background:#f0fdf4;color:#166534;border-color:#86efac}}
{report_level_css()}
.card{{background:#ffffff;border:1px solid #d5e0e2;border-radius:10px;padding:16px 20px;margin-bottom:14px}}
.care-card.urgent-border{{border-left:4px solid #b91c1c}}
.section-head{{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;margin-bottom:10px}}
.primary{{font-size:16px;font-weight:500;color:#0f2930;margin:6px 0 10px}}
.quality-line{{margin:0 0 8px;color:#375158;font-size:14px}}
.dr-scale{{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-top:10px}}
.scale-step{{background:#f8fafb;border:1px solid #dbe4e6;border-radius:6px;padding:6px 8px;font-size:12px;color:#52686e;display:flex;align-items:center;gap:6px}}
.scale-step .step-num{{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50%;background:#e2ecee;color:#14282e;font-weight:700;font-size:11px;flex-shrink:0}}
.scale-step.active{{background:#ecfeff;border-color:#155e75;color:#0f2930;font-weight:700}}
.scale-step.active .step-num{{background:#155e75;color:#ffffff}}
.image-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.image-grid.single-col{{grid-template-columns:1fr}}
figure{{margin:0;min-width:0;background:#fafcfd;border:1px solid #e1eaec;border-radius:8px;padding:10px}}
img{{width:100%;height:auto;display:block;background:#161a1c;border-radius:4px}}
figcaption{{font-weight:650;margin-bottom:6px;font-size:14px;color:#0f2930}}
.key{{margin-top:8px;font-size:13px}}
.key span{{display:inline-block;margin:0 12px 4px 0}}
i{{display:inline-block;width:13px;height:13px;vertical-align:-2px;margin-right:5px;border:1px solid #687e80;border-radius:2px}}
.meta-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 16px;background:#f8fafb;border:1px solid #e3ecef;border-radius:8px;padding:10px 12px;margin-bottom:10px;font-size:13px}}
.meta-k{{color:#587077;margin-right:6px}}
.meta-v{{color:#0f2930;font-weight:500}}
.care-columns{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}}
.subcard{{background:#f9fbfb;border:1px solid #e2ebed;border-radius:8px;padding:10px 12px;font-size:13.5px}}
.subcard ul{{margin:0;padding-inline-start:18px}}
.subcard li{{margin-bottom:4px}}
.care-next{{font-weight:650;margin:0 0 6px}}
.basis{{color:#425861;font-size:13px;margin-top:3px}}
ol{{padding-inline-start:1.4em;margin:0}}
li{{margin-bottom:6px}}
.note-line{{font-size:13.5px;color:#344d54;margin:10px 0 0}}
.credit{{font-size:12.5px;color:#52686e;margin:6px 0 0}}
.sample-credit{{display:inline-block;background:#f1f5f9;color:#475569;border:1px solid #e2e8f0;border-radius:6px;padding:3px 8px;font-size:12.5px;margin:8px 0 0}}
.scope-note{{font-size:13px;color:#4a6369;border-top:1px dashed #d5e0e2;padding-top:8px;margin-top:8px}}
.decision-next{{margin:0 0 8px;color:#375158;font-size:14px}}
.decision-table td,.decision-table th{{padding:8px;vertical-align:top}}
.decision-table tbody th{{background:#ffffff;min-width:8em}}
.decision-safety{{font-weight:600;color:#991b1b;background:#fef2f2;border-left:3px solid #dc2626;padding:8px 10px}}
details summary{{cursor:pointer;font-weight:600;color:#155e75;padding:4px 0}}
.scroll{{overflow:auto;margin-top:8px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
td,th{{text-align:left;border-bottom:1px solid #dbe4e6;padding:6px 10px 6px 0}}
th{{color:#0f2930;background:#f8fafb}}
td{{font-variant-numeric:tabular-nums}}
@media(max-width:720px){{
  .image-grid,.meta-grid,.dr-scale{{grid-template-columns:1fr}}
  body{{padding:10px}}
}}
@media screen and (max-width:720px){{
  .decision-table{{min-width:620px}}
}}
@media print{{
  body{{background:#ffffff;padding:0;font-size:12.5px;line-height:1.5}}
  .no-print{{display:none !important}}
  .card,.report-header,figure,.subcard{{box-shadow:none;break-inside:avoid}}
  .care-card{{break-inside:auto}}
  .decision-card{{break-inside:auto}}
  .decision-card .scroll{{overflow:visible}}
  .image-grid{{grid-template-columns:1fr 1fr !important}}
  details#tech-details{{display:block}}
  details#tech-details > summary{{list-style:none;margin-bottom:4px}}
  table tr{{break-inside:avoid}}
}}
</style>
<div class="page">
  <div class="toolbar no-print">
    <div class="toolbar-actions">
      {image_toggle}
      <button type="button" class="btn" onclick="exportPdf()">导出 / 打印 PDF 报告</button>
    </div>
  </div>

  <header class="report-header">
    <div class="header-top">
      <div>
        <h1>{title}</h1>
        <div class="header-sub">
          {patient_facts}
          <span>检查时间：{check_time}</span>
          <span>眼别：{eye_badge}</span>
          <span>报告编号：{assessment_id}</span>
        </div>
      </div>
      {header_chip}
    </div>
  </header>

  <section class="card">
    <h2>主要判读</h2>
    <p class="quality-line"><strong>图像质控：</strong>{escape(review["quality"])}</p>
    <p class="primary">{escape(primary)}</p>
    {dr_scale}
  </section>

  {decision_section}
  {comparison_block}

  <section class="card">
    <h2>眼底双图可视化对照</h2>
    <main id="image-grid" class="{image_grid_class}">
      {original}
      {right}
    </main>
    <p class="note-line">{escape(image_note)}</p>
    {credit}
  </section>

  {review_block}
  {explanation_card}
  {care_section}

  <section class="card">
    <h2>专业技术附录</h2>
    <details id="tech-details">
      <summary>查看逐项模型得分和研究筛查线（导出 PDF 时将自动完整展开）</summary>
      <div class="scroll"><table><thead><tr><th>项目</th><th>模型</th><th>得分</th><th>研究线</th><th>比较结果</th></tr></thead>
      <tbody>{rows}</tbody></table></div>
      {other}
    </details>
  </section>
</div>
<script>
function exportPdf() {{
  var d = document.getElementById('tech-details');
  var wasOpen = d ? d.open : false;
  if (d) d.open = true;
  window.print();
  if (d && !wasOpen) {{
    setTimeout(function() {{ d.open = false; }}, 400);
  }}
}}
function toggleImageLayout() {{
  var grid = document.getElementById('image-grid');
  if (grid) grid.classList.toggle('single-col');
}}
{_REPORT_EXPLANATION_SCRIPT if explanation_card else ""}
</script>
</html>"""


def write_visual_report(image, response, image_sha256, artifact_root, request_id, *, url_prefix=None):
    """Store complete matched artifacts atomically. Call with the decoded frame used in inference."""
    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
        raise ValueError("Invalid report identifier")
    if not isinstance(image_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", image_sha256):
        raise ValueError("Invalid image digest")
    if response.get("image_sha256") not in (None, image_sha256) or response.get("request_id") not in (None, request_id):
        raise ValueError("Screening response belongs to a different request or image")
    if not _image_readable(response):
        return _quality_blocked_info()
    root = Path(artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / request_id
    if target.exists() or target.is_symlink():
        raise FileExistsError("Report identifier already exists")
    original = image.convert("RGB")
    overlay, info = _comparison(original, response, image_sha256)
    temporary = Path(tempfile.mkdtemp(prefix=".partial-", dir=root))
    try:
        # JPEG export keeps the original field and dimensions, dropping source metadata.
        original.save(temporary / "original.jpg", quality=95, subsampling=0, exif=b"")
        if overlay is not None:
            overlay.save(temporary / "overlay.jpg", quality=95, subsampling=0, exif=b"")
        (temporary / "report.html").write_text(_report_html(info, response, overlay is not None), encoding="utf-8")
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    prefix = url_prefix.rstrip("/") + "/" if url_prefix else ""
    return dict(info, original_url=prefix + "original.jpg", report_url=prefix + "report.html",
                overlay_url=prefix + "overlay.jpg" if overlay is not None else None)
