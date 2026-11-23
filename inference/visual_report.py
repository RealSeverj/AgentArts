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

from inference.clinical_review import REFERENCE_SHA, build_clinical_review, screening_groups


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


def _resolve_care_and_intake(response, assessment_record):
    if assessment_record is not None:
        evidence = assessment_record.get("evidence_packet") or {}
        intake = evidence.get("intake") or {}
        plan = assessment_record.get("latest_care_plan")
        if plan is not None:
            return intake, plan, True
    else:
        intake = {"eye": "unknown", "symptoms": [], "symptom_onset": None,
                  "history": [], "confirmed_plan": None, "revision": 1}
    from inference.clinical_workflow import evaluate_care_rules
    return intake, evaluate_care_rules(response, intake), False


def _care_section_html(intake, care, has_saved_plan):
    priority_code = care.get("care_priority") or "annual_screening"
    badge_cls, default_label = PRIORITY_BADGE.get(priority_code, ("badge-info", "常规随访"))
    priority_label = care.get("priority_label") or default_label
    eye_text = EYE_LABELS.get(intake.get("eye") or "unknown", "眼别待确认")
    symptoms_text = "、".join(intake.get("symptoms") or []) or "尚未补充（请在对话中告知有无视力下降、飞蚊或闪光感）"
    history_text = "、".join(intake.get("history") or []) or "尚未补充"
    onset_text = intake.get("symptom_onset") or "未提供"
    confirmed_plan = intake.get("confirmed_plan") or "无既有线下医嘱记录"
    revision = intake.get("revision") or 1

    def _li(items):
        if not items:
            return "<li>暂无额外条目</li>"
        return "".join("<li>" + escape(str(x)) + "</li>" for x in items)

    confirm_html = ""
    pending = care.get("questions_to_confirm") or []
    if pending:
        confirm_html = (
            '<div class="subcard subcard-muted"><h3>建议补充确认的问题</h3><ul>'
            + _li(pending)
            + "</ul></div>"
        )

    plan_tag = f"随访计划 v{escape(str(care.get('version', 1)))}" if has_saved_plan else "初步规则参考（待补充信息）"
    rule_refs = " / ".join([*(care.get("rule_ids") or []), *(care.get("references") or [])]) or "标准规则"
    card_cls = "card care-card urgent-border" if priority_code == "urgent_ophthalmology" else "card care-card"

    return (
        f'<section class="{card_cls}">'
        '<div class="section-head">'
        '<h2>受检者信息与个体化随访计划</h2>'
        f'<span class="pill {badge_cls}">{escape(priority_label)}</span>'
        "</div>"
        '<div class="meta-grid">'
        f'<div><span class="meta-k">检查眼别</span><span class="meta-v">{escape(eye_text)}</span></div>'
        f'<div><span class="meta-k">问询状态</span><span class="meta-v">{escape(plan_tag)}（信息修订 v{escape(str(revision))}）</span></div>'
        f'<div><span class="meta-k">自述症状</span><span class="meta-v">{escape(symptoms_text)}</span></div>'
        f'<div><span class="meta-k">起病时间</span><span class="meta-v">{escape(onset_text)}</span></div>'
        f'<div><span class="meta-k">既往病史</span><span class="meta-v">{escape(history_text)}</span></div>'
        f'<div><span class="meta-k">既往医嘱</span><span class="meta-v">{escape(confirmed_plan)}</span></div>'
        "</div>"
        '<div class="followup-banner">'
        f'<div><strong>建议复查窗口：</strong>{escape(str(care.get("followup_time") or "请遵医嘱"))}</div>'
        f'<div class="basis"><strong>优先级依据：</strong>{escape(str(care.get("priority_basis") or ""))}'
        f' <span class="code-tag">[{escape(rule_refs)}]</span></div>'
        "</div>"
        '<div class="care-columns">'
        f'<div class="subcard"><h3>下一步行动建议</h3><ul>{_li(care.get("actions"))}</ul></div>'
        f'<div class="subcard"><h3>居家监测事项</h3><ul>{_li(care.get("monitoring_items"))}</ul></div>'
        f'<div class="subcard"><h3>建议向接诊医生提问</h3><ul>{_li(care.get("questions_for_clinician"))}</ul></div>'
        f"{confirm_html}"
        "</div>"
        "</section>"
    )


def _report_html(info, response, has_overlay, *, assessment_record=None):
    review = build_clinical_review(response, info)
    intake, care, has_saved_plan = _resolve_care_and_intake(response, assessment_record)
    rows = _score_rows(response)
    tasks = "".join("<li><strong>" + escape(item["topic"]) + "</strong>　" + escape(item["action"]) +
                    ("<div class='basis'>" + escape(item["basis"]) + "</div>" if item["basis"] else "") +
                    "</li>" for item in review["review_items"])
    review_block = "<section class='card'><h2>阅片意见</h2><ol>" + tasks + "</ol></section>" if tasks else ""
    other = "".join("<p>" + escape(line) + "</p>" for line in review["other_results"])
    caption = {"expert_reference": "病灶标注图（专家参考）", "model_candidates": "病灶候选区（模型预测）",
               "anatomy": "视盘视杯结构图", "no_candidates": "本次未检出可标注候选区",
               "unavailable": "本次无法定位"}[info["kind"]]
    right = '<a href="overlay.jpg" target="_blank" rel="noreferrer"><img src="overlay.jpg" alt="' + caption + '"></a>' if has_overlay else ""
    keys = "".join('<span><i style="background:' + x["color"] + '"></i>' + escape(x["name"]) + '</span>'
                   for x in info["legend"])
    credit = ""
    if info.get("reference_grade") is not None and response.get("image_sha256") == REFERENCE_SHA:
        origin = "右图叠加了公开像素标注。" if info["kind"] == "expert_reference" else "右图显示本次模型运行结果。"
        credit = ('<p class="credit">案例来源：<a rel="noreferrer" href="https://doi.org/10.21227/H25W98">IDRiD</a>，分割图 IDRiD_03 与分级图 IDRiD_369 同源同图。'
                  '© Prasanna Porwal、Samiksha Pachade、Manesh Kokare，<a rel="noreferrer" href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>。'
                  + origin + '</p>')

    assessment_id = escape(str(response.get("request_id") or "本地离线报告"))
    created_utc = escape(str(response.get("created_utc") or "即时生成")[:19].replace("T", " "))
    eye_badge = escape(EYE_LABELS.get(intake.get("eye") or "unknown", "眼别待确认"))
    priority_code = care.get("care_priority") or "annual_screening"
    priority_cls, priority_default = PRIORITY_BADGE.get(priority_code, ("badge-info", "常规随访"))
    priority_text = escape(str(care.get("priority_label") or priority_default))
    dr_scale = _dr_scale_html(response)
    care_section = _care_section_html(intake, care, has_saved_plan)

    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>眼底影像智能筛查与随访报告</title><style>
@page{{size:A4 portrait;margin:12mm 11mm}}
*{{box-sizing:border-box}}
body{{font:15px/1.65 system-ui,-apple-system,"Microsoft YaHei","PingFang SC",sans-serif;color:#14282e;background:#f4f7f8;margin:0;padding:20px 14px;-webkit-print-color-adjust:exact;print-color-adjust:exact}}
.page{{max-width:1100px;margin:0 auto}}
.toolbar{{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:12px;background:#ffffff;border:1px solid #d5e0e2;border-radius:10px;padding:12px 16px;margin-bottom:16px;box-shadow:0 1px 3px rgba(15,41,48,.05)}}
.toolbar-meta{{display:flex;flex-wrap:wrap;align-items:center;gap:8px;font-size:13px;color:#415a61}}
.toolbar-actions{{display:flex;flex-wrap:wrap;gap:8px}}
button.btn{{border:1px solid #155e75;background:#155e75;color:#ffffff;padding:7px 14px;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer}}
button.btn-secondary{{background:#ffffff;color:#155e75}}
button.btn:hover{{opacity:.92}}
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
.followup-banner{{background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:10px 12px;margin-bottom:12px;font-size:14px}}
.care-columns{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}}
.subcard{{background:#f9fbfb;border:1px solid #e2ebed;border-radius:8px;padding:10px 12px;font-size:13.5px}}
.subcard-muted{{background:#fffbeb;border-color:#fde68a}}
.subcard ul{{margin:0;padding-inline-start:18px}}
.subcard li{{margin-bottom:4px}}
.code-tag{{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#415a61}}
.basis{{color:#425861;font-size:13px;margin-top:3px}}
ol{{padding-inline-start:1.4em;margin:0}}
li{{margin-bottom:6px}}
.note-line{{font-size:13.5px;color:#344d54;margin:10px 0 0}}
.credit{{font-size:12.5px;color:#52686e;margin:6px 0 0}}
.scope-note{{font-size:13px;color:#4a6369;border-top:1px dashed #d5e0e2;padding-top:8px;margin-top:8px}}
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
@media print{{
  body{{background:#ffffff;padding:0;font-size:12.5px;line-height:1.5}}
  .no-print{{display:none !important}}
  .card,.report-header,figure,.subcard,.followup-banner{{box-shadow:none;break-inside:avoid}}
  .image-grid{{grid-template-columns:1fr 1fr !important}}
  details#tech-details{{display:block}}
  details#tech-details > summary{{list-style:none;margin-bottom:4px}}
  table tr{{break-inside:avoid}}
}}
</style>
<div class="page">
  <div class="toolbar no-print">
    <div class="toolbar-meta">
      <span><strong>报告编号：</strong><span class="code-tag">{assessment_id}</span></span>
      <span><strong>眼别：</strong>{eye_badge}</span>
      <span class="pill {priority_cls}">{priority_text}</span>
    </div>
    <div class="toolbar-actions">
      <button type="button" class="btn btn-secondary" onclick="toggleImageLayout()">切换单图大图 / 双图对照</button>
      <button type="button" class="btn" onclick="exportPdf()">导出 / 打印 PDF 报告</button>
    </div>
  </div>

  <header class="report-header">
    <div class="header-top">
      <div>
        <h1>眼底影像智能筛查与随访报告</h1>
        <div class="header-sub">
          <span>检查时间：{created_utc} (UTC)</span>
          <span>眼别归属：{eye_badge}</span>
          <span>报告编号：{assessment_id}</span>
        </div>
      </div>
      <span class="pill {priority_cls}">{priority_text}</span>
    </div>
  </header>

  <section class="card">
    <h2>主要判读</h2>
    <p class="quality-line"><strong>图像质控：</strong>{escape(review["quality"])}</p>
    <p class="primary">{escape(review["primary"])}</p>
    {dr_scale}
  </section>

  <section class="card">
    <h2>眼底双图可视化对照</h2>
    <main id="image-grid" class="image-grid">
      <figure><figcaption>原图（完整视野）</figcaption><a href="original.jpg" target="_blank" rel="noreferrer"><img src="original.jpg" alt="本次眼底原图"></a></figure>
      <figure><figcaption>{caption}</figcaption>{right}<div class="key">{keys}</div></figure>
    </main>
    <p class="note-line"><strong>{escape(info["source_label"])}：</strong>{escape(info["note"])}</p>
    {credit}
  </section>

  {care_section}
  {review_block}

  <section class="card">
    <h2>其余结果与适用范围</h2>
    {other}
    <p class="scope-note">{escape(review["scope"])} 临床诊断与治疗决策需由执业眼科医生结合完整专科检查完成。</p>
  </section>

  <section class="card">
    <h2>专业技术附录</h2>
    <details id="tech-details">
      <summary>查看逐项模型得分和研究筛查线（导出 PDF 时将自动完整展开）</summary>
      <div class="scroll"><table><thead><tr><th>项目</th><th>模型</th><th>得分</th><th>研究线</th><th>比较结果</th></tr></thead>
      <tbody>{rows}</tbody></table></div>
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
    quality = (response.get("results") or {}).get("quality") or {}
    if (quality.get("status") != "completed" or (quality.get("output") or {}).get("predicted_label")
            not in ("Good", "Usable")):
        return {"kind": "unavailable", "localization_status": "unavailable", "source_label": "图像可判读性不足",
                "candidate_status": "quality_blocked", "reference_grade": None, "legend": [],
                "note": "本次图像未通过质控，请核对原图或重拍。", "original_url": None, "overlay_url": None, "report_url": None}
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
