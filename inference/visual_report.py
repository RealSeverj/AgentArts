"""Render a same-view image pair from spatial evidence tied to this request."""
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


def _report_html(info, response, has_overlay):
    review = build_clinical_review(response, info)
    rows = _score_rows(response)
    tasks = "".join("<li><strong>" + escape(item["topic"]) + "</strong>　" + escape(item["action"]) +
                    ("<div class='basis'>" + escape(item["basis"]) + "</div>" if item["basis"] else "") +
                    "</li>" for item in review["review_items"])
    review_block = "<section><h2>阅片意见</h2><ol>" + tasks + "</ol></section>" if tasks else ""
    other = "".join("<p>" + escape(line) + "</p>" for line in review["other_results"])
    caption = {"expert_reference": "病灶标注图（专家参考）", "model_candidates": "病灶候选区（模型预测）",
               "anatomy": "视盘视杯结构图", "no_candidates": "本次未检出可标注候选区",
               "unavailable": "本次无法定位"}[info["kind"]]
    right = '<a href="overlay.jpg"><img src="overlay.jpg" alt="' + caption + '"></a>' if has_overlay else ""
    keys = "".join('<span><i style="background:' + x["color"] + '"></i>' + escape(x["name"]) + '</span>'
                   for x in info["legend"])
    credit = ""
    if info.get("reference_grade") is not None and response.get("image_sha256") == REFERENCE_SHA:
        origin = "右图叠加了公开像素标注。" if info["kind"] == "expert_reference" else "右图显示本次模型运行结果。"
        credit = ('<p>案例来源：<a rel="noreferrer" href="https://doi.org/10.21227/H25W98">IDRiD</a>，分割图 IDRiD_03 与分级图 IDRiD_369 同源同图。'
                  '© Prasanna Porwal、Samiksha Pachade、Manesh Kokare，<a rel="noreferrer" href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>。'
                  + origin + '</p>')
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>眼底影像辅助阅片报告</title><style>
*{{box-sizing:border-box}}body{{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#193941;max-width:1180px;padding:24px;margin:auto}}
h1{{font-size:28px}}h2{{font-size:20px}}main{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}figure{{margin:0;min-width:0}}
.basis{{color:#425861;font-size:.94em}}li{{margin-bottom:.55em}}ol{{padding-inline-start:1.5em}}.primary{{font-size:1.12em}}
img{{width:100%;height:auto;display:block;background:#161a1c}}figcaption{{font-weight:650;margin-bottom:5px}}.key span{{display:inline-block;margin:0 1em .25em 0}}
i{{display:inline-block;width:15px;height:15px;vertical-align:-2px;margin-right:6px;border:1px solid #687e80}}.scroll{{overflow:auto}}
td,th{{text-align:left;border-bottom:1px solid #d6dfdf;padding:5px 14px 5px 0}}td{{font-variant-numeric:tabular-nums}}
@media(max-width:680px){{main{{grid-template-columns:1fr}}body{{padding:14px}}}}</style>
<h1>眼底影像辅助阅片报告</h1><p>{escape(review["quality"])}</p><h2>主要判读</h2><p class="primary">{escape(review["primary"])}</p>{review_block}
<main><figure><figcaption>原图（完整视野）</figcaption><a href="original.jpg"><img src="original.jpg" alt="本次眼底原图"></a></figure>
<figure><figcaption>{caption}</figcaption>{right}<div class="key">{keys}</div></figure></main>
<p><strong>{escape(info["source_label"])}</strong><br>{escape(info["note"])}</p>{credit}<h2>其余结果</h2>{other}<p>{escape(review["scope"])}</p>
<details><summary>查看逐项模型得分和研究筛查线</summary><div class="scroll"><table><thead><tr><th>项目</th><th>模型</th><th>得分</th><th>研究线</th><th>比较结果</th></tr></thead>
<tbody>{rows}</tbody></table></div></details></html>"""


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
