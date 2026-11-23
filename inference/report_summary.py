"""Build display fields from the current image's completed inference stages."""
from __future__ import annotations

from decimal import Decimal

# Display labels apply only to the DR grading task. Keep the original clinical stage.
DR_LEVELS = (
    ("未见明显糖网病变（本项正常档）", "未见明显糖尿病视网膜病变"),
    ("轻度（非增殖期）", "轻度非增殖期糖尿病视网膜病变"),
    ("中度（非增殖期）", "中度非增殖期糖尿病视网膜病变"),
    ("重度（非增殖期）", "重度非增殖期糖尿病视网膜病变"),
    ("严重（增殖期，最高档）", "增殖期糖尿病视网膜病变"),
)

# Names follow the archived RFMiD label dictionary. RVO is the grouped label.
LABELS = {
    "DR": ("糖尿病视网膜病变", "疾病"),
    "AMD": ("年龄相关性黄斑变性", "疾病"),
    "MYOPIC_FUNDUS": ("近视性眼底改变", "影像表现"),
    "OPTIC_DISC_CUPPING": ("视盘凹陷表型", "影像表现"),
    "ABNORMAL": ("总体异常", "综合标签"),
    "RVO": ("视网膜静脉阻塞", "疾病"),
    "LS": ("激光瘢痕", "治疗痕迹"),
    "DN": ("玻璃膜疣", "影像表现"),
    "CSR": ("中心性浆液性脉络膜视网膜病变", "疾病"),
    "ODE": ("视盘水肿", "影像表现"),
    "ODP": ("视盘苍白", "影像表现"),
    "TSLN": ("豹纹状眼底", "影像表现"),
    "glaucoma": ("青光眼", "疾病"),
}
FIVE_LABELS = ("DR", "AMD", "MYOPIC_FUNDUS", "OPTIC_DISC_CUPPING", "ABNORMAL")
ELEVEN_LABELS = ("RVO", "LS", "DN", "AMD", "CSR", "OPTIC_DISC_CUPPING",
                 "ODE", "ODP", "MYOPIC_FUNDUS", "TSLN", "ABNORMAL")

# Frozen validation work points, in the original 0–1 output scale. The model 5
# fingerprint identifies the ensemble recipe, which pins both member weights.
RESEARCH_LINES = {
    # model3_multidisease/outputs/validation_lock.json: candidates.original.thresholds
    "multidisease": (
        "multidisease_rfmid_vitb14_392_best21_20261114",
        "f5352ccb0834b41a2d9e6ed6d43c80b7b7b2e87dcfa1c0b3690f96eddafca640",
        {"DR": 0.043762970715761185, "AMD": 0.005200723651796579,
         "MYOPIC_FUNDUS": 0.06336989253759384, "OPTIC_DISC_CUPPING": 1.2753240298479795e-05,
         "ABNORMAL": 0.8418160676956177},
    ),
    # 07_模型5云端优化结果_20261122/ensemble/thresholds.json: per_label
    "multilabel_11": (
        "rfmid11_vitl_vitb_disc_mean_20261122",
        "22568735bd84cced4eb5be06ab8ef45139a35d58ea139994681338d4171a2edf",
        {"RVO": 0.011820118874311447, "LS": 0.0009931727981893346, "DN": 0.0675105344853364,
         "AMD": 0.006723892467562109, "CSR": 0.0720701176323928, "OPTIC_DISC_CUPPING": 0.0009663661439844873,
         "ODE": 0.030557316611520946, "ODP": 0.0009341061486338731, "MYOPIC_FUNDUS": 0.07395362388342619,
         "TSLN": 0.05557347844296601, "ABNORMAL": 0.7770775258541107},
    ),
    # model6_glaucoma/vitl392/final_selection.json: validation.historical
    "glaucoma": (
        "glaucoma_historical_vitb392_selected_20261122",
        "378fc1a1e15e4f67e016bacaa49fd1b81e016c973043919e3a0a290edadc1aca",
        {"glaucoma": 0.14928317070007324},
    ),
}


def _score_fields(raw):
    if type(raw) not in (float, int) or not 0 <= raw <= 1:
        return {"score": None, "score_display": None}
    display = f"{100 * raw:.3f}"
    if 0 < raw < 0.00001:
        display = "<0.001"
    elif raw < 1 and display == "100.000":
        display = ">99.999"
    return {"score": raw, "score_display": display + "分"}


def _screening_score(raw, threshold):
    fields = _score_fields(raw)
    fields.update(research_threshold=threshold, research_threshold_display=None,
                  research_threshold_met=None, research_threshold_status="无法比较")
    if threshold is None:
        return fields
    line = Decimal(str(threshold)).scaleb(2)
    places = 6
    fields["research_threshold_display"] = f"{line:.{places}f}分"
    if fields["score"] is None:
        return fields
    reached = raw >= threshold
    fields.update(research_threshold_met=reached,
                  research_threshold_status="达到研究筛查线" if reached else "未达到研究筛查线")
    # Resolve a display tie near the line without changing the raw comparison.
    score = Decimal(str(raw)).scaleb(2)
    if abs(score - line) < Decimal("0.001") and not fields["score_display"].startswith(("<", ">")):
        while score != line and round(score, places) == round(line, places):
            places += 1
        fields["score_display"] = f"{score:.{places}f}分"
        fields["research_threshold_display"] = f"{line:.{places}f}分"
    return fields


def _dr_report(stage, quality_block):
    grade_result = {"name": LABELS["DR"][0], "source": "dr_grading",
                    "status": "not_run", "reason": quality_block or stage.get("reason"),
                    "grade": None, "level": None, "clinical_stage": None, **_score_fields(None)}
    if quality_block:
        return grade_result
    grade_result["status"] = stage.get("status", "not_run")
    if grade_result["status"] != "completed":
        return grade_result
    output = stage.get("output") or {}
    grade = output.get("predicted_grade")
    if type(grade) is not int or not 0 <= grade < len(DR_LEVELS):
        grade_result.update(status="ungradable", reason="invalid_grade")
        return grade_result
    level, clinical_stage = DR_LEVELS[grade]
    grade_result.update(grade=grade, level=level, clinical_stage=clinical_stage)
    order, scores = output.get("grade_order"), output.get("scores")
    if (isinstance(order, list) and isinstance(scores, list) and len(order) == len(scores) == 5
            and all(type(value) is int for value in order) and set(order) == set(range(5))):
        grade_result.update(_score_fields(scores[order.index(grade)]))
    return grade_result


def _label_report(stage, stage_name, codes, quality_block):
    status = "not_run" if quality_block else stage.get("status", "not_run")
    report = {"status": status, "reason": quality_block or stage.get("reason"),
              "model_id": stage.get("model_id"), "items": []}
    if status != "completed":
        return report
    model_id, fingerprint, lines = RESEARCH_LINES[stage_name]
    if (stage.get("model_id"), stage.get("weights_sha256")) != (model_id, fingerprint):
        lines = {}
    scores = (stage.get("output") or {}).get("scores") or {}
    for code in codes:
        name, kind = LABELS[code]
        if code == "DR":
            grading = "见糖网专病分级"
        elif kind == "疾病":
            grading = "暂无法分级"
        else:
            grading = "该标签不适用疾病分级"
        report["items"].append({"code": code, "name": name, "kind": kind,
                                **_screening_score(scores.get(code), lines.get(code)),
                                "severity_grade": None, "grading": grading})
    return report


def build_report_summary(results: dict, research_results: dict | None = None) -> dict:
    """Provide disease-specific grades and separately attributed screening scores.

    A display score is the raw model output scaled to 100. It never determines
    a disease stage. Missing research stages also cover older service responses.
    """
    quality = results.get("quality") or {}
    quality_label = (quality.get("output") or {}).get("predicted_label")
    if quality.get("status") != "completed" or quality_label not in ("Good", "Usable", "Reject"):
        quality_block = "quality_unavailable"
    elif quality_label == "Reject":
        quality_block = "quality_rejected"
    else:
        quality_block = None
    research = research_results or {}
    missing = {"status": "not_run", "reason": "not_configured"}
    return {
        "severity_scale": {
            "applies_to": "糖尿病视网膜病变",
            "levels": [{"grade": i, "level": level, "clinical_stage": stage}
                       for i, (level, stage) in enumerate(DR_LEVELS)],
        },
        "dr_severity": _dr_report(results.get("dr_grading") or missing, quality_block),
        "five_label_screening": _label_report(results.get("multidisease") or missing, "multidisease",
                                              FIVE_LABELS, quality_block),
        "research": {
            "scope": "分数与对应模型的研究筛查线比较。达到或未达到均不能据此判定疾病阳性、阴性或严重程度。",
            "multilabel_11": _label_report(research.get("multilabel_11") or missing,
                                          "multilabel_11", ELEVEN_LABELS, quality_block),
            "glaucoma": _label_report(research.get("glaucoma") or missing, "glaucoma", ("glaucoma",), quality_block),
        },
    }
