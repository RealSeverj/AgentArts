"""Four-level referral guidance for accepted fundus screening evidence."""
from __future__ import annotations

from dataclasses import dataclass

from inference.clinical_review import REFERENCE_SHA
from inference.report_summary import DR_LEVELS, LABELS


TITLES = {
    "L4": "紧急 / 优先转诊级",
    "L3": "限期专科复核级",
    "L2": "轻度 / 常规观察级",
    "L1": "正常 / 低风险级",
}

SCOPE = "本级别用于眼底筛查分诊，依据原图、病史及专科检查确认诊断和实际转诊时限。"
SAFETY_NOTE = "如果视力突然下降、眼睛剧痛、眼前突然出现大量黑点或闪光，或视野像被帘子挡住，请立即去眼科急诊。"
RETINA = "眼底病专科"
EYE_CLINIC = "眼科专科门诊"
PRIORITY_WINDOW = "尽快转诊，筛查流程参考 1–2 周内"
SPECIALIST_WINDOW = "2 周至 1 个月内"
DR_CHECKS = ("黄斑 OCT", "散瞳眼底检查", "由专科酌情安排眼底荧光素血管造影（FFA）")
DISC_CHECKS = ("眼压测量（IOP）", "自动视野检查", "视盘 RNFL-OCT")
MACULA_CHECKS = ("黄斑 OCT，必要时加做 OCTA", "散瞳眼底检查，必要时行三面镜检查")


@dataclass(frozen=True)
class ReferralRule:
    level: str | None
    timeframe: str
    destination: str
    basis: str
    tests: tuple[str, ...]


# Within a level, the shortest window is listed first, followed by findings
# with the same window. This order determines the headline recommendation.
RULES = {
    "ODE": ReferralRule("L4", "尽快联系专科，建议 24 小时内评估", "眼科急诊或神经眼科",
                        "视盘水肿线索需鉴别颅内压增高、视神经炎及缺血性视神经病变。"
                        "临床怀疑颅内压增高时，当日急诊评估。",
                        ("血压测量", "视盘 OCT", "视野检查", "怀疑颅内压增高时由急诊医生安排颅脑影像")),
    "DR_SEVERE_REVIEW": ReferralRule("L4", PRIORITY_WINDOW, RETINA,
                                     "现有证据提示可能低估重度糖网，需人工复核出血象限、静脉串珠、"
                                     "视网膜内微血管异常及新生血管。", DR_CHECKS),
    "DR4": ReferralRule("L4", PRIORITY_WINDOW, RETINA,
                        "增殖期糖网线索提示新生血管及玻璃体积血、牵拉性视网膜脱离或新生血管性青光眼风险。",
                        DR_CHECKS),
    "DR3": ReferralRule("L4", PRIORITY_WINDOW, RETINA,
                        "重度非增殖期糖网线索提示缺血及向增殖期进展的风险，需及时确认分期与黄斑受累情况。",
                        DR_CHECKS),
    "RVO": ReferralRule("L4", PRIORITY_WINDOW, RETINA,
                        "静脉阻塞线索需尽快评估黄斑水肿、缺血及新生血管风险，并结合血压等全身情况评估。",
                        ("黄斑 OCT", "散瞳眼底检查", "由专科酌情安排 FFA", "血压测量")),
    "DR2": ReferralRule("L3", SPECIALIST_WINDOW, RETINA,
                        "中度非增殖期糖网需建立专科随访并评估糖尿病性黄斑水肿。", MACULA_CHECKS),
    "glaucoma": ReferralRule("L3", SPECIALIST_WINDOW, "青光眼专科",
                             "彩照提供视盘形态线索，确诊需要结合眼压、视神经结构和视功能检查。", DISC_CHECKS),
    "OPTIC_DISC_CUPPING": ReferralRule("L3", SPECIALIST_WINDOW, "青光眼专科",
                                       "杯盘比及盘沿改变需结合眼压和视野，鉴别生理性大视杯与青光眼性损害。",
                                       DISC_CHECKS),
    "ODP": ReferralRule("L3", SPECIALIST_WINDOW, "神经眼科或眼科专科门诊",
                        "视盘苍白可提示视神经受损，需要结合病史追查缺血、压迫等病因。", DISC_CHECKS),
    "AMD": ReferralRule("L3", SPECIALIST_WINDOW, RETINA,
                        "眼底彩照提示黄斑改变，需用 OCT 判断积液及新生血管相关改变。"
                        "临床怀疑活动性湿性 AMD 时，建议 1 个工作日内发起黄斑专科转诊。", MACULA_CHECKS),
    "CSR": ReferralRule("L3", SPECIALIST_WINDOW, RETINA,
                        "需用断层成像核实视网膜下液及色素上皮改变，并结合病程复核。", MACULA_CHECKS),
    "MYOPIC_FUNDUS": ReferralRule("L3", SPECIALIST_WINDOW, RETINA,
                                  "近视性眼底改变需排查黄斑劈裂、近视性脉络膜新生血管及周边视网膜变性或裂孔。",
                                  MACULA_CHECKS),
    "LS": ReferralRule("L2", "3–6 个月专科随访，结合原发病与既往医嘱", RETINA,
                       "激光瘢痕线索常见于既往光凝治疗，需核对原发病和治疗史，检查周边有无活动性病灶。",
                       ("核对既往眼底光凝及复查记录", "散瞳眼底检查或眼底彩照对比")),
    "DR1": ReferralRule("L2", "6–12 个月复查", EYE_CLINIC,
                        "轻度非增殖期糖网通常以微动脉瘤为主要表现，重点是控制血糖、血压和血脂。",
                        ("眼底彩照对比", "糖尿病患者按医嘱监测糖化血红蛋白（HbA1c）")),
    "DN": ReferralRule("L2", "6–12 个月复查", EYE_CLINIC,
                       "需观察玻璃膜疣大小、数量及形态变化，并关注新发视物变形。",
                       ("眼底彩照对比", "必要时由医生指导使用 Amsler 方格表")),
    "TSLN": ReferralRule("L2", "12 个月常规复查", EYE_CLINIC,
                         "豹纹状眼底是可见脉络膜纹理的形态表现，常见于近视，需结合屈光情况和既往图像。",
                         ("眼底彩照对比",)),
}

FINDING_NAMES = {
    "glaucoma": "青光眼风险",
    "DR_SEVERE_REVIEW": "重度糖网升阶复核预警",
    **{f"DR{grade}": f"糖网 {grade} 级（{DR_LEVELS[grade][1]}）" for grade in range(1, 5)},
}

MODEL_NAMES = {"multidisease": "五标签模型", "multilabel_11": "十一标签模型", "glaucoma": "青光眼模型"}


def _item(code, name, evidence, rule):
    return {"code": code, "name": name, "evidence": evidence, "basis": rule.basis, "level": rule.level,
            "timeframe": rule.timeframe, "destination": rule.destination, "tests": list(rule.tests)}


def _decision(status, items, *, timeframe=None, destination=None, title=None):
    primary = items[0] if items else {}
    level = primary.get("level")
    return {"level": level, "title": TITLES.get(level, title), "status": status,
            "timeframe": primary.get("timeframe", timeframe), "destination": primary.get("destination", destination),
            "trigger_items": items, "scope": SCOPE, "safety_note": SAFETY_NOTE}


def _severe_dr_evidence(screening, dr_grade):
    alert = (screening.get("report_summary") or {}).get("dr_severe_review_alert")
    if isinstance(alert, dict) and alert.get("triggered") is True:
        basis = alert.get("basis")
        return basis.strip() if isinstance(basis, str) and basis.strip() else "已记录重度糖网人工复核预警。"
    visual = screening.get("visual_report") or {}
    if dr_grade == 2 and screening.get("image_sha256") == REFERENCE_SHA and visual.get("reference_grade") == 3:
        return "本图糖网模型预测为 2 级，同图公开数据集专家标注为 3 级。保留模型原始分期。"
    return None


def build_clinical_decision(screening, predictions, dr_grade, *, quality_ok, urgent_hits):
    if urgent_hits:
        rule = ReferralRule("L4", "立即（当日尽快）", "眼科急诊",
                            "警示症状需要立即评估急性视力威胁，包括视网膜裂孔、脱离或眼底出血。",
                            ("按急诊医生判断行散瞳眼底检查及其他检查",))
        return _decision("emergency", [_item("URGENT_SYMPTOMS", "急症警示症状", "；".join(urgent_hits) + "。", rule)])
    if not quality_ok:
        return _decision("image_ungradable", [], title="暂无法分级 · 需核实图像质量",
                         timeframe="重拍或人工复核后评估", destination=EYE_CLINIC)

    comparisons = {}
    for prediction in predictions:
        comparisons.setdefault(prediction["label_code"], []).append(prediction)

    evidence = {}
    for code, entries in comparisons.items():
        if any(entry["screening_flag"] == "positive" for entry in entries):
            evidence[code] = "；".join(
                MODEL_NAMES[entry["source_model"]] + ("达到研究筛查线" if entry["screening_flag"] == "positive"
                                                     else "未达到研究筛查线")
                for entry in entries if entry["screening_flag"] != "undetermined"
            ) + "。"
    if dr_grade is not None and dr_grade > 0:
        evidence[f"DR{dr_grade}"] = f"糖网专病模型预测为 {dr_grade} 级。"
    alert = _severe_dr_evidence(screening, dr_grade)
    if alert:
        evidence["DR_SEVERE_REVIEW"] = alert

    items = []
    for code, rule in RULES.items():
        if code in evidence:
            name = FINDING_NAMES.get(code) or LABELS[code][0]
            items.append(_item(code, name, evidence[code], rule))
    if items:
        return _decision("graded", items)

    all_below = bool(predictions) and all(item["screening_flag"] == "negative" for item in predictions)
    if dr_grade == 0 and all_below:
        rule = ReferralRule("L1", "12 个月常规复查", "常规眼科体检",
                            "当前照片在筛查覆盖范围内未提示明显异常。慢病或其他眼病高风险者遵循医生安排的复查计划。",
                            ("常规眼底筛查",))
        item = _item("SCREEN_NEGATIVE", "本次筛查低风险",
                     "图像质控合格，糖网分期为 0 级，全部多病种筛查项均低于各自研究筛查线。", rule)
        return _decision("graded", [item])

    review_reasons = [f"{LABELS[code][0]}筛查提示需复核。{evidence[code]}" for code in ("DR", "ABNORMAL")
                      if code in evidence]
    if dr_grade is None:
        review_reasons.append("缺少有效的糖网专病分期。")
    if any(item["screening_flag"] == "undetermined" for item in predictions):
        review_reasons.append("有筛查项目未获得可比较的结果。")
    review_rule = ReferralRule(None, "请联系眼科复阅并补齐结果", EYE_CLINIC,
                               "需结合原图确认异常线索和随访优先级。", ("复阅眼底原图及已完成的筛查结果",))
    item = _item("INCOMPLETE_SCREENING", "待专科复核", "".join(review_reasons), review_rule)
    return _decision("review_required", [item], title="暂无法分级 · 待专科复核")
