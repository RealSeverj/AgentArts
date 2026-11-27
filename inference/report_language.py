"""Clinician and patient wording for structured referral decisions."""
from __future__ import annotations

import re


CLINICAL_EVIDENCE = {
    "ODE": "眼底筛查提示疑似视盘水肿。",
    "DR_SEVERE_REVIEW": "当前糖网初评分期存在低估风险，需优先复核。",
    "DR4": "糖网暂评为 4 级，提示增殖期改变。",
    "DR3": "糖网暂评为 3 级，提示重度非增殖期改变。",
    "RVO": "眼底筛查提示视网膜静脉阻塞的可能。",
    "DR2": "糖网暂评为 2 级，提示中度非增殖期改变。",
    "glaucoma": "眼底筛查提示青光眼相关风险。",
    "OPTIC_DISC_CUPPING": "眼底筛查提示视盘凹陷相关征象。",
    "ODP": "眼底筛查提示视盘苍白征象。",
    "AMD": "眼底筛查提示年龄相关的黄斑改变。",
    "CSR": "眼底筛查提示中心性浆液性脉络膜视网膜病变的可能。",
    "MYOPIC_FUNDUS": "眼底筛查提示与近视相关的眼底改变。",
    "LS": "眼底可见疑似既往激光治疗留下的瘢痕。",
    "DR1": "糖网暂评为 1 级，提示轻度非增殖期改变。",
    "DN": "眼底筛查提示玻璃膜疣相关征象。",
    "TSLN": "眼底筛查提示豹纹状眼底改变。",
    "SCREEN_NEGATIVE": "照片可判读，糖网暂评为 0 级，已完成的筛查项目均未提示明确异常。",
}

MACULAR_FINDINGS = {"DR2", "DR3", "DR4", "DR_SEVERE_REVIEW", "RVO", "AMD", "CSR", "MYOPIC_FUNDUS"}
DISC_FINDINGS = {"glaucoma", "OPTIC_DISC_CUPPING", "ODP"}


def decision_evidence(item: dict) -> str:
    code = item.get("code")
    if code in CLINICAL_EVIDENCE:
        return CLINICAL_EVIDENCE[code]
    evidence = str(item.get("evidence") or "")
    if code == "URGENT_SYMPTOMS":
        clauses = re.sub(r"（命中警示：[^）]*）", "", evidence).strip("。； ").split("；")
        symptoms = "、".join(dict.fromkeys(clause.strip() for clause in clauses if clause.strip()))
        return f"本次问诊记录了需要立即评估的症状（{symptoms}）。" if symptoms else "本次问诊记录了需要立即评估的症状。"
    if code == "INCOMPLETE_SCREENING":
        reasons = []
        if "糖尿病视网膜病变筛查提示需复核" in evidence:
            reasons.append("糖网筛查线索需结合原图复核。")
        if "总体异常筛查提示需复核" in evidence:
            reasons.append("眼底整体筛查提示异常线索，需人工复阅定位。")
        if "缺少有效的糖网专病分期" in evidence:
            reasons.append("本次暂未取得可用的糖网分期。")
        if "有筛查项目未获得可比较的结果" in evidence:
            reasons.append("部分筛查结果尚待补充。")
        return "".join(reasons) or "当前检查资料有待补充或人工复核。"
    return "请结合眼底原图和就诊时的检查结果复核。"


def patient_followup(decision: dict, intake: dict) -> dict[str, list[str]]:
    level = decision.get("level")
    status = decision.get("status")
    if status == "emergency":
        return {
            "actions": ["请现在去眼科急诊。"],
            "monitoring": ["就医途中如有新出现或加重的症状，到院后告诉医生。"],
            "questions": ["我现在需要做哪些检查？哪些结果需要尽快处理？"],
            "preparation": ["请告诉医生症状何时开始、哪只眼受影响。眼科资料在手边的话可以一并带上。"],
        }

    findings = {item.get("code") for item in decision.get("trigger_items") or []}
    actions = {
        "L4": ["请尽快联系上方建议的科室，确认最早可以就诊的时间。"],
        "L3": ["请预约眼科复核，就诊时把这份报告带给医生。"],
        "L2": ["请按建议时间安排复查，帮助医生了解眼底变化。"],
        "L1": ["请按建议安排下一次常规眼底检查。"],
    }.get(level, ["请联系眼科补齐检查，由医生确认接下来的安排。"])
    if status == "image_ungradable":
        actions = ["请联系拍片点补拍清晰的眼底照片，或带原图请眼科医生复核。"]
    if level in {"L1", "L2", "L3"} and intake.get("confirmed_plan"):
        actions = ["如果近期眼部感受没有变化，请先按医生原来定好的计划复查。"]

    monitoring = ["如果新出现看直线变弯，或读字时中间一片发糊，请尽快联系眼科。"]
    questions = []
    if findings & MACULAR_FINDINGS:
        questions.append("我需要做眼底断层扫描（黄斑 OCT），看看眼底中心有没有变化吗？")
    if findings & DISC_FINDINGS:
        questions.append("我需要量眼压、检查能看见的范围（视野）或进一步查视神经吗？")
    elif "ODE" in findings:
        questions.append("是否需要查血压、视野或拍眼底断层照片来弄清原因？")
    if not questions:
        questions.append("这次还需要补做检查吗？下次复查约在什么时候？")

    preparation = ["可以带上以往的眼底照片、检查单和用药清单，方便医生对照。"]
    if "LS" in findings:
        preparation.append("如果以前做过眼底激光，有治疗记录的话也一起带来。")
    return {"actions": actions, "monitoring": monitoring, "questions": questions, "preparation": preparation}
