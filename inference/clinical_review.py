"""Translate this image's graded, screening and spatial outputs into reading tasks."""
from inference.report_summary import DR_LEVELS, ELEVEN_LABELS, FIVE_LABELS, LABELS


REFERENCE_SHA = "092d05e8bb47568b511d896dd8ec79dccf9f5dd53bcb8d3ce5ec1a6c38d3e1b3"
NAMES = {name: code for code, (name, _) in LABELS.items()}
LESION_NAMES = {"MA": "微动脉瘤", "HE": "视网膜出血", "EX": "硬性渗出", "SE": "棉絮斑"}
AREAS = (
    ("视盘与视神经", ("glaucoma", "OPTIC_DISC_CUPPING", "ODP", "ODE"),
     "在原图核对视盘盘缘的形态、颜色及边界。"),
    ("黄斑区", ("AMD", "DN", "CSR"), "放大原图核对黄斑区的形态和色素改变，并与既往影像对照。"),
    ("视网膜静脉", ("RVO",), "沿静脉走行复核血管形态和相应区域有无出血，记录可见分布。"),
    ("近视性眼底表现", ("MYOPIC_FUNDUS", "TSLN"), "对照原图核实后极部纹理与脉络膜显露范围。"),
    ("激光治疗痕迹", ("LS",), "在原图核实有无光凝斑，并对照既往眼底光凝记录。"),
)
DR_ACTIONS = (
    "放大原图核对是否有微动脉瘤或视网膜出血。",
    "放大原图核对微动脉瘤，检查有无支持更高分期的病变。",
    "在可见视野核对出血的象限分布，并检查静脉串珠及视网膜内微血管异常。",
    "在可见视野核对出血分布、静脉串珠和视网膜内微血管异常，并检查有无新生血管。",
    "复核视盘和视网膜有无新生血管、视网膜前或玻璃体积血。",
)


def screening_groups(summary):
    research = summary.get("research") or {}
    return (("五标签模型", summary.get("five_label_screening") or {}, FIVE_LABELS),
            ("十一标签模型", research.get("multilabel_11") or {}, ELEVEN_LABELS),
            ("青光眼模型", research.get("glaucoma") or {}, ("glaucoma",)))


def _signals(summary):
    signals, missing = {}, []
    for source, group, expected in screening_groups(summary):
        if group.get("status") != "completed":
            missing.append(source + ("运行失败" if group.get("status") == "error" else "未评估"))
            continue
        compared = set()
        for item in group.get("items", []):
            code = item.get("code") or NAMES.get(item.get("name"))
            met = item.get("research_threshold_met")
            if code in expected and type(met) is bool:
                compared.add(code)
                signals.setdefault(code, {})[source] = met
        if len(compared) < len(expected):
            missing.append(f"{source}有{len(expected) - len(compared)}项缺少可比较的得分或筛查线")
    return signals, missing


def _basis(codes, signals):
    labels = ["青光眼筛查" if code == "glaucoma" else LABELS[code][0] for code in codes]
    basis = "、".join(labels) + "达到研究筛查线。"
    for code in codes:
        sources = signals[code]
        if False in sources.values():
            attribution = "、".join(source + ("达到" if met else "未达到") for source, met in sources.items())
            basis += f"{LABELS[code][0]}的筛查提示不一致（{attribution}）。"
    return basis


def _item(topic, basis, action):
    return {"topic": topic, "basis": basis, "action": action}


def build_clinical_review(response, visual=None):
    """Return the same physician-facing narrative fields for the HTML and the agent."""
    quality = (response.get("results") or {}).get("quality") or {}
    label = (quality.get("output") or {}).get("predicted_label") if quality.get("status") == "completed" else None
    review = {"quality": "", "primary": "", "review_items": [], "other_results": [], "scope": ""}
    if label not in ("Good", "Usable"):
        review["quality"] = "图像质量不足。" if label == "Reject" else "图像可判读性未确认。"
        review["primary"] = ("请重拍清晰的眼底照片，再进行分期和病灶分析。" if label == "Reject"
                             else "请先人工核实图像质量，或重新进行质控。")
        return review

    review["quality"] = "图像质量良好，可判读。" if label == "Good" else "图像可判读，质量受限。"
    summary = response.get("report_summary") or {}
    signals, missing = _signals(summary)
    reached = {code for code, sources in signals.items() if any(sources.values())}
    visual = (response.get("visual_report") if visual is None else visual) or {}
    dr = summary.get("dr_severity") or {}
    grade = dr.get("grade") if dr.get("status") == "completed" else None
    grade = grade if type(grade) is int and 0 <= grade <= 4 else None
    ref = visual.get("reference_grade") if response.get("image_sha256") == REFERENCE_SHA else None
    ref = ref if type(ref) is int and ref == 3 else None
    review["primary"] = (f"糖网专病模型预测为{grade}级，{DR_LEVELS[grade][1]}。" if grade is not None
                         else "本次未取得有效的糖网专病分期。")
    if ref is not None:
        review["primary"] += f"同图的公开数据集专家标注为{ref}级，{DR_LEVELS[ref][1]}。"
        if grade is not None and grade != ref:
            direction = "低" if grade < ref else "高"
            review["primary"] += f"模型分期较专家标注{direction}{abs(grade - ref)}级。"

    dr_evidence = []
    if grade == 0 and "DR" in reached:
        dr_evidence.append("五标签模型的糖网项达到研究筛查线，与专病模型的0级判断不一致。")
    elif grade is not None and grade > 0 and False in signals.get("DR", {}).values():
        dr_evidence.append("五标签模型的糖网项未达研究筛查线，与专病分期提示不一致。")
    elif grade is None and "DR" in reached:
        dr_evidence.append("五标签模型的糖网项达到研究筛查线。")
    candidates = []
    if visual.get("localization_status") == "localized" and visual.get("candidate_status") == "candidates_present":
        legend = {entry.get("code") for entry in visual.get("legend", [])}
        candidates = [name for code, name in LESION_NAMES.items() if code in legend]
        if candidates:
            dr_evidence.append("分割模型标出" + "、".join(candidates) + "候选区。")
    if grade is None or grade > 0 or dr_evidence or (ref is not None and ref != grade):
        action = DR_ACTIONS[grade] if grade is not None else "对照原图人工评估糖网分期。"
        if grade == 0 and candidates:
            action = "对照着色候选区与同位置原图，核实病灶形态并复核0级判断。"
        if {grade, ref} == {2, 3}:
            action = DR_ACTIONS[2]
        review["review_items"].append(_item("糖网分期", "".join(dr_evidence), action))

    for topic, codes, action in AREAS:
        triggers = [code for code in codes if code in reached]
        if triggers:
            if "glaucoma" in triggers:
                action += "结合已有眼压及视野资料评估青光眼线索。"
            review["review_items"].append(_item(topic, _basis(triggers, signals), action))
    if "ABNORMAL" in reached and not reached.difference({"ABNORMAL"}) and not review["review_items"]:
        review["review_items"].append(_item("全图复阅", "综合异常指标达到研究筛查线，缺少具体定位。",
                                         "依次复阅后极部、视盘和可见血管，记录实际所见。"))
    below = sum(met is False for sources in signals.values() for met in sources.values())
    if below:
        review["other_results"].append(f"{below}项模型与标签的比较低于各自研究筛查线，详见明细。" if reached
                                       else "已完成的研究筛查未触发额外复核项目。")
    if missing:
        review["other_results"].append("本次评估缺项：" + "，".join(missing) + "。")
    if visual.get("candidate_status") == "no_candidates":
        review["other_results"].append("四类病灶定位已完成，本图未生成可绘制的候选区。")
    review["scope"] = "研究筛查线用于选择复阅项目。病变性质和分期由阅片医生结合原图及临床资料确认。"
    return review
