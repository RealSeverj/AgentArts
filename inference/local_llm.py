"""Explain eye screening reports with local Qwen or Huawei Cloud MaaS."""
from __future__ import annotations

from contextlib import closing
from http.client import HTTPConnection, HTTPException, HTTPSConnection
import json
import re

from inference.clinical_referral import RULES
from inference.report_language import CLINICAL_EVIDENCE
from inference.report_levels import display_report_level


DEFAULT_LOCAL_LLM_PORT = 8091
LOCAL_LLM_MODEL = "qwen2.5-1.5b-local"
MAAS_HOST = "api.modelarts-maas.com"
MAAS_PATH = "/v2/chat/completions"
MAAS_ENDPOINT = f"https://{MAAS_HOST}{MAAS_PATH}"
MAAS_MODEL = "deepseek-v4.1-flash-1017"
MAX_RESPONSE_BYTES = 65536

REFERRAL_SUMMARIES = {
    "L4": "报告的转诊级别为 L4，需优先安排专科评估。",
    "L3": "报告的转诊级别为 L3，需在报告所列时限内进行眼科专科复核。",
    "L2": "报告的转诊级别为 L2，可按报告所列安排复查。",
    "L1": "报告的转诊级别为 L1，可按报告所列安排常规筛查。",
    "L0": "报告标为 L0，需补充资料或人工复核后再确认转诊级别。",
}
QUALITY_SUMMARIES = {
    "Good": "本次眼底照片质控良好，可用于辅助阅片。",
    "Usable": "本次眼底照片可以判读，图像质量有限。",
    "Reject": "本次照片质控未通过，需要重拍或由医生核实原图。",
}
_POINT_CODES = {text: code for code, text in CLINICAL_EVIDENCE.items()}
_ANALYSIS_TOPICS = (
    ("质控", "质量", "可判读", "清晰度", "图像可用"),
    ("糖网", "糖尿病视网膜病变"),
    ("重度低估", "低估", "复核预警"),
    ("轻度非增殖期", "轻度糖网"),
    ("中度非增殖期", "中度糖网"),
    ("重度非增殖期", "重度糖网"),
    ("视盘水肿", "视乳头水肿"),
    ("视网膜静脉阻塞", "视网膜静脉堵塞"),
    ("青光眼",),
    ("视盘凹陷", "视杯凹陷", "杯盘比"),
    ("视盘苍白", "视神经苍白"),
    ("年龄相关的黄斑改变", "黄斑变性"),
    ("中心性浆液性", "中浆"),
    ("近视",),
    ("激光", "光凝", "瘢痕"),
    ("玻璃膜疣",),
    ("豹纹状眼底", "豹纹眼底"),
    ("候选区域", "候选标注", "病灶图"),
    ("急症症状", "急症警示", "警示症状"),
    ("结果尚缺", "资料不完整", "证据不完整"),
    ("未提示明确异常", "未见明确异常"),
)
_LIMIT = re.compile(r"有限|局限|不足|仍需|尚需|尚不能|尚难|难以|无法|不能|待核|待结合|待确认|仅凭|单凭")
_UNSUPPORTED_CLAIM = re.compile(
    r"已经确诊|已确诊|可以确诊|确认患有|确认为|诊断为|已经发生|病情恶化"
    r"|已排除|可以排除|排除所有|无需复核|无需检查"
    r"|视网膜脱离|黄斑水肿|颅内压|肿瘤|眼底出血|新生血管|视神经萎缩"
    r"|视网膜出血|玻璃体积血|微动脉瘤|硬性渗出|软性渗出"
    r"|治疗方案|手术|用药|停药|给药|口服|服用|注射|滴眼|剂量|疗程|复诊时间|何时就诊"
    r"|立即|尽快|限期|紧急|优先转诊|急诊|延期|择期|不必就医|转诊级别|提高分诊|降低分诊"
    r"|[一二三四五六七八九十两半]+(?:级|小时|分钟|天|周|个月|年)"
    r"|(?:不存在|没有|未提示|未见|未发现|不支持|无)(?:明确的?|明显的?)*"
    r"(?:低估风险|青光眼(?:相关)?风险|视盘水肿|视盘凹陷|视盘苍白|视网膜静脉阻塞|黄斑变性)"
)


class LocalExplanationUnavailable(RuntimeError):
    """An explanation provider cannot return a verified selection."""


def _report_points(screening: dict, decision: dict) -> dict[str, str]:
    if decision.get("status") == "emergency":
        return {"A": "问诊记录中有需要立即评估的急症症状。"}

    quality = (screening.get("results") or {}).get("quality") or {}
    label = (quality.get("output") or {}).get("predicted_label") if quality.get("status") == "completed" else None
    points = [QUALITY_SUMMARIES.get(label, "本次照片的可判读性仍待确认。")]
    if label not in {"Good", "Usable"}:
        return {"A": points[0]}

    grade_result = (screening.get("report_summary") or {}).get("dr_severity") or {}
    grade = grade_result.get("grade") if grade_result.get("status") == "completed" else None
    if type(grade) is int and 0 <= grade <= 4:
        points.append(
            "糖网影像筛查暂评为 0 级。" if grade == 0 else CLINICAL_EVIDENCE[f"DR{grade}"]
        )

    # Only use recognized finding codes. Narrative reasons and user text never
    # enter the model request.
    for trigger in decision.get("trigger_items") or []:
        code = trigger.get("code")
        if code in {"DR1", "DR2", "DR3", "DR4", "SCREEN_NEGATIVE"}:
            continue
        sentence = CLINICAL_EVIDENCE.get(code)
        if sentence and sentence not in points:
            points.append(sentence)

    visual = screening.get("visual_report") or {}
    if visual.get("localization_status") == "localized" and visual.get("candidate_status") == "candidates_present":
        points.append("病灶图标出了待人工核查的候选区域。")
    return dict(zip("ABCDEFGH", points))


def _report_input(screening: dict, decision: dict) -> tuple[str, dict[str, str]]:
    lead = REFERRAL_SUMMARIES.get(display_report_level(decision), "请查看当前报告中的分诊结论。")
    if decision.get("status") == "emergency":
        lead = "报告建议立即前往眼科急诊。"
    return lead, _report_points(screening, decision)


def _choose_points_payload(lead: str, points: dict[str, str]) -> dict:
    return {
        "model": LOCAL_LLM_MODEL,
        "stream": False,
        "temperature": 0,
        "max_tokens": 80,
        "messages": [
            {
                "role": "system",
                "content": (
                    "从已核实的眼底报告要点中选出最能说明当前分诊结果的 1 至 3 个编号。"
                    "优先选择需要复核的关键异常，再考虑分期或图像质量。"
                    '每个编号至多选一次，按重要性排序，只返回形如 {"ids":["A"]} 的 JSON 对象。'
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"分诊结论": lead, "可选要点": points}, ensure_ascii=False),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "report_points",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "ids": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(points)},
                            "minItems": 1,
                            "maxItems": min(3, len(points)),
                        },
                    },
                    "required": ["ids"],
                    "additionalProperties": False,
                },
            },
        },
    }


def _completion_json(reply: dict, model: str) -> dict:
    if reply.get("model") != model:
        raise LocalExplanationUnavailable("解读服务返回的模型名称与项目配置不一致。")

    try:
        choice = reply["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError
        text = choice["message"]["content"]
        if not isinstance(text, str):
            raise ValueError
        text = text.strip()
        # Some MaaS replies wrap the complete object in a Markdown code fence.
        if text.startswith("```"):
            language, newline, inner = text.partition("\n")
            if newline and language.strip().lower() in {"```", "```json"} and inner.endswith("\n```"):
                text = inner[:-4]
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError
    except (KeyError, IndexError, TypeError, ValueError, RecursionError) as error:
        raise LocalExplanationUnavailable("模型没有返回可核验的解读，请重试。") from error
    return result


def _valid_evidence_ids(ids, evidence: dict, *, maximum: int) -> bool:
    return (isinstance(ids, list) and 1 <= len(ids) <= maximum
            and all(isinstance(key, str) and key in evidence for key in ids)
            and len(set(ids)) == len(ids))


def _verified_selection(reply: dict, points: dict[str, str]) -> list[str]:
    selection = _completion_json(reply, LOCAL_LLM_MODEL)
    ids = selection.get("ids")
    if set(selection) != {"ids"} or not _valid_evidence_ids(ids, points, maximum=3):
        raise LocalExplanationUnavailable("模型没有返回可核验的解读，请重试。")
    return [points[key] for key in ids]


def _analysis_evidence(screening: dict, decision: dict) -> tuple[str, dict[str, dict[str, str]], str | None]:
    lead, points = _report_input(screening, decision)
    status = decision.get("status")
    if status == "review_required":
        points["I"] = "筛查资料不完整或部分结果需要人工复核。"
    elif display_report_level(decision) == "L1":
        points["I"] = CLINICAL_EVIDENCE["SCREEN_NEGATIVE"]

    primary_code = next(iter(decision.get("trigger_items") or []), {}).get("code")
    primary_id = {"emergency": "A", "image_ungradable": "A", "review_required": "I"}.get(status)
    evidence = {}
    for key, text in points.items():
        fact = {"发现": text}
        code = _POINT_CODES.get(text)
        if code == "DR_SEVERE_REVIEW":
            fact["发现"] = "现有报告提示重度糖网低估风险，当前糖网分期仍需人工核对。"
            fact["复核依据"] = "初步分期可能低估病变严重程度，复核原图有助于核对实际分期。"
        rule = RULES.get(code)
        if rule and "复核依据" not in fact:
            # Some rules list conditions to rule out, rather than current findings.
            # Avoid supplying them as material for patient-specific generated prose.
            rationale = rule.basis.split("。", 1)[0]
            if not _UNSUPPORTED_CLAIM.search(rationale):
                fact["复核依据"] = rationale + "。"
        if code is not None and code == primary_code:
            primary_id = key
        evidence[key] = fact
    return lead, evidence, primary_id


def _focused_evidence(evidence: dict[str, dict[str, str]], primary_id: str | None) -> dict[str, dict[str, str]]:
    return {
        key: {"发现": fact["发现"]}
        for key, fact in evidence.items()
        if key in {"A", primary_id}
        or fact["发现"].startswith(("糖网暂评为", "糖网影像筛查暂评为"))
    }


def _analysis_payload(
    lead: str, evidence: dict[str, dict[str, str]], primary_id: str | None, *, repair: bool = False,
) -> dict:
    if repair:
        instruction = (
            "根据列出的少量事实写两句临床分析。第一句沿用首要线索中的筛查术语，说明它为何值得复核。"
            "第二句说明现有证据的局限。仅用汉字和中文标点，不改写分期或转诊安排，不引入其他病名。"
            '只返回 JSON 对象，字段为 "analysis" 和 "evidence_ids"，后者列出使用过的编号并包含首要线索。'
        )
    else:
        instruction = (
            "你协助眼科医生复核筛查报告。用你自己的话写两至三句简短分析。"
            "说明已列的本次表现如何影响医生复核的重点，以及现有资料为什么不足以定论。"
            "如给出首要复核线索，请体现该线索并引用它的证据编号。"
            "只使用证据中的已知表现，复核依据仅供解释，不可写成患者已有的异常。"
            "诊断、转诊级别、科室、时间、检查或治疗安排由原报告独立呈现，不在分析里改写或复述。"
            "分析使用中文和句号，写出临床词语和证据局限，省去数字、字母缩写、引用、网址和身份信息。"
            '只返回 JSON 对象，字段为 "analysis" 和 "evidence_ids"。后一字段仅列出实际使用的证据编号。'
        )
    return {
        "model": MAAS_MODEL,
        "stream": False,
        "temperature": 0,
        "max_tokens": 160 if repair else 240,
        "thinking": {"type": "disabled"},
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(
                {"分诊状态": lead, "首要复核线索": primary_id, "报告证据": evidence}, ensure_ascii=False,
            )},
        ],
    }


def _verified_analysis(reply: dict, evidence: dict[str, dict[str, str]], primary_id: str | None) -> str:
    result = _completion_json(reply, MAAS_MODEL)
    analysis, ids = result.get("analysis"), result.get("evidence_ids")
    message = "华为云 MaaS 返回的分析无法依据报告核验，请重试。"
    if (set(result) != {"analysis", "evidence_ids"} or not isinstance(analysis, str)
            or not _valid_evidence_ids(ids, evidence, maximum=len(evidence))
            or (primary_id is not None and primary_id not in ids)):
        raise LocalExplanationUnavailable(message)

    analysis = analysis.strip()
    sentences = re.split("[。！？]+", analysis.rstrip("。！？"))
    if (not 24 <= len(analysis) <= 260 or not 1 <= len(sentences) <= 3
            or not re.fullmatch(r"[㐀-鿿，。！？、（）“”‘’ \t]+", analysis)
            or not _LIMIT.search(analysis) or _UNSUPPORTED_CLAIM.search(analysis)):
        raise LocalExplanationUnavailable(message)

    findings = [fact["发现"] for fact in evidence.values()]
    cited_findings = [evidence[key]["发现"] for key in ids]
    mentioned = {topic for topic in _ANALYSIS_TOPICS if any(word in analysis for word in topic)}
    # Citations identify the analysis focus. Contextual mentions still have to be
    # supported by a finding from this same report.
    if not any(word in finding for topic in mentioned for word in topic for finding in cited_findings):
        raise LocalExplanationUnavailable(message)
    if any(not any(word in finding for word in topic for finding in findings) for topic in mentioned):
        raise LocalExplanationUnavailable(message)
    evidence_sentences = {text for fact in evidence.values() for text in fact.values()}
    if all(f"{sentence}。" in evidence_sentences for sentence in sentences):
        raise LocalExplanationUnavailable(message)
    return analysis


def _http_json(
    connection: HTTPConnection, method: str, path: str, *, payload: dict | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict]:
    request_headers = dict(headers or {})
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    # The standard-library connection talks directly to its configured host
    # and does not use environment proxy variables or follow redirects.
    with closing(connection) as conn:
        conn.request(method, path, body=body, headers=request_headers)
        with conn.getresponse() as response:
            if response.status != 200:
                return response.status, {}
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Reply exceeds the response budget")
    reply = json.loads(raw)
    if not isinstance(reply, dict):
        raise ValueError("Expected a JSON object")
    return 200, reply


class LocalReportExplainer:
    label = "本地 · Qwen"

    def __init__(self, port: int = DEFAULT_LOCAL_LLM_PORT):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("本地模型端口应在 1 至 65535 之间。")
        self.port = port

    def _request(self, method: str, path: str, *, payload: dict | None = None, timeout: float) -> dict:
        try:
            status, reply = _http_json(HTTPConnection("127.0.0.1", self.port, timeout=timeout),
                                       method, path, payload=payload)
        except (OSError, HTTPException, ValueError, RecursionError) as error:
            raise LocalExplanationUnavailable("无法获得本地模型解读，请检查本机模型服务后重试。") from error
        if status != 200:
            raise LocalExplanationUnavailable("本地模型服务暂时无法处理请求。")
        return reply

    def is_available(self) -> bool:
        try:
            models = self._request("GET", "/v1/models", timeout=2).get("data")
        except LocalExplanationUnavailable:
            return False
        return isinstance(models, list) and any(
            isinstance(model, dict) and model.get("id") == LOCAL_LLM_MODEL for model in models
        )

    def explain(self, screening: dict, decision: dict) -> dict[str, str]:
        if not self.is_available():
            raise LocalExplanationUnavailable("本机 Qwen 眼底报告解读服务尚未就绪，请启动后重试。")

        lead, points = _report_input(screening, decision)
        reply = self._request("POST", "/v1/chat/completions", payload=_choose_points_payload(lead, points), timeout=25)
        sentences = _verified_selection(reply, points)
        return {"model": LOCAL_LLM_MODEL, "explanation": lead + "".join(sentences)}


class MaaSReportExplainer:
    label = "华为 MaaS · DeepSeek"

    def __init__(self, api_key: str | None = None):
        key = (api_key or "").strip()
        self._api_key = key if all("!" <= character <= "~" for character in key) else ""

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _request_analysis(self, payload: dict, *, timeout: float) -> dict:
        try:
            status, reply = _http_json(
                HTTPSConnection(MAAS_HOST, timeout=timeout), "POST", MAAS_PATH, payload=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except (OSError, HTTPException, ValueError, RecursionError) as error:
            raise LocalExplanationUnavailable("暂时无法连接华为云 MaaS 解读服务，请重试。") from error
        if status in {401, 403}:
            raise LocalExplanationUnavailable("华为云 MaaS 鉴权未通过，请检查 API 密钥和模型权限。")
        if status != 200:
            raise LocalExplanationUnavailable(f"华为云 MaaS 解读服务返回 HTTP {status}，请重试。")
        if reply.get("model") != MAAS_MODEL:
            raise LocalExplanationUnavailable("解读服务返回的模型名称与项目配置不一致。")
        return reply

    def explain(self, screening: dict, decision: dict) -> dict[str, str]:
        if not self.is_available():
            raise LocalExplanationUnavailable("华为云 MaaS 解读尚未配置 API 密钥。")

        lead, evidence, primary_id = _analysis_evidence(screening, decision)
        reply = self._request_analysis(_analysis_payload(lead, evidence, primary_id), timeout=25)
        try:
            analysis = _verified_analysis(reply, evidence, primary_id)
        except LocalExplanationUnavailable:
            # Retry the same model once with fewer facts and a more focused prompt.
            focused = _focused_evidence(evidence, primary_id)
            reply = self._request_analysis(_analysis_payload(lead, focused, primary_id, repair=True), timeout=15)
            analysis = _verified_analysis(reply, focused, primary_id)
        return {"model": MAAS_MODEL, "explanation": analysis, "analysis": analysis, "analysis_mode": "generated"}
