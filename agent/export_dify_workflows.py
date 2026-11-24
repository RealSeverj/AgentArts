"""Export standard Dify Workflow DSL (.yml) files for Huawei Cloud AgentArts multi-agent import."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import yaml

DEFAULT_ORIGIN = "https://severj.top"
PLACEHOLDER_API_KEY = "YOUR_RETINA_API_KEY"
DEFAULT_KB_ID = "REPLACE_WITH_AGENTARTS_KB_ID"


class _NoAliasDumper(yaml.SafeDumper):
    """Ensure clean YAML output without &id001/*id001 anchors for cross-platform DSL converters."""

    def ignore_aliases(self, data):
        return True


def origin_url(value: str) -> str:
    url = urlsplit(value)
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.path not in ("", "/")
        or url.fragment
        or url.query
        or any(c.isspace() for c in value)
        or url.hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
        or url.hostname.endswith(".invalid")
    ):
        raise argparse.ArgumentTypeError("Supply a public HTTPS origin without path, credentials, or query")
    return "https://" + url.netloc


def _base_app(name: str, description: str, icon: str) -> dict:
    return {
        "app": {
            "description": description,
            "icon": icon,
            "icon_background": "#EFF6FF",
            "mode": "workflow",
            "name": name,
            "use_icon_as_answer_icon": False,
        },
        "kind": "app",
        "version": "0.1.5",
    }


def _features() -> dict:
    return {
        "file_upload": {
            "enabled": False,
            "image": {"enabled": False, "number_limits": 3, "transfer_methods": ["remote_url", "local_file"]},
        },
        "opening_statement": "",
        "retriever_resource": {"enabled": True},
        "sensitive_word_avoidance": {"enabled": False},
        "speech_to_text": {"enabled": False},
        "text_to_speech": {"enabled": False},
    }


def _node(node_id: str, node_type: str, title: str, x: int, y: int, data: dict, desc: str = "") -> dict:
    payload = {"type": node_type, "title": title, "desc": desc, **data}
    return {
        "id": node_id,
        "type": "custom",
        "position": {"x": x, "y": y},
        "data": payload,
    }


def _edge(source: str, source_type: str, target: str, target_type: str, source_handle: str = "source") -> dict:
    return {
        "id": f"{source}__{source_handle}__to__{target}",
        "source": source,
        "sourceHandle": source_handle,
        "target": target,
        "targetHandle": "target",
        "type": "custom",
        "data": {
            "isInIteration": False,
            "sourceType": source_type,
            "targetType": target_type,
        },
    }


def _var(name: str, label: str, *, required: bool = False, max_length: int = 256, var_type: str = "text-input") -> dict:
    return {
        "variable": name,
        "label": label,
        "type": var_type,
        "required": required,
        "max_length": max_length,
        "options": [],
    }


def _http_node(
    node_id: str,
    title: str,
    x: int,
    y: int,
    *,
    method: str,
    url: str,
    api_key: str,
    json_body_template: str | None = None,
    read_timeout: int = 30,
) -> dict:
    headers_lines = [f"X-API-Key: {api_key}"]
    if method.lower() == "post":
        headers_lines.insert(0, "Content-Type: application/json")
        body = {
            "type": "json",
            "data": [
                {
                    "id": f"{node_id}_json_body",
                    "key": "",
                    "type": "text",
                    "value": json_body_template or "{}",
                }
            ],
        }
    else:
        body = {"type": "none", "data": []}

    return _node(
        node_id,
        "http-request",
        title,
        x,
        y,
        {
            "method": method.lower(),
            "url": url,
            "authorization": {"type": "no-auth", "config": None},
            "headers": "\n".join(headers_lines),
            "params": "",
            "body": body,
            "timeout": {
                "max_connect_timeout": 10,
                "max_read_timeout": read_timeout,
                "max_write_timeout": 30,
            },
            "ssl_verify": True,
        },
    )


def _llm_node(
    node_id: str,
    title: str,
    x: int,
    y: int,
    *,
    system_prompt: str,
    user_prompt: str,
    context_selector: list[str] | None = None,
) -> dict:
    ctx = (
        {"enabled": True, "variable_selector": context_selector}
        if context_selector
        else {"enabled": False, "variable_selector": []}
    )
    return _node(
        node_id,
        "llm",
        title,
        x,
        y,
        {
            "model": {
                "provider": "deepseek",
                "name": "deepseek-chat",
                "mode": "chat",
                "completion_params": {"temperature": 0.2},
            },
            "prompt_template": [
                {"id": f"{node_id}_sys", "role": "system", "text": system_prompt},
                {"id": f"{node_id}_usr", "role": "user", "text": user_prompt},
            ],
            "context": ctx,
            "vision": {"enabled": False},
            "memory": {"role_prefix": {"user": "", "assistant": ""}},
            "selected_variables": [],
        },
    )


def _kb_node(node_id: str, title: str, x: int, y: int, query_selector: list[str]) -> dict:
    return _node(
        node_id,
        "knowledge-retrieval",
        title,
        x,
        y,
        {
            "query_variable_selector": query_selector,
            "dataset_ids": [DEFAULT_KB_ID],
            "retrieval_mode": "multiple",
            "multiple_retrieval_config": {
                "top_k": 5,
                "score_threshold": 0.35,
                "reranking_enable": False,
            },
        },
    )


def build_w00_context(origin: str, api_key: str) -> dict:
    """W00: Multi-agent initial workflow - load session context, detect audience and urgent red flags."""
    doc = _base_app(
        "W00_上下文与急症前置检查",
        "多智能体起始工作流：每轮必先加载当前会话绑定检查与近期列表，识别受众身份，并抢先扫描眼科急症警示症状。",
        "W0",
    )
    nodes = [
        _node(
            "start",
            "start",
            "开始",
            80,
            240,
            {
                "variables": [
                    _var("query", "用户本轮输入", required=True, max_length=2000, var_type="paragraph"),
                    _var("session_id", "会话ID（默认 default）", required=False, max_length=128),
                ]
            },
        ),
        _http_node(
            "http_context",
            "获取当前检查上下文",
            360,
            240,
            method="get",
            url=f"{origin}/v1/context?session_id={{{{#start.session_id#}}}}",
            api_key=api_key,
            read_timeout=15,
        ),
        _llm_node(
            "llm_triage_entry",
            "入口状态与急症受众识别",
            660,
            240,
            system_prompt=(
                "你是眼底筛查多智能体应用的起始路由预处理助手。禁止使用任何 Emoji。\n"
                "根据用户输入和后端返回的上下文 JSON，提炼如下结构化摘要供下游子工作流引用：\n"
                "1. active_assessment_id 与 eye（若无记录则明确写无）；\n"
                "2. urgent_flag：若用户文本含突发大量飞蚊/黑影飘动、闪光感、黑幕样视野遮挡、突发视力骤降或剧烈眼痛伴头痛恶心（且非否定表述），标记为 URGENT；否则 NORMAL；\n"
                "3. audience：识别为 patient（受检者/大众体检）、caregiver（陪诊家属/替父母子女咨询）或 clinician（基层医生）；\n"
                "4. 若 urgent_flag=URGENT，在输出第一行醒目提醒立即前往眼科急诊。"
            ),
            user_prompt=(
                "用户输入：{{#start.query#}}\n"
                "后端上下文返回状态码：{{#http_context.status_code#}}\n"
                "后端上下文内容：{{#http_context.body#}}"
            ),
        ),
        _node(
            "end",
            "end",
            "结束",
            960,
            240,
            {
                "outputs": [
                    {"variable": "triage_summary", "value_selector": ["llm_triage_entry", "text"]},
                    {"variable": "context_json", "value_selector": ["http_context", "body"]},
                ]
            },
        ),
    ]
    edges = [
        _edge("start", "start", "http_context", "http-request"),
        _edge("http_context", "http-request", "llm_triage_entry", "llm"),
        _edge("llm_triage_entry", "llm", "end", "end"),
    ]
    doc["workflow"] = {
        "conversation_variables": [],
        "environment_variables": [],
        "features": _features(),
        "graph": {"nodes": nodes, "edges": edges},
    }
    return doc


def build_w10_screen(origin: str, api_key: str) -> dict:
    """W10: Upload new eye image -> atomic screen_fundus -> QC branch -> KB retrieval -> A2 report summary."""
    doc = _base_app(
        "W10_眼底筛查与首屏解读",
        "接收眼底照片 HTTPS 地址及随图眼别症状，原子调用六模型与病灶定位，经质控分支后结合医学知识库输出首屏解读与 PDF 链接。",
        "W1",
    )
    body_tpl = json.dumps(
        {
            "image_url": "{{#start.image_url#}}",
            "eye": "{{#start.eye#}}",
            "symptoms": "{{#start.symptoms#}}",
            "history": "{{#start.history#}}",
        },
        ensure_ascii=False,
    )
    nodes = [
        _node(
            "start",
            "start",
            "开始",
            80,
            260,
            {
                "variables": [
                    _var("image_url", "眼底图像 HTTPS 地址", required=True, max_length=4096),
                    _var("eye", "眼别（left / right / unknown）", required=False, max_length=32),
                    _var("symptoms", "随图眼部症状（逗号分隔）", required=False, max_length=512),
                    _var("history", "随图慢病与治疗史（逗号分隔）", required=False, max_length=512),
                    _var("audience", "受众视角（patient / caregiver / clinician）", required=False, max_length=32),
                    _var("user_query", "用户附带提问", required=False, max_length=1000, var_type="paragraph"),
                ]
            },
        ),
        _http_node(
            "http_screen",
            "六模型与病灶定位推理",
            360,
            260,
            method="post",
            url=f"{origin}/v1/screen_fundus",
            api_key=api_key,
            json_body_template=body_tpl,
            read_timeout=120,
        ),
        _node(
            "if_qc_ok",
            "if-else",
            "校验请求与图像质控",
            660,
            260,
            {
                "cases": [
                    {
                        "case_id": "true",
                        "id": "true",
                        "logical_operator": "and",
                        "conditions": [
                            {
                                "id": "cond_status_200",
                                "variable_selector": ["http_screen", "status_code"],
                                "comparison_operator": "is",
                                "value": "200",
                            },
                            {
                                "id": "cond_not_rejected",
                                "variable_selector": ["http_screen", "body"],
                                "comparison_operator": "not contains",
                                "value": "\"quality_rejected\"",
                            },
                        ],
                    }
                ]
            },
        ),
        _kb_node("kb_medical", "检索医学知识库", 960, 140, ["start", "user_query"]),
        _llm_node(
            "llm_explain",
            "A2 医学解读与首屏输出",
            1260,
            140,
            context_selector=["kb_medical", "result"],
            system_prompt=(
                "你是眼底医学解读智能体（A2）。绝对禁止使用任何 Emoji 表情符号。\n"
                "必须严格依据筛查接口返回的 JSON 数据与检索到的医学知识库内容作答：\n"
                "1. 先明确图像质控状态（Good/Usable）及六模型运行情况，严禁修改分数、阈值或糖网分期数字；\n"
                "2. 严格区分疾病（DR/AMD/RVO/CSR/glaucoma）、影像体征（DN/ODC/ODE/ODP/MYOPIC_FUNDUS/TSLN）与治疗痕迹（LS 激光瘢痕，绝不写成新发眼病）；\n"
                "3. 根据受众模式（patient 通俗去焦虑 / caregiver 陪诊与红线 / clinician 全量指标明细）组织回复，并在专业概念后标注来源编号（如 [SRC_DR_GUIDELINE_2024]、[SRC_DICT_LS]）；\n"
                "4. 使用 Markdown 三级标题与对齐表格展示核心发现；保留展示 visual_report.original_url 与 overlay_url；\n"
                "5. 若返回存在 visual_report.report_url，必须在回复末尾以引用块给出：[打开完整交互报告（支持导出标准 A4 PDF）](report_url)。"
            ),
            user_prompt=(
                "受众模式：{{#start.audience#}}\n"
                "眼别：{{#start.eye#}}；症状：{{#start.symptoms#}}；病史：{{#start.history#}}\n"
                "用户问题：{{#start.user_query#}}\n"
                "筛查接口返回JSON：{{#http_screen.body#}}\n"
                "医学知识库检索片段：{{#kb_medical.result#}}"
            ),
        ),
        _llm_node(
            "llm_retake",
            "质控拦截与重拍引导",
            960,
            390,
            system_prompt=(
                "你是眼底筛查质控引导助手。禁止使用任何 Emoji。\n"
                "当图像质控为 Reject、图像不可读或请求报错时：\n"
                "1. 清晰告知无法进行自动分期或筛查的具体原因（参考返回 JSON 中的 message 与 reason）；\n"
                "2. 绝不能把未运行的模型项写成“正常”或“阴性”；\n"
                "3. 给出重拍清晰眼底照片的具体动作建议（暗室对焦、避免反光遮挡），若用户已自述紧急视力症状，同时提醒立即眼科面诊 [SRC_QC_RETAKE]。"
            ),
            user_prompt=(
                "HTTP 状态码：{{#http_screen.status_code#}}\n"
                "返回内容：{{#http_screen.body#}}"
            ),
        ),
        _node(
            "end_ok",
            "end",
            "输出解读结果",
            1560,
            140,
            {
                "outputs": [
                    {"variable": "answer", "value_selector": ["llm_explain", "text"]},
                    {"variable": "screening_json", "value_selector": ["http_screen", "body"]},
                ]
            },
        ),
        _node(
            "end_reject",
            "end",
            "输出质控指引",
            1260,
            390,
            {
                "outputs": [
                    {"variable": "answer", "value_selector": ["llm_retake", "text"]},
                    {"variable": "screening_json", "value_selector": ["http_screen", "body"]},
                ]
            },
        ),
    ]
    edges = [
        _edge("start", "start", "http_screen", "http-request"),
        _edge("http_screen", "http-request", "if_qc_ok", "if-else"),
        _edge("if_qc_ok", "if-else", "kb_medical", "knowledge-retrieval", source_handle="true"),
        _edge("kb_medical", "knowledge-retrieval", "llm_explain", "llm"),
        _edge("llm_explain", "llm", "end_ok", "end"),
        _edge("if_qc_ok", "if-else", "llm_retake", "llm", source_handle="false"),
        _edge("llm_retake", "llm", "end_reject", "end"),
    ]
    doc["workflow"] = {
        "conversation_variables": [],
        "environment_variables": [],
        "features": _features(),
        "graph": {"nodes": nodes, "edges": edges},
    }
    return doc


def build_w20_qa_and_edu(origin: str, api_key: str) -> dict:
    """W20: Follow-up Q&A on frozen evidence packet OR no-image paper checkup consultation."""
    doc = _base_app(
        "W20_报告追问与无图科普",
        "支持有图场景读取冻结 EvidencePacket 进行精确追问，也支持无图场景下纸质体检名词（TSLN/DN/ODC/LS）去焦虑科普与检查引导。",
        "W2",
    )
    nodes = [
        _node(
            "start",
            "start",
            "开始",
            80,
            260,
            {
                "variables": [
                    _var("user_query", "用户提问内容", required=True, max_length=2000, var_type="paragraph"),
                    _var("assessment_id", "当前检查 assessment_id（无图留空）", required=False, max_length=128),
                    _var("audience", "受众视角（patient / caregiver / clinician）", required=False, max_length=32),
                ]
            },
        ),
        _node(
            "if_has_id",
            "if-else",
            "判断是否存在已建档检查ID",
            360,
            260,
            {
                "cases": [
                    {
                        "case_id": "true",
                        "id": "true",
                        "logical_operator": "and",
                        "conditions": [
                            {
                                "id": "cond_asmt_not_empty",
                                "variable_selector": ["start", "assessment_id"],
                                "comparison_operator": "not empty",
                                "value": "",
                            }
                        ],
                    }
                ]
            },
        ),
        _http_node(
            "http_assessment",
            "读取冻结检查证据",
            660,
            140,
            method="get",
            url=f"{origin}/v1/assessments/{{{{#start.assessment_id#}}}}",
            api_key=api_key,
            read_timeout=15,
        ),
        _kb_node("kb_with_img", "检索证据依据与标签字典", 960, 140, ["start", "user_query"]),
        _llm_node(
            "llm_answer_img",
            "A2 证据溯源问答",
            1260,
            140,
            context_selector=["kb_with_img", "result"],
            system_prompt=(
                "你是眼底医学解读问答智能体（A2）。绝对禁止使用任何 Emoji。\n"
                "先直接回答用户问题，再说明本次检查证据（EvidencePacket），最后给出下一步建议：\n"
                "1. 严格以冻结 EvidencePacket 中的眼别、质控、糖网分级与 13 项标签得分为准，绝不窜用其他眼别或改写数值；\n"
                "2. 不把 0~100 标度的得分解释为患病概率百分比；\n"
                "3. 关键解释须关联真实来源编号（如 [SRC_DICT_DN_AMD]、[SRC_DICT_LS]、[SRC_DR_GUIDELINE_2024]）；\n"
                "4. 按 audience（patient/caregiver/clinician）调整话术详略，并在结尾保留报告下载入口（若 evidence 中有 report_url）。"
            ),
            user_prompt=(
                "受众模式：{{#start.audience#}}\n"
                "用户问题：{{#start.user_query#}}\n"
                "冻结证据接口返回：{{#http_assessment.body#}}\n"
                "知识库检索片段：{{#kb_with_img.result#}}"
            ),
        ),
        _kb_node("kb_no_img", "检索大众体检与慢病科普", 660, 390, ["start", "user_query"]),
        _llm_node(
            "llm_answer_no_img",
            "A4 无图体检名词科普与分诊",
            960,
            390,
            context_selector=["kb_no_img", "result"],
            system_prompt=(
                "你是眼底医学科普与无图导诊助手（A2+A4）。绝对禁止使用任何 Emoji。\n"
                "当前尚未上传或未绑定具体眼底照片，你必须严守无图咨询边界：\n"
                "1. 先排除急症：如用户提及突发大量飞蚊、闪光感、幕帘样遮挡、突发视力骤降或剧烈眼痛，立刻置顶提示当日前往眼科急诊 [SRC_NEI_RD_URGENT]；\n"
                "2. 对用户询问的体检报告词汇（如 TSLN 豹纹状眼底、DN 玻璃膜疣、OPTIC_DISC_CUPPING 视盘凹陷、LS 激光瘢痕）结合知识库做通俗去焦虑科普并标注 [SRC_*]；\n"
                "3. 严禁在未见眼底图片的情况下对用户个人作出“眼底正常”或“排除病变”的判断，引导需要明确筛查时上传清晰眼底照并告知左右眼别 [SRC_EDU_NO_IMAGE_TRIAGE]。"
            ),
            user_prompt=(
                "受众模式：{{#start.audience#}}\n"
                "用户问题：{{#start.user_query#}}\n"
                "知识库检索片段：{{#kb_no_img.result#}}"
            ),
        ),
        _node(
            "end_with_evidence",
            "end",
            "结束（有图追问）",
            1560,
            140,
            {"outputs": [{"variable": "answer", "value_selector": ["llm_answer_img", "text"]}]},
        ),
        _node(
            "end_no_image",
            "end",
            "结束（无图科普）",
            1260,
            390,
            {"outputs": [{"variable": "answer", "value_selector": ["llm_answer_no_img", "text"]}]},
        ),
    ]
    edges = [
        _edge("start", "start", "if_has_id", "if-else"),
        _edge("if_has_id", "if-else", "http_assessment", "http-request", source_handle="true"),
        _edge("http_assessment", "http-request", "kb_with_img", "knowledge-retrieval"),
        _edge("kb_with_img", "knowledge-retrieval", "llm_answer_img", "llm"),
        _edge("llm_answer_img", "llm", "end_with_evidence", "end"),
        _edge("if_has_id", "if-else", "kb_no_img", "knowledge-retrieval", source_handle="false"),
        _edge("kb_no_img", "knowledge-retrieval", "llm_answer_no_img", "llm"),
        _edge("llm_answer_no_img", "llm", "end_no_image", "end"),
    ]
    doc["workflow"] = {
        "conversation_variables": [],
        "environment_variables": [],
        "features": _features(),
        "graph": {"nodes": nodes, "edges": edges},
    }
    return doc


def build_w30_care_plan(origin: str, api_key: str) -> dict:
    """W30: Record user intake facts -> run deterministic symptom-first care rules -> output plan."""
    doc = _base_app(
        "W30_症状补录与随访计划",
        "固化记录眼别/症状/病史（record_intake）后立即调用确定性规则引擎（evaluate_care），严格贯彻急症优先红线并输出随访清单。",
        "W3",
    )
    intake_tpl = json.dumps(
        {
            "assessment_id": "{{#start.assessment_id#}}",
            "session_id": "{{#start.session_id#}}",
            "eye": "{{#start.eye#}}",
            "symptoms": "{{#start.symptoms#}}",
            "symptom_onset": "{{#start.symptom_onset#}}",
            "history": "{{#start.history#}}",
            "confirmed_plan": "{{#start.confirmed_plan#}}",
        },
        ensure_ascii=False,
    )
    care_tpl = json.dumps(
        {
            "assessment_id": "{{#start.assessment_id#}}",
            "session_id": "{{#start.session_id#}}",
        },
        ensure_ascii=False,
    )
    nodes = [
        _node(
            "start",
            "start",
            "开始",
            80,
            240,
            {
                "variables": [
                    _var("assessment_id", "目标检查 assessment_id", required=True, max_length=128),
                    _var("session_id", "会话ID（默认 default）", required=False, max_length=128),
                    _var("eye", "眼别（left / right / unknown，未提留空）", required=False, max_length=32),
                    _var("symptoms", "症状补充（逗号分隔）", required=False, max_length=512),
                    _var("symptom_onset", "起病时间", required=False, max_length=128),
                    _var("history", "慢病与既往激光/手术史（逗号分隔）", required=False, max_length=512),
                    _var("confirmed_plan", "线下医生既有医嘱安排", required=False, max_length=512),
                    _var("audience", "受众视角（patient / caregiver / clinician）", required=False, max_length=32),
                ]
            },
        ),
        _http_node(
            "http_intake",
            "保存结构化症状与眼别",
            360,
            240,
            method="post",
            url=f"{origin}/v1/intake",
            api_key=api_key,
            json_body_template=intake_tpl,
            read_timeout=15,
        ),
        _http_node(
            "http_care",
            "执行确定性随访规则引擎",
            660,
            240,
            method="post",
            url=f"{origin}/v1/care/evaluate",
            api_key=api_key,
            json_body_template=care_tpl,
            read_timeout=15,
        ),
        _kb_node("kb_care", "检索急症与专科随访依据", 960, 240, ["start", "symptoms"]),
        _llm_node(
            "llm_care_plan",
            "A3 生成可执行随访计划",
            1260,
            240,
            context_selector=["kb_care", "result"],
            system_prompt=(
                "你是眼底随访管理智能体（A3）。绝对禁止使用任何 Emoji 表情符号。\n"
                "严格依据 evaluate_care 工具返回的结构化计划与规则编号生成回复：\n"
                "1. 安全红线：若返回的 care_priority 为 urgent_ophthalmology，必须在第一行加粗置顶 **立即眼科急诊评估** 并解释命中症状（引用 [SRC_NEI_RD_URGENT]），说明影像结果不能降低急症优先级；\n"
                "2. 按工具返回的 priority_label、followup_time、actions、monitoring_items、questions_for_clinician 组织清晰的表格或要点清单；若面向 caregiver（陪诊家属），附加诊前资料准备提示 [SRC_EDU_CAREGIVER_GUIDE]；\n"
                "3. 若 missing_facts/questions_to_confirm 非空，仅温和追问这 1~2 个关键缺失项；\n"
                "4. 严禁修改工具给定的优先级、复查时间或规则依据；末尾若存在 report_url，务必附上实时同步后的报告与 PDF 导出链接。"
            ),
            user_prompt=(
                "受众模式：{{#start.audience#}}\n"
                "入库回执：{{#http_intake.body#}}\n"
                "规则引擎随访计划：{{#http_care.body#}}\n"
                "知识库参考：{{#kb_care.result#}}"
            ),
        ),
        _node(
            "end",
            "end",
            "结束",
            1560,
            240,
            {
                "outputs": [
                    {"variable": "answer", "value_selector": ["llm_care_plan", "text"]},
                    {"variable": "care_plan_json", "value_selector": ["http_care", "body"]},
                ]
            },
        ),
    ]
    edges = [
        _edge("start", "start", "http_intake", "http-request"),
        _edge("http_intake", "http-request", "http_care", "http-request"),
        _edge("http_care", "http-request", "kb_care", "knowledge-retrieval"),
        _edge("kb_care", "knowledge-retrieval", "llm_care_plan", "llm"),
        _edge("llm_care_plan", "llm", "end", "end"),
    ]
    doc["workflow"] = {
        "conversation_variables": [],
        "environment_variables": [],
        "features": _features(),
        "graph": {"nodes": nodes, "edges": edges},
    }
    return doc


def build_w40_compare(origin: str, api_key: str) -> dict:
    """W40: Compare two same-eye assessments or list recent left/right eye records."""
    doc = _base_app(
        "W40_同眼历史与双眼对照",
        "在明确两次检查 ID 时调用 compare_assessments 执行同眼与版本可比性核验；缺 ID 或查询双眼对照时读取 get_context 汇总展示。",
        "W4",
    )
    cmp_tpl = json.dumps(
        {
            "previous_assessment_id": "{{#start.previous_assessment_id#}}",
            "current_assessment_id": "{{#start.current_assessment_id#}}",
        },
        ensure_ascii=False,
    )
    nodes = [
        _node(
            "start",
            "start",
            "开始",
            80,
            260,
            {
                "variables": [
                    _var("user_query", "用户的对比或双眼查看诉求", required=True, max_length=1000, var_type="paragraph"),
                    _var("previous_assessment_id", "较早检查 assessment_id（可选）", required=False, max_length=128),
                    _var("current_assessment_id", "较新检查 assessment_id（可选）", required=False, max_length=128),
                    _var("session_id", "会话ID（默认 default）", required=False, max_length=128),
                ]
            },
        ),
        _node(
            "if_two_ids",
            "if-else",
            "判断是否已指定两条对比ID",
            360,
            260,
            {
                "cases": [
                    {
                        "case_id": "true",
                        "id": "true",
                        "logical_operator": "and",
                        "conditions": [
                            {
                                "id": "cond_prev",
                                "variable_selector": ["start", "previous_assessment_id"],
                                "comparison_operator": "not empty",
                                "value": "",
                            },
                            {
                                "id": "cond_curr",
                                "variable_selector": ["start", "current_assessment_id"],
                                "comparison_operator": "not empty",
                                "value": "",
                            },
                        ],
                    }
                ]
            },
        ),
        _http_node(
            "http_compare",
            "调用同眼对比接口",
            660,
            140,
            method="post",
            url=f"{origin}/v1/comparisons",
            api_key=api_key,
            json_body_template=cmp_tpl,
            read_timeout=15,
        ),
        _llm_node(
            "llm_compare",
            "A4 输出可比性与差异解读",
            960,
            140,
            system_prompt=(
                "你是眼底历史对比与科普教育智能体（A4）。绝对禁止使用任何 Emoji。\n"
                "严格依据 compare_assessments 返回结果说明：\n"
                "1. 若 comparable=false，明确解释不可比原因（如眼别为 unknown 需先确认，两个 unknown 不代表同一只眼；或左右眼不一致；或质控不合格）；\n"
                "2. 若 comparable=true，展示版本一致性、糖网分期变化及标签达线变化，并用 Markdown 表格汇总关键变化；\n"
                "3. **医学红线**：明确重申模型分类分数不具备严重度量纲，得分波动仅代表同方法下两张图输出响应差异，绝不能直接写成临床病情恶化或好转。"
            ),
            user_prompt=(
                "用户问题：{{#start.user_query#}}\n"
                "对比接口返回JSON：{{#http_compare.body#}}"
            ),
        ),
        _http_node(
            "http_ctx",
            "拉取近期检查与双眼列表",
            660,
            390,
            method="get",
            url=f"{origin}/v1/context?session_id={{{{#start.session_id#}}}}",
            api_key=api_key,
            read_timeout=15,
        ),
        _llm_node(
            "llm_list_eyes",
            "整理左/右眼汇总并引导选择",
            960,
            390,
            system_prompt=(
                "你是眼底历史对比助手（A4）。绝对禁止使用任何 Emoji。\n"
                "根据 get_context 返回的 recent_assessments 列表，向用户清晰展示已保存检查的 assessment_id、时间、眼别（left/right/unknown）：\n"
                "1. 若分别存在左眼与右眼记录，整理成双眼并列摘要表格；\n"
                "2. 若用户希望对比同一只眼变化但记录仍标为 unknown 或仅有一条记录，清晰指引补充确认左右眼别或补充对应眼别检查 ID 后再对比。"
            ),
            user_prompt=(
                "用户问题：{{#start.user_query#}}\n"
                "上下文记录列表：{{#http_ctx.body#}}"
            ),
        ),
        _node(
            "end_cmp",
            "end",
            "结束（对比报告）",
            1260,
            140,
            {
                "outputs": [
                    {"variable": "answer", "value_selector": ["llm_compare", "text"]},
                    {"variable": "comparison_json", "value_selector": ["http_compare", "body"]},
                ]
            },
        ),
        _node(
            "end_ctx",
            "end",
            "结束（检查汇总）",
            1260,
            390,
            {"outputs": [{"variable": "answer", "value_selector": ["llm_list_eyes", "text"]}]},
        ),
    ]
    edges = [
        _edge("start", "start", "if_two_ids", "if-else"),
        _edge("if_two_ids", "if-else", "http_compare", "http-request", source_handle="true"),
        _edge("http_compare", "http-request", "llm_compare", "llm"),
        _edge("llm_compare", "llm", "end_cmp", "end"),
        _edge("if_two_ids", "if-else", "http_ctx", "http-request", source_handle="false"),
        _edge("http_ctx", "http-request", "llm_list_eyes", "llm"),
        _edge("llm_list_eyes", "llm", "end_ctx", "end"),
    ]
    doc["workflow"] = {
        "conversation_variables": [],
        "environment_variables": [],
        "features": _features(),
        "graph": {"nodes": nodes, "edges": edges},
    }
    return doc


def build_w90_fallback(origin: str, _api_key: str) -> dict:
    """W90: Default fallback workflow for out-of-scope queries or invalid media types."""
    doc = _base_app(
        "W90_边界与异常兜底",
        "多智能体默认工作流：处理非眼科闲聊、普通自拍/眼表/OCT 图像范围说明，并提供正确使用指引与集成工作台入口。",
        "W9",
    )
    nodes = [
        _node(
            "start",
            "start",
            "开始",
            80,
            240,
            {"variables": [_var("user_query", "用户输入", required=True, max_length=2000, var_type="paragraph")]},
        ),
        _llm_node(
            "llm_fallback",
            "边界说明与操作引导",
            380,
            240,
            system_prompt=(
                "你是眼底筛查与随访多智能体系统的默认兜底助手（W90）。绝对禁止使用任何 Emoji。\n"
                "当用户提问超出眼底筛查范围、上传了普通自拍/眼表外照/OCT 图像，或遇到任务异常时：\n"
                "1. 礼貌说明本系统适用范围：基于彩色眼底相机照片的六模型辅助筛查、四类病灶定位、症状优先随访与大众科普；\n"
                "2. 如用户描述了突发飞蚊、闪光感、幕帘遮挡、突发视力骤降或剧烈眼痛，无论场景必须优先提醒立即线下眼科急诊；\n"
                "3. 引导用户可上传清晰的眼底彩色照（附带左/右眼说明），或前往在线可视化演示工作台体验："
                f" {origin}/v1/demo 。"
            ),
            user_prompt="用户输入：{{#start.user_query#}}",
        ),
        _node(
            "end",
            "end",
            "结束",
            680,
            240,
            {"outputs": [{"variable": "answer", "value_selector": ["llm_fallback", "text"]}]},
        ),
    ]
    edges = [
        _edge("start", "start", "llm_fallback", "llm"),
        _edge("llm_fallback", "llm", "end", "end"),
    ]
    doc["workflow"] = {
        "conversation_variables": [],
        "environment_variables": [],
        "features": _features(),
        "graph": {"nodes": nodes, "edges": edges},
    }
    return doc


WORKFLOW_BUILDERS = {
    "W00_上下文与急症前置检查.yml": build_w00_context,
    "W10_眼底筛查与首屏解读.yml": build_w10_screen,
    "W20_报告追问与无图科普.yml": build_w20_qa_and_edu,
    "W30_症状补录与随访计划.yml": build_w30_care_plan,
    "W40_同眼历史与双眼对照.yml": build_w40_compare,
    "W90_边界与异常兜底.yml": build_w90_fallback,
}


def export_dify_workflows(output_dir: Path, *, origin: str = DEFAULT_ORIGIN, api_key: str = PLACEHOLDER_API_KEY) -> list[Path]:
    output = output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "dify_workflows"
        stage.mkdir()
        for filename, builder in WORKFLOW_BUILDERS.items():
            payload = builder(origin, api_key)
            text = yaml.dump(payload, Dumper=_NoAliasDumper, sort_keys=False, allow_unicode=True)
            target = stage / filename
            target.write_text(text, encoding="utf-8")
        if output.exists():
            import shutil
            shutil.rmtree(output)
        stage.rename(output)
    for filename in WORKFLOW_BUILDERS:
        written.append(output / filename)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default=DEFAULT_ORIGIN, type=origin_url)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "dify_workflows")
    parser.add_argument("--api-key", default=None, help="Explicit X-API-Key to embed")
    parser.add_argument("--embed-local-key", action="store_true", help="Read ~/.retina-screening/api-key.txt and embed X-API-Key")
    args = parser.parse_args()

    api_key = args.api_key or PLACEHOLDER_API_KEY
    if args.embed_local_key and not args.api_key:
        key_path = Path.home() / ".retina-screening" / "api-key.txt"
        if not key_path.is_file():
            parser.error(f"Local API key file not found: {key_path}")
        api_key = key_path.read_text(encoding="ascii").strip()

    paths = export_dify_workflows(args.output, origin=args.origin, api_key=api_key)
    for p in paths:
        print(f"Created {p}")


if __name__ == "__main__":
    main()
