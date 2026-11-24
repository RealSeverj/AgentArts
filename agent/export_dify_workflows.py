"""Export Dify 0.6.0 Workflow DSL (.yml) packages for AgentArts dual-app architecture (Patient + Doctor)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
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
        "dependencies": [],
        "kind": "app",
        "version": "0.6.0",
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
        "suggested_questions": [],
        "suggested_questions_after_answer": {"enabled": False},
        "text_to_speech": {"enabled": False},
    }


def _workflow_block(nodes: list[dict], edges: list[dict]) -> dict:
    return {
        "conversation_variables": [],
        "environment_variables": [],
        "features": _features(),
        "graph": {
            "edges": edges,
            "nodes": nodes,
            "viewport": {"x": 0, "y": 0, "zoom": 1},
        },
    }


def _node(node_id: str, node_type: str, title: str, x: int, y: int, data: dict, desc: str = "") -> dict:
    payload = {"type": node_type, "title": title, "desc": desc, "selected": False, **data}
    return {
        "id": node_id,
        "type": "custom",
        "width": 244,
        "height": 98,
        "selected": False,
        "position": {"x": x, "y": y},
        "positionAbsolute": {"x": x, "y": y},
        "data": payload,
    }


def _start_query_only(x: int = 80, y: int = 240) -> dict:
    """Single mandatory 'query' parameter avoids AgentArts multi-slot extraction errors."""
    return _node(
        "start",
        "start",
        "开始",
        x,
        y,
        {
            "variables": [
                {
                    "variable": "query",
                    "label": "用户输入",
                    "type": "paragraph",
                    "required": True,
                    "max_length": 4000,
                    "options": [],
                }
            ]
        },
    )


def _edge(source: str, source_type: str, target: str, target_type: str, source_handle: str = "source") -> dict:
    return {
        "id": f"{source}__{source_handle}__to__{target}",
        "source": source,
        "sourceHandle": source_handle,
        "target": target,
        "targetHandle": "target",
        "type": "custom",
        "zIndex": 0,
        "data": {
            "isInIteration": False,
            "sourceType": source_type,
            "targetType": target_type,
        },
    }


def _http_post_query(node_id: str, title: str, x: int, y: int, *, url: str, api_key: str, read_timeout: int = 60) -> dict:
    return _node(
        node_id,
        "http-request",
        title,
        x,
        y,
        {
            "method": "post",
            "url": url,
            "authorization": {"type": "no-auth", "config": None},
            "headers": f"Content-Type: application/json\nX-API-Key: {api_key}",
            "params": "",
            "body": {
                "type": "json",
                "data": [
                    {
                        "id": f"{node_id}_json_body",
                        "key": "",
                        "type": "text",
                        "value": json.dumps({"query": "{{#start.query#}}"}, ensure_ascii=False),
                    }
                ],
            },
            "timeout": {
                "max_connect_timeout": 10,
                "max_read_timeout": read_timeout,
                "max_write_timeout": 30,
            },
            "retry_config": {
                "max_retries": 1,
                "retry_enabled": False,
                "retry_interval": 100,
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
                "completion_params": {"temperature": 0.25},
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


def _kb_node(node_id: str, title: str, x: int, y: int) -> dict:
    return _node(
        node_id,
        "knowledge-retrieval",
        title,
        x,
        y,
        {
            "query_variable_selector": ["start", "query"],
            "dataset_ids": [DEFAULT_KB_ID],
            "retrieval_mode": "multiple",
            "multiple_retrieval_config": {
                "top_k": 5,
                "score_threshold": 0.35,
                "reranking_enable": False,
            },
        },
    )


def _simple_http_llm_workflow(
    *,
    name: str,
    desc: str,
    icon: str,
    http_title: str,
    url: str,
    api_key: str,
    read_timeout: int,
    llm_title: str,
    system_prompt: str,
    user_prompt: str,
    with_kb: bool = False,
) -> dict:
    doc = _base_app(name, desc, icon)
    nodes = [
        _start_query_only(80, 240),
        _http_post_query("http_api", http_title, 360, 240, url=url, api_key=api_key, read_timeout=read_timeout),
    ]
    edges = [_edge("start", "start", "http_api", "http-request")]
    if with_kb:
        nodes.append(_kb_node("kb_search", "检索医学与科普知识库", 660, 240))
        nodes.append(
            _llm_node(
                "llm_reply",
                llm_title,
                960,
                240,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                context_selector=["kb_search", "result"],
            )
        )
        nodes.append(
            _node("end", "end", "结束", 1260, 240, {"outputs": [{"variable": "answer", "value_selector": ["llm_reply", "text"]}]})
        )
        edges.extend([
            _edge("http_api", "http-request", "kb_search", "knowledge-retrieval"),
            _edge("kb_search", "knowledge-retrieval", "llm_reply", "llm"),
            _edge("llm_reply", "llm", "end", "end"),
        ])
    else:
        nodes.append(_llm_node("llm_reply", llm_title, 660, 240, system_prompt=system_prompt, user_prompt=user_prompt))
        nodes.append(
            _node("end", "end", "结束", 960, 240, {"outputs": [{"variable": "answer", "value_selector": ["llm_reply", "text"]}]})
        )
        edges.extend([
            _edge("http_api", "http-request", "llm_reply", "llm"),
            _edge("llm_reply", "llm", "end", "end"),
        ])
    doc["workflow"] = _workflow_block(nodes, edges)
    return doc


# ==================== App 1: Patient Personal Eye Care Agent ====================

def build_p00_patient_session(origin: str, api_key: str) -> dict:
    return _simple_http_llm_workflow(
        name="P00_患者身份与档案绑定",
        desc="患者端起始工作流：通过手机号识别患者并同步医院拍片室档案状态，给予亲切欢迎与引导。",
        icon="P0",
        http_title="同步患者院内档案",
        url=f"{origin}/v1/his/agent/patient/session",
        api_key=api_key,
        read_timeout=15,
        llm_title="患者端亲切接待",
        system_prompt=(
            "你是「视健康·患者端个人眼健康管家」。严禁使用任何 Emoji。\n"
            "根据后端返回的档案接口结果，用温暖、亲切、像家人朋友一样自然的中文和患者打招呼：\n"
            "- 如果 bound=false，请温和礼貌地请患者告诉您他的手机号以便调出医院档案；\n"
            "- 如果 bound=true，亲切称呼患者姓名，简单告知已关联好他的既往病史和医院刚刚同步的新照片状态（若有），并热情询问今天想了解检查报告、平时护眼问题还是突然的眼部不适。"
        ),
        user_prompt="患者本轮说：{{#start.query#}}\n院内档案接口返回：{{#http_api.body#}}",
    )


def build_p10_patient_report(origin: str, api_key: str) -> dict:
    return _simple_http_llm_workflow(
        name="P10_患者报告温暖解读",
        desc="患者免传图查报告：自动检查医院拍片室是否同步了新片、执行六模型分析与跨期对比，并用通俗温暖的话语解读报告并附报告链接。",
        icon="P1",
        http_title="院内影像自动分析与对比",
        url=f"{origin}/v1/his/agent/patient/report",
        api_key=api_key,
        read_timeout=120,
        llm_title="温暖版报告翻译润色",
        system_prompt=(
            "你是「视健康·患者端个人眼健康管家」。严禁使用任何 Emoji。\n"
            "后端接口已经替您完成了院内影像排查、六模型推理、与上次同眼报告的进步对比，并生成了患者专属报告链接。\n"
            "请绝对以接口返回JSON中的事实、patient_report_url 和 reply 建议为基础，用通俗、亲切、让人放心的口语整理回复：\n"
            "1. 绝对不要生硬堆砌“图像质控 Good”“0~100 标度”“微动脉瘤候选区”“[SRC_*]”等科研术语；\n"
            "2. 先说明是查了医院新同步的照片还是调出了上一次存档报告；\n"
            "3. 若未见明显病变，用类似“很高兴告诉您：模型检查后没有发现明显的视网膜病变，您可以放宽心”表达；若有进步对比，说明“整体往好的方向走（具体恢复情况门诊让医生最终拍板）”；若有需要留意的问题，温和建议复查时间；\n"
            "4. 务必保留并展示接口返回的 patient_report_url 链接，告诉患者点开即可查看及保存 PDF 报告。"
        ),
        user_prompt="患者本轮说：{{#start.query#}}\n院内报告接口返回：{{#http_api.body#}}",
    )


def build_p20_patient_symptom(origin: str, api_key: str) -> dict:
    return _simple_http_llm_workflow(
        name="P20_结合病史症状问诊",
        desc="患者突发不适或症状咨询：自动调取该患者慢病史与最近眼底记录，结合急症红线与知识库给出贴合本人病情的分诊建议。",
        icon="P2",
        http_title="拉取患者病史与急症判定",
        url=f"{origin}/v1/his/agent/patient/symptom",
        api_key=api_key,
        read_timeout=15,
        llm_title="个体化症状解答",
        with_kb=True,
        system_prompt=(
            "你是「视健康·患者端个人眼健康管家」。严禁使用任何 Emoji。\n"
            "结合后端返回的该患者个人慢病史、既往眼底情况、急症检测标志（urgent_detected）以及知识库检索内容，贴心地回答患者不适：\n"
            "1. 先复述并关切患者的症状感受，把医学可能性用大白话讲明白；\n"
            "2. 务必主动把【他本人的既往慢病史和眼底底子】联系起来分析（例如原本就有几年糖尿病、之前左眼查过糖网等）；\n"
            "3. 如果 urgent_detected=true，务必第一时间清晰温和地劝导患者别耽误、今天停下剧烈活动尽快让家人陪着去眼科急诊或做散瞳检查；\n"
            "4. 若暂未命中急症，安抚情绪并告知哪些情况出现时需及时看医生。"
        ),
        user_prompt=(
            "患者描述症状：{{#start.query#}}\n"
            "个人画像与急症判定：{{#http_api.body#}}\n"
            "知识库检索参考：{{#kb_search.result#}}"
        ),
    )


def build_p30_patient_education(_origin: str, _api_key: str) -> dict:
    doc = _base_app(
        "P30_眼健康日常科普答疑",
        "患者日常闲聊或主动询问青光眼、糖网、飞蚊症、豹纹状眼底等眼病科普知识与护眼视频/文章推荐。",
        "P3",
    )
    nodes = [
        _start_query_only(80, 240),
        _kb_node("kb_edu", "检索患者通俗科普库", 360, 240),
        _llm_node(
            "llm_edu",
            "亲切科普讲解与资源推荐",
            680,
            240,
            context_selector=["kb_edu", "result"],
            system_prompt=(
                "你是「视健康·患者端个人眼健康管家」。严禁使用任何 Emoji。\n"
                "患者在闲暇或体检后向你咨询常见眼病知识（如青光眼是怎么发生的、平时注意什么、玻璃膜疣/豹纹眼底是什么意思等）：\n"
                "1. 用生活化的比喻把病因讲清楚（例如把青光眼比作水球水压排不畅），绝不吓唬患者；\n"
                "2. 列出 3~5 条在家就能照着做到的日常护眼小贴士；\n"
                "3. 附带推荐权威科普文章主题与视频检索建议（后续若在画布串联市场视频 MCP 插件，可直接附真实视频链接）。"
            ),
            user_prompt="患者提问：{{#start.query#}}\n科普知识库片段：{{#kb_edu.result#}}",
        ),
        _node("end", "end", "结束", 980, 240, {"outputs": [{"variable": "answer", "value_selector": ["llm_edu", "text"]}]}),
    ]
    edges = [
        _edge("start", "start", "kb_edu", "knowledge-retrieval"),
        _edge("kb_edu", "knowledge-retrieval", "llm_edu", "llm"),
        _edge("llm_edu", "llm", "end", "end"),
    ]
    doc["workflow"] = _workflow_block(nodes, edges)
    return doc


# ==================== App 2: Doctor Clinical Assistant Agent ====================

def build_d00_doctor_patient(origin: str, api_key: str) -> dict:
    return _simple_http_llm_workflow(
        name="D00_会诊患者接诊与影像概览",
        desc="医生端起始/接诊工作流：按手机号检索并绑定当前会诊患者，展示慢病史及院内拍片室同步的左右眼影像状态。",
        icon="D0",
        http_title="调阅会诊患者档案",
        url=f"{origin}/v1/his/agent/doctor/patient",
        api_key=api_key,
        read_timeout=15,
        llm_title="输出接诊摘要",
        system_prompt=(
            "你是「视健康·眼科医生端临床辅助智能体」。严禁使用任何 Emoji。\n"
            "根据后端返回的患者档案 JSON 向会诊医生简洁、准确地汇报：\n"
            "- 若未绑定手机号，提示医生输入会诊患者手机号；\n"
            "- 若已找到患者，清晰展示“您正在会诊：姓名（性别、年龄、慢病史/眼科史），当前院内左右眼影像拍摄与待分析状态”，并提示医生可直接下达“辅助诊断左眼/右眼”或“调出历史报告”指令。"
        ),
        user_prompt="医生指令：{{#start.query#}}\n院内患者档案接口返回：{{#http_api.body#}}",
    )


def build_d10_doctor_diagnose(origin: str, api_key: str) -> dict:
    return _simple_http_llm_workflow(
        name="D10_院内影像辅助诊断",
        desc="医生端核心阅片工作流：自动抓取患者左/右眼最新院内影像执行六模型与四类病灶定位推理，结合医学指南库输出临床风险提示与医生版 PDF 报告。",
        icon="D1",
        http_title="执行六模型与病灶定位推理",
        url=f"{origin}/v1/his/agent/doctor/diagnose",
        api_key=api_key,
        read_timeout=120,
        llm_title="临床专业辅助诊断报告",
        with_kb=True,
        system_prompt=(
            "你是「视健康·眼科医生端临床辅助智能体」。严禁使用任何 Emoji。\n"
            "请严格忠于后端六模型推理与临床规则接口返回的结果及医学知识库指南：\n"
            "1. 简洁专业汇报：当前会诊患者、分析眼别、DR 分级（0~4级）、达到研究筛查线的异常标签、四类病灶定位关注区（MA/HE/EX/SE）及同眼纵向趋势；\n"
            "2. 明确给出专科进一步重点核查方向（如黄斑 OCT、眼压、视野、FFA 等）及知识库指南来源依据；\n"
            "3. 务必保留并输出接口返回的临床版报告链接（doctor_report_url），方便医生点击查看双图并导出标准 A4 PDF。"
        ),
        user_prompt=(
            "医生指令：{{#start.query#}}\n"
            "辅助推理与规则返回：{{#http_api.body#}}\n"
            "医学指南参考片段：{{#kb_search.result#}}"
        ),
    )


def build_d20_doctor_records(origin: str, api_key: str) -> dict:
    return _simple_http_llm_workflow(
        name="D20_患者报告调阅与对比",
        desc="医生端报告调阅工作流：快速检索当前会诊患者或指定手机号名下的左/右眼归档临床 PDF 报告链接。",
        icon="D2",
        http_title="调取归档临床报告列表",
        url=f"{origin}/v1/his/agent/doctor/records",
        api_key=api_key,
        read_timeout=15,
        llm_title="呈现临床报告索引",
        system_prompt=(
            "你是「视健康·眼科医生端临床辅助智能体」。严禁使用任何 Emoji。\n"
            "根据后端返回的归档报告列表，简洁列出患者各次左/右眼检查的时间、DR 分级与可直接点击打开的临床报告链接（doctor_report_url），绝不篡改链接与分期数字。"
        ),
        user_prompt="医生指令：{{#start.query#}}\n归档报告查询返回：{{#http_api.body#}}",
    )


def build_w90_fallback(origin: str, _api_key: str) -> dict:
    doc = _base_app(
        "W90_通用兜底引导",
        "多智能体默认兜底工作流：处理超出眼底健康范围的问题，并提示门户与拍片室演示入口。",
        "W9",
    )
    nodes = [
        _start_query_only(80, 240),
        _llm_node(
            "llm_fallback",
            "礼貌边界与引导说明",
            380,
            240,
            system_prompt=(
                "你是「视健康」眼科智能体系统的默认兜底助手。严禁使用任何 Emoji。\n"
                "当用户输入与眼科检查/日常护眼/患者会诊无关时，礼貌说明自己的服务范围（医院眼底报告解读、症状分诊、眼科科普与医生辅助会诊），"
                "若提及突发黑影、闪光感、遮挡或剧痛等症状仍立即提醒去眼科急诊；并可告知医患门户体验地址："
                f"{origin}/v1/demo ，拍片室同步模拟地址：{origin}/v1/imaging_room 。"
            ),
            user_prompt="用户输入：{{#start.query#}}",
        ),
        _node("end", "end", "结束", 680, 240, {"outputs": [{"variable": "answer", "value_selector": ["llm_fallback", "text"]}]}),
    ]
    edges = [
        _edge("start", "start", "llm_fallback", "llm"),
        _edge("llm_fallback", "llm", "end", "end"),
    ]
    doc["workflow"] = _workflow_block(nodes, edges)
    return doc


WORKFLOW_BUILDERS = {
    "patient/P00_患者身份与档案绑定.yml": build_p00_patient_session,
    "patient/P10_患者报告温暖解读.yml": build_p10_patient_report,
    "patient/P20_结合病史症状问诊.yml": build_p20_patient_symptom,
    "patient/P30_眼健康日常科普答疑.yml": build_p30_patient_education,
    "doctor/D00_会诊患者接诊与影像概览.yml": build_d00_doctor_patient,
    "doctor/D10_院内影像辅助诊断.yml": build_d10_doctor_diagnose,
    "doctor/D20_患者报告调阅与对比.yml": build_d20_doctor_records,
    "W90_通用兜底引导.yml": build_w90_fallback,
}


def export_dify_workflows(output_dir: Path, *, origin: str = DEFAULT_ORIGIN, api_key: str = PLACEHOLDER_API_KEY) -> list[Path]:
    output = output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "dify_workflows"
        stage.mkdir()
        for rel_name, builder in WORKFLOW_BUILDERS.items():
            payload = builder(origin, api_key)
            text = yaml.dump(payload, Dumper=_NoAliasDumper, sort_keys=False, allow_unicode=True)
            target = stage / rel_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        stage.rename(output)
    for rel_name in WORKFLOW_BUILDERS:
        written.append(output / rel_name)
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
