"""解读 portable/run.py demo 生成的公开样图结果。"""

import argparse
import getpass
import json
import os
from pathlib import Path

import httpx


MAAS_ENDPOINT = "https://api.modelarts-maas.com/v2/chat/completions"
MAAS_MODEL = "deepseek-v4.1-flash-1017"
LOCAL_MODEL = "qwen2.5-1.5b-local"
QUALITY_LABELS = {"Good": "良好", "Usable": "可用", "Reject": "拒收"}
LESION_STATUS = {
    "candidates_present": "分割模型标出了候选区域，供人工核查",
    "no_candidates": "分割模型本次未标出候选区域",
    "reference_only": "叠加图使用公开专家参考标注",
    "quality_blocked": "图像质控未通过，未执行病灶定位",
}


def report_evidence(report: dict) -> dict:
    quality = (report["results"]["quality"].get("output") or {}).get("predicted_label")
    grade = report["report_summary"]["dr_severity"].get("grade")
    candidate_status = report["visual_report"].get("candidate_status")
    return {
        "图像质控": QUALITY_LABELS.get(quality, "未获得结果"),
        "模型糖网分级": f"{grade}级" if type(grade) is int else "未获得分级",
        "病灶标注状态": LESION_STATUS.get(candidate_status, "未获得定位结果"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="公开样图的 result.json")
    parser.add_argument("--port", type=int, default=8091, help="本机大模型 API 端口")
    parser.add_argument("--model", help="覆盖所选服务的默认模型")
    parser.add_argument("--maas", action="store_true", help="通过华为 MaaS 解读结构化结果")
    args = parser.parse_args()

    report = json.loads(args.result.read_text(encoding="utf-8"))
    evidence = report_evidence(report)
    opening = (
        f"这张公开样图的图像质控为「{evidence['图像质控']}」，"
        f"筛查模型给出的糖网分级为「{evidence['模型糖网分级']}」。"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "用一句通俗中文说明这张公开眼底样图的病灶标注状态。"
                "如果有候选区域，说明这些标记供人工核查。"
                "仅使用提供的事实，禁止新增诊断、病灶名称、治疗或复查建议。"
            ),
        },
        {"role": "user", "content": evidence["病灶标注状态"]},
    ]

    payload = {"messages": messages, "temperature": 0, "max_tokens": 240}
    headers = {}
    output_name = "llm_interpretation.json"
    if args.maas:
        key = os.environ.get("MAAS_API_KEY") or getpass.getpass("华为 MaaS API Key: ")
        if not key:
            parser.error("MaaS 需要 API 密钥")
        endpoint = MAAS_ENDPOINT
        headers["Authorization"] = f"Bearer {key}"
        payload.update(model=args.model or MAAS_MODEL, thinking={"type": "disabled"}, stream=False)
        output_name = "maas_interpretation.json"
    else:
        endpoint = f"http://127.0.0.1:{args.port}/v1/chat/completions"
        payload["model"] = args.model or LOCAL_MODEL

    with httpx.Client(timeout=90, trust_env=False) as client:
        response = client.post(endpoint, headers=headers, json=payload)
        if not response.is_success:
            raise SystemExit(f"解读服务返回 HTTP {response.status_code}")
        reply = response.json()

    answer = reply["choices"][0]
    text = (answer["message"].get("content") or "").strip()
    if not text or answer.get("finish_reason") != "stop":
        raise RuntimeError("模型未返回完整解读")
    interpretation = f"{opening}{text}"

    output = args.result.with_name(output_name)
    output.write_text(json.dumps({
        "model": reply["model"],
        "endpoint": endpoint,
        "image_sha256": report["image_sha256"],
        "report_evidence": evidence,
        "llm_input": evidence["病灶标注状态"],
        "llm_explanation": text,
        "interpretation": interpretation,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(interpretation)
    print(output)


if __name__ == "__main__":
    main()
