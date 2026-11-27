"""Portable entry points for same-image offline reporting and the AgentArts HTTPS backend."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import html
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def service_settings(device):
    from inference.settings import Settings

    return Settings(registry=ROOT / "portable/models.json", weights_root=ROOT,
                    research_root=ROOT,
                    workflow_db=ROOT / "inference_runtime/clinical_workflow.sqlite3",
                    hospital_root=ROOT / "inference_runtime/hospital_storage",
                    device=device, precision="auto",
                    cpu_threads=min(8, os.cpu_count() or 1))


def new_machine_key():
    supplied = os.environ.get("RETINA_API_KEY")
    if supplied:
        return supplied
    key_file = Path.home() / ".retina-screening" / "api-key.txt"
    key_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        fd = os.open(key_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(secrets.token_urlsafe(48) + "\n")
    print(f"API key file: {key_file}")
    return key_file.read_text(encoding="ascii").strip()


def serve(args):
    import uvicorn
    from inference.app import create_app

    settings = replace(service_settings(args.device), api_key=new_machine_key(),
                       visual_origin=args.public_origin)
    llm_context = nullcontext()
    use_local = getattr(args, "local_llm", False)
    use_maas = getattr(args, "maas_llm", False)
    if use_local and use_maas:
        raise ValueError("请选择一种报告解读服务：--local-llm 或 --maas-llm。")
    if (use_local or use_maas) and (args.host not in {"127.0.0.1", "localhost", "::1"} or args.public_origin):
        print("[大模型解读] 启用条件为本机回环地址和本机门户。", flush=True)
    elif use_maas:
        from inference.local_llm import MaaSReportExplainer

        api_key = os.environ.get("MAAS_API_KEY", "").strip()
        if api_key:
            llm_context = nullcontext(MaaSReportExplainer(api_key=api_key))
            print("[华为 MaaS] 已配置 DeepSeek 报告解读。", flush=True)
        else:
            print("[华为 MaaS] 请设置 MAAS_API_KEY 以启用 DeepSeek 解读，病例报告可继续使用。", flush=True)
    elif use_local:
        from inference.local_llm import DEFAULT_LOCAL_LLM_PORT
        from portable.local_llm_server import local_llm_server

        llm_port = getattr(args, "local_llm_port", None)
        if llm_port is None:
            llm_port = DEFAULT_LOCAL_LLM_PORT
        if not 1 <= llm_port <= 65535 or llm_port == args.port:
            print("[本地大模型] 请指定一个与门户不同的有效端口。", flush=True)
        else:
            llm_context = local_llm_server(ROOT, llm_port)

    with llm_context as report_explainer:
        app = create_app(settings, report_explainer=report_explainer)
        uvicorn.run(app, host=args.host, port=args.port, access_log=False, proxy_headers=False)


def demo(args):
    import torch
    from inference.clinical_review import build_clinical_review
    from inference.clinical_workflow import evaluate_care_rules
    from inference.engine import FundusEngine
    from inference.image_io import URLFetchPolicy, decode_image_bytes
    from inference.registry import ModelRegistry
    from inference.research_experts import ResearchExperts
    from inference.visual_report import write_visual_report

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    policy = URLFetchPolicy()
    with args.image.open("rb") as stream:
        payload = stream.read(policy.max_download_bytes + 1)
    decoded = decode_image_bytes(payload, policy=policy)
    settings = service_settings(args.device)
    device = settings.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(settings.cpu_threads)
    registry = ModelRegistry.from_file(settings.registry, settings.weights_root)
    engine = FundusEngine(registry, device=device, routing=settings.routing,
                          research=ResearchExperts(settings.research_root, device))
    output.parent.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex
    with decoded.image as image, tempfile.TemporaryDirectory(dir=output.parent) as folder:
        engine.load()
        result = engine.screen(image, request_id=request_id, image_sha256=decoded.sha256)
        result["warnings"].extend(decoded.warnings)
        visual = write_visual_report(image, result, decoded.sha256, folder, request_id)
        review = build_clinical_review(result, visual)
        result["visual_report"] = visual
        result["report_summary"]["clinical_review"] = review
        result["report_summary"]["clinical_decision"] = evaluate_care_rules(result, {})["clinical_decision"]
        completed = Path(folder) / request_id
        if not completed.exists():
            completed.mkdir()
            message = " ".join((review["quality"], review["primary"]))
            (completed / "report.html").write_text(
                '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>眼底阅片</title>'
                + "<h1>图像质控结果</h1><p>" + html.escape(message) + "</p></html>", encoding="utf-8")
        (completed / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                          allow_nan=False), encoding="utf-8")
        completed.rename(output)
    print(f"Status: {result['status']}; localization: {visual['localization_status']}")
    print(output / "report.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    local = sub.add_parser("demo", help="Generate a report from a local eye image")
    local.add_argument("--image", type=Path, default=ROOT / "examples/IDRiD_03.jpg")
    local.add_argument("--output", type=Path, required=True, help="New directory for report files")
    server = sub.add_parser("serve", help="Start the inference API")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--public-origin", default=os.environ.get("RETINA_PUBLIC_ORIGIN"),
                        help="Public HTTPS origin serving this API and /v1/view for AgentArts")
    llm_provider = server.add_mutually_exclusive_group()
    llm_provider.add_argument("--local-llm", action="store_true", help="Enable the bundled offline report explanation model")
    llm_provider.add_argument("--maas-llm", action="store_true", help="Enable Huawei MaaS DeepSeek report explanation using MAAS_API_KEY")
    server.add_argument("--local-llm-port", type=int,
                        help="Loopback port for the local language model (default: 8091)")
    for command in (local, server):
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.command == "demo":
        demo(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
