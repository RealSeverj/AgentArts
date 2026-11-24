"""Portable entry points for same-image offline reporting and the AgentArts HTTPS backend."""
import argparse
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
    app = create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False, proxy_headers=False)


def demo(args):
    import torch
    from inference.clinical_review import build_clinical_review
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
