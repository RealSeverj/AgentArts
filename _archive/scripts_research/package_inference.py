"""Build an allowlisted source delivery, with no secrets, datasets or model binaries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = Path("delivery/agentarts_integration_20261114")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / DELIVERY / "retina_inference_v0.1.0.zip")
    args = parser.parse_args()
    paths = [ROOT / name for name in (
        "inference/__init__.py", "inference/__main__.py", "inference/app.py", "inference/engine.py",
        "inference/image_io.py", "inference/registry.py", "inference/schemas.py", "inference/settings.py",
        "configs/inference/models.json", "configs/inference/service.json",
        "deployment/inference/install_runtime.sh", "deployment/inference/start.sh",
        "deployment/inference/retina-inference.service.example", "deployment/inference/nginx.conf.example",
        "deployment/inference/environment.example", "tests/test_inference_api.py", "tests/test_inference_engine.py",
        "tests/test_inference_image_io.py", "tests/test_inference_settings.py",
        "requirements-inference.txt", "requirements-inference-test.txt", "scripts/__init__.py",
        "scripts/manifest_dataset.py", "scripts/prepare_inference_weights.py", "scripts/package_inference.py",
        "scripts/check_inference_environment.py", "scripts/smoke_inference_real.py",
        "training/__init__.py", "training/classification.py", "training/structural.py",
        "training/fundus_preprocessing.py", "training/metrics.py",
    )]
    paths += [ROOT / DELIVERY / name for name in (
        "tool_contract.json", "agent_system_prompt.txt", "推理接口使用说明.txt", "openapi.json",
        "verification.json", "real_smoke.json", "real_smoke.receipt.json", "INFERENCE_REVIEW.txt",
    )]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "kind": "inference_source_only",
                "weights_included": False, "production_endpoint": None, "files": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(paths)):
            if not path.is_file():
                raise FileNotFoundError(path)
            data = path.read_bytes()
            relative = path.relative_to(ROOT).as_posix()
            archive.writestr(relative, data)
            manifest["files"].append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        archive.writestr("SOURCE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".zip.sha256").write_text(f"{digest}  {args.output.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(args.output), "bytes": args.output.stat().st_size,
                      "files": len(paths), "sha256": digest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
