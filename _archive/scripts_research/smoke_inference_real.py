"""Exercise real owned weights through the authenticated HTTP app, locally.

This offline smoke check never downloads an image or weights and never starts
training. The injected fetcher accepts one fixed test URL only; production URL
validation remains unchanged. Its response is per-image evidence, not an
accuracy/QWK evaluation or a deployment receipt.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import secrets
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _versions():
    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "timm", "Pillow", "numpy", "fastapi", "pydantic", "httpx", "jsonschema"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _response_checks(response, *, source_sha256, schema):
    import numpy as np
    from PIL import Image
    from jsonschema import Draft202012Validator, FormatChecker
    from inference.registry import DISEASE_LABELS, MASK_LABELS, QUALITY_LABELS, SPECS, STAGES

    checks = {"response_schema_valid": not list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(response)),
              "image_sha256_matches_original_bytes": response.get("image_sha256") == source_sha256,
              "request_id_recorded": bool(response.get("request_id")),
              "overall_no_runtime_error": response.get("status") in {"ok", "quality_rejected"},
              "default_all_policy_reported": (response.get("routing") or {}).get("mode") == "all"}
    # The HTTP response itself must remain valid finite JSON, including nested outputs.
    try:
        json.dumps(response, allow_nan=False)
        checks["finite_json"] = True
    except (ValueError, TypeError):
        checks["finite_json"] = False
    stages = response.get("results", {})
    for stage in STAGES:
        result = stages.get(stage, {})
        if result.get("status") == "completed":
            checks[stage + "_provenance"] = (result.get("model_id") == SPECS[stage]["model_id"] and
                                              result.get("weights_sha256") == SPECS[stage]["sha256"] and
                                              bool(result.get("preprocessing_version")) and
                                              isinstance(result.get("elapsed_ms"), (int, float)) and result["elapsed_ms"] >= 0)
        else:
            checks[stage + "_unexecuted_has_no_output"] = result.get("status") == "not_run" and result.get("output") is None and result.get("elapsed_ms") is None
    quality = stages.get("quality", {}).get("output")
    if quality:
        scores = quality.get("scores", {})
        values = [scores.get(label) for label in QUALITY_LABELS]
        valid = all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in values)
        checks["quality_scores_and_argmax"] = (valid and quality.get("class_order") == list(QUALITY_LABELS) and
                                                abs(sum(values) - 1) < 1e-5 and
                                                quality.get("predicted_label") == QUALITY_LABELS[max(range(3), key=values.__getitem__)])
        if quality.get("predicted_label") == "Reject":
            checks["quality_reject_stops_downstream"] = (response.get("status") == "quality_rejected" and
                all(stages.get(stage, {}).get("status") == "not_run" and stages[stage].get("reason") == "quality_rejected" for stage in STAGES[1:]))
        if quality.get("predicted_label") == "Usable":
            checks["usable_warning_preserved"] = any("Usable" in warning for warning in response.get("warnings", []))
    multi = stages.get("multidisease", {}).get("output")
    if multi:
        scores = multi.get("scores", {})
        checks["multidisease_order_and_raw_score_semantics"] = (list(scores) == list(DISEASE_LABELS) and
            multi.get("label_order") == list(DISEASE_LABELS) and multi.get("binary_decisions") is None and
            multi.get("decision_thresholds") is None and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in scores.values()))
    dr = stages.get("dr_grading", {}).get("output")
    if dr:
        values = dr.get("scores", [])
        valid = len(values) == 5 and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in values)
        checks["dr_scores_and_argmax"] = (valid and dr.get("grade_order") == list(range(5)) and abs(sum(values) - 1) < 1e-5 and
                                          dr.get("predicted_grade") == max(range(5), key=values.__getitem__))
    segmentation = stages.get("disc_cup_segmentation", {}).get("output")
    if segmentation:
        masks = {}
        try:
            checks["segmentation_channel_order"] = segmentation.get("channel_order") == list(MASK_LABELS)
            for label in MASK_LABELS:
                artifact = segmentation["mask_artifacts"][label]
                payload = base64.b64decode(artifact["data"], validate=True)
                with Image.open(io.BytesIO(payload)) as image:
                    masks[label] = np.asarray(image).copy() > 0
                    checks[label + "_actual_png"] = (image.format == "PNG" and image.mode == "L" and image.size == (512, 512) and
                        set(np.unique(np.asarray(image))).issubset({0, 255}) and artifact.get("sha256") == hashlib.sha256(payload).hexdigest() and
                        artifact.get("encoding") == "base64" and artifact.get("media_type") == "image/png")
            checks["cup_outside_disc_is_computed_from_artifacts"] = (int((masks["optic_cup"] & ~masks["optic_disc"]).sum()) ==
                                                                    segmentation.get("cup_outside_disc_pixels"))
            checks["no_unvalidated_measurements_or_containment_fix"] = (segmentation.get("derived_measurements") is None and
                                                                         segmentation.get("containment_correction_applied") is False)
        except (KeyError, ValueError, TypeError, OSError):
            checks["segmentation_artifacts_decodable"] = False
    checks["all_four_stages_completed"] = all(stages.get(stage, {}).get("status") == "completed" for stage in STAGES)
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path, help="SHA-pinned four-model registry JSON")
    parser.add_argument("--weights-root", type=Path, help="Optional root for relative checkpoint paths")
    parser.add_argument("--image", required=True, type=Path, help="One explicitly selected local JPEG/PNG/WebP")
    parser.add_argument("--output", required=True, type=Path, help="Save actual full HTTP response JSON")
    parser.add_argument("--threads", type=int, default=4, help="CPU intra-op threads (default: 4)")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16"), default="auto")
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 64:
        parser.error("--threads must be in [1,64]")
    started = time.monotonic()
    receipt_path = args.output.with_suffix(".receipt.json")
    receipt = {"receipt_version": "local-real-inference-smoke-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "local_in_process_HTTP_with_real_owned_models_not_network_deployment_or_accuracy_evaluation",
               "versions": _versions(), "device": "cpu", "precision": "fp32", "cpu_threads": args.threads,
               "all_checks_passed": False, "checks": {}, "load_seconds": None, "http_request_seconds": None}
    try:
        import torch
        from fastapi.testclient import TestClient
        from inference.app import create_app
        from inference.engine import FundusEngine
        from inference.image_io import decode_image_bytes
        from inference.registry import ModelRegistry, STAGES
        from inference.settings import Settings

        torch.set_num_threads(args.threads)
        registry = ModelRegistry.from_file(args.registry, weights_root=args.weights_root)
        protected = {args.registry.resolve(), args.image.resolve(), *(entry.checkpoint.resolve() for entry in registry.entries.values())}
        if args.output.resolve() in protected or receipt_path.resolve() in protected or args.output.resolve() == receipt_path.resolve():
            # argparse exits immediately; the failure receipt must not overwrite
            # that same protected path through the general exception handler.
            parser.error("Smoke output must not overwrite inputs or checkpoints")
        payload = args.image.read_bytes()
        image_sha256 = hashlib.sha256(payload).hexdigest()
        receipt["image"] = {"sha256": image_sha256, "bytes": len(payload), "filename": args.image.name}
        engine = FundusEngine(registry, device=args.device, precision=args.precision)
        receipt["device"] = str(engine.device)
        receipt["precision"] = engine.precision
        load_started = time.monotonic()
        engine.load()
        receipt["load_seconds"] = round(time.monotonic() - load_started, 4)
        receipt["models"] = engine.model_info()
        if engine.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(engine.device)
            receipt["cuda_memory_after_load"] = {
                "allocated_bytes": torch.cuda.memory_allocated(engine.device),
                "reserved_bytes": torch.cuda.memory_reserved(engine.device),
            }
        local_url = "https://local-smoke.invalid/explicit-local-image"
        fetch_count = 0

        def local_fetcher(image_url, *, policy):
            nonlocal fetch_count
            if image_url != local_url:
                raise ValueError("The local smoke fetcher accepts its one fixed resource only")
            fetch_count += 1
            decoded = decode_image_bytes(payload, policy=policy)
            receipt["image"].update(width=decoded.width, height=decoded.height, format=decoded.original_format,
                                    exif_orientation=decoded.exif_orientation)
            return decoded

        ephemeral_key = secrets.token_urlsafe(48)
        settings = Settings(api_key=ephemeral_key, device=str(engine.device), precision=engine.precision, cpu_threads=args.threads,
                            routing=engine.routing)
        app = create_app(settings, engine=engine, fetcher=local_fetcher)
        with TestClient(app) as client:
            unauthenticated = client.post("/v1/screen_fundus", json={"image_url": local_url})
            receipt["checks"]["unauthenticated_request_rejected_before_decode"] = unauthenticated.status_code == 401 and fetch_count == 0
            receipt["checks"]["readyz_after_real_load"] = client.get("/readyz").status_code == 200
            request_started = time.monotonic()
            http_response = client.post("/v1/screen_fundus", headers={"X-API-Key": ephemeral_key}, json={"image_url": local_url})
            receipt["http_request_seconds"] = round(time.monotonic() - request_started, 4)
            response = http_response.json()
        _dump(args.output, response)
        contract = json.loads((ROOT / "delivery/agentarts_integration_20261114/tool_contract.json").read_text(encoding="utf-8-sig"))
        receipt["checks"].update(_response_checks(response, source_sha256=image_sha256, schema=contract["response_schema"]))
        receipt["checks"].update(http_status_200=http_response.status_code == 200, image_decoded_once=fetch_count == 1,
                                request_id_matches_header=response.get("request_id") == http_response.headers.get("x-request-id"))
        receipt["http_status"] = http_response.status_code
        receipt["response_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
        receipt["response_status"] = response.get("status")
        receipt["stages"] = {stage: {key: response.get("results", {}).get(stage, {}).get(key) for key in
                                     ("status", "reason", "model_id", "weights_sha256", "preprocessing_version", "elapsed_ms")} for stage in STAGES}
        if engine.device.type == "cuda":
            receipt["cuda_memory_after_request"] = {
                "allocated_bytes": torch.cuda.memory_allocated(engine.device),
                "reserved_bytes": torch.cuda.memory_reserved(engine.device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(engine.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(engine.device),
            }
        receipt["all_checks_passed"] = bool(receipt["checks"]) and all(receipt["checks"].values())
    except Exception as error:
        # The receipt records failure without persisting keys, URLs or raw paths.
        receipt["error_type"] = type(error).__name__
        receipt["all_checks_passed"] = False
        print("Smoke check failed: " + type(error).__name__, file=sys.stderr)
    receipt["elapsed_seconds"] = round(time.monotonic() - started, 4)
    _dump(receipt_path, receipt)
    print(json.dumps({"all_checks_passed": receipt["all_checks_passed"], "response_status": receipt.get("response_status"),
                      "load_seconds": receipt["load_seconds"], "http_request_seconds": receipt["http_request_seconds"],
                      "failed_checks": [key for key, passed in receipt["checks"].items() if not passed],
                      "response_file": str(args.output), "receipt_file": str(receipt_path)}, ensure_ascii=False))
    return 0 if receipt["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
