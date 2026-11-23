"""Verify an authenticated deployed endpoint using one public scientific image.

The API key stays in memory. No signed attachment URLs, headers, or credentials
are written to receipts. This verifies deployment, not diagnostic accuracy.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PUBLIC_IMAGE = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/retina.jpg"
PUBLIC_SHA = "38a07f36f27f095e818aea7b96d34202c05176d30253c66733f2e00379e9e0e6"


def read_api_key():
    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    payload = (ROOT / "inference_runtime/retina_api_key.dpapi").read_bytes()
    buffer = ctypes.create_string_buffer(payload)
    encrypted = Blob(len(payload), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    decrypted = Blob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptUnprotectData.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    if not crypt32.CryptUnprotectData(ctypes.byref(encrypted), None, None, None, None, 0,
                                    ctypes.byref(decrypted)):
        raise RuntimeError("Current-user DPAPI decryption failed")
    try:
        return ctypes.string_at(decrypted.pbData, decrypted.cbData).decode("utf-8")
    finally:
        kernel32.LocalFree(decrypted.pbData)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-url-stdin", action="store_true",
                        help="Read a temporary URL for the same public sample from stdin; never persist it")
    args = parser.parse_args()
    parsed = urlsplit(args.origin)
    valid = (parsed.scheme == "https" or
             (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}))
    if not valid or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        parser.error("Use an HTTPS origin, or an explicit loopback HTTP origin")
    origin = args.origin.rstrip("/")
    image_url = sys.stdin.read().strip() if args.image_url_stdin else PUBLIC_IMAGE
    if not image_url:
        parser.error("The test image URL is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {"recorded_utc": datetime.now(timezone.utc).isoformat(), "origin": origin,
               "scope": "deployed_authenticated_HTTP_with_public_sample_not_accuracy_evaluation",
               "source_image": ({"host": urlsplit(image_url).hostname, "temporary_url_omitted": True}
                                if args.image_url_stdin else PUBLIC_IMAGE),
               "source_sha256": PUBLIC_SHA, "checks": {},
               "tls_verification_enabled": True, "training_started": False}
    try:
        import httpx
        from scripts.smoke_inference_real import _response_checks

        schema = json.loads((ROOT / "delivery/agentarts_integration_20261114/tool_contract.json").read_text(
            encoding="utf-8-sig"))["response_schema"]
        key = read_api_key()
        with httpx.Client(timeout=60, trust_env=False, follow_redirects=False) as client:
            alive = client.get(origin + "/healthz")
            ready = client.get(origin + "/readyz")
            receipt["checks"]["healthz_200"] = alive.status_code == 200
            receipt["checks"]["readyz_true"] = ready.status_code == 200 and ready.json().get("ready") is True
            if not receipt["checks"]["readyz_true"]:
                raise RuntimeError("Endpoint is not ready")
            unauthorized = client.post(origin + "/v1/screen_fundus", json={"image_url": image_url})
            receipt["checks"]["missing_key_rejected401"] = unauthorized.status_code == 401
            headers = {"X-API-Key": key}
            private = client.post(origin + "/v1/screen_fundus", headers=headers,
                                  json={"image_url": "https://127.0.0.1/private.png"})
            receipt["checks"]["private_image_source_rejected422"] = (
                private.status_code == 422 and private.json().get("status") == "input_error")
            started = time.monotonic()
            result = client.post(origin + "/v1/screen_fundus", headers=headers, json={"image_url": image_url})
            receipt["request_seconds"] = round(time.monotonic() - started, 4)
            response = result.json()
            key = None
        receipt["checks"].update(_response_checks(response, source_sha256=PUBLIC_SHA, schema=schema))
        receipt["checks"]["http200"] = result.status_code == 200
        receipt["checks"]["request_id_matches_header"] = response.get("request_id") == result.headers.get("x-request-id")
        receipt["http_status"] = result.status_code
        receipt["response_status"] = response.get("status")
        receipt["request_id"] = response.get("request_id")
        receipt["stage_summary"] = {name: {field: stage.get(field) for field in (
            "status", "reason", "model_id", "weights_sha256", "preprocessing_version", "elapsed_ms")}
            for name, stage in response.get("results", {}).items()}
        (args.output_dir / "public_response.json").write_text(json.dumps(
            response, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    except Exception as error:
        receipt["error_type"] = type(error).__name__
    receipt["all_checks_passed"] = bool(receipt["checks"]) and all(receipt["checks"].values()) and "error_type" not in receipt
    (args.output_dir / "verification.json").write_text(json.dumps(
        receipt, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_passed": receipt["all_checks_passed"], "http_status": receipt.get("http_status"),
                      "response_status": receipt.get("response_status"), "request_seconds": receipt.get("request_seconds"),
                      "request_id": receipt.get("request_id"), "failed_checks": [
                          name for name, passed in receipt["checks"].items() if not passed],
                      "error_type": receipt.get("error_type"),
                      "receipt": str(args.output_dir / "verification.json")}, ensure_ascii=False))
    return 0 if receipt["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
