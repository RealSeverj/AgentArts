"""Owned Windows process entry point. API credentials stay in the environment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import uuid


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "inference_runtime" / "local_gpu"
CONFIG = RUNTIME / "service.json"
RECORD = RUNTIME / "server_process.json"


def _write_request_receipt(*, request_id: str | None, status_code: int, instance_id: str) -> None:
    """Write only response identity, never request/response content or addresses."""
    target = RUNTIME / "last_screen_request.json"
    temporary = target.with_name("last_screen_request." + uuid.uuid4().hex + ".tmp")
    try:
        payload = {
            "request_id": request_id,
            "http_status": status_code,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "instance_id": instance_id,
        }
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
    finally:
        # A failed optional receipt must neither change nor interrupt HTTP output.
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass


class LocalRequestReceipt:
    """Observe completion of the single inference route without reading its body."""

    def __init__(self, app, instance_id: str):
        self.app = app
        self.instance_id = instance_id

    async def __call__(self, scope, receive, send):
        if (scope.get("type") != "http" or scope.get("method") != "POST"
                or scope.get("path") != "/v1/screen_fundus"):
            return await self.app(scope, receive, send)
        request_id = None
        status_code = None

        async def observed_send(message):
            nonlocal request_id, status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                for name, value in message.get("headers", ()):
                    if name.lower() == b"x-request-id":
                        # The owned API uses lowercase UUID hex identifiers.
                        try:
                            candidate = value.decode("ascii")
                            if len(candidate) == 32 and all(c in "0123456789abcdef" for c in candidate):
                                request_id = candidate
                        except (UnicodeError, AttributeError):
                            pass
            await send(message)
            if (message["type"] == "http.response.body" and not message.get("more_body", False)
                    and status_code is not None):
                try:
                    _write_request_receipt(request_id=request_id, status_code=status_code,
                                           instance_id=self.instance_id)
                except Exception:
                    pass

        return await self.app(scope, receive, observed_send)


def create_local_app(instance_id: str):
    from fastapi import HTTPException, Response
    from inference.app import create_app
    # Use the operator-owned tunnel record, never an inbound Host header.
    try:
        tunnel = json.loads((RUNTIME / "tunnel-state.json").read_text(encoding="utf-8-sig"))
        origin = tunnel.get("public_url") if not tunnel.get("stopped_utc") else None
    except (FileNotFoundError, ValueError):
        origin = None
    app = create_app(visual_origin=origin)

    def public_jpeg(source, expected_sha256, headers):
        limit = 2 * 1024 * 1024
        try:
            if source.is_symlink():
                raise OSError("Validation sample must be an ordinary file")
            with source.open("rb") as handle:
                payload = handle.read(limit + 1)
        except OSError:
            raise HTTPException(status_code=404, detail="Public validation sample unavailable.") from None
        if len(payload) > limit or hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise HTTPException(status_code=404, detail="Public validation sample unavailable.")
        return Response(payload, media_type="image/jpeg", headers={"cache-control": "no-store", **headers})

    @app.get("/validation/public-retina.jpg", include_in_schema=False)
    def public_validation_sample():
        return public_jpeg(RUNTIME / "retina_public_sample_proxy.jpg",
                           "38a07f36f27f095e818aea7b96d34202c05176d30253c66733f2e00379e9e0e6",
                           {"x-validation-sample": "public-scikit-image-retina"})

    @app.get("/validation/idrid-03.jpg", include_in_schema=False)
    def public_idrid_sample():
        source = ROOT / "data/IDRiD/extracted/A. Segmentation/1. Original Images/a. Training Set/IDRiD_03.jpg"
        return public_jpeg(source, "092d05e8bb47568b511d896dd8ec79dccf9f5dd53bcb8d3ce5ec1a6c38d3e1b3", {
            "x-validation-sample": "IDRiD_03, Porwal et al., CC BY 4.0",
            "Link": '<https://creativecommons.org/licenses/by/4.0/>; rel="license", '
                    '<https://doi.org/10.3390/data3030025>; rel="describedby"',
        })

    return LocalRequestReceipt(app, instance_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", required=True, type=uuid.UUID)
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)

    # Lock the owned runtime file, so two simultaneous starts cannot both load
    # the models. Windows releases this lock when its owning process exits.
    import msvcrt
    with (RUNTIME / "server.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("Another owned local inference process is starting or running.") from None

        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        os.environ["RETINA_SERVICE_CONFIG"] = str(CONFIG)
        os.environ["RETINA_DEVICE"] = "cuda"
        os.environ["RETINA_PRECISION"] = "fp16"
        os.environ.pop("RETINA_REGISTRY", None)
        os.environ.pop("RETINA_WEIGHTS_ROOT", None)
        os.environ.setdefault("PYTHONUTF8", "1")
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        if not CONFIG.is_file():
            raise SystemExit("The owned local service configuration is missing.")
        if len(os.environ.get("RETINA_API_KEY", "").encode("utf-8")) < 32:
            raise SystemExit("A valid RETINA_API_KEY must be provided in the process environment.")

        record = {
            "kind": "retina_local_gpu_inference_process_v1",
            "instance_id": str(args.instance_id),
            "pid": os.getpid(),
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "python": str(Path(sys.executable).resolve()),
            "launcher": str(Path(__file__).resolve()),
            "project_root": str(ROOT),
            "config": str(CONFIG),
            "listen_url": "http://127.0.0.1:8000",
        }
        temporary_record = RECORD.with_name("server_process." + str(args.instance_id) + ".tmp")
        temporary_record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary_record.replace(RECORD)

        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable; install the intended CUDA-enabled inference environment first.")
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        logging.info("local_inference_starting instance=%s device=cuda precision=fp16", args.instance_id)

        import uvicorn
        uvicorn.run(
            create_local_app(str(args.instance_id)), host="127.0.0.1", port=8000,
            workers=1, access_log=False, proxy_headers=False,
        )


if __name__ == "__main__":
    main()
