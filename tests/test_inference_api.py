"""HTTP and contract tests use fixtures, not trained-model quality estimates."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import threading
import unittest

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from PIL import Image

from inference.app import create_app
from inference.image_io import DecodedImage, InputImageError
from inference.schemas import failure_response
from inference.settings import Settings

KEY = "test-only-" + "x" * 40
HEADERS = {"X-API-Key": KEY}
URL = "https://public.example/image.jpg?signature=SECRET"
CONTRACT = json.loads((Path(__file__).resolve().parents[1] / "delivery/agentarts_integration_20261114/tool_contract.json").read_text(encoding="utf-8"))


def decoded_fetcher(url, *, policy):
    return DecodedImage(Image.new("RGB", (10, 8)), "a" * 64, "PNG", 10, 8, 80)


class FixtureEngine:
    is_ready = False
    def __init__(self, *, failure=False):
        self.failure = failure
        self.calls = 0
        self.entered, self.release = threading.Event(), threading.Event()
        self.block = False
        self.nonfinite = False
        self.inference_error = False

    def load(self):
        if self.failure:
            raise RuntimeError("private/path/SECRET")
        self.is_ready = True

    def model_info(self):
        return {}

    def screen(self, image, *, request_id, image_sha256):
        self.calls += 1
        self.entered.set()
        if self.block:
            assert self.release.wait(10)
        response = failure_response("quality_rejected", "fixture rejection", request_id, image_sha256=image_sha256)
        response["results"]["quality"] = {
            "status": "completed", "reason": None, "model_id": "fixture-model",
            "weights_sha256": "b" * 64, "preprocessing_version": "fixture-preprocessor",
            "elapsed_ms": 1.0, "output": {"predicted_label": "Reject"},
        }
        if self.nonfinite:
            response["results"]["quality"]["output"]["score"] = float("nan")
        if self.inference_error:
            response["status"] = "error"
            response["results"]["quality"].update(status="error", output=None, reason="inference_failed")
            for stage in ("multidisease", "dr_grading", "disc_cup_segmentation"):
                response["results"][stage]["reason"] = "quality_failed"
        return response


class ApiTests(unittest.TestCase):
    def client(self, engine=None, fetcher=decoded_fetcher, **overrides):
        return TestClient(create_app(Settings(api_key=KEY, **overrides), engine=engine or FixtureEngine(), fetcher=fetcher))

    def assert_contract(self, response):
        Draft202012Validator(CONTRACT["response_schema"]).validate(response.json())
        self.assertIn("x-request-id", response.headers)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_ready_auth_reject_and_contract(self):
        engine = FixtureEngine()
        with self.client(engine) as client:
            self.assertEqual(client.get("/readyz").status_code, 200)
            unauth = client.post("/v1/screen_fundus", json={"image_url": URL})
            self.assertEqual(unauth.status_code, 401)
            self.assert_contract(unauth)
            response = client.post("/v1/screen_fundus", json={"image_url": URL}, headers=HEADERS)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "quality_rejected")
            self.assertEqual(engine.calls, 1)
            self.assert_contract(response)
            self.assertEqual(client.get("/docs").status_code, 404)

    def test_no_key_or_failed_load_fail_closed(self):
        for settings, engine in ((Settings(), FixtureEngine()), (Settings(api_key=KEY), FixtureEngine(failure=True))):
            with TestClient(create_app(settings, engine=engine, fetcher=decoded_fetcher)) as client:
                self.assertEqual(client.get("/healthz").status_code, 200)
                self.assertEqual(client.get("/readyz").status_code, 503)
                response = client.post("/v1/screen_fundus", json={"image_url": URL}, headers=HEADERS)
                self.assertIn(response.status_code, (401, 503))
                self.assert_contract(response)
                self.assertEqual(engine.calls, 0)
                self.assertNotIn("SECRET", response.text)

    def test_validation_redacts_input(self):
        with self.client() as client:
            for body in ({"image_url": URL, "extra": "SECRET"}, {"image_url": {"url": URL}}, {}):
                response = client.post("/v1/screen_fundus", json=body, headers=HEADERS)
                self.assertEqual(response.status_code, 422)
                self.assertNotIn("SECRET", response.text)
                self.assert_contract(response)

    def test_chunked_body_limit_and_invalid_json(self):
        with self.client(max_body_bytes=1024) as client:
            response = client.post("/v1/screen_fundus", content=iter([b"x" * 700, b"y" * 700]), headers=HEADERS)
            self.assertEqual(response.status_code, 413)
            self.assert_contract(response)
            response = client.post("/v1/screen_fundus", content=b'{"image_url":SECRET', headers={**HEADERS, "Content-Type": "application/json"})
            self.assertEqual(response.status_code, 422)
            self.assertNotIn("SECRET", response.text)

    def test_input_errors_map_status_and_release_gate(self):
        for code, expected in (("blocked_address", 422), ("invalid_image", 422), ("image_too_large", 413),
                               ("image_pixel_limit", 413), ("unsupported_image_format", 415), ("image_animated", 415)):
            def fetcher(url, *, policy):
                raise InputImageError(code)
            with self.client(fetcher=fetcher) as client:
                for _ in range(2):
                    response = client.post("/v1/screen_fundus", json={"image_url": URL}, headers=HEADERS)
                    self.assertEqual(response.status_code, expected)
                    self.assertEqual(response.json()["status"], "input_error")
                    self.assertEqual(response.json()["contract_version"], "draft-20261114.1")
                    self.assert_contract(response)

    def test_nonfinite_output_is_safe_error(self):
        engine = FixtureEngine()
        engine.nonfinite = True
        with self.client(engine) as client:
            response = client.post("/v1/screen_fundus", json={"image_url": URL}, headers=HEADERS)
            self.assertEqual(response.status_code, 500)
            self.assertNotIn("NaN", response.text)
            self.assert_contract(response)

    def test_quality_execution_failure_is_http500_with_real_stage_state(self):
        engine = FixtureEngine()
        engine.inference_error = True
        with self.client(engine) as client:
            response = client.post("/v1/screen_fundus", json={"image_url": URL}, headers=HEADERS)
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json()["image_sha256"], "a" * 64)
            self.assertEqual(response.json()["results"]["quality"]["status"], "error")
            self.assertEqual(response.json()["results"]["quality"]["elapsed_ms"], 1.0)
            self.assertIsNone(response.json()["results"]["quality"]["output"])
            self.assert_contract(response)

    def test_busy_does_not_fetch_another_image(self):
        engine = FixtureEngine()
        engine.block = True
        fetches = []
        def fetcher(url, *, policy):
            fetches.append(url)
            return decoded_fetcher(url, policy=policy)
        with self.client(engine, fetcher=fetcher) as client, ThreadPoolExecutor(2) as pool:
            pending = pool.submit(client.post, "/v1/screen_fundus", json={"image_url": URL}, headers=HEADERS)
            self.assertTrue(engine.entered.wait(5))
            try:
                response = client.post("/v1/screen_fundus", json={"image_url": URL}, headers=HEADERS)
                self.assertEqual(response.status_code, 429)
                self.assertEqual(len(fetches), 1)
                self.assert_contract(response)
                self.assertTrue(all(x["reason"] == "tool_unavailable" for x in response.json()["results"].values()))
            finally:
                engine.release.set()
            self.assertEqual(pending.result().status_code, 200)

    def test_openapi_has_auth_tool_name_and_four_stage_schema(self):
        document = create_app(Settings(api_key=KEY)).openapi()
        operation = document["paths"]["/v1/screen_fundus"]["post"]
        self.assertEqual(operation["operationId"], "screen_fundus")
        self.assertIn("APIKeyHeader", document["components"]["securitySchemes"])
        self.assertTrue(operation["security"])
        self.assertNotIn("servers", document)


if __name__ == "__main__":
    unittest.main()
