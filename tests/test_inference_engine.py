"""Inference behavior fixtures, not estimates of trained model accuracy.

No training data, real weights, downloads, or GPU are required by these tests.
The fixed-output modules deliberately exercise routing and artifact generation.
"""
from copy import deepcopy
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from fastapi.testclient import TestClient
from pydantic import ValidationError

from inference.app import create_app
from inference.engine import FundusEngine, validate_routing
from inference.image_io import DecodedImage, decode_image_bytes
from inference.report_summary import ELEVEN_LABELS
from inference.schemas import FundusResponse
from inference.settings import Settings
from inference.registry import (DISEASE_LABELS, MASK_LABELS, MEAN, SPECS, STAGES, STD,
                                LoadedExpert, ModelRegistry, RegistryEntry, load_verified_checkpoint,
                                validate_checkpoint_config)
from training.classification import ClassificationTransform
from training.fundus_preprocessing import DEFAULT_SPEC
from training.structural import _tensor


def fixture_config(stage):
    spec = SPECS[stage]
    result = {"task": spec["task"], "kind": spec["kind"], "labels": spec["labels"],
              "model": {"name": spec["architecture"], "img_size": spec["size"],
                        "mean": MEAN, "std": STD},
              "data": {"datasets": [spec["source"]], "source_isolated": True,
                       "protocol": "strict_cross_domain"},
              "training": {"epochs": 40, "workers": 2, "log_every": 10}}
    if "num_classes" in spec:
        result["num_classes"] = spec["num_classes"]
    if stage == "dr_grading":
        result["preprocessing"] = deepcopy(DEFAULT_SPEC)
        result["initialization"] = {"parent_training_sources": ["APTOS2019"]}
        result["augmentation"] = {"resize_before_augmentation": True}
    elif stage == "disc_cup_segmentation":
        result["model"].update(base_channels=32, pretrained=False)
        result["evaluation"] = {"threshold": 0.5}
    return deepcopy(result)


def fixture_checkpoint(stage):
    config = fixture_config(stage)
    identity = {k: v for k, v in config.items() if k != "notes"}
    identity["training"] = {k: v for k, v in config["training"].items() if k not in ("workers", "log_every")}
    return {"config": config, "epoch": SPECS[stage]["epoch"] - 1,
            "data_identity": {"experiment": identity}, "model": {"fixture": torch.zeros(1)}}


class FixedModel(nn.Module):
    def __init__(self, value, *, delay=0):
        super().__init__()
        self.register_buffer("value", value)
        self.calls = 0
        self.inputs = []
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0
        self.guard = threading.Lock()

    def forward(self, image):
        with self.guard:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            self.calls += 1
            self.inputs.append(image.detach().cpu().clone())
            if self.delay:
                time.sleep(self.delay)
            return self.value.clone()
        finally:
            with self.guard:
                self.in_flight -= 1


class FakeRegistry:
    def __init__(self, quality=(10., 0., 0.), multidisease=None, *, delay=0):
        seg = torch.full((1, 2, 512, 512), -10.)
        seg[0, 1, 10:14, 10:14] = 10.  # 16 disc pixels
        seg[0, 0, 11:13, 11:13] = 10.  # 4 cup pixels contained in disc
        seg[0, 0, 2, 2] = 10.  # 1 genuinely outside disc
        values = {"quality": torch.tensor([quality], dtype=torch.float32),
                  "multidisease": torch.tensor([multidisease if multidisease is not None else [-10.] * 5]),
                  "dr_grading": torch.tensor([[0., 1., 3., 0., -1.]]),
                  "disc_cup_segmentation": seg}
        self.experts = {stage: LoadedExpert(stage, SPECS[stage]["model_id"], SPECS[stage]["sha256"],
                                            fixture_config(stage), FixedModel(values[stage], delay=delay),
                                            SPECS[stage]["epoch"]) for stage in STAGES}
        self.calls = 0

    def load_models(self):
        self.calls += 1
        return self.experts


class InferenceEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.image = Image.new("RGB", (400, 200), (150, 45, 20))
        self.digest = hashlib.sha256(self.image.tobytes()).hexdigest()

    def screen(self, engine):
        return engine.screen(self.image, request_id="unit-fixture", image_sha256=self.digest)

    def make_engine(self, **registry_kwargs):
        registry = FakeRegistry(**registry_kwargs)
        return registry, FundusEngine(registry, device="cpu").load()

    def post_fixture_screen(self, registry, research=None, *, decoded=None):
        engine = FundusEngine(registry, device="cpu", research=research)
        key = "integration-test-" + "x" * 40

        def fetch_fixture(url, *, policy):
            if decoded is not None:
                return decoded
            fixture = self.image.copy()
            fixture.putpixel((0, 0), (149, 45, 20))
            return DecodedImage(fixture, hashlib.sha256(fixture.tobytes()).hexdigest(), "PNG", 400, 200, 128)

        with TestClient(create_app(Settings(api_key=key), engine=engine, fetcher=fetch_fixture)) as client:
            return client.post("/v1/screen_fundus", json={"image_url": "https://example.org/fundus.png"},
                               headers={"X-API-Key": key})

    def test_blank_and_invisible_images_are_input_errors_before_all_six_models(self):
        for invisible in (False, True):
            with self.subTest(invisible=invisible):
                with Image.new("RGBA", (4, 4), (255, 255, 255, 255)) as image, io.BytesIO() as buffer:
                    if invisible:
                        image.putalpha(0)
                        image.putpixel((2, 2), (0, 255, 0, 0))
                    image.save(buffer, format="PNG")
                    decoded = decode_image_bytes(buffer.getvalue())
                registry, research = FakeRegistry(), Mock()
                response = self.post_fixture_screen(registry, research, decoded=decoded)
                self.assertEqual(200, response.status_code)
                body = response.json()
                FundusResponse.model_validate(body)
                self.assertEqual("input_error", body["status"])
                self.assertEqual("research-20261122.1", body["contract_version"])
                reason = "image_fully_transparent" if invisible else "image_uniform"
                self.assertEqual(decoded.sha256, body["image_sha256"])
                self.assertIn("透明" if invisible else "一种颜色", body["message"])
                stages = {**body["results"], **body["research_results"]}
                self.assertEqual({*STAGES, "multilabel_11", "glaucoma"}, set(stages))
                for stage in stages.values():
                    self.assertEqual("not_run", stage["status"])
                    self.assertIsNone(stage["output"])
                    self.assertEqual(reason, stage["reason"])
                self.assertIsNone(body["report_summary"]["dr_severity"]["grade"])
                self.assertTrue(all(expert.model.calls == 0 for expert in registry.experts.values()))
                research.execute.assert_not_called()

    def test_research_stages_follow_quality_even_if_envelope_is_mislabelled(self):
        for quality_logits, expected_status, http_status in (
            ((0., 0., 10.), "quality_rejected", 200),
            ((float("nan"), 0., 0.), "error", 500),
        ):
            with self.subTest(status=expected_status):
                research = Mock()
                response = self.post_fixture_screen(FakeRegistry(quality=quality_logits), research)
                self.assertEqual(http_status, response.status_code)
                body = response.json()
                self.assertEqual(expected_status, body["status"])
                self.assertEqual("research-20261122.1", body["contract_version"])
                self.assertTrue(all(stage["status"] == "not_run" for stage in body["research_results"].values()))
                self.assertIsNone(body["report_summary"]["dr_severity"]["grade"])
                research.execute.assert_not_called()

                # Even an error envelope must reject a leaked post-quality execution.
                body["status"] = "error"
                body["research_results"]["glaucoma"]["status"] = "error"
                with self.assertRaises(ValidationError):
                    FundusResponse.model_validate(body)

    def test_http_reports_four_six_and_partial_results(self):
        legacy = self.post_fixture_screen(FakeRegistry())
        self.assertEqual(200, legacy.status_code)
        self.assertEqual("draft-20261114.1", legacy.json()["contract_version"])
        self.assertNotIn("report_summary", legacy.json())
        self.assertNotIn("research_results", legacy.json())

        for failing_stage in (None, "glaucoma", "multidisease"):
            with self.subTest(failing_stage=failing_stage):
                def predict(stage, image):
                    labels = ELEVEN_LABELS if stage == "multilabel_11" else ("glaucoma",)
                    result = {"status": "completed", "reason": None, "model_id": stage,
                              "weights_sha256": "b" * 64, "elapsed_ms": 0.1,
                              "preprocessing_version": "test", "output": {"scores": dict.fromkeys(labels, 0.42)}}
                    if stage == failing_stage:
                        result.update(status="error", reason="inference_failed", output=None)
                    return result

                research = Mock()
                research.execute.side_effect = predict
                registry = FakeRegistry()
                if failing_stage == "multidisease":
                    registry.experts["multidisease"].model.value.fill_(float("nan"))
                response = self.post_fixture_screen(registry, research)
                self.assertEqual(200, response.status_code)
                body = response.json()
                self.assertEqual("partial" if failing_stage else "ok", body["status"])
                self.assertEqual("all", body["routing"]["mode"])
                for stage, result in body["results"].items():
                    self.assertEqual("error" if stage == failing_stage else "completed", result["status"])
                    self.assertEqual(1, registry.experts[stage].model.calls)
                self.assertEqual(["multilabel_11", "glaucoma"],
                                 [call.args[0] for call in research.execute.call_args_list])
                report = body["report_summary"]
                self.assertEqual(2, report["dr_severity"]["grade"])
                self.assertEqual(11, len(report["research"]["multilabel_11"]["items"]))
                glaucoma = report["research"]["glaucoma"]
                if failing_stage == "glaucoma":
                    self.assertEqual("error", glaucoma["status"])
                    self.assertFalse(glaucoma["items"])
                else:
                    self.assertEqual("42.000分", glaucoma["items"][0]["score_display"])
                    self.assertIsNone(glaucoma["items"][0]["severity_grade"])

    def test_unloaded_and_failed_load_never_claim_readiness(self):
        registry = FakeRegistry()
        engine = FundusEngine(registry, device="cpu")
        self.assertFalse(engine.is_ready)
        response = self.screen(engine)
        self.assertEqual(response["status"], "tool_unavailable")
        self.assertTrue(all(item["output"] is None and item["model_id"] is None for item in response["results"].values()))
        registry.experts.pop("quality")
        with self.assertRaises(ValueError):
            engine.load()
        self.assertFalse(engine.is_ready)
        self.assertEqual(engine.model_info(), {})
        self.assertEqual(engine.last_load_error, "ValueError")

    def test_successful_load_is_once_and_frozen(self):
        registry, engine = self.make_engine()
        self.assertIs(engine.load(), engine)
        self.assertTrue(engine.is_ready)
        self.assertEqual(registry.calls, 1)
        self.assertTrue(all(not expert.model.training and not any(p.requires_grad for p in expert.model.parameters())
                            for expert in registry.experts.values()))
        info = engine.model_info()
        self.assertEqual(info["dr_grading"]["completed_epoch"], 24)
        self.assertNotIn("checkpoint", json.dumps(info))

    def test_quality_reject_has_no_downstream_execution(self):
        registry, engine = self.make_engine(quality=(0., 0., 10.))
        response = self.screen(engine)
        self.assertEqual(response["status"], "quality_rejected")
        for stage in STAGES[1:]:
            result = response["results"][stage]
            self.assertEqual(result["status"], "not_run")
            self.assertEqual(result["reason"], "quality_rejected")
            self.assertIsNone(result["output"])
            self.assertIsNone(result["elapsed_ms"])
            self.assertEqual(registry.experts[stage].model.calls, 0)

    def test_usable_warns_and_default_all_does_not_apply_candidate_gate(self):
        registry, engine = self.make_engine(quality=(0., 10., 0.))
        response = self.screen(engine)
        self.assertEqual(response["status"], "ok")
        self.assertTrue(any("Usable" in warning for warning in response["warnings"]))
        self.assertEqual(response["routing"]["mode"], "all")
        self.assertLess(response["results"]["multidisease"]["output"]["scores"]["DR"], .3)
        self.assertTrue(all(result["status"] == "completed" for result in response["results"].values()))
        self.assertTrue(all(expert.model.calls == 1 for expert in registry.experts.values()))
        json.dumps(response, allow_nan=False)

    def test_multidisease_scores_are_independent_and_not_binary_decisions(self):
        _, engine = self.make_engine(multidisease=[8., 7., 6., 5., -8.])
        output = self.screen(engine)["results"]["multidisease"]["output"]
        self.assertEqual(output["label_order"], list(DISEASE_LABELS))
        self.assertGreater(sum(output["scores"].values()), 3.)
        self.assertLess(output["scores"]["ABNORMAL"], .001)
        self.assertIsNone(output["binary_decisions"])
        self.assertIsNone(output["decision_thresholds"])
        self.assertEqual(output["calibration_state"], "not_validated_for_production")

    def test_threshold_routing_is_explicit_and_preserves_skips(self):
        policy = {"mode": "threshold", "policy_id": "fixture-candidate", "validation_passed": False,
                  "thresholds": {"dr_grading": .7, "disc_cup_segmentation": .4}}
        with self.assertRaises(ValueError):
            FundusEngine(FakeRegistry(), device="cpu", routing=policy)
        policy["allow_unvalidated"] = True
        registry = FakeRegistry(multidisease=[0.] * 5)
        engine = FundusEngine(registry, device="cpu", routing=policy).load()
        response = self.screen(engine)
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["results"]["dr_grading"]["reason"], "score_below_gate")
        self.assertIsNone(response["results"]["dr_grading"]["output"])
        self.assertEqual(registry.experts["dr_grading"].model.calls, 0)
        self.assertEqual(response["results"]["disc_cup_segmentation"]["status"], "completed")
        self.assertTrue(any("尚未经独立验证" in warning for warning in response["warnings"]))
        policy.update(validation_passed=True, validation_reference="fixture-only-declaration")
        self.assertTrue(validate_routing(policy)["validation_passed"])
        policy["validation_reference"] = ""
        with self.assertRaises(ValueError):
            validate_routing(policy)
        policy.update(validation_passed=False, thresholds={"dr_grading": float("nan"), "disc_cup_segmentation": .4})
        with self.assertRaises(ValueError):
            validate_routing(policy)

    def test_nonfinite_quality_fails_closed(self):
        registry, engine = self.make_engine(quality=(float("nan"), 0., 0.))
        response = self.screen(engine)
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["results"]["quality"]["status"], "error")
        self.assertIsNone(response["results"]["quality"]["output"])
        self.assertTrue(all(response["results"][stage]["reason"] == "quality_failed" for stage in STAGES[1:]))
        self.assertTrue(all(registry.experts[stage].model.calls == 0 for stage in STAGES[1:]))

    def test_multidisease_failure_retains_quality_and_skips_dependents(self):
        registry, engine = self.make_engine(multidisease=[float("inf")] * 5)
        response = self.screen(engine)
        self.assertEqual(response["status"], "partial")
        self.assertEqual(response["results"]["quality"]["status"], "completed")
        self.assertEqual(response["results"]["multidisease"]["status"], "error")
        for stage in STAGES[2:]:
            self.assertEqual(response["results"][stage]["reason"], "multidisease_failed")
            self.assertEqual(registry.experts[stage].model.calls, 0)

    def test_dr_failure_does_not_erase_independent_segmentation(self):
        registry, engine = self.make_engine()
        registry.experts["dr_grading"].model.value.fill_(float("nan"))
        response = self.screen(engine)
        self.assertEqual(response["status"], "partial")
        self.assertEqual(response["results"]["dr_grading"]["status"], "error")
        self.assertEqual(response["results"]["disc_cup_segmentation"]["status"], "completed")

    def test_segmentation_artifacts_decode_to_actual_masks_and_geometry(self):
        _, engine = self.make_engine()
        response = self.screen(engine)
        output = response["results"]["disc_cup_segmentation"]["output"]
        self.assertEqual(output["channel_order"], list(MASK_LABELS))
        decoded = {}
        for label, artifact in output["mask_artifacts"].items():
            payload = base64.b64decode(artifact["data"], validate=True)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), artifact["sha256"])
            with Image.open(io.BytesIO(payload)) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.mode, "L")
                self.assertEqual(image.size, (512, 512))
                decoded[label] = np.asarray(image).copy()
                self.assertEqual(set(np.unique(decoded[label])), {0, 255})
        self.assertEqual(int((decoded["optic_cup"] > 0).sum()), 5)
        self.assertEqual(int((decoded["optic_disc"] > 0).sum()), 16)
        self.assertEqual(output["cup_outside_disc_pixels"], 1)
        self.assertFalse(output["containment_correction_applied"])
        self.assertEqual(decoded["optic_cup"][2, 2], 255)
        self.assertEqual(decoded["optic_disc"][2, 2], 0)
        self.assertEqual(output["source_size_wh"], [400, 200])
        self.assertEqual(output["mapping_to_source"]["scale_x"], 400 / 512)
        self.assertEqual(output["mapping_to_source"]["scale_y"], 200 / 512)
        self.assertIsNone(output["derived_measurements"])

    def test_each_stage_receives_original_with_its_own_validation_transform(self):
        registry, engine = self.make_engine()
        before = self.image.tobytes()
        self.screen(engine)
        for stage in STAGES:
            expert = registry.experts[stage]
            expected = (_tensor(self.image.resize((512, 512), Image.Resampling.BILINEAR), expert.config)
                        if stage == "disc_cup_segmentation"
                        else ClassificationTransform(expert.config, training=False)(self.image))
            self.assertTrue(torch.equal(expert.model.inputs[0], expected.unsqueeze(0)), stage)
        quality = registry.experts["quality"].model.inputs[0]
        dr = registry.experts["dr_grading"].model.inputs[0]
        self.assertGreater(float(quality[0, 0, -1, 10]), 0.)
        self.assertLess(float(dr[0, 0, -1, 10]), -2.)  # DR black padding only
        self.assertEqual(self.image.tobytes(), before)
        self.assertEqual(tuple(registry.experts["quality"].model.inputs[0].shape), (1, 3, 392, 392))
        self.assertEqual(tuple(dr.shape), (1, 3, 336, 336))

    def test_requests_share_serialized_models(self):
        registry, engine = self.make_engine(delay=.01)
        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(lambda _: self.screen(engine), range(2)))
        self.assertTrue(all(response["status"] == "ok" for response in responses))
        self.assertTrue(all(expert.model.calls == 2 and expert.model.max_in_flight == 1 for expert in registry.experts.values()))


class RegistryTests(unittest.TestCase):
    def test_registry_constructs_exact_models_offline_and_loads_strictly(self):
        class StateOnlyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fixture = nn.Parameter(torch.ones(1))
                self.strict_loads = []

            def load_state_dict(self, state_dict, strict=True):
                self.strict_loads.append(strict)
                return super().load_state_dict(state_dict, strict=strict)

        entries = {stage: RegistryEntry(stage, Path(stage + ".pt"), SPECS[stage]["sha256"],
                                        SPECS[stage]["model_id"]) for stage in STAGES}
        seen = []

        def build(config, pretrained):
            seen.append((config["task"], pretrained))
            return StateOnlyModel()

        with patch("inference.registry.load_verified_checkpoint", side_effect=lambda path, sha: fixture_checkpoint(path.stem)), \
             patch("training.classification.build_model", side_effect=build), \
             patch("training.structural.build_model", side_effect=build):
            loaded = ModelRegistry(entries).load_models()
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(pretrained is False for _, pretrained in seen))
        self.assertEqual(set(loaded), set(STAGES))
        for expert in loaded.values():
            self.assertEqual(expert.model.strict_loads, [True])
            self.assertFalse(expert.model.training)
            self.assertFalse(expert.model.fixture.requires_grad)
            self.assertTrue(torch.equal(expert.model.fixture, torch.zeros(1)))

    def test_bad_hash_never_calls_unpickler(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "untrusted.pt"
            path.write_bytes(b"this must not reach torch.load")
            with patch("inference.registry.torch.load") as load:
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    load_verified_checkpoint(path, "0" * 64)
                load.assert_not_called()

    def test_matching_hash_loads_small_local_checkpoint_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            torch.save({"model": {"x": torch.ones(2)}}, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result = load_verified_checkpoint(path, digest)
            self.assertTrue(torch.equal(result["model"]["x"], torch.ones(2)))
            self.assertEqual(result["model"]["x"].device.type, "cpu")

    def test_config_schema_rejects_wrong_labels_epoch_source_geometry(self):
        mutations = [lambda ckpt: ckpt["config"]["labels"].reverse(),
                     lambda ckpt: ckpt.update(epoch=33),
                     lambda ckpt: ckpt["config"]["data"].update(datasets=["DDR"]),
                     lambda ckpt: ckpt["config"]["model"].update(img_size=336),
                     lambda ckpt: ckpt["model"].update(fixture=torch.tensor([float("nan")])),
                     lambda ckpt: ckpt["config"].update(preprocessing=deepcopy(DEFAULT_SPEC))]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                checkpoint = fixture_checkpoint("multidisease")
                mutate(checkpoint)
                with self.assertRaises(ValueError):
                    validate_checkpoint_config("multidisease", checkpoint)
        for stage in STAGES:
            self.assertEqual(validate_checkpoint_config(stage, fixture_checkpoint(stage)), fixture_config(stage))

    def test_registry_pins_audited_version_and_confines_weights_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = {"models": {stage: {"checkpoint": stage + ".pt", "sha256": SPECS[stage]["sha256"],
                                          "model_id": SPECS[stage]["model_id"]} for stage in STAGES}}
            path = root / "registry.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            registry = ModelRegistry.from_file(path, weights_root=root)
            self.assertEqual(set(registry.entries), set(STAGES))
            document["models"]["quality"]["sha256"] = "0" * 64
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "audited"):
                ModelRegistry.from_file(path)
            document["models"]["quality"]["sha256"] = SPECS["quality"]["sha256"]
            document["models"]["quality"]["checkpoint"] = "../outside.pt"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "relative"):
                ModelRegistry.from_file(path, weights_root=root)


if __name__ == "__main__":
    unittest.main()
