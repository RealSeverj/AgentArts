"""Tests for SQLite clinical workflow store, symptom-first care rules, and eye comparisons."""
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from inference.app import create_app
from inference.clinical_workflow import ClinicalWorkflowStore
from inference.image_io import DecodedImage
from inference.settings import Settings

KEY = "workflow-test-" + "k" * 32
HEADERS = {"X-API-Key": KEY}
URL = "https://public.example/fundus.jpg"


def fake_fetcher(url, *, policy):
    return DecodedImage(Image.new("RGB", (64, 64), (100, 60, 40)), "c" * 64, "JPEG", 64, 64, 128)


class SequencedEngine:
    """Fixture engine supporting Good/Reject/Normal/DR2 responses across multiple calls."""

    def __init__(self):
        self.is_ready = False
        self.mode = "dr2"

    def load(self):
        self.is_ready = True

    def model_info(self):
        return {}

    def screen(self, image, *, request_id, image_sha256):
        if self.mode == "reject":
            return {
                "contract_version": "research-20261122.1",
                "request_id": request_id,
                "status": "quality_rejected",
                "message": "图像质量不足，请重拍或人工复核；未执行下游筛查。",
                "image_sha256": image_sha256,
                "created_utc": "2026-11-23T08:00:00+00:00",
                "routing": {"mode": "all", "policy_id": "test", "validation_passed": False, "stage_reasons": {}},
                "results": {
                    "quality": {
                        "status": "completed", "reason": None, "model_id": "qc_v1",
                        "weights_sha256": "1" * 64, "preprocessing_version": "prep_v1",
                        "elapsed_ms": 5.0, "output": {"predicted_label": "Reject"},
                    },
                    "multidisease": {"status": "not_run", "reason": "quality_rejected", "model_id": None, "weights_sha256": None, "preprocessing_version": None, "elapsed_ms": None, "output": None},
                    "dr_grading": {"status": "not_run", "reason": "quality_rejected", "model_id": None, "weights_sha256": None, "preprocessing_version": None, "elapsed_ms": None, "output": None},
                    "disc_cup_segmentation": {"status": "not_run", "reason": "quality_rejected", "model_id": None, "weights_sha256": None, "preprocessing_version": None, "elapsed_ms": None, "output": None},
                },
                "research_results": {
                    "multilabel_11": {"status": "not_run", "reason": "quality_rejected", "model_id": None, "weights_sha256": None, "preprocessing_version": None, "elapsed_ms": None, "output": None},
                    "glaucoma": {"status": "not_run", "reason": "quality_rejected", "model_id": None, "weights_sha256": None, "preprocessing_version": None, "elapsed_ms": None, "output": None},
                },
                "report_summary": {"dr_severity": {"status": "not_run", "grade": None}, "clinical_review": {"quality": "图像质量不足。"}},
                "warnings": [],
            }

        dr_grade = 2 if self.mode == "dr2" else 0
        dr_score = 0.85 if self.mode == "dr2" else 0.96
        dr_met = self.mode == "dr2"
        ls_met = self.mode == "dr2"
        return {
            "contract_version": "research-20261122.1",
            "request_id": request_id,
            "status": "ok",
            "message": "图像分析完成；结果供人工复核。",
            "image_sha256": image_sha256,
            "created_utc": "2026-11-23T08:01:00+00:00",
            "routing": {"mode": "all", "policy_id": "test", "validation_passed": False, "stage_reasons": {}},
            "results": {
                "quality": {
                    "status": "completed", "reason": None, "model_id": "qc_v1",
                    "weights_sha256": "1" * 64, "preprocessing_version": "prep_v1",
                    "elapsed_ms": 5.0, "output": {"predicted_label": "Good"},
                },
                "multidisease": {
                    "status": "completed", "reason": None, "model_id": "five_v1",
                    "weights_sha256": "2" * 64, "preprocessing_version": "prep_v1",
                    "elapsed_ms": 5.0, "output": {"scores": {"DR": 0.9 if dr_met else 0.01}},
                },
                "dr_grading": {
                    "status": "completed", "reason": None, "model_id": "dr_v1",
                    "weights_sha256": "3" * 64, "preprocessing_version": "prep_v1",
                    "elapsed_ms": 5.0, "output": {"predicted_grade": dr_grade},
                },
                "disc_cup_segmentation": {
                    "status": "completed", "reason": None, "model_id": "seg_v1",
                    "weights_sha256": "4" * 64, "preprocessing_version": "prep_v1",
                    "elapsed_ms": 5.0, "output": {"mask_artifacts": {}},
                },
            },
            "research_results": {
                "multilabel_11": {
                    "status": "completed", "reason": None, "model_id": "eleven_v1",
                    "weights_sha256": "5" * 64, "preprocessing_version": "prep_v1",
                    "elapsed_ms": 5.0, "output": {"scores": {"LS": 0.02 if ls_met else 0.0001}},
                },
                "glaucoma": {
                    "status": "completed", "reason": None, "model_id": "glau_v1",
                    "weights_sha256": "6" * 64, "preprocessing_version": "prep_v1",
                    "elapsed_ms": 5.0, "output": {"scores": {"glaucoma": 0.05}},
                },
            },
            "report_summary": {
                "dr_severity": {
                    "status": "completed", "grade": dr_grade,
                    "level": "中度（非增殖期）" if dr_grade == 2 else "未见明显糖网病变（本项正常档）",
                    "clinical_stage": "中度非增殖期糖尿病视网膜病变" if dr_grade == 2 else "未见明显糖尿病视网膜病变",
                    "score": dr_score, "score_display": f"{dr_score * 100:.3f}分",
                },
                "five_label_screening": {
                    "status": "completed", "model_id": "five_v1",
                    "items": [{"code": "DR", "name": "糖尿病视网膜病变", "kind": "疾病",
                               "score": 0.9 if dr_met else 0.01, "research_threshold": 0.04,
                               "research_threshold_met": dr_met}],
                },
                "research": {
                    "multilabel_11": {
                        "status": "completed", "model_id": "eleven_v1",
                        "items": [{"code": "LS", "name": "激光瘢痕", "kind": "治疗痕迹",
                                   "score": 0.02 if ls_met else 0.0001, "research_threshold": 0.001,
                                   "research_threshold_met": ls_met}],
                    },
                    "glaucoma": {
                        "status": "completed", "model_id": "glau_v1",
                        "items": [{"code": "glaucoma", "name": "青光眼", "kind": "疾病",
                                   "score": 0.05, "research_threshold": 0.15,
                                   "research_threshold_met": False}],
                    },
                },
                "clinical_review": {"quality": "图像质量良好，可判读。", "primary": "ok"},
            },
            "warnings": [],
        }


class ClinicalWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = Path(self.tmpdir.name) / "workflow.sqlite3"
        self.engine = SequencedEngine()

    def make_client(self):
        settings = Settings(api_key=KEY, workflow_db=self.db_path)
        return TestClient(create_app(settings, engine=self.engine, fetcher=fake_fetcher))

    def test_auth_empty_context_and_validation(self):
        with self.make_client() as client:
            self.assertEqual(client.get("/v1/context").status_code, 401)
            empty = client.get("/v1/context", headers=HEADERS)
            self.assertEqual(empty.status_code, 200)
            self.assertFalse(empty.json()["data"]["has_active_assessment"])
            missing = client.get("/v1/assessments/nonexistent", headers=HEADERS)
            self.assertEqual(missing.status_code, 404)
            bad = client.post("/v1/intake", json={"eye": "both"}, headers=HEADERS)
            self.assertEqual(bad.status_code, 422)
            self.assertEqual(bad.json()["error"]["code"], "invalid_request")

    def test_screening_intake_care_and_symptom_first_override(self):
        with self.make_client() as client:
            # 1. DR grade 2 + LS triggered
            r1 = client.post("/v1/screen_fundus", json={"image_url": URL}, headers=HEADERS)
            self.assertEqual(r1.status_code, 200)
            id1 = r1.json()["request_id"]

            ctx = client.get("/v1/context", headers=HEADERS).json()["data"]
            self.assertTrue(ctx["has_active_assessment"])
            self.assertEqual(ctx["active_assessment_id"], id1)
            self.assertEqual(ctx["eye"], "unknown")

            asmt = client.get(f"/v1/assessments/{id1}", headers=HEADERS).json()["data"]["evidence_packet"]
            self.assertEqual(asmt["schema_version"], "evidence.v1")
            self.assertEqual(asmt["ood"]["status"], "not_evaluated")
            self.assertEqual(asmt["quality"]["clinical_gradability"], "not_evaluated")

            care1 = client.post("/v1/care/evaluate", json={"assessment_id": id1}, headers=HEADERS).json()["data"]
            self.assertEqual(care1["care_priority"], "scheduled_ophthalmology")
            self.assertLessEqual(len(care1["questions_to_confirm"]), 2)
            self.assertIn("eye", care1["missing_facts"])

            # 2. Add negative symptom + eye + diabetes + laser history (as comma-separated string & empty optional fields) -> ready plan
            intake1 = client.post(
                "/v1/intake",
                json={
                    "assessment_id": id1,
                    "session_id": "",
                    "eye": "left",
                    "symptoms": "无飞蚊，无闪光感",
                    "history": "2型糖尿病6年, 2024年左眼激光光凝",
                    "symptom_onset": "",
                    "confirmed_plan": None,
                },
                headers=HEADERS,
            )
            self.assertEqual(intake1.status_code, 200)
            self.assertEqual(intake1.json()["data"]["intake_revision"], 2)

            care2 = client.post("/v1/care/evaluate", json={"assessment_id": "", "session_id": ""}, headers=HEADERS).json()["data"]
            self.assertEqual(care2["care_priority"], "scheduled_ophthalmology")
            self.assertEqual(care2["missing_facts"], [])
            self.assertEqual(care2["plan_status"], "ready")
            self.assertEqual(care2["version"], 2)

            # 3. Normal DR0 image BUT urgent warning symptoms -> urgent_ophthalmology override
            self.engine.mode = "normal"
            r2 = client.post("/v1/screen_fundus", json={"image_url": URL}, headers=HEADERS)
            id2 = r2.json()["request_id"]
            client.post(
                "/v1/intake",
                json={"assessment_id": id2, "eye": "left", "symptoms": ["今天突然出现大量飞蚊伴闪光感"]},
                headers=HEADERS,
            )
            care_urgent = client.post("/v1/care/evaluate", json={"assessment_id": id2}, headers=HEADERS).json()["data"]
            self.assertEqual(care_urgent["care_priority"], "urgent_ophthalmology")
            self.assertIn("RULE_URGENT_SYMPTOM_01", care_urgent["rule_ids"])
            self.assertIn("SRC_NEI_RD_URGENT", care_urgent["references"])

        # 4. Verify persistence across store reopen
        reopened = ClinicalWorkflowStore(self.db_path)
        try:
            persisted = reopened.get_assessment(id1)
            self.assertIsNotNone(persisted)
            self.assertEqual(persisted["evidence_packet"]["eye"], "left")
            self.assertEqual(persisted["latest_care_plan"]["version"], 2)
        finally:
            reopened.close()

    def test_comparisons_and_quality_rejection_rules(self):
        with self.make_client() as client:
            id_a = client.post("/v1/screen_fundus", json={"image_url": URL}, headers=HEADERS).json()["request_id"]
            self.engine.mode = "normal"
            id_b = client.post("/v1/screen_fundus", json={"image_url": URL}, headers=HEADERS).json()["request_id"]

            # Unknown eyes must not be compared
            cmp_unknown = client.post(
                "/v1/comparisons",
                json={"previous_assessment_id": id_a, "current_assessment_id": id_b},
                headers=HEADERS,
            ).json()["data"]
            self.assertFalse(cmp_unknown["comparable"])
            self.assertEqual(cmp_unknown["reason"], "eye_unconfirmed")

            # Different eyes must be rejected
            client.post("/v1/intake", json={"assessment_id": id_a, "eye": "left"}, headers=HEADERS)
            client.post("/v1/intake", json={"assessment_id": id_b, "eye": "right"}, headers=HEADERS)
            cmp_diff = client.post(
                "/v1/comparisons",
                json={"previous_assessment_id": id_a, "current_assessment_id": id_b},
                headers=HEADERS,
            ).json()["data"]
            self.assertFalse(cmp_diff["comparable"])
            self.assertEqual(cmp_diff["reason"], "different_eye")

            # Same eye succeeds
            client.post("/v1/intake", json={"assessment_id": id_b, "eye": "left"}, headers=HEADERS)
            cmp_ok = client.post(
                "/v1/comparisons",
                json={"previous_assessment_id": id_a, "current_assessment_id": id_b},
                headers=HEADERS,
            ).json()["data"]
            self.assertTrue(cmp_ok["comparable"])
            self.assertEqual(cmp_ok["dr_grade_comparison"]["grade_delta"], -2)
            self.assertIn("LS", cmp_ok["screening_flag_changes"]["resolved_below_threshold"])

            # Quality rejected image -> retake_image
            self.engine.mode = "reject"
            id_rej = client.post("/v1/screen_fundus", json={"image_url": URL}, headers=HEADERS).json()["request_id"]
            care_rej = client.post("/v1/care/evaluate", json={"assessment_id": id_rej}, headers=HEADERS).json()["data"]
            self.assertEqual(care_rej["care_priority"], "retake_image")
            client.post("/v1/intake", json={"assessment_id": id_rej, "eye": "left"}, headers=HEADERS)
            cmp_rej = client.post(
                "/v1/comparisons",
                json={"previous_assessment_id": id_a, "current_assessment_id": id_rej},
                headers=HEADERS,
            ).json()["data"]
            self.assertFalse(cmp_rej["comparable"])
            self.assertEqual(cmp_rej["reason"], "quality_not_comparable")

    def test_dynamic_report_syncs_latest_care_plan_and_pdf_print_styles(self):
        settings = Settings(api_key=KEY, workflow_db=self.db_path, visual_origin="https://severj.top")
        with TestClient(create_app(settings, engine=self.engine, fetcher=fake_fetcher)) as client:
            screen = client.post("/v1/screen_fundus", json={"image_url": URL}, headers=HEADERS)
            self.assertEqual(screen.status_code, 200)
            payload = screen.json()
            report_url = payload["visual_report"]["report_url"]
            self.assertTrue(report_url.startswith("https://severj.top/v1/view/"))
            view_path = "/" + report_url.split("https://severj.top/", 1)[1]

            initial_html = client.get(view_path)
            self.assertEqual(initial_html.status_code, 200)
            self.assertIn("导出 / 打印 PDF 报告", initial_html.text)
            self.assertIn("@media print", initial_html.text)
            self.assertIn("眼别待确认", initial_html.text)

            client.post(
                "/v1/intake",
                json={"assessment_id": payload["request_id"], "eye": "left", "symptoms": "突发大量飞蚊"},
                headers=HEADERS,
            )
            care = client.post("/v1/care/evaluate", json={"assessment_id": payload["request_id"]}, headers=HEADERS)
            self.assertEqual(care.status_code, 200)
            self.assertEqual(care.json()["data"]["report_url"], report_url)

            updated_html = client.get(view_path)
            self.assertEqual(updated_html.status_code, 200)
            self.assertIn("左眼（OS）", updated_html.text)
            self.assertIn("突发大量飞蚊", updated_html.text)
            self.assertIn("立即眼科急诊评估", updated_html.text)
            self.assertIn("随访计划 v1", updated_html.text)

    def test_two_sequential_images_preserve_distinct_eye_bindings(self):
        settings = Settings(api_key=KEY, workflow_db=self.db_path, visual_origin="https://severj.top")
        with TestClient(create_app(settings, engine=self.engine, fetcher=fake_fetcher)) as client:
            r_left = client.post(
                "/v1/screen_fundus",
                json={"image_url": URL, "eye": "左眼", "symptoms": "轻微视物模糊"},
                headers=HEADERS,
            )
            self.assertEqual(r_left.status_code, 200)
            id_left = r_left.json()["request_id"]
            path_left = "/" + r_left.json()["visual_report"]["report_url"].split("https://severj.top/", 1)[1]

            self.engine.mode = "normal"
            r_right = client.post(
                "/v1/screen_fundus",
                json={"image_url": URL, "eye": "right"},
                headers=HEADERS,
            )
            self.assertEqual(r_right.status_code, 200)
            id_right = r_right.json()["request_id"]
            path_right = "/" + r_right.json()["visual_report"]["report_url"].split("https://severj.top/", 1)[1]

            html_left = client.get(path_left).text
            html_right = client.get(path_right).text
            self.assertIn("左眼（OS）", html_left)
            self.assertNotIn("右眼（OD）", html_left)
            self.assertIn("轻微视物模糊", html_left)
            self.assertIn("右眼（OD）", html_right)
            self.assertNotIn("左眼（OS）", html_right)

            # Updating left image later with explicit assessment_id never mutates right image
            client.post(
                "/v1/intake",
                json={"assessment_id": id_left, "history": "2型糖尿病3年"},
                headers=HEADERS,
            )
            left_asmt = client.get(f"/v1/assessments/{id_left}", headers=HEADERS).json()["data"]["evidence_packet"]
            right_asmt = client.get(f"/v1/assessments/{id_right}", headers=HEADERS).json()["data"]["evidence_packet"]
            self.assertEqual(left_asmt["eye"], "left")
            self.assertEqual(left_asmt["intake"]["history"], ["2型糖尿病3年"])
            self.assertEqual(right_asmt["eye"], "right")
            self.assertEqual(right_asmt["intake"]["history"], [])

    def test_demo_workbench_multi_agent_and_no_image_flows(self):
        settings = Settings(api_key=KEY, workflow_db=self.db_path, visual_origin="https://severj.top")
        with TestClient(create_app(settings, engine=self.engine, fetcher=fake_fetcher)) as client:
            page = client.get("/v1/demo")
            self.assertEqual(page.status_code, 200)
            self.assertIn("眼底筛查与随访多智能体演示工作台", page.text)
            self.assertIn("connect-src 'self'", page.headers.get("content-security-policy", ""))

            sample_img = client.get("/v1/demo/sample/idrid_03")
            self.assertEqual(sample_img.status_code, 200)
            self.assertEqual(client.get("/v1/demo/sample/unknown_id").status_code, 404)

            # 1. Sample eye screening run (A1 -> W10 -> A2 -> A3 -> A4)
            run1 = client.post(
                "/v1/demo/run",
                json={
                    "mode": "sample",
                    "sample_id": "idrid_03",
                    "audience": "caregiver",
                    "eye": "left",
                    "symptoms": "视物模糊",
                    "history": "糖尿病5年",
                },
            )
            self.assertEqual(run1.status_code, 200)
            data1 = run1.json()
            self.assertTrue(data1["ok"])
            self.assertEqual(data1["eye"], "left")
            self.assertTrue(data1["visual_urls"]["report"].startswith("/v1/view/"))
            self.assertIn("patient", data1["narratives"])
            self.assertIn("caregiver", data1["narratives"])
            self.assertIn("clinician", data1["narratives"])
            self.assertGreaterEqual(len(data1["trace"]), 4)

            # 2. Same-eye second run triggers automatic A4 comparison & isolates default session
            self.engine.mode = "normal"
            run2 = client.post(
                "/v1/demo/run",
                json={"mode": "sample", "sample_id": "public_proxy", "audience": "clinician", "eye": "left"},
            )
            self.assertEqual(run2.status_code, 200)
            self.assertTrue(run2.json()["comparison_preview"]["comparable"])
            ctx_default = client.get("/v1/context", headers=HEADERS).json()["data"]
            self.assertFalse(ctx_default["has_active_assessment"])

            # 3. No-image urgent symptom override vs general checkup Q&A
            no_img_urgent = client.post(
                "/v1/demo/run",
                json={"mode": "no_image", "query_text": "今天突然眼前大量飞蚊飘动伴闪光感"},
            )
            self.assertEqual(no_img_urgent.status_code, 200)
            self.assertEqual(no_img_urgent.json()["care_priority"], "urgent_ophthalmology")
            self.assertIn("RULE_URGENT_SYMPTOM_01", no_img_urgent.json()["rule_ids"])

            no_img_benign = client.post(
                "/v1/demo/run",
                json={"mode": "no_image", "query_text": "体检写着豹纹状眼底TSLN是什么意思"},
            )
            self.assertEqual(no_img_benign.status_code, 200)
            self.assertEqual(no_img_benign.json()["care_priority"], "no_image_guidance")


if __name__ == "__main__":
    unittest.main()
