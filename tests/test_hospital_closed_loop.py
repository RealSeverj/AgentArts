"""End-to-end tests for Hospital Imaging Room, Phone-keyed PACS, and Patient/Doctor Agents."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

from fastapi.testclient import TestClient
from PIL import Image

from inference.app import create_app
from inference.local_llm import LocalExplanationUnavailable
from inference.settings import Settings
from inference.visual_report import LESIONS
from tests.test_clinical_workflow import VISUAL_ORIGIN, SequencedEngine, write_sample_image

KEY = "hospital-loop-" + "k" * 32
HEADERS = {"X-API-Key": KEY}


def _jpeg_b64(color=(120, 60, 40)) -> str:
    buf = BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _lesion_masks(image: Image.Image) -> dict:
    masks = {code: Image.new("L", image.size) for code in LESIONS}
    masks["HE"].paste(255, (12, 12, 28, 28))
    return {"masks": masks, "source_label": "测试病灶定位"}


class HospitalClosedLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.engine = SequencedEngine()

    def make_client(self):
        settings = Settings(
            api_key=KEY,
            workflow_db=self.root / "workflow.sqlite3",
            hospital_root=self.root / "hospital",
            visual_origin=VISUAL_ORIGIN,
        )
        return TestClient(create_app(settings, engine=self.engine))

    def test_report_explanation_uses_saved_report_and_leaves_report_available_on_failure(self):
        explainer = Mock(label="华为 MaaS · DeepSeek")
        analysis = "筛查提示中度糖网改变，仍需结合专科复核确认。"
        explainer.explain.return_value = {
            "model": "deepseek-v4.1-flash-1017",
            "explanation": analysis,
            "analysis": analysis,
            "analysis_mode": "generated",
        }
        settings = Settings(api_key=KEY, hospital_root=self.root / "local-hospital")
        report = "/v1/his/reports/rep_seed_138001_l0"
        write_sample_image(self.root / "examples/IDRiD_03.jpg")
        write_sample_image(
            self.root / "inference_runtime/local_gpu/retina_public_sample_proxy.jpg", (150, 80, 52),
        )
        with patch("inference.hospital_system.ROOT", self.root):
            app = create_app(settings, engine=self.engine, report_explainer=explainer)
        with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
            doctor = client.get(f"{report}/doctor.html")
            self.assertEqual(doctor.status_code, 200)
            self.assertIn("华为 MaaS · DeepSeek", doctor.text)
            self.assertIn('id="report-explanation-button"', doctor.text)
            self.assertNotIn("report-explanation-button", client.get(f"{report}/patient.html").text)
            generated = client.post(f"{report}/report-explanation")
            self.assertEqual(generated.status_code, 200)
            self.assertEqual(generated.headers["cache-control"], "no-store")
            self.assertEqual(generated.json()["model"], "deepseek-v4.1-flash-1017")
            self.assertEqual(generated.json()["explanation"], analysis)
            self.assertEqual(generated.json()["analysis"], analysis)
            self.assertEqual(generated.json()["analysis_mode"], "generated")
            self.assertEqual(explainer.explain.call_args.args[0]["report_summary"]["dr_severity"]["grade"], 2)
            self.assertEqual(explainer.explain.call_args.args[1]["level"], "L3")

            remote = client.post(f"{report}/report-explanation", headers={"Host": urlsplit(VISUAL_ORIGIN).netloc})
            cross_origin = client.post(f"{report}/report-explanation", headers={"Origin": VISUAL_ORIGIN})
            self.assertEqual(remote.status_code, 404)
            self.assertEqual(cross_origin.status_code, 404)
            explainer.explain.assert_called_once()

            explainer.explain.side_effect = LocalExplanationUnavailable()
            unavailable = client.post(f"{report}/report-explanation")
            self.assertEqual(unavailable.status_code, 503)
            self.assertFalse(unavailable.json()["ok"])
            self.assertEqual(client.get(f"{report}/doctor.html").status_code, 200)

    def test_imaging_room_doctor_diagnosis_and_patient_friendly_loop(self):
        with self.make_client() as client, patch("inference.visual_report._predict_masks", side_effect=_lesion_masks):
            # 1. Imaging Room UI
            room = client.get("/v1/imaging_room")
            self.assertEqual(room.status_code, 200)
            self.assertIn("医院眼科拍片室", room.text)

            portal = client.get("/v1/portal")
            self.assertEqual(portal.status_code, 200)
            self.assertIn("患者端：个人眼健康管家", portal.text)

            # 2. Imaging room auto-creates new patient 13900139009 and uploads left eye baseline (DR 2)
            up1 = client.post(
                "/v1/his/images/upload",
                json={
                    "phone": "13900139009",
                    "name": "王明远",
                    "age": 58,
                    "gender": "男",
                    "chronic_history": "2型糖尿病6年",
                    "eye_history": "首次眼底检查",
                    "eye": "left",
                    "source": "upload",
                    "image_base64": _jpeg_b64((150, 70, 50)),
                    "note": "门诊初查左眼",
                },
            )
            self.assertEqual(up1.status_code, 200)
            self.assertEqual(up1.json()["overview"]["pending_eyes"], ["left"])

            # 3. Doctor agent binds patient via phone query & triggers diagnosis -> doctor PDF report
            self.engine.mode = "dr2"
            d_bind = client.post("/v1/his/agent/doctor/patient", json={"query": "会诊患者 13900139009"}, headers=HEADERS)
            self.assertEqual(d_bind.status_code, 200)
            self.assertIn("王明远", d_bind.json()["message"])

            d_diag = client.post("/v1/his/agent/doctor/diagnose", json={"query": "请辅助诊断一下病人左眼情况"}, headers=HEADERS)
            self.assertEqual(d_diag.status_code, 200)
            self.assertTrue(d_diag.json()["has_new_image"])
            doc_url = d_diag.json()["doctor_report_url"]
            self.assertIn("/v1/his/reports/", doc_url)
            doc_path = urlsplit(doc_url).path
            doc_html = client.get(doc_path)
            self.assertEqual(doc_html.status_code, 200)
            self.assertIn("眼科临床辅助诊断与病灶分析报告", doc_html.text)
            self.assertIn("王明远", doc_html.text)
            body = doc_html.text.split("</style>", 1)[-1]
            narrative, _, appendix = body.partition('id="tech-details"')
            self.assertIn("中度非增殖期糖尿病视网膜病变", narrative)
            for level in ("L3", "L2"):
                self.assertIn(f'data-level="{level}"', narrative)
            self.assertIn("就诊前可以准备", narrative)
            self.assertIn("五标签模型", appendix)
            self.assertIn("85.000分", appendix)
            self.assertRegex(doc_html.text, r"RET-\d{8}-[A-Z0-9]+")
            self.assertNotRegex(doc_html.text, r"\b(?:RULE|SRC)_[A-Z0-9_]+")
            for name in ("original.jpg", "overlay.jpg"):
                self.assertIn(f'src="{name}"', doc_html.text)
                media = client.get(doc_path.rsplit("/", 1)[0] + f"/{name}")
                self.assertEqual(media.status_code, 200)
                self.assertEqual(media.headers["content-type"], "image/jpeg")

            d_rec = client.post("/v1/his/agent/doctor/records", json={"query": "调出左眼报告"}, headers=HEADERS)
            self.assertEqual(d_rec.status_code, 200)
            self.assertEqual(len(d_rec.json()["reports"]), 1)

            # 4. Patient logs in -> checks left eye (no new image -> returns existing report calmly)
            p_sess = client.post("/v1/his/agent/patient/session", json={"query": "您好，我的手机号是13900139009"}, headers=HEADERS)
            self.assertEqual(p_sess.status_code, 200)
            self.assertTrue(p_sess.json()["bound"])

            p_rep1 = client.post("/v1/his/agent/patient/report", json={"query": "帮我看看左眼最新报告"}, headers=HEADERS)
            self.assertEqual(p_rep1.status_code, 200)
            self.assertFalse(p_rep1.json()["has_new_image"])
            self.assertIn("医院暂时还没有上传新的左眼照片", p_rep1.json()["reply"])

            # 5. Imaging room uploads an improved (DR0) left eye photo -> patient automatically detects new photo & improvement!
            up2 = client.post(
                "/v1/his/images/upload",
                json={
                    "phone": "13900139009",
                    "eye": "left",
                    "source": "upload",
                    "image_base64": _jpeg_b64((90, 50, 35)),
                    "note": "复查左眼新拍片",
                },
            )
            self.assertEqual(up2.status_code, 200)

            self.engine.mode = "normal"
            p_rep2 = client.post("/v1/his/agent/patient/report", json={"query": "帮我看下左眼新检查吧"}, headers=HEADERS)
            self.assertEqual(p_rep2.status_code, 200)
            p_data2 = p_rep2.json()
            self.assertTrue(p_data2["has_new_image"])
            self.assertEqual(p_data2["comparison"]["trend"], "improved")
            self.assertIn("往好的方向发展", p_data2["reply"])
            self.assertIn("没有发现明显的糖尿病视网膜病变", p_data2["reply"])
            self.assertNotIn("0~100", p_data2["reply"])

            pat_url = p_data2["patient_report_url"]
            pat_html = client.get(urlsplit(pat_url).path)
            self.assertEqual(pat_html.status_code, 200)
            self.assertIn("王明远先生/女士的左眼眼健康解读单", pat_html.text)

            latest_doc = client.post(
                "/v1/his/portal/chat",
                json={"role": "doctor", "phone": "13900139009", "eye": "left",
                      "action": "report_or_diagnose", "query": "查看左眼最新结果"},
            )
            self.assertEqual(latest_doc.status_code, 200)
            self.assertFalse(latest_doc.json()["has_new_image"])
            self.assertIsNone(latest_doc.json()["clinical_decision"]["level"])
            latest_doc_html = client.get(urlsplit(latest_doc.json()["doctor_report_url"]).path)
            self.assertEqual(latest_doc_html.status_code, 200)
            latest_body = latest_doc_html.text.split("</style>", 1)[-1]
            self.assertIn('data-level="L0"', latest_body)
            self.assertNotIn('data-level="L1"', latest_body)
            self.assertIn("暂无法分级", latest_body)
            self.assertIn("糖网分级由 2 级降至 0 级", latest_body)

            # 6. Patient reports sudden warning symptoms -> personalized urgent guidance referencing chronic history
            p_sym = client.post(
                "/v1/his/agent/patient/symptom",
                json={"query": "我今早起床突然眼前有很多黑影飘还有闪光感"},
                headers=HEADERS,
            )
            self.assertEqual(p_sym.status_code, 200)
            self.assertTrue(p_sym.json()["urgent_detected"])
            self.assertIn("2型糖尿病6年", p_sym.json()["patient"]["chronic_history"])

            # 7. Portal interactive chat endpoint works seamlessly for education & symptoms
            chat_edu = client.post(
                "/v1/his/portal/chat",
                json={"role": "patient", "phone": "13900139009", "action": "education", "query": "青光眼怎么发生的"},
            )
            self.assertEqual(chat_edu.status_code, 200)
            self.assertIn("眼压", chat_edu.json()["reply"])

    def test_rejected_image_has_l0_with_original_and_no_lesion_overlay(self):
        self.engine.mode = "reject"
        with self.make_client() as client, patch("inference.visual_report._predict_masks") as localize:
            upload = client.post(
                "/v1/his/images/upload",
                json={"phone": "13900139009", "name": "王明远", "age": 58, "gender": "男",
                      "eye": "left", "source": "upload", "image_base64": _jpeg_b64()},
            )
            self.assertEqual(upload.status_code, 200)
            diagnosis = client.post(
                "/v1/his/portal/chat",
                json={"role": "doctor", "phone": "13900139009", "eye": "left",
                      "action": "report_or_diagnose", "query": "辅助诊断左眼"},
            )
            self.assertEqual(diagnosis.status_code, 200)
            localize.assert_not_called()

            doc_path = urlsplit(diagnosis.json()["doctor_report_url"]).path
            report = client.get(doc_path)
            self.assertEqual(report.status_code, 200)
            body = report.text.split("</style>", 1)[-1]
            self.assertIn('data-level="L0"', body)
            self.assertIn("暂无法分级", body)
            self.assertIn('src="original.jpg"', body)
            self.assertNotIn('src="overlay.jpg"', body)
            self.assertEqual(client.get(doc_path.rsplit("/", 1)[0] + "/overlay.jpg").status_code, 404)

    def test_root_launcher_parser(self):
        import run as root_run

        parser = root_run.build_parser()
        self.assertIsNone(parser.parse_args([]).command)
        self.assertEqual(parser.parse_args(["local"]).command, "local")
        self.assertEqual(parser.parse_args(["test"]).command, "test")
        self.assertEqual(parser.parse_args(["export"]).command, "export")


if __name__ == "__main__":
    unittest.main()
