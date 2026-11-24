"""End-to-end tests for Hospital Imaging Room, Phone-keyed PACS, and Patient/Doctor Agents."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from inference.app import create_app
from inference.settings import Settings
from tests.test_clinical_workflow import SequencedEngine

KEY = "hospital-loop-" + "k" * 32
HEADERS = {"X-API-Key": KEY}


def _jpeg_b64(color=(120, 60, 40)) -> str:
    buf = BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


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
            visual_origin="https://severj.top",
        )
        return TestClient(create_app(settings, engine=self.engine))

    def test_imaging_room_doctor_diagnosis_and_patient_friendly_loop(self):
        with self.make_client() as client:
            # 1. Imaging Room UI & pre-seeded patient 13800138001
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
            doc_html = client.get("/" + doc_url.split("https://severj.top/", 1)[1])
            self.assertEqual(doc_html.status_code, 200)
            self.assertIn("眼科临床辅助诊断与病灶分析报告", doc_html.text)
            self.assertIn("王明远", doc_html.text)

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
            pat_html = client.get("/" + pat_url.split("https://severj.top/", 1)[1])
            self.assertEqual(pat_html.status_code, 200)
            self.assertIn("王明远先生/女士的左眼眼健康解读单", pat_html.text)

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


if __name__ == "__main__":
    unittest.main()
