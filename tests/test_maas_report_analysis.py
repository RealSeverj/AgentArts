"""Regression checks for grounded MaaS report analysis without cloud access."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from inference.local_llm import MAAS_HOST, MAAS_MODEL, MAAS_PATH, LocalExplanationUnavailable, MaaSReportExplainer


GENERATED_ANALYSIS = (
    "当前筛查提示青光眼相关风险，可将视盘评估作为复核重点。"
    "现有筛查照片提供的证据有限，仍需综合其他专科结果判断。"
)


def _completion(content: dict) -> dict:
    return {
        "model": MAAS_MODEL,
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content, ensure_ascii=False)}}],
    }


def _report_user_data(payload: dict) -> dict:
    return json.loads(next(message["content"] for message in payload["messages"] if message["role"] == "user"))


class MaaSReportAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.explainer = MaaSReportExplainer(api_key="stub-maas-secret")
        self.screening = {
            "patient": {"name": "PRIVATE-PATIENT-NAME", "phone": "PRIVATE-PATIENT-PHONE"},
            "assessment_id": "PRIVATE-CASE-ID",
            "results": {
                "quality": {"status": "completed", "output": {"predicted_label": "Good"}},
                "raw_image": "PRIVATE-IMAGE-BLOB",
            },
            "report_summary": {
                "dr_severity": {"status": "completed", "grade": 2},
                "symptoms": "PRIVATE-SYMPTOM-TEXT",
            },
            "visual_report": {
                "localization_status": "localized",
                "candidate_status": "candidates_present",
                "original_url": "https://example.invalid/PRIVATE-IMAGE-URL",
            },
        }
        self.decision = {
            "level": "L3",
            "trigger_items": [
                {"code": "glaucoma", "reason": "PRIVATE-TRIAGE-NOTE"},
                {"code": "PRIVATE-UNKNOWN-TRIGGER"},
            ],
        }

    def test_returns_deepseek_prose_and_sends_only_projected_evidence(self):
        completion = _completion({"analysis": GENERATED_ANALYSIS, "evidence_ids": ["C"]})
        with patch("inference.local_llm._http_json", return_value=(200, completion)) as transport:
            result = self.explainer.explain(self.screening, self.decision)

        self.assertEqual(result["model"], MAAS_MODEL)
        self.assertEqual(result["analysis_mode"], "generated")
        self.assertEqual(result["analysis"], GENERATED_ANALYSIS)
        self.assertEqual(result["explanation"], GENERATED_ANALYSIS)

        transport.assert_called_once()
        connection, method, path = transport.call_args.args
        self.assertEqual(connection.host, MAAS_HOST)
        self.assertEqual((method, path), ("POST", MAAS_PATH))
        payload = transport.call_args.kwargs["payload"]
        self.assertEqual(payload["model"], MAAS_MODEL)
        self.assertFalse(payload["stream"])
        self.assertNotIn("PRIVATE-", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("stub-maas-secret", json.dumps(payload, ensure_ascii=False))
        user_data = _report_user_data(payload)
        self.assertIn("L3", json.dumps(user_data["分诊状态"], ensure_ascii=False))
        evidence = user_data["报告证据"]
        self.assertIn("2 级", evidence["B"]["发现"])
        self.assertIn("青光眼相关风险", evidence["C"]["发现"])

    def test_retries_invalid_analysis_once_with_focused_report_evidence(self):
        rejected = _completion({"analysis": "UNTRUSTED-PROVIDER-TEXT", "evidence_ids": ["Z"]})
        valid = _completion({"analysis": GENERATED_ANALYSIS, "evidence_ids": ["C"]})
        with patch("inference.local_llm._http_json", side_effect=[(200, rejected), (200, valid)]) as transport:
            result = self.explainer.explain(self.screening, self.decision)

        self.assertEqual(result["analysis"], GENERATED_ANALYSIS)
        self.assertEqual(result["analysis_mode"], "generated")
        self.assertEqual(transport.call_count, 2)
        initial, retry = [call.kwargs["payload"] for call in transport.call_args_list]
        initial_evidence = _report_user_data(initial)["报告证据"]
        retry_data = _report_user_data(retry)
        retry_evidence = retry_data["报告证据"]
        self.assertEqual(retry_data["首要复核线索"], "C")
        self.assertLess(len(retry_evidence), len(initial_evidence))
        self.assertEqual(set(retry_evidence), {"A", "B", "C"})
        self.assertTrue(all(set(item) == {"发现"} for item in retry_evidence.values()))
        self.assertNotIn("UNTRUSTED-PROVIDER-TEXT", json.dumps(retry, ensure_ascii=False))
        self.assertNotIn("PRIVATE-", json.dumps(retry, ensure_ascii=False))

    def test_returns_an_error_after_one_unsuccessful_retry(self):
        invalid = _completion({"ids": ["A"]})
        with patch("inference.local_llm._http_json", side_effect=[(200, invalid), (200, invalid)]) as transport:
            with self.assertRaises(LocalExplanationUnavailable):
                self.explainer.explain(self.screening, self.decision)

        self.assertEqual(transport.call_count, 2)

    def test_does_not_retry_transport_http_or_model_identity_failures(self):
        different_model = _completion({"analysis": GENERATED_ANALYSIS, "evidence_ids": ["C"]})
        different_model["model"] = "unexpected-provider-model"
        failures = {
            "unauthorized": (401, {}),
            "forbidden": (403, {}),
            "provider error": (500, {}),
            "network timeout": TimeoutError(),
            "wrong model": (200, different_model),
        }
        for case, failure in failures.items():
            with self.subTest(case=case), patch("inference.local_llm._http_json", side_effect=[failure]) as transport:
                with self.assertRaises(LocalExplanationUnavailable):
                    self.explainer.explain(self.screening, self.decision)
                transport.assert_called_once()

    def test_rejects_selector_only_output_invalid_citations_and_invalid_clinical_claims(self):
        rejected = {
            "old selector": {"ids": ["A"]},
            "missing evidence": {"analysis": GENERATED_ANALYSIS, "evidence_ids": []},
            "unknown evidence": {"analysis": GENERATED_ANALYSIS, "evidence_ids": ["Z"]},
            "duplicate evidence": {"analysis": GENERATED_ANALYSIS, "evidence_ids": ["C", "C"]},
            "unrelated evidence": {"analysis": GENERATED_ANALYSIS, "evidence_ids": ["A"]},
            "no limitation": {"analysis": "筛查提示青光眼相关风险，应将视盘评估作为复核重点。", "evidence_ids": ["C"]},
            "unsupported grade": {"analysis": GENERATED_ANALYSIS + "可改判为糖网 4 级。", "evidence_ids": ["B", "C"]},
            "invented triage": {"analysis": GENERATED_ANALYSIS + "请立即前往急诊。", "evidence_ids": ["C"]},
            "treatment directive": {"analysis": GENERATED_ANALYSIS + "建议立即手术治疗。", "evidence_ids": ["C"]},
            "contradicts positive finding": {
                "analysis": "当前没有青光眼风险。仅凭筛查照片尚难确认，仍需结合结果判断。",
                "evidence_ids": ["C"],
            },
            "mentions a condition outside the report": {
                "analysis": GENERATED_ANALYSIS.replace("综合其他专科结果", "核对玻璃膜疣"),
                "evidence_ids": ["C"],
            },
        }
        for case, content in rejected.items():
            with self.subTest(case=case), patch("inference.local_llm._http_json", return_value=(200, _completion(content))):
                with self.assertRaises(LocalExplanationUnavailable):
                    self.explainer.explain(self.screening, self.decision)

    def test_cites_all_submitted_report_evidence(self):
        decision = {
            "level": "L4",
            "trigger_items": [{"code": code} for code in (
                "DR_SEVERE_REVIEW", "glaucoma", "OPTIC_DISC_CUPPING", "ODP", "LS", "DN",
            )],
        }
        analysis = (
            "糖网初步分期与低估风险同时提示了复核分期的必要性。"
            "现有筛查影像不能明确严重程度，判断仍需核对原始影像。"
        )
        completion = _completion({"analysis": analysis, "evidence_ids": list("ABCDEFGH")})
        with patch("inference.local_llm._http_json", return_value=(200, completion)) as transport:
            result = self.explainer.explain(self.screening, decision)

        self.assertEqual(result["analysis"], analysis)
        request = transport.call_args.kwargs["payload"]
        evidence = json.loads(request["messages"][1]["content"])["报告证据"]
        self.assertIn("低估", evidence["C"]["复核依据"])
        self.assertNotIn("新生血管", evidence["C"]["复核依据"])

        speculative = _completion({
            "analysis": "糖网初步分期存在低估风险，目前有新生血管的证据。筛查图像仍需结合原图核对。",
            "evidence_ids": ["C"],
        })
        with patch("inference.local_llm._http_json", return_value=(200, speculative)):
            with self.assertRaises(LocalExplanationUnavailable):
                self.explainer.explain(self.screening, decision)

    def test_can_mention_additional_report_findings_while_citing_the_review_focus(self):
        decision = {
            "level": "L4",
            "trigger_items": [{"code": "DR_SEVERE_REVIEW"}, {"code": "glaucoma"}],
        }
        analysis = (
            "现有筛查提示糖网分期存在低估风险，复核时应核对原图。"
            "眼底照片所提供的信息有限，尚不足以对青光眼相关风险定论。"
        )
        completion = _completion({"analysis": analysis, "evidence_ids": ["A", "B", "C"]})
        with patch("inference.local_llm._http_json", return_value=(200, completion)):
            self.assertEqual(self.explainer.explain(self.screening, decision)["analysis"], analysis)


if __name__ == "__main__":
    unittest.main()
