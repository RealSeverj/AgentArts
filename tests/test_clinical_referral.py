"""Regression coverage for clinical screening priorities and incomplete evidence."""

import unittest

from inference.clinical_review import REFERENCE_SHA
from inference.clinical_workflow import CareEvaluateRequest, ClinicalWorkflowStore, IntakeRequest, evaluate_care_rules
from inference.report_summary import RESEARCH_LINES, build_report_summary


def make_screening(grade=0, *, quality="Good", hits=(), missing=()):
    results = {
        "quality": {"status": "completed", "output": {"predicted_label": quality}},
        "dr_grading": {"status": "completed", "output": {"predicted_grade": grade}},
    }
    research = {}
    for group, (model_id, fingerprint, thresholds) in RESEARCH_LINES.items():
        scores = {
            code: threshold if (group, code) in hits else 0.0
            for code, threshold in thresholds.items()
            if (group, code) not in missing
        }
        target = results if group == "multidisease" else research
        target[group] = {
            "status": "completed",
            "model_id": model_id,
            "weights_sha256": fingerprint,
            "output": {"scores": scores},
        }
    return {
        "status": "ok" if quality != "Reject" else "quality_rejected",
        "image_sha256": "a" * 64,
        "results": results,
        "research_results": research,
        "report_summary": build_report_summary(results, research),
    }


class ClinicalReferralTests(unittest.TestCase):
    def evaluate(self, screening, *, symptoms=()):
        return evaluate_care_rules(screening, {"eye": "left", "symptoms": list(symptoms)})

    def test_dr_model_grades_map_to_referral_levels(self):
        expected = (
            ("L1", "annual_screening", r"12\s*个月"),
            ("L2", "routine_followup", r"6\s*[–—~～至-]\s*12\s*个月"),
            ("L3", "scheduled_ophthalmology", r"2\s*周\s*[–—~～至-]\s*1\s*个?月"),
            ("L4", "prompt_ophthalmology", r"1\s*[–—~～至-]\s*2\s*周"),
            ("L4", "prompt_ophthalmology", r"1\s*[–—~～至-]\s*2\s*周"),
        )
        for grade, (level, priority, timeframe) in enumerate(expected):
            with self.subTest(grade=grade):
                screening = make_screening(grade, quality="Usable", hits=(("multidisease", "DR"),) if grade else ())
                care = self.evaluate(screening)
                decision = care["clinical_decision"]
                self.assertEqual(care["care_priority"], priority)
                self.assertEqual(decision["level"], level)
                self.assertRegex(decision["timeframe"], timeframe)
                self.assertTrue(decision["title"])
                self.assertEqual(screening["report_summary"]["dr_severity"]["grade"], grade)

    def test_every_specific_research_signal_can_trigger_a_referral(self):
        for group, code, level, name, exam in (
            ("multilabel_11", "ODE", "L4", "视盘水肿", "OCT"),
            ("multilabel_11", "RVO", "L4", "视网膜静脉阻塞", "OCT"),
            ("glaucoma", "glaucoma", "L3", "青光眼", "眼压"),
            ("multidisease", "OPTIC_DISC_CUPPING", "L3", "视盘凹陷", "眼压"),
            ("multidisease", "AMD", "L3", "黄斑变性", "OCT"),
            ("multilabel_11", "CSR", "L3", "中心性浆液性", "OCT"),
            ("multilabel_11", "ODP", "L3", "视盘苍白", "眼压"),
            ("multidisease", "MYOPIC_FUNDUS", "L3", "近视性眼底改变", "OCT"),
            ("multilabel_11", "DN", "L2", "玻璃膜疣", "Amsler"),
            ("multilabel_11", "TSLN", "L2", "豹纹状眼底", "眼底彩照"),
            ("multilabel_11", "LS", "L2", "激光瘢痕", "眼底彩照"),
        ):
            with self.subTest(code=code):
                decision = self.evaluate(make_screening(hits=((group, code),)))["clinical_decision"]
                self.assertEqual(decision["level"], level)
                trigger = next(item for item in decision["trigger_items"] if name in item["name"])
                self.assertEqual(trigger["level"], level)
                self.assertTrue(trigger["evidence"])
                self.assertTrue(trigger["basis"])
                self.assertTrue(trigger["destination"])
                self.assertIn(exam, str(trigger["tests"]))

    def test_multiple_signals_use_the_highest_level_and_shortest_window(self):
        decision = self.evaluate(make_screening(1, hits=(
            ("multilabel_11", "LS"), ("multilabel_11", "TSLN"), ("multilabel_11", "DN"),
        )))["clinical_decision"]
        self.assertEqual(decision["level"], "L2")
        self.assertRegex(decision["timeframe"], r"3\s*[–—~～至-]\s*6\s*个月")
        self.assertIn("激光瘢痕", decision["trigger_items"][0]["name"])
        self.assertIn("12", next(item for item in decision["trigger_items"] if "豹纹" in item["name"])["timeframe"])

        decision = self.evaluate(make_screening(4, hits=(
            ("multilabel_11", "RVO"), ("multilabel_11", "ODE"), ("multilabel_11", "DN"),
        )))["clinical_decision"]
        self.assertEqual(decision["level"], "L4")
        self.assertIn("视盘水肿", decision["trigger_items"][0]["name"])
        ode = next(item for item in decision["trigger_items"] if "视盘水肿" in item["name"])
        self.assertEqual(decision["timeframe"], ode["timeframe"])
        self.assertRegex(ode["timeframe"], r"24\s*小时内")
        self.assertTrue(any("视网膜静脉阻塞" in item["name"] for item in decision["trigger_items"]))
        self.assertTrue(any(item["level"] == "L2" for item in decision["trigger_items"]))

    def test_missing_or_nonspecific_results_do_not_count_as_a_negative_screen(self):
        for screening in (
            make_screening(None),
            make_screening(missing=(("multilabel_11", "RVO"),)),
            make_screening(hits=(("multidisease", "ABNORMAL"),)),
            make_screening(hits=(("multidisease", "DR"),)),
        ):
            with self.subTest(screening=screening["report_summary"]["dr_severity"]):
                care = self.evaluate(screening)
                self.assertIsNone(care["clinical_decision"]["level"])
                self.assertNotEqual(care["care_priority"], "annual_screening")

        failed_with_stale_score = make_screening(hits=(("multilabel_11", "ODE"),))
        failed_with_stale_score["report_summary"]["research"]["multilabel_11"]["status"] = "error"
        care = self.evaluate(failed_with_stale_score)
        self.assertIsNone(care["clinical_decision"]["level"])

    def test_severe_review_alert_can_raise_priority_without_changing_the_dr_grade(self):
        screening = make_screening(2)
        screening["report_summary"]["dr_severe_review_alert"] = {
            "triggered": True,
            "basis": "已触发边界病例人工复核规则。",
        }
        decision = self.evaluate(screening)["clinical_decision"]
        self.assertEqual(decision["level"], "L4")
        self.assertTrue(any("升阶复核" in item["name"] for item in decision["trigger_items"]))
        self.assertEqual(screening["report_summary"]["dr_severity"]["grade"], 2)

        screening["report_summary"]["dr_severe_review_alert"]["triggered"] = False
        self.assertEqual(self.evaluate(screening)["clinical_decision"]["level"], "L3")

    def test_public_grade_disagreement_requires_same_image(self):
        screening = make_screening(2)
        screening["visual_report"] = {"reference_grade": 3}
        self.assertEqual(self.evaluate(screening)["clinical_decision"]["level"], "L3")

        screening["image_sha256"] = REFERENCE_SHA
        decision = self.evaluate(screening)["clinical_decision"]
        self.assertEqual(decision["level"], "L4")
        self.assertEqual(screening["report_summary"]["dr_severity"]["grade"], 2)

    def test_emergency_symptoms_override_even_rejected_images(self):
        screening = make_screening(grade=3, quality="Reject", hits=(("multilabel_11", "ODE"),))
        care = self.evaluate(screening, symptoms=("无突发视力骤降",))
        self.assertEqual(care["care_priority"], "retake_image")
        self.assertIsNone(care["clinical_decision"]["level"])
        care = self.evaluate(screening, symptoms=("今天突发视力骤降",))
        self.assertEqual(care["care_priority"], "urgent_ophthalmology")
        self.assertEqual(care["clinical_decision"]["level"], "L4")
        self.assertIn("立即", care["clinical_decision"]["timeframe"])
        self.assertFalse(any("视盘水肿" in item["name"] for item in care["clinical_decision"]["trigger_items"]))

    def test_new_intake_updates_the_persisted_decision(self):
        store = ClinicalWorkflowStore()
        self.addCleanup(store.close)
        screening = make_screening(0)
        screening["request_id"] = "assessment-referral-test"
        assessment_id = store.record_screening(screening, eye="left")
        request = CareEvaluateRequest(assessment_id=assessment_id)
        saved = store.evaluate_care(request)
        self.assertEqual(saved["clinical_decision"]["level"], "L1")
        self.assertEqual(store.get_context()["clinical_decision"]["level"], "L1")

        intake = store.record_intake(IntakeRequest(assessment_id=assessment_id, symptoms=["突然失明"]))
        self.assertEqual(intake["care_preview"]["clinical_decision"]["status"], "emergency")
        evidence = store.get_assessment(assessment_id)["evidence_packet"]
        self.assertEqual(evidence["clinical_decision"]["level"], "L4")
        self.assertEqual(evidence["dr_grading"]["grade"], 0)
        self.assertEqual(store.evaluate_care(request)["clinical_decision"], evidence["clinical_decision"])


if __name__ == "__main__":
    unittest.main()
