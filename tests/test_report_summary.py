"""The grade belongs to the DR expert even when the other classifiers disagree."""
import math
import unittest
from decimal import Decimal

from inference.report_summary import ELEVEN_LABELS, RESEARCH_LINES, build_report_summary


def completed(**output):
    return {"status": "completed", "output": output}


class ReportSummaryTests(unittest.TestCase):
    def screened(self, stage_name, scores, **metadata):
        model_id, fingerprint, _ = RESEARCH_LINES[stage_name]
        stage = dict(completed(scores=scores), model_id=model_id, weights_sha256=fingerprint)
        stage.update(metadata)
        results = {"quality": completed(predicted_label="Good")}
        if stage_name == "multidisease":
            results[stage_name] = stage
            report = build_report_summary(results)["five_label_screening"]
        else:
            report = build_report_summary(results, {stage_name: stage})["research"][stage_name]
        return {item["code"]: item for item in report["items"]}

    def test_every_research_line_is_inclusive_and_its_display_preserves_comparison(self):
        for stage_name, (_, _, lines) in RESEARCH_LINES.items():
            for code, line in lines.items():
                for score in (math.nextafter(line, 0), line, math.nextafter(line, 1)):
                    with self.subTest(stage=stage_name, code=code, score=score):
                        item = self.screened(stage_name, {code: score})[code]
                        self.assertEqual(item["research_threshold"], line)
                        self.assertEqual(item["research_threshold_met"], score >= line)
                        self.assertEqual("达到研究筛查线" if score >= line else "未达到研究筛查线",
                                         item["research_threshold_status"])
                        displayed_score = Decimal(item["score_display"][:-1])
                        displayed_line = Decimal(item["research_threshold_display"][:-1])
                        self.assertEqual(displayed_score.compare(displayed_line),
                                         (score > line) - (score < line))

    def test_shared_amd_label_compares_to_its_own_model(self):
        score = {"AMD": 0.006}
        five = self.screened("multidisease", score)["AMD"]
        eleven = self.screened("multilabel_11", score)["AMD"]
        self.assertEqual(five["research_threshold"], 0.005200723651796579)
        self.assertEqual(eleven["research_threshold"], 0.006723892467562109)
        self.assertEqual("0.520072分", five["research_threshold_display"])
        self.assertEqual("0.672389分", eleven["research_threshold_display"])
        self.assertTrue(five["research_threshold_met"])
        self.assertFalse(eleven["research_threshold_met"])
        self.assertEqual("暂无法分级", five["grading"])
        self.assertEqual("暂无法分级", eleven["grading"])

    def test_model_identity_is_required_and_missing_scores_stay_unknown(self):
        for metadata in ({"model_id": None}, {"weights_sha256": None},
                         {"weights_sha256": "different"}, {"model_id": "different"}):
            with self.subTest(metadata=metadata):
                item = self.screened("glaucoma", {"glaucoma": 0.92}, **metadata)["glaucoma"]
                self.assertEqual(0.92, item["score"])
                self.assertIsNone(item["research_threshold"])
                self.assertIsNone(item["research_threshold_display"])
                self.assertIsNone(item["research_threshold_met"])
                self.assertEqual("无法比较", item["research_threshold_status"])
        for raw in (None, False, float("nan"), float("inf"), 1.00001):
            with self.subTest(raw=raw):
                item = self.screened("glaucoma", {"glaucoma": raw})["glaucoma"]
                self.assertIsNone(item["score"])
                self.assertIsNone(item["score_display"])
                self.assertEqual(0.14928317070007324, item["research_threshold"])
                self.assertEqual("14.928317分", item["research_threshold_display"])
                self.assertIsNone(item["research_threshold_met"])
                self.assertEqual("无法比较", item["research_threshold_status"])

    def test_stage_and_score_follow_the_specialist_grade(self):
        results = {"quality": completed(predicted_label="Usable"),
                   "multidisease": completed(scores={"DR": 0.9999999})}
        for grade in range(5):
            with self.subTest(grade=grade):
                results["dr_grading"] = completed(predicted_grade=grade, grade_order=[4, 3, 2, 1, 0],
                                                  scores=[0.4, 0.3, 0.2, 0.1, 0.0])
                report = build_report_summary(results)
                self.assertEqual(grade, report["dr_severity"]["grade"])
                self.assertEqual(grade / 10, report["dr_severity"]["score"])
                dr_signal = report["five_label_screening"]["items"][0]
                self.assertIsNone(dr_signal["severity_grade"])
                self.assertEqual(">99.999分", dr_signal["score_display"])
        self.assertEqual("严重（增殖期，最高档）", report["dr_severity"]["level"])

    def test_research_items_keep_their_semantics_without_a_grade(self):
        research = {"multilabel_11": completed(scores=dict.fromkeys(ELEVEN_LABELS, 1e-10)),
                    "glaucoma": completed(scores={"glaucoma": 1.0})}
        report = build_report_summary({"quality": completed(predicted_label="Good")}, research)
        items = report["research"]["multilabel_11"]["items"]
        self.assertEqual(list(ELEVEN_LABELS), [item["code"] for item in items])
        self.assertEqual("治疗痕迹", next(item["kind"] for item in items if item["code"] == "LS"))
        self.assertTrue(all(item["severity_grade"] is None for item in items))
        self.assertTrue(all(item["score_display"] == "<0.001分" for item in items))
        glaucoma = report["research"]["glaucoma"]["items"][0]
        self.assertEqual("100.000分", glaucoma["score_display"])
        self.assertIsNone(glaucoma["severity_grade"])
        self.assertIsNone(report["dr_severity"]["grade"])

    def test_unreadable_image_blocks_results_and_missing_is_not_zero(self):
        results = {"quality": completed(predicted_label="Reject"),
                   "dr_grading": completed(predicted_grade=0)}
        research = {"glaucoma": completed(scores={"glaucoma": 0.0})}
        for quality in (results["quality"], {"status": "error"}, {}):
            results["quality"] = quality
            report = build_report_summary(results, research)
            self.assertIsNone(report["dr_severity"]["grade"])
            self.assertEqual("not_run", report["research"]["glaucoma"]["status"])
            self.assertFalse(report["research"]["glaucoma"]["items"])
        results["quality"] = completed(predicted_label="Good")
        for grade in (None, False, -1, 5, "0"):
            results["dr_grading"] = completed(predicted_grade=grade)
            report = build_report_summary(results)
            self.assertIsNone(report["dr_severity"]["grade"])
            self.assertEqual("ungradable", report["dr_severity"]["status"])
            self.assertEqual("not_configured", report["research"]["glaucoma"]["reason"])
