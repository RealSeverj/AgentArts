import base64
import hashlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from inference.clinical_referral import build_clinical_decision
from inference.clinical_review import build_clinical_review
from inference.report_levels import display_report_level
from inference.visual_report import LESIONS, REFERENCE_SHA, _comparison, _decision_section_html, _reference_masks, _report_html, write_visual_report


class VisualReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.image = Image.new("RGB", (180, 100), (80, 50, 40))
        self.response = {"results": {"quality": {"status": "completed", "output": {"predicted_label": "Good"}}}}
        patcher = patch("inference.visual_report._predict_masks", return_value=None)
        self.pred = patcher.start()
        self.addCleanup(patcher.stop)

    def render(self, request="check", **kwargs):
        return write_visual_report(self.image, self.response, "a"*64, self.root, request, **kwargs)

    def test_quality_gate_avoids_processing_and_artifacts(self):
        self.response["results"]["quality"]["output"]["predicted_label"] = "Reject"
        self.assertIsNone(self.render()["report_url"])
        self.pred.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_no_spatial_result_cannot_become_a_lesion(self):
        group = {"status": "completed", "items": [{"name": "年龄相关性黄斑变性", "research_threshold_met": True}]}
        self.response["report_summary"] = {"five_label_screening": group, "research": {"multilabel_11": {
            "status": "completed", "items": [{"name": "年龄相关性黄斑变性", "research_threshold_met": False}]}}}
        out = self.render(url_prefix="https://example.org/v1/view/cap123")
        self.assertEqual(out["report_url"], "https://example.org/v1/view/cap123/report.html")
        self.assertEqual(out["localization_status"], "unavailable")
        self.assertIsNone(out["overlay_url"])
        html = (self.root / "check/report.html").read_text(encoding="utf-8")
        self.assertIn('src="original.jpg"', html)
        self.assertIn("提示不一致", html)
        self.assertNotIn("cap123", html)
        self.assertFalse((self.root / "check/overlay.jpg").exists())

    def masks(self):
        return {k: Image.new("L", self.image.size) for k in LESIONS}

    def test_empty_prediction_has_distinct_status(self):
        self.pred.return_value = {"masks": self.masks(), "source_label": "自动模型"}
        out = self.render()
        self.assertEqual(out["candidate_status"], "no_candidates")
        self.assertEqual(out["legend"], [])
        self.assertEqual((self.root / "check/original.jpg").read_bytes(), (self.root / "check/overlay.jpg").read_bytes())

    def test_masks_are_drawn_without_flipping_or_stretching_source(self):
        masks = self.masks()
        ImageDraw.Draw(masks["HE"]).rectangle((28, 10, 42, 30), fill=255)
        self.pred.return_value = {"masks": masks, "source_label": "自动模型"}
        out = self.render()
        self.assertEqual(out["localization_status"], "localized")
        self.assertEqual([x["code"] for x in out["legend"]], ["HE"])
        with Image.open(self.root / "check/overlay.jpg") as view:
            self.assertEqual(view.size, self.image.size)
            self.assertGreater(view.getpixel((36, 19))[0], 100)
            self.assertLess(view.getpixel((150, 19))[0], 90)
        masks["MA"] = Image.new("L", (200, 100))
        self.assertEqual(self.render("other")["candidate_status"], "unavailable")

    def test_real_structure_masks_are_attributed_only_to_anatomy(self):
        result = {"status": "completed", "output": {"source_size_wh": list(self.image.size), "size_hw": [512, 512],
                  "coordinate_space": "whole_image_resized_to_model_img_size", "mapping_to_source": {"kind": "stretch"},
                  "mask_artifacts": {}}}
        self.response["results"]["disc_cup_segmentation"] = result
        for code, radius in (("optic_disc", 50), ("optic_cup", 24)):
            mask = Image.new("L", (512, 512))
            ImageDraw.Draw(mask).ellipse((120-radius, 100-radius, 120+radius, 100+radius), fill=255)
            buf = io.BytesIO()
            mask.save(buf, format="PNG")
            payload = buf.getvalue()
            result["output"]["mask_artifacts"][code] = {
                "encoding": "base64", "media_type": "image/png", "sha256": hashlib.sha256(payload).hexdigest(),
                "data": base64.b64encode(payload).decode()}
        out = self.render()
        self.assertEqual(out["localization_status"], "anatomical_only")
        self.assertIsNone(out["reference_grade"])
        result["output"]["mask_artifacts"]["optic_disc"]["sha256"] = "b" * 64
        self.assertIsNone(self.render("tampered")["overlay_url"])

    def test_reference_only_matches_exact_known_image_and_dimensions(self):
        size = (4288, 2848)
        encoded_mask = self.root / "masks.png"
        with Image.new("L", size) as mask:
            mask.putpixel((0, 0), 1 | 2)
            mask.putpixel((1, 0), 4)
            mask.putpixel((2, 0), 8)
            mask.save(encoded_mask)
        manifest = {
            "image_sha256": REFERENCE_SHA,
            "mask_sha256": hashlib.sha256(encoded_mask.read_bytes()).hexdigest(),
            "dr_grade": 3,
            "source_image_id": "IDRiD_03",
            "grading_image_id": "IDRiD_369",
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with patch("inference.visual_report.REFERENCE_FOLDER", self.root):
            self.assertIsNone(_reference_masks("b" * 64, size))
            self.assertIsNone(_reference_masks(REFERENCE_SHA, (2048, 2048)))
            masks = _reference_masks(REFERENCE_SHA, size)
            self.assertIsNotNone(masks)
            for mask in masks.values():
                self.addCleanup(mask.close)
            counts = {key: mask.histogram()[255] for key, mask in masks.items()}
            self.assertEqual(counts, {"MA": 1, "HE": 1, "EX": 1, "SE": 1})
            encoded_mask.write_bytes(b"tampered")
            self.assertIsNone(_reference_masks(REFERENCE_SHA, size))

    def test_reference_case_does_not_override_an_empty_real_prediction(self):
        frame = Image.new("RGB", (4288, 2848))
        empty = Image.new("L", frame.size)
        self.pred.return_value = {"source_label": "公开模型", "masks": {code: empty for code in LESIONS}}
        with patch("inference.visual_report._reference_masks") as source:
            _, info = _comparison(frame, self.response, REFERENCE_SHA)
            source.assert_not_called()
        self.assertEqual(info["candidate_status"], "no_candidates")
        self.assertEqual(info["reference_grade"], 3)
        self.response.update(status="ok", image_sha256=REFERENCE_SHA)
        self.response["report_summary"] = {
            "dr_severity": {"grade": 2, "status": "completed", "score_display": "61分"},
            "research": {"glaucoma": {"status": "completed", "items": [{
                "code": "glaucoma", "name": "青光眼", "research_threshold_met": True, "score_display": "95分",
            }]}},
        }
        page = _report_html(info, self.response, True)
        _, _, following = page.partition('<h2>主要判读</h2>')
        primary, _, following = following.partition("</section>")
        narrative, _, appendix = following.partition('id="tech-details"')
        self.assertIn("中度非增殖期", primary)
        self.assertIn("存在重度糖网低估风险", primary)
        self.assertNotIn("专家标注", primary)
        self.assertNotIn("模型分期较专家标注低1级", page)
        self.assertIn("演示样例对照：IDRiD_03 专家标注 3 级", narrative)
        self.assertIn("重度糖网升阶复核预警", narrative)
        self.assertNotIn("青光眼模型", narrative)
        for value in ("青光眼模型", "61分", "95分"):
            self.assertIn(value, appendix)
        self.response["image_sha256"] = "b" * 64
        unrelated_page = _report_html(info, self.response, True)
        self.assertNotIn("演示样例对照", unrelated_page)
        self.assertNotIn("重度糖网升阶复核预警", unrelated_page)

    def test_report_metadata_and_patient_plan_omit_internal_identifiers(self):
        request_id = "b2698a772664a9f06cb00b1d85b1c8d2"
        self.response.update(
            status="ok", request_id=request_id, created_utc="2026-11-26T05:35:35+00:00",
            report_summary={"dr_severity": {"grade": 2, "status": "completed"}},
        )
        self.render(request_id)
        page = (self.root / request_id / "report.html").read_text(encoding="utf-8")
        self.assertIn("2026-11-26 13:35:35", page)
        self.assertNotIn("(UTC)", page)
        self.assertRegex(page, r"RET-20261126-[A-Z0-9]{3,}")
        self.assertNotIn(request_id, page)
        self.assertNotRegex(page, r"\b(?:RULE|SRC)_[A-Z0-9_]+")
        for heading in ("下一步怎么做", "在家留意什么", "就诊时可以问", "就诊前可以准备"):
            self.assertIn(f"<h3>{heading}</h3>", page)

    def test_referral_chips_show_all_levels_with_distinct_colors(self):
        self.render()
        page = (self.root / "check/report.html").read_text(encoding="utf-8")
        colors = dict(re.findall(r'\.level-chip\[data-level="(L[0-4])"\]\{[^}]*background:([^;]+);', page))
        self.assertEqual(set(colors), {"L4", "L3", "L2", "L1", "L0"})
        self.assertEqual(len(set(colors.values())), 5)

        predictions = [{"label_code": "DR", "screening_flag": "negative"}]
        for grade, quality_ok, shown in ((3, True, "L4"), (2, True, "L3"), (1, True, "L2"),
                                         (0, True, "L1"), (0, False, "L0"), (None, True, "L0")):
            with self.subTest(grade=grade, quality_ok=quality_ok):
                decision = build_clinical_decision({}, predictions, grade, quality_ok=quality_ok, urgent_hits=[])
                self.assertEqual(display_report_level(decision), shown)
                body = _decision_section_html(decision)
                self.assertIn(f'data-level="{shown}">{shown}</span>', body)
                if shown == "L0":
                    self.assertIsNone(decision["level"])
                    self.assertIn("暂无法分级", body)
        self.assertIsNone(display_report_level(None))

    def test_same_nerve_head_triaged_once_and_reference_bound_to_image(self):
        def group(rows):
            return {"status": "completed", "items": [{"code": code, "research_threshold_met": reached}
                                                       for code, reached in rows]}
        self.response["report_summary"] = {
            "dr_severity": {"status": "completed", "grade": 2},
            "five_label_screening": group([("OPTIC_DISC_CUPPING", True), ("ABNORMAL", True)]),
            "research": {"multilabel_11": group([("OPTIC_DISC_CUPPING", False), ("ODP", True), ("LS", True)]),
                         "glaucoma": group([("glaucoma", True)])}}
        plain = build_clinical_review(self.response, {"reference_grade": 3})
        self.assertNotIn("专家", plain["primary"])
        self.response["image_sha256"] = REFERENCE_SHA
        review = build_clinical_review(self.response, {"reference_grade": 3})
        self.assertIn("模型分期较专家标注低1级", review["primary"])
        self.assertEqual([item["topic"] for item in review["review_items"]], ["糖网分期", "视盘与视神经", "激光治疗痕迹"])
        self.assertIn("可见视野", review["review_items"][0]["action"])
        self.assertIn("五标签模型达到、十一标签模型未达到", review["review_items"][1]["basis"])
        self.assertIn("光凝记录", review["review_items"][2]["action"])

    def test_grade_zero_with_spatial_candidates_or_binary_signal(self):
        self.response["report_summary"] = {"dr_severity": {"status": "completed", "grade": 0}}
        visual = {"localization_status": "localized", "candidate_status": "candidates_present", "legend": [{"code": "EX"}]}
        review = build_clinical_review(self.response, visual)
        self.assertIn("0级", review["primary"])
        self.assertIn("硬性渗出候选区", review["review_items"][0]["basis"])
        self.assertIn("同位置原图", review["review_items"][0]["action"])
        self.assertEqual(build_clinical_review(self.response)["review_items"], [])
        self.response["report_summary"]["five_label_screening"] = {
            "status": "completed", "items": [{"code": "DR", "research_threshold_met": True}]}
        self.assertIn("0级判断不一致", build_clinical_review(self.response)["review_items"][0]["basis"])
        self.response["results"]["quality"]["output"]["predicted_label"] = "Reject"
        blocked = build_clinical_review(self.response, visual)
        self.assertEqual(blocked["review_items"], [])
        self.assertIn("重拍", blocked["primary"])

    def test_unassessed_screening_does_not_become_normal(self):
        self.response["report_summary"] = {"research": {"glaucoma": {"status": "error"}}}
        review = build_clinical_review(self.response)
        self.assertNotIn("正常", str(review))
        self.assertIn("青光眼模型运行失败", str(review["other_results"]))
        self.assertIn("五标签模型未评估", str(review["other_results"]))

    def test_cross_request_response_and_path_escape_are_rejected(self):
        with self.assertRaises(ValueError):
            self.render("../escape")
        self.response["image_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            self.render()
        self.assertEqual(list(self.root.iterdir()), [])
