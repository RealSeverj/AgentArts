"""Deterministic geometry fixtures; no real dataset files or training needed."""
import copy
import unittest

import numpy as np
from PIL import Image, ImageDraw

from training.fundus_preprocessing import (MODE, apply_preprocessing,
                                           preprocessing_spec)


class FundusPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.spec = preprocessing_spec({"preprocessing": {"mode": MODE}})

    def test_legacy_stretch_is_pixel_identical_and_metadata_supported(self):
        source = Image.fromarray(np.random.default_rng(3).integers(0, 256, (31, 79, 3), dtype=np.uint8))
        spec = preprocessing_spec({})
        self.assertEqual(spec, {"mode": "stretch"})
        result, meta = apply_preprocessing(source, (40, 50), spec, return_metadata=True)
        expected = source.convert("RGB").resize((40, 50), Image.Resampling.BILINEAR)
        self.assertEqual(result.tobytes(), expected.tobytes())
        self.assertEqual(meta["crop_bbox"], [0, 0, 79, 31])
        self.assertFalse(meta["fallback"])

    def test_black_border_is_cropped_and_detached_corner_label_ignored(self):
        source = Image.new("RGB", (800, 600), "black")
        draw = ImageDraw.Draw(source)
        draw.ellipse((200, 100, 599, 499), fill=(100, 35, 20))
        draw.rectangle((4, 4, 65, 14), fill="white")
        output, meta = apply_preprocessing(source, 336, self.spec, return_metadata=True)
        left, top, right, bottom = meta["crop_bbox"]
        self.assertGreater(left, 150)
        self.assertGreater(top, 70)
        self.assertLess(right, 650)
        self.assertLess(bottom, 540)
        self.assertEqual(output.size, (336, 336))
        self.assertFalse(meta["fallback"])
        self.assertTrue(meta["cropped"])
        # The source rectangle and its mask/labels never change.
        self.assertEqual(source.size, (800, 600))
        self.assertEqual(source.getpixel((5, 5)), (255, 255, 255))

    def test_equal_aspect_fit_centred_padding_and_no_stretch(self):
        source = Image.new("RGB", (400, 200), (90, 20, 10))
        output, meta = apply_preprocessing(source, (300, 300), self.spec, return_metadata=True)
        self.assertEqual(meta["crop_bbox"], [0, 0, 400, 200])
        self.assertEqual(meta["resized_size"], [300, 150])
        self.assertEqual(meta["padding"], [0, 75, 0, 75])
        self.assertEqual(output.getpixel((150, 74)), (0, 0, 0))
        self.assertEqual(output.getpixel((150, 75)), (90, 20, 10))
        self.assertEqual(output.getpixel((150, 225)), (0, 0, 0))

    def test_portrait_rectangular_target_and_odd_padding(self):
        source = Image.new("RGB", (101, 202), (90, 20, 10))
        output, meta = apply_preprocessing(source, (204, 201), self.spec, return_metadata=True)
        self.assertEqual(output.size, (204, 201))
        self.assertEqual(meta["resized_size"], [100, 201])
        self.assertEqual(meta["padding"], [52, 0, 52, 0])

    def test_dark_or_tiny_foreground_falls_back_without_crop(self):
        for source in [Image.new("RGB", (400, 200), "black"),
                       Image.new("RGB", (400, 200), (7, 3, 1))]:
            result, meta = apply_preprocessing(source, 100, self.spec, return_metadata=True)
            self.assertTrue(meta["fallback"])
            self.assertEqual(meta["fallback_reason"], "no_reliable_foreground")
            self.assertEqual(meta["crop_bbox"], [0, 0, 400, 200])
            self.assertEqual(meta["resized_size"], [100, 50])
        source = Image.new("RGB", (500, 500), "black")
        ImageDraw.Draw(source).ellipse((220, 220, 280, 280), fill="white")
        _, meta = apply_preprocessing(source, 100, self.spec, return_metadata=True)
        self.assertEqual(meta["fallback_reason"], "foreground_too_small")
        self.assertEqual(meta["crop_bbox"], [0, 0, 500, 500])

    def test_connected_dim_periphery_is_retained(self):
        source = Image.new("RGB", (700, 500), "black")
        draw = ImageDraw.Draw(source)
        draw.ellipse((120, 20, 579, 479), fill=(6, 2, 1))
        draw.ellipse((200, 100, 499, 399), fill=(100, 35, 20))
        _, meta = apply_preprocessing(source, 336, self.spec, return_metadata=True)
        self.assertLessEqual(meta["crop_bbox"][0], 120)
        self.assertLessEqual(meta["crop_bbox"][1], 20)
        self.assertGreaterEqual(meta["crop_bbox"][2], 580)
        self.assertGreaterEqual(meta["crop_bbox"][3], 480)

    def test_thin_bright_outer_frame_does_not_win_over_retina(self):
        source = Image.new("RGB", (800, 600), "black")
        draw = ImageDraw.Draw(source)
        draw.rectangle((0, 0, 799, 599), outline="white", width=2)
        draw.ellipse((200, 100, 599, 499), fill=(100, 35, 20))
        _, meta = apply_preprocessing(source, 336, self.spec, return_metadata=True)
        self.assertGreater(meta["crop_bbox"][0], 150)
        self.assertGreater(meta["crop_bbox"][1], 70)

    def test_spec_is_canonical_independent_and_strict(self):
        spec = preprocessing_spec({"preprocessing": {"mode": MODE, "crop_margin": 0}})
        self.assertEqual(spec["crop_margin"], 0.0)
        self.assertEqual(spec["min_axis_fraction"], 0.25)
        spec["padding_color"][0] = 5
        self.assertEqual(preprocessing_spec({"preprocessing": {"mode": MODE}})["padding_color"], [0, 0, 0])
        invalid = [None, {}, {"mode": "other"}, {"mode": "stretch", "crop_margin": 0.02},
                   {"mode": MODE, "unknown": True}, {"mode": MODE, "black_threshold": True},
                   {"mode": MODE, "black_threshold": 255}, {"mode": MODE, "analysis_size": 0},
                   {"mode": MODE, "crop_margin": float("nan")},
                   {"mode": MODE, "min_component_fraction": 0},
                   {"mode": MODE, "min_axis_fraction": 2}, {"mode": MODE, "padding_color": [1, 0, 0]}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                preprocessing_spec({"preprocessing": value})
        for size in [True, 0, -2, [16], [16, 0], [16.0, 32]]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                apply_preprocessing(Image.new("RGB", (20, 20)), size, self.spec)

    def test_output_is_deterministic_and_input_unchanged(self):
        source = Image.new("RGB", (433, 277), "black")
        ImageDraw.Draw(source).ellipse((75, 9, 350, 267), fill=(80, 30, 20))
        before = source.tobytes()
        config_before = copy.deepcopy(self.spec)
        a, ma = apply_preprocessing(source, 336, self.spec, return_metadata=True)
        b, mb = apply_preprocessing(source, 336, self.spec, return_metadata=True)
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(ma, mb)
        self.assertEqual(source.tobytes(), before)
        self.assertEqual(self.spec, config_before)


if __name__ == "__main__":
    unittest.main()
