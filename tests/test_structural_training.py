"""Geometry and missing-supervision regression checks; no downloads required."""
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from training import structural


class StructuralTrainingTests(unittest.TestCase):
    def test_unknown_channel_has_no_loss_or_gradient(self):
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(1, 2, 16, 16))

            def forward(self, images):
                return self.logits.expand(len(images), -1, -1, -1)

        model = Constant()
        batch = {"images": torch.zeros(1, 3, 16, 16), "targets": torch.ones(1, 2, 16, 16),
                 "known": torch.tensor([[True, False]]), "sample_ids": ["one"], "datasets": ["fixture"]}
        config = {"kind": "segmentation", "labels": ["known", "unknown"]}
        loss, output = structural.step(model, batch, "cpu", config, True)
        loss.backward()
        self.assertGreater(model.logits.grad[:, 0].abs().sum().item(), 0)
        self.assertEqual(model.logits.grad[:, 1].abs().sum().item(), 0)
        self.assertEqual(structural.metrics([output], config)["per_label"]["unknown"]["known"], 0)

    def test_ddp_loss_counts_known_targets_across_ranks_but_val_has_no_collective(self):
        numerator = torch.tensor(3., requires_grad=True)
        known = torch.tensor([[True, False]])
        # Other rank contributes two known targets; DDP averages gradients by 2.
        with mock.patch.object(torch.distributed, "is_initialized", return_value=True), \
             mock.patch.object(torch.distributed, "get_world_size", return_value=2), \
             mock.patch.object(torch.distributed, "all_reduce", side_effect=lambda value, **kwargs: value.add_(2)) as reduce:
            loss = structural._known_mean(numerator, known, True)
            self.assertAlmostEqual(loss.item(), 2.)
            loss.backward()
            self.assertAlmostEqual(numerator.grad.item(), 2/3, places=6)
            reduce.assert_called_once()
            reduce.reset_mock()
            self.assertEqual(structural._known_mean(numerator, known, False).item(), 3.)
            reduce.assert_not_called()

    def test_flips_keep_image_and_mask_aligned_and_val_uses_whole_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[:8, :12, 1] = 255
            Image.fromarray(image).save(root / "image.png")
            Image.fromarray(image[:, :, 1]).save(root / "mask.png")
            record = {"sample_id": "geometry", "dataset": "fixture", "image_paths": ["image.png"],
                      "label_mask": {"target": True}, "annotations": {"masks": {
                          "target": {"path": "mask.png", "encoding": "binary_0_255", "source_path": "never_decode.tif"}}}}
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            config = {"kind": "segmentation", "labels": ["target"], "model": {"img_size": 32},
                      "augmentation": {"horizontal_flip": 1., "vertical_flip": 1.}}
            sample = structural.build_dataset(manifest, root, config, True)[0]
            self.assertTrue(torch.equal(sample["image"][1], sample["targets"][0]))
            self.assertEqual(sample["targets"][0, -1, -1].item(), 1.)
            # A training patch must never turn into an undocumented validation center crop.
            config["augmentation"]["patch_size"] = 16
            sample = structural.build_dataset(manifest, root, config, False)[0]
            self.assertEqual(sample["targets"].sum().item(), 8 * 12)
            self.assertEqual(sample["targets"][0, 0, 0].item(), 1.)

    def test_xml_policy_is_explicit_and_converts_only_declared_convention(self):
        with tempfile.TemporaryDirectory() as temporary:
            xml = Path(temporary) / "boxes.xml"
            xml.write_text("""<annotation><size><width>32</width><height>20</height></size>
                <object><name>ma</name><bndbox><xmin>1</xmin><ymin>1</ymin>
                <xmax>12</xmax><ymax>8</ymax></bndbox></object></annotation>""", encoding="utf-8")
            with self.assertRaises(ValueError):
                structural.parse_detection_xml(xml, ["MA"], (32, 20), None)
            source = structural.parse_detection_xml(xml, ["MA"], (32, 20), "source_xyxy_no_offset")
            voc = structural.parse_detection_xml(xml, ["MA"], (32, 20), "voc_1based_inclusive")
            self.assertEqual(source["boxes"].tolist(), [[1., 1., 12., 8.]])
            self.assertEqual(voc["boxes"].tolist(), [[0., 0., 12., 8.]])

    def test_tiny_landmark_performs_real_backward(self):
        config = {"kind": "landmark", "labels": ["point"], "model": {"name": "tiny_cnn", "img_size": 32}}
        model = structural.build_model(config, pretrained=False)
        batch = {"images": torch.rand(2, 3, 32, 32), "targets": torch.zeros(2, 1, 2),
                 "known": torch.tensor([[True], [False]]), "original_sizes": torch.tensor([[100., 100.], [100., 100.]]),
                 "sample_ids": ["a", "b"], "datasets": ["fixture", "fixture"]}
        loss, output = structural.step(model, batch, "cpu", config, True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(structural.metrics([output], config)["per_label"]["point"]["known"], 1)

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "pycocotools not installed")
    def test_coco_ap_perfect_empty_and_undefined_category(self):
        config = {"kind": "detection", "labels": ["MA", "HE"],
                  "data": {"box_coordinate_mode": "source_xyxy_no_offset"}}
        truth = {"boxes": torch.tensor([[1., 2., 11., 12.]]), "labels": torch.tensor([1]),
                 "difficult": torch.tensor([0])}
        perfect = {"boxes": truth["boxes"].clone(), "labels": truth["labels"].clone(),
                   "scores": torch.tensor([.99])}
        score = structural._coco_metrics([perfect], [truth], config)
        self.assertAlmostEqual(score["map"], 1.0)
        self.assertIsNone(score["per_label"]["HE"]["ap"])
        empty = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long), "scores": torch.zeros(0)}
        self.assertEqual(structural._coco_metrics([empty], [truth], config)["map"], 0.0)


if __name__ == "__main__":
    unittest.main()
