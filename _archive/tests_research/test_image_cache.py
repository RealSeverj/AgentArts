"""Small fixtures for pixel equivalence and strict resized cache boundaries."""
import copy
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from scripts.build_classification_cache import build_cache
from scripts.manifest_dataset import ResizeToTensor
from training.classification import ClassificationTransform
from training.image_cache import CachedImageLoader


class ImageCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix="image_cache_")
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.cache = self.root / "cache"
        self.data.mkdir()
        rng = np.random.default_rng(7)
        self.source = self.data / "a.png"
        Image.fromarray(rng.integers(0, 256, (19, 27, 3), dtype=np.uint8)).save(self.source)
        self.config = {"task": "fixture", "kind": "multiclass", "labels": ["label"], "num_classes": 2,
                       "model": {"name": "tiny_cnn", "img_size": 16, "mean": [0.5] * 3, "std": [0.25] * 3},
                       "augmentation": {"horizontal_flip": 0.5, "rotation_degrees": 10, "color_jitter": 0.1}}
        task = self.data / "prepared/tasks/fixture"
        task.mkdir(parents=True)
        for split in ("train", "val"):
            row = {"sample_id": split, "task": "fixture", "dataset": "fixture", "split": split,
                   "image_paths": ["a.png"], "labels": {"label": 1}, "label_mask": {"label": True}}
            (task / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def build(self):
        return build_cache(self.config, self.data, self.cache, workers=1)

    def test_training_pre_resize_and_validation_tensor_equivalence(self):
        manifest = self.build()
        self.assertTrue(manifest["pixel_equivalence_verified"])
        loader = CachedImageLoader(self.cache, self.data, 16)
        with Image.open(self.source) as image:
            native = image.convert("RGB")
        direct = native.resize((16, 16), resample=Image.Resampling.BILINEAR)
        cached = loader(self.source.resolve())
        self.assertEqual(cached.tobytes(), direct.tobytes())
        training_transform = ClassificationTransform(self.config, True)
        random.seed(123)
        expected = training_transform(direct)
        random.seed(123)
        actual = training_transform(cached)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        val = ResizeToTensor(16, mean=[0.5] * 3, std=[0.25] * 3)
        torch.testing.assert_close(val(native), val(cached), rtol=0, atol=0)
        reused = self.build()
        self.assertEqual(reused["build_counts"], {"reused": 1})

    def test_changed_source_and_missing_or_corrupt_cache_are_rejected(self):
        manifest = self.build()
        loader = CachedImageLoader(self.cache, self.data, 16)
        before = self.source.stat()
        os.utime(self.source, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
        with self.assertRaisesRegex(ValueError, "source changed"):
            loader(self.source.resolve())
        os.utime(self.source, ns=(before.st_atime_ns, before.st_mtime_ns))
        png = self.cache / manifest["entries"]["a.png"]["cache_path"]
        png.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Cached PNG is missing"):
            loader(self.source.resolve())
        self.build()  # Explicit rebuilding repairs a missing cache.
        Image.new("RGB", (16, 16), "black").save(png)
        with self.assertRaisesRegex(ValueError, "pixel SHA256 mismatch"):
            loader(self.source.resolve())

    def test_size_path_and_incomplete_manifest_are_rejected(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "dimensions"):
            CachedImageLoader(self.cache, self.data, 32)
        with self.assertRaisesRegex(ValueError, "separate"):
            build_cache(self.config, self.data, self.data / "cache", workers=1)
        path = self.cache / "cache_manifest.json"
        manifest = json.loads(path.read_text())
        bad = copy.deepcopy(manifest)
        bad["entries"]["a.png"]["cache_path"] = "../outside.png"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            CachedImageLoader(self.cache, self.data, 16)
        manifest["pixel_equivalence_verified"] = False
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            CachedImageLoader(self.cache, self.data, 16)

    def test_parallel_build_uses_the_same_verified_contract(self):
        manifest = build_cache(self.config, self.data, self.cache, workers=2)
        self.assertTrue(manifest["pixel_equivalence_verified"])
        image = CachedImageLoader(self.cache, self.data, 16)(self.source.resolve())
        self.assertEqual(image.size, (16, 16))


if __name__ == "__main__":
    unittest.main()
