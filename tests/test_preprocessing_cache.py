"""Crop/letterbox caches are separate, pixel-verified preprocessing artifacts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from scripts.build_classification_cache import build_cache
from scripts.dataset_routing import validate_training_route
from scripts.isolate_training_datasets import build_isolation, SPLITS
from scripts.package_cloud_training import make_plan
from training.fundus_preprocessing import apply_preprocessing, preprocessing_spec
from training.image_cache import CachedImageLoader, validate_manifest


class PreprocessingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fundus_cache_")
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.image = Image.new("RGB", (240, 140), "black")
        ImageDraw.Draw(self.image).ellipse((65, 15, 175, 125), fill=(140, 65, 25))
        old = self.data / "prepared/tasks/dr_grading"
        old.mkdir(parents=True)
        for split in SPLITS:
            rows = []
            for grade in range(5):
                relative = f"DDR/{split}_{grade}.png"
                path = self.data / relative
                path.parent.mkdir(exist_ok=True)
                self.image.save(path)
                identifier = f"DDR:{split}:{grade}"
                rows.append({"sample_id": identifier, "task": "dr_grading", "dataset": "DDR", "split": split,
                             "image_path": relative, "image_paths": [relative], "labels": {"dr_grade": grade},
                             "label_mask": {"dr_grade": True}, "annotations": {}, "patient_id": identifier,
                             "content_group_id": identifier, "leakage_group_id": identifier,
                             "image_sha256": [hashlib.sha256(identifier.encode()).hexdigest()]})
            (old / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        isolated = self.data / "prepared/source_isolated"
        self.assertTrue(build_isolation(self.data, isolated, ("strict_cross_domain",))["all_checks_passed"])
        self.cache = self.root / "crop_cache"
        self.config = {"task": "dr_grading", "kind": "multiclass", "labels": ["dr_grade"], "num_classes": 5,
                       "data": {"protocol": "strict_cross_domain", "datasets": ["DDR"], "source_isolated": True,
                                "isolation_manifest_sha256": hashlib.sha256((isolated / "isolation_manifest.json").read_bytes()).hexdigest(),
                                "image_cache_dir": str(self.cache)},
                       "model": {"img_size": 48}, "augmentation": {"resize_before_augmentation": True},
                       "preprocessing": {"mode": "fundus_crop_letterbox_v1"}}
        self.source = self.data / "DDR/train_0.png"

    def tearDown(self):
        self.temporary.cleanup()

    def test_crop_cache_matches_shared_source_processing_and_binds_route(self):
        manifest = build_cache(self.config, self.data, self.cache, workers=1)
        spec = preprocessing_spec(self.config)
        self.assertEqual(manifest["preprocessing"], spec)
        self.assertEqual(manifest["expected_unique_images"], 10)
        self.assertEqual(manifest["split_supervised_records"], {"train": 5, "val": 5})
        self.assertFalse(any("test_" in path for path in manifest["entries"]))
        actual = CachedImageLoader(self.cache, self.data, 48, expected_preprocessing=spec)(self.source.resolve())
        expected = apply_preprocessing(self.image, 48, spec)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertIn("preprocessing_metadata", manifest["entries"]["DDR/train_0.png"])
        self.assertEqual(validate_training_route(self.config, self.data)["source"], "DDR")
        reused = build_cache(self.config, self.data, self.cache, workers=1)
        self.assertEqual(reused["build_counts"], {"reused": 10})

    def test_rectangular_cache_config_uses_model_height_width_order(self):
        self.config['model']['img_size'] = [32, 48]
        manifest = build_cache(self.config, self.data, self.cache, workers=1)
        self.assertEqual(manifest['image_size'], [48, 32])
        loaded = CachedImageLoader(self.cache, self.data, (48, 32),
                                  expected_preprocessing=preprocessing_spec(self.config))(self.source.resolve())
        self.assertEqual(loaded.size, (48, 32))
        self.assertEqual(validate_training_route(self.config, self.data)['source'], 'DDR')

    def test_source_upload_can_be_planned_before_cloud_cache_exists(self):
        path = self.root / 'config.json'
        path.write_text(json.dumps(self.config), encoding='utf-8')
        with patch('scripts.package_cloud_training.code_files', return_value=[]):
            report, files = make_plan(self.root, self.data, [path], False)
        self.assertEqual(report['configs'][0]['datasets'], ['DDR'])
        self.assertIn('data/DDR/train_0.png', files)
        self.assertFalse(self.cache.exists())

    def test_crop_and_stretch_cannot_reuse_or_overwrite_the_same_cache(self):
        build_cache(self.config, self.data, self.cache, workers=1)
        manifest_path = self.cache / "cache_manifest.json"
        original = manifest_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            CachedImageLoader(self.cache, self.data, 48)
        stretch = deepcopy(self.config)
        stretch.pop("preprocessing")
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            build_cache(stretch, self.data, self.cache, workers=1)
        self.assertEqual(manifest_path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            validate_training_route(stretch, self.data)
        wrong = deepcopy(self.config)
        wrong["preprocessing"]["black_threshold"] = 12
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            CachedImageLoader(self.cache, self.data, 48, expected_preprocessing=preprocessing_spec(wrong))

    def test_legacy_stretch_without_spec_is_readable_but_not_a_crop_cache(self):
        cfg = deepcopy(self.config)
        cfg.pop("preprocessing")
        build_cache(cfg, self.data, self.cache, workers=1)
        path = self.cache / "cache_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.pop("preprocessing")  # The exact legacy schema.
        path.write_text(json.dumps(manifest), encoding="utf-8")
        cached = CachedImageLoader(self.cache, self.data, 48)(self.source.resolve())
        direct = self.image.resize((48, 48), Image.Resampling.BILINEAR)
        self.assertEqual(cached.tobytes(), direct.tobytes())
        self.assertEqual(validate_manifest(manifest, 48), (48, 48))
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            validate_manifest(manifest, 48, expected_preprocessing=preprocessing_spec(self.config))
        self.assertEqual(validate_training_route(cfg, self.data)["source"], "DDR")

    def test_cache_rejects_noncanonical_spec_and_other_source_binding(self):
        build_cache(self.config, self.data, self.cache, workers=1)
        path = self.cache / "cache_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        incomplete = deepcopy(manifest)
        incomplete["preprocessing"] = {"mode": "fundus_crop_letterbox_v1"}
        with self.assertRaisesRegex(ValueError, "canonical"):
            validate_manifest(incomplete, 48, expected_preprocessing=preprocessing_spec(self.config))
        manifest["datasets"] = ["APTOS2019"]
        path.write_text(json.dumps(manifest), encoding="utf-8")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "another source/task"):
            build_cache(self.config, self.data, self.cache, workers=1)
        self.assertEqual(path.read_bytes(), before)
        unknown = self.root / "unrecognized_cache"
        unknown.mkdir()
        kept = unknown / "keep.txt"
        kept.write_bytes(b"unrelated bytes must not be overwritten")
        with self.assertRaisesRegex(ValueError, "no recognized metadata"):
            build_cache(self.config, self.data, unknown, workers=1)
        self.assertEqual(kept.read_bytes(), b"unrelated bytes must not be overwritten")


if __name__ == "__main__":
    unittest.main()
