"""Image geometry and checkpoint identity tests; no model training."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
import torch

from training.classification import ClassificationTransform
from training.fundus_preprocessing import apply_preprocessing, preprocessing_spec
from training.train import evaluation_runtime_config, validate_config
from scripts.cloud_preflight import run_checks
from pathlib import Path


def config():
    return {'task': 'dr_grading', 'kind': 'multiclass', 'labels': ['dr_grade'], 'num_classes': 5,
            'model': {'name': 'tiny_cnn', 'img_size': 64}, 'data': {},
            'preprocessing': {'mode': 'fundus_crop_letterbox_v1'},
            'augmentation': {'resize_before_augmentation': True, 'horizontal_flip': 0,
                             'rotation_degrees': 0, 'color_jitter': 0},
            'training': {'epochs': 30, 'batch_size': 16, 'grad_accum_steps': 1}}


def image():
    result = Image.new('RGB', (200, 100), 'black')
    ImageDraw.Draw(result).ellipse((55, 10, 145, 90), fill=(100, 40, 20))
    return result


class CropIntegrationTests(unittest.TestCase):
    def test_source_transform_uses_exact_shared_geometry(self):
        cfg = config()
        transform = ClassificationTransform(cfg, False)
        expected = transform.tensor(apply_preprocessing(image(), 64, preprocessing_spec(cfg)))
        self.assertTrue(torch.equal(transform(image()), expected))

    def test_cached_transform_does_not_crop_again(self):
        cfg = config()
        prepared = apply_preprocessing(image(), 64, preprocessing_spec(cfg))
        direct = ClassificationTransform(cfg, False)(image())
        transform = ClassificationTransform(cfg, False, input_preprocessed=True)
        with patch('training.classification.apply_preprocessing', side_effect=AssertionError('double crop')):
            self.assertTrue(torch.equal(transform(prepared), direct))

    def test_disabled_augmentation_has_matching_train_val_geometry(self):
        cfg = config()
        self.assertTrue(torch.equal(ClassificationTransform(cfg, True)(image()),
                                    ClassificationTransform(cfg, False)(image())))

    def test_evaluation_cannot_silently_change_geometry(self):
        saved = config()
        for mutation in [{}, {'mode': 'stretch'}, {'mode': 'fundus_crop_letterbox_v1', 'black_threshold': 12}]:
            other = deepcopy(saved)
            other['preprocessing'] = mutation
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                evaluation_runtime_config(saved, other, True)
        runtime, metadata = evaluation_runtime_config(saved, deepcopy(saved), True)
        self.assertEqual(runtime['preprocessing'], saved['preprocessing'])
        self.assertEqual(metadata['deterministic_preprocessing'], preprocessing_spec(saved))

    def test_classification_only_and_explicit_augmentation_order(self):
        cfg = config()
        validate_config(cfg)
        other = deepcopy(cfg)
        other['kind'] = 'segmentation'
        with self.assertRaises(ValueError):
            validate_config(other)
        other = deepcopy(cfg)
        other['augmentation']['resize_before_augmentation'] = False
        with self.assertRaises(ValueError):
            validate_config(other)

    def test_preflight_rejects_invalid_geometry_before_data_access(self):
        cfg = config()
        for changes in [{'kind': 'segmentation'}, {'preprocessing': {'mode': 'unknown'}}]:
            other = deepcopy(cfg)
            other.update(changes)
            report = run_checks(other, Path('not_present'), 1, True)
            self.assertFalse(report['all_checks_passed'])
            self.assertIn('preprocessing' if 'preprocessing' in changes else 'classification',report['errors'][0])

    def test_rectangular_model_config_has_height_width_tensor_shape(self):
        cfg = config()
        cfg['model']['img_size'] = [48, 64]
        direct = ClassificationTransform(cfg, False)(image())
        prepared = apply_preprocessing(image(), (64, 48), preprocessing_spec(cfg))
        cached = ClassificationTransform(cfg, False, input_preprocessed=True)(prepared)
        self.assertEqual(tuple(direct.shape), (3, 48, 64))
        self.assertTrue(torch.equal(direct, cached))


if __name__ == '__main__':
    unittest.main()
