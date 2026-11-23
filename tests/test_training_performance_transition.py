"""The opt-in performance branch must never become a general resume bypass."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from training.train import (data_identity, load_image_cache_manifest, model_image_size,
                            stop_requested_after_epoch, validate_performance_transition,
                            validate_resume_output, validate_transition_cache)


def config():
    return {'seed': 19, 'task': 'dr_grading', 'kind': 'multiclass', 'labels': ['dr_grade'],
            'num_classes': 5, 'data': {'protocol': 'strict_cross_domain'},
            'model': {'name': 'tiny_cnn', 'img_size': 16, 'pretrained': False},
            'augmentation': {'rotation_degrees': 10, 'color_jitter': .1},
            'loss': {}, 'training': {'epochs': 30, 'batch_size': 4, 'grad_accum_steps': 4,
                                    'lr': .0001, 'amp': 'fp16', 'workers': 4, 'log_every': 25}}


def identity(c=None):
    experiment = deepcopy(c or config())
    experiment['training'] = {k: v for k, v in experiment['training'].items() if k not in ('workers', 'log_every')}
    return {'experiment': experiment, 'manifests_sha256': {'train': 'a' * 64, 'val': 'b' * 64}}


def transitioned(old, cache_path=None):
    new = deepcopy(old)
    new['experiment'].setdefault('augmentation', {})['resize_before_augmentation'] = True
    if cache_path:
        new['experiment']['data']['image_cache_dir'] = cache_path
        new['image_cache'] = {'format_version': 1, 'representation': 'rgb_resized_lossless_png',
                              'image_size': [16, 16], 'resize': 'PIL_BILINEAR',
                              'pixel_equivalence_verified': True, 'manifest_sha256': 'c' * 64}
    return new


class IdentityWhitelistTests(unittest.TestCase):
    def test_absent_or_false_to_true_only(self):
        for explicit_false in (False, True):
            old = identity()
            if explicit_false:
                old['experiment']['augmentation']['resize_before_augmentation'] = False
            new = transitioned(old)
            result = validate_performance_transition(old, new)
            self.assertEqual(result['augmentation.resize_before_augmentation'], {'before': False, 'after': True})
            self.assertNotIn('resize_before_augmentation', old['experiment']['augmentation']) if not explicit_false else None

    def test_missing_whole_augmentation_dictionary(self):
        old = identity()
        old['experiment'].pop('augmentation')
        validate_performance_transition(old, transitioned(old))

    def test_cache_path_requires_content_identity(self):
        old = identity()
        new = transitioned(old, '/cache/verified')
        changes = validate_performance_transition(old, new)
        self.assertIn('data.image_cache_dir', changes)
        new.pop('image_cache')
        with self.assertRaises(ValueError):
            validate_performance_transition(old, new)

    def test_cache_semantics_not_whitelisted(self):
        old = identity()
        for field, value in [('representation', 'jpeg'), ('image_size', [32, 32]),
                             ('resize', 'PIL_NEAREST'), ('pixel_equivalence_verified', False),
                             ('manifest_sha256', 'bad')]:
            new = transitioned(old, '/cache/verified')
            new['image_cache'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_performance_transition(old, new)

    def test_other_experiment_changes_are_strictly_rejected(self):
        mutations = [
            lambda x: x['experiment']['training'].update(batch_size=8),
            lambda x: x['experiment']['training'].update(grad_accum_steps=2),
            lambda x: x['experiment']['training'].update(lr=.001),
            lambda x: x['experiment']['training'].update(epochs=31),
            lambda x: x['experiment']['training'].update(amp='none'),
            lambda x: x['experiment']['model'].update(name='other'),
            lambda x: x['experiment']['model'].update(img_size=32),
            lambda x: x['experiment'].update(labels=['other']),
            lambda x: x['experiment'].update(seed=20),
            lambda x: x['experiment'].update(task='other'),
            lambda x: x['experiment']['data'].update(protocol='within_source'),
            lambda x: x['experiment']['data'].update(datasets=['EyeQ']),
            lambda x: x['experiment']['augmentation'].update(color_jitter=.2),
            lambda x: x['experiment']['loss'].update(label_smoothing=.1),
            lambda x: x['manifests_sha256'].update(train='changed'),
            lambda x: x['manifests_sha256'].update(val='changed'),
        ]
        for index, mutate in enumerate(mutations):
            old = identity()
            new = transitioned(old, '/cache/verified')
            mutate(new)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                validate_performance_transition(old, new)

    def test_not_a_general_resume_override(self):
        old = identity()
        for old_flag, new_flag in [(True, True), (False, False), (True, False), (0, True), (False, 1)]:
            before, after = deepcopy(old), deepcopy(old)
            before['experiment']['augmentation']['resize_before_augmentation'] = old_flag
            after['experiment']['augmentation']['resize_before_augmentation'] = new_flag
            with self.subTest(flags=(old_flag, new_flag)), self.assertRaises(ValueError):
                validate_performance_transition(before, after)
        old['experiment']['kind'] = 'segmentation'
        with self.assertRaises(ValueError):
            validate_performance_transition(old, transitioned(old))


class OutputAndStopTests(unittest.TestCase):
    def test_new_branch_output_required_without_weakening_normal_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / 'parent'
            parent.mkdir()
            checkpoint = parent / 'last.pt'
            checkpoint.write_bytes(b'parent must stay unchanged')
            branch = root / 'branch'
            with self.assertRaises(ValueError):
                validate_resume_output(parent, checkpoint, True)
            with self.assertRaises(ValueError):
                validate_resume_output(branch, checkpoint, False)
            validate_resume_output(parent, checkpoint, False)
            validate_resume_output(branch, checkpoint, True)
            branch.mkdir()
            validate_resume_output(branch, checkpoint, True)
            (branch / 'history.jsonl').write_text('existing')
            with self.assertRaises(ValueError):
                validate_resume_output(branch, checkpoint, True)
            self.assertEqual(checkpoint.read_bytes(), b'parent must stay unchanged')

    def test_stop_request_consumed_only_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(stop_requested_after_epoch(root, 3, 0, 1, torch.device('cpu')))
            (root / 'STOP_AFTER_EPOCH').write_text('pause')
            self.assertTrue(stop_requested_after_epoch(root, 3, 0, 1, torch.device('cpu')))
            self.assertFalse((root / 'STOP_AFTER_EPOCH').exists())
            ack = list(root.glob('STOP_AFTER_EPOCH.ack.epoch_3.*'))
            self.assertEqual(len(ack), 1)
            self.assertEqual(ack[0].read_text(), 'pause')
            self.assertFalse(stop_requested_after_epoch(root, 4, 0, 1, torch.device('cpu')))

    def test_stop_boolean_broadcast_to_all_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'STOP_AFTER_EPOCH').touch()
            with patch('training.train.dist.broadcast') as broadcast:
                self.assertTrue(stop_requested_after_epoch(root, 3, 0, 2, torch.device('cpu')))
                broadcast.assert_called_once()
            def receive(flag, src):
                self.assertEqual(src, 0)
                flag.fill_(1)
            with patch('training.train.dist.broadcast', side_effect=receive):
                self.assertTrue(stop_requested_after_epoch(root, 3, 1, 2, torch.device('cpu')))


class CacheProofTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data, self.cache = self.root / 'data', self.root / 'cache'
        self.data.mkdir()
        self.cache.mkdir()
        self.cfg = config()
        self.cfg['data']['image_cache_dir'] = str(self.cache)
        self.cfg['augmentation']['resize_before_augmentation'] = True
        source = self.data / 'source.png'
        image = Image.new('RGB', (31, 25))
        image.putdata([(x % 255, (x * 3) % 255, (x * 7) % 255) for x in range(31 * 25)])
        image.save(source)
        resized = image.resize((16, 16), Image.Resampling.BILINEAR)
        resized.save(self.cache / 'pixels.png')
        source_stat = source.stat()
        self.manifest = {'format_version': 1, 'representation': 'rgb_resized_lossless_png',
                         'image_size': [16, 16], 'resize': 'PIL_BILINEAR', 'pixel_equivalence_verified': True,
                         'entries': {'source.png': {'cache_path': 'pixels.png', 'source_size': source_stat.st_size,
                                                   'source_mtime_ns': source_stat.st_mtime_ns,
                                                   'pixel_sha256': hashlib.sha256(resized.tobytes()).hexdigest()}}}
        self.save_manifest()
        task = self.data / 'prepared/tasks/dr_grading'
        task.mkdir(parents=True)
        row = {'dataset': 'fixture', 'labels': {'dr_grade': 1}, 'label_mask': {'dr_grade': True}, 'image_paths': ['source.png']}
        for split in ('train', 'val'):
            (task / (split + '.jsonl')).write_text(json.dumps(row) + '\n', encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def save_manifest(self):
        (self.cache / 'cache_manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_cache_contract_reference_pixels_and_bound_manifest(self):
        result = validate_transition_cache(self.cfg, self.data)
        self.assertEqual(result['referenced_images_checked'], 1)
        self.assertEqual(result['cache_pixel_pairs_checked'], 1)
        before = data_identity(self.cfg, self.data)
        self.manifest['extra_creation_note'] = 'a changed content index is bound even when paths are unchanged'
        self.save_manifest()
        self.assertNotEqual(before, data_identity(self.cfg, self.data))

    def test_cache_rejects_different_pixels_even_with_declared_contract(self):
        Image.new('RGB', (16, 16), 'red').save(self.cache / 'pixels.png')
        with self.assertRaises(ValueError):
            validate_transition_cache(self.cfg, self.data)

    def test_source_stat_change_coverage_and_cache_escape_rejected(self):
        for key, value in [('source_size', 1), ('source_mtime_ns', 1), ('cache_path', '../outside.png')]:
            original = deepcopy(self.manifest)
            self.manifest['entries']['source.png'][key] = value
            self.save_manifest()
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_transition_cache(self.cfg, self.data)
            self.manifest = original
        self.manifest['entries'] = {}
        self.save_manifest()
        with self.assertRaises(ValueError):
            validate_transition_cache(self.cfg, self.data)

    def test_wrong_cache_semantics_rejected(self):
        for key, value in [('resize', 'PIL_NEAREST'), ('image_size', [15, 16]),
                           ('representation', 'lossy_jpeg'), ('pixel_equivalence_verified', False)]:
            original = deepcopy(self.manifest)
            self.manifest[key] = value
            self.save_manifest()
            with self.subTest(field=key), self.assertRaises(ValueError):
                load_image_cache_manifest(self.cfg, self.data)
            self.manifest = original


if __name__ == '__main__':
    unittest.main()
