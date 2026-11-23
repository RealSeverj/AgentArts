"""Cloud checks/packages/cache selection honor explicit source isolation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

from scripts import dataset_routing as routing
from scripts.build_classification_cache import build_cache, selected_paths
from scripts.cloud_preflight import run_checks
from scripts.isolate_training_datasets import SPLITS, build_isolation
from scripts.package_cloud_training import (code_files, make_plan, selected_rows,
                                            validate_packaged_leaves, write_archive)

PROJECT = Path(__file__).resolve().parents[1]


class IsolatedEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='isolated_entrypoints_')
        self.project = Path(self.temp.name)
        self.data = self.project / 'data'
        self.data.mkdir()
        self.legacy = self.data / 'prepared/tasks/dr_grading'
        self.legacy.mkdir(parents=True)
        rows = {split: [] for split in SPLITS}
        for source in ('DDR', 'APTOS2019'):
            for split in ('train', 'val', 'test'):
                relative = source + '/' + split + '.png'
                path = self.data / relative
                path.parent.mkdir(exist_ok=True)
                Image.new('RGB', (23, 19), (70, 90, 100)).save(path)
                rows[split].append(self.row(source, split, relative))
        relative = 'IDRiD/external.png'
        path = self.data / relative
        path.parent.mkdir(exist_ok=True)
        Image.new('RGB', (25, 20), (100, 80, 50)).save(path)
        rows['external_test'].append(self.row('IDRiD', 'external_test', relative))
        for split in SPLITS:
            (self.legacy / (split + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows[split]), encoding='utf-8')
        isolation = self.data / 'prepared/source_isolated'
        result = build_isolation(self.data, isolation, ('strict_cross_domain',))
        self.assertTrue(result['all_checks_passed'])
        self.cfg = {'task': 'dr_grading', 'kind': 'multiclass', 'labels': ['dr_grade'], 'num_classes': 5,
                    'data': {'source_isolated': True, 'protocol': 'strict_cross_domain', 'datasets': ['DDR'],
                             'isolation_manifest_sha256': hashlib.sha256((isolation / 'isolation_manifest.json').read_bytes()).hexdigest()},
                    'model': {'name': 'tiny_cnn', 'pretrained': False, 'img_size': 16,
                              'mean': [.5] * 3, 'std': [.25] * 3},
                    'training': {'epochs': 1, 'batch_size': 1, 'grad_accum_steps': 1, 'workers': 0}}
        self.config_path = self.project / 'configs/source_isolated/strict_cross_domain/dr_grading_ddr.json'
        self.config_path.parent.mkdir(parents=True)
        self.config_path.write_text(json.dumps(self.cfg), encoding='utf-8')
        self.dummy_code = self.project / 'code.py'
        self.dummy_code.write_text('# package fixture')

    def tearDown(self):
        self.temp.cleanup()

    def row(self, source, split, path):
        return {'sample_id': f'dr_grading:{source}:{split}', 'task': 'dr_grading', 'dataset': source, 'split': split,
                'image_paths': [path], 'image_path': path, 'labels': {'dr_grade': 2}, 'label_mask': {'dr_grade': True},
                'annotations': {}, 'patient_id': f'{source}:{split}',
                # Distinct declared IDs suffice for this metadata route fixture;
                # actual image decoding is tested independently by preflight.
                'image_sha256': [hashlib.sha256(path.encode()).hexdigest()],
                'leakage_group_id': f'group:{source}:{split}', 'content_group_id': f'content:{source}:{split}'}

    def package(self, config=None, code_only=False):
        self.config_path.write_text(json.dumps(config or self.cfg), encoding='utf-8')
        with patch('scripts.package_cloud_training.code_files', return_value=[(self.dummy_code, 'code.py')]):
            return make_plan(self.project, self.data, [self.config_path], code_only)

    def test_preflight_uses_single_source_and_only_explicit_external_source(self):
        report = run_checks(self.cfg, self.data, 1, True)
        self.assertTrue(report['all_checks_passed'], report['errors'])
        self.assertEqual(report['splits']['train']['sources'], {'DDR': 1})
        self.assertEqual(report['splits']['val']['sources'], {'DDR': 1})
        self.assertEqual(report['splits']['test']['sources'], {'DDR': 1})
        self.assertEqual(report['splits']['external_test']['records'], 0)
        external = deepcopy(self.cfg)
        external['data']['external_dataset'] = 'IDRiD'
        report = run_checks(external, self.data, 1, True)
        self.assertTrue(report['all_checks_passed'], report['errors'])
        self.assertEqual(report['splits']['external_test']['sources'], {'IDRiD': 1})

    def test_legacy_fresh_preflight_and_data_package_are_rejected(self):
        legacy = deepcopy(self.cfg)
        legacy['data'].pop('source_isolated')
        legacy['data'].pop('isolation_manifest_sha256')
        report = run_checks(legacy, self.data, 1, True)
        self.assertFalse(report['all_checks_passed'])
        self.assertIn('Source-isolated training route rejected', report['errors'][0])
        with self.assertRaises(ValueError):
            self.package(legacy)
        report, files = self.package(legacy, code_only=True)
        self.assertTrue(report['code_only'])
        self.assertFalse(any(name.startswith('data/') for name in files))

    def test_wrong_source_is_rejected_before_any_dataset_filter(self):
        path = routing.manifest_path(self.cfg, self.data, 'test')
        contaminated = self.row('APTOS2019', 'test', 'APTOS2019/test.png')
        path.write_text(json.dumps(contaminated) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'contamination'):
            list(selected_rows(path, self.cfg, 'test'))
        report = run_checks(self.cfg, self.data, 1, True)
        self.assertFalse(report['all_checks_passed'])
        self.assertTrue(any('isolation index' in error for error in report['errors']))

    def test_test_and_external_leaf_hashes_bound_to_index(self):
        external = deepcopy(self.cfg)
        external['data']['external_dataset'] = 'IDRiD'
        verified = validate_packaged_leaves(external, self.data)
        self.assertEqual(verified['external_test']['records'], 1)
        path = routing.manifest_path(external, self.data, 'external_test')
        content = json.loads(path.read_text())
        content['labels']['dr_grade'] = 3  # Still a valid label, but changed bytes.
        path.write_text(json.dumps(content) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'external_test leaf'):
            validate_packaged_leaves(external, self.data)
        with self.assertRaises(ValueError):
            self.package(external)

    def test_package_contains_index_four_routed_leaves_and_only_selected_images(self):
        cfg = deepcopy(self.cfg)
        cfg['data']['external_dataset'] = 'IDRiD'
        report, files = self.package(cfg)
        self.assertIn('data/prepared/source_isolated/isolation_manifest.json', files)
        for split in SPLITS:
            source = 'IDRiD' if split == 'external_test' else 'DDR'
            expected = f'data/prepared/source_isolated/strict_cross_domain/tasks/dr_grading/{source}/{split}.jsonl'
            self.assertIn(expected, files)
        self.assertFalse(any(name.startswith('data/APTOS2019/') for name in files))
        self.assertFalse(any(name.startswith('data/prepared/tasks/') for name in files))
        self.assertIn('data/IDRiD/external.png', files)
        archive = self.project / 'bundle.zip'
        write_archive(archive, report, files)
        extracted = self.project / 'extracted'
        with zipfile.ZipFile(archive) as package:
            package.extractall(extracted)
        # A subset upload must remain independently routable without any old
        # mixed manifests or the unselected APTOS image source.
        routing.validate_training_route(cfg, extracted / 'data')
        self.assertEqual(validate_packaged_leaves(cfg, extracted / 'data')['test']['records'], 1)

    def test_cache_selected_paths_use_isolated_leaves_and_ignore_external(self):
        cfg = deepcopy(self.cfg)
        cfg['data']['external_dataset'] = 'IDRiD'
        paths, hashes, counts = selected_paths(cfg, self.data)
        self.assertEqual(paths, ['DDR/train.png', 'DDR/val.png'])
        self.assertEqual(counts, {'train': 1, 'val': 1})
        self.assertEqual(hashes['train'], hashlib.sha256(routing.manifest_path(cfg, self.data, 'train').read_bytes()).hexdigest())
        built = build_cache(cfg, self.data, self.project / 'cache', workers=1)
        self.assertTrue(built['source_isolated'])
        self.assertEqual(built['datasets'], ['DDR'])
        self.assertEqual(set(built['entries']), set(paths))

    def test_cache_rejects_isolated_contamination_but_preserves_legacy_filter_reads(self):
        path = routing.manifest_path(self.cfg, self.data, 'train')
        path.write_text(json.dumps(self.row('APTOS2019', 'train', 'APTOS2019/train.png')) + '\n')
        with self.assertRaises(ValueError):
            selected_paths(self.cfg, self.data)
        legacy = deepcopy(self.cfg)
        legacy['data'].pop('source_isolated')
        legacy['data'].pop('isolation_manifest_sha256')
        paths, _, _ = selected_paths(legacy, self.data)
        self.assertEqual(paths, ['DDR/train.png', 'DDR/val.png'])
        self.assertEqual(len(list(selected_rows(self.legacy / 'train.jsonl', legacy))), 1)

    def test_routing_builder_and_generator_are_in_code_package(self):
        included = {relative for _, relative in code_files(PROJECT)}
        for relative in ('scripts/dataset_routing.py', 'scripts/isolate_training_datasets.py', 'scripts/generate_source_configs.py'):
            self.assertIn(relative, included)


if __name__ == '__main__':
    unittest.main()
