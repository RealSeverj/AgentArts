"""Verify a completed, extracted DR training backup before cloud shutdown.

All listed file sizes/SHA256 values are checked before loading the user's own
four trusted PyTorch checkpoints. The original dataset is never modified or
downloaded. Run only after archive extraction has completely finished.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import time

PROJECT = Path(__file__).resolve().parents[1]
EXPECTED = {'test': 20501, 'external_test': 506, 'train': 19480, 'val': 4713}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def manifest_entries(value):
    items = value.get('files', value.get('entries', value)) if isinstance(value, dict) else value
    if isinstance(items, dict):
        items = [dict(record, path=name) if isinstance(record, dict) else {'path': name, 'sha256': record}
                 for name, record in items.items()]
    if not isinstance(items, list):
        raise ValueError('SHA256SUMS.json requires files/entries array or a relative-path mapping')
    result = []
    for item in items:
        name = item.get('path', item.get('relative_path'))
        size = item.get('bytes', item.get('size', item.get('size_bytes')))
        digest = item.get('sha256', '')
        if (not isinstance(name, str) or not name or type(size) is not int or size < 0
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdefABCDEF' for c in digest)):
            raise ValueError('Each checksum entry requires relative path, bytes/size and 64-character SHA256')
        normalized = name.replace('\\', '/')
        relative = PurePosixPath(normalized)
        if relative.is_absolute() or '..' in relative.parts or PureWindowsPath(normalized).drive:
            raise ValueError('Checksum entry leaves the backup root')
        result.append({'path': relative.as_posix(), 'bytes': size, 'sha256': digest.lower()})
    if len({item['path'] for item in result}) != len(result):
        raise ValueError('Checksum manifest contains repeated paths')
    return result


class Audit:
    def __init__(self, root):
        self.root = root.resolve()
        self.errors, self.warnings, self.checks = [], [], []
        self.listed, self.verified = {}, {}
        self.report = {'format_version': 1, 'backup_root': str(self.root),
                       'verified_utc': datetime.now(timezone.utc).isoformat(),
                       'errors': self.errors, 'warnings': self.warnings, 'checks': self.checks}

    def require(self, condition, code, detail=None):
        if not condition:
            self.errors.append({'code': code, **({'detail': detail} if detail is not None else {})})
        return bool(condition)

    def close(self, first, second, code, tolerance=1e-10):
        return self.require(isinstance(first, (int, float)) and isinstance(second, (int, float))
                            and math.isfinite(first) and math.isfinite(second)
                            and math.isclose(first, second, abs_tol=tolerance, rel_tol=tolerance), code,
                            {'observed': first, 'expected': second})

    def locate(self, suffix, required=True):
        matches = [relative for relative in self.listed if relative == suffix or relative.endswith('/' + suffix)]
        if len(matches) != 1:
            if required:
                self.require(False, 'required_backup_path_missing_or_ambiguous', {'suffix': suffix, 'matches': matches})
            return None
        relative = matches[0]
        if relative not in self.verified:
            if required:
                self.require(False, 'required_file_not_hash_verified', relative)
            return None
        return self.root / relative

    def document(self, suffix, required=True):
        path = self.locate(suffix, required)
        if path is None:
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            self.require(False, 'invalid_json_document', suffix)
            return None

    def file_checks(self, manifest_path):
        entries = manifest_entries(read_json(manifest_path))
        self.listed = {item['path']: item for item in entries}
        done_bytes, last_tick, started = 0, time.monotonic(), time.monotonic()
        for index, item in enumerate(entries, 1):
            path = (self.root / item['path']).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                self.require(False, 'listed_file_missing_or_outside_backup', item['path'])
                continue
            size = path.stat().st_size
            if not self.require(size == item['bytes'], 'file_size_mismatch', {'path': item['path'], 'actual': size, 'expected': item['bytes']}):
                continue
            digest = sha256(path)
            if self.require(digest == item['sha256'], 'file_sha256_mismatch', item['path']):
                self.verified[item['path']] = digest
            done_bytes += size
            if time.monotonic() - last_tick >= 30:
                print(json.dumps({'phase': 'file_hashes', 'files_checked': index, 'files_total': len(entries),
                                  'bytes_read': done_bytes}), flush=True)
                last_tick = time.monotonic()
        self.report['file_integrity'] = {'listed_files': len(entries), 'verified_files': len(self.verified),
                                         'bytes_read': done_bytes, 'elapsed_seconds': round(time.monotonic() - started, 2)}
        self.require(len(entries) > 0 and len(self.verified) == len(entries), 'not_all_manifest_files_verified')

    def required_code_checks(self):
        suffixes = ['project/training/train.py', 'project/training/classification.py', 'project/training/metrics.py',
                    'project/training/image_cache.py', 'project/scripts/manifest_dataset.py',
                    'project/scripts/build_classification_cache.py', 'project/requirements-training.txt']
        for suffix in suffixes:
            self.locate(suffix)
        self.checks.append({'scope': 'offline_code_and_dependencies', 'required_suffixes': suffixes})

    def task_manifests(self):
        result, hashes = {}, {}
        for split, expected in EXPECTED.items():
            path = self.locate('prepared/tasks/dr_grading/' + split + '.jsonl')
            if path is None:
                continue
            records = {}
            try:
                with path.open(encoding='utf-8-sig') as handle:
                    for line in handle:
                        if line.strip():
                            row = json.loads(line)
                            sample = row.get('sample_id')
                            if sample in records:
                                self.require(False, 'duplicate_task_manifest_sample_id', {'split': split, 'sample_id': sample})
                            records[sample] = row
                self.require(len(records) == expected, 'task_manifest_count_mismatch', {'split': split, 'observed': len(records), 'expected': expected})
                result[split], hashes[split] = records, sha256(path)
            except (OSError, ValueError):
                self.require(False, 'invalid_task_manifest', split)
        self.report['dr_manifests'] = {split: {'records': len(rows), 'sha256': hashes[split]} for split, rows in result.items()}
        return result, hashes

    def histories(self):
        result = {}
        for run, expected_epochs in [('dr_baseline', list(range(1, 4))), ('dr_fast', list(range(4, 31)))]:
            path = self.locate('runs/' + run + '/history.jsonl')
            if path is None:
                continue
            try:
                rows = [json.loads(text) for text in path.read_text(encoding='utf-8-sig').splitlines() if text.strip()]
                self.require([row.get('epoch') for row in rows] == expected_epochs, 'run_epoch_history_mismatch', run)
                self.require(all(row.get('val', {}).get('n') == EXPECTED['val'] for row in rows), 'validation_count_history_mismatch', run)
                best = max(rows, key=lambda row: row['val']['qwk'])
                result[run] = {'rows': rows, 'best': best}
                self.checks.append({'scope': 'epoch_history', 'run': run, 'epochs': [row['epoch'] for row in rows],
                                    'best_validation_epoch': best['epoch'], 'best_validation_qwk': best['val']['qwk']})
            except (OSError, ValueError, KeyError, TypeError):
                self.require(False, 'invalid_history', run)
        epochs = [row['epoch'] for value in result.values() for row in value['rows']]
        self.require(sorted(epochs) == list(range(1, 31)), 'combined_training_is_not_exactly_30_epochs')
        return result

    def checkpoints(self, histories, manifest_hashes, reference):
        try:
            import torch
        except ImportError:
            self.require(False, 'pytorch_unavailable_for_checkpoint_verification')
            return {}
        summaries, configurations, identities = {}, {}, {}
        for run in ('dr_baseline', 'dr_fast'):
            config = self.document('runs/' + run + '/config.resolved.json')
            identity = self.document('runs/' + run + '/data_identity.json')
            self.document('runs/' + run + '/runtime.json')
            for name in ('best', 'last'):
                label = run + '/' + name
                path = self.locate('runs/' + label + '.pt')
                if path is None:
                    continue
                try:
                    # Only the user's own hash-verified training artifacts are
                    # accepted; optimizer/RNG metadata requires weights_only=False.
                    state = torch.load(path, map_location='cpu', weights_only=False)
                    required = {'epoch', 'model', 'optimizer', 'scheduler', 'scaler', 'config', 'data_identity', 'rng_states', 'world_size', 'best_metric'}
                    self.require(isinstance(state, dict) and required.issubset(state), 'checkpoint_missing_resume_state', label)
                    self.require(state['config'] == config, 'checkpoint_config_differs_from_run_config', label)
                    self.require(state['data_identity'] == identity, 'checkpoint_identity_differs_from_run_identity', label)
                    cfg = state['config']
                    self.require(cfg.get('task') == 'dr_grading' and cfg.get('kind') == 'multiclass'
                                 and cfg.get('labels') == ['dr_grade'] and cfg.get('num_classes') == 5,
                                 'checkpoint_is_not_dr_five_class_model', label)
                    self.require(cfg['model'].get('name') == 'vit_small_patch14_dinov2.lvd142m', 'checkpoint_backbone_unexpected', label)
                    self.require(cfg['training'].get('epochs') == 30, 'checkpoint_total_training_epochs_not_30', label)
                    self.require(state['world_size'] == 2 and len(state['rng_states']) == 2, 'checkpoint_rank_rng_count_mismatch', label)
                    self.require(all({'python', 'numpy', 'torch', 'cuda'}.issubset(rng) for rng in state['rng_states']), 'checkpoint_rng_state_incomplete', label)
                    weights = state['model']
                    self.require(bool(weights) and all(torch.is_tensor(value) for value in weights.values()), 'checkpoint_model_weights_invalid', label)
                    head = weights.get('head.1.weight')
                    self.require(torch.is_tensor(head) and tuple(head.shape) == (5, 384), 'checkpoint_classification_head_shape_mismatch', label)
                    self.require(bool(state['optimizer'].get('state')) and bool(state['optimizer'].get('param_groups')), 'checkpoint_optimizer_state_empty', label)
                    for split in ('train', 'val'):
                        self.require(identity.get('manifests_sha256', {}).get(split) == manifest_hashes.get(split), 'checkpoint_training_manifest_hash_mismatch', {'checkpoint': label, 'split': split})
                    epoch = int(state['epoch']) + 1
                    expected_epoch = (histories[run]['best']['epoch'] if name == 'best' else histories[run]['rows'][-1]['epoch'])
                    self.require(epoch == expected_epoch, 'checkpoint_epoch_does_not_match_history', {'checkpoint': label, 'epoch': epoch, 'expected': expected_epoch})
                    self.close(state['best_metric'], histories[run]['best']['val']['qwk'], 'checkpoint_best_score_does_not_match_history')
                    self.require(state['scheduler'].get('last_epoch') == epoch, 'checkpoint_scheduler_epoch_mismatch', label)
                    summaries[label] = {'completed_epoch': epoch, 'model': cfg['model']['name'],
                                        'model_tensor_count': len(weights), 'model_parameters_and_buffers': sum(value.numel() for value in weights.values()),
                                        'optimizer_state_entries': len(state['optimizer']['state']), 'scheduler_last_epoch': state['scheduler'].get('last_epoch'),
                                        'rank_rng_states': len(state['rng_states']), 'best_validation_qwk': state['best_metric'],
                                        'source': str(path.relative_to(self.root))}
                    configurations[run], identities[run] = cfg, state['data_identity']
                    del head, weights, state
                    gc.collect()
                except Exception as error:
                    self.require(False, 'checkpoint_load_or_metadata_check_failed', {'checkpoint': label, 'error_type': type(error).__name__})
        self.require(len(summaries) == 4, 'not_all_four_checkpoints_loaded_and_checked')
        for run, ref_key in [('dr_baseline', 'parent_best_validation'), ('dr_fast', 'final_validation')]:
            ref = reference.get(ref_key, {})
            actual = summaries.get(run + '/best')
            if actual and ref:
                self.require(actual['completed_epoch'] == ref.get('epoch'), 'checkpoint_best_epoch_differs_from_reference', run)
                self.close(actual['best_validation_qwk'], ref.get('qwk'), 'checkpoint_best_validation_differs_from_reference')
        transition = self.document('runs/dr_fast/transition.json')
        if transition:
            self.require(transition.get('parent_completed_epoch') == 3 and transition.get('continuation_start_epoch') == 4,
                         'transition_epoch_lineage_mismatch')
            self.require(transition.get('bitwise_continuation') is False, 'preprocessing_transition_not_disclosed')
            self.require(transition.get('old_identity') == identities.get('dr_baseline'), 'transition_parent_identity_mismatch')
            self.require(transition.get('new_identity') == identities.get('dr_fast'), 'transition_child_identity_mismatch')
        cache_manifest = self.locate('cache/images/dr392_v1/cache_manifest.json')
        if cache_manifest and identities.get('dr_fast'):
            self.require(sha256(cache_manifest) == identities['dr_fast'].get('image_cache', {}).get('manifest_sha256'), 'backup_cache_manifest_identity_mismatch')
        self.report['checkpoints'] = summaries
        return configurations

    def predictions(self, records, manifest_hashes, configurations):
        summary = self.document('evaluation/paired_summary.json')
        checked = {}
        for model, run in [('baseline', 'dr_baseline'), ('finetuned', 'dr_fast')]:
            for split in ('test', 'external_test'):
                label, expected = model + '/' + split, EXPECTED[split]
                report = self.document('evaluation/' + model + '/evaluation_' + split + '.json')
                meta = self.document('evaluation/' + model + '/predictions_' + split + '.meta.json')
                path = self.locate('evaluation/' + model + '/predictions_' + split + '.jsonl')
                if not report or not meta or path is None or split not in records:
                    continue
                self.require(report.get('evaluated_dataset_count') == expected and report['metrics'].get('n') == expected, 'evaluation_report_count_mismatch', label)
                for field in ('sample_count', 'expected_dataset_count', 'unique_sample_ids'):
                    self.require(meta.get(field) == expected, 'prediction_meta_count_mismatch', {'evaluation': label, 'field': field})
                self.require(meta.get('complete_dataset_evaluated') is True and meta.get('all_targets_known') is True, 'prediction_meta_not_complete', label)
                digest = sha256(path)
                self.require(meta.get('jsonl_sha256') == digest and report.get('predictions', {}).get('sha256') == digest, 'prediction_hash_metadata_mismatch', label)
                self.require(meta.get('manifest_sha256') == manifest_hashes.get(split), 'prediction_test_manifest_hash_mismatch', label)
                self.require(meta.get('checkpoint_completed_epoch') == (1 if model == 'baseline' else 23), 'prediction_checkpoint_epoch_mismatch', label)
                for key in ('task', 'kind', 'labels', 'model', 'data'):
                    self.require(report['config'].get(key) == configurations.get(run, {}).get(key), 'evaluation_config_differs_from_saved_checkpoint', {'evaluation': label, 'field': key})
                matrix, seen, source_counts = [[0] * 5 for _ in range(5)], set(), Counter()
                try:
                    with path.open(encoding='utf-8') as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            sample = row['sample_id']
                            if sample in seen or sample not in records[split]:
                                raise ValueError('Prediction ID is duplicated or absent from manifest')
                            seen.add(sample)
                            reference = records[split][sample]
                            target, prediction = row['target'], row['predicted_class']
                            probabilities = row['probabilities']
                            if (type(target) is not int or type(prediction) is not int or not 0 <= target < 5 or not 0 <= prediction < 5
                                    or target != reference['labels']['dr_grade'] or row['dataset'] != reference['dataset']
                                    or len(probabilities) != 5 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
                                    or not math.isclose(sum(probabilities), 1, abs_tol=1e-5)
                                    or prediction != max(range(5), key=lambda i: probabilities[i])
                                    or row['content_group_id'] != (reference.get('content_group_id') or sample)
                                    or not math.isclose(row['evaluation_weight'], float(reference.get('content_group_evaluation_weight') or 1), abs_tol=1e-12)):
                                raise ValueError('Prediction fields differ from source manifest or valid probabilities')
                            matrix[target][prediction] += 1
                            source_counts[row['dataset']] += 1
                    self.require(len(seen) == expected and seen == set(records[split]), 'prediction_export_not_full_manifest', label)
                    self.require(dict(source_counts) == meta.get('source_counts'), 'prediction_source_counts_mismatch', label)
                    metrics = matrix_metrics(matrix)
                    for key in ('qwk', 'accuracy', 'macro_f1'):
                        self.close(metrics[key], report['metrics'].get(key), 'recomputed_' + key + '_differs_from_evaluation')
                    self.require(matrix == report['metrics'].get('confusion_matrix'), 'recomputed_confusion_matrix_differs_from_evaluation', label)
                    if summary:
                        paired = summary['splits'][split]
                        self.require(paired.get('audited_count') == expected and paired.get('required_manifest_count') == expected, 'paired_summary_count_mismatch', split)
                        self.close(metrics['qwk'], paired['models'][model]['all']['qwk'], 'prediction_qwk_differs_from_paired_summary')
                        self.require(matrix == paired['models'][model]['all']['confusion_matrix'], 'prediction_matrix_differs_from_paired_summary', label)
                    checked[label] = {'n': len(seen), 'source_counts': dict(source_counts), **metrics}
                except (OSError, ValueError, KeyError, TypeError) as error:
                    self.require(False, 'prediction_content_verification_failed', {'evaluation': label, 'error_type': type(error).__name__})
        self.require(len(checked) == 4, 'not_all_four_full_evaluations_checked')
        self.report['evaluations'] = checked


def matrix_metrics(matrix):
    n = sum(map(sum, matrix))
    rows, columns = list(map(sum, matrix)), [sum(matrix[i][j] for i in range(5)) for j in range(5)]
    observed = sum((i-j)**2 * matrix[i][j] for i in range(5) for j in range(5))
    expected = sum((i-j)**2 * rows[i] * columns[j] / n for i in range(5) for j in range(5))
    f1 = [2 * matrix[i][i] / (rows[i] + columns[i]) if rows[i] + columns[i] else 0 for i in range(5)]
    return {'qwk': 1 - observed / expected if expected else None,
            'accuracy': sum(matrix[i][i] for i in range(5)) / n, 'macro_f1': sum(f1) / 5}


def verify_backup(root, manifest_path, reference):
    audit = Audit(root)
    audit.file_checks(manifest_path)
    audit.required_code_checks()
    records, hashes = audit.task_manifests()
    histories = audit.histories()
    configurations = audit.checkpoints(histories, hashes, reference)
    audit.predictions(records, hashes, configurations)
    audit.report['all_checks_passed'] = not audit.errors
    audit.report['scope'] = ['All manifest-listed file sizes and SHA256 values; four own trusted checkpoints loaded on CPU.',
                             'Thirty completed epochs across parent and explicit preprocessing branch; validation-only best checkpoint provenance.',
                             'All predictions aligned to backed-up held-out manifests, distributions checked and QWK/accuracy/macro-F1/confusion matrices recomputed.',
                             'Raw local source images and derived cache PNGs are not duplicated in this backup; this is an artifact integrity/consistency check, not a new model evaluation.']
    return audit.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup-root', '--backup-dir', dest='backup_root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference-state', type=Path, default=PROJECT / 'cloud_upload/deployment_state.json')
    args = parser.parse_args()
    root = args.backup_root.resolve()
    manifest = args.manifest.resolve() if args.manifest else root / 'SHA256SUMS.json'
    output = args.output.resolve() if args.output else root / 'verification.json'
    try:
        reference = read_json(args.reference_state) if args.reference_state.is_file() else {}
        report = verify_backup(root, manifest, reference)
    except Exception as error:
        report = {'all_checks_passed': False, 'backup_root': str(root),
                  'errors': [{'code': 'verification_setup_or_unhandled_error', 'error_type': type(error).__name__}]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(output)
    print(json.dumps({'all_checks_passed': report['all_checks_passed'], 'errors': len(report.get('errors', [])),
                      'files': report.get('file_integrity', {}), 'report': str(output)}, ensure_ascii=False), flush=True)
    return 0 if report['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
