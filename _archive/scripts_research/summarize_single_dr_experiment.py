"""Audit one complete DDR test export and describe fixed-prediction uncertainty.

This tool reads only JSON/JSONL files. It never loads weights, reads image pixels,
trains a model, selects epochs, fits thresholds, or requires another cohort.
The optional reference must have its matching standard evaluation and metadata
files beside it, and must cover exactly the same audited DDR test manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import dataset_routing as routing
from scripts.analyze_dr_evaluation import (assert_report_matches, confusion,
    load_predictions, qwk, read_json, sha256_file, summarize_predictions)
from scripts.report_dr_safety_metrics import (bootstrap_metrics, connected_groups,
    fixed_cohort_manifest, interval, prioritized_errors)

SOURCE = 'DDR'
EXPECTED_COUNT = 3898
DEFAULT_REPLICATES = 5000
GRADE_ORDER = list(range(5))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_record(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': sha256_file(path)}


def load_export(prediction_path, data_root):
    """Bind a standard evaluation export to one complete audited DDR manifest."""
    prediction_path, data_root = Path(prediction_path), Path(data_root)
    require(prediction_path.name == 'predictions_test.jsonl',
            'Expected the standard predictions_test.jsonl export filename')
    evaluation_path = prediction_path.with_name('evaluation_test.json')
    meta_path = prediction_path.with_suffix('.meta.json')
    evaluation, meta = read_json(evaluation_path), read_json(meta_path)
    require(isinstance(evaluation, dict) and isinstance(meta, dict),
            'Evaluation and export metadata must be JSON objects')
    config = evaluation.get('config')
    require(isinstance(config, dict), 'Evaluation has no saved configuration')
    data = config.get('data', {})
    require(config.get('task') == 'dr_grading' and config.get('kind') == 'multiclass'
            and config.get('labels') == ['dr_grade'] and config.get('num_classes') == 5
            and data.get('datasets') == [SOURCE] and data.get('source_isolated') is True
            and data.get('protocol') == 'strict_cross_domain',
            'Evaluation must use the isolated DDR five-class test contract')
    index_path = routing.isolation_root(config, data_root) / 'isolation_manifest.json'
    index = read_json(index_path)
    require(type(index.get('format_version')) is int and index['format_version'] == 1
            and index.get('all_checks_passed') is True
            and index.get('record_bytes_preserved') is True
            and 'strict_cross_domain' in index.get('base_protocols', []),
            'Isolation index was not successfully audited with preserved record bytes')
    manifest_path, manifest = fixed_cohort_manifest(data_root, config, 'test', SOURCE, EXPECTED_COUNT)
    relative = manifest_path.relative_to(index_path.parent).as_posix()
    entries = [entry for combination in index.get('combinations', [])
               for entry in combination.get('splits', {}).values() if entry.get('path') == relative]
    require(len(entries) == 1 and entries[0].get('records') == EXPECTED_COUNT,
            'Audited DDR test row count differs from 3898')
    require(sum(bool(line.strip()) for line in manifest_path.read_bytes().splitlines()) == EXPECTED_COUNT
            and all(row.get('label_mask', {}).get('dr_grade') is True for row in manifest.values()),
            'DDR test manifest contains extra or unknown-label rows')
    manifest_sha, prediction_sha = sha256_file(manifest_path), sha256_file(prediction_path)
    require(type(meta.get('format_version')) is int and meta['format_version'] == 1
            and meta.get('split') == 'test' and meta.get('sample_count') == EXPECTED_COUNT
            and meta.get('expected_dataset_count') == EXPECTED_COUNT
            and meta.get('unique_sample_ids') == EXPECTED_COUNT
            and meta.get('source_counts') == {SOURCE: EXPECTED_COUNT}
            and meta.get('all_targets_known') is True
            and meta.get('complete_dataset_evaluated') is True
            and meta.get('target_label') == 'dr_grade' and meta.get('class_order') == GRADE_ORDER
            and meta.get('manifest_sha256') == manifest_sha
            and meta.get('jsonl_sha256') == prediction_sha,
            'Prediction metadata source/count/manifest SHA/JSONL SHA/label contract differs')
    checkpoint = evaluation.get('checkpoint')
    epoch = meta.get('checkpoint_completed_epoch')
    require(isinstance(checkpoint, str) and checkpoint.strip()
            and meta.get('checkpoint') == checkpoint and type(epoch) is int and epoch > 0,
            'Export checkpoint path/epoch is missing or inconsistent')
    if evaluation.get('checkpoint_completed_epoch') is not None:
        require(evaluation['checkpoint_completed_epoch'] == epoch, 'Evaluation checkpoint epoch differs from metadata')
    for owner in (evaluation, meta):
        digest = owner.get('checkpoint_sha256')
        if digest is not None:
            require(isinstance(digest, str) and len(digest) == 64
                    and all(char in '0123456789abcdef' for char in digest),
                    'Export checkpoint SHA256 is malformed')
    if evaluation.get('checkpoint_sha256') is not None and meta.get('checkpoint_sha256') is not None:
        require(evaluation['checkpoint_sha256'] == meta['checkpoint_sha256'],
                'Evaluation checkpoint SHA256 differs from metadata')
    require(evaluation.get('split') == 'test'
            and evaluation.get('evaluated_dataset_count') == EXPECTED_COUNT,
            'Evaluation report does not cover the complete DDR test split')
    recorded = evaluation.get('metrics')
    require(isinstance(recorded, dict) and {'n', 'qwk', 'accuracy', 'confusion_matrix', 'by_dataset'} <= set(recorded),
            'Evaluation report lacks required five-class metrics')
    predictions = load_predictions(prediction_path, manifest)
    require(meta.get('ordering') == 'sample_id_lexicographic'
            and list(predictions) == sorted(manifest), 'Export prediction ordering differs from metadata')
    metrics = summarize_predictions(sorted(manifest), manifest, predictions)
    assert_report_matches(evaluation, 'test', metrics, {SOURCE: metrics}, str(evaluation_path))
    if evaluation.get('predictions') is not None:
        exported = evaluation['predictions']
        require(isinstance(exported, dict) and exported.get('sha256') == prediction_sha
                and exported.get('sample_count') == EXPECTED_COUNT
                and exported.get('unique_sample_ids') == EXPECTED_COUNT,
                'Evaluation export reference differs from prediction bytes/count')
    metrics['confusion_matrix'] = np.asarray(metrics['confusion_matrix'], dtype=np.int64).tolist()
    result = {'checkpoint': checkpoint, 'checkpoint_completed_epoch': epoch,
              'checkpoint_sha256': meta.get('checkpoint_sha256', evaluation.get('checkpoint_sha256')),
              'checkpoint_identity_basis': 'Recorded export checkpoint path/epoch; this statistical tool does not load or hash the weight file.',
              'model': config.get('model'), 'preprocessing': config.get('preprocessing', {'mode': 'stretch'}),
              'current_run_training_sources': data['datasets'],
              'declared_parent_training_sources': (config.get('initialization') or {}).get('parent_training_sources', []),
              'metrics': metrics, 'prioritized_errors': prioritized_errors(metrics['confusion_matrix']),
              'verification': {'full_manifest_ids_targets_sources_content_groups_weights_argmax_verified': True,
                               'prediction_jsonl_sha256': prediction_sha,
                               'manifest_sha256': manifest_sha,
                               'isolation_manifest_sha256': sha256_file(index_path)}}
    for name, grades in (('true_3_to_0_or_1', [3]), ('true_4_to_0_or_1', [4]),
                         ('true_3_or_4_to_0_or_1', [3, 4])):
        sample_ids = [sid for sid in sorted(manifest) if manifest[sid]['labels']['dr_grade'] in grades
                      and predictions[sid]['predicted_class'] in (0, 1)]
        require(len(sample_ids) == result['prioritized_errors'][name]['count'], 'Prioritized error count/ID mismatch')
        result['prioritized_errors'][name]['sample_ids'] = sample_ids
    paths = [evaluation_path, prediction_path, meta_path, index_path, manifest_path]
    return result, manifest, predictions, paths


def supplementary_bootstrap(manifest, predictions_by_model, replicates, seed):
    """Use the same connected groups, seed and chunked draws as bootstrap_metrics.

    Existing bootstrap_metrics supplies QWK and prioritized error-rate CIs.
    These additional draws provide accuracy/recall and optional paired QWK delta;
    neither model is reselected or updated.
    """
    groups, _ = connected_groups(manifest)
    matrices = {name: np.asarray([confusion([manifest[s]['labels']['dr_grade'] for s in members],
                                          [predictions[s]['predicted_class'] for s in members])
                                  for members in groups], dtype=np.int64)
                for name, predictions in predictions_by_model.items()}
    values = {name: {'accuracy': [], **{f'recall_grade_{grade}': [] for grade in GRADE_ORDER}}
              for name in matrices}
    deltas = []
    rng = np.random.default_rng(seed)
    distance = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2
    for start in range(0, replicates, 128):
        count = min(128, replicates - start)
        draw = rng.integers(0, len(groups), size=(count, len(groups)))
        batch_qwk = {}
        for name, grouped in matrices.items():
            sampled = grouped[draw].sum(axis=1)
            total = sampled.sum(axis=(1, 2))
            support = sampled.sum(axis=2)
            diagonal = sampled.diagonal(axis1=1, axis2=2)
            values[name]['accuracy'].extend(diagonal.sum(axis=1) / total)
            for grade in GRADE_ORDER:
                rates = np.full(count, np.nan)
                np.divide(diagonal[:, grade], support[:, grade], out=rates, where=support[:, grade] > 0)
                values[name][f'recall_grade_{grade}'].extend(rates)
            if 'reference' in matrices:
                expected = support[:, :, None] * sampled.sum(axis=1)[:, None, :] / total[:, None, None]
                denominator = (expected * distance).sum(axis=(1, 2))
                ratio = np.full(count, np.nan)
                np.divide((sampled * distance).sum(axis=(1, 2)), denominator, out=ratio, where=denominator > 0)
                batch_qwk[name] = 1 - ratio
        if 'reference' in matrices:
            deltas.extend(batch_qwk['current'] - batch_qwk['reference'])
    result = {'models': {name: {metric: interval(samples) for metric, samples in metrics.items()}
                         for name, metrics in values.items()}}
    if 'reference' in matrices:
        current, reference = (qwk(matrices[name].sum(axis=0)) for name in ('current', 'reference'))
        result['paired_qwk_delta'] = {**interval(deltas),
            'point_delta_current_minus_reference': current - reference if current is not None and reference is not None else None,
            'delta_definition': 'Unweighted sample QWK of current fixed checkpoint minus reference fixed checkpoint',
            'method': 'Paired whole-connected-group percentile bootstrap; identical group draws for both models.'}
    return result


def build_report(run_root, data_root, reference_predictions=None, replicates=DEFAULT_REPLICATES, seed=20261114):
    require(type(replicates) is int and replicates > 0, 'Bootstrap replicates must be a positive integer')
    require(type(seed) is int and seed >= 0, 'Bootstrap seed must be a nonnegative integer')
    run_root, data_root = Path(run_root), Path(data_root)
    current, manifest, predictions, paths = load_export(run_root / 'predictions_test.jsonl', data_root)
    models, exports = {'current': current}, {'current': predictions}
    if reference_predictions is not None:
        reference, reference_manifest, reference_export, reference_paths = load_export(reference_predictions, data_root)
        require(reference_manifest == manifest
                and reference['verification']['manifest_sha256'] == current['verification']['manifest_sha256'],
                'Reference does not use the exact same DDR test manifest')
        models['reference'], exports['reference'] = reference, reference_export
        paths.extend(reference_paths)
    bootstrap = bootstrap_metrics(manifest, exports, replicates=replicates, seed=seed)
    extra = supplementary_bootstrap(manifest, exports, replicates, seed)
    for name, record in models.items():
        record['uncertainty'] = {**bootstrap['models'][name], **extra['models'][name]}
        record['qwk_bootstrap'] = {**record['uncertainty']['qwk'], 'point_qwk': record['metrics']['qwk']}
        for metric in ('true_3_to_0_or_1', 'true_4_to_0_or_1', 'true_3_or_4_to_0_or_1'):
            record['prioritized_errors'][metric]['bootstrap'] = record['uncertainty'][metric]
    evidence = [file_record(path) for path in dict.fromkeys(Path(p).resolve() for p in paths)]
    report = {'format_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'all_checks_passed': True, 'source': SOURCE, 'split': 'test', 'n': EXPECTED_COUNT,
              'purpose': 'Descriptive statistics for already fixed exported predictions; no training, image processing, epoch selection or threshold fitting.',
              'grade_order': GRADE_ORDER, 'confusion_matrix_axes': {'rows': 'true grade', 'columns': 'predicted grade'},
              **current, 'bootstrap_design': {key: value for key, value in bootstrap.items() if key != 'models'},
              'input_files': evidence,
              'verification_scope': 'Audited manifest/index bytes and exported prediction SHA/IDs/targets/groups/probabilities/metrics; source image pixels and checkpoint files are not re-read.',
              'limitations': [*bootstrap['limitations'],
                  'The prioritized true-3/4 to predicted-0/1 pattern is not an exhaustive account of consequential errors; the full confusion matrix is retained.',
                  'Five-class accuracy is the exact grade-match fraction; QWK is a separate agreement statistic.']}
    report['bootstrap_design']['method'] = 'Nonparametric whole-connected-group percentile bootstrap; same group draws for all fixed models and reported intervals.'
    if 'reference' in models:
        report['reference'] = models['reference']
        report['paired_qwk_delta'] = extra['paired_qwk_delta']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True,
                        help='Directory containing evaluation_test.json and predictions_test.jsonl/.meta.json')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--reference-predictions', type=Path)
    parser.add_argument('--replicates', type=int, default=DEFAULT_REPLICATES)
    parser.add_argument('--seed', type=int, default=20261114)
    parser.add_argument('--output', type=Path, help='Default: RUN_ROOT/single_model_report.json')
    args = parser.parse_args()
    report = build_report(args.run_root, args.data_root, args.reference_predictions, args.replicates, args.seed)
    output = (args.output or args.run_root / 'single_model_report.json').resolve()
    require(output not in {Path(item['path']).resolve() for item in report['input_files']},
            'Report output cannot overwrite an input file')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    critical = report['prioritized_errors']['true_3_or_4_to_0_or_1']
    print(json.dumps({'report': str(output), 'n': report['n'], 'qwk': report['metrics']['qwk'],
                      'accuracy': report['metrics']['accuracy'], 'qwk_ci95': report['qwk_bootstrap']['ci95_percentile'],
                      'true_3_or_4_to_0_or_1': {'count': critical['count'], 'denominator': critical['true_grade_denominator']},
                      'replicates': args.replicates}, ensure_ascii=False))


if __name__ == '__main__':
    main()
