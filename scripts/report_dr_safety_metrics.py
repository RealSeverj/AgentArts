"""Describe fixed DR predictions and grouped-bootstrap uncertainty; never tune.

The prioritized true-3/4 -> predicted-0/1 pattern is not an exhaustive account
of consequential errors. Full confusion matrices and other errors remain in
the output. No checkpoint, threshold, class merge or fusion rule is fitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import numpy as np

try:
    from scripts import dataset_routing as routing
    from scripts.analyze_dr_evaluation import (confusion, load_manifest, load_predictions,
                                               matrix_summary, qwk, read_json, sha256_file)
except ImportError:
    import dataset_routing as routing
    from analyze_dr_evaluation import (confusion, load_manifest, load_predictions,
                                       matrix_summary, qwk, read_json, sha256_file)

BASE = Path(__file__).resolve().parents[1]
MODELS = {'new_head': ('head_best', 'aptos_vitl_head_best'),
          'new_partial': ('best', 'aptos_vitl_best')}
COHORTS = {'test': ('APTOS2019', 354), 'external_test': ('IDRiD', 506)}


def cohort_specification(test_source='APTOS2019'):
    counts = {'APTOS2019': 354, 'DDR': 3898}
    if test_source not in counts:
        raise ValueError('Explicit test source must be APTOS2019 or DDR')
    return {'test': (test_source, counts[test_source]), 'external_test': ('IDRiD', 506)}


def select_summary_model(cohort, evaluation_name):
    matches = [(key, model) for key, model in cohort['models'].items()
               if model.get('evaluation_name') == evaluation_name]
    if not matches:
        legacy = dict((folder, key) for folder, key in MODELS.values()).get(evaluation_name)
        if legacy in cohort['models']:
            matches = [(legacy, cohort['models'][legacy])]
    if len(matches) != 1:
        raise ValueError('Comparison summary must identify one fixed new model for each evaluation_name')
    return matches[0]


def fixed_cohort_manifest(data_root, config, split, source, expected):
    routing.validate_isolated_config(config)
    if routing.expected_source(config, split) != source:
        raise ValueError('Evaluation config routes this split to another source')
    path = routing.manifest_path(config, data_root, split)
    index_path = routing.isolation_root(config, data_root) / 'isolation_manifest.json'
    if config['data'].get('isolation_manifest_sha256') != sha256_file(index_path):
        raise ValueError('Evaluation config isolation index hash differs from the local source index')
    index = read_json(index_path)
    relative = path.relative_to(index_path.parent).as_posix()
    indexed = [entry for combination in index.get('combinations', []) for entry in combination['splits'].values()
               if entry.get('path') == relative]
    if (index.get('all_checks_passed') is not True or len(indexed) != 1
            or indexed[0].get('sha256') != sha256_file(path) or indexed[0].get('bytes') != path.stat().st_size):
        raise ValueError('Fixed source manifest differs from its successfully audited isolation index')
    rows = load_manifest(path, split, expected)
    if {row['dataset'] for row in rows.values()} != {source}:
        raise ValueError('Source-isolated manifest contains another dataset/source')
    return path, rows


def actual_stage(run_root, completed_epoch, config, checkpoint_record):
    stages = set()
    for name in ('history.jsonl', 'head_history.jsonl'):
        path = run_root / name
        if path.is_file():
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get('epoch') == completed_epoch and isinstance(row.get('stage'), str):
                        stages.add(row['stage'])
    for field in ('actual_stage', 'stage'):
        if isinstance(checkpoint_record.get(field), str) and checkpoint_record[field]:
            stages.add(checkpoint_record[field])
    if len(stages) > 1:
        raise ValueError('Checkpoint stage differs across fixed metadata/history')
    if stages:
        return next(iter(stages)), 'checkpoint_metadata/history'
    return None, 'unavailable_no_recorded_stage; not inferred from evaluation role or an assumed epoch'


def prioritized_errors(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    result = {}
    for name, grades in [('true_3_to_0_or_1', [3]), ('true_4_to_0_or_1', [4]), ('true_3_or_4_to_0_or_1', [3, 4])]:
        numerator = int(matrix[grades, :2].sum())
        denominator = int(matrix[grades, :].sum())
        result[name] = {'count': numerator, 'true_grade_denominator': denominator,
                        'rate': numerator / denominator if denominator else None}
    return result


def connected_groups(manifest):
    identifiers = sorted(manifest)
    parent = list(range(len(identifiers)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    tokens, coverage = {}, Counter()
    for index, sample in enumerate(identifiers):
        row = manifest[sample]
        for field in ('leakage_group_id', 'content_group_id', 'patient_id'):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                continue
            coverage[field] += 1
            token = (field, row['dataset'] if field == 'patient_id' else '', value)
            if token in tokens:
                first, second = find(index), find(tokens[token])
                if first != second:
                    parent[second] = first
            else:
                tokens[token] = index
    grouped = {}
    for i, sample in enumerate(identifiers):
        grouped.setdefault(find(i), []).append(sample)
    groups = sorted(grouped.values(), key=lambda group: group[0])
    return groups, {'resampling_unit': 'Connected components of any shared declared leakage_group_id/content_group_id and known dataset-scoped patient_id; each component is sampled whole.',
                    'group_count': len(groups), 'group_size_histogram': dict(sorted(Counter(map(len, groups)).items())),
                    'identifier_coverage_rows': dict(coverage),
                    'known_patient_id_rows': coverage['patient_id'], 'rows_without_patient_id': len(identifiers) - coverage['patient_id'],
                    'patient_independence_established': False}


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {'ci95_percentile': np.quantile(finite, [.025, .975], method='linear').tolist() if len(finite) else None,
            'bootstrap_mean': float(finite.mean()) if len(finite) else None,
            'valid_replicates': int(len(finite)), 'undefined_replicates': int(len(values) - len(finite))}


def bootstrap_metrics(manifest, predictions_by_model, replicates=5000, seed=20261113):
    if replicates < 1 or not manifest:
        raise ValueError('Bootstrap requires positive replicates and a nonempty cohort')
    groups, grouping = connected_groups(manifest)
    group_matrices = {}
    for name, predictions in predictions_by_model.items():
        group_matrices[name] = np.asarray([confusion([manifest[s]['labels']['dr_grade'] for s in members],
                                                     [predictions[s]['predicted_class'] for s in members]) for members in groups])
    results = {name: {metric: [] for metric in ('qwk', 'true_3_to_0_or_1', 'true_4_to_0_or_1', 'true_3_or_4_to_0_or_1')}
               for name in predictions_by_model}
    rng = np.random.default_rng(seed)
    distance = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2
    for start in range(0, replicates, 128):
        count = min(128, replicates - start)
        draw = rng.integers(0, len(groups), size=(count, len(groups)))
        for name, matrices in group_matrices.items():
            sampled = matrices[draw].sum(axis=1)
            total = sampled.sum(axis=(1, 2))
            expected = sampled.sum(axis=2)[:, :, None] * sampled.sum(axis=1)[:, None, :] / total[:, None, None]
            denominator = (expected * distance).sum(axis=(1, 2))
            observed = (sampled * distance).sum(axis=(1, 2))
            values = np.full(count, np.nan)
            np.divide(observed, denominator, out=values, where=denominator > 0)
            results[name]['qwk'].extend(1 - values)
            for metric, grades in [('true_3_to_0_or_1', [3]), ('true_4_to_0_or_1', [4]), ('true_3_or_4_to_0_or_1', [3, 4])]:
                numerator = sampled[:, grades, :2].sum(axis=(1, 2))
                support = sampled[:, grades, :].sum(axis=(1, 2))
                rates = np.full(count, np.nan)
                np.divide(numerator, support, out=rates, where=support > 0)
                results[name][metric].extend(rates)
    return {'replicates': replicates, 'seed': seed, 'numpy_version': np.__version__,
            'method': 'Nonparametric whole-group percentile bootstrap, fixed number of observed groups sampled with replacement; same draws used for both fixed models.',
            'grouping': grouping,
            'models': {name: {metric: interval(values) for metric, values in metrics.items()} for name, metrics in results.items()},
            'limitations': ['Intervals condition on these fixed predictions/cohorts and declared grouping; retraining and checkpoint-selection uncertainty are not included.',
                            'A leakage/content group is not established to be an independent clinical patient; absent patient identities limit the uncertainty interpretation.',
                            'Undefined QWK or zero-denominator rate replicates are excluded and counted, never replaced with zero.',
                            'When no errors of a pattern were observed, its percentile-bootstrap interval may be [0,0]; this does not establish zero population risk.']}


def build_report(run_root, data_root, replicates=5000, seed=20261113, test_source='APTOS2019'):
    summary_path = run_root / 'comparison_summary.json'
    summary = read_json(summary_path)
    if summary.get('all_checks_passed') is not True:
        raise ValueError('Original comparison summary was not successfully audited')
    cohorts = cohort_specification(test_source)
    checkpoint_metadata_path = run_root / 'checkpoint_metadata.json'
    checkpoint_metadata = read_json(checkpoint_metadata_path)
    if checkpoint_metadata.get('all_checks_passed') is not True:
        raise ValueError('Checkpoint metadata was not successfully audited')
    report = {'format_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'all_checks_passed': True,
              'test_source': test_source,
              'current_run_training_sources': [test_source],
              'training_history_sources': ['APTOS2019', 'DDR'] if test_source == 'DDR' else ['APTOS2019'],
              'purpose': 'Additional descriptive error counts and absolute-metric uncertainty for already fixed head/partial checkpoints. No test-driven tuning, fusion, threshold fitting or training-epoch selection.',
              'prioritized_pattern': 'True grade 3 or 4 classified as grade 0 or 1',
              'scope_note': 'This prioritized pattern is not the only potentially consequential error. Complete five-class confusion matrices, other underestimation and overestimation are retained.',
              'confusion_matrix_axes': {'rows': 'true grade', 'columns': 'predicted grade'},
              'grade_order': [0, 1, 2, 3, 4], 'cohorts': {},
              'input_files': [{'path': str(path.resolve()), 'sha256': sha256_file(path)}
                              for path in (summary_path, checkpoint_metadata_path)]}
    for key in ('initialization_lineage', 'lineage_evidence'):
        if key in summary:
            report[key] = summary[key]
    for name in ('history.jsonl', 'head_history.jsonl'):
        path = run_root / name
        if path.is_file():
            report['input_files'].append({'path': str(path.resolve()), 'sha256': sha256_file(path)})
    for offset, (split, (source, expected)) in enumerate(cohorts.items()):
        manifest = None
        predicted, models = {}, {}
        for name, (folder, _) in MODELS.items():
            root = run_root / 'evaluation' / folder
            predictions_path, meta_path = root / f'predictions_{split}.jsonl', root / f'predictions_{split}.meta.json'
            evaluation_path = root / f'evaluation_{split}.json'
            evaluation = read_json(evaluation_path)
            config = evaluation['config']
            if config.get('data', {}).get('datasets') != [test_source]:
                raise ValueError('Saved evaluation config current training source differs from --test-source')
            manifest_path, current_manifest = fixed_cohort_manifest(data_root, config, split, source, expected)
            index_path = routing.isolation_root(config, data_root) / 'isolation_manifest.json'
            report['input_files'].append({'path': str(index_path.resolve()), 'sha256': sha256_file(index_path)})
            if manifest is not None and current_manifest != manifest:
                raise ValueError('The two fixed model evaluations do not use the same source manifest')
            if manifest is None:
                manifest = current_manifest
                report['input_files'].append({'path': str(manifest_path.resolve()), 'sha256': sha256_file(manifest_path)})
            meta = read_json(meta_path)
            if (meta.get('sample_count') != expected or meta.get('unique_sample_ids') != expected
                    or meta.get('jsonl_sha256') != sha256_file(predictions_path)
                    or meta.get('manifest_sha256') != sha256_file(manifest_path)
                    or meta.get('source_counts') != {source: expected}):
                raise ValueError('Prediction metadata count/source/SHA differs from the fixed manifest')
            predictions = load_predictions(predictions_path, manifest)
            predicted[name] = predictions
            identifiers = sorted(manifest)
            targets = [manifest[s]['labels']['dr_grade'] for s in identifiers]
            labels = [predictions[s]['predicted_class'] for s in identifiers]
            matrix = confusion(targets, labels).astype(np.int64)
            summary_name, original = select_summary_model(summary['splits'][split], folder)
            current_sources = original.get('current_run_training_sources', original.get('training_sources'))
            history_sources = original.get('training_history_sources', original.get('training_sources'))
            if current_sources != [test_source] or history_sources != report['training_history_sources']:
                raise ValueError('Comparison summary current source or supervision history differs from the declared experiment')
            checkpoint_record = checkpoint_metadata['checkpoints'][folder + '.pt']
            if (summary['splits'][split]['source'] != source or summary['splits'][split]['n'] != expected
                    or original['checkpoint_completed_epoch'] != meta['checkpoint_completed_epoch']
                    or checkpoint_record['completed_epoch'] != meta['checkpoint_completed_epoch']
                    or not np.array_equal(matrix, original['metrics']['confusion_matrix'])
                    or not math.isclose(qwk(matrix), original['metrics']['qwk'], rel_tol=0, abs_tol=1e-12)):
                raise ValueError('Recomputed fixed predictions differ from the original comparison report')
            critical = prioritized_errors(matrix)
            cases = [{'sample_id': s, 'dataset': source, 'true_grade': manifest[s]['labels']['dr_grade'],
                      'predicted_grade': predictions[s]['predicted_class'],
                      'leakage_group_id': manifest[s].get('leakage_group_id'),
                      'content_group_id': manifest[s].get('content_group_id'), 'patient_id': manifest[s].get('patient_id')}
                     for s in identifiers if manifest[s]['labels']['dr_grade'] in (3, 4) and predictions[s]['predicted_class'] in (0, 1)]
            total_errors = int(matrix.sum() - matrix.trace())
            for metric, grades in [('true_3_to_0_or_1', [3]), ('true_4_to_0_or_1', [4]), ('true_3_or_4_to_0_or_1', [3, 4])]:
                critical[metric]['sample_ids'] = [row['sample_id'] for row in cases if row['true_grade'] in grades]
                if len(critical[metric]['sample_ids']) != critical[metric]['count']:
                    raise ValueError('Critical error case list/count mismatch')
            stage, stage_origin = actual_stage(run_root, meta['checkpoint_completed_epoch'], config, checkpoint_record)
            models[name] = {'checkpoint': original['checkpoint'], 'checkpoint_completed_epoch': meta['checkpoint_completed_epoch'],
                            'comparison_summary_model': summary_name, 'evaluation_name': folder,
                            'actual_stage': stage, 'stage_verified_from': stage_origin,
                            'current_run_training_sources': current_sources, 'training_history_sources': history_sources,
                            'fixed_prediction_metrics': matrix_summary(matrix, expected),
                            'prioritized_errors': critical, 'prioritized_error_cases': cases,
                            'other_errors_retained': {'all_misclassified_count': total_errors,
                                                      'outside_prioritized_pattern_count': total_errors - len(cases),
                                                      'all_underestimation_count': int(np.tril(matrix, -1).sum()),
                                                      'all_overestimation_count': int(np.triu(matrix, 1).sum())}}
            report['input_files'].extend([{'path': str(p.resolve()), 'sha256': sha256_file(p)} for p in (predictions_path, meta_path, evaluation_path)])
        bootstrap = bootstrap_metrics(manifest, predicted, replicates, seed + offset)
        for name, values in bootstrap['models'].items():
            models[name]['qwk_bootstrap'] = {**values['qwk'], 'point_qwk': models[name]['fixed_prediction_metrics']['qwk']}
            for metric in ('true_3_to_0_or_1', 'true_4_to_0_or_1', 'true_3_or_4_to_0_or_1'):
                models[name]['prioritized_errors'][metric]['bootstrap'] = values[metric]
        report['cohorts'][split] = {'source': source, 'n': expected, 'models': models,
                                    'bootstrap_design': {k: v for k, v in bootstrap.items() if k != 'models'}}
    paired_path = run_root / 'paired_head_vs_partial.json'
    if paired_path.is_file():
        paired = read_json(paired_path)
        report['existing_paired_delta_reference'] = {'path': str(paired_path.resolve()), 'sha256': sha256_file(paired_path),
                                                    'replicates': paired.get('bootstrap_replicates'),
                                                    'delta_definition': paired.get('delta_definition'), 'recomputed': False}
        report['input_files'].append({'path': str(paired_path.resolve()), 'sha256': sha256_file(paired_path)})
    report['input_files'] = list({item['path']: item for item in report['input_files']}.values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=BASE / 'cloud_upload/aptos_vitl_20261113')
    parser.add_argument('--data-root', type=Path, default=BASE / 'data')
    parser.add_argument('--replicates', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--test-source', choices=['APTOS2019', 'DDR'], default='APTOS2019',
                        help='Explicit held-out source: APTOS2019=354 or DDR=3898; external IDRiD remains 506')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (args.output or args.run_root / 'critical_errors_and_bootstrap.json').resolve()
    result = build_report(args.run_root.resolve(), args.data_root.resolve(), args.replicates, args.seed, args.test_source)
    if output in {Path(info['path']).resolve() for info in result['input_files']}:
        parser.error('Output must not replace an original input report, manifest or prediction file')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, output)
    print(json.dumps({'all_checks_passed': True, 'report': str(output),
                      'summary': {split: {name: {'qwk': model['qwk_bootstrap'], 'prioritized_errors': model['prioritized_errors']}
                                          for name, model in cohort['models'].items()} for split, cohort in result['cohorts'].items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
