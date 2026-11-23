"""Audit a COMPLETE trusted DR ViT-L run and export checkpoint metadata.

Reads local run artifacts only. No inference, retraining, model movement, or
checkpoint rewriting occurs. Full checkpoint deserialization is restricted to
the user's own trusted training outputs; do not use arbitrary downloaded .pt
files. Incomplete runs fail rather than emitting a completion claim.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import torch

CHUNK = 8 * 1024**2
MODEL_HASH_FORMAT = "legacy_APTOS_state_v1: for sorted(state.items()), hash UTF-8 (key+'|'+str(dtype)+'|'+str(tuple(shape))+'\\n'), then contiguous little-endian CPU tensor bytes; matches the existing APTOS checkpoint_metadata.json recipe"
CHECKPOINTS = ('head_best.pt', 'best.pt', 'last.pt')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def same_number(actual, expected, context):
    if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
            or not math.isfinite(actual) or not math.isclose(float(actual), float(expected), abs_tol=1e-10, rel_tol=0)):
        raise ValueError(f'Numeric metadata differs at {context}: {actual!r} vs {expected!r}')


def experiment_identity(config):
    value = deepcopy(config)
    value.pop('notes', None)
    value['training'] = {k: v for k, v in value['training'].items() if k not in ('workers', 'log_every')}
    return value


def validate_run_contract(config):
    if (config.get('task') != 'dr_grading' or config.get('kind') != 'multiclass'
            or config.get('labels') != ['dr_grade'] or type(config.get('num_classes')) is not int or config['num_classes'] != 5):
        raise ValueError('Metadata export requires the five-class DR grading contract')
    model, training = config.get('model', {}), config.get('training', {})
    if model.get('name') != 'vit_large_patch14_dinov2.lvd142m' or model.get('img_size') not in (336, [336, 336]):
        raise ValueError('This run audit requires the configured DINOv2 ViT-L/14 at 336')
    if (type(training.get('epochs')) is not int or training['epochs'] != 30 or training.get('freeze_backbone_epochs') != 5
            or training.get('unfreeze_last_n_blocks') != 4 or training.get('full_finetune_epoch', 0) < 30
            or training.get('best_metric') != 'qwk' or training.get('best_mode') != 'max'):
        raise ValueError('Expected 30 epochs, first 5 head-only then last 4 blocks, selected by maximum validation QWK')
    sources = config.get('data', {}).get('datasets')
    if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], str) or not sources[0]:
        raise ValueError('The current run must declare exactly one training dataset')
    return sources


def validate_metrics(metrics, expected_n, source, context):
    if not isinstance(metrics, dict) or type(metrics.get('n')) is not int or metrics['n'] != expected_n:
        raise ValueError(f'Validation count mismatch: {context}')
    matrix = np.asarray(metrics.get('confusion_matrix'), dtype=np.float64)
    if (matrix.shape != (5, 5) or not np.isfinite(matrix).all() or (matrix < 0).any()
            or not np.equal(matrix, np.floor(matrix)).all() or int(matrix.sum()) != expected_n):
        raise ValueError(f'Invalid ordinary validation confusion matrix: {context}')
    distance = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / expected_n
    denominator = float((distance * expected).sum())
    if not denominator:
        raise ValueError(f'Undefined validation QWK: {context}')
    calculated = 1 - float((distance * matrix).sum()) / denominator
    same_number(metrics.get('qwk'), calculated, context + '/qwk')
    sources = metrics.get('by_dataset')
    if not isinstance(sources, dict) or set(sources) != {source} or sources[source].get('n') != expected_n:
        raise ValueError(f'Validation metrics contain another/missing dataset: {context}')
    return float(metrics['qwk'])


def model_state_fingerprint(state):
    """Retain the existing APTOS key/dtype/shape header and CPU-byte recipe."""
    if sys.byteorder != 'little':
        raise ValueError('Canonical tensor export currently requires a little-endian CPU host')
    if not isinstance(state, dict) or not state or any(not isinstance(key, str) or not key for key in state):
        raise ValueError('Checkpoint model state must be a nonempty string-keyed mapping')
    digest, schema_digest = hashlib.sha256(), hashlib.sha256()
    schema, numel = [], 0
    for key in sorted(state):
        if '|' in key or '\n' in key or '\r' in key:
            raise ValueError('Model state keys cannot contain hash-header delimiters')
        value = state[key]
        if not torch.is_tensor(value) or value.layout != torch.strided or value.is_quantized:
            raise ValueError(f'Unsupported non-dense/quantized model state value: {key}')
        value = value.detach().cpu().contiguous()
        record = {'key': key, 'dtype': str(value.dtype), 'shape': list(value.shape)}
        metadata = (key + '|' + str(value.dtype) + '|' + str(tuple(value.shape)) + '\n').encode('utf-8')
        digest.update(metadata)
        schema_digest.update(metadata)
        raw = memoryview(value.reshape(-1).view(torch.uint8).numpy())
        for start in range(0, raw.nbytes, CHUNK):
            digest.update(raw[start:start + CHUNK])
        schema.append({**record, 'bytes': raw.nbytes})
        numel += value.numel()
    return {'model_state_sha256': digest.hexdigest(), 'model_state_schema_sha256': schema_digest.hexdigest(),
            'model_state_tensor_count': len(schema), 'model_state_numel': numel}, schema


def runtime_history(log_path, latest):
    result, text, seen = [], '', set()
    if log_path is not None:
        text = Path(log_path).read_text(encoding='utf-8', errors='replace')
        for line in text.splitlines():
            start = line.find('{')
            if start < 0:
                continue
            try:
                value = json.loads(line[start:])
            except ValueError:
                continue
            if (isinstance(value, dict) and {'utc', 'torch', 'world_size', 'train_records', 'val_records'} <= value.keys()
                    and value['train_records'] == latest['train_records'] and value['val_records'] == latest['val_records']
                    and value['world_size'] == latest['world_size']
                    and value.get('smoke_test') is not True):
                encoded = json.dumps(value, sort_keys=True)
                if encoded not in seen:
                    seen.add(encoded)
                    result.append(value)
    if json.dumps(latest, sort_keys=True) not in seen:
        result.append(latest)
    return result, text


def check_lineage(config, lineage, baseline, current_sources, validation_n):
    specification = config.get('initialization')
    if specification is None:
        if lineage is not None or baseline is not None:
            raise ValueError('Unexpected initialization lineage/baseline for a fresh-pretrained config')
        return list(current_sources)
    if not isinstance(lineage, dict) or not isinstance(baseline, dict) or specification.get('mode') != 'weights_only':
        raise ValueError('A model-only transferred run requires its saved initialization lineage and validation baseline')
    for key in ('mode', 'checkpoint_sha256', 'parent_training_sources'):
        if lineage.get(key) != specification.get(key):
            raise ValueError(f'Initialization lineage disagrees with the configured parent: {key}')
    if (lineage.get('new_run_training_sources') != current_sources or lineage.get('starting_epoch_zero_based') != 0
            or lineage.get('optimizer_state_imported') is not False or lineage.get('parent_rng_restored') is not False):
        raise ValueError('Initialization lineage does not describe a fresh model-only transfer')
    expected_history = list(dict.fromkeys(specification['parent_training_sources'] + current_sources))
    if lineage.get('supervised_training_source_history') != expected_history:
        raise ValueError('Supervised source history is missing or misordered')
    if (baseline.get('split') != 'val' or baseline.get('complete_dataset_evaluated') is not True
            or baseline.get('evaluated_dataset_count') != validation_n or baseline.get('optimizer_updates') != 0
            or baseline.get('completed_training_epochs') != 0 or baseline.get('initialization') != lineage):
        raise ValueError('Initialization baseline was not the complete pre-update validation split')
    validate_metrics(baseline.get('metrics'), validation_n, current_sources[0], 'initialization_validation')
    return expected_history


def export_metadata(run_root, output=None, training_log=None):
    run_root = Path(run_root).expanduser().resolve()
    output = Path(output).expanduser().resolve() if output else run_root / 'checkpoint_metadata.json'
    files = {name: run_root / name for name in ('config.resolved.json', 'history.jsonl', 'runtime.json', 'data_identity.json')}
    files.update({name: run_root / name for name in CHECKPOINTS})
    for name in ('initialization_lineage.json', 'initialization_validation.json'):
        if (run_root / name).exists():
            files[name] = run_root / name
    if training_log is not None:
        training_log = Path(training_log).expanduser().resolve()
        files['training_log'] = training_log
    # Check before deserializing large checkpoints or touching the destination.
    for path in files.values():
        if output == path.resolve() or (output.exists() and path.exists() and os.path.samefile(output, path)):
            raise ValueError('Metadata output must not overwrite an input file or a hardlink alias')
    if output.exists():
        raise FileExistsError(f'Metadata output already exists; choose a new --output to preserve it: {output}')
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f'Required completed-run artifact is missing: {path}')
    config, runtime, identity = (read_json(files[key]) for key in ('config.resolved.json', 'runtime.json', 'data_identity.json'))
    current_sources = validate_run_contract(config)
    if identity.get('experiment') != experiment_identity(config):
        raise ValueError('Run data_identity experiment does not match the resolved config')
    manifests = identity.get('manifests_sha256')
    if (not isinstance(manifests, dict) or set(manifests) != {'train', 'val'} or
            any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in manifests.values())):
        raise ValueError('Run identity requires exact train/val SHA256 bindings')
    for field in ('train_records', 'val_records', 'world_size'):
        if type(runtime.get(field)) is not int or runtime[field] < 1:
            raise ValueError(f'Invalid runtime {field}')
    if runtime.get('smoke_test') is not False:
        raise ValueError('Smoke/debug runs cannot be exported as a complete production run')
    training_counts = config.get('loss', {}).get('training_label_counts', {}).get('dr_grade')
    if training_counts is not None and sum(training_counts.values()) != runtime['train_records']:
        raise ValueError('Source-specific training label counts do not match runtime records')
    rows = [json.loads(line) for line in files['history.jsonl'].read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    epochs = config['training']['epochs']
    if (len(rows) != epochs or any(not isinstance(row, dict) or type(row.get('epoch')) is not int for row in rows)
            or [row.get('epoch') for row in rows] != list(range(1, epochs + 1))):
        raise ValueError('History must contain each completed epoch 1..30 exactly once, in order; incomplete/duplicate runs are rejected')
    scores, running_best = [], None
    for row in rows:
        epoch = row['epoch']
        expected_stage = 'head_only' if epoch <= 5 else 'last_4_blocks'
        if row.get('stage') != expected_stage:
            raise ValueError(f'Unexpected training stage at epoch {epoch}: {row.get("stage")}')
        score = validate_metrics(row.get('val'), runtime['val_records'], current_sources[0], f'epoch{epoch}')
        scores.append(score)
        running_best = score if running_best is None else max(running_best, score)
        same_number(row.get('best_metric'), running_best, f'epoch{epoch}/best_metric')
        if row.get('best_metric_name') != 'qwk':
            raise ValueError('History changed the checkpoint-selection metric')
    best_epoch = max(range(1, epochs + 1), key=lambda e: scores[e - 1])
    head_epoch = max(range(1, 6), key=lambda e: scores[e - 1])
    selected_epochs = {'best.pt': best_epoch, 'head_best.pt': head_epoch, 'last.pt': epochs}
    saved_lineage = read_json(files['initialization_lineage.json']) if 'initialization_lineage.json' in files else None
    baseline = read_json(files['initialization_validation.json']) if 'initialization_validation.json' in files else None
    history_sources = check_lineage(config, saved_lineage, baseline, current_sources, runtime['val_records'])
    checkpoints, tensor_schema = {}, None
    for name in CHECKPOINTS:
        path = files[name]
        with path.open('rb') as handle:
            before = os.fstat(handle.fileno())
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(CHUNK), b''):
                digest.update(block)
            handle.seek(0)
            checkpoint = torch.load(handle, map_location='cpu', weights_only=False)
            if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('config'), dict):
                raise ValueError(f'Checkpoint lacks its saved configuration: {name}')
            checkpoint_config = checkpoint['config']
            for key in ('task', 'kind', 'labels', 'num_classes', 'model'):
                if checkpoint_config.get(key) != config.get(key):
                    raise ValueError(f'Checkpoint {name} differs in {key}')
            if (experiment_identity(checkpoint_config) != experiment_identity(config)
                    or checkpoint.get('data_identity') != identity or checkpoint.get('world_size') != runtime['world_size']):
                raise ValueError(f'Checkpoint identity belongs to another experiment/runtime: {name}')
            expected_epoch = selected_epochs[name]
            if type(checkpoint.get('epoch')) is not int or checkpoint['epoch'] + 1 != expected_epoch:
                raise ValueError(f'Checkpoint {name} completed epoch does not match the first best/history position: expected {expected_epoch}')
            if checkpoint.get('initialization_lineage') != saved_lineage:
                raise ValueError(f'Checkpoint initialization lineage differs from the run: {name}')
            expected_best = max(scores[:expected_epoch])
            same_number(checkpoint.get('best_metric'), expected_best, name + '/best_metric')
            fingerprint, schema = model_state_fingerprint(checkpoint.get('model'))
            if tensor_schema is None:
                tensor_schema = schema
            elif tensor_schema != schema:
                raise ValueError('Checkpoint model state key/dtype/shape schemas differ')
            after, named = os.fstat(handle.fileno()), path.stat()
            before_tuple = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            if before_tuple != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or before_tuple != (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns):
                raise ValueError(f'Checkpoint changed during metadata export: {name}')
            entry = {'sha256': digest.hexdigest(), **fingerprint, 'bytes': before.st_size,
                     'completed_epoch': expected_epoch, 'val_qwk': scores[expected_epoch - 1],
                     'saved_global_best_metric': float(checkpoint['best_metric']), 'manifests_sha256': manifests}
            if saved_lineage is not None:
                entry['initialization_lineage'] = saved_lineage
            checkpoints[name] = entry
        del checkpoint
        gc.collect()
        print(json.dumps({'event': 'checkpoint_metadata_verified', 'checkpoint': name, 'completed_epoch': expected_epoch,
                          'bytes': checkpoints[name]['bytes']}, ensure_ascii=False), flush=True)
    runtimes, log_text = runtime_history(training_log, runtime)
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'all_checks_passed': True,
              'run_root': str(run_root), 'training_records': runtime['train_records'],
              'validation_records': runtime['val_records'], 'total_epochs': epochs,
              'training_started': runtimes[0].get('utc'),
              'training_finished': datetime.fromtimestamp(files['history.jsonl'].stat().st_mtime, timezone.utc).isoformat(),
              'timestamp_basis': {'training_started': 'first matching runtime JSON UTC record', 'training_finished': 'history.jsonl filesystem modification time after its final row'},
              'checkpoints': checkpoints, 'current_datasets': current_sources,
              'actual_source_history': history_sources,
              'independent_source_only_model': len(set(history_sources)) == 1,
              'initialization_baseline': baseline, 'initialization_lineage': saved_lineage,
              'runtime_history': runtimes,
              'selection': f'Maximum {current_sources[0]} validation QWK only; first occurrence wins ties. No test/external data are read by this exporter.',
              'traceback_in_log': 'Traceback (most recent call last)' in log_text if training_log else None,
              'model_state_hash_format': MODEL_HASH_FORMAT, 'model_state_tensor_schema': tensor_schema,
              'data_identity': identity,
              'validation_scope': ['Completed epoch/stage/validation histories and checkpoint identities were checked against this run.',
                                   'Recorded manifest/cache hashes are preserved and checked for equality; raw dataset/image bytes are not reread.',
                                   'The configured architecture identity and saved tensor schemas are verified without constructing timm or performing inference.',
                                   'Canonical model hashes use the documented legacy_APTOS_state_v1 recipe, matching the existing APTOS metadata algorithm.'],
              'input_files': [{'path': str(path), 'bytes': path.stat().st_size} for path in files.values()]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        if output.exists():
            raise FileExistsError('Metadata output appeared while exporting; refusing overwrite')
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--training-log', type=Path)
    args = parser.parse_args()
    report = export_metadata(args.run_root, args.output, args.training_log)
    print(json.dumps({'all_checks_passed': True, 'total_epochs': report['total_epochs'],
                      'current_datasets': report['current_datasets'], 'actual_source_history': report['actual_source_history'],
                      'checkpoints': {name: {'completed_epoch': item['completed_epoch'], 'val_qwk': item['val_qwk'], 'sha256': item['sha256']}
                                      for name, item in report['checkpoints'].items()},
                      'output': str(args.output or args.run_root / 'checkpoint_metadata.json')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
