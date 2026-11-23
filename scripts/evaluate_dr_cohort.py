"""Explicitly evaluate fixed, SHA-pinned DR weights on a registered cohort.

This separate entry point authorizes a descriptive cross-cohort evaluation.
It does not relax training.train's ordinary checkpoint/config identity guard,
change the original saved config, initialize from another model, or train.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import torch
from training import classification as adapter
from training.train import evaluation, make_loader, save_dr_predictions, write_json

COHORTS = {'APTOS2019': ('test', 354), 'DDR': ('test', 3898), 'IDRiD': ('external_test', 506)}
KNOWN_MODELS = {'vit_small_patch14_dinov2.lvd142m': (392, 336),
                'vit_large_patch14_dinov2.lvd142m': (336,)}
CHUNK = 8 * 1024**2


def load_pinned_checkpoint(path, expected_sha256):
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('A lowercase expected SHA256 is required')
    path = Path(path).expanduser().resolve()
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(CHUNK), b''):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError('Checkpoint SHA256 mismatch; model was not deserialized')
        handle.seek(0)
        checkpoint = torch.load(handle, map_location='cpu', weights_only=False)
        after = os.fstat(handle.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Checkpoint changed while loading')
    config = checkpoint.get('config') if isinstance(checkpoint, dict) else None
    if (not isinstance(config, dict) or config.get('task') != 'dr_grading'
            or config.get('kind') != 'multiclass' or config.get('labels') != ['dr_grade'] or config.get('num_classes') != 5):
        raise ValueError('Checkpoint must be an owned five-class DR training artifact')
    model = config.get('model', {})
    size = model.get('img_size')
    size = size[0] if isinstance(size, (tuple, list)) and len(size) == 2 and size[0] == size[1] else size
    if model.get('name') not in KNOWN_MODELS or size not in KNOWN_MODELS[model['name']]:
        raise ValueError('Checkpoint architecture/input size is outside the reviewed DINOv2 DR models')
    if type(checkpoint.get('epoch')) is not int or checkpoint['epoch'] < 0 or not isinstance(checkpoint.get('model'), dict):
        raise ValueError('Checkpoint model/epoch metadata is missing')
    identity = checkpoint.get('data_identity')
    normalized = deepcopy(config)
    normalized.pop('notes', None)
    normalized['training'] = {k: v for k, v in normalized.get('training', {}).items() if k not in ('workers', 'log_every')}
    if not isinstance(identity, dict) or identity.get('experiment') != normalized:
        raise ValueError('Original checkpoint configuration contradicts its saved data identity')
    return checkpoint


def evaluation_cohort(config, data_root, source):
    if source not in COHORTS:
        raise ValueError('Unknown evaluation cohort')
    data_root = Path(data_root).resolve()
    root = data_root / 'prepared/source_isolated'
    index_path = root / 'isolation_manifest.json'
    index_bytes = index_path.read_bytes()
    index_hash = hashlib.sha256(index_bytes).hexdigest()
    index = json.loads(index_bytes.decode('utf-8-sig'))
    if (index.get('all_checks_passed') is not True or index.get('format_version') != 1
            or index.get('record_bytes_preserved') is not True or 'strict_cross_domain' not in index.get('base_protocols', [])):
        raise ValueError('The evaluation isolation index is not a successfully audited strict protocol')
    original_pin = config.get('data', {}).get('isolation_manifest_sha256')
    if original_pin is not None and original_pin != index_hash:
        raise ValueError('Evaluation isolation index differs from the original checkpoint index binding')
    split, expected_count = COHORTS[source]
    matches = [combo for combo in index['combinations']
               if (combo['base_protocol'], combo['task'], combo['source']) == ('strict_cross_domain', 'dr_grading', source)]
    if len(matches) != 1:
        raise ValueError('Evaluation cohort does not have one unambiguous isolation-index entry')
    info = matches[0]['splits'][split]
    relative = f'strict_cross_domain/tasks/dr_grading/{source}/{split}.jsonl'
    if info.get('path') != relative:
        raise ValueError('Evaluation manifest path differs from the requested source/task/split')
    manifest = (root / relative).resolve()
    if not manifest.is_relative_to(root.resolve()):
        raise ValueError('Evaluation manifest escapes its isolation root')
    payload = manifest.read_bytes()
    manifest_hash = hashlib.sha256(payload).hexdigest()
    if manifest_hash != info['sha256'] or len(payload) != info['bytes']:
        raise ValueError('Evaluation manifest bytes/SHA256 differ from the isolation index')
    ids = set()
    for raw in payload.splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get('dataset') != source or row.get('task') != 'dr_grading' or row.get('split') != split:
            raise ValueError('Cross-source/task/split contamination in evaluation manifest')
        identifier = row.get('sample_id')
        target = row.get('labels', {}).get('dr_grade')
        if (not isinstance(identifier, str) or not identifier or identifier in ids
                or row.get('label_mask', {}).get('dr_grade') is not True
                or type(target) is not int or not 0 <= target <= 4):
            raise ValueError('Invalid/duplicate sample ID or unavailable five-class target in evaluation cohort')
        ids.add(identifier)
    if len(ids) != info['records'] or len(ids) != expected_count:
        raise ValueError(f'Incomplete evaluation cohort: expected {expected_count}, found {len(ids)}')
    result = deepcopy(config)
    data = result.setdefault('data', {})
    removed = {key: data.pop(key) for key in ('image_cache_dir', 'image_cache_root') if key in data}
    data.update(protocol='strict_cross_domain', source_isolated=True, datasets=[source],
                isolation_root='prepared/source_isolated', isolation_manifest_sha256=index_hash)
    data.pop('external_dataset', None)
    if split == 'external_test':
        data['external_dataset'] = source
    return result, manifest, {'source': source, 'split': split, 'sample_count': len(ids),
                              'manifest_sha256': manifest_hash, 'isolation_manifest_sha256': index_hash,
                              'removed_training_image_cache': removed}


def supervision_history(checkpoint):
    config = checkpoint['config']
    current = config.get('data', {}).get('datasets')
    lineage = checkpoint.get('initialization_lineage')
    if config.get('initialization') is not None:
        if not isinstance(lineage, dict) or lineage.get('new_run_training_sources') != current:
            raise ValueError('Transferred checkpoint lacks its recorded supervised initialization lineage')
        history = lineage.get('supervised_training_source_history')
        expected = list(dict.fromkeys(config['initialization']['parent_training_sources'] + current))
        if history != expected:
            raise ValueError('Checkpoint source history contradicts the original initialization contract')
        return {'current_training_datasets': current, 'actual_supervised_source_history': history, 'initialization_lineage': lineage}
    return {'current_training_datasets': current,
            'actual_supervised_source_history': current if current else None,
            'initialization_lineage': lineage,
            'history_limitation': None if current else 'Legacy config did not enumerate training sources; do not infer source independence.'}


def evaluate_cohort(checkpoint_path, expected_sha256, source, data_root, output, device='cuda', workers=6, batch=16):
    if workers < 0 or batch < 1:
        raise ValueError('workers must be nonnegative and batch must be positive')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Run this controlled exporter as one process, not torchrun/DDP')
    checkpoint_path, data_root, output = Path(checkpoint_path).resolve(), Path(data_root).resolve(), Path(output).resolve()
    if output.is_relative_to(data_root) or output == checkpoint_path:
        raise ValueError('Evaluation output must be separate from immutable data/checkpoint inputs')
    split = COHORTS[source][0]
    destinations = [output / f'evaluation_{split}.json', output / f'predictions_{split}.jsonl', output / f'predictions_{split}.meta.json']
    if any(path.exists() for path in destinations):
        raise FileExistsError('Evaluation outputs already exist; choose a new output directory to preserve them')
    checkpoint = load_pinned_checkpoint(checkpoint_path, expected_sha256)
    original = deepcopy(checkpoint['config'])
    original_serialized = json.dumps(original, sort_keys=True, ensure_ascii=False)
    original_identity = deepcopy(checkpoint['data_identity'])
    source_history = supervision_history(checkpoint)
    cohort_config, manifest, cohort = evaluation_cohort(original, data_root, source)
    runtime_config = deepcopy(cohort_config)
    runtime_config.setdefault('training', {})['batch_size'] = batch
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; choose --device cpu explicitly for CPU evaluation')
    if device not in ('cuda', 'cpu'):
        raise ValueError('device must be cuda or cpu')
    target_device = torch.device(device)
    if device == 'cpu':
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    # No pretrained download or config.initialization application occurs here.
    model = adapter.build_model(original, pretrained=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    completed_epoch = checkpoint['epoch'] + 1
    del checkpoint
    gc.collect()
    model.to(target_device).eval().requires_grad_(False)
    dataset = adapter.build_dataset(manifest, data_root, cohort_config, training=False)
    if len(dataset) != cohort['sample_count']:
        raise ValueError('Adapter omitted records from the complete audited evaluation cohort')
    loader = make_loader(adapter, dataset, runtime_config, False, 0, 1, workers)
    precision = 'fp16' if device == 'cuda' else 'none'
    metrics, predictions = evaluation(adapter, model, loader, target_device, cohort_config, 1, precision, max_steps=None)
    if metrics.get('n') != cohort['sample_count'] or set(metrics.get('by_dataset', {})) != {source}:
        raise ValueError('Evaluation result count/source differs from the audited complete cohort')
    if json.dumps(original, sort_keys=True, ensure_ascii=False) != original_serialized or cohort_config['model'] != original['model']:
        raise AssertionError('Original model configuration was modified')
    authorization = {'entrypoint': 'scripts/evaluate_dr_cohort.py', 'mode': 'explicit_fixed_weights_cross_cohort',
                     'scope': 'Evaluate the requested registered cohort without training, threshold fitting, checkpoint selection, or changing the saved model configuration.',
                     'ordinary_identity_guard': 'training.train is unchanged; its ordinary same-config guard is not invoked by this separately authorized cross-cohort entrypoint.',
                     'original_config_mutated': False, 'original_data_identity_replaced': False}
    output.mkdir(parents=True, exist_ok=True)
    exported = save_dr_predictions(output / f'predictions_{split}.jsonl', predictions, original, cohort['sample_count'],
                                   {'created_utc': datetime.now(timezone.utc).isoformat(), 'split': split,
                                    'checkpoint': str(checkpoint_path), 'checkpoint_sha256': expected_sha256,
                                    'checkpoint_completed_epoch': completed_epoch, 'source': source,
                                    'manifest_sha256': cohort['manifest_sha256'], 'cross_cohort_authorization': authorization})
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'evaluation_mode': 'explicit_cross_cohort',
              'cross_cohort_authorization': authorization, 'split': split, 'source': source,
              'checkpoint': str(checkpoint_path), 'checkpoint_sha256': expected_sha256,
              'checkpoint_completed_epoch': completed_epoch, 'config': original, 'original_config': original,
              'original_data_identity': original_identity, 'evaluation_cohort_config': cohort_config,
              'supervision_history': source_history, 'cohort': cohort,
              'manifest_sha256': cohort['manifest_sha256'], 'evaluated_dataset_count': len(dataset),
              'complete_dataset_evaluated': True, 'training_performed': False, 'optimizer_loaded': False,
              'initialization_applied': False, 'model_weights_updated': False,
              'runtime': {'torch': torch.__version__, 'device': device, 'precision': precision,
                          'workers': workers, 'batch_size': batch}, 'metrics': metrics, 'predictions': exported}
    write_json(output / f'evaluation_{split}.json', report)
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--source', choices=tuple(COHORTS), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    report = evaluate_cohort(args.checkpoint, args.expected_sha256, args.source, args.data_root, args.output,
                             args.device, args.workers, args.batch)
    print(json.dumps({'success': True, 'source': report['source'], 'split': report['split'],
                      'n': report['evaluated_dataset_count'], 'qwk': report['metrics']['qwk'],
                      'checkpoint_completed_epoch': report['checkpoint_completed_epoch'],
                      'training_performed': False, 'output': str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
