"""Single-GPU / torchrun DDP training. Test sets are accessed only explicitly."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler, Subset

PROJECT = Path(__file__).resolve().parents[1]


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if torch.is_tensor(value):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class EvaluationSampler(Sampler):
    """Rank-strided validation without DistributedSampler's duplicate padding."""
    def __init__(self, dataset, rank, world):
        self.n, self.rank, self.world = len(dataset), rank, world

    def __iter__(self):
        return iter(range(self.rank, self.n, self.world))

    def __len__(self):
        return len(range(self.rank, self.n, self.world))


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def prepared_root(config, data_root):
    protocol = config.get('data', {}).get('protocol', 'strict_cross_domain')
    if protocol not in ('strict_cross_domain', 'within_source'):
        raise ValueError('Unknown data.protocol: ' + protocol)
    root = data_root / 'prepared'
    return root if protocol == 'strict_cross_domain' else root / 'protocols' / 'within_source'


def manifest_path(config, data_root, split):
    from scripts.dataset_routing import manifest_path as routed_manifest_path
    return routed_manifest_path(config, data_root, split)


def data_identity(config, data_root):
    fingerprints = {}
    # Test sets do not need to be read during training or model selection.
    for split in ('train', 'val'):
        p = manifest_path(config, data_root, split)
        fingerprints[split] = hashlib.sha256(p.read_bytes()).hexdigest()
    experiment = {k: v for k, v in config.items() if k != 'notes'}
    experiment['training'] = {k: v for k, v in config['training'].items() if k not in ('workers', 'log_every')}
    identity = {'experiment': experiment, 'manifests_sha256': fingerprints}
    if config.get('data', {}).get('image_cache_dir'):
        _, _, cache_identity = load_image_cache_manifest(config, data_root)
        identity['image_cache'] = cache_identity
    return identity


def model_image_size(config):
    value = config['model'].get('img_size', 384)
    if isinstance(value, int):
        return [value, value]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('model.img_size must be an int or [height,width]')
    return [int(value[1]), int(value[0])]


def load_image_cache_manifest(config, data_root):
    """Bind cache content index and its pixel representation to the experiment."""
    value = config.get('data', {}).get('image_cache_dir')
    if not isinstance(value, str) or not value.strip():
        raise ValueError('data.image_cache_dir must be a nonempty local path')
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = data_root / root
    root = root.resolve()
    payload = (root / 'cache_manifest.json').read_bytes()
    manifest = json.loads(payload.decode('utf-8-sig'))
    required = {'format_version': 1, 'representation': 'rgb_resized_lossless_png',
                'image_size': model_image_size(config), 'resize': 'PIL_BILINEAR',
                'pixel_equivalence_verified': True}
    if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError('Image cache does not satisfy the fixed RGB resized lossless PNG contract')
    if manifest.get('pixel_equivalence_verified') is not True or type(manifest.get('format_version')) is not int:
        raise ValueError('Image cache verification and format fields have invalid types')
    if not isinstance(manifest.get('entries'), dict):
        raise ValueError('Image cache manifest entries must map source paths to cache records')
    from training.fundus_preprocessing import preprocessing_spec
    from training.image_cache import validate_manifest
    specification = preprocessing_spec(config)
    validate_manifest(manifest, model_image_size(config), expected_preprocessing=specification)
    # Keep the identity of already completed legacy runs byte-for-byte compatible.
    # New geometric preprocessing is additionally bound to the cache identity.
    if specification['mode'] != 'stretch':
        required['preprocessing'] = specification
    return root, manifest, {**required, 'manifest_sha256': hashlib.sha256(payload).hexdigest()}


def validate_performance_transition(old_identity, new_identity):
    """Allow only the explicit classification resize-order transition.

    DataLoader workers/log_every are already runtime-only in data_identity.
    Every other field, including manifests, batch layout and LR plan, remains
    byte-for-byte/equality constrained after the two named fields are removed.
    """
    if not isinstance(old_identity, dict) or not isinstance(new_identity, dict):
        raise ValueError('Performance transition requires complete old and new data identities')
    old, new = deepcopy(old_identity), deepcopy(new_identity)
    old_exp, new_exp = old.get('experiment'), new.get('experiment')
    if not isinstance(old_exp, dict) or not isinstance(new_exp, dict):
        raise ValueError('Performance transition requires experiment identities')
    if old_exp.get('kind') not in ('multiclass', 'multilabel') or new_exp.get('kind') != old_exp['kind']:
        raise ValueError('Resize-order transition is restricted to the same classification kind')
    old_aug, new_aug = old_exp.setdefault('augmentation', {}), new_exp.setdefault('augmentation', {})
    if not isinstance(old_aug, dict) or not isinstance(new_aug, dict):
        raise ValueError('Augmentation identities must be dictionaries')
    if old_aug.get('resize_before_augmentation', False) is not False or new_aug.get('resize_before_augmentation') is not True:
        raise ValueError('Transition requires resize_before_augmentation absent/false -> true')
    old_aug.pop('resize_before_augmentation', None)
    new_aug.pop('resize_before_augmentation', None)
    old_cache = old_exp.setdefault('data', {}).pop('image_cache_dir', None)
    new_cache = new_exp.setdefault('data', {}).pop('image_cache_dir', None)
    for path in (old_cache, new_cache):
        if path is not None and (not isinstance(path, str) or not path.strip()):
            raise ValueError('Transition cache paths must be nonempty local path strings')
    for identity, experiment, cache_path in ((old, old_exp, old_cache), (new, new_exp, new_cache)):
        cache = identity.pop('image_cache', None)
        if cache_path is None:
            if cache is not None:
                raise ValueError('Unexpected cache identity without a configured cache')
            continue
        expected = {'format_version': 1, 'representation': 'rgb_resized_lossless_png',
                    'image_size': model_image_size(experiment), 'resize': 'PIL_BILINEAR',
                    'pixel_equivalence_verified': True}
        if (not isinstance(cache, dict) or set(cache) != set(expected) | {'manifest_sha256'}
                or any(cache.get(k) != v for k, v in expected.items())
                or cache.get('pixel_equivalence_verified') is not True
                or not isinstance(cache.get('manifest_sha256'), str)
                or len(cache['manifest_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in cache['manifest_sha256'])):
            raise ValueError('Transition requires a validated cache content identity')
    if old != new:
        raise ValueError('Performance transition changes fields outside resize order and cache location')
    changes = {'augmentation.resize_before_augmentation': {'before': False, 'after': True}}
    if old_cache != new_cache:
        changes['data.image_cache_dir'] = {'before': old_cache, 'after': new_cache}
    return changes


def validate_resume_output(output, checkpoint_path, performance_transition=False):
    output, parent = Path(output).resolve(), Path(checkpoint_path).resolve().parent
    if performance_transition:
        if output == parent:
            raise ValueError('Performance transition requires a new output directory, separate from the parent run')
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError('Performance transition output must be a new or empty directory')
    elif output != parent:
        raise ValueError('--resume requires --output to be the checkpoint directory, preserving its best.pt and history')


def validate_transition_cache(config, data_root):
    """Check all selected references and sample actual reference/cache pixels.

    The builder's contract attests every entry was compared at creation time.
    This startup check covers all source stats/paths and deterministically checks
    up to two real pixel pairs per source and split, without another serial
    full-resolution decode of the entire training collection.
    """
    if not config.get('data', {}).get('image_cache_dir'):
        return {'enabled': False, 'cache_pixel_pairs_checked': 0}
    from PIL import Image
    from scripts.manifest_dataset import resolve_data_path
    from training.fundus_preprocessing import apply_preprocessing, preprocessing_spec
    specification = preprocessing_spec(config)
    root, manifest, cache_identity = load_image_cache_manifest(config, data_root)
    entries, expected_size = manifest['entries'], tuple(model_image_size(config))
    allowed = config.get('data', {}).get('datasets')
    allowed = set(allowed) if allowed else None
    references, sample_groups = {}, {}
    for split in ('train', 'val'):
        with manifest_path(config, data_root, split).open(encoding='utf-8-sig') as handle:
            for text in handle:
                if not text.strip():
                    continue
                row = json.loads(text)
                if allowed is not None and row['dataset'] not in allowed:
                    continue
                if not any(row.get('label_mask', {}).get(label, False) for label in config['labels']):
                    continue
                for value in row['image_paths']:
                    group = sample_groups.setdefault((split, row['dataset']), [])
                    if value not in references:
                        source = resolve_data_path(value, data_root)
                        item = entries.get(value)
                        if not isinstance(item, dict):
                            raise ValueError('Cache does not cover a selected supervised source image')
                        source_stat = source.stat()
                        if item.get('source_size') != source_stat.st_size or item.get('source_mtime_ns') != source_stat.st_mtime_ns:
                            raise ValueError('Cache source fingerprint differs from the selected source image')
                        cached_value = item.get('cache_path')
                        if not isinstance(cached_value, str) or not cached_value:
                            raise ValueError('Cache entry requires a local cache_path')
                        cached = (root / cached_value).resolve()
                        if not cached.is_relative_to(root) or not cached.is_file() or cached.suffix.lower() != '.png':
                            raise ValueError('Cache PNG path is absent or leaves the cache directory')
                        checksum = item.get('pixel_sha256')
                        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
                            raise ValueError('Cache entry requires an RGB pixel SHA256')
                        references[value] = (source, cached, checksum)
                    if len(group) < 2 and value not in group:
                        group.append(value)
    checked = set()
    for group in sample_groups.values():
        for value in group:
            if value in checked:
                continue
            source, cached, checksum = references[value]
            with Image.open(source) as original:
                expected = apply_preprocessing(original, expected_size, specification).tobytes()
            with Image.open(cached) as result:
                if result.format != 'PNG' or result.mode != 'RGB' or result.size != expected_size:
                    raise ValueError('Cache pixels have an invalid format, channel mode or size')
                pixels = result.tobytes()
            if pixels != expected or hashlib.sha256(pixels).hexdigest() != checksum:
                raise ValueError('Cache RGB pixels differ from the configured source preprocessing with PIL BILINEAR')
            checked.add(value)
    if not references:
        raise ValueError('Cache validation found no supervised train/val images')
    return {'enabled': True, 'cache_identity': cache_identity, 'referenced_images_checked': len(references),
            'cache_pixel_pairs_checked': len(checked),
            'scope': 'All selected train/val reference paths and source size/mtime checked; two deterministic pixel pairs per source/split; builder attests all entries.'}


def stop_requested_after_epoch(output, completed_epoch, rank, world, device):
    """Consume an explicit request only after a durable epoch checkpoint."""
    stop, acknowledgment, failure = False, None, None
    if rank == 0:
        request = Path(output) / 'STOP_AFTER_EPOCH'
        if request.is_file():
            acknowledgment = request.with_name('STOP_AFTER_EPOCH.ack.epoch_' + str(completed_epoch) + '.' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            try:
                os.replace(request, acknowledgment)
                stop = True
            except OSError as error:
                failure = type(error).__name__
    flag = torch.tensor(-1 if failure else int(stop), device=device, dtype=torch.int32)
    if world > 1:
        dist.broadcast(flag, src=0)
    if flag.item() < 0:
        raise RuntimeError('Cannot acknowledge STOP_AFTER_EPOCH request; request was preserved')
    stop = bool(flag.item())
    if stop and rank == 0:
        print(json.dumps({'event': 'stop_after_epoch_acknowledged', 'epoch': completed_epoch,
                          'acknowledgment': str(acknowledgment)}, ensure_ascii=False), flush=True)
    return stop


def validate_config(c):
    for key in ('task', 'kind', 'labels', 'model', 'training'):
        if key not in c:
            raise ValueError(f'Configuration requires {key}')
    if c['kind'] not in ('multiclass', 'multilabel', 'segmentation', 'landmark', 'detection'):
        raise ValueError('Unsupported kind')
    if not c['labels'] or len(c['labels']) != len(set(c['labels'])):
        raise ValueError('labels must be a nonempty ordered list without duplicates')
    t = c['training']
    for k in ('epochs', 'batch_size', 'grad_accum_steps'):
        if int(t.get(k, 1)) < 1:
            raise ValueError(f'training.{k} must be positive')
    if t.get('amp', 'fp16') not in ('fp16', 'bf16', 'none'):
        raise ValueError('amp must be fp16, bf16 or none')
    differential_learning_rates(t)
    validate_initialization_config(c)
    from training.fundus_preprocessing import validate_preprocessing_config
    validate_preprocessing_config(c)


def validate_initialization_config(config):
    """Validate the explicit, pinned model-only transfer contract, without I/O."""
    specification = config.get('initialization')
    if specification is None:
        return None
    if not isinstance(specification, dict):
        raise ValueError('initialization must be an object')
    allowed = {'mode', 'checkpoint', 'checkpoint_sha256', 'parent_training_sources', 'evaluate_before_training'}
    if set(specification) != allowed or specification.get('mode') != 'weights_only':
        raise ValueError('initialization requires the reviewed weights_only fields exactly')
    path = specification.get('checkpoint')
    if not isinstance(path, str) or not path.strip() or '://' in path:
        raise ValueError('initialization.checkpoint must be a nonempty local path')
    checksum = specification.get('checkpoint_sha256')
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
        raise ValueError('initialization.checkpoint_sha256 must be a lowercase SHA256 digest')
    sources = specification.get('parent_training_sources')
    if (not isinstance(sources, list) or not sources or any(not isinstance(s, str) or not s for s in sources)
            or len(sources) != len(set(sources))):
        raise ValueError('initialization.parent_training_sources must name distinct nonempty sources')
    if specification.get('evaluate_before_training') is not True:
        raise ValueError('weights_only initialization requires complete validation before training')
    return specification


def should_apply_weights_only_initialization(config, resume=False, eval_only=False):
    # Resuming/evaluating this run must not require the old parent file to
    # remain present or accidentally replace already-trained model parameters.
    return config.get('initialization') is not None and not resume and not eval_only


def apply_weights_only_initialization(model, config, resume=False, eval_only=False):
    """Import only model tensors from a SHA-pinned trusted training artifact.

    Optimizer/scheduler/scaler, epoch/best metric and parent RNG are never read
    into the new run. Full checkpoint deserialization is needed because older
    trusted artifacts contain Python RNG objects; verify the same open file's
    complete digest before deserialization and check metadata before copying.
    """
    if not should_apply_weights_only_initialization(config, resume, eval_only):
        return None
    specification = validate_initialization_config(config)
    path = Path(specification['checkpoint']).expanduser()
    if not path.is_absolute():
        path = PROJECT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Initialization checkpoint is missing: {path}')
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        checksum = hashlib.sha256()
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            checksum.update(block)
        actual_digest = checksum.hexdigest()
        if actual_digest != specification['checkpoint_sha256']:
            raise ValueError('Initialization checkpoint SHA256 mismatch; no parent weights were loaded')
        handle.seek(0)
        parent = torch.load(handle, map_location='cpu', weights_only=False)
        after = os.fstat(handle.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Initialization checkpoint changed while loading')
    if not isinstance(parent, dict) or not isinstance(parent.get('config'), dict):
        raise ValueError('Initialization checkpoint lacks its parent configuration')
    saved = parent['config']
    for key in ('task', 'kind', 'labels', 'num_classes', 'model'):
        if saved.get(key) != config.get(key):
            raise ValueError(f'Initialization parent {key} differs from the new run')
    actual_sources = list(saved.get('data', {}).get('datasets') or [])
    inherited = parent.get('initialization_lineage') or {}
    actual_sources.extend(inherited.get('supervised_training_source_history', []))
    actual_sources.extend((saved.get('initialization') or {}).get('parent_training_sources', []))
    actual_sources = list(dict.fromkeys(actual_sources))
    if not actual_sources or set(actual_sources) != set(specification['parent_training_sources']):
        raise ValueError('Declared parent_training_sources differ from checkpoint supervision history')
    state = parent.get('model')
    raw = unwrap(model)
    expected = raw.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('Initialization model state keys do not strictly match the configured architecture')
    if any(not torch.is_tensor(state[key]) or state[key].shape != value.shape for key, value in expected.items()):
        raise ValueError('Initialization model tensor shapes do not strictly match the configured architecture')
    raw.load_state_dict(state, strict=True)
    current_sources = list(config.get('data', {}).get('datasets') or [])
    history = list(dict.fromkeys(specification['parent_training_sources'] + current_sources))
    return {'mode': 'weights_only', 'checkpoint': str(path), 'checkpoint_sha256': actual_digest,
            'parent_completed_epoch': int(parent['epoch']) + 1 if isinstance(parent.get('epoch'), int) else None,
            'parent_task': saved['task'], 'parent_training_sources': specification['parent_training_sources'],
            'new_run_training_sources': current_sources, 'supervised_training_source_history': history,
            'independent_source_only_model': len(set(history)) == 1,
            'starting_epoch_zero_based': 0,
            'reset_components': ['optimizer', 'scheduler', 'scaler', 'epoch', 'best_metric', 'RNG'],
            'optimizer_state_imported': False, 'parent_rng_restored': False,
            'explanation': 'The new manifests may be single-source while supervised model history spans the declared parent and new sources. Only model tensors are transferred.'}


def differential_learning_rates(training):
    """Resolve optional backbone/head rates, preserving the legacy single group.

    A missing member of the pair falls back to training.lr. Explicit override
    values must be positive finite JSON numbers, rather than strings or bools.
    """
    keys = ('backbone_lr', 'head_lr')
    if not any(key in training for key in keys):
        return None
    for key in keys:
        if key in training:
            value = training[key]
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f'training.{key} must be a positive finite number')
    fallback = float(training.get('lr', 1e-4))
    result = {key: float(training.get(key, fallback)) for key in keys}
    for key, value in result.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'Effective training.{key} must be positive and finite; check training.lr fallback')
    return result


def build_optimizer(model, training):
    """Keep all parameters in AdamW, including currently frozen future stages."""
    rates = differential_learning_rates(training)
    options = {'lr': float(training.get('lr', 1e-4)),
               'weight_decay': float(training.get('weight_decay', 0.05))}
    if rates is None:
        # Identical parameter ordering/group structure to legacy checkpoints.
        return torch.optim.AdamW(model.parameters(), **options)
    raw = unwrap(model)
    if not hasattr(raw, 'backbone') or not isinstance(raw.backbone, torch.nn.Module):
        raise ValueError('Differential learning rates require a model.backbone module')
    parameters = list(raw.parameters())
    backbone_ids = {id(parameter) for parameter in raw.backbone.parameters()}
    backbone = [parameter for parameter in parameters if id(parameter) in backbone_ids]
    head = [parameter for parameter in parameters if id(parameter) not in backbone_ids]
    if not backbone or not head:
        raise ValueError('Differential learning rates require nonempty backbone and remaining head parameter groups')
    return torch.optim.AdamW([
        {'params': backbone, 'lr': rates['backbone_lr'], 'group_name': 'backbone'},
        {'params': head, 'lr': rates['head_lr'], 'group_name': 'head'},
    ], **options)


def optimizer_learning_rates(optimizer):
    return {group.get('group_name', 'all' if len(optimizer.param_groups) == 1 else f'group_{index}'): group['lr']
            for index, group in enumerate(optimizer.param_groups)}


def set_trainable(model, config, epoch):
    """Freeze -> last N transformer blocks -> full tuning, including optimizer state."""
    raw = unwrap(model)
    if not hasattr(raw, 'backbone') or config['kind'] in ('segmentation', 'detection'):
        return 'full'
    t = config['training']
    warm = int(t.get('freeze_backbone_epochs', 0))
    full = int(t.get('full_finetune_epoch', warm))
    last_n = int(t.get('unfreeze_last_n_blocks', 0))
    for p in raw.parameters():
        p.requires_grad_(True)
    if epoch < warm:
        for p in raw.backbone.parameters():
            p.requires_grad_(False)
        return 'head_only'
    if last_n and epoch < full:
        blocks = getattr(raw.backbone, 'blocks', None)
        if blocks is None:
            raise ValueError('unfreeze_last_n_blocks requires a backbone.blocks transformer')
        for p in raw.backbone.parameters():
            p.requires_grad_(False)
        for block in list(blocks)[-last_n:]:
            for p in block.parameters():
                p.requires_grad_(True)
        for name in ('norm', 'fc_norm'):
            module = getattr(raw.backbone, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(True)
        return f'last_{last_n}_blocks'
    return 'full'


def make_loader(adapter, dataset, config, training, rank, world, workers):
    sampler = (DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                  seed=int(config.get('seed', 20261113)), drop_last=False)
               if training and world > 1 else
               EvaluationSampler(dataset, rank, world) if world > 1 else None)
    generator = torch.Generator().manual_seed(int(config.get('seed', 20261113)) + rank)
    options = dict(batch_size=int(config['training']['batch_size']), shuffle=training and sampler is None,
                   sampler=sampler, num_workers=workers, collate_fn=adapter.collate,
                   pin_memory=torch.cuda.is_available(), drop_last=False,
                   worker_init_fn=seed_worker, generator=generator)
    if workers:
        options.update(persistent_workers=False, prefetch_factor=2)
    return DataLoader(dataset, **options)


def gather(value, world):
    if world == 1:
        return [value]
    result = [None] * world
    dist.all_gather_object(result, value)
    return result


def amp_context(device, precision):
    if device.type != 'cuda' or precision == 'none':
        return nullcontext()
    return torch.autocast('cuda', dtype=torch.float16 if precision == 'fp16' else torch.bfloat16)


def evaluation(adapter, model, loader, device, config, world, precision, max_steps=None):
    # Use unwrapped model: validation ranks have unequal batch counts, so DDP
    # buffer broadcasts would deadlock on the last batches.
    raw = unwrap(model)
    raw.eval()
    outputs = []
    loss_sum, loss_weight = 0.0, 0.0
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if max_steps is not None and i >= max_steps:
                break
            with amp_context(device, precision):
                loss, result = adapter.step(raw, batch, device, config, training=False)
            if result.get('validation_loss_available', True) and not torch.isfinite(loss).item():
                raise FloatingPointError('Nonfinite validation loss; the run cannot select a valid best checkpoint')
            weight = (float(result.get('loss_weight', len(batch.get('sample_ids', []))))
                      if result.get('validation_loss_available', True) else 0.0)
            if weight and torch.isfinite(loss):
                loss_sum += float(loss) * weight
                loss_weight += weight
            outputs.append(result)
    shards = gather({'outputs': outputs, 'sum': loss_sum, 'weight': loss_weight}, world)
    flat = [entry for shard in shards for entry in shard['outputs']]
    metrics = adapter.metrics(flat, config)
    total_weight = sum(s['weight'] for s in shards)
    metrics['loss'] = sum(s['sum'] for s in shards) / total_weight if total_weight else None
    return jsonable(metrics), flat


def evaluation_runtime_config(saved_config, config, use_source_images=False):
    """Check the original checkpoint configuration before any input override."""
    for key in ('task', 'kind', 'labels', 'model', 'data'):
        if saved_config.get(key) != config.get(key):
            raise ValueError(f'Evaluation configuration {key} differs from checkpoint')
    from training.fundus_preprocessing import preprocessing_spec
    if preprocessing_spec(saved_config) != preprocessing_spec(config):
        raise ValueError('Evaluation preprocessing differs from checkpoint')
    runtime_config = deepcopy(config)
    removed_cache = None
    if use_source_images:
        if config['kind'] not in ('multiclass', 'multilabel'):
            raise ValueError('--eval-use-source-images is restricted to classification evaluation')
        removed_cache = runtime_config.setdefault('data', {}).pop('image_cache_dir', None)
    return runtime_config, {
        'eval_use_source_images': bool(use_source_images),
        'removed_image_cache_dir': removed_cache,
        'checkpoint_configuration_checked_before_override': True,
        'deterministic_preprocessing': preprocessing_spec(config),
        'preprocessing': ('Source RGB images with deterministic configured resize and normalization; training=False. '
                          'Only the cache location is removed; model, labels, source filters and protocol remain fixed.')
                         if use_source_images else 'Configured evaluation inputs without a runtime cache override.'}


def require_dr_prediction_export(config):
    if (config.get('task') != 'dr_grading' or config.get('kind') != 'multiclass'
            or config.get('labels') != ['dr_grade'] or config.get('num_classes') != 5):
        raise ValueError('--save-predictions currently supports only DR five-class multiclass evaluation with labels=[dr_grade]')


def save_dr_predictions(path, outputs, config, expected_count, metadata=None):
    """Write complete, unique per-sample DR results after checking all batches."""
    require_dr_prediction_export(config)
    rows, seen, source_counts = [], set(), {}
    classes = int(config['num_classes'])
    for output in outputs:
        sample_ids = output.get('sample_ids', [])
        datasets = output.get('datasets', [])
        groups = output.get('content_group_ids', [])
        targets = np.asarray(output.get('targets'))
        masks = np.asarray(output.get('mask'))
        probabilities = np.asarray(output.get('probabilities'), dtype=np.float64)
        weights = np.asarray(output.get('evaluation_weights'), dtype=np.float64)
        n = len(sample_ids)
        if (len(datasets) != n or len(groups) != n or targets.shape != (n,) or masks.shape != (n,)
                or probabilities.shape != (n, classes) or weights.shape != (n,)):
            raise ValueError('Prediction batch fields have inconsistent lengths or shapes')
        if not np.all(masks == True):
            raise ValueError('DR prediction export requires a known target for every evaluated sample')
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)):
            raise ValueError('Prediction probabilities must be finite, nonnegative five-class distributions')
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Prediction evaluation weights must be finite and nonnegative')
        for i, sample_id in enumerate(sample_ids):
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ValueError('Prediction sample IDs must be nonempty and unique across all evaluation ranks')
            if not isinstance(datasets[i], str) or not datasets[i] or not isinstance(groups[i], str) or not groups[i]:
                raise ValueError('Prediction source and content-group IDs must be nonempty strings')
            target = targets[i]
            if not np.isfinite(target) or int(target) != target or not 0 <= int(target) < classes:
                raise ValueError('Prediction target is outside DR class range 0..4')
            seen.add(sample_id)
            source_counts[datasets[i]] = source_counts.get(datasets[i], 0) + 1
            rows.append({'sample_id': sample_id, 'dataset': datasets[i], 'target': int(target),
                         'predicted_class': int(probabilities[i].argmax()),
                         'probabilities': probabilities[i].tolist(), 'content_group_id': groups[i],
                         'evaluation_weight': float(weights[i])})
    if len(rows) != expected_count or expected_count <= 0:
        raise ValueError(f'Prediction export count {len(rows)} does not match complete evaluation dataset count {expected_count}')
    rows.sort(key=lambda row: row['sample_id'])
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    digest = hashlib.sha256()
    with temporary.open('wb') as handle:
        for row in rows:
            payload = (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')
            handle.write(payload)
            digest.update(payload)
    os.replace(temporary, path)
    meta_path = path.with_suffix('.meta.json')
    meta = {**(metadata or {}), 'format_version': 1, 'sample_count': len(rows),
            'expected_dataset_count': expected_count, 'unique_sample_ids': len(seen),
            'all_targets_known': True, 'complete_dataset_evaluated': True,
            'target_label': 'dr_grade', 'class_order': list(range(classes)),
            'ordering': 'sample_id_lexicographic', 'source_counts': source_counts,
            'jsonl_sha256': digest.hexdigest(),
            'fields': ['sample_id', 'dataset', 'target', 'predicted_class', 'probabilities', 'content_group_id', 'evaluation_weight']}
    write_json(meta_path, meta)
    return {'path': str(path), 'metadata_path': str(meta_path), 'sample_count': len(rows),
            'unique_sample_ids': len(seen), 'sha256': digest.hexdigest()}


def metric_value(metrics, path):
    value = metrics
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f'Best-checkpoint metric {path!r} is absent; available: {list(metrics)}')
        value = value[key]
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f'Best-checkpoint metric {path!r} is undefined on this validation set')
    return float(value)


def save_checkpoint(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=PROJECT / 'data')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--grad-accum-steps', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-use-source-images', action='store_true',
                        help='Evaluation only: verify checkpoint config first, then bypass a train/val-only image cache and read deterministic source images')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Evaluation only: export complete unique per-sample DR five-class predictions and metadata')
    parser.add_argument('--split', choices=['val', 'test', 'external_test'], default='val')
    parser.add_argument('--smoke-test', action='store_true', help='Tiny model, 1 epoch, 2 batches; does not measure model quality')
    parser.add_argument('--max-train-steps', type=int)
    parser.add_argument('--max-val-steps', type=int)
    parser.add_argument('--stop-after-epoch', type=int, help='Stop after this completed epoch without changing the configured LR schedule; resumable')
    parser.add_argument('--allow-resize-before-augmentation-change', action='store_true',
                        help='Explicit classification-only preprocessing transition from a parent checkpoint into a NEW output directory; other experiment settings stay fixed')
    args = parser.parse_args()
    if (args.eval_use_source_images or args.save_predictions) and not args.eval_only:
        parser.error('--eval-use-source-images and --save-predictions require --eval-only')
    if args.save_predictions and args.max_val_steps is not None:
        parser.error('--save-predictions requires the complete evaluation dataset without --max-val-steps')
    if args.allow_resize_before_augmentation_change and (not args.resume or args.eval_only or args.smoke_test or args.output is None):
        parser.error('--allow-resize-before-augmentation-change requires --resume and an explicit new --output, and cannot be combined with evaluation or smoke mode')
    if args.allow_resize_before_augmentation_change and (args.max_train_steps is not None or args.max_val_steps is not None):
        parser.error('A preprocessing transition must use the complete train/val loaders, without step limits')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    validate_config(config)
    if args.smoke_test and config.get('initialization') is not None:
        parser.error('--smoke-test changes architecture and cannot apply a real weights_only initialization')
    if args.save_predictions:
        require_dr_prediction_export(config)
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        parser.error('--stop-after-epoch must be positive')
    for arg, key in ((args.epochs, 'epochs'), (args.batch_size, 'batch_size'), (args.grad_accum_steps, 'grad_accum_steps')):
        if arg is not None:
            config['training'][key] = arg
    if args.smoke_test:
        if args.resume or args.eval_only:
            parser.error('--smoke-test is a separate fresh training run')
        config['model'].update(pretrained=False, img_size=64, grad_checkpointing=False)
        if config['kind'] in ('multiclass', 'multilabel', 'landmark'):
            config['model']['name'] = 'tiny_cnn'
        elif config['kind'] == 'segmentation':
            config['model']['base_channels'] = 4
        config.setdefault('augmentation', {})['patch_size'] = None
        config['training'].update(epochs=1, batch_size=2, grad_accum_steps=2, workers=0,
                                  freeze_backbone_epochs=0, unfreeze_last_n_blocks=0, full_finetune_epoch=0)
        if config['kind'] != 'detection':
            config['training'].update(best_metric='loss', best_mode='min')
        args.max_train_steps = 2
        args.max_val_steps = 2
    validate_config(config)
    if args.workers is not None and args.workers < 0:
        parser.error('--workers must be nonnegative')
    for x in (args.max_train_steps, args.max_val_steps):
        if x is not None and x < 1:
            parser.error('step limits must be positive')
    if not args.eval_only and not args.smoke_test:
        from scripts.dataset_routing import validate_training_route
        validate_training_route(config, args.data_root.resolve())
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    use_cuda = args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment. Install the CUDA PyTorch wheel or explicitly use --device cpu for debugging.')
    device = torch.device(f'cuda:{local_rank}' if use_cuda else 'cpu')
    if use_cuda:
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if use_cuda else 'gloo')
    seed = int(config.get('seed', 20261113))
    seed_all(seed + rank)
    precision = config['training'].get('amp', 'fp16') if use_cuda else 'none'
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support BF16; V100 configurations use FP16.')
    if not use_cuda:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    data_root = args.data_root.resolve()
    adapter = importlib.import_module('training.classification' if config['kind'] in ('multiclass', 'multilabel') else 'training.structural')
    output = (args.output or PROJECT / 'runs' / (config['task'] + ('_smoke' if args.smoke_test else ''))).resolve()
    if args.resume and not args.eval_only:
        validate_resume_output(output, args.resume, args.allow_resize_before_augmentation_change)
    if not args.eval_only and (output / 'last.pt').exists() and not args.resume:
        raise FileExistsError(f'{output}/last.pt exists. Use --resume or a new --output directory.')
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    workers = int(args.workers if args.workers is not None else config['training'].get('workers', 4))
    if args.smoke_test:
        workers = 0
    checkpoint = None
    checkpoint_path = args.resume or args.checkpoint
    if args.eval_only and not checkpoint_path:
        raise ValueError('--eval-only requires --checkpoint (or --resume)')
    if args.checkpoint and not args.eval_only:
        raise ValueError('--checkpoint is for --eval-only; use --resume to continue training')
    if checkpoint_path:
        # Checkpoints contain optimizer/RNG Python objects. Load only your own
        # trusted local training output; never download arbitrary .pt files.
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    apply_initialization = should_apply_weights_only_initialization(config, args.resume is not None, args.eval_only)
    model = adapter.build_model(config, pretrained=checkpoint is None and not apply_initialization).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    initialization_lineage = apply_weights_only_initialization(model, config, args.resume is not None, args.eval_only)
    if initialization_lineage is None and checkpoint:
        initialization_lineage = checkpoint.get('initialization_lineage')
    if args.eval_only:
        saved = checkpoint.get('config', {})
        dataset_config, runtime_overrides = evaluation_runtime_config(saved, config, args.eval_use_source_images)
        from scripts.dataset_routing import config_for_split
        dataset_config = config_for_split(dataset_config, args.split)
        path = manifest_path(config, data_root, args.split)
        dataset = adapter.build_dataset(path, data_root, dataset_config, training=False)
        if not len(dataset):
            raise ValueError(f'No supervised samples for {args.split}: {path}')
        loader = make_loader(adapter, dataset, config, False, rank, world, workers)
        result, flat = evaluation(adapter, model, loader, device, config, world, precision, args.max_val_steps)
        if rank == 0:
            report = {'split': args.split, 'checkpoint': str(checkpoint_path), 'config': config,
                      'runtime_overrides': runtime_overrides, 'evaluated_dataset_count': len(dataset), 'metrics': result}
            if args.save_predictions:
                report['predictions'] = save_dr_predictions(output / f'predictions_{args.split}.jsonl', flat, config, len(dataset),
                                                           {'created_utc': datetime.now(timezone.utc).isoformat(),
                                                            'split': args.split, 'checkpoint': str(checkpoint_path.resolve()),
                                                            'checkpoint_completed_epoch': int(checkpoint['epoch']) + 1 if 'epoch' in checkpoint else None,
                                                            'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                                            'runtime_overrides': runtime_overrides})
            write_json(output / f'evaluation_{args.split}.json', report)
            print(json.dumps(jsonable(result), ensure_ascii=False), flush=True)
        if world > 1:
            dist.destroy_process_group()
        return
    identity = data_identity(config, data_root)
    transition = None
    if checkpoint:
        if args.allow_resize_before_augmentation_change:
            changes = validate_performance_transition(checkpoint.get('data_identity'), identity)
            cache_checks = validate_transition_cache(config, data_root)
            parent_best = checkpoint_path.resolve().parent / 'best.pt'
            transition = {'kind': 'classification_resize_before_augmentation',
                          'created_utc': datetime.now(timezone.utc).isoformat(),
                          'parent_checkpoint': str(checkpoint_path.resolve()),
                          'parent_run': str(checkpoint_path.resolve().parent),
                          'parent_best_checkpoint': str(parent_best) if parent_best.is_file() else None,
                          'parent_best_metric': checkpoint.get('best_metric'),
                          'parent_completed_epoch': int(checkpoint['epoch']) + 1,
                          'continuation_start_epoch': int(checkpoint['epoch']) + 2,
                          'old_identity': checkpoint['data_identity'], 'new_identity': identity,
                          'allowed_changes': changes, 'cache_verification': cache_checks,
                          'bitwise_continuation': False,
                          'explanation': 'Training augmentation now operates after a fixed BILINEAR resize. This is an explicit preprocessing change, not bitwise reproduction. Model, optimizer, scheduler, scaler, epoch and per-rank RNG state continue; batch layout, LR schedule, labels and manifests remain fixed. A verified lossless RGB PNG cache preserves validation resize pixels.',
                          'branch_best_policy': 'Select best from this branch only; parent best checkpoint and score remain in the original run.'}
        elif checkpoint.get('data_identity') != identity:
            raise ValueError('Resume task/model/labels/protocol or train/val manifests differ from the saved run')
    if checkpoint and checkpoint.get('world_size', world) != world:
        raise ValueError('Exact resume requires the same number of GPUs; use a fresh run to change world size')
    train_ds = adapter.build_dataset(manifest_path(config, data_root, 'train'), data_root, config, training=True)
    val_ds = adapter.build_dataset(manifest_path(config, data_root, 'val'), data_root, config, training=False)
    if not len(train_ds) or not len(val_ds):
        raise ValueError('Training and validation must both have supervised samples. Check the selected protocol and source filters.')
    if args.smoke_test:
        train_ds = Subset(train_ds, list(range(min(4, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(4, len(val_ds)))))
    train_loader = make_loader(adapter, train_ds, config, True, rank, world, workers)
    val_loader = make_loader(adapter, val_ds, config, False, rank, world, workers)
    t = config['training']
    epochs = int(t['epochs'])
    optimizer = build_optimizer(model, t)
    warmup = int(t.get('warmup_epochs', 1))
    def lr_factor(epoch):
        if warmup and epoch < warmup:
            return max(0.01, (epoch + 1) / warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * min(1, progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda and precision == 'fp16')
    start_epoch, best = 0, None
    if checkpoint:
        # Changing total epochs changes the LR trajectory; reject silent drift.
        saved_t = checkpoint['config']['training']
        for key in ('epochs', 'batch_size', 'grad_accum_steps', 'lr', 'backbone_lr', 'head_lr', 'weight_decay', 'warmup_epochs', 'freeze_backbone_epochs', 'unfreeze_last_n_blocks', 'full_finetune_epoch'):
            if saved_t.get(key) != t.get(key):
                raise ValueError(f'Resume training.{key} differs from saved configuration')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = int(checkpoint['epoch']) + 1
        best = None if transition else checkpoint['best_metric']
        states = checkpoint.get('rng_states')
        if states:
            restore_rng(states[rank])
    runtime = {'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
               'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': str(device),
               'gpu': torch.cuda.get_device_name(device) if use_cuda else None,
               'world_size': world, 'train_records': len(train_ds), 'val_records': len(val_ds),
               'effective_batch_size': int(t['batch_size']) * int(t['grad_accum_steps']) * world,
               'workers_per_rank': workers, 'preprocessing_transition': bool(transition),
               'smoke_test': args.smoke_test, 'metrics_scope': 'sample-level; adapter may additionally report content-weighted metrics'}
    if rank == 0:
        if transition:
            write_json(output / 'transition.json', transition)
        write_json(output / 'config.resolved.json', config)
        write_json(output / 'runtime.json', runtime)
        write_json(output / 'data_identity.json', identity)
        print(json.dumps(runtime, ensure_ascii=False), flush=True)
    if apply_initialization:
        # This is a fixed initial-state baseline, not checkpoint/threshold
        # selection. Ignore debug validation step limits for this audit.
        initial_metrics, _ = evaluation(adapter, model, val_loader, device, config, world, precision, max_steps=None)
        observed = initial_metrics.get('n', initial_metrics.get('samples'))
        if observed != len(val_ds):
            raise ValueError('Initialization baseline did not evaluate the complete validation dataset')
        if rank == 0:
            write_json(output / 'initialization_lineage.json', initialization_lineage)
            write_json(output / 'initialization_validation.json',
                       {'split': 'val', 'completed_training_epochs': 0, 'optimizer_updates': 0,
                        'complete_dataset_evaluated': True, 'evaluated_dataset_count': len(val_ds),
                        'initialization': initialization_lineage, 'metrics': initial_metrics,
                        'selection_policy': 'Report initial validation separately; trained best.pt is selected from new-run epochs only. Preserve the parent checkpoint for rollback.'})
            print(json.dumps({'event': 'initialization_validation_complete', 'split': 'val',
                              'n': len(val_ds), 'optimizer_updates': 0, 'metrics': initial_metrics}, ensure_ascii=False), flush=True)
        if world > 1:
            dist.barrier()
    best_name = t.get('best_metric', 'loss')
    best_mode = t.get('best_mode', 'min' if best_name == 'loss' else 'max')
    if best_mode not in ('min', 'max'):
        raise ValueError('best_mode must be min or max')
    accum = int(t.get('grad_accum_steps', 1))
    previous_stage = None
    for epoch in range(start_epoch, epochs):
        # Seed by epoch makes DataLoader worker augmentation reproducible on
        # epoch-boundary resume; sampler shuffle has its own fixed epoch seed.
        seed_all(seed + epoch * 100003 + rank)
        train_loader.generator.manual_seed(seed + epoch * 100003 + rank)
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        raw = unwrap(model)
        stage = set_trainable(raw, config, epoch)
        if world > 1 and (not isinstance(model, DDP) or stage != previous_stage):
            # Frozen parameters are omitted by DDP's reducer. Reconstruct it
            # only when the set of trainable parameters changes.
            if isinstance(model, DDP):
                old_wrapper = model
                model = raw
                del old_wrapper
                gc.collect()
            dist.barrier()
            model = DDP(raw, device_ids=[local_rank] if use_cuda else None, broadcast_buffers=False)
        previous_stage = stage
        model.train()
        if stage == 'head_only':
            raw.backbone.eval()
        steps = min(len(train_loader), args.max_train_steps or len(train_loader))
        optimizer.zero_grad(set_to_none=True)
        train_sum, train_weight = 0.0, 0.0
        started = time.monotonic()
        for i, batch in enumerate(train_loader):
            if i >= steps:
                break
            window_size = min(accum, steps - (i // accum) * accum)
            boundary = (i + 1) % accum == 0 or i + 1 == steps
            sync = model.no_sync() if world > 1 and not boundary else nullcontext()
            with sync:
                with amp_context(device, precision):
                    loss, result = adapter.step(model, batch, device, config, training=True)
                finite = torch.tensor(int(torch.isfinite(loss).item()), device=device)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError(f'Nonfinite loss at epoch {epoch + 1}, step {i + 1}')
                scaler.scale(loss / window_size).backward()
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(t.get('clip_grad_norm', 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            # Training losses already use a globally reduced known-target
            # denominator. Rank-average them once for logging.
            weight = 1.0
            train_sum += float(loss.detach()) * weight
            train_weight += weight
            if rank == 0 and (i == 0 or (i + 1) % int(t.get('log_every', 25)) == 0 or i + 1 == steps):
                print(json.dumps({'epoch': epoch + 1, 'step': i + 1, 'steps': steps,
                                  'loss': round(float(loss.detach()), 6), 'stage': stage}, ensure_ascii=False), flush=True)
        sums = gather((train_sum, train_weight), world)
        train_loss = sum(v[0] for v in sums) / max(1, sum(v[1] for v in sums))
        val_metrics, _ = evaluation(adapter, model, val_loader, device, config, world, precision, args.max_val_steps)
        score = metric_value(val_metrics, best_name)
        improved = best is None or (score > best if best_mode == 'max' else score < best)
        if improved:
            best = score
        scheduler.step()
        states = gather(rng_state(), world)
        if rank == 0:
            record = {'epoch': epoch + 1, 'stage': stage, 'train_loss': train_loss, 'val': val_metrics,
                      'next_lr': optimizer.param_groups[0]['lr'], 'next_lr_by_group': optimizer_learning_rates(optimizer),
                      'elapsed_seconds': round(time.monotonic() - started, 2),
                      'best_metric': best, 'best_metric_name': best_name, 'improved': improved}
            state = {'epoch': epoch, 'model': unwrap(model).state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'config': config,
                     'best_metric': best, 'data_identity': identity, 'world_size': world, 'rng_states': states}
            if initialization_lineage is not None:
                state['initialization_lineage'] = initialization_lineage
            lineage = transition or (checkpoint.get('transition') if checkpoint else None)
            if lineage:
                state['transition'] = lineage
            checkpoint_started = time.monotonic()
            save_checkpoint(output / 'last.pt', state)
            if improved:
                from training.checkpoint_io import publish_checkpoint_snapshot
                record['best_checkpoint_publish_mode'] = publish_checkpoint_snapshot(output / 'last.pt', output / 'best.pt')
            record['checkpoint_io_seconds'] = round(time.monotonic() - checkpoint_started, 2)
            record['epoch_wall_seconds'] = round(time.monotonic() - started, 2)
            with (output / 'history.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(jsonable(record), ensure_ascii=False, allow_nan=False) + '\n')
            print(json.dumps(jsonable(record), ensure_ascii=False, allow_nan=False), flush=True)
        if world > 1:
            dist.barrier()
        requested_stop = stop_requested_after_epoch(output, epoch + 1, rank, world, device)
        if requested_stop or (args.stop_after_epoch is not None and epoch + 1 >= args.stop_after_epoch):
            break
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
