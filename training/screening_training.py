"""Run and resume the two versioned retinal screening experiments on one GPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch
import timm
from timm.models.vision_transformer import checkpoint_filter_fn
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.manifest_dataset import resolve_data_path
from training.classification import ClassificationTransform
from training.metrics import binary_metrics
from training.train import amp_context, restore_rng, rng_state, save_checkpoint, seed_all, seed_worker, write_json


OFFICIAL_SHA = '0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'
BACKBONE = 'vit_base_patch14_dinov2.lvd142m'
LABELS = {
    'grouped': ['RVO', 'LS', 'DN', 'AMD', 'CSR', 'OPTIC_DISC_CUPPING', 'ODE', 'ODP',
                'MYOPIC_FUNDUS', 'TSLN', 'ABNORMAL'],
    'glaucoma': ['glaucoma'],
}
SOURCES = {
    'grouped': {'train': {'RFMiD'}, 'val': {'RFMiD'}, 'test': {'RFMiD'}, 'external_test': {'RFMiD2'}},
    'glaucoma': {'train': {'REFUGE', 'PAPILA'}, 'val': {'PAPILA'}, 'test': {'PAPILA'}, 'external_test': {'FIVES'}},
}
TRANSFORM_CONFIG = {
    'model': {'name': BACKBONE, 'img_size': 392, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    'preprocessing': {'mode': 'stretch'},
    'augmentation': {'horizontal_flip': 0.5, 'rotation_degrees': 10, 'color_jitter': 0.1,
                     'resize_before_augmentation': False},
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def backbone_hash(backbone):
    digest = hashlib.sha256()
    for name, tensor in backbone.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ScreeningDataset(Dataset):
    def __init__(self, args, audit, task, split):
        key = f'{task}/{split}'
        path = args.release / (key + '.jsonl')
        self.manifest_sha = sha256(path)
        if audit['manifests'][key]['sha256'] != self.manifest_sha:
            raise ValueError(f'Manifest fingerprint mismatch for {key}')
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(records) != audit['manifests'][key]['records']:
            raise ValueError(f'Unexpected record count for {key}')
        sources = {r['dataset'] for r in records}
        if sources != SOURCES[task][split]:
            raise ValueError(f'Unexpected sources for {key}: {sources}')
        n = len(LABELS[task])
        for row in records:
            if (row['split'] != split or len(row['labels']) != n or len(row['mask']) != n
                    or any(type(v) is not bool for v in row['mask'])
                    or any(v not in (0, 1) for v in row['labels'])):
                raise ValueError(f"Invalid labels, mask or split for {row['sample_id']}")
        self.rows = [r for r in records if split != 'train' or any(r['mask'])]
        self.paths = [resolve_data_path(r['image_path'], args.data_root) for r in self.rows]
        self.transform = ClassificationTransform(TRANSFORM_CONFIG, training=(split == 'train'))
        self.cache_dir = None
        if args.rgb_cache_dir:
            self.cache_dir = args.rgb_cache_dir / ('native_rgb_pillow_' + pillow_version)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.rows)

    def source_rgb(self, index):
        if self.cache_dir:
            sha = self.rows[index]['image_sha256']
            if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                raise ValueError('Invalid image fingerprint in the release manifest')
            target = self.cache_dir / (sha + '.npy')
            if target.is_file():
                return Image.fromarray(np.load(target, mmap_mode='r', allow_pickle=False))
        with Image.open(self.paths[index]) as image:
            raw = image.convert('RGB')
        if self.cache_dir:
            # Concurrent tasks/workers may encounter the same image. Only publish a complete array.
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix='.tmp', delete=False) as out:
                temp = Path(out.name)
                try:
                    np.save(out, np.asarray(raw), allow_pickle=False)
                except BaseException:
                    out.close()
                    temp.unlink(missing_ok=True)
                    raise
            try:
                os.replace(temp, target)
            finally:
                temp.unlink(missing_ok=True)
        return raw

    def __getitem__(self, index):
        r = self.rows[index]
        x = self.transform(self.source_rgb(index))
        return x, torch.tensor(r['labels'], dtype=torch.float32), torch.tensor(r['mask']), index


class ScreeningModel(nn.Module):
    def __init__(self, task, weights):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE, pretrained=False, img_size=392, num_classes=0)
        official = torch.load(weights, map_location='cpu', weights_only=True)
        self.backbone.load_state_dict(checkpoint_filter_fn(official, self.backbone), strict=True)
        self.dropout = nn.Dropout(0.1)
        widths = [2, 3, 3, 2, 1] if task == 'grouped' else [1]
        self.heads = nn.ModuleList(nn.Linear(self.backbone.num_features, w) for w in widths)

    def forward(self, x):
        features = self.dropout(self.backbone(x))
        return torch.cat([head(features) for head in self.heads], dim=1)

    def configure(self, variant, blocks):
        self.backbone.requires_grad_(variant == 'full')
        if variant == 'partial':
            if not 1 <= blocks <= len(self.backbone.blocks):
                raise ValueError('Invalid number of trainable backbone blocks')
            self.backbone.blocks[-blocks:].requires_grad_(True)
            self.backbone.norm.requires_grad_(True)

    def train_mode(self, variant, blocks):
        self.train()
        if variant != 'full':
            self.backbone.eval()
        if variant == 'partial':
            self.backbone.blocks[-blocks:].train()
            self.backbone.norm.train()


def loader(dataset, args, training):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=training, num_workers=args.workers,
                      pin_memory=True, worker_init_fn=seed_worker,
                      **({'prefetch_factor': 1} if args.workers else {}))


@torch.inference_mode()
def evaluate(model, batches, args, labels, collect=False):
    model.eval()
    all_probs, all_y, all_m, predictions = [], [], [], []
    for x, y, mask, indices in batches:
        with amp_context(torch.device(args.device), args.precision):
            logits = model(x.to(args.device, non_blocking=True)).float()
        if not torch.isfinite(logits).all():
            raise ValueError('Evaluation produced nonfinite logits')
        probs = logits.sigmoid().cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.numpy())
        all_m.append(mask.numpy())
        if collect:
            logits = logits.cpu().tolist()
            for j, index in enumerate(indices.tolist()):
                r = batches.dataset.rows[index]
                record = {k: r.get(k) for k in ('sample_id', 'split', 'dataset', 'patient_id', 'content_group_id',
                                                 'image_path', 'labels', 'mask')}
                predictions.append(dict(record, logits=logits[j], probabilities=probs[j].tolist()))
    y, m, p = [np.concatenate(a) for a in (all_y, all_m, all_probs)]
    by_label = {}
    for i, label in enumerate(labels):
        sel = m[:, i]
        by_label[label] = binary_metrics(y[sel, i], p[sel, i])
    aps = [v['auprc'] for v in by_label.values() if v['auprc'] is not None]
    return {'macro_ap': float(np.mean(aps)) if aps else None, 'per_label': by_label}, predictions


def export(model, args, audit, task, split, checkpoint, expected_identity):
    best = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if best['identity'] != expected_identity:
        raise ValueError('Checkpoint provenance does not match this config, release or initialization')
    model.load_state_dict(best['model'], strict=True)
    dataset = ScreeningDataset(args, audit, task, split)
    labels = LABELS[task]
    metrics, records = evaluate(model, loader(dataset, args, False), args, labels, collect=True)
    digest = sha256(checkpoint)
    dst = args.output / f'{split}_predictions.jsonl'
    tmp = dst.with_suffix('.jsonl.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for r in records:
            r['checkpoint_sha256'] = digest
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, dst)
    write_json(dst.with_suffix('.meta.json'), {'labels': labels, 'split': split, 'checkpoint_sha256': digest,
               'manifest_sha256': dataset.manifest_sha, 'epoch': best['epoch'], 'metrics_at_0_5': metrics,
               'precision': args.precision})


def run_epoch(model, batches, args, optimizer, scaler, pos_weight, stage):
    model.configure(stage, args.blocks)
    model.train_mode(stage, args.blocks)
    optimizer.zero_grad(set_to_none=True)
    epoch_sum, epoch_count, group_count = 0., 0, 0
    for step, (x, y, mask, _) in enumerate(batches, 1):
        x, y, mask = [v.to(args.device, non_blocking=True) for v in (x, y, mask)]
        with amp_context(torch.device(args.device), args.precision):
            logits = model(x)
        losses = nn.functional.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pos_weight, reduction='none')
        loss_sum = (losses * mask).sum()
        if not torch.isfinite(loss_sum):
            raise FloatingPointError(f'Nonfinite training loss at batch {step}')
        # Normalize by actual known entries over the complete accumulation window.
        # This also handles the final short batch without giving it extra weight.
        batch_count = int(mask.sum())
        group_count += batch_count
        epoch_sum += float(loss_sum.detach())
        epoch_count += batch_count
        scale_base = args.batch_size * args.accum * y.shape[1]
        scaler.scale(loss_sum / scale_base).backward()
        if step % args.accum == 0 or step == len(batches):
            scaler.unscale_(optimizer)
            params = [p for g in optimizer.param_groups for p in g['params'] if p.grad is not None]
            for p in params:
                p.grad.mul_(scale_base / group_count)
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=not scaler.is_enabled())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            group_count = 0
    return epoch_sum / epoch_count


def train(args, audit, task, model, identity):
    trainset = ScreeningDataset(args, audit, task, 'train')
    valset = ScreeningDataset(args, audit, task, 'val')
    train_batches, val_batches = loader(trainset, args, True), loader(valset, args, False)
    mask = np.asarray([r['mask'] for r in trainset.rows], dtype=bool)
    y = np.asarray([r['labels'] for r in trainset.rows])
    positives, negatives = ((y == 1) & mask).sum(axis=0), ((y == 0) & mask).sum(axis=0)
    if not np.all((positives > 0) & (negatives > 0)):
        raise ValueError('Each trained label needs known positives and negatives')
    pos_weight = torch.tensor(np.minimum(10, np.sqrt(negatives / positives)), device=args.device, dtype=torch.float32)
    model.configure(args.variant, args.blocks)
    groups = [{'params': model.heads.parameters(), 'lr': args.head_lr}]
    trainable = [p for p in model.backbone.parameters() if p.requires_grad]
    if trainable:
        groups.append({'params': trainable, 'lr': args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0001)
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16' and args.device.startswith('cuda'))
    settings = {k: getattr(args, k) for k in ('variant', 'blocks', 'head_only_epochs', 'epochs', 'precision', 'seed', 'head_lr',
                                                'backbone_lr', 'batch_size', 'accum', 'workers')}
    resume_id = dict(identity, settings=settings, engine_sha256=sha256(Path(__file__)))
    best_path, last_path = args.output / 'best.pt', args.output / 'last.pt'
    best_ap, best_epoch, history = -1., 0, []
    initial_hash = backbone_hash(model.backbone) if args.variant == 'frozen' else None
    def publish_best(epoch):
        save_checkpoint(best_path, {'model': model.state_dict(), 'identity': identity, 'labels': LABELS[task],
                                    'epoch': epoch, 'settings': settings, 'best_validation_macro_ap': best_ap})
    if args.resume:
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        if state['identity'] != resume_id:
            raise ValueError('Resume settings, code, data or initialization differ from the saved run')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'])
        best_ap, best_epoch, history = state['best_ap'], state['best_epoch'], state['history']
        if len(history) == best_epoch:
            publish_best(best_epoch)
        del state
    elif any((args.output / name).exists() for name in ('run.json', 'last.pt', 'history.json', 'best.pt', 'complete.json')):
        raise FileExistsError('Use a new empty output directory or explicitly resume an existing run')
    write_json(args.output / 'run.json', dict(identity=resume_id, labels=LABELS[task], config=str(args.config),
               transform=TRANSFORM_CONFIG, train_positives=positives, train_negatives=negatives,
               positive_weights=pos_weight, init_backbone_sha256=initial_hash, torch=str(torch.__version__),
               timm=timm.__version__, device=(torch.cuda.get_device_name() if args.device.startswith('cuda') else 'cpu'),
               command=[sys.executable, *sys.argv]))
    print(f"Starting {task} {args.variant} on {args.device}; {len(trainset)} train / {len(valset)} val records", flush=True)
    for epoch in range(len(history) + 1, args.epochs + 1):
        started = time.monotonic()
        factor = epoch / 2 if epoch <= 2 else 0.5 * (1 + math.cos(math.pi * (epoch-3) / max(1, args.epochs-2)))
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = lr * factor
        stage = 'frozen' if epoch <= args.head_only_epochs else args.variant
        loss = run_epoch(model, train_batches, args, optimizer, scaler, pos_weight, stage)
        metrics, _ = evaluate(model, val_batches, args, LABELS[task])
        if metrics['macro_ap'] is None or any(v['auprc'] is None for v in metrics['per_label'].values()):
            raise ValueError('Validation lacks required class support')
        if metrics['macro_ap'] > best_ap:
            best_epoch, best_ap = epoch, metrics['macro_ap']
        history.append({'epoch': epoch, 'stage': stage, 'train_loss': loss, 'val': metrics, 'best_epoch': best_epoch,
                        'learning_rates': [g['lr'] for g in optimizer.param_groups],
                        'elapsed_seconds': time.monotonic() - started})
        save_checkpoint(last_path, dict(identity=resume_id, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(), rng=rng_state(), history=history, best_ap=best_ap, best_epoch=best_epoch))
        if epoch == best_epoch:
            publish_best(epoch)
        write_json(args.output / 'history.json', history)
        print(f"Epoch {epoch}/{args.epochs} loss={loss:.5f} val_macro_AP={metrics['macro_ap']:.5f} best={best_ap:.5f}", flush=True)
    if initial_hash and backbone_hash(model.backbone) != initial_hash:
        raise ValueError('Frozen backbone tensors changed during training')
    write_json(args.output / 'history.json', history)
    export(model, args, audit, task, 'val', best_path, identity)
    write_json(args.output / 'complete.json', dict(best_epoch=best_epoch, best_validation_macro_ap=best_ap,
               checkpoint_sha256=sha256(best_path), epochs=len(history), frozen_backbone_sha256=initial_hash))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'predict'])
    for name in ('config', 'release', 'data-root', 'weights', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--variant', choices=['frozen', 'partial', 'full'], default='frozen')
    p.add_argument('--blocks', type=int, default=4)
    p.add_argument('--head-only-epochs', type=int, default=0,
                   help='Train only heads for these initial epochs before applying --variant.')
    p.add_argument('--rgb-cache-dir', type=Path,
                   help='Optional local scratch directory for lossless native-resolution decoded RGB arrays.')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--accum', type=int, default=2)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--precision', choices=['fp16', 'bf16', 'none'], default='fp16')
    p.add_argument('--head-lr', type=float, default=0.001)
    p.add_argument('--backbone-lr', type=float, default=0.00001)
    p.add_argument('--seed', type=int, default=20261114)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--split', choices=['val', 'test', 'external_test'], default='val')
    args = p.parse_args()
    if min(args.batch_size, args.accum, args.epochs) < 1 or args.workers < 0:
        p.error('Batch size, accumulation and epochs must be positive; workers must be nonnegative')
    if args.head_lr <= 0 or args.backbone_lr <= 0:
        p.error('Learning rates must be positive')
    if not 0 <= args.head_only_epochs <= args.epochs:
        p.error('Head-only epochs must be between zero and the total epochs')
    if args.action == 'train' and args.split != 'val':
        p.error('Training uses only train and validation manifests')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA is unavailable in this process')
    if args.device == 'cpu' and args.precision != 'none':
        p.error('CPU execution requires --precision none')
    plan = json.loads(args.config.read_text(encoding='utf-8-sig'))
    order = plan['head_architecture']['output_order']
    task = next((k for k, labels in LABELS.items() if labels == order), None)
    if task is None or plan['backbone']['name'] != BACKBONE:
        p.error('Unsupported head order or backbone in config')
    digest = sha256(args.weights)
    if digest != OFFICIAL_SHA:
        p.error('Pretrained weights do not match the pinned official DINOv2 ViT-B/14 file')
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    audit_path = args.release / 'audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
    if audit.get('status') != 'verified' or audit.get('label_order', {}).get(task) != order:
        p.error('Dataset release must be verified with the same label order as the requested task')
    identity = dict(plan_sha256=sha256(args.config), task=task, release_sha256=sha256(audit_path), pretrained_sha256=digest)
    model = ScreeningModel(task, args.weights).to(args.device)
    if args.action == 'train':
        train(args, audit, task, model, identity)
    else:
        export(model, args, audit, task, args.split, args.output / 'best.pt', identity)


if __name__ == '__main__':
    main()
