"""Controlled RFMiD backbone and predicted optic-disc-view comparisons.

Outputs follow screening_training's checkpoint, validation prediction, and AP
contracts. The three optic-disc outputs may use a second, predicted local view.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import timm
import torch
from timm.models.vision_transformer import checkpoint_filter_fn
from torch import nn

from training import screening_training as baseline
from training.train import restore_rng, rng_state, save_checkpoint, seed_all, write_json


MODELS = {
    'vitb': (baseline.BACKBONE, baseline.OFFICIAL_SHA),
    'vitl': ('vit_large_patch14_dinov2.lvd142m',
             'd5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428'),
}
LOCATOR_SHA = 'bb25840047eedca5b115c3a81c9f7b8e3e73a8a0729cc6c67b9f55b1c38ef5e7'
ROI_POLICY = 'disc_lcc8_square2_v1'
ROI_POLICY_SHA = '0e5333977ca662baeab6025ea1ed0ec6f9bcd8f5c73dd63f07142376052a1f85'
LABELS = baseline.LABELS['grouped']
GROUP_WIDTHS = (2, 3, 3, 2, 1)
DISC_GROUP = 2


class GroupedModel(baseline.ScreeningModel):
    def __init__(self, architecture, weights, use_roi):
        nn.Module.__init__(self)
        name, _ = MODELS[architecture]
        self.backbone = timm.create_model(name, pretrained=False, img_size=392, num_classes=0)
        raw = torch.load(weights, map_location='cpu', weights_only=True)
        self.backbone.load_state_dict(checkpoint_filter_fn(raw, self.backbone), strict=True)
        self.use_roi = use_roi
        self.dropout = nn.Dropout(0.1)
        dim = self.backbone.num_features
        self.heads = nn.ModuleList(nn.Linear(2 * dim if use_roi and i == DISC_GROUP else dim, w)
                                   for i, w in enumerate(GROUP_WIDTHS))

    def forward(self, x):
        if not self.use_roi:
            return super().forward(x)
        global_features = self.dropout(self.backbone(x[:, 0]))
        local_features = self.dropout(self.backbone(x[:, 1]))
        disc_features = torch.cat((global_features, local_features), dim=1)
        return torch.cat([head(disc_features if i == DISC_GROUP else global_features)
                          for i, head in enumerate(self.heads)], dim=1)


class GroupedViews(baseline.ScreeningDataset):
    def __init__(self, args, audit, split):
        super().__init__(args, audit, 'grouped', split)
        self.roi_sha = None
        self.locations = None
        self.training = split == 'train'
        if args.roi_dir:
            path = args.roi_dir / f'{split}.jsonl'
            self.roi_sha = baseline.sha256(path)
            meta = json.loads((args.roi_dir / f'{split}.meta.json').read_text(encoding='utf-8-sig'))
            if (meta['source_manifest_sha256'] != self.manifest_sha or meta['coordinate_sha256'] != self.roi_sha
                    or meta['split'] != split or meta['samples'] != len(self.rows)
                    or meta['policy_sha256'] != ROI_POLICY_SHA or meta['locator_sha256'] != LOCATOR_SHA):
                raise ValueError(f'ROI metadata does not bind the expected source, coordinates and policy for {split}')
            self.locations = {}
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                r = json.loads(line)
                sid = r['sample_id']
                if (sid in self.locations or r['locator_sha256'].lower() != LOCATOR_SHA
                        or r.get('policy_id') != ROI_POLICY or r.get('policy_sha256') != ROI_POLICY_SHA):
                    raise ValueError(f'Invalid or duplicate predicted ROI for {sid}')
                self.locations[sid] = r
            if set(self.locations) != {r['sample_id'] for r in self.rows}:
                raise ValueError(f'Predicted ROI IDs must exactly match {split}')
            for row in self.rows:
                r = self.locations[row['sample_id']]
                if (any(r[k] != row[k] for k in ('image_sha256', 'split', 'dataset', 'image_path'))
                        or r.get('status') != ('fallback' if r['bbox_xyxy'] is None else 'ok')
                        or len(r['source_size_wh']) != 2 or any(type(v) is not int or v < 1 for v in r['source_size_wh'])):
                    raise ValueError(f"ROI missing or bound to a different image: {row['sample_id']}")

    def __getitem__(self, index):
        if self.locations is None:
            return super().__getitem__(index)
        row = self.rows[index]
        image = self.source_rgb(index)
        record = self.locations[row['sample_id']]
        box = record['bbox_xyxy']
        if tuple(record['source_size_wh']) != image.size:
            raise ValueError(f"ROI coordinates use a different image size: {row['sample_id']}")
        state = random.getstate() if self.training else None
        full = self.transform(image)
        if box is None:
            local = full
        else:
            if (len(box) != 4 or any(type(v) is not int for v in box) or box[2]-box[0] != box[3]-box[1]
                    or not 0 <= box[0] < box[2] <= image.width or not 0 <= box[1] < box[3] <= image.height):
                raise ValueError(f"Invalid original-pixel crop: {row['sample_id']}")
            if state is not None:
                # Both views get the same flip, rotation and colour factors. The
                # baseline transform draws the same number of values per call.
                random.setstate(state)
            local = self.transform(image.crop(box))
        return torch.stack((full, local)), torch.tensor(row['labels'], dtype=torch.float32), torch.tensor(row['mask']), index


def read_best(path, signature):
    best = torch.load(path, map_location='cpu', weights_only=True)
    if best['identity'] != signature:
        raise ValueError('Best checkpoint uses a different data release, model or ROI contract')
    return best


def export(args, audit, model, signature, split):
    checkpoint = args.output / 'best.pt'
    best = read_best(checkpoint, signature)
    model.load_state_dict(best['model'], strict=True)
    precision = best['settings']['precision']
    args.precision = precision
    data = GroupedViews(args, audit, split)
    metrics, records = baseline.evaluate(model, baseline.loader(data, args, False), args, LABELS, collect=True)
    digest = baseline.sha256(checkpoint)
    path = args.output / f'{split}_predictions.jsonl'
    tmp = path.with_suffix('.jsonl.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        for row in records:
            if data.locations is not None:
                roi = data.locations[row['sample_id']]
                row['roi_fallback'] = roi['bbox_xyxy'] is None
                row['bbox_xyxy'] = roi['bbox_xyxy']
            row['checkpoint_sha256'] = digest
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)
    write_json(path.with_suffix('.meta.json'), dict(labels=LABELS, split=split, epoch=best['epoch'],
               precision=precision, checkpoint_sha256=digest, metrics_at_0_5=metrics, manifest_sha256=data.manifest_sha,
               roi_manifest_sha256=data.roi_sha, locator_sha256=signature['locator_sha256']))


def train(args, audit, model, signature):
    datasets = {s: GroupedViews(args, audit, s) for s in ('train', 'val')}
    batches = {s: baseline.loader(data, args, s == 'train') for s, data in datasets.items()}
    rows = datasets['train'].rows
    mask = np.array([r['mask'] for r in rows], dtype=bool)
    y = np.array([r['labels'] for r in rows])
    positive, negative = ((y == 1) & mask).sum(0), ((y == 0) & mask).sum(0)
    if not np.all((positive > 0) & (negative > 0)):
        raise ValueError('Missing training class')
    pos_weight = torch.tensor(np.minimum(10, np.sqrt(negative / positive)), device=args.device, dtype=torch.float32)
    model.configure(args.variant, args.blocks)
    params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{'params': model.heads.parameters(), 'lr': args.head_lr}]
    if params:
        groups.append({'params': params, 'lr': args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0001)
    base_lr = [g['lr'] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16' and args.device.startswith('cuda'))
    settings = {k: getattr(args, k) for k in ('epochs', 'batch_size', 'accum', 'workers', 'blocks', 'variant',
                                               'seed', 'precision', 'head_only_epochs', 'head_lr', 'backbone_lr')}
    resume_id = dict(signature, settings=settings, code_sha256=baseline.sha256(Path(__file__)),
                     baseline_code_sha256=baseline.sha256(Path(baseline.__file__)))
    best_ap, best_epoch, history = -1., 0, []
    best_path, last_path = args.output / 'best.pt', args.output / 'last.pt'
    def publish_best():
        save_checkpoint(best_path, dict(model=model.state_dict(), labels=LABELS, identity=signature,
                        epoch=best_epoch, settings=settings, best_validation_macro_ap=best_ap))
    if args.resume and last_path.is_file():
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        if state['identity'] != resume_id:
            raise ValueError('Resume configuration, data, weights or code changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'])
        best_ap, best_epoch, history = state['best_ap'], state['best_epoch'], state['history']
        del state
        if len(history) == best_epoch and not best_path.is_file():
            publish_best()
        elif len(history) == best_epoch and read_best(best_path, signature)['epoch'] != best_epoch:
            publish_best()
    elif args.resume:
        record = json.loads((args.output / 'run.json').read_text(encoding='utf-8'))
        if record['identity'] != resume_id or best_path.exists() or (args.output/'complete.json').exists():
            raise ValueError('Cannot restart uncommitted first epoch with conflicting run state')
    elif any((args.output / n).exists() for n in ('run.json', 'best.pt', 'last.pt', 'complete.json')):
        raise FileExistsError('Choose a fresh optimization output directory')
    first_hash = baseline.backbone_hash(model.backbone) if args.variant == 'frozen' else None
    transform = dict(baseline.TRANSFORM_CONFIG, model=dict(baseline.TRANSFORM_CONFIG['model'], name=signature['backbone']))
    write_json(args.output / 'run.json', dict(identity=resume_id, labels=LABELS, transform=transform,
               positive_weights=pos_weight, train_positives=positive, train_negatives=negative,
               torch=str(torch.__version__), timm=timm.__version__,
               device=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'cpu',
               command=[sys.executable, *sys.argv]))
    for epoch in range(len(history) + 1, args.epochs + 1):
        start = time.monotonic()
        factor = epoch / 2 if epoch <= 2 else 0.5 * (1 + math.cos(math.pi * (epoch-3) / max(1, args.epochs-2)))
        for g, lr in zip(optimizer.param_groups, base_lr):
            g['lr'] = lr * factor
        stage = 'frozen' if epoch <= args.head_only_epochs else args.variant
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        loss = baseline.run_epoch(model, batches['train'], args, optimizer, scaler, pos_weight, stage)
        metrics, _ = baseline.evaluate(model, batches['val'], args, LABELS)
        if any(v['auprc'] is None for v in metrics['per_label'].values()):
            raise ValueError('Validation AP is missing for a label')
        if metrics['macro_ap'] > best_ap:
            best_epoch, best_ap = epoch, metrics['macro_ap']
        history.append(dict(epoch=epoch, stage=stage, train_loss=loss, val=metrics, best_epoch=best_epoch,
                            learning_rates=[g['lr'] for g in optimizer.param_groups], elapsed_seconds=time.monotonic()-start,
                            peak_cuda_allocated_gib=(torch.cuda.max_memory_allocated()/2**30 if args.device.startswith('cuda') else None)))
        save_checkpoint(last_path, dict(identity=resume_id, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(), rng=rng_state(), history=history, best_ap=best_ap, best_epoch=best_epoch))
        if epoch == best_epoch:
            publish_best()
        write_json(args.output / 'history.json', history)
        print(f"Epoch {epoch}/{args.epochs} {stage} val_macro_AP={metrics['macro_ap']:.5f} best={best_ap:.5f}", flush=True)
    if first_hash and baseline.backbone_hash(model.backbone) != first_hash:
        raise ValueError('Frozen backbone changed')
    write_json(args.output / 'history.json', history)
    export(args, audit, model, signature, 'val')
    write_json(args.output / 'complete.json', dict(best_epoch=best_epoch, best_validation_macro_ap=best_ap,
               epochs=len(history), checkpoint_sha256=baseline.sha256(best_path)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'predict'])
    p.add_argument('--backbone', choices=MODELS, required=True)
    for name in ('weights', 'release', 'data-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--roi-dir', type=Path, help='Predicted ROI JSONL directory. Omit for full-image control.')
    p.add_argument('--rgb-cache-dir', type=Path)
    p.add_argument('--variant', choices=('frozen','partial','full'), default='partial')
    for name, default in [('epochs',30),('batch-size',8),('accum',4),('workers',4),('blocks',4),
                          ('head-only-epochs',5),('seed',20261114)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--head-lr', type=float, default=0.001)
    p.add_argument('--backbone-lr', type=float, default=0.00001)
    p.add_argument('--precision', choices=['fp16','bf16','none'], default='fp16')
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--split', choices=['val','test','external_test'], default='val')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if (min(args.epochs, args.batch_size, args.accum, args.blocks, args.head_lr, args.backbone_lr) <= 0
            or not 0 <= args.head_only_epochs <= args.epochs or args.workers < 0):
        p.error('Invalid training schedule, optimizer parameters or batch size')
    if args.action == 'train' and args.split != 'val':
        p.error('Training selects checkpoints on validation only')
    if args.device == 'cpu' and args.precision != 'none':
        p.error('CPU execution requires --precision none')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    name, digest = MODELS[args.backbone]
    if baseline.sha256(args.weights) != digest:
        p.error('Incompatible pretrained file for the selected official backbone')
    audit_path = args.release / 'audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
    if audit.get('status') != 'verified' or audit['label_order']['grouped'] != LABELS:
        p.error('RFMiD data release or label order is incompatible')
    signature = dict(task='grouped', backbone=name, pretrained_sha256=digest, release_sha256=baseline.sha256(audit_path),
                     image_size=392, input_mode='global_disc_roi' if args.roi_dir else 'global',
                     locator_sha256=LOCATOR_SHA if args.roi_dir else None)
    if args.roi_dir:
        signature['selection_roi_sha256'] = {s: baseline.sha256(args.roi_dir / f'{s}.jsonl') for s in ('train','val')}
        signature.update(roi_policy=ROI_POLICY, roi_policy_sha256=ROI_POLICY_SHA)
    seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    model = GroupedModel(args.backbone, args.weights, bool(args.roi_dir)).to(args.device)
    if args.action == 'train':
        train(args, audit, model, signature)
    else:
        export(args, audit, model, signature, args.split)


if __name__ == '__main__':
    main()
