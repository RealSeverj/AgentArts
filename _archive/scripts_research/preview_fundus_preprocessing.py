"""Preview fixed TRAIN samples only; never modify images or dataset manifests."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts.dataset_routing import manifest_path, validate_training_route
from scripts.manifest_dataset import resolve_data_path
from training.fundus_preprocessing import apply_preprocessing, preprocessing_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=PROJECT / 'data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-class', type=int, default=2)
    args = parser.parse_args()
    if args.per_class < 1 or args.per_class > 10:
        parser.error('--per-class must be between 1 and 10')
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    cfg['data'].pop('image_cache_dir', None)
    validate_training_route(cfg, args.data_root)
    manifest = manifest_path(cfg, args.data_root, 'train')
    groups = defaultdict(list)
    for line in manifest.read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('label_mask', {}).get('dr_grade'):
                groups[int(row['labels']['dr_grade'])].append(row)
    selected = []
    for grade, rows in sorted(groups.items()):
        rows.sort(key=lambda x: x['sample_id'])
        random.Random(20261114 + grade).shuffle(rows)
        selected.extend(rows[:args.per_class])
    args.output.mkdir(parents=True, exist_ok=True)
    target = cfg['model']['img_size']
    spec = preprocessing_spec(cfg)
    cell, header, line_height = 280, 32, 26
    sheet = Image.new('RGB', (cell * 3, header + len(selected) * (cell + line_height)), (236, 236, 236))
    draw = ImageDraw.Draw(sheet)
    for column, text in enumerate(['Source + crop box', 'Previous stretch resize', 'Crop + aspect fit + padding']):
        draw.text((column * cell + 6, 8), text, fill=(0, 0, 0))
    records = []
    for index, row in enumerate(selected):
        source = resolve_data_path(row['image_paths'][0], args.data_root)
        with Image.open(source) as opened:
            image = opened.convert('RGB')
        result, metadata = apply_preprocessing(image, target, spec, return_metadata=True)
        old = image.resize((target, target), Image.Resampling.BILINEAR)
        annotated = image.copy()
        bounds = metadata['crop_bbox']
        ImageDraw.Draw(annotated).rectangle(tuple(bounds), outline=(0, 255, 0), width=max(1, round(min(image.size) / 200)))
        y = header + index * (cell + line_height)
        for column, frame in enumerate([annotated, old, result]):
            thumbnail = ImageOps.contain(frame, (cell, cell), Image.Resampling.BILINEAR)
            sheet.paste(thumbnail, (column * cell + (cell - thumbnail.width) // 2,
                                    y + (cell - thumbnail.height) // 2))
        draw.text((6, y + cell + 4), f"{row['sample_id']} | grade {row['labels']['dr_grade']}", fill=(0, 0, 0))
        draw.text((cell + 6, y + cell + 4), f"source {image.width}x{image.height}", fill=(0, 0, 0))
        draw.text((2 * cell + 6, y + cell + 4), f"fit {metadata['resized_size']} | {metadata['fallback']}", fill=(0, 0, 0))
        result.save(args.output / f'processed_{index:02d}.png')
        records.append({'sample_id': row['sample_id'], 'grade': row['labels']['dr_grade'],
                        'image_path': row['image_paths'][0], **metadata})
    sheet.save(args.output / 'comparison.jpg', quality=93)
    report = {'source': cfg['data']['datasets'], 'split': 'train', 'preprocessing': spec,
              'train_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'selection': 'fixed per-class random sample; seed20261114+grade; train only',
              'original_files_modified': False, 'validation_or_test_used': False, 'samples': records}
    (args.output / 'preview.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'preview': str(args.output / 'comparison.jpg'), 'samples': len(records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
