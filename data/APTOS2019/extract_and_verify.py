"""Validate the public APTOS mirror without misrepresenting its split."""
import collections
import csv
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile
import zlib

from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'public_mirror'


def crc32(path):
    value = 0
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            value = zlib.crc32(block, value)
    return value & 0xffffffff


def main():
    archive = ROOT / 'archives' / 'aptos2019_public_mirror_v3.zip'
    OUTPUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for index, member in enumerate(package.infolist(), 1):
            name = PurePosixPath(member.filename.replace('\\', '/'))
            if name.is_absolute() or '..' in name.parts or any(':' in part for part in name.parts):
                raise ValueError('Unsafe ZIP path')
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Unexpected symbolic link')
            target = OUTPUT.joinpath(*name.parts).resolve()
            if not target.is_relative_to(OUTPUT.resolve()):
                raise ValueError('Path outside dataset output')
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size == member.file_size and crc32(target) == member.CRC:
                    continue
                raise ValueError('Existing extracted file differs from archive')
            temporary = target.with_name(target.name + '.extracting')
            with package.open(member) as source, temporary.open('wb') as output:
                for block in iter(lambda: source.read(1024**2), b''):
                    output.write(block)
            if temporary.stat().st_size != member.file_size:
                raise ValueError('Incomplete extracted file')
            os.replace(temporary, target)
            if index % 500 == 0:
                print(json.dumps({'event': 'extracting', 'files': index}), flush=True)
    images = list(OUTPUT.rglob('*.png'))
    mapping = {}
    for path in images:
        if path.stem in mapping:
            raise ValueError('Duplicate image ID across mirror folders')
        mapping[path.stem] = path
    all_rows = []
    splits = []
    for file, expected in [('train_1.csv', 2930), ('valid.csv', 366), ('test.csv', 366)]:
        with (OUTPUT / file).open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ['id_code', 'diagnosis']:
                raise ValueError('Unexpected label schema')
            rows = list(reader)
        if len(rows) != expected:
            raise ValueError('Unexpected mirror split count')
        for row in rows:
            if row['diagnosis'] not in {'0', '1', '2', '3', '4'} or row['id_code'] not in mapping:
                raise ValueError('Invalid diagnosis or missing image')
            all_rows.append({'id_code': row['id_code'], 'diagnosis': row['diagnosis'],
                             'image_path': mapping[row['id_code']].relative_to(ROOT).as_posix(),
                             'mirror_split': file.removesuffix('.csv')})
        splits.append({'file': file, 'rows': len(rows)})
    if len(images) != 3662 or len(all_rows) != 3662 or len({r['id_code'] for r in all_rows}) != 3662:
        raise ValueError('Image and label counts differ')
    counts = dict(collections.Counter(row['diagnosis'] for row in all_rows))
    if counts != {'0': 1805, '1': 370, '2': 999, '3': 193, '4': 295}:
        raise ValueError('Class counts differ from original labelled training set')
    dimensions = collections.Counter()
    for path in images:
        with Image.open(path) as image:
            dimensions[image.size] += 1
            image.verify()
    with (ROOT / 'all_labeled_images.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id_code', 'diagnosis', 'image_path', 'mirror_split'])
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda row: row['id_code']))
    report = {'dataset': 'APTOS2019 publicly redistributed and re-split mirror', 'all_checks_passed': True,
              'archive_zip_crc_checked': True, 'png_verification_passed': True, 'images': len(images),
              'label_rows': len(all_rows), 'image_label_ids_match': True, 'label_counts': counts,
              'mirror_splits': splits, 'unique_dimensions': len(dimensions),
              'most_common_dimensions': [{'size': size, 'count': count} for size, count in dimensions.most_common(10)],
              'original_competition_test_images_downloaded': False,
              'note': 'Mirror train/valid/test are re-partitions of original 3662 labelled training images. They are not original competition train/public-test splits.'}
    (ROOT / 'verification_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
