"""Get a clearly labelled 480px EyeQ image mirror and match official labels."""
import collections
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time
import zipfile
import zlib

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCES = [
    {'split': 'train', 'ref': 'bishoyekramy/eyeq-train-resized', 'bytes': 178142642, 'rows': 12543, 'folder': 'resized_train'},
    {'split': 'test', 'ref': 'bishoyekramy/eyeq-test-resized', 'bytes': 226289071, 'rows': 16249, 'folder': 'resized_test'},
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def crc32(path):
    value = 0
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            value = zlib.crc32(block, value)
    return value & 0xffffffff


def main():
    archives = ROOT / 'archives'
    archives.mkdir(exist_ok=True)
    output_root = ROOT / 'images_480px_public_mirror'
    output_root.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / 'download_manifest.json').read_text(encoding='utf-8'))
    reports = []
    session = requests.Session()
    for spec in SOURCES:
        url = 'https://www.kaggle.com/api/v1/datasets/download/' + spec['ref'] + '?datasetVersionNumber=1'
        archive = archives / ('eyeq_' + spec['split'] + '_480px_public_mirror_v1.zip')
        part = archive.with_suffix('.zip.part')
        if not archive.exists():
            for attempt in range(6):
                try:
                    start = part.stat().st_size if part.exists() else 0
                    if start > spec['bytes']:
                        raise ValueError('Partial file larger than source')
                    if start == spec['bytes']:
                        break
                    with session.get(url, headers={'Range': f"bytes={start}-{spec['bytes'] - 1}"},
                                     stream=True, timeout=(30, 120)) as response:
                        expected = f"bytes {start}-{spec['bytes'] - 1}/{spec['bytes']}"
                        if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                            raise ValueError(f'Unexpected range HTTP {response.status_code}')
                        with part.open('ab') as handle:
                            for block in response.iter_content(1024**2):
                                if block:
                                    handle.write(block)
                    if part.stat().st_size != spec['bytes']:
                        raise ValueError('Incomplete download')
                    break
                except Exception as error:
                    print(json.dumps({'event': 'retry', 'split': spec['split'], 'error_type': type(error).__name__}), flush=True)
                    if attempt == 5:
                        raise RuntimeError('EyeQ mirror download failed') from None
                    time.sleep(min(2**attempt, 15))
            os.replace(part, archive)
        if archive.stat().st_size != spec['bytes']:
            raise ValueError('Archive size mismatch')
        print(json.dumps({'event': 'downloaded', 'split': spec['split'], 'bytes': spec['bytes']}), flush=True)
        with zipfile.ZipFile(archive) as package:
            for index, member in enumerate(package.infolist(), 1):
                name = PurePosixPath(member.filename.replace('\\', '/'))
                if name.is_absolute() or '..' in name.parts or any(':' in p for p in name.parts):
                    raise ValueError('Unsafe ZIP member')
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError('Unexpected symlink')
                target = output_root.joinpath(*name.parts).resolve()
                if not target.is_relative_to(output_root.resolve()):
                    raise ValueError('Unsafe extracted path')
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.stat().st_size == member.file_size and crc32(target) == member.CRC:
                        continue
                    raise ValueError('Existing file differs from source')
                temporary = target.with_name(target.name + '.extracting')
                with package.open(member) as source, temporary.open('wb') as output:
                    for block in iter(lambda: source.read(1024**2), b''):
                        output.write(block)
                if temporary.stat().st_size != member.file_size:
                    raise ValueError('Extracted size mismatch')
                os.replace(temporary, target)
                if index % 5000 == 0:
                    print(json.dumps({'event': 'extracting', 'split': spec['split'], 'images': index}), flush=True)
        with (ROOT / 'labels' / ('Label_EyeQ_' + spec['split'] + '.csv')).open(encoding='utf-8-sig', newline='') as handle:
            labels = list(csv.DictReader(handle))
        paths = list((output_root / spec['folder']).glob('*.jpeg'))
        if len(paths) != spec['rows'] or len(labels) != spec['rows'] or {p.name for p in paths} != {r['image'] for r in labels}:
            raise ValueError('Mirror images do not match official label filenames')
        dimensions = collections.Counter()
        for path in paths:
            with Image.open(path) as image:
                dimensions[image.size] += 1
                image.verify()
        if set(dimensions) != {(480, 480)}:
            raise ValueError('Unexpected preprocessed mirror dimensions')
        reports.append({'split': spec['split'], 'image_count': len(paths), 'label_rows': len(labels),
                        'image_label_filenames_match': True, 'jpeg_verification_passed': True,
                        'dimensions': [480, 480], 'archive_bytes': spec['bytes'], 'archive_sha256': digest(archive),
                        'source': 'https://www.kaggle.com/datasets/' + spec['ref'], 'version': 1,
                        'quality_counts': dict(collections.Counter(r['quality'] for r in labels))})
        print(json.dumps({'event': 'split_verified', **reports[-1]}), flush=True)
    manifest['processed_image_mirrors'] = reports
    manifest['images_downloaded'] = sum(s['image_count'] for s in reports)
    manifest['image_download_status'] = 'Complete 480x480 public mirror; original EyePACS images not obtained.'
    manifest['original_images_downloaded'] = 0
    manifest['raw_images_downloaded'] = 0
    manifest['processed_images_downloaded'] = manifest['images_downloaded']
    manifest['license_note'] = 'Mirror Apache 2.0 metadata does not replace EyePACS source-image terms or establish a license for author labels.'
    (ROOT / 'download_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {'all_checks_passed': True, 'image_representation': 'Public resized 480x480 JPEG mirror, not original images',
              'images': manifest['images_downloaded'], 'label_rows': 28792, 'original_images': 0,
              'raw_images_downloaded': 0, 'processed_images': manifest['images_downloaded'],
              'official_labels_complete': True,
              'zip_crc_checked': True, 'splits': reports}
    (ROOT / 'verification_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'complete', 'images': manifest['images_downloaded'], 'original_images': 0}), flush=True)


if __name__ == '__main__':
    main()
