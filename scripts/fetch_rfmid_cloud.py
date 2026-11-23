"""Fetch the three exact original RFMiD ZIPs and preserve published image bytes.

Thin adapter around fetch_dr_cloud_ranges and its SHA-gated safe extractor.
The historical download_manifest.json is read, never rewritten. Downloads use
public Google Drive IDs, resumable range journals and fixed whole-file SHA256;
no URL, response body, cookie or HTTP exception text is printed. No data split,
label or image conversion is performed. Requires requests.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import queue
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import fetch_dr_cloud_ranges as ranges
from scripts import fetch_dr_cloud_sources as legacy

PINNED = (
    {'split': 'training', 'name': 'Training_Set.zip', 'id': '1_pzNv3z3H1_ao8nOrp-B40S5TE1Rcx1W',
     'bytes': 4681355361, 'expected_images': 1920,
     'sha256': 'cc39852c4d83cebb627e70383004bd41efccb4141df06817d6f2663e882be15e',
     'folder': 'Training_Set/Training', 'csv': 'Training_Set/RFMiD_Training_Labels.csv',
     'csv_sha256': '03f3d0afcbe6829eef111ceb73ee6fe1e1c1b18531f6a789502279bfda371878'},
    {'split': 'validation', 'name': 'Evaluation_Set.zip', 'id': '1ORmTGxwXrU9QjufVL8H93yj8GZHjxrXC',
     'bytes': 1564446026, 'expected_images': 640,
     'sha256': '26c5e775338d3895d4ad5a42bc10fe6c531b20d9fc01b44c62c9196d73119d85',
     'folder': 'Evaluation_Set/Validation', 'csv': 'Evaluation_Set/RFMiD_Validation_Labels.csv',
     'csv_sha256': '465ed8cbea77f269d17bf4055a90b2bd5b75edacb82e0519c876f28d98b24fff'},
    {'split': 'test', 'name': 'Test_Set.zip', 'id': '167Km-NVmvQLlbegWKC3Biwu0ETs4dWUK',
     'bytes': 1742451361, 'expected_images': 640,
     'sha256': 'b7ca643913f1bf7c9fa80ae6913b5d61aaa24227b993ed4a717aa05b80274966',
     'folder': 'Test_Set/Test', 'csv': 'Test_Set/RFMiD_Testing_Labels.csv',
     'csv_sha256': '4ebc844ddd280777cb18112f60b4890692efcb6e5121672b888d2d5a3fdf5e05'},
)


def read_pinned_sources(metadata):
    """Require the original exact sizes, public IDs, counts and historical SHAs."""
    manifest = json.loads(Path(metadata).read_text(encoding='utf-8-sig'))
    files = manifest.get('files')
    if not isinstance(files, list) or len(files) != 3 or any(not isinstance(row, dict) for row in files):
        raise legacy.FetchError('RFMiD metadata must contain exactly the three original archives')
    if len({row.get('name') for row in files}) != 3:
        raise legacy.FetchError('RFMiD archive names are repeated')
    indexed = {row.get('name'): row for row in files}
    sources = []
    for pinned in PINNED:
        recorded = indexed.get(pinned['name'])
        if not recorded or any(recorded.get(key) != pinned[key]
                               for key in ('split', 'name', 'id', 'bytes', 'expected_images', 'sha256')):
            raise legacy.FetchError('RFMiD metadata differs from the fixed original archive identity')
        sources.append({'name': 'RFMiD_' + pinned['split'], 'archive_name': pinned['name'],
                        'bytes': pinned['bytes'], 'sha256': pinned['sha256'], 'extract_root': 'RFMiD',
                        'download_url': 'https://drive.usercontent.google.com/download'
                                        f"?id={pinned['id']}&export=download&confirm=t"})
    return sorted(sources, key=lambda source: source['bytes'])


def verify_published_split(data_root, pinned):
    """Validate exact original CSV bytes and unchanged published image paths."""
    root = Path(data_root).resolve() / 'RFMiD'
    label_path = root / pinned['csv']
    label_hash = hashlib.sha256(label_path.read_bytes()).hexdigest()
    if label_hash != pinned['csv_sha256']:
        raise legacy.FetchError('Extracted original RFMiD CSV SHA256 differs; existing files preserved')
    with label_path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    if (not fields or len(fields) != 47 or fields[:2] != ['ID', 'Disease_Risk']
            or len(rows) != pinned['expected_images']):
        raise legacy.FetchError('RFMiD original label count/schema differs')
    ids = [str(int(row['ID'])) for row in rows]
    if len(set(ids)) != len(ids) or any(row[field] not in {'0', '1'} for row in rows for field in fields[1:]):
        raise legacy.FetchError('RFMiD original label IDs or binary values differ')
    images = list((root / pinned['folder']).glob('*.png'))
    if len(images) != pinned['expected_images'] or {path.stem for path in images} != set(ids):
        raise legacy.FetchError('RFMiD published PNG paths and original CSV IDs differ')
    legacy.emit('rfmid_split_verified', 'RFMiD_' + pinned['split'], images=len(images),
                label_sha256=label_hash, original_png_bytes_preserved=True)


def range_adapter(original_fetch_chunk):
    """Keep the original RFMiD downloader's per-range public cache-busting key."""
    def fetch(source, index, chunk_bytes, partial, state, retries, stopped):
        adapted = deepcopy(source)
        start, end = ranges.chunk_bounds(index, chunk_bytes, source['bytes'])
        adapted['download_url'] += f'&rfmid_range={start}-{end}'
        return original_fetch_chunk(adapted, index, chunk_bytes, partial, state, retries, stopped)
    return fetch


def fetch_all(sources, data_root, cache_root, connections=32, retries=6, chunk_mib=32):
    """One archive at a time: connections is the total active range limit."""
    data_root, cache_root = Path(data_root).resolve(), Path(cache_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    state, events = ranges.TransferState(), queue.Queue()
    original_fetch_chunk = ranges.fetch_chunk
    ranges.fetch_chunk = range_adapter(original_fetch_chunk)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            for source in sources:
                future = pool.submit(ranges.process_source, source, cache_root, data_root,
                                     chunk_mib * 1024**2, connections, retries, state, events)
                last_progress = time.monotonic()
                while not future.done():
                    wait([future], timeout=1)
                    while not events.empty():
                        event, name, fields = events.get_nowait()
                        legacy.emit(event, name, **fields)
                    if time.monotonic() - last_progress >= 30:
                        for name, fields in state.snapshot().items():
                            if fields.get('phase') not in {'complete', 'failed'}:
                                legacy.emit('progress', name, **fields)
                        last_progress = time.monotonic()
                future.result()
                while not events.empty():
                    event, name, fields = events.get_nowait()
                    legacy.emit(event, name, **fields)
                pinned = next(item for item in PINNED if item['name'] == source['archive_name'])
                verify_published_split(data_root, pinned)
    finally:
        ranges.fetch_chunk = original_fetch_chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--metadata', type=Path,
                        help='Default: DATA_ROOT/RFMiD/download_manifest.json; copy the small historical file first')
    parser.add_argument('--connections', type=int, default=32)
    parser.add_argument('--chunk-mib', type=int, default=32)
    parser.add_argument('--retries', type=int, default=6)
    parser.add_argument('--plan', action='store_true', help='Validate metadata and print fixed identities without network or writes')
    args = parser.parse_args()
    if not 1 <= args.connections <= 256 or not 1 <= args.chunk_mib <= 64 or not 1 <= args.retries <= 20:
        parser.error('connections=1..256, chunk-mib=1..64, retries=1..20')
    try:
        sources = read_pinned_sources(args.metadata or args.data_root / 'RFMiD/download_manifest.json')
        if args.plan:
            legacy.emit('rfmid_plan', sources=[{key: value for key, value in source.items() if key != 'download_url'}
                                              for source in sources], total_bytes=sum(source['bytes'] for source in sources),
                        connections=args.connections, network_performed=False, writes_performed=False)
            return 0
        fetch_all(sources, args.data_root, args.cache_root, args.connections, args.retries, args.chunk_mib)
        legacy.emit('rfmid_complete', archives=3, images=3200, all_checks_passed=True,
                    original_png_bytes_preserved=True, task_manifests_changed=False)
        return 0
    except BaseException as error:
        # HTTP exception strings can contain redirect URLs/cookies. Even local
        # setup errors are logged only by class unless this is our safe type.
        legacy.emit('rfmid_failed', error_type=type(error).__name__,
                    reason=str(error) if isinstance(error, legacy.FetchError) else 'details suppressed')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
