"""Download the public mariaherrerot/aptos2019 v3 mirror, preserving provenance."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import requests

ROOT = Path(__file__).resolve().parent
SIZE = 8602506535
CHUNK = 32 * 1024**2
URL = 'https://www.kaggle.com/api/v1/datasets/download/mariaherrerot/aptos2019?datasetVersionNumber=3'
ARCHIVE = ROOT / 'archives' / 'aptos2019_public_mirror_v3.zip'
PART = ARCHIVE.with_suffix('.zip.part')
STATE = ARCHIVE.with_suffix('.ranges.json')
LOCAL = threading.local()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def get_range(start, end):
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    for attempt in range(6):
        try:
            with LOCAL.session.get(URL, headers={'Range': f'bytes={start}-{end}', 'Cache-Control': 'no-cache'},
                                   stream=True, timeout=(30, 120)) as response:
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                    raise ValueError(f'Unexpected HTTP range response {response.status_code}')
                n = 0
                with PART.open('r+b') as handle:
                    handle.seek(start)
                    for block in response.iter_content(1024**2):
                        if block:
                            n += len(block)
                            if n > end - start + 1:
                                raise ValueError('Range too long')
                            handle.write(block)
                    handle.flush()
                if n != end - start + 1:
                    raise ValueError('Range incomplete')
            return start // CHUNK, n
        except Exception as error:
            # Do not emit redirect URLs, which contain temporary signing values.
            print(json.dumps({'event': 'retry', 'start': start, 'attempt': attempt + 1,
                              'error_type': type(error).__name__}), flush=True)
            if attempt == 5:
                raise RuntimeError('Range download failed after retries') from None
            time.sleep(min(2**attempt, 15))


def main():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        if ARCHIVE.stat().st_size != SIZE:
            raise ValueError('Existing archive has wrong size')
        print(json.dumps({'event': 'already_downloaded', 'bytes': SIZE}), flush=True)
        return
    if PART.exists() or STATE.exists():
        if not (PART.exists() and STATE.exists()):
            raise ValueError('Missing resume state')
        state = json.loads(STATE.read_text(encoding='utf-8'))
        if state['source_url'] != URL or state['bytes'] != SIZE or state['chunk_bytes'] != CHUNK or PART.stat().st_size != SIZE:
            raise ValueError('Resume state differs from expected source')
    else:
        state = {'source_url': URL, 'bytes': SIZE, 'chunk_bytes': CHUNK, 'completed': []}
        with PART.open('xb') as handle:
            handle.truncate(SIZE)
        save(STATE, state)
    done = set(state['completed'])
    initial = sum(min(CHUNK, SIZE - i * CHUNK) for i in done)
    received = initial
    began = time.monotonic()
    last = 0
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(get_range, start, min(start + CHUNK, SIZE) - 1)
                   for start in range(0, SIZE, CHUNK) if start // CHUNK not in done}
        while pending:
            completed, _ = cf.wait(pending, timeout=10, return_when=cf.FIRST_COMPLETED)
            for future in completed:
                index, n = future.result()
                pending.remove(future)
                state['completed'].append(index)
                state['completed'].sort()
                save(STATE, state)
                received += n
            now = time.monotonic()
            if now - last > 20 or not pending:
                p = {'event': 'progress', 'completed_bytes': received, 'total_bytes': SIZE,
                     'percent': round(received / SIZE * 100, 2),
                     'average_MiB_per_second': round((received - initial) / max(now - began, .01) / 1024**2, 2)}
                save(ROOT / 'download_progress.json', p)
                print(json.dumps(p), flush=True)
                last = now
    os.replace(PART, ARCHIVE)
    h = hashlib.sha256()
    with ARCHIVE.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    report = {'dataset': 'APTOS2019 public mirror', 'mirror': 'https://www.kaggle.com/datasets/mariaherrerot/aptos2019',
              'mirror_version': 3, 'archive_bytes': SIZE, 'sha256': h.hexdigest(),
              'original_source': 'https://www.kaggle.com/competitions/aptos2019-blindness-detection',
              'original_api_status': 401, 'license_note': 'Original competition terms apply; mirror Unknown does not replace them.',
              'split_note': 'Mirror reorganizes original 3662 labelled images into 2930/366/366. Not original competition test set.'}
    save(ROOT / 'download_manifest.json', report)
    print(json.dumps({'event': 'download_complete', 'bytes': SIZE, 'sha256': h.hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
