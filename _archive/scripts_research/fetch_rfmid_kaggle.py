"""Transport the byte-identical original RFMiD files through a fixed Kaggle mirror.

The container is Kaggle andrewmvd/retinal-disease-classification v1, NOT one of
the three original release ZIPs. Its pinned MD5 identifies transport bytes only.
Every published PNG/CSV must match a separately pinned original-file SHA256
manifest. All files are staged and verified before any new RFMiD file is
published. Original images, labels, prepared splits and training code are never
converted, rebuilt or overwritten. Signed URLs remain in memory and are never
printed or written into journals/reports. Requires requests.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import threading
import time
import zipfile

import requests

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts import fetch_dr_cloud_ranges as ranges
from scripts import fetch_dr_cloud_sources as legacy

PUBLIC_URL = 'https://www.kaggle.com/api/v1/datasets/download/andrewmvd/retinal-disease-classification?datasetVersionNumber=1'
CONTAINER_BYTES = 7973757355
CONTAINER_MD5 = 'fe823452f514e4d7d0a67cf5f5c08495'
ORIGINAL_MANIFEST_SHA256 = 'de4d1b6686fa344bab2b6cb040d87338aab0688a8e5b9248aa33a5bcbe34d147'
SPLITS = (
    ('Training_Set', 'Training', 1920, 'RFMiD_Training_Labels.csv', '03f3d0afcbe6829eef111ceb73ee6fe1e1c1b18531f6a789502279bfda371878', 'cc39852c4d83cebb627e70383004bd41efccb4141df06817d6f2663e882be15e'),
    ('Evaluation_Set', 'Validation', 640, 'RFMiD_Validation_Labels.csv', '465ed8cbea77f269d17bf4055a90b2bd5b75edacb82e0519c876f28d98b24fff', '26c5e775338d3895d4ad5a42bc10fe6c531b20d9fc01b44c62c9196d73119d85'),
    ('Test_Set', 'Test', 640, 'RFMiD_Testing_Labels.csv', '4ebc844ddd280777cb18112f60b4890692efcb6e5121672b888d2d5a3fdf5e05', 'b7ca643913f1bf7c9fa80ae6913b5d61aaa24227b993ed4a717aa05b80274966'),
)
FetchError = legacy.FetchError


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def expected_paths():
    paths = []
    for folder, images, count, csv, _, _ in SPLITS:
        paths.extend(f'{folder}/{images}/{i}.png' for i in range(1, count + 1))
        paths.append(f'{folder}/{csv}')
    return sorted(paths)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name + '.', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_original_manifest(original_root, output):
    """Local preparation only: stream-hash the exact original published files."""
    root, output = Path(original_root).resolve(), Path(output)
    if output.exists():
        raise FetchError('Original-file manifest already exists; it was preserved')
    def record(relative):
        path = checked_target(root, relative)
        if not path.is_file() or path.is_symlink():
            raise FetchError('An expected original published PNG/CSV is missing or is a symlink')
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise FetchError('An original file changed while creating the SHA256 manifest')
        return relative, {'bytes': before.st_size, 'sha256': digest}
    with ThreadPoolExecutor(max_workers=8) as pool:
        files = dict(pool.map(record, expected_paths()))
    for folder, _, _, csv, expected, _ in SPLITS:
        if files[f'{folder}/{csv}']['sha256'] != expected:
            raise FetchError('Local original CSV differs from the previously verified release')
    payload = {'format_version': 1, 'dataset': 'RFMiD original release',
               'purpose': 'Canonical original file identities for mirror transport verification; no split or label changes.',
               'relative_to': 'data/RFMiD', 'file_count': 3203, 'png_count': 3200, 'csv_count': 3,
               'original_archive_sha256': {folder + '.zip': digest for folder, _, _, _, _, digest in SPLITS},
               'files': files}
    atomic_json(output, payload)
    return sha256_file(output)


def read_original_manifest(path, expected_sha256=ORIGINAL_MANIFEST_SHA256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise FetchError('Original-file SHA256 manifest checksum mismatch')
    obj = json.loads(raw.decode('utf-8-sig'))
    if (obj.get('format_version') != 1 or obj.get('dataset') != 'RFMiD original release'
            or obj.get('relative_to') != 'data/RFMiD' or obj.get('file_count') != 3203
            or obj.get('png_count') != 3200 or obj.get('csv_count') != 3
            or not isinstance(obj.get('files'), dict) or set(obj['files']) != set(expected_paths())):
        raise FetchError('Original-file manifest does not describe the complete original release')
    for relative, info in obj['files'].items():
        if (not isinstance(info, dict) or type(info.get('bytes')) is not int or info['bytes'] <= 0
                or not isinstance(info.get('sha256'), str) or len(info['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in info['sha256'])):
            raise FetchError('Invalid original file size/SHA256 record')
    for folder, _, _, csv, digest, _ in SPLITS:
        if obj['files'][f'{folder}/{csv}']['sha256'] != digest:
            raise FetchError('Original-file manifest label CSV identity differs')
    return obj


def checked_target(root, relative):
    root = Path(root)
    text = relative
    if not isinstance(text, str) or not text or '\\' in text or ':' in text:
        raise FetchError('Unsafe RFMiD file path')
    rel = PurePosixPath(text)
    if rel.is_absolute() or any(part in {'', '.', '..'} for part in text.split('/')) or rel.as_posix() != text:
        raise FetchError('Unsafe or noncanonical RFMiD file path')
    # Reject symlinked parents as well as a symlinked final target. Resolving
    # before these checks would conceal a link that points inside the root.
    current = root
    if current.is_symlink():
        raise FetchError('RFMiD root must not be a symbolic link')
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise FetchError('RFMiD target path contains a symbolic link')
    return current


def matches_file(path, info):
    return path.is_file() and not path.is_symlink() and path.stat().st_size == info['bytes'] and sha256_file(path) == info['sha256']


def resolve_container_url(total):
    # Resolve a fresh ordinary Kaggle redirect for each invocation, never store
    # the resulting expiring URL. A small Range also verifies the expected size.
    from urllib.parse import urlparse
    with requests.get(PUBLIC_URL, headers={'Range': 'bytes=0-0', 'Accept-Encoding': 'identity'},
                      stream=True, timeout=(20, 45)) as response:
        parsed = urlparse(response.url)
        if (response.status_code != 206 or response.headers.get('Content-Range') != f'bytes 0-0/{total}'
                or response.headers.get('Content-Length') != '1'
                or parsed.scheme != 'https' or parsed.hostname != 'storage.googleapis.com'):
            raise FetchError('Fixed public Kaggle container redirect/range identity could not be verified')
        received = 0
        for block in response.iter_content(chunk_size=1):
            received += len(block)
            if received > 1:
                raise FetchError('Kaggle container range probe exceeded one byte')
        if received != 1:
            raise FetchError('Kaggle container range probe returned an unexpected length')
        return response.url


def container_md5(path):
    value = hashlib.md5(usedforsecurity=False)
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def download_container(cache_root, connections=128, chunk_bytes=8 * 1024**2, retries=6,
                       total=CONTAINER_BYTES, expected_md5=CONTAINER_MD5):
    """Journal the mirror's MD5 container identity, never an original-ZIP SHA."""
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    archive = cache_root / 'rfmid_kaggle_v1.zip'
    partial = cache_root / 'rfmid_kaggle_v1.md5.ranges.partial'
    journal = cache_root / 'rfmid_kaggle_v1.md5.ranges.json'
    lock = cache_root / 'rfmid_kaggle_v1.container.lock'
    for path in (archive, partial, journal, lock):
        if path.is_symlink():
            raise FetchError('Container cache paths must not be symbolic links')
    with ranges.exclusive_source(lock):
        if archive.exists():
            if not archive.is_file() or archive.stat().st_size != total or container_md5(archive) != expected_md5:
                raise FetchError('Existing Kaggle container size/MD5 differs; it was preserved')
            legacy.emit('container_verified', 'RFMiD_Kaggle_v1', bytes=total, container_md5=expected_md5, reused=True)
            return archive
        identity = {'format_version': 1, 'source': 'RFMiD_Kaggle_v1_container', 'bytes': total,
                    'container_md5': expected_md5, 'chunk_bytes': chunk_bytes,
                    'container_is_original_release_zip': False}
        count = (total + chunk_bytes - 1) // chunk_bytes
        if partial.exists() or journal.exists():
            if not partial.is_file() or not journal.is_file() or partial.stat().st_size != total:
                raise FetchError('Incomplete mirror range state; existing files preserved')
            saved = json.loads(journal.read_text(encoding='utf-8'))
            if any(saved.get(key) != value for key, value in identity.items()):
                raise FetchError('Mirror range journal container MD5/size/chunk identity differs')
            completed_list = saved.get('completed_chunks')
            if (not isinstance(completed_list, list) or len(set(completed_list)) != len(completed_list)
                    or any(type(i) is not int or not 0 <= i < count for i in completed_list)):
                raise FetchError('Mirror range journal completed chunk indexes are invalid')
            completed = set(completed_list)
        else:
            with partial.open('xb') as output:
                ranges.mark_sparse(output)
                output.truncate(total)
                output.flush()
                os.fsync(output.fileno())
            completed = set()
            ranges.write_journal(journal, identity, completed)
        source = {'name': 'RFMiD_Kaggle_v1', 'bytes': total}
        pending_indexes = [i for i in range(count) if i not in completed]
        if pending_indexes:
            source['download_url'] = resolve_container_url(total)
        state, stopped = ranges.TransferState(), threading.Event()
        legacy.emit('mirror_download_started', source['name'], connections=connections,
                    completed_chunks=len(completed), total_chunks=count, container_md5=expected_md5)
        with ThreadPoolExecutor(max_workers=connections) as pool:
            pending = {pool.submit(ranges.fetch_chunk, source, i, chunk_bytes, partial, state, retries, stopped)
                       for i in pending_indexes}
            last_tick = time.monotonic()
            try:
                while pending:
                    done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        completed.add(future.result())
                        ranges.write_journal(journal, identity, completed)
                    if time.monotonic() - last_tick >= 30:
                        received = state.snapshot().get(source['name'], {}).get('received_bytes', 0)
                        legacy.emit('mirror_download_progress', source['name'], completed_chunks=len(completed),
                                    total_chunks=count, received_this_session_bytes=received)
                        last_tick = time.monotonic()
            except BaseException:
                stopped.set()
                for future in pending:
                    future.cancel()
                raise
        if len(completed) != count or container_md5(partial) != expected_md5:
            raise FetchError('Downloaded mirror container MD5 mismatch; range state preserved, extraction skipped')
        # Publish without clobbering a file that appeared during the download.
        os.link(partial, archive)
        partial.unlink()
        legacy.emit('container_verified', source['name'], bytes=total, container_md5=expected_md5, reused=False)
        return archive


def publish_noclobber(staged, target, info):
    if target.exists():
        if not matches_file(target, info):
            raise FetchError('Existing RFMiD file differs from original SHA256; it was preserved')
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(staged, target)
    except OSError as error:
        if error.errno == errno.EEXIST:
            if not matches_file(target, info):
                raise FetchError('RFMiD target appeared with different bytes; it was preserved') from None
            return False
        if error.errno != errno.EXDEV:
            raise
        # Cross-filesystem: copy into the destination filesystem, then use its
        # same-filesystem hard link as an atomic no-overwrite publication.
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.' + target.name + '.', suffix='.publish', delete=False) as output:
            temporary = Path(output.name)
            with staged.open('rb') as incoming:
                for block in iter(lambda: incoming.read(1024**2), b''):
                    output.write(block)
        try:
            if not matches_file(temporary, info):
                raise FetchError('Copied publication bytes differ from original SHA256')
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return True


def extract_and_publish(archive, original, data_root, stage_root):
    """All expected ZIP files pass original SHA before publishing any new file."""
    data_root, stage_root = Path(data_root).resolve(), Path(stage_root)
    destination = data_root / 'RFMiD'
    if destination.is_symlink() or stage_root.is_symlink():
        raise FetchError('RFMiD destination/staging root must not be a symbolic link')
    stage_root = stage_root.resolve()
    files = original['files']
    remote_to_original = {relative.split('/', 1)[0] + '/' + relative: relative for relative in files}
    allowed_directories = {str(parent) for remote in remote_to_original for parent in PurePosixPath(remote).parents if str(parent) != '.'}
    with zipfile.ZipFile(archive) as package:
        prepared, seen = [], set()
        for member in package.infolist():
            remote = member.filename
            if (stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1
                    or '\\' in remote or ':' in remote):
                raise FetchError('Mirror ZIP contains an unsupported/symlink/encrypted path')
            if member.is_dir():
                if remote.rstrip('/') not in allowed_directories:
                    raise FetchError('Mirror ZIP contains an unexpected directory')
                continue
            if remote not in remote_to_original or remote in seen:
                raise FetchError('Mirror ZIP contains an unknown/duplicate file')
            seen.add(remote)
            relative = remote_to_original[remote]
            info = files[relative]
            if member.file_size != info['bytes']:
                raise FetchError('Mirror ZIP file size differs from original file manifest')
            target, staged = checked_target(destination, relative), checked_target(stage_root, relative)
            for existing in (target, staged):
                if existing.exists() and not matches_file(existing, info):
                    raise FetchError('Existing destination/staged file conflicts with original SHA256; preserved')
            prepared.append((member, relative, staged, target, info))
        if seen != set(remote_to_original):
            raise FetchError('Mirror ZIP does not contain exactly the complete original-file set')
        # No new RFMiD target files are published in this phase.
        for index, (member, relative, staged, target, info) in enumerate(prepared, 1):
            if not staged.exists():
                staged.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=staged.parent, prefix='.' + staged.name + '.', suffix='.verify', delete=False) as output:
                    temporary = Path(output.name)
                    digest, copied = hashlib.sha256(), 0
                    try:
                        with package.open(member) as incoming:
                            for block in iter(lambda: incoming.read(1024**2), b''):
                                copied += len(block)
                                if copied > info['bytes']:
                                    raise FetchError('Mirror ZIP member exceeds original size')
                                digest.update(block)
                                output.write(block)
                    except BaseException:
                        output.close()
                        temporary.unlink(missing_ok=True)
                        raise
                try:
                    if copied != info['bytes'] or digest.hexdigest() != info['sha256']:
                        raise FetchError('Mirror file SHA256 differs from original; no new RFMiD files published')
                    os.link(temporary, staged)
                finally:
                    temporary.unlink(missing_ok=True)
            if index % 250 == 0 or index == len(prepared):
                legacy.emit('mirror_original_sha256_progress', 'RFMiD', verified_files=index, total_files=len(prepared))
    # Recheck all destination/staging identities before the first publication.
    for _, _, staged, target, info in prepared:
        if not matches_file(staged, info) or (target.exists() and not matches_file(target, info)):
            raise FetchError('Destination/staging changed before publication; original files preserved')
    published = reused = 0
    for _, relative, staged, _, info in prepared:
        target = checked_target(destination, relative)
        if publish_noclobber(staged, target, info):
            published += 1
        else:
            reused += 1
    return {'files_verified_against_original_sha256': len(prepared), 'published_files': published,
            'reused_identical_files': reused, 'all_original_file_sha256_matched': True,
            'path_mapping': 'Remove only the duplicate leading release folder; preserve original canonical RFMiD paths.',
            'image_conversion_performed': False, 'labels_rebuilt': False, 'training_manifests_changed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--original-manifest', type=Path, required=True)
    parser.add_argument('--build-original-manifest', type=Path, help='Local-only preparation: hash this original RFMiD directory and write --original-manifest, then exit')
    parser.add_argument('--connections', type=int, default=128)
    parser.add_argument('--chunk-mib', type=int, default=8)
    parser.add_argument('--retries', type=int, default=6)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.connections <= 256 or not 1 <= args.chunk_mib <= 64 or not 1 <= args.retries <= 20:
        parser.error('connections=1..256, chunk-mib=1..64, retries=1..20')
    try:
        if args.build_original_manifest is not None:
            digest = build_original_manifest(args.build_original_manifest, args.original_manifest)
            legacy.emit('original_manifest_created', 'RFMiD', files=3203, pngs=3200, csvs=3, sha256=digest,
                        network_performed=False, training_performed=False)
            return 0
        if args.data_root is None or args.cache_root is None:
            parser.error('--data-root and --cache-root are required for mirror transport')
        original = read_original_manifest(args.original_manifest)
        if args.plan:
            legacy.emit('rfmid_mirror_plan', 'RFMiD', container_bytes=CONTAINER_BYTES, container_md5=CONTAINER_MD5,
                        original_manifest_sha256=ORIGINAL_MANIFEST_SHA256, expected_files=len(original['files']),
                        connections=args.connections, network_performed=False, writes_performed=False)
            return 0
        cache_root = args.cache_root.resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        # This additional lock spans extraction/publication, beyond the download
        # engine's container lock. It prevents concurrent destination publishing.
        transfer_lock = cache_root / 'rfmid_kaggle_v1.transfer.lock'
        if transfer_lock.is_symlink():
            raise FetchError('Mirror transfer lock must not be a symbolic link')
        with ranges.exclusive_source(transfer_lock):
            archive = download_container(cache_root, args.connections, args.chunk_mib * 1024**2, args.retries)
            result = extract_and_publish(archive, original, args.data_root,
                                         cache_root / ('rfmid_original_stage_' + ORIGINAL_MANIFEST_SHA256[:16]))
            report = {'format_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                      'dataset': 'RFMiD original release', 'transport_source': 'Kaggle andrewmvd/retinal-disease-classification v1',
                      'public_container_entry': PUBLIC_URL, 'container_bytes': CONTAINER_BYTES, 'container_md5': CONTAINER_MD5,
                      'container_is_original_release_zip': False, 'original_manifest_sha256': ORIGINAL_MANIFEST_SHA256,
                      'png_count': 3200, 'csv_count': 3, 'all_checks_passed': True, **result}
            atomic_json(cache_root / 'rfmid_kaggle_v1_transfer_report.json', report)
            legacy.emit('rfmid_original_files_ready', 'RFMiD', **result, all_checks_passed=True)
        return 0
    except BaseException as error:
        legacy.emit('rfmid_mirror_failed', 'RFMiD', error_type=type(error).__name__,
                    reason=str(error) if isinstance(error, FetchError) else 'details suppressed')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
