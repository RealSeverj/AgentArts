"""Read-only verification of the archived DDR/APTOS image references."""
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json

PROJECT = Path(__file__).resolve().parents[1]
BACKUP = PROJECT / 'cloud_backup/20261113_DDR_from_APTOS'
DATA = (PROJECT / 'data').resolve()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    bundle = BACKUP / 'bundle_contents'
    manifest = json.loads((bundle / 'SHA256SUMS.json').read_text(encoding='utf-8-sig'))
    for entry in manifest['files']:
        path = bundle / entry['path']
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], entry['path']
    references = {}
    counts = {}
    for source in ('DDR', 'APTOS2019'):
        counts[source] = {}
        folder = bundle / 'project/data/prepared/source_isolated/strict_cross_domain/tasks/dr_grading' / source
        for split in ('train', 'val', 'test'):
            rows = [json.loads(line) for line in (folder / (split + '.jsonl')).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            counts[source][split] = len(rows)
            for row in rows:
                assert row['dataset'] == source and row['split'] == split and row['task'] == 'dr_grading'
                paths, hashes = row['image_paths'], row['image_sha256']
                assert len(paths) == len(hashes)
                for name, checksum in zip(paths, hashes):
                    relative = PurePosixPath(name.replace('\\', '/'))
                    assert not relative.is_absolute() and '..' not in relative.parts
                    assert isinstance(checksum, str) and len(checksum) == 64
                    if name in references:
                        assert references[name] == checksum
                    references[name] = checksum
    def check(item):
        name, checksum = item
        path = DATA.joinpath(*PurePosixPath(name.replace('\\', '/')).parts)
        assert path.resolve().is_relative_to(DATA)
        before = path.stat()
        assert sha(path) == checksum, 'Original image hash differs: ' + name
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        return before.st_size
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        for index, size in enumerate(pool.map(check, references.items()), 1):
            total_bytes += size
            if index % 4000 == 0:
                print('ORIGINAL_IMAGES_VERIFIED', index, '/', len(references), flush=True)
    result = {'all_checks_passed': True, 'checked_at': datetime.now().astimezone().isoformat(),
              'archive_files_sha256_verified': len(manifest['files']), 'source_split_records': counts,
              'original_images_sha256_verified': len(references), 'original_image_bytes': total_bytes,
              'original_images_directory': str(DATA), 'local_backup_directory': str(BACKUP),
              'scope': 'Selected DDR run and its APTOS source images; raw images remain in data, not duplicated in the backup directory.'}
    (BACKUP / 'pre_cloud_deletion_data_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
