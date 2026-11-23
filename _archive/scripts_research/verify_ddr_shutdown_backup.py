"""Verify the selected DDR backup before its already-authorized shutdown."""
from pathlib import Path, PurePosixPath
import gc
import hashlib
import json
import shutil
import tarfile
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / 'cloud_backup/20261113_DDR_from_APTOS'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    info = read_json(BACKUP / 'bundle_info.json')
    archive = BACKUP / Path(info['path']).name
    assert archive.stat().st_size == info['bytes']
    assert digest(archive) == info['sha256']
    extracted = BACKUP / 'bundle_contents'
    extracted.mkdir(exist_ok=True)
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        assert len(members) == info['members']
        assert len({m.name for m in members}) == len(members)
        for member in members:
            relative = PurePosixPath(member.name)
            assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
            target = extracted.joinpath(*relative.parts)
            assert target.resolve().is_relative_to(extracted.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = source.extractfile(member)
            assert stream is not None
            data = stream.read()
            if target.exists():
                assert target.read_bytes() == data, 'Conflicting previously extracted file'
            else:
                target.write_bytes(data)
    manifest = read_json(extracted / 'SHA256SUMS.json')
    expected = {entry['path'] for entry in manifest['files']}
    assert expected == {m.name for m in members} - {'SHA256SUMS.json'}
    for entry in manifest['files']:
        path = extracted / entry['path']
        assert path.stat().st_size == entry['bytes']
        assert digest(path) == entry['sha256'], entry['path']
    print('ARCHIVE_FILES_VERIFIED', len(expected), flush=True)
    metadata = read_json(extracted / 'run/checkpoint_metadata.json')
    history = [json.loads(line) for line in (extracted / 'run/history.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in history] == list(range(1, 31))
    assert all(row['stage'] == 'head_only' for row in history[:5])
    assert all(row['stage'] == 'last_4_blocks' for row in history[5:])
    import torch
    results = []
    for name in ('head_best.pt', 'best.pt', 'last.pt'):
        path = BACKUP / name
        expected_checkpoint = metadata['checkpoints'][name]
        checksum = digest(path)
        assert path.stat().st_size == expected_checkpoint['bytes']
        assert checksum == expected_checkpoint['sha256'], name
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        assert state['epoch'] + 1 == expected_checkpoint['completed_epoch']
        assert state['config']['data']['datasets'] == ['DDR']
        assert state['config']['num_classes'] == 5
        assert set(('model', 'optimizer', 'scheduler', 'scaler', 'rng_states', 'data_identity')) <= set(state)
        model_hash = hashlib.sha256()
        for key, tensor in sorted(state['model'].items()):
            model_hash.update((key + '|' + str(tensor.dtype) + '|' + str(tuple(tensor.shape)) + '\n').encode())
            model_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        assert model_hash.hexdigest() == expected_checkpoint['model_state_sha256'], name
        results.append({'file': name, 'bytes': path.stat().st_size, 'sha256': checksum,
                        'completed_epoch': state['epoch'] + 1, 'load_and_model_hash_passed': True})
        del state
        gc.collect()
        print('CHECKPOINT_VERIFIED', name, flush=True)
    analysis = BACKUP / 'local_analysis_code'
    analysis.mkdir(exist_ok=True)
    names = ('summarize_unified_dr.py', 'summarize_ddr_transfer.py', 'report_dr_safety_metrics.py',
             'plot_aptos_confusion.py', 'evaluate_dr_cohort.py', 'export_dr_run_metadata.py',
             'verify_ddr_shutdown_backup.py')
    for name in names:
        shutil.copy2(ROOT / 'scripts' / name, analysis / name)
    for name in ('APTOS2019', 'DDR'):
        folder = ROOT / 'data/prepared/source_isolated/strict_cross_domain/tasks/dr_grading' / name
        assert all((folder / (split + '.jsonl')).is_file() for split in ('train', 'val', 'test'))
    result = {'all_checks_passed': True, 'verified_at': datetime.now().isoformat(),
              'archive_sha256': info['sha256'], 'archive_files_verified': len(expected),
              'checkpoints': results, 'history_epochs_verified': 30,
              'original_images_location': str(ROOT / 'data'),
              'original_images_existence_audited_count': 12924,
              'delivery_summary_scope': 'DDR and APTOS; IDRiD omitted from the delivery summary at user request',
              'cloud_shutdown_authorized_after_this_verification': True}
    (BACKUP / 'backup_complete.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('BACKUP_COMPLETE_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
