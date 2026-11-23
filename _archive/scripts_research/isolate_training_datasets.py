"""Build single-source, single-task manifests without changing any records.

Both existing preparation protocols are treated as legacy mixed roots. Their
alternative splits are audited separately, never merged or re-partitioned.
Only metadata is read; source images and annotations are not opened or moved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

BASE = Path(__file__).resolve().parents[1]
PROTOCOLS = ('strict_cross_domain', 'within_source')
SPLITS = ('train', 'val', 'test', 'external_test')
SAFE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
VERSION = 1


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def member_digest(counter):
    digest = hashlib.sha256()
    for sample, count in sorted(counter.items()):
        digest.update(json.dumps([sample, count], ensure_ascii=False, separators=(',', ':')).encode('utf-8') + b'\n')
    return digest.hexdigest()


def protocol_root(data_root, protocol):
    root = data_root / 'prepared'
    return root if protocol == 'strict_cross_domain' else root / 'protocols/within_source'


def validate_output_target(data_root, output_root):
    data_root, output_root = data_root.resolve(), output_root.resolve()
    for protocol in PROTOCOLS:
        original = protocol_root(data_root, protocol).resolve()
        if output_root == original or original.is_relative_to(output_root) or output_root.is_relative_to(original / 'tasks'):
            raise ValueError('Output must be separate from legacy roots and legacy tasks directories')
    return output_root


def canonical_image_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError('Image path must be a nonempty data-relative string')
    text = value.replace('\\', '/')
    relative = PurePosixPath(text)
    if relative.is_absolute() or '..' in relative.parts or PureWindowsPath(text).drive or '://' in text:
        raise ValueError('Image path is absolute or leaves data root')
    # Case folding also catches a Windows alias that would be ambiguous after a
    # Linux upload. No filesystem lookup, image decode, or content edit occurs.
    return relative.as_posix().casefold()


def supported_annotation(label, annotations):
    masks, experts, points = (annotations.get(name, {}) for name in ('masks', 'experts', 'points'))
    if isinstance(masks, dict) and masks.get(label):
        return True
    if isinstance(experts, dict) and any(isinstance(expert, dict) and expert.get(label) for expert in experts.values()):
        return True
    if isinstance(points, dict) and points.get(label):
        return True
    return label == 'objects' and bool(annotations.get('xml_path'))


class Planner:
    def __init__(self, data_root, protocols, max_details=100):
        self.data_root, self.protocols = Path(data_root).resolve(), tuple(protocols)
        self.max_details = max_details
        self.error_counts, self.errors = Counter(), []
        self.inputs, self.groups, self.combinations = [], {}, set()
        self.members = defaultdict(Counter)
        self.leakage_checked = {}

    def error(self, code, context=None):
        self.error_counts[code] += 1
        if len(self.errors) < self.max_details:
            self.errors.append({'code': code, **(context or {})})

    def scan(self):
        portable_names = {}
        for protocol in self.protocols:
            root = protocol_root(self.data_root, protocol)
            task_root = root / 'tasks'
            if not task_root.is_dir():
                self.error('missing_legacy_task_root', {'protocol': protocol})
                continue
            trackers = defaultdict(dict)
            reported_leaks = set()
            def track(category, identity, split, context):
                if identity is None:
                    return
                previous = trackers[category].setdefault(identity, {})
                for earlier, first in previous.items():
                    if earlier != split:
                        key = (category, identity, tuple(sorted((earlier, split))))
                        if key not in reported_leaks:
                            reported_leaks.add(key)
                            self.error(category + '_cross_split_leakage', {'protocol': protocol, 'first': first, 'other': context})
                previous.setdefault(split, context)
            tasks = sorted(path for path in task_root.iterdir() if path.is_dir())
            if not tasks:
                self.error('empty_legacy_task_root', {'protocol': protocol})
            for task_dir in tasks:
                task = task_dir.name
                if not SAFE_NAME.fullmatch(task):
                    self.error('unsafe_task_name', {'protocol': protocol, 'task': task})
                    continue
                for extra in task_dir.glob('*.jsonl'):
                    if extra.stem not in SPLITS:
                        self.error('unexpected_split_manifest', {'path': extra.relative_to(self.data_root).as_posix()})
                for split in SPLITS:
                    path = task_dir / (split + '.jsonl')
                    if not path.is_file():
                        self.error('missing_legacy_split_manifest', {'protocol': protocol, 'task': task, 'split': split})
                        continue
                    whole, file_ids, records, blank_lines = hashlib.sha256(), set(), 0, 0
                    with path.open('rb') as handle:
                        for number, raw in enumerate(handle, 1):
                            whole.update(raw)
                            if not raw.strip():
                                blank_lines += 1
                                continue
                            context = {'protocol': protocol, 'task': task, 'split': split, 'line': number,
                                       'manifest': path.relative_to(self.data_root).as_posix()}
                            try:
                                row = json.loads(raw)
                                if not isinstance(row, dict):
                                    raise ValueError('non-object record')
                            except (ValueError, UnicodeError):
                                self.error('invalid_json_record', context)
                                continue
                            source, sample = row.get('dataset'), row.get('sample_id')
                            context = {**context, 'sample_id': sample, 'dataset': source}
                            if row.get('task') != task or row.get('split') != split:
                                self.error('record_task_or_split_mismatch', context)
                            if not isinstance(source, str) or not SAFE_NAME.fullmatch(source):
                                self.error('unsafe_source_name', context)
                                continue
                            if not isinstance(sample, str) or not sample:
                                self.error('invalid_sample_id', context)
                                continue
                            if sample in file_ids:
                                self.error('duplicate_sample_id_within_file', context)
                            file_ids.add(sample)
                            portable = (protocol, task.casefold(), source.casefold())
                            names = (task, source)
                            if portable in portable_names and portable_names[portable] != names:
                                self.error('portable_task_source_case_collision', context)
                            portable_names[portable] = names
                            self.combinations.add((protocol, task, source))
                            key = (protocol, task, source, split)
                            group = self.groups.setdefault(key, {'sha': hashlib.sha256(), 'records': 0,
                                                                  'bytes': 0, 'supervised': 0, 'labels': Counter()})
                            group['sha'].update(raw)
                            group['records'] += 1
                            group['bytes'] += len(raw)
                            self.members[(protocol, split)][sample] += 1
                            records += 1
                            labels, masks, annotations = row.get('labels', {}), row.get('label_mask', {}), row.get('annotations', {})
                            if not all(isinstance(v, dict) for v in (labels, masks, annotations)):
                                self.error('invalid_supervision_containers', context)
                                labels, masks, annotations = {}, {}, {}
                            effective = []
                            for label, known in masks.items():
                                if not isinstance(known, (bool, int)) or known not in (0, 1):
                                    self.error('invalid_known_label_mask', context)
                                elif known:
                                    scalar = labels.get(label)
                                    if scalar is not None and not (isinstance(scalar, float) and not math.isfinite(scalar)):
                                        effective.append(label)
                                    elif supported_annotation(label, annotations):
                                        effective.append(label)
                                    else:
                                        self.error('known_supervision_has_no_target', {**context, 'label': label})
                            if effective:
                                group['supervised'] += 1
                                group['labels'].update(effective)
                            paths = row.get('image_paths') or ([row['image_path']] if row.get('image_path') else [])
                            if not isinstance(paths, list) or not paths:
                                self.error('invalid_image_paths', context)
                                paths = []
                            for image in paths:
                                try:
                                    canonical = canonical_image_path(image)
                                    track('same_source_image_path', (source, canonical), split, context)
                                except ValueError:
                                    self.error('unsafe_image_path', context)
                            patient = row.get('patient_id')
                            if patient is not None:
                                if not isinstance(patient, str) or not patient:
                                    self.error('invalid_patient_id', context)
                                else:
                                    track('dataset_scoped_patient', (source, patient), split, context)
                            for field, category in [('leakage_group_id', 'global_leakage_group'), ('content_group_id', 'global_content_group')]:
                                value = row.get(field)
                                if value is not None:
                                    if not isinstance(value, str) or not value:
                                        self.error('invalid_' + field, context)
                                    else:
                                        track(category, value, split, context)
                            hashes = row.get('image_sha256')
                            if hashes is not None:
                                if not isinstance(hashes, list) or len(hashes) != len(paths):
                                    self.error('invalid_image_hash_array', context)
                                else:
                                    for digest in hashes:
                                        if not isinstance(digest, str) or not re.fullmatch('[0-9a-fA-F]{64}', digest):
                                            self.error('invalid_image_sha256', context)
                                        else:
                                            track('global_image_content_sha256', digest.lower(), split, context)
                    self.inputs.append({'protocol': protocol, 'task': task, 'split': split,
                                        'path': path.relative_to(self.data_root).as_posix(), 'sha256': whole.hexdigest(),
                                        'records': records, 'blank_lines': blank_lines, 'bytes': path.stat().st_size})
            self.leakage_checked[protocol] = {kind: len(values) for kind, values in trackers.items()}
        for protocol, task, source in self.combinations:
            for split in SPLITS:
                self.groups.setdefault((protocol, task, source, split), {'sha': hashlib.sha256(), 'records': 0,
                                                                       'bytes': 0, 'supervised': 0, 'labels': Counter()})
        return self

    def manifest(self):
        identity = {'builder_version': VERSION, 'base_protocols': list(self.protocols), 'input_files': self.inputs}
        identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        combinations = []
        for protocol, task, source in sorted(self.combinations):
            files = {}
            for split in SPLITS:
                group = self.groups[(protocol, task, source, split)]
                relative = '/'.join((protocol, 'tasks', task, source, split + '.jsonl'))
                files[split] = {'path': relative, 'records': group['records'], 'bytes': group['bytes'],
                                'sha256': group['sha'].hexdigest(), 'supervised_records': group['supervised'],
                                'known_supervision_by_label': dict(sorted(group['labels'].items()))}
            reasons = []
            if not files['train']['supervised_records']:
                reasons.append('no_effective_training_supervision')
            if not files['val']['supervised_records']:
                reasons.append('no_effective_validation_supervision')
            combinations.append({'base_protocol': protocol, 'task': task, 'source': source,
                                 'trainable': not reasons, 'non_trainable_reasons': reasons, 'splits': files})
        return {'format_version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'identity': identity, 'identity_sha256': identity_sha,
                'all_checks_passed': not self.error_counts, 'errors_total': sum(self.error_counts.values()),
                'error_counts': dict(self.error_counts), 'errors': self.errors,
                'base_protocols': list(self.protocols), 'record_bytes_preserved': True,
                'legacy_mixed_roots': [{'base_protocol': p, 'path': str(protocol_root(self.data_root, p)),
                                        'legacy': True, 'status': 'legacy_mixed_task_root',
                                        'instruction': 'Use isolated task/source manifests for new training; never mix alternative protocols.'} for p in self.protocols],
                'combinations': combinations, 'combination_count': len(combinations),
                'trainable_combinations': sum(item['trainable'] for item in combinations),
                'global_split_membership': {p: {s: {'records': sum(self.members[(p, s)].values()),
                                                        'unique_sample_ids': len(self.members[(p, s)]),
                                                        'sample_id_multiset_sha256': member_digest(self.members[(p, s)])} for s in SPLITS} for p in self.protocols},
                'leakage_identity_counts': self.leakage_checked,
                'limitations': ['Leakage checks compare declared hashes/group identities and canonical metadata paths; source image bytes are not read.',
                               'Patient IDs are scoped by dataset; absent IDs do not prove clinical independence.',
                               'Alternative base protocols are audited independently and cannot be combined.',
                               'Trainable means nonzero effective supervision in both train and val; it does not establish class balance, label compatibility or model accuracy.']}


def output_files(manifest):
    return {info['path']: info for combination in manifest['combinations'] for info in combination['splits'].values()}


def verify_outputs(output_root, manifest):
    expected = output_files(manifest)
    actual = {path.relative_to(output_root).as_posix() for path in output_root.rglob('*') if path.is_file()}
    extras = actual - set(expected) - {'isolation_manifest.json'}
    if extras:
        raise ValueError('Isolation directory contains unexpected files; refusing reuse')
    observed = defaultdict(Counter)
    for relative, info in expected.items():
        path = output_root / relative
        if not path.is_file() or path.stat().st_size != info['bytes'] or digest_file(path) != info['sha256']:
            raise ValueError('Isolated output bytes/hash differ from the planned source record stream')
        protocol, _, task, source, filename = PurePosixPath(relative).parts
        count = 0
        with path.open('rb') as handle:
            for raw in handle:
                if raw.strip():
                    row = json.loads(raw)
                    if row.get('task') != task or row.get('dataset') != source or row.get('split') != filename[:-6]:
                        raise ValueError('Isolated file contains records from another task, source or split')
                    observed[(protocol, filename[:-6])][row['sample_id']] += 1
                    count += 1
        if count != info['records']:
            raise ValueError('Isolated file record count differs from source plan')
    for protocol in manifest['base_protocols']:
        for split in SPLITS:
            expected_member = manifest['global_split_membership'][protocol][split]
            counter = observed[(protocol, split)]
            if (sum(counter.values()) != expected_member['records']
                    or member_digest(counter) != expected_member['sample_id_multiset_sha256']):
                raise ValueError('Copied global sample-ID multiset differs from legacy source split')


def build_isolation(data_root, output_root, protocols=PROTOCOLS, dry_run=False):
    data_root = Path(data_root).resolve()
    protocols = tuple(protocols)
    if not protocols or len(set(protocols)) != len(protocols) or any(p not in PROTOCOLS for p in protocols):
        raise ValueError('Select distinct known base protocols only')
    output_root = validate_output_target(data_root, Path(output_root))
    planner = Planner(data_root, protocols).scan()
    manifest = planner.manifest()
    manifest['output_root'] = str(output_root)
    manifest['dry_run'] = bool(dry_run)
    if not manifest['all_checks_passed']:
        manifest['publication_status'] = 'audit_failed_no_output_written'
        return manifest
    if output_root.exists():
        existing_path = output_root / 'isolation_manifest.json'
        if not output_root.is_dir() or not existing_path.is_file():
            raise ValueError('Output already exists without an isolation identity; refusing overwrite')
        existing = json.loads(existing_path.read_text(encoding='utf-8-sig'))
        if existing.get('identity_sha256') != manifest['identity_sha256']:
            raise ValueError('Output has another isolation identity; refusing overwrite')
        for field in ('format_version', 'identity', 'base_protocols', 'record_bytes_preserved', 'combinations', 'global_split_membership'):
            if existing.get(field) != manifest[field]:
                raise ValueError('Existing isolation manifest metadata differs from its input identity; refusing reuse')
        verify_outputs(output_root, manifest)
        manifest['publication_status'] = 'existing_identical_output_reused_read_only'
        return manifest
    if dry_run:
        manifest['publication_status'] = 'dry_run_passed_no_output_written'
        return manifest
    # Reserve a NEW root. Files publish atomically one by one; the manifest is
    # the final commit marker. On failure no originals are touched and an
    # incomplete new root is preserved for review, never recursively deleted.
    output_root.mkdir(parents=True, exist_ok=False)
    expected = output_files(manifest)
    for relative in expected:
        temporary = output_root / (relative + '.tmp')
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open('xb'):
            pass
    for source_file in planner.inputs:
        path = data_root / source_file['path']
        opened, checksum = {}, hashlib.sha256()
        try:
            with path.open('rb') as handle:
                for raw in handle:
                    checksum.update(raw)
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    relative = '/'.join((source_file['protocol'], 'tasks', row['task'], row['dataset'], row['split'] + '.jsonl'))
                    if relative not in expected:
                        raise ValueError('Legacy input changed between audit and copying')
                    if relative not in opened:
                        opened[relative] = (output_root / (relative + '.tmp')).open('ab')
                    opened[relative].write(raw)
        finally:
            for handle in opened.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
        if checksum.hexdigest() != source_file['sha256']:
            raise ValueError('Legacy input bytes changed between audit and copying')
    for relative in expected:
        temporary, final = output_root / (relative + '.tmp'), output_root / relative
        if final.exists():
            raise ValueError('Unexpected final output appeared; refusing overwrite')
        os.replace(temporary, final)
    verify_outputs(output_root, manifest)
    manifest['publication_status'] = 'published_and_verified'
    temporary = output_root / 'isolation_manifest.json.tmp'
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_root / 'isolation_manifest.json')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=BASE / 'data')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--protocol', choices=['both', *PROTOCOLS], default='both')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    protocols = PROTOCOLS if args.protocol == 'both' else (args.protocol,)
    output = args.output_root or args.data_root / 'prepared/source_isolated'
    try:
        result = build_isolation(args.data_root, output, protocols, args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if result['all_checks_passed'] else 1
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({'all_checks_passed': False, 'error': str(error), 'error_type': type(error).__name__}, ensure_ascii=False), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
