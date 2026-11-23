"""Read-only SHA-256 verification of original classification image files.

Hashes are pinned by each isolated manifest's image_paths/image_sha256 pairs.
This does not decode pixels, build a cache, alter splits, or start training.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts import dataset_routing as routing
from scripts.manifest_dataset import resolve_data_path
from scripts.package_cloud_training import validate_packaged_leaves


SPLITS = ("train", "val", "test")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _stat_signature(info):
    return (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino)


def _hash_file(relative, path, expected):
    """Check the opened file and path remain the same throughout the read."""
    result = {"path": relative, "expected_sha256": expected, "passed": False}
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Original image reference is not a regular file")
        digest, read_bytes = hashlib.sha256(), 0
        with path.open("rb") as handle:
            if _stat_signature(os.fstat(handle.fileno())) != _stat_signature(before):
                raise ValueError("Original file changed between stat and open")
            while True:
                chunk = handle.read(4 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                read_bytes += len(chunk)
            after_open = os.fstat(handle.fileno())
        after_path = path.stat()
        if (_stat_signature(before) != _stat_signature(after_open) or
                _stat_signature(before) != _stat_signature(after_path) or read_bytes != before.st_size):
            raise ValueError("Original file changed during SHA-256 verification")
        actual = digest.hexdigest()
        result.update(bytes=read_bytes, actual_sha256=actual, stat_stable=True)
        if actual != expected:
            raise ValueError("Original file SHA-256 differs from manifest image_sha256")
        result["passed"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def verify_source_images(config_path, data_root, workers=8):
    """Return a report on success or failure; input files are never modified."""
    started = time.monotonic()
    config_path, data_root = Path(config_path).resolve(), Path(data_root).resolve()
    report = {"format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "all_checks_passed": False, "config": str(config_path), "data_root": str(data_root),
              "algorithm": "SHA-256", "expected_hash_source": "isolated manifest row.image_sha256",
              "splits_checked": list(SPLITS), "workers": workers, "splits": {}, "files": [],
              "unique_files": 0, "verified_files": 0, "verified_bytes": 0, "errors": [],
              "original_inputs_modified": False, "pixels_decoded": False,
              "cache_identity_used_as_original_image_verification": False,
              "formal_training_started": False}
    try:
        if type(workers) is not int or not 1 <= workers <= 128:
            raise ValueError("workers must be an integer in [1,128]")
        config_payload = config_path.read_bytes()
        report["config_sha256"] = _sha(config_payload)
        config = json.loads(config_payload.decode("utf-8-sig"))
        if config.get("kind") not in {"multiclass", "multilabel"}:
            raise ValueError("This original-image verifier supports classification configs only")
        source_config = deepcopy(config)
        report["image_cache_ignored_for_source_routing"] = bool(source_config.get("data", {}).pop("image_cache_dir", None))
        report["source_route"] = routing.validate_training_route(source_config, data_root)
        leaves = validate_packaged_leaves(source_config, data_root)
        report["audited_leaf_integrity"] = leaves
        source = routing.expected_source(source_config, "train")
        report["declared_source"] = source
        expected_files, paths, split_references = {}, {}, {}
        for split in SPLITS:
            manifest = routing.manifest_path(source_config, data_root, split)
            payload = manifest.read_bytes()
            if _sha(payload) != leaves[split]["sha256"]:
                raise ValueError(f"{split} manifest changed after leaf integrity validation")
            identifiers, references, count = set(), [], 0
            for line_number, text in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
                if not text.strip():
                    continue
                row = json.loads(text)
                routing.validate_row(source_config, row, split)
                if row.get("dataset") != source:
                    raise ValueError(f"{split}:{line_number} contains an undeclared source")
                identifier = row.get("sample_id")
                if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                    raise ValueError(f"{split}:{line_number} has an empty or duplicate sample_id")
                identifiers.add(identifier)
                image_paths, hashes = row.get("image_paths"), row.get("image_sha256")
                if (not isinstance(image_paths, list) or not image_paths or not isinstance(hashes, list) or
                        len(image_paths) != len(hashes)):
                    raise ValueError(f"{split}:{line_number} image_paths/image_sha256 must be nonempty aligned lists")
                for relative, expected in zip(image_paths, hashes):
                    if not isinstance(relative, str) or not relative:
                        raise ValueError(f"{split}:{line_number} has an invalid original image path")
                    if not _valid_sha(expected):
                        raise ValueError(f"{split}:{line_number} has an invalid image_sha256")
                    path = resolve_data_path(relative, data_root)
                    if relative in expected_files and expected_files[relative] != expected:
                        raise ValueError(f"Conflicting manifest SHA-256 values for original image {relative}")
                    expected_files[relative], paths[relative] = expected, path
                    references.append(relative)
                count += 1
            if count != leaves[split]["records"]:
                raise ValueError(f"{split} row count differs from its audited leaf")
            split_references[split] = references
            report["splits"][split] = {"records": count, "image_references": len(references),
                                       "unique_files": len(set(references)), "manifest_sha256": leaves[split]["sha256"]}
        if not expected_files:
            raise ValueError("Selected train/val/test manifests have no original image references")
        report["unique_files"] = len(expected_files)
        results = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_hash_file, relative, paths[relative], expected): relative
                       for relative, expected in expected_files.items()}
            for future in as_completed(futures):
                result = future.result()
                results[result["path"]] = result
        report["files"] = [results[relative] for relative in sorted(results)]
        for relative, result in results.items():
            if result["passed"]:
                report["verified_files"] += 1
                report["verified_bytes"] += result["bytes"]
            else:
                report["errors"].append({"path": relative, "message": result["error"]})
        for split, references in split_references.items():
            unique = set(references)
            good = [results[name] for name in unique if results[name]["passed"]]
            report["splits"][split].update(verified_unique_files=len(good),
                verified_unique_bytes=sum(item["bytes"] for item in good),
                all_original_SHA256_matched=len(good) == len(unique),
                all_original_file_stats_stable=all(results[name].get("stat_stable") is True for name in unique))
        # Preserve the manifest/index/config binding while the potentially long
        # image hashing operation was in progress.
        if config_path.read_bytes() != config_payload:
            raise ValueError("Config changed during verification")
        index = routing.isolation_root(source_config, data_root) / "isolation_manifest.json"
        if _sha(index.read_bytes()) != report["source_route"]["isolation_manifest_sha256"]:
            raise ValueError("Isolation index changed during verification")
        for split in SPLITS:
            if _sha(routing.manifest_path(source_config, data_root, split).read_bytes()) != leaves[split]["sha256"]:
                raise ValueError(f"{split} manifest changed during image verification")
        report["all_checks_passed"] = not report["errors"] and report["verified_files"] == len(expected_files)
    except Exception as error:
        report["errors"].append({"message": f"{type(error).__name__}: {error}"})
    report["error_count"] = len(report["errors"])
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


def _write_report(output, report, config_path, data_root):
    output, data_root, config_path = Path(output).resolve(), Path(data_root).resolve(), Path(config_path).resolve()
    if output == config_path or output == data_root or output.is_relative_to(data_root):
        raise ValueError("Verification output must be outside original data_root and must not replace the config")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    report = verify_source_images(args.config, args.data_root, args.workers)
    try:
        _write_report(args.output, report, args.config, args.data_root)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot write verification report: {error}\n")
    print(json.dumps({key: report[key] for key in ("all_checks_passed", "unique_files", "verified_files", "verified_bytes", "error_count", "elapsed_seconds")}, ensure_ascii=False), flush=True)
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
