"""Plan or build a portable training archive without changing source data.

No archive is written unless --output is supplied. Only the requested task
manifests and runtime image/annotation files are included; audit source_path
references are intentionally excluded. Paths inside the archive are relative to
the project root, so data-relative manifest paths remain valid on Linux.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import posixpath
import sys
import tarfile
import tempfile
import zipfile

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from scripts import dataset_routing as routing
SPLITS = ("train", "val", "test", "external_test")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("task"), str):
        raise ValueError(f"Config must contain a task name: {path}")
    task = config["task"]
    if not task or task in {".", ".."} or "/" in task or "\\" in task:
        raise ValueError(f"Invalid task name: {task!r}")
    return config


def protocol_root(data_root: Path, config: dict) -> Path:
    protocol = config.get("data", {}).get("protocol", "strict_cross_domain")
    if protocol == "strict_cross_domain":
        return data_root / "prepared"
    if protocol == "within_source":
        return data_root / "prepared/protocols/within_source"
    raise ValueError(f"Unknown data.protocol: {protocol!r}")


def resolve_reference(data_root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError(f"Invalid data-relative path: {value!r}")
    normalized = value.replace("\\", "/")
    if Path(normalized).is_absolute() or PureWindowsPath(normalized).drive:
        raise ValueError(f"Absolute path in manifest: {value!r}")
    target = (data_root / normalized).resolve()
    if not target.is_relative_to(data_root.resolve()):
        raise ValueError(f"Path leaves data root: {value!r}")
    return target


def archive_reference(value: str) -> str:
    """Preserve manifest case even when Windows resolves a different spelling.

    Call resolve_reference first for containment checks. The archive must use
    the spelling that Linux readers will request, rather than the resolved
    source file's case-insensitive Windows spelling.
    """
    return "data/" + posixpath.normpath(value.replace("\\", "/"))


def runtime_annotations(value):
    """Yield training mask and XML references, excluding provenance-only files."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"path", "xml_path"} and isinstance(child, str):
                yield child
            elif key not in {"source_path", "rasterization_reference"}:
                yield from runtime_annotations(child)
    elif isinstance(value, list):
        for child in value:
            yield from runtime_annotations(child)


def selected_rows(manifest: Path, config: dict, split: str | None = None):
    sources = config.get("data", {}).get("datasets")
    if sources is not None and (not isinstance(sources, list) or not all(isinstance(x, str) for x in sources)):
        raise ValueError("data.datasets must be a list of source names")
    with manifest.open(encoding="utf-8-sig") as handle:
        for number, text in enumerate(handle, 1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
            except ValueError as error:
                raise ValueError(f"Malformed JSONL at {manifest}:{number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Non-object record at {manifest}:{number}")
            if routing.is_source_isolated(config):
                # A wrong source must fail before any old dataset filter can
                # silently discard it and hide a contaminated leaf manifest.
                routing.validate_row(config, row, split or manifest.stem)
                yield row
            elif sources is None or row.get("dataset") in sources:
                yield row


def validate_packaged_leaves(config, data_root):
    """Bind every routed split, including an explicit external source, to index."""
    if not routing.is_source_isolated(config):
        raise ValueError("Fresh cloud training requires a source-isolated route")
    root = routing.isolation_root(config, data_root)
    index = json.loads((root / "isolation_manifest.json").read_text(encoding="utf-8-sig"))
    entries = {info["path"]: info for combo in index["combinations"] for info in combo["splits"].values()}
    result = {}
    for split in SPLITS:
        path = routing.manifest_path(config, data_root, split)
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        records = sum(bool(line.strip()) for line in content.splitlines())
        expected = entries.get(relative)
        if (not expected or expected.get("sha256") != digest or expected.get("bytes") != len(content)
                or expected.get("records") != records):
            raise ValueError(f"Routed {split} leaf differs from its audited isolation index")
        result[split] = {"path": relative, "records": records, "sha256": digest}
    return result


def code_files(project: Path):
    required = ["requirements-training.txt", "CLOUD_TRAINING.md",
                "scripts/manifest_dataset.py", "scripts/validate_training_sets.py",
                "scripts/check_manifest_reader.py", "scripts/cloud_preflight.py",
                "scripts/package_cloud_training.py", "scripts/setup_cloud.sh", "scripts/launch_v100.sh",
                "scripts/check_cloud_gpu.py", "scripts/profile_dr_input.py", "scripts/build_classification_cache.py"]
    required += ["scripts/dataset_routing.py", "scripts/isolate_training_datasets.py", "scripts/generate_source_configs.py", "scripts/check_expert_suite.py"]
    for relative in required:
        path = project / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required training file is missing: {path}")
        yield path, relative
    for directory, extensions in [("training", {".py", ".md"}), ("configs", {".json", ".yaml", ".yml"}), ("tests", {".py"})]:
        folder = project / directory
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing training directory: {folder}")
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix in extensions and "__pycache__" not in path.parts:
                yield path, path.relative_to(project).as_posix()
    verification = project / "TRAINING_VERIFICATION.json"
    if verification.is_file():
        yield verification, verification.name


def make_plan(project: Path, data_root: Path, config_paths: list[Path], code_only: bool):
    project, data_root = Path(project).resolve(), Path(data_root).resolve()
    files = {relative: path for path, relative in code_files(project)}
    records, sources = {}, set()
    missing = []
    reference_cache = {}
    configs = []
    for path in config_paths:
        config = load_config(path)
        if not path.resolve().is_relative_to(project.resolve()):
            raise ValueError("Package configs must be inside the project so their paths are portable")
        files[path.relative_to(project).as_posix()] = path
        configs.append({"path": path.relative_to(project).as_posix(), "task": config["task"],
                        "protocol": config.get("data", {}).get("protocol", "strict_cross_domain"),
                        "datasets": config.get("data", {}).get("datasets"),
                        "source_isolated": routing.is_source_isolated(config),
                        "external_dataset": config.get("data", {}).get("external_dataset")})
        if code_only:
            continue
        # Fresh data uploads must use the same source/task guard as training.
        # Legacy configurations remain in code-only archives for past evals.
        # Uploads contain original inputs, not the derived cache that will be
        # built on the destination host. A cloud-only cache path may not exist
        # on the packaging machine. Still validate all source/manifest guards.
        source_config = deepcopy(config)
        source_config.get('data', {}).pop('image_cache_dir', None)
        route = routing.validate_training_route(source_config, data_root)
        configs[-1]["training_route"] = json.loads(json.dumps(route, default=str))
        configs[-1]["leaf_integrity"] = validate_packaged_leaves(config, data_root)
        index = routing.isolation_root(config, data_root) / "isolation_manifest.json"
        if not index.is_file() or not index.resolve().is_relative_to(data_root):
            raise ValueError("Cloud training packages require an isolation index inside data_root")
        files["data/" + index.relative_to(data_root).as_posix()] = index
        configs[-1]["isolation_manifest_sha256"] = hashlib.sha256(index.read_bytes()).hexdigest()
        root = protocol_root(data_root, config)
        for name in ("label_schema.json", "summary.json", "split_counts.csv", "class_counts.csv", "split_policy.json", "README.md"):
            candidate = root / name
            if candidate.is_file():
                files["data/" + candidate.relative_to(data_root).as_posix()] = candidate
        # The root policy documents the relationship between both protocols.
        for name in ("split_policy.json", "README.md"):
            candidate = data_root / "prepared" / name
            if candidate.is_file():
                files["data/" + candidate.relative_to(data_root).as_posix()] = candidate
        for split in SPLITS:
            manifest = routing.manifest_path(config, data_root, split)
            if not manifest.resolve().is_relative_to(data_root):
                raise ValueError("Cloud training task manifests must be inside data_root")
            if not manifest.is_file():
                raise FileNotFoundError(f"Task manifest is missing: {manifest}")
            files["data/" + manifest.relative_to(data_root).as_posix()] = manifest
            count = 0
            for row in selected_rows(manifest, config, split):
                count += 1
                sources.add(row["dataset"])
                paths = row.get("image_paths") or ([row["image_path"]] if row.get("image_path") else [])
                if not paths:
                    raise ValueError(f"No image paths for {row.get('sample_id')}")
                for value in [*paths, *runtime_annotations(row.get("annotations", {}))]:
                    if value not in reference_cache:
                        target = resolve_reference(data_root, value)
                        reference_cache[value] = (target, target.is_file())
                    target, exists = reference_cache[value]
                    relative = archive_reference(value)
                    if exists:
                        files[relative] = target
                    else:
                        missing.append(relative)
            records[f"{config_paths.index(path)}:{config['task']}:{split}"] = count
    # Keep compact local provenance beside the selected image sources.
    for source in sources:
        for name in ("README.md", "LICENSE", "LICENSE.txt", "LICENSE.md", "source_metadata.json"):
            candidate = data_root / source / name
            if candidate.is_file() and candidate.stat().st_size < 2 * 1024 * 1024:
                files["data/" + candidate.relative_to(data_root).as_posix()] = candidate
    if missing:
        raise FileNotFoundError(f"{len(set(missing))} required data files missing; examples: {sorted(set(missing))[:8]}")
    entries = [{"path": relative, "bytes": path.stat().st_size} for relative, path in sorted(files.items())]
    kinds = Counter()
    for entry in entries:
        group = "data" if entry["path"].startswith("data/") else "code"
        kinds[group] += entry["bytes"]
    total = sum(entry["bytes"] for entry in entries)
    report = {"format_version": 1, "code_only": code_only, "configs": configs,
              "task_records_selected": records, "file_count": len(entries), "bytes_uncompressed": total,
              "gib_uncompressed": round(total / 1024**3, 3), "bytes_by_category": dict(kinds),
              "recommended_free_gib_for_archive_and_extraction": round(2 * total / 1024**3 + 20, 1),
              "notes": ["Data source paths and manifest records are preserved; no repartition or relabeling.",
                        "Archive data entry spelling preserves manifest path case for Linux compatibility.",
                        "Only selected tasks are packaged. Use cloud_preflight, not full-protocol validate_training_sets, on this subset.",
                        "Fresh data packages contain only explicitly routed task/source leaves; unexpected source records are rejected before filtering.",
                        "Multiple protocols in one archive are separate experiment collections, never one mixed-protocol training plan or suite.",
                        "Legacy configs may be present as historical code artifacts; new training requires explicit source-isolated routing.",
                        "Provenance source_path files, archives, staging and fingerprint databases are not runtime dependencies.",
                        "Checkpoint/model downloads and output runs need additional disk space."],
              "files": entries}
    return report, files


def write_archive(output: Path, report: dict, files: dict[str, Path]):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing archive: {output}")
    if not (output.name.lower().endswith(".zip") or output.name.lower().endswith(".tar.gz")):
        raise ValueError("--output must end in .zip or .tar.gz")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.stem + ".", suffix=".partial", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        payload = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        if output.name.lower().endswith(".zip"):
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                for relative, path in sorted(files.items()):
                    archive.write(path, relative)
                archive.writestr("cloud_package_manifest.json", payload)
        else:
            import io
            with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
                for relative, path in sorted(files.items()):
                    archive.add(path, arcname=relative, recursive=False)
                item = tarfile.TarInfo("cloud_package_manifest.json")
                item.size = len(payload)
                archive.addfile(item, io.BytesIO(payload))
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=BASE)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", type=Path, action="append", help="Repeat to include multiple task configurations")
    parser.add_argument("--all-configs", action="store_true", help="Include runnable configs recursively under configs/source_isolated only")
    parser.add_argument("--code-only", action="store_true", help="Exclude all images, masks and manifests")
    parser.add_argument("--output", type=Path, help="Create .zip or .tar.gz; omitted means dry-run only")
    parser.add_argument("--manifest-out", type=Path, help="Optional detailed JSON plan, written only when explicitly requested")
    args = parser.parse_args()
    project = args.project_root.resolve()
    data_root = args.data_root.resolve() if args.data_root else project / "data"
    if args.all_configs and args.config:
        parser.error("Choose --all-configs or --config, not both")
    if args.all_configs:
        configs = []
        for path in sorted((project / "configs/source_isolated").rglob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(document, dict) and "task" in document:
                configs.append(path)
    else:
        configs = args.config or [Path("configs/source_isolated/strict_cross_domain/dr_grading_ddr.json")]
    configs = [(path if path.is_absolute() else project / path).resolve() for path in configs]
    if not configs:
        parser.error("No task configurations found")
    try:
        report, files = make_plan(project, data_root, configs, args.code_only)
        if args.manifest_out:
            args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
            args.manifest_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.output:
            write_archive(args.output, report, files)
        summary = {key: value for key, value in report.items() if key != "files"}
        summary.update({"dry_run": not bool(args.output), "archive": str(args.output.resolve()) if args.output else None})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Package error: {error}\n")


if __name__ == "__main__":
    main()
