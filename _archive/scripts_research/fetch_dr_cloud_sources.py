"""Resume exact DR source ZIPs, verify SHA256, then safely extract them.

Input metadata may contain temporary signed URLs. No URL, response body, HTTP
exception text, or metadata dump is printed. Existing different data files are
never overwritten. This program does not alter the prepared task manifests.
Requires requests (python -m pip install requests).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import threading
import time
from urllib.parse import urlparse
import zipfile
import zlib

import requests


class FetchError(RuntimeError):
    """A message constructed exclusively from safe local details."""


class Progress:
    def __init__(self):
        self.lock = threading.Lock()
        self.values = {}

    def set(self, name, **fields):
        with self.lock:
            self.values.setdefault(name, {}).update(fields)

    def snapshot(self):
        with self.lock:
            return {name: dict(values) for name, values in self.values.items()}


def emit(event, source=None, **fields):
    value = {"event": event}
    if source is not None:
        value["source"] = source
    value.update(fields)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def relative_path(value, what):
    if not isinstance(value, str) or not value:
        raise FetchError(f"{what} must be a nonempty relative path")
    text = value.replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or PureWindowsPath(text).drive or ".." in path.parts or any(":" in part for part in path.parts):
        raise FetchError(f"Unsafe relative path in {what}")
    return path


def contained(root, relative):
    destination = root.joinpath(*relative.parts).resolve()
    if not destination.is_relative_to(root):
        raise FetchError("Destination leaves the selected local directory")
    return destination


def read_metadata(path, names):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("format_version") != 1 or not isinstance(data.get("sources"), list):
        raise FetchError("Metadata requires format_version=1 and a sources array")
    selected = []
    seen_names, seen_archives = set(), set()
    for item in data["sources"]:
        name = item.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise FetchError("Source names may contain only letters, digits, underscore, period and hyphen")
        if name in seen_names:
            raise FetchError("Duplicate source name in metadata")
        seen_names.add(name)
        if names and name not in names:
            continue
        size, checksum = item.get("bytes"), item.get("sha256", "")
        if not isinstance(size, int) or size <= 0 or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
            raise FetchError("Invalid source size or SHA256")
        url = item.get("download_url")
        if not isinstance(url, str) or urlparse(url).scheme not in {"https", "http"}:
            raise FetchError("Source requires an HTTP(S) download_url")
        archive = relative_path(item.get("archive_name"), "archive_name")
        if len(archive.parts) != 1 or not archive.name.lower().endswith(".zip"):
            raise FetchError("archive_name must be one ZIP filename")
        if archive.name in seen_archives:
            raise FetchError("Archive filenames must be unique")
        seen_archives.add(archive.name)
        relative_path(item.get("extract_root"), "extract_root")
        selected.append(item)
    missing = set(names) - seen_names
    if missing:
        raise FetchError("Requested source names are absent: " + ", ".join(sorted(missing)))
    if not selected:
        raise FetchError("No sources selected")
    # Small sources finish first, exposing a routing/extraction problem before
    # spending hours on the two larger archives.
    return sorted(selected, key=lambda value: value["bytes"])


def sha256_file(path, name, progress):
    digest, done = hashlib.sha256(), 0
    progress.set(name, phase="verifying_sha256", verified_bytes=0)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
            done += len(block)
            progress.set(name, verified_bytes=done)
    return digest.hexdigest()


def download(source, cache_root, progress, retries):
    name, size = source["name"], source["bytes"]
    archive = contained(cache_root, relative_path(source["archive_name"], "archive_name"))
    partial = archive.with_name(archive.name + ".part")
    if archive.is_file():
        if archive.stat().st_size != size:
            raise FetchError("Existing ZIP has a different size; it was preserved")
        if sha256_file(archive, name, progress) != source["sha256"].lower():
            raise FetchError("Existing ZIP SHA256 mismatch; it was preserved")
        emit("archive_verified", name, bytes=size, reused=True)
        return archive
    if partial.is_symlink():
        raise FetchError("Partial download must not be a symbolic link")
    last_reason = "download did not complete"
    for attempt in range(retries):
        start = partial.stat().st_size if partial.is_file() else 0
        if start > size:
            raise FetchError("Partial ZIP is larger than expected; it was preserved")
        if start == size:
            break
        progress.set(name, phase="downloading", downloaded_bytes=start, session_start_bytes=start, total_bytes=size)
        try:
            with requests.get(source["download_url"], headers={"Range": f"bytes={start}-{size-1}",
                              "Accept-Encoding": "identity"}, stream=True, timeout=(20, 90)) as response:
                status = response.status_code
                if status not in {200, 206}:
                    raise FetchError(f"HTTP status {status}; response body and URL suppressed")
                expected_range = f"bytes {start}-{size-1}/{size}"
                if status == 206 and response.headers.get("Content-Range") != expected_range:
                    raise FetchError("HTTP Content-Range differs from the requested source bytes")
                if status == 200 and start != 0:
                    raise FetchError("Server ignored Range for a resumed download; partial data was preserved")
                length = response.headers.get("Content-Length")
                if length is not None and (not length.isdigit() or int(length) != size - start):
                    raise FetchError("HTTP Content-Length differs from the remaining source bytes")
                received = start
                with partial.open("ab") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if not block:
                            continue
                        if received + len(block) > size:
                            raise FetchError("HTTP body exceeds expected source size")
                        handle.write(block)
                        received += len(block)
                        progress.set(name, downloaded_bytes=received)
                if received != size:
                    raise FetchError("HTTP body ended before the expected source size")
            break
        except requests.RequestException as error:
            # requests exceptions often contain a signed URL. Only their class
            # name is safe to expose in progress or error messages.
            last_reason = type(error).__name__
        except FetchError as error:
            last_reason = str(error)
        if attempt + 1 == retries:
            raise FetchError("Download retries exhausted: " + last_reason)
        emit("retry", name, attempt=attempt + 1, reason=last_reason)
        time.sleep(min(2 ** attempt, 15))
    if not partial.is_file() or partial.stat().st_size != size:
        raise FetchError("Final ZIP size differs from source metadata")
    if sha256_file(partial, name, progress) != source["sha256"].lower():
        raise FetchError("Downloaded ZIP SHA256 mismatch; partial preserved and extraction skipped")
    if archive.exists():
        raise FetchError("Another file appeared at the final ZIP path; partial was preserved")
    os.replace(partial, archive)
    emit("archive_verified", name, bytes=size, reused=False)
    return archive


def crc32_file(path):
    checksum = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xffffffff


def safe_extract(archive, source, data_root, progress):
    name = source["name"]
    destination = contained(data_root, relative_path(source["extract_root"], "extract_root"))
    destination.mkdir(parents=True, exist_ok=True)
    completed = reused = 0
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        # Validate the complete member namespace before creating any members.
        # The full archive SHA256 has already been verified by download().
        prepared, seen = [], set()
        for member in members:
            relative = relative_path(member.filename, "ZIP member")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise FetchError("ZIP contains a symbolic link")
            target = contained(destination, relative)
            key = target.as_posix()
            if key in seen and not member.is_dir():
                raise FetchError("ZIP contains a duplicate member target")
            seen.add(key)
            prepared.append((member, target))
        total = sum(not member.is_dir() for member in members)
        progress.set(name, phase="extracting", extracted_files=0, total_files=total)
        for member, target in prepared:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_symlink():
                raise FetchError("Existing extraction target is a symbolic link")
            if target.exists():
                if not target.is_file() or target.stat().st_size != member.file_size or crc32_file(target) != member.CRC:
                    raise FetchError("Existing extracted file differs from the verified ZIP; it was preserved")
                reused += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + f".fetching-{os.getpid()}-{threading.get_ident()}")
                try:
                    with package.open(member) as incoming, temporary.open("xb") as outgoing:
                        for block in iter(lambda: incoming.read(1024 * 1024), b""):
                            outgoing.write(block)
                    # Reading to EOF also performs ZIP member CRC validation.
                    if temporary.stat().st_size != member.file_size:
                        raise FetchError("Extracted file length mismatch")
                    if target.exists():
                        raise FetchError("Another file appeared at an extraction target")
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            completed += 1
            progress.set(name, extracted_files=completed)
    progress.set(name, phase="complete", extracted_files=completed)
    emit("source_complete", name, files=completed, reused_files=reused)
    return {"source": name, "files": completed, "reused_files": reused}


def process_source(source, cache_root, data_root, progress, retries):
    name = source["name"]
    emit("source_started", name, bytes=source["bytes"])
    archive = download(source, cache_root, progress, retries)
    return safe_extract(archive, source, data_root, progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], help="Repeat to fetch selected source names only")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.retries <= 20:
        parser.error("workers must be 1..8 and retries 1..20")
    try:
        sources = read_metadata(args.metadata, args.source)
        data_root, cache_root = args.data_root.resolve(), args.cache_root.resolve()
        data_root.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)
        progress, failures, completed = Progress(), [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_source, source, cache_root, data_root, progress, args.retries): source["name"]
                       for source in sources}
            pending, last_progress = set(futures), time.monotonic()
            last_bytes = {}
            while pending:
                done, pending = wait(pending, timeout=max(.1, 30 - (time.monotonic() - last_progress)), return_when=FIRST_COMPLETED)
                for future in done:
                    name = futures[future]
                    try:
                        completed.append(future.result())
                    except FetchError as error:
                        failures.append(name)
                        progress.set(name, phase="failed")
                        emit("source_failed", name, reason=str(error))
                    except BaseException as error:
                        failures.append(name)
                        progress.set(name, phase="failed")
                        # Never print exception repr or a traceback containing
                        # HTTP request arguments/signed URLs.
                        emit("source_failed", name, error_type=type(error).__name__)
                if time.monotonic() - last_progress >= 30:
                    interval = time.monotonic() - last_progress
                    for name, fields in progress.snapshot().items():
                        if fields.get("phase") not in {"complete", "failed"}:
                            current = fields.get("downloaded_bytes", 0)
                            if fields.get("phase") == "downloading":
                                baseline = last_bytes.get(name, fields.get("session_start_bytes", current))
                                fields["rate_mib_s"] = round(max(0, current - baseline) / interval / 1024**2, 2)
                            last_bytes[name] = current
                            emit("progress", name, **fields)
                    last_progress = time.monotonic()
        emit("finished", sources_complete=len(completed), failed_sources=failures)
        return 1 if failures else 0
    except (FetchError, OSError, ValueError, KeyError) as error:
        if isinstance(error, FetchError):
            emit("setup_failed", reason=str(error))
        else:
            emit("setup_failed", error_type=type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
