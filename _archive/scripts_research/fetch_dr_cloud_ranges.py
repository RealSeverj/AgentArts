"""Concurrent resumable HTTP ranges for exact DR source ZIPs.

Signed URLs are read from metadata, never printed. Completed ranges are
journaled only after each entire chunk has been written. Whole-file SHA256 is
verified before publishing the ZIP or invoking the original safe extractor.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time

import requests

try:
    from scripts import fetch_dr_cloud_sources as legacy
except ImportError:
    import fetch_dr_cloud_sources as legacy

FetchError = legacy.FetchError
LOCAL = threading.local()


@contextmanager
def exclusive_source(path):
    """OS locks release automatically after a crash, allowing later resume."""
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise FetchError("Another range downloader holds this source lock") from None
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_journal(path, identity, completed):
    temporary = path.with_name(path.name + ".tmp")
    payload = {**identity, "completed_chunks": sorted(completed)}
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def mark_sparse(handle):
    # Linux truncate creates a sparse file. NTFS needs FSCTL_SET_SPARSE first;
    # filesystems without sparse support still work as ordinary allocated files.
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        import msvcrt
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        control = kernel.DeviceIoControl
        control.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                            wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
        control.restype = wintypes.BOOL
        returned = wintypes.DWORD()
        control(wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno())), 0x900C4,
                None, 0, None, 0, ctypes.byref(returned), None)


class TransferState(legacy.Progress):
    def add_received(self, name, count):
        with self.lock:
            self.values.setdefault(name, {})["received_bytes"] = self.values.get(name, {}).get("received_bytes", 0) + count

    def inflight(self, name, chunk, count):
        with self.lock:
            chunks = self.values.setdefault(name, {}).setdefault("_inflight", {})
            if count is None:
                chunks.pop(chunk, None)
            else:
                chunks[chunk] = count

    def snapshot(self):
        with self.lock:
            result = {}
            for name, value in self.values.items():
                result[name] = {k: v for k, v in value.items() if k != "_inflight"}
                result[name]["in_flight_bytes"] = sum(value.get("_inflight", {}).values())
            return result


def chunk_bounds(index, chunk_bytes, total):
    start = index * chunk_bytes
    return start, min(total, start + chunk_bytes) - 1


def fetch_chunk(source, index, chunk_bytes, partial, state, retries, stopped):
    name, total = source["name"], source["bytes"]
    start, end = chunk_bounds(index, chunk_bytes, total)
    length = end - start + 1
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    reason = "incomplete range"
    for attempt in range(retries):
        if stopped.is_set():
            raise FetchError("Source canceled after another chunk failed")
        count = 0
        state.inflight(name, index, 0)
        try:
            with LOCAL.session.get(source["download_url"], headers={"Range": f"bytes={start}-{end}",
                                   "Accept-Encoding": "identity"}, stream=True, timeout=(20, 90)) as response:
                if response.status_code != 206:
                    raise FetchError(f"Range HTTP status {response.status_code}")
                if response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                    raise FetchError("Range Content-Range mismatch")
                header_length = response.headers.get("Content-Length", "")
                if not header_length.isdigit() or int(header_length) != length:
                    raise FetchError("Range Content-Length mismatch")
                # Each worker owns its file object and its non-overlapping byte
                # range. Retrying a failed chunk rewrites that entire range.
                with partial.open("r+b", buffering=0) as output:
                    output.seek(start)
                    for block in response.iter_content(chunk_size=256 * 1024):
                        if stopped.is_set():
                            raise FetchError("Source canceled after another chunk failed")
                        if not block:
                            continue
                        if count + len(block) > length:
                            raise FetchError("Range body exceeds requested length")
                        view = memoryview(block)
                        while view:
                            written = output.write(view)
                            if not written:
                                raise FetchError("Range file write made no progress")
                            view = view[written:]
                        count += len(block)
                        state.add_received(name, len(block))
                        state.inflight(name, index, count)
                    if count != length:
                        raise FetchError("Range body ended before requested length")
                    output.flush()
                    os.fsync(output.fileno())
            state.inflight(name, index, None)
            return index
        except requests.RequestException as error:
            reason = type(error).__name__
        except FetchError as error:
            reason = str(error)
        finally:
            state.inflight(name, index, None)
        if attempt + 1 < retries:
            # No successful-download throttling. Interruptible retry backoff.
            stopped.wait(min(2 ** attempt, 15))
    raise FetchError("Chunk retries exhausted: " + reason)


def process_source(source, cache_root, data_root, chunk_bytes, connections, retries, state, events):
    name, total = source["name"], source["bytes"]
    archive = legacy.contained(cache_root, legacy.relative_path(source["archive_name"], "archive_name"))
    partial = archive.with_name(archive.name + ".ranges.partial")
    journal = archive.with_name(archive.name + ".ranges.json")
    lock = archive.with_name(archive.name + ".ranges.lock")
    if partial.is_symlink() or journal.is_symlink() or lock.is_symlink():
        raise FetchError("Range state paths must not be symbolic links")
    with exclusive_source(lock):
        if archive.is_file():
            if archive.stat().st_size != total or legacy.sha256_file(archive, name, state) != source["sha256"].lower():
                raise FetchError("Existing final ZIP differs; it was preserved")
            events.put(("archive_verified", name, {"bytes": total, "reused": True}))
            return legacy.safe_extract(archive, source, data_root, state)
        identity = {"format_version": 1, "source": name, "bytes": total,
                    "sha256": source["sha256"].lower(), "chunk_bytes": chunk_bytes}
        count = (total + chunk_bytes - 1) // chunk_bytes
        if partial.exists() or journal.exists():
            if not partial.is_file() or not journal.is_file() or partial.stat().st_size != total:
                raise FetchError("Incomplete or mismatched range state; existing files preserved")
            saved = json.loads(journal.read_text(encoding="utf-8"))
            if any(saved.get(k) != v for k, v in identity.items()):
                raise FetchError("Range journal source, size, checksum or chunk size differs")
            values = saved.get("completed_chunks", [])
            if (not isinstance(values, list) or any(type(i) is not int or not 0 <= i < count for i in values)
                    or len(set(values)) != len(values)):
                raise FetchError("Range journal completed chunk indexes are invalid")
            completed = set(values)
        else:
            with partial.open("xb") as output:
                mark_sparse(output)
                output.truncate(total)
                output.flush()
                os.fsync(output.fileno())
            completed = set()
            write_journal(journal, identity, completed)
        complete_bytes = sum(chunk_bounds(i, chunk_bytes, total)[1] - i * chunk_bytes + 1 for i in completed)
        state.set(name, phase="downloading_ranges", completed_bytes=complete_bytes, total_bytes=total,
                  chunks_done=len(completed), chunks_total=count, received_bytes=0)
        events.put(("source_started", name, {"bytes": total, "connections": connections,
                                            "resumed_chunks": len(completed), "chunk_bytes": chunk_bytes}))
        stopped = threading.Event()
        with ThreadPoolExecutor(max_workers=connections) as pool:
            futures = {pool.submit(fetch_chunk, source, i, chunk_bytes, partial, state, retries, stopped)
                       for i in range(count) if i not in completed}
            try:
                while futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = future.result()
                        completed.add(index)
                        start, end = chunk_bounds(index, chunk_bytes, total)
                        complete_bytes += end - start + 1
                        # Only this coordinator publishes a completed chunk.
                        write_journal(journal, identity, completed)
                        state.set(name, completed_bytes=complete_bytes, chunks_done=len(completed))
            except BaseException:
                stopped.set()
                for future in futures:
                    future.cancel()
                raise
        if len(completed) != count:
            raise FetchError("Not all chunks are complete; extraction skipped")
        if legacy.sha256_file(partial, name, state) != source["sha256"].lower():
            raise FetchError("Whole ZIP SHA256 mismatch; range state preserved and extraction skipped")
        if archive.exists():
            raise FetchError("Final ZIP appeared while downloading; range state preserved")
        os.replace(partial, archive)
        # Keep the small journal as completion evidence; it contains no URLs.
        events.put(("archive_verified", name, {"bytes": total, "reused": False}))
        return legacy.safe_extract(archive, source, data_root, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--workers", type=int, default=2, help="Concurrent sources")
    parser.add_argument("--connections", type=int, default=128, help="Concurrent range requests per source")
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 5 or not 1 <= args.connections <= 256 or not 1 <= args.chunk_mib <= 64 or not 1 <= args.retries <= 20:
        parser.error("workers=1..5, connections=1..256, chunk-mib=1..64, retries=1..20")
    try:
        sources = legacy.read_metadata(args.metadata, args.source)
        data_root, cache_root = args.data_root.resolve(), args.cache_root.resolve()
        data_root.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)
        state, events = TransferState(), queue.Queue()
        # The legacy safe extractor emits only safe source-level events. Route
        # them through the main thread rather than printing from worker threads.
        legacy.emit = lambda event, source=None, **fields: events.put((event, source, fields))
        failures, complete = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_source, source, cache_root, data_root, args.chunk_mib * 1024**2,
                                   args.connections, args.retries, state, events): source["name"] for source in sources}
            pending, last_tick, last_bytes = set(futures), time.monotonic(), {}
            while pending:
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                while not events.empty():
                    event, name, fields = events.get_nowait()
                    emit(event, name, **fields)
                for future in done:
                    name = futures[future]
                    try:
                        complete.append(future.result())
                    except BaseException as error:
                        failures.append(name)
                        state.set(name, phase="failed")
                        emit("source_failed", name, error_type=type(error).__name__,
                             reason=str(error) if isinstance(error, FetchError) else "details suppressed")
                now = time.monotonic()
                if now - last_tick >= 30:
                    interval = now - last_tick
                    for name, values in state.snapshot().items():
                        if values.get("phase") not in {"complete", "failed"}:
                            received = values.get("received_bytes", 0)
                            values["rate_mib_s"] = round(max(0, received - last_bytes.get(name, 0)) / interval / 1024**2, 2)
                            emit("progress", name, **values)
                            last_bytes[name] = received
                    last_tick = now
        emit("finished", sources_complete=len(complete), failed_sources=failures)
        return 1 if failures else 0
    except BaseException as error:
        emit("setup_failed", error_type=type(error).__name__,
             reason=str(error) if isinstance(error, FetchError) else "details suppressed")
        return 1


def emit(event, source=None, **fields):
    value = {"event": event}
    if source is not None:
        value["source"] = source
    value.update(fields)
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
