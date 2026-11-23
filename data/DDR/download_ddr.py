"""Download the original DDR release from the links on the RIADD website.

Files are downloaded in independently verified HTTP ranges. Completion state is
saved after every range so an interrupted run can safely resume.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from datetime import datetime, timezone

import requests

ROOT = Path(__file__).resolve().parent
ARCHIVES = ROOT / "chunks"
CHUNK = 32 * 1024 * 1024
WORKERS = 3
OFFICIAL = "https://github.com/nkicsl/DDR-dataset"
FILES = json.loads((ROOT / "source_files.json").read_text(encoding="utf-8"))
LOCAL = threading.local()


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_range(spec: dict, start: int, end: int) -> tuple[str, int]:
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    expected_range = f"bytes {start}-{end}/{spec['bytes']}"
    url = ("https://drive.usercontent.google.com/download"
           f"?id={spec['id']}&export=download&confirm=t&ddr_range={start}-{end}")
    for attempt in range(6):
        try:
            with LOCAL.session.get(url, headers={"Range": f"bytes={start}-{end}", "Cache-Control": "no-cache"},
                                   stream=True, timeout=(25, 90)) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError(f"Unexpected range response: {response.status_code}, {response.headers.get('Content-Range')}")
                received = 0
                with (ARCHIVES / (spec["name"] + ".part")).open("r+b") as output:
                    output.seek(start)
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if not block:
                            continue
                        received += len(block)
                        if received > end - start + 1:
                            raise RuntimeError("Response exceeds requested range")
                        output.write(block)
                    output.flush()
                if received != end - start + 1:
                    raise RuntimeError(f"Incomplete range: {received} bytes")
            return spec["name"], start // CHUNK
        except Exception as error:
            print(json.dumps({"event": "retry", "file": spec["name"], "start": start,
                              "attempt": attempt + 1, "error": str(error)[:180]}), flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError("Unreachable")


def main() -> None:
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    states = {}
    jobs = []
    manifest_path = ROOT / "download_manifest.json"
    manifest = {"dataset": "OIA-DDR original 10-part ZIP", "official_page": OFFICIAL,
                "started_utc": datetime.now(timezone.utc).isoformat(), "chunk_bytes": CHUNK, "files": FILES}
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        for spec in FILES:
            for old in prior.get("files", []):
                if spec["id"] == old.get("id") and old.get("sha256"):
                    spec["sha256"] = old["sha256"]
    for spec in FILES:
        spec["source_url"] = f"https://drive.google.com/file/d/{spec['id']}/view"
        final = ARCHIVES / spec["name"]
        if final.exists():
            if final.stat().st_size != spec["bytes"]:
                raise RuntimeError(f"Existing file has unexpected size: {final.name}")
            print(json.dumps({"event": "already_downloaded", "file": spec["name"]}), flush=True)
            continue
        part = ARCHIVES / (spec["name"] + ".part")
        state_path = ARCHIVES / (spec["name"] + ".ranges.json")
        if part.exists() or state_path.exists():
            if not (part.exists() and state_path.exists()):
                raise RuntimeError(f"Incomplete resume metadata for {spec['name']}")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (state["id"], state["bytes"], state["chunk_bytes"]) != (spec["id"], spec["bytes"], CHUNK):
                raise RuntimeError("Resume metadata does not match official file")
            if part.stat().st_size != spec["bytes"]:
                raise RuntimeError("Partial file size differs from saved metadata")
        else:
            state = {"id": spec["id"], "bytes": spec["bytes"], "chunk_bytes": CHUNK, "completed": []}
            with part.open("xb") as handle:
                handle.truncate(spec["bytes"])
            save_json(state_path, state)
        states[spec["name"]] = state
        completed = set(state["completed"])
        for start in range(0, spec["bytes"], CHUNK):
            if start // CHUNK not in completed:
                jobs.append((spec, start, min(start + CHUNK, spec["bytes"]) - 1))
    save_json(manifest_path, manifest)
    total = sum(spec["bytes"] for spec in FILES)
    initial = sum(spec["bytes"] if (ARCHIVES / spec["name"]).exists() else
                  sum(min(CHUNK, spec["bytes"] - i * CHUNK) for i in states[spec["name"]]["completed"])
                  for spec in FILES)
    received = initial
    started = time.monotonic()
    last_report = 0.0
    # Interleave splits so all three make progress even on a partial run.
    jobs.sort(key=lambda job: (job[1], job[0]["split"]))
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = {pool.submit(download_range, *job): job for job in jobs}
        while pending:
            done, _ = cf.wait(pending, timeout=10, return_when=cf.FIRST_COMPLETED)
            for future in done:
                spec, start, end = pending.pop(future)
                name, index = future.result()
                states[name]["completed"].append(index)
                states[name]["completed"].sort()
                save_json(ARCHIVES / (name + ".ranges.json"), states[name])
                received += end - start + 1
            now = time.monotonic()
            if now - last_report >= 15 or not pending:
                progress = {"event": "progress", "completed_bytes": received, "total_bytes": total,
                            "percent": round(received / total * 100, 2), "pending_ranges": len(pending),
                            "average_MiB_per_second": round((received - initial) / max(now - started, 0.001) / 1024**2, 2)}
                save_json(ROOT / "download_progress.json", progress)
                print(json.dumps(progress), flush=True)
                last_report = now
    for spec in FILES:
        final = ARCHIVES / spec["name"]
        if not final.exists():
            os.replace(ARCHIVES / (spec["name"] + ".part"), final)
        spec["sha256"] = sha256(final)
        print(json.dumps({"event": "downloaded", "file": spec["name"], "bytes": spec["bytes"], "sha256": spec["sha256"]}), flush=True)
        save_json(manifest_path, manifest)
    manifest["download_finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(manifest_path, manifest)
    print(json.dumps({"event": "all_downloads_complete", "total_bytes": total}), flush=True)


if __name__ == "__main__":
    main()
