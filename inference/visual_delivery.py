"""Serve short-lived, image-specific viewing links without retaining uploads on disk."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import secrets
from tempfile import TemporaryDirectory
import threading
from time import monotonic
from urllib.parse import urlsplit


MEDIA = {"original.jpg": "image/jpeg", "overlay.jpg": "image/jpeg", "report.html": "text/html"}
TTL_SECONDS = 30 * 60


class VisualDelivery:
    def __init__(self, origin, *, clock=monotonic, max_bytes=96 * 1024**2):
        origin = origin.rstrip("/")
        url = urlsplit(origin)
        if (url.scheme != "https" or not url.hostname or url.username is not None or
                url.password is not None or url.path or url.query or url.fragment):
            raise ValueError("The viewing origin must be a configured HTTPS origin")
        self.origin = origin
        self.clock = clock
        self.max_bytes = max_bytes
        self.total_bytes = 0
        self.entries = OrderedDict()
        self.lock = threading.Lock()

    def _remove_oldest(self):
        _, (_, _, size, _) = self.entries.popitem(last=False)
        self.total_bytes -= size

    def _expire(self):
        now = self.clock()
        while self.entries and next(iter(self.entries.values()))[0] <= now:
            self._remove_oldest()

    def expire(self):
        with self.lock:
            self._expire()

    def create(self, image, screening, image_sha256, request_id):
        from inference.visual_report import write_visual_report

        token = secrets.token_urlsafe(32)
        prefix = f"{self.origin}/v1/view/{token}"
        with TemporaryDirectory(prefix="retina-view-") as work:
            root = Path(work)
            result = write_visual_report(image, screening, image_sha256, root, request_id, url_prefix=prefix)
            folder = root / request_id
            files = {name: (folder / name).read_bytes() for name in MEDIA if (folder / name).is_file()}
        if not files:
            return result
        size = sum(map(len, files.values()))
        if size > self.max_bytes:
            raise ValueError("Rendered report exceeds the delivery budget")
        meta = {
            "request_id": request_id,
            "info": deepcopy(result),
            "screening": deepcopy(screening),
            "has_overlay": "overlay.jpg" in files,
        }
        with self.lock:
            self._expire()
            while self.entries and (self.total_bytes + size > self.max_bytes or len(self.entries) >= 48):
                self._remove_oldest()
            self.entries[token] = (self.clock() + TTL_SECONDS, files, size, meta)
            self.total_bytes += size
        result["expires_in_seconds"] = TTL_SECONDS
        return result

    def get_request_id(self, token):
        if len(token) != 43:
            return None
        with self.lock:
            self._expire()
            item = self.entries.get(token)
            return item[3]["request_id"] if item else None

    def read(self, token, name, *, assessment_record=None):
        if name not in MEDIA or len(token) != 43:
            return None
        with self.lock:
            self._expire()
            item = self.entries.get(token)
            if not item:
                return None
            files = item[1]
            meta = item[3]
        if name == "report.html" and assessment_record is not None:
            from inference.visual_report import _report_html

            return _report_html(
                meta["info"], meta["screening"], meta["has_overlay"], assessment_record=assessment_record
            ).encode("utf-8")
        return files.get(name)
