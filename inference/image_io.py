"""Bounded public-image download with DNS pinning and one RGB decode.

The original URL (which may contain a signed query) is neither returned nor
logged. Only HTTP(S) is supported, HTTPS is the default, no ambient proxies or
cookies are used, and every redirect is validated again. TLS connects to the
validated IP while checking the original hostname and using that hostname for
SNI. No global DNS, Pillow-limit, or TLS-verification settings are modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import ipaddress
import json
import math
import queue
import socket
import ssl
import threading
import time
from typing import Callable, TypeVar
from urllib.parse import quote, urlencode, urljoin, urlsplit
import warnings

import certifi
from PIL import Image, UnidentifiedImageError
import urllib3


_MESSAGES = {
    "invalid_url": "Image URL is invalid or contains unsupported components.",
    "insecure_url": "Image URL must use HTTPS unless HTTP was explicitly enabled.",
    "blocked_host": "Image host is not allowed by the server policy.",
    "blocked_address": "Image host resolves to a non-public network address.",
    "disallowed_port": "Image URL uses a port not allowed by the server policy.",
    "dns_error": "Image host could not be resolved.",
    "dns_timeout": "Image host resolution exceeded the request time limit.",
    "download_timeout": "Image download or decode exceeded the request time limit.",
    "download_error": "Image could not be downloaded.",
    "download_status": "Image server returned an unsuccessful HTTP status.",
    "tls_error": "Image server failed TLS certificate verification.",
    "too_many_redirects": "Image URL exceeded the allowed redirect count.",
    "redirect_invalid": "Image server returned an invalid redirect.",
    "redirect_downgrade": "Image redirect attempted to downgrade HTTPS to HTTP.",
    "content_encoding": "Compressed HTTP transfer encodings are not accepted for images.",
    "image_too_large": "Image download exceeds the configured byte limit.",
    "image_pixel_limit": "Image dimensions exceed the configured pixel limit.",
    "image_animated": "Animated or multi-frame images are not supported.",
    "unsupported_image_format": "Only single-frame JPEG, PNG, and WebP images are supported.",
    "invalid_image": "Downloaded content is not a valid supported image.",
    "image_uniform": "整张图解码后只有一种颜色，无法进行眼底筛查。请上传原始眼底照片。",
    "image_fully_transparent": "整张图完全透明，没有可见像素可供筛查。请上传可见的眼底照片。",
}


class InputImageError(ValueError):
    """Public error whose message never includes the input URL or raw exception."""

    def __init__(self, code: str):
        if code not in _MESSAGES:
            raise ValueError("Unknown input-image error code")
        self.code = code
        super().__init__(_MESSAGES[code])


_TRANSPARENT_MARKER = "_screening_fully_transparent"


def _fully_transparent(source: Image.Image) -> bool:
    if not source.has_transparency_data:
        return False
    if "A" in source.getbands():
        with source.getchannel("A") as alpha:
            return alpha.getbbox() is None
    # PNG can represent transparency through a palette or a color key instead
    # of an explicit alpha band. Pillow applies the file's transparency rules.
    with source.convert("RGBA") as rgba, rgba.getchannel("A") as alpha:
        return alpha.getbbox() is None


def validate_screening_pixels(image: Image.Image) -> None:
    """Reject exact blanks only, before models receive the decoded RGB image."""
    if image.mode != "RGB":
        raise ValueError("Screening pixel validation requires a decoded RGB image")
    if image.info.get(_TRANSPARENT_MARKER) is True:
        raise InputImageError("image_fully_transparent")
    if image.getcolors(maxcolors=1) is not None:
        raise InputImageError("image_uniform")


def _normalize_host(host: str) -> str:
    if not isinstance(host, str) or not host or any(c.isspace() for c in host):
        raise InputImageError("invalid_url")
    if any(c in host for c in ("%", "\\", "\0", "/", "@")):
        raise InputImageError("invalid_url")
    host = host.rstrip(".")
    if not host:
        raise InputImageError("invalid_url")
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        try:
            normalized = host.encode("idna").decode("ascii").lower()
        except (UnicodeError, ValueError):
            raise InputImageError("invalid_url") from None
        if len(normalized) > 253:
            raise InputImageError("invalid_url")
        labels = normalized.split(".")
        if any(not label or len(label) > 63 or label[0] == "-" or label[-1] == "-"
               or not all(c.isalnum() or c == "-" for c in label) for label in labels):
            raise InputImageError("invalid_url")
        return normalized


@dataclass(frozen=True)
class URLFetchPolicy:
    allow_http: bool = False
    allowed_hosts: tuple[str, ...] = ()
    public_dns_hosts: tuple[str, ...] = ()
    allowed_ports: tuple[int, ...] = (443,)
    max_download_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 60_000_000
    max_redirects: int = 3
    total_timeout_seconds: float = 15.0
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 5.0

    def __post_init__(self):
        if type(self.allow_http) is not bool:
            raise ValueError("allow_http must be boolean")
        for name in ("max_download_bytes", "max_pixels"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if type(self.max_redirects) is not int or not 0 <= self.max_redirects <= 10:
            raise ValueError("max_redirects must be an integer in [0,10]")
        for name in ("total_timeout_seconds", "connect_timeout_seconds", "read_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be positive and finite")
        if not isinstance(self.allowed_ports, (tuple, list)) or not self.allowed_ports or any(
                type(port) is not int or not 1 <= port <= 65535 for port in self.allowed_ports):
            raise ValueError("allowed_ports must be a nonempty sequence of TCP port numbers")
        for name in ("allowed_hosts", "public_dns_hosts"):
            if not isinstance(getattr(self, name), (tuple, list)):
                raise ValueError(name + " must be a sequence of exact hostnames")
            object.__setattr__(self, name, tuple(dict.fromkeys(_normalize_host(h) for h in getattr(self, name))))
        object.__setattr__(self, "allowed_ports", tuple(sorted(set(self.allowed_ports))))


@dataclass(frozen=True)
class DecodedImage:
    image: Image.Image
    sha256: str
    original_format: str
    width: int
    height: int
    bytes: int
    exif_orientation: int | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Target:
    scheme: str
    host: str
    port: int
    request_target: str
    host_header: str
    addresses: tuple[str, ...]


_FETCH_SLOTS = threading.BoundedSemaphore(4)
_DNS_SLOTS = threading.BoundedSemaphore(4)
# Python versions before context-aware warnings share the warning filter state.
# Serialize this small decode section so concurrent requests do not overwrite
# each other's DecompressionBombWarning handling.
_DECODE_LOCK = threading.Lock()
_T = TypeVar("_T")
_PUBLIC_DNS_HOST = "dns.alidns.com"
_MAX_DNS_RESPONSE_BYTES = 16 * 1024


def _remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise InputImageError("download_timeout")
    return value


def _bounded_call(function: Callable[[], _T], slots: threading.BoundedSemaphore,
                  deadline: float, timeout_code: str) -> _T:
    """Bound OS DNS and slow decoders without unbounded background threads.

    A timed-out OS resolver or decoder cannot be forcibly cancelled safely.
    Its daemon worker keeps a slot until it ends; therefore at most four DNS
    and four fetch/decode workers may remain active, and process shutdown is
    not held up by those workers. The caller always has a bounded wait.
    """
    budget = deadline - time.monotonic()
    if budget <= 0 or not slots.acquire(timeout=max(0.0, budget)):
        raise InputImageError(timeout_code)
    result: queue.Queue = queue.Queue(maxsize=1)

    def worker():
        try:
            result.put((True, function()))
        except Exception as error:
            result.put((False, error))
        finally:
            slots.release()

    try:
        threading.Thread(target=worker, name="fundus-image-io", daemon=True).start()
    except Exception:
        slots.release()
        raise InputImageError("download_error") from None
    try:
        good, value = result.get(timeout=max(0.0, deadline - time.monotonic()))
    except queue.Empty:
        raise InputImageError(timeout_code) from None
    if not good:
        raise value
    if time.monotonic() >= deadline:
        raise InputImageError(timeout_code)
    return value


def _resolve_addresses(host: str, port: int, deadline: float) -> tuple[str, ...]:
    try:
        return (str(ipaddress.ip_address(host)),)
    except ValueError:
        pass
    try:
        records = _bounded_call(
            lambda: socket.getaddrinfo(host, port, type=socket.SOCK_STREAM),
            _DNS_SLOTS, deadline, "dns_timeout")
    except InputImageError:
        raise
    except (OSError, UnicodeError):
        raise InputImageError("dns_error") from None
    addresses = tuple(dict.fromkeys(str(record[4][0]) for record in records))
    if not addresses:
        raise InputImageError("dns_error")
    return addresses


def _public_address(address: str) -> str:
    try:
        value = ipaddress.ip_address(address)
    except ValueError:
        raise InputImageError("dns_error") from None
    mapped = getattr(value, "ipv4_mapped", None)
    effective = mapped if mapped is not None else value
    # Transition mechanisms can encode an otherwise blocked IPv4 destination.
    # They are not needed by the public image hosts supported by this service.
    if isinstance(value, ipaddress.IPv6Address) and (
            value.sixtofour is not None or value.teredo is not None or
            value in ipaddress.ip_network("64:ff9b::/96") or
            value in ipaddress.ip_network("64:ff9b:1::/48")):
        raise InputImageError("blocked_address")
    if (not value.is_global or not effective.is_global or value.is_multicast or
            value.is_reserved or value.is_unspecified or value.is_loopback or
            value.is_link_local or effective.is_multicast or effective.is_reserved):
        raise InputImageError("blocked_address")
    return str(value)


def _dns_json_addresses(payload: bytes, host: str) -> tuple[str, ...]:
    """Parse one bounded A response, including the observed OBS CNAME chain."""
    try:
        document = json.loads(payload.decode("utf-8"))
        if (not isinstance(document, dict) or type(document.get("Status")) is not int or
                document["Status"] != 0 or document.get("TC", False) is not False):
            raise ValueError("DNS response status")
        question = document.get("Question")
        # AliDNS uses an object; the list form is also used by JSON DoH APIs.
        if isinstance(question, list) and len(question) == 1:
            question = question[0]
        if (not isinstance(question, dict) or type(question.get("type")) is not int or question["type"] != 1 or
                _normalize_host(question.get("name")) != host):
            raise ValueError("DNS response question")
        records = document.get("Answer")
        if not isinstance(records, list) or not 1 <= len(records) <= 64:
            raise ValueError("DNS response answer count")
        aliases, addresses = [], []
        for record in records:
            if not isinstance(record, dict) or type(record.get("type")) is not int:
                raise ValueError("DNS record")
            owner = _normalize_host(record.get("name"))
            if record["type"] == 5:
                aliases.append((owner, _normalize_host(record.get("data"))))
            elif record["type"] == 1:
                if not isinstance(record.get("data"), str):
                    raise ValueError("DNS A data")
                addresses.append((owner, str(ipaddress.IPv4Address(record.get("data")))))
        names = {host}
        while aliases:
            pending = []
            for owner, target in aliases:
                if owner in names:
                    if target in names:
                        raise ValueError("DNS CNAME loop")
                    names.add(target)
                else:
                    pending.append((owner, target))
            if len(pending) == len(aliases):
                raise ValueError("DNS unrelated CNAME")
            aliases = pending
        if not addresses or any(owner not in names for owner, _ in addresses):
            raise ValueError("DNS unrelated A answer")
        # The caller validates EVERY returned A address through _public_address.
        return tuple(dict.fromkeys(address for _, address in addresses))
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise InputImageError("dns_error") from None


def _query_public_dns(host: str, bootstrap: tuple[str, ...], policy: URLFetchPolicy,
                      deadline: float) -> tuple[str, ...]:
    # Only the normalized hostname is sent, never the path, signed query or user credentials.
    target = _Target("https", _PUBLIC_DNS_HOST, 443,
                     "/resolve?" + urlencode({"name": host, "type": "A"}), _PUBLIC_DNS_HOST, bootstrap)
    last_error = None
    for address in bootstrap:
        response = None
        try:
            response = _request_pinned(target, address, policy, deadline, accept="application/dns-json")
            if response.status != 200:
                raise InputImageError("dns_error")
            media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
            if media_type not in {"application/json", "application/dns-json"} or encoding not in {"", "identity"}:
                raise InputImageError("dns_error")
            length = response.headers.get("Content-Length")
            if length is not None and (not isinstance(length, str) or not length.isascii() or
                                       not length.isdecimal() or int(length) > _MAX_DNS_RESPONSE_BYTES):
                raise InputImageError("dns_error")
            payload = bytearray()
            while True:
                remaining = _remaining(deadline)
                amount = min(4096, _MAX_DNS_RESPONSE_BYTES - len(payload) + 1)
                block = response.read(amount, min(policy.read_timeout_seconds, remaining))
                _remaining(deadline)
                if not block:
                    break
                payload.extend(block)
                if len(payload) > _MAX_DNS_RESPONSE_BYTES:
                    raise InputImageError("dns_error")
            if length is not None and len(payload) != int(length):
                raise InputImageError("dns_error")
            return _dns_json_addresses(bytes(payload), host)
        except Exception as error:
            last_error = error
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
    if (time.monotonic() >= deadline or isinstance(last_error, (TimeoutError, socket.timeout)) or
            isinstance(last_error, InputImageError) and last_error.code in {"download_timeout", "dns_timeout"}):
        raise InputImageError("dns_timeout")
    raise InputImageError("dns_error") from None


def _resolve_public_dns_addresses(host: str, policy: URLFetchPolicy, deadline: float) -> tuple[str, ...]:
    # Bootstrap finishes and releases its DNS slot before a DoH worker acquires
    # one. Four concurrent overrides must not nest acquisitions of _DNS_SLOTS.
    bootstrap = tuple(dict.fromkeys(_public_address(address) for address in
                                   _resolve_addresses(_PUBLIC_DNS_HOST, 443, deadline)))
    return _bounded_call(lambda: _query_public_dns(host, bootstrap, policy, deadline),
                         _DNS_SLOTS, deadline, "dns_timeout")


def _validated_target(url: str, policy: URLFetchPolicy, deadline: float) -> _Target:
    if (not isinstance(url, str) or not url or len(url) > 8192 or
            any(c.isspace() or ord(c) < 32 or ord(c) == 127 or c == "\\" for c in url)):
        raise InputImageError("invalid_url")
    try:
        parts = urlsplit(url)
        if parts.scheme.lower() not in {"http", "https"} or not parts.netloc or parts.fragment:
            raise InputImageError("invalid_url")
        if parts.username is not None or parts.password is not None:
            raise InputImageError("invalid_url")
        host = _normalize_host(parts.hostname or "")
        scheme = parts.scheme.lower()
        explicit_port = parts.port
        port = explicit_port if explicit_port is not None else (443 if scheme == "https" else 80)
    except (ValueError, UnicodeError) as error:
        if isinstance(error, InputImageError):
            raise
        raise InputImageError("invalid_url") from None
    if scheme == "http" and not policy.allow_http:
        raise InputImageError("insecure_url")
    allowed_ports = set(policy.allowed_ports) | ({80} if policy.allow_http else set())
    if port not in allowed_ports:
        raise InputImageError("disallowed_port")
    if host == "localhost" or host.endswith(".localhost") or (policy.allowed_hosts and host not in policy.allowed_hosts):
        raise InputImageError("blocked_host")
    dns_deadline = min(deadline, time.monotonic() + policy.connect_timeout_seconds)
    resolved = (_resolve_public_dns_addresses(host, policy, dns_deadline) if host in policy.public_dns_hosts
                else _resolve_addresses(host, port, dns_deadline))
    addresses = tuple(dict.fromkeys(_public_address(address) for address in resolved))
    if not addresses:
        raise InputImageError("dns_error")
    path = quote(parts.path or "/", safe="/%:@!$&'()*+,;=-._~")
    query = quote(parts.query, safe="/%?:@!$&'()*+,;=-._~")
    request_target = path + ("?" + query if query else "")
    host_header = "[" + host + "]" if ":" in host else host
    if port != (443 if scheme == "https" else 80):
        host_header += ":" + str(port)
    return _Target(scheme, host, port, request_target, host_header, addresses)


class _Response:
    def __init__(self, pool, response):
        self.pool, self.response = pool, response
        self.status, self.headers = response.status, response.headers

    def read(self, amount: int, timeout: float) -> bytes:
        connection = getattr(self.response, "connection", None)
        sock = getattr(connection, "sock", None)
        if sock is not None:
            sock.settimeout(timeout)
        return self.response.read(amount, decode_content=False)

    def close(self):
        try:
            self.response.close()
        finally:
            self.pool.close()


def _network_error(error: Exception) -> InputImageError:
    if isinstance(error, (ssl.SSLError, urllib3.exceptions.SSLError)):
        return InputImageError("tls_error")
    if isinstance(error, (TimeoutError, socket.timeout, urllib3.exceptions.TimeoutError)):
        return InputImageError("download_timeout")
    return InputImageError("download_error")


def _request_pinned(target: _Target, address: str, policy: URLFetchPolicy, deadline: float,
                    *, accept: str = "image/jpeg,image/png,image/webp") -> _Response:
    remaining = _remaining(deadline)
    timeout = urllib3.Timeout(total=remaining, connect=min(policy.connect_timeout_seconds, remaining),
                              read=min(policy.read_timeout_seconds, remaining))
    if target.scheme == "https":
        pool = urllib3.HTTPSConnectionPool(
            host=address, port=target.port, timeout=timeout, retries=False,
            server_hostname=target.host, assert_hostname=target.host,
            cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())
    else:
        pool = urllib3.HTTPConnectionPool(host=address, port=target.port, timeout=timeout, retries=False)
    try:
        response = pool.urlopen(
            "GET", target.request_target,
            headers={"Host": target.host_header, "User-Agent": "FundusInference/1.0",
                     "Accept": accept, "Accept-Encoding": "identity"},
            redirect=False, retries=False, preload_content=False, decode_content=False,
            assert_same_host=False, timeout=timeout)
        return _Response(pool, response)
    except Exception as error:
        pool.close()
        raise _network_error(error) from None


def _download(url: str, policy: URLFetchPolicy, deadline: float) -> bytes:
    for hop in range(policy.max_redirects + 1):
        target = _validated_target(url, policy, deadline)
        response = None
        last_error = None
        # Only validated addresses are attempted; urllib3 sees a literal IP,
        # so a later DNS answer cannot redirect this connection to a private IP.
        for address in target.addresses[:4]:
            try:
                response = _request_pinned(target, address, policy, deadline)
                break
            except InputImageError as error:
                last_error = error
                if error.code != "download_error":
                    raise
        if response is None:
            raise last_error or InputImageError("download_error")
        try:
            _remaining(deadline)
            if response.status in {301, 302, 303, 307, 308}:
                if hop == policy.max_redirects:
                    raise InputImageError("too_many_redirects")
                location = response.headers.get("Location")
                if not isinstance(location, str) or not location or len(location) > 8192:
                    raise InputImageError("redirect_invalid")
                redirected = urljoin(url, location)
                if target.scheme == "https" and urlsplit(redirected).scheme.lower() == "http":
                    raise InputImageError("redirect_downgrade")
                url = redirected
                continue
            if response.status != 200:
                raise InputImageError("download_status")
            encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
            if encoding not in {"", "identity"}:
                raise InputImageError("content_encoding")
            length = response.headers.get("Content-Length")
            if length is not None:
                if not isinstance(length, str) or not length.isascii() or not length.isdecimal():
                    raise InputImageError("download_error")
                if int(length) > policy.max_download_bytes:
                    raise InputImageError("image_too_large")
            payload = bytearray()
            while True:
                remaining = _remaining(deadline)
                amount = min(64 * 1024, policy.max_download_bytes - len(payload) + 1)
                try:
                    block = response.read(amount, min(policy.read_timeout_seconds, remaining))
                except Exception as error:
                    raise _network_error(error) from None
                _remaining(deadline)
                if not block:
                    if length is not None and len(payload) != int(length):
                        raise InputImageError("download_error")
                    return bytes(payload)
                payload.extend(block)
                if len(payload) > policy.max_download_bytes:
                    raise InputImageError("image_too_large")
        finally:
            try:
                response.close()
            except Exception:
                pass
    raise InputImageError("too_many_redirects")


def _decode_locked(payload: bytes, policy: URLFetchPolicy, deadline: float) -> DecodedImage:
    _remaining(deadline)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as source:
                width, height = source.size
                if width < 1 or height < 1 or width * height > policy.max_pixels:
                    raise InputImageError("image_pixel_limit")
                if getattr(source, "is_animated", False) or getattr(source, "n_frames", 1) != 1:
                    raise InputImageError("image_animated")
                image_format = source.format
                if image_format not in {"JPEG", "PNG", "WEBP"}:
                    raise InputImageError("unsupported_image_format")
                orientation = source.getexif().get(274)
                image_warnings = []
                if orientation is not None and (type(orientation) is not int or orientation not in range(1, 9)):
                    orientation = None
                    image_warnings.append("invalid_exif_orientation_ignored")
                if orientation is not None and orientation != 1:
                    image_warnings.append("exif_orientation_not_applied_to_preserve_training_geometry")
                # This performs the pixel decode once, detaches the result from
                # its stream, and preserves the training pipeline's RGB behavior.
                # In particular: no EXIF transpose, extra crop, or resize here.
                invisible = _fully_transparent(source)
                rgb = source.convert("RGB")
                # File-supplied text metadata must never set an internal gate.
                rgb.info.pop(_TRANSPARENT_MARKER, None)
                if invisible:
                    rgb.info[_TRANSPARENT_MARKER] = True
        _remaining(deadline)
        return DecodedImage(rgb, hashlib.sha256(payload).hexdigest(), image_format,
                            width, height, len(payload), orientation, tuple(image_warnings))
    except InputImageError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise InputImageError("image_pixel_limit") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError):
        raise InputImageError("invalid_image") from None


def _decode(payload: bytes, policy: URLFetchPolicy, deadline: float) -> DecodedImage:
    if not _DECODE_LOCK.acquire(timeout=_remaining(deadline)):
        raise InputImageError("download_timeout")
    try:
        return _decode_locked(payload, policy, deadline)
    finally:
        _DECODE_LOCK.release()


def fetch_and_decode_image(image_url: str, *, policy: URLFetchPolicy | None = None) -> DecodedImage:
    """Download one public image and return its original bytes SHA and RGB pixels.

    ``allowed_hosts`` is an optional exact-host allowlist. Explicit HTTP enables
    port80; other nonstandard public ports must be named in ``allowed_ports``.
    Redirects never downgrade HTTPS. The bounded worker covers the full DNS,
    connection, body-read and decode operation under a single caller deadline.
    ``public_dns_hosts`` opts only exact configured hosts into a fresh AliDNS
    HTTPS A lookup, for cloud split-DNS domains. Its answers still must all be
    public, with the same pinned image connection and original-host TLS checks.
    There is no production switch for private or loopback destinations.
    """
    if policy is None:
        policy = URLFetchPolicy()
    if not isinstance(policy, URLFetchPolicy):
        raise TypeError("policy must be URLFetchPolicy")
    deadline = time.monotonic() + policy.total_timeout_seconds
    try:
        return _bounded_call(lambda: _decode(_download(image_url, policy, deadline), policy, deadline),
                             _FETCH_SLOTS, deadline, "download_timeout")
    except InputImageError:
        raise
    except Exception:
        raise InputImageError("download_error") from None


def decode_image_bytes(payload: bytes, *, policy: URLFetchPolicy | None = None) -> DecodedImage:
    """Decode bounded in-memory bytes for offline evaluation and local smoke checks.

    This accepts no path or URL and provides no exception to the public-address
    rules of ``fetch_and_decode_image``. It uses the same decode and worker limits.
    """
    if policy is None:
        policy = URLFetchPolicy()
    if not isinstance(policy, URLFetchPolicy):
        raise TypeError("policy must be URLFetchPolicy")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if len(payload) > policy.max_download_bytes:
        raise InputImageError("image_too_large")
    deadline = time.monotonic() + policy.total_timeout_seconds
    try:
        return _bounded_call(lambda: _decode(payload, policy, deadline),
                             _FETCH_SLOTS, deadline, "download_timeout")
    except InputImageError:
        raise
    except Exception:
        raise InputImageError("invalid_image") from None
