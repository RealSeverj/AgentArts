"""Offline input validation, pinned transport and bounded image decode tests."""
from __future__ import annotations

import hashlib
import io
import json
import socket
import ssl
import threading
import time
import unittest
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from inference import image_io


PUBLIC_IP = "93.184.216.34"
PUBLIC_URL = "https://images.example.org/fundus.png?token=private-test-token"


def image_bytes(mode="RGB", size=(9, 7), fmt="PNG", **save_options):
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format=fmt, **save_options)
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, payload=b"", *, status=200, headers=None):
        self.status = status
        self.headers = headers or {}
        self.stream = io.BytesIO(payload)
        self.closed = False
        self.read_sizes = []

    def read(self, amount, timeout):
        self.read_sizes.append(amount)
        return self.stream.read(amount)

    def close(self):
        self.closed = True


def dns_response(addresses=(PUBLIC_IP,), *, host="images.example.org", **fields):
    body = {"Status": 0, "TC": False, "Question": {"name": host + ".", "type": 1},
            "Answer": [{"name": host + ".", "type": 5, "TTL": 120, "data": "edge.example.net."},
                       *[{"name": "edge.example.net.", "type": 1, "TTL": 300, "data": value}
                         for value in addresses]]}
    body.update(fields)
    return FakeResponse(json.dumps(body).encode(), headers={"Content-Type": "application/json"})


class ImageInputTests(unittest.TestCase):
    def fetch(self, payload=None, *, response=None, url=PUBLIC_URL, policy=None):
        response = response or FakeResponse(image_bytes() if payload is None else payload)
        with patch.object(image_io, "_resolve_addresses", return_value=(PUBLIC_IP,)), \
                patch.object(image_io, "_request_pinned", return_value=response):
            result = image_io.fetch_and_decode_image(url, policy=policy)
        self.assertTrue(response.closed)
        return result

    def assert_code(self, code, function, *args, **kwargs):
        with self.assertRaises(image_io.InputImageError) as context:
            function(*args, **kwargs)
        self.assertEqual(context.exception.code, code)
        self.assertNotIn("private-test-token", str(context.exception))
        self.assertNotIn("images.example.org", str(context.exception))
        return context.exception

    def test_rgb_decoded_once_original_dimensions_and_original_bytes_hash(self):
        for mode in ("RGB", "L", "RGBA"):
            with self.subTest(mode=mode):
                payload = image_bytes(mode)
                result = self.fetch(payload)
                self.assertEqual(result.image.mode, "RGB")
                self.assertEqual(result.image.size, (9, 7))
                self.assertEqual((result.width, result.height), (9, 7))
                self.assertEqual(result.bytes, len(payload))
                self.assertEqual(result.sha256, hashlib.sha256(payload).hexdigest())
                self.assertEqual(result.original_format, "PNG")
                self.assertEqual(result.warnings, ())

    def test_actual_format_accepted_without_trusting_content_type(self):
        payload = image_bytes(fmt="JPEG")
        result = self.fetch(response=FakeResponse(payload, headers={"Content-Type": "text/plain"}))
        self.assertEqual(result.original_format, "JPEG")

    def test_screening_rejects_exact_rgb_blanks_but_preserves_low_contrast_detail(self):
        for color in ((0, 0, 0), (255, 255, 255), (175, 33, 12), (37, 37, 37)):
            with self.subTest(color=color), Image.new("RGB", (48, 48), color) as image:
                self.assert_code("image_uniform", image_io.validate_screening_pixels, image)
        for changed_pixel in ((1, 1, 1), (0, 0, 1), (255, 255, 255)):
            with self.subTest(pixel=changed_pixel), Image.new("RGB", (48, 48)) as image:
                image.putpixel((47, 47), changed_pixel)
                image_io.validate_screening_pixels(image)

    def test_fully_transparent_rgba_and_palette_images_preserve_their_visibility_gate(self):
        with Image.new("RGBA", (2, 1)) as rgba, Image.new("P", (2, 1)) as palette:
            rgba.putdata([(255, 0, 0, 0), (0, 255, 0, 0)])
            palette.putdata([0, 1])
            palette.putpalette([255, 0, 0, 0, 255, 0] + [0] * (768 - 6))
            for source, options in ((rgba, {}), (palette, {"transparency": bytes((0, 0))})):
                with self.subTest(mode=source.mode), io.BytesIO() as buffer:
                    source.save(buffer, format="PNG", **options)
                    decoded = image_io.decode_image_bytes(buffer.getvalue())
                    with decoded.image as rgb:
                        self.assertNotEqual(rgb.getpixel((0, 0)), rgb.getpixel((1, 0)))
                        self.assert_code("image_fully_transparent", image_io.validate_screening_pixels, rgb)

    def test_visible_rgb_pixels_not_blocked_by_file_metadata(self):
        meta = PngInfo()
        meta.add_text("_screening_fully_transparent", "true")
        with Image.new("RGBA", (2, 1)) as image, io.BytesIO() as buffer:
            image.putdata([(19, 19, 19, 255), (22, 22, 22, 254)])
            image.save(buffer, format="PNG", pnginfo=meta)
            decoded = image_io.decode_image_bytes(buffer.getvalue())
            with decoded.image as rgb:
                self.assertNotIn("_screening_fully_transparent", rgb.info)
                image_io.validate_screening_pixels(rgb)

    def test_exif_orientation_recorded_without_rotating_training_geometry(self):
        exif = Image.Exif()
        exif[274] = 6
        result = self.fetch(image_bytes(size=(11, 5), fmt="JPEG", exif=exif))
        self.assertEqual(result.image.size, (11, 5))
        self.assertEqual(result.exif_orientation, 6)
        self.assertIn("exif_orientation_not_applied_to_preserve_training_geometry", result.warnings)

    def test_invalid_url_components_rejected_before_network(self):
        cases = {
            "file:///etc/passwd": "invalid_url",
            "data:image/png;base64,abc": "invalid_url",
            "ftp://images.example.org/a.png": "invalid_url",
            "http://images.example.org/a.png": "insecure_url",
            "https://name:password@images.example.org/a": "invalid_url",
            "https://images.example.org/a#fragment": "invalid_url",
            "https://images.example.org/a\r\nHeader:x": "invalid_url",
            "https://images.example.org\\@127.0.0.1/a": "invalid_url",
            "https://localhost/a": "blocked_host",
            "https://nested.localhost/a": "blocked_host",
            "https://images.example.org:0/a": "disallowed_port",
            "https://images.example.org:8443/a": "disallowed_port",
            "https://images.example.org:invalid/a": "invalid_url",
            "https://[::1/a": "invalid_url",
        }
        with patch.object(image_io, "_request_pinned") as request:
            for url, code in cases.items():
                with self.subTest(url=url):
                    self.assert_code(code, image_io.fetch_and_decode_image, url)
            request.assert_not_called()

    def test_private_reserved_and_encoded_transition_addresses_rejected(self):
        addresses = (
            "127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.1.1",
            "169.254.169.254", "100.64.0.1", "0.0.0.0", "224.0.0.1",
            "240.0.0.1", "192.0.2.1", "::1", "::", "fc00::1", "fe80::1",
            "ff02::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1",
            "2002:7f00:1::", "64:ff9b::7f00:1", "2001:db8::1",
        )
        with patch.object(image_io, "_request_pinned") as request:
            for address in addresses:
                host = "[" + address + "]" if ":" in address else address
                with self.subTest(address=address):
                    self.assert_code("blocked_address", image_io.fetch_and_decode_image, "https://" + host + "/a")
            request.assert_not_called()

    def test_all_dns_results_must_be_public(self):
        with patch.object(image_io, "_resolve_addresses", return_value=(PUBLIC_IP, "10.10.0.1")), \
                patch.object(image_io, "_request_pinned") as request:
            self.assert_code("blocked_address", image_io.fetch_and_decode_image, PUBLIC_URL)
            request.assert_not_called()

    def test_policy_exact_allowlist_http_opt_in_and_custom_port(self):
        policy = image_io.URLFetchPolicy(allow_http=True, allowed_hosts=("IMAGES.EXAMPLE.ORG.",), allowed_ports=(443, 8443))
        for url in ("http://images.example.org/a", "https://images.example.org:8443/a"):
            self.fetch(url=url, policy=policy)
        self.assert_code("blocked_host", image_io.fetch_and_decode_image,
                         "https://images.example.org.attacker.org/a", policy=policy)
        self.assert_code("blocked_host", image_io.fetch_and_decode_image,
                         "https://different.example.org/a", policy=policy)

    def test_redirect_revalidates_private_destination_and_closes_response(self):
        response = FakeResponse(status=302, headers={"Location": "https://127.0.0.1/secret"})
        with patch.object(image_io, "_resolve_addresses", side_effect=[(PUBLIC_IP,), ("127.0.0.1",)]), \
                patch.object(image_io, "_request_pinned", return_value=response) as request:
            self.assert_code("blocked_address", image_io.fetch_and_decode_image, PUBLIC_URL)
            self.assertEqual(request.call_count, 1)
        self.assertTrue(response.closed)

    def test_redirect_revalidates_exact_host_allowlist(self):
        response = FakeResponse(status=302, headers={"Location": "https://other.example.org/a"})
        policy = image_io.URLFetchPolicy(allowed_hosts=("images.example.org",))
        with patch.object(image_io, "_resolve_addresses", return_value=(PUBLIC_IP,)), \
                patch.object(image_io, "_request_pinned", return_value=response) as request:
            self.assert_code("blocked_host", image_io.fetch_and_decode_image, PUBLIC_URL, policy=policy)
            self.assertEqual(request.call_count, 1)

    def test_relative_redirect_works_and_has_a_hard_limit(self):
        redirect = FakeResponse(status=302, headers={"Location": "/new.png?token=private-test-token"})
        final = FakeResponse(image_bytes())
        with patch.object(image_io, "_resolve_addresses", return_value=(PUBLIC_IP,)), \
                patch.object(image_io, "_request_pinned", side_effect=[redirect, final]) as request:
            result = image_io.fetch_and_decode_image(PUBLIC_URL)
            self.assertEqual(result.width, 9)
            self.assertEqual(request.call_args_list[1].args[0].request_target, "/new.png?token=private-test-token")
        self.assertTrue(redirect.closed and final.closed)
        loop = FakeResponse(status=302, headers={"Location": "/again"})
        with patch.object(image_io, "_resolve_addresses", return_value=(PUBLIC_IP,)), \
                patch.object(image_io, "_request_pinned", return_value=loop) as request:
            self.assert_code("too_many_redirects", image_io.fetch_and_decode_image, PUBLIC_URL,
                             policy=image_io.URLFetchPolicy(max_redirects=2))
            self.assertEqual(request.call_count, 3)

    def test_https_downgrade_rejected_even_if_http_opted_in(self):
        response = FakeResponse(status=302, headers={"Location": "http://images.example.org/a"})
        with patch.object(image_io, "_resolve_addresses", return_value=(PUBLIC_IP,)), \
                patch.object(image_io, "_request_pinned", return_value=response):
            self.assert_code("redirect_downgrade", image_io.fetch_and_decode_image, PUBLIC_URL,
                             policy=image_io.URLFetchPolicy(allow_http=True))

    def test_declared_and_streaming_byte_limits_and_length_mismatch(self):
        policy = image_io.URLFetchPolicy(max_download_bytes=100)
        declared = FakeResponse(b"unused", headers={"Content-Length": "101"})
        with patch.object(image_io, "_resolve_addresses", return_value=(PUBLIC_IP,)), \
                patch.object(image_io, "_request_pinned", return_value=declared):
            self.assert_code("image_too_large", image_io.fetch_and_decode_image, PUBLIC_URL, policy=policy)
        self.assertEqual(declared.read_sizes, [])
        for response, code in (
                (FakeResponse(b"a" * 101), "image_too_large"),
                (FakeResponse(b"a" * 10, headers={"Content-Length": "11"}), "download_error"),
                (FakeResponse(b"a", headers={"Content-Length": "not-a-size"}), "download_error"),
                (FakeResponse(b"a", headers={"Content-Encoding": "gzip"}), "content_encoding"),
                (FakeResponse(status=404), "download_status")):
            with self.subTest(code=code), patch.object(image_io, "_resolve_addresses", return_value=(PUBLIC_IP,)), \
                    patch.object(image_io, "_request_pinned", return_value=response):
                self.assert_code(code, image_io.fetch_and_decode_image, PUBLIC_URL, policy=policy)
                self.assertTrue(response.closed)
                self.assertTrue(all(amount <= 65536 for amount in response.read_sizes))

    def test_bad_image_unsupported_and_multiframe(self):
        for payload, code in ((b"not an image", "invalid_image"), (image_bytes(fmt="GIF"), "unsupported_image_format")):
            with self.subTest(code=code):
                self.assert_code(code, self.fetch, payload)
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), "red").save(buffer, "GIF", save_all=True,
                                              append_images=[Image.new("RGB", (4, 4), "blue")])
        self.assert_code("image_animated", self.fetch, buffer.getvalue())

    def test_pixel_limit_and_decompression_bomb_warning(self):
        self.assert_code("image_pixel_limit", self.fetch, image_bytes(size=(11, 10)),
                         policy=image_io.URLFetchPolicy(max_pixels=100))
        original_limit = Image.MAX_IMAGE_PIXELS
        with patch.object(image_io.Image, "open", side_effect=Image.DecompressionBombWarning("internal detail")):
            self.assert_code("image_pixel_limit", image_io._decode, b"bytes", image_io.URLFetchPolicy(), time.monotonic() + 1)
        self.assertEqual(Image.MAX_IMAGE_PIXELS, original_limit)

    def test_policy_rejects_invalid_limits(self):
        for kwargs in ({"max_pixels": 0}, {"max_download_bytes": True}, {"max_redirects": 11},
                       {"total_timeout_seconds": float("inf")}, {"allowed_ports": (0,)},
                       {"allowed_hosts": "images.example.org"}, {"allow_http": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                image_io.URLFetchPolicy(**kwargs)

    def test_offline_decode_uses_same_hash_geometry_and_byte_limits(self):
        payload = image_bytes()
        result = image_io.decode_image_bytes(payload)
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(result.image.size, (9, 7))
        self.assert_code("image_too_large", image_io.decode_image_bytes, payload,
                         policy=image_io.URLFetchPolicy(max_download_bytes=len(payload) - 1))
        self.assert_code("invalid_image", image_io.decode_image_bytes, b"not-an-image")
        with self.assertRaises(TypeError):
            image_io.decode_image_bytes("file:///tmp/image.png")

    def test_dns_error_never_exposes_original_exception(self):
        with patch.object(image_io.socket, "getaddrinfo", side_effect=socket.gaierror("private-test-token")):
            self.assert_code("dns_error", image_io.fetch_and_decode_image, PUBLIC_URL)

    def test_pinned_https_transport_keeps_hostname_for_host_sni_and_cert_validation(self):
        response = Mock(status=200, headers={})
        pool = Mock()
        pool.urlopen.return_value = response
        records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 443))]
        with patch.object(image_io.socket, "getaddrinfo", return_value=records) as resolve, \
                patch.object(image_io.urllib3, "HTTPSConnectionPool", return_value=pool) as construct:
            deadline = time.monotonic() + 2
            policy = image_io.URLFetchPolicy()
            target = image_io._validated_target(PUBLIC_URL, policy, deadline)
            wrapped = image_io._request_pinned(target, target.addresses[0], policy, deadline)
            wrapped.close()
        self.assertEqual(resolve.call_count, 1)
        self.assertEqual(resolve.call_args.args[:2], ("images.example.org", 443))
        options = construct.call_args.kwargs
        self.assertEqual(options["host"], PUBLIC_IP)
        self.assertEqual(options["server_hostname"], "images.example.org")
        self.assertEqual(options["assert_hostname"], "images.example.org")
        self.assertEqual(options["cert_reqs"], "CERT_REQUIRED")
        self.assertEqual(options["ca_certs"], image_io.certifi.where())
        request = pool.urlopen.call_args
        self.assertEqual(request.args, ("GET", "/fundus.png?token=private-test-token"))
        self.assertEqual(request.kwargs["headers"]["Host"], "images.example.org")
        for name in ("redirect", "retries", "preload_content", "decode_content"):
            self.assertFalse(request.kwargs[name])
        pool.close.assert_called_once()

    def test_ipv6_host_header_and_percent_encoded_query_are_preserved(self):
        url = "https://[2606:4700:4700::1111]:8443/a%20b.png?sig=a%2Fb%3D"
        policy = image_io.URLFetchPolicy(allowed_ports=(8443,))
        target = image_io._validated_target(url, policy, time.monotonic() + 1)
        self.assertEqual(target.host_header, "[2606:4700:4700::1111]:8443")
        self.assertEqual(target.request_target, "/a%20b.png?sig=a%2Fb%3D")

    def test_tls_and_socket_timeout_errors_are_safe(self):
        target = image_io._Target("https", "images.example.org", 443, "/?token=private-test-token",
                                 "images.example.org", (PUBLIC_IP,))
        for cause, code in ((ssl.SSLError("private-test-token"), "tls_error"),
                            (socket.timeout("private-test-token"), "download_timeout")):
            pool = Mock()
            pool.urlopen.side_effect = cause
            with self.subTest(code=code), patch.object(image_io.urllib3, "HTTPSConnectionPool", return_value=pool):
                self.assert_code(code, image_io._request_pinned, target, PUBLIC_IP,
                                 image_io.URLFetchPolicy(), time.monotonic() + 1)
                pool.close.assert_called_once()

    def test_total_deadline_bounds_a_slow_response_and_releases_worker(self):
        release = threading.Event()
        completed = threading.Event()

        class SlowResponse(FakeResponse):
            def read(self, amount, timeout):
                release.wait(1)
                return super().read(amount, timeout)

            def close(self):
                super().close()
                completed.set()

        response = SlowResponse(image_bytes())
        try:
            started = time.monotonic()
            with patch.object(image_io, "_resolve_addresses", return_value=(PUBLIC_IP,)), \
                    patch.object(image_io, "_request_pinned", return_value=response):
                self.assert_code("download_timeout", image_io.fetch_and_decode_image, PUBLIC_URL,
                                 policy=image_io.URLFetchPolicy(total_timeout_seconds=0.06))
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            release.set()
            self.assertTrue(completed.wait(1))

    def test_dns_has_its_own_bounded_wait_and_releases_worker(self):
        release = threading.Event()
        completed = threading.Event()

        def slow_resolve(*args, **kwargs):
            try:
                release.wait(1)
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 443))]
            finally:
                completed.set()

        try:
            with patch.object(image_io.socket, "getaddrinfo", side_effect=slow_resolve), \
                    patch.object(image_io, "_request_pinned") as request:
                self.assert_code("dns_timeout", image_io.fetch_and_decode_image, PUBLIC_URL,
                                 policy=image_io.URLFetchPolicy(connect_timeout_seconds=0.06, total_timeout_seconds=0.5))
                request.assert_not_called()
        finally:
            release.set()
            self.assertTrue(completed.wait(1))

    def test_exact_public_dns_override_preserves_signature_and_pins_both_connections(self):
        policy = image_io.URLFetchPolicy(public_dns_hosts=["IMAGES.EXAMPLE.ORG."])
        self.assertEqual(policy.public_dns_hosts, ("images.example.org",))
        doh, image = dns_response(), FakeResponse(image_bytes())
        def request(target, address, *args, **kwargs):
            if target.host == "dns.alidns.com":
                self.assertEqual(kwargs["accept"], "application/dns-json")
                self.assertEqual(target.request_target, "/resolve?name=images.example.org&type=A")
                self.assertNotIn("private-test-token", target.request_target)
                return doh
            self.assertEqual(target.host, "images.example.org")
            self.assertEqual(address, PUBLIC_IP)
            self.assertEqual(target.request_target, "/fundus.png?token=private-test-token")
            self.assertEqual(target.host_header, "images.example.org")
            return image
        with patch.object(image_io, "_resolve_addresses", return_value=("223.5.5.5",)) as resolve, \
                patch.object(image_io, "_request_pinned", side_effect=request):
            result = image_io.fetch_and_decode_image(PUBLIC_URL, policy=policy)
        self.assertEqual(result.image.size, (9, 7))
        self.assertEqual(resolve.call_count, 1)
        self.assertEqual(resolve.call_args.args[:2], ("dns.alidns.com", 443))
        self.assertTrue(doh.closed and image.closed)

    def test_public_dns_is_opt_in_and_exact_match_not_suffix(self):
        policy = image_io.URLFetchPolicy(public_dns_hosts=("images.example.org",))
        self.assertEqual(image_io.URLFetchPolicy().public_dns_hosts, ())
        for host in ("other.example.org", "images.example.org.evil.net", "child.images.example.org"):
            with self.subTest(host=host), patch.object(image_io, "_resolve_public_dns_addresses") as override, \
                    patch.object(image_io, "_resolve_addresses", return_value=("100.125.6.3",)) as system, \
                    patch.object(image_io, "_request_pinned") as request:
                self.assert_code("blocked_address", image_io.fetch_and_decode_image, "https://" + host + "/a", policy=policy)
                override.assert_not_called()
                request.assert_not_called()
                self.assertEqual(system.call_args.args[0], host)
        with self.assertRaises(ValueError):
            image_io.URLFetchPolicy(public_dns_hosts="images.example.org")

    def test_public_dns_mixed_private_answers_and_private_bootstrap_are_rejected(self):
        policy = image_io.URLFetchPolicy(public_dns_hosts=("images.example.org",))
        response = dns_response((PUBLIC_IP, "100.125.6.3"))
        with patch.object(image_io, "_resolve_addresses", return_value=("223.5.5.5",)), \
                patch.object(image_io, "_request_pinned", return_value=response) as request:
            self.assert_code("blocked_address", image_io.fetch_and_decode_image, PUBLIC_URL, policy=policy)
            self.assertEqual(request.call_count, 1)  # DNS only; no image connection.
            self.assertEqual(request.call_args.args[0].host, "dns.alidns.com")
        self.assertTrue(response.closed)
        with patch.object(image_io, "_resolve_addresses", return_value=("100.125.6.3",)), \
                patch.object(image_io, "_request_pinned") as request:
            self.assert_code("blocked_address", image_io.fetch_and_decode_image, PUBLIC_URL, policy=policy)
            request.assert_not_called()

    def test_public_dns_never_falls_back_to_system_private_host_on_failure(self):
        policy = image_io.URLFetchPolicy(public_dns_hosts=("images.example.org",))
        with patch.object(image_io, "_resolve_addresses", return_value=("223.5.5.5",)) as resolve, \
                patch.object(image_io, "_request_pinned", side_effect=image_io.InputImageError("tls_error")):
            self.assert_code("dns_error", image_io.fetch_and_decode_image, PUBLIC_URL, policy=policy)
        self.assertEqual(resolve.call_count, 1)
        self.assertEqual(resolve.call_args.args[0], "dns.alidns.com")

    def test_public_dns_rejects_bad_status_media_body_lengths_and_json(self):
        responses = [FakeResponse(status=302), FakeResponse(b"{}", headers={"Content-Type": "text/html"}),
                     FakeResponse(b"bad json", headers={"Content-Type": "application/json"}),
                     FakeResponse(b"x" * (16 * 1024 + 1), headers={"Content-Type": "application/json"}),
                     FakeResponse(b"{}", headers={"Content-Type": "application/json", "Content-Length": "20000"}),
                     FakeResponse(b"{}", headers={"Content-Type": "application/json", "Content-Length": "3"}),
                     dns_response(Status=3), dns_response(TC=True), dns_response(Question={"name": "other.example.org", "type": 1}),
                     dns_response(Question={"name": "images.example.org", "type": True}),
                     dns_response(Answer=[]), dns_response(Answer=[{"name": "unrelated.example.org", "type": 1, "data": PUBLIC_IP}])]
        policy = image_io.URLFetchPolicy(public_dns_hosts=("images.example.org",))
        for response in responses:
            with self.subTest(response=response), patch.object(image_io, "_resolve_addresses", return_value=("223.5.5.5",)), \
                    patch.object(image_io, "_request_pinned", return_value=response):
                self.assert_code("dns_error", image_io.fetch_and_decode_image, PUBLIC_URL, policy=policy)
                self.assertTrue(response.closed)

    def test_public_dns_transport_uses_provider_tls_identity_and_dns_accept_header(self):
        pool = Mock()
        with patch.object(image_io.urllib3, "HTTPSConnectionPool", return_value=pool) as factory:
            response = image_io._request_pinned(image_io._Target("https", "dns.alidns.com", 443,
                "/resolve?name=images.example.org&type=A", "dns.alidns.com", ("223.5.5.5",)),
                "223.5.5.5", image_io.URLFetchPolicy(), time.monotonic() + 1, accept="application/dns-json")
            self.assertEqual(factory.call_args.kwargs["host"], "223.5.5.5")
            self.assertEqual(factory.call_args.kwargs["server_hostname"], "dns.alidns.com")
            self.assertEqual(factory.call_args.kwargs["assert_hostname"], "dns.alidns.com")
            self.assertEqual(factory.call_args.kwargs["cert_reqs"], "CERT_REQUIRED")
            self.assertEqual(pool.urlopen.call_args.kwargs["headers"]["Accept"], "application/dns-json")
            self.assertFalse(pool.urlopen.call_args.kwargs["redirect"])
            response.close()
        pool.close.assert_called_once()

    def test_public_dns_network_wait_is_bounded_and_response_is_closed(self):
        release, completed = threading.Event(), threading.Event()
        original = dns_response()
        class SlowDNSResponse(FakeResponse):
            def read(self, amount, timeout):
                release.wait(1)
                return super().read(amount, timeout)
            def close(self):
                super().close()
                completed.set()
        response = SlowDNSResponse(original.stream.getvalue(), headers=original.headers)
        policy = image_io.URLFetchPolicy(public_dns_hosts=("images.example.org",), connect_timeout_seconds=0.05)
        try:
            started = time.monotonic()
            with patch.object(image_io, "_resolve_addresses", return_value=("223.5.5.5",)), \
                    patch.object(image_io, "_request_pinned", return_value=response):
                self.assert_code("dns_timeout", image_io.fetch_and_decode_image, PUBLIC_URL, policy=policy)
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            release.set()
            self.assertTrue(completed.wait(1))

    def test_four_public_dns_queries_do_not_deadlock_nested_dns_semaphores(self):
        barrier = threading.Barrier(4)
        policy = image_io.URLFetchPolicy(public_dns_hosts=("images.example.org",), connect_timeout_seconds=2)
        def records(host, port, **kwargs):
            self.assertEqual(host, "dns.alidns.com")
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("223.5.5.5", 443))]
        def request(*args, **kwargs):
            barrier.wait(1.5)
            return dns_response()
        with patch.object(image_io.socket, "getaddrinfo", side_effect=records), \
                patch.object(image_io, "_request_pinned", side_effect=request), ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(image_io._validated_target, PUBLIC_URL, policy, time.monotonic() + 3) for _ in range(4)]
            self.assertTrue(all(future.result(timeout=4).addresses == (PUBLIC_IP,) for future in futures))


if __name__ == "__main__":
    unittest.main()
