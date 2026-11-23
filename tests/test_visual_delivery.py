import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from inference.visual_delivery import TTL_SECONDS, VisualDelivery


class VisualDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.now = 100
        self.delivery = VisualDelivery("https://reports.example.org", clock=lambda: self.now, max_bytes=10)

    @staticmethod
    def render(image, screening, sha, root, request_id, *, url_prefix):
        folder = root / request_id
        folder.mkdir()
        (folder / "original.jpg").write_bytes(image)
        (folder / "report.html").write_bytes(b"ok")
        return {"report_url": url_prefix + "/report.html", "original_url": url_prefix + "/original.jpg"}

    def create(self, data):
        with patch("inference.visual_report.write_visual_report", side_effect=self.render):
            result = self.delivery.create(data, {}, "0" * 64, "1" * 32)
        return urlsplit(result["report_url"]).path.split("/")[-2]

    def test_tokens_are_scoped_to_one_report_and_expire(self):
        first = self.create(b"ab")
        second = self.create(b"cd")
        self.assertNotEqual(first, second)
        self.assertEqual(self.delivery.read(first, "original.jpg"), b"ab")
        self.assertEqual(self.delivery.read(second, "original.jpg"), b"cd")
        self.assertIsNone(self.delivery.read(first, "../original.jpg"))
        self.assertIsNone(self.delivery.read(first, "overlay.jpg"))
        self.now += TTL_SECONDS
        self.assertIsNone(self.delivery.read(first, "original.jpg"))
        self.assertEqual(self.delivery.total_bytes, 0)

    def test_budget_evicts_oldest_view(self):
        first = self.create(b"first")
        second = self.create(b"next")
        self.assertIsNone(self.delivery.read(first, "report.html"))
        self.assertEqual(self.delivery.read(second, "original.jpg"), b"next")
