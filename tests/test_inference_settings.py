import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference.settings import Settings


class SettingsTests(unittest.TestCase):
    def test_relative_paths_are_relative_to_config(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "service.json"
            config.write_text(json.dumps({"registry": "models.json", "weights_root": "weights"}), encoding="utf-8")
            with patch.dict(os.environ, {"RETINA_SERVICE_CONFIG": str(config), "RETINA_API_KEY": "x" * 40}, clear=True):
                settings = Settings.from_env()
            self.assertEqual(settings.registry, Path(folder) / "models.json")
            self.assertEqual(settings.weights_root, Path(folder) / "weights")
            self.assertNotIn("x" * 40, repr(settings))

    def test_reject_insecure_secret_or_unbounded_capacity(self):
        for params in ({"api_key": "short"}, {"max_concurrent_requests": 1000}, {"max_body_bytes": 10**9},
                       {"cpu_threads": 0}, {"device": "bogus"}):
            with self.assertRaises(ValueError):
                Settings(**params)


if __name__ == "__main__":
    unittest.main()
