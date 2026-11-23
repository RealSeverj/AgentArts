"""Pure file fixtures: image bytes are deliberately not valid encoded images."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.verify_isolated_source_images import (PROJECT, _hash_file, main,
                                                    verify_source_images)


class SourceImageVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="source_image_verify_")
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.leaves = self.data / "prepared/source_isolated/strict_cross_domain/tasks/dr_grading/DDR"
        self.leaves.mkdir(parents=True)
        self.rows = {}
        for split in ("train", "val", "test", "external_test"):
            paths = [f"DDR/{split}_{view}.bytes" for view in range(2 if split == "train" else 1)]
            for relative in paths:
                path = self.data / relative
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(("not an image: " + relative).encode())
            self.rows[split] = [] if split == "external_test" else [{
                "sample_id": f"DDR:{split}", "dataset": "DDR", "task": "dr_grading", "split": split,
                "image_paths": paths, "image_sha256": [hashlib.sha256((self.data / p).read_bytes()).hexdigest() for p in paths],
                "labels": {"dr_grade": 0}, "label_mask": {"dr_grade": True},
                "patient_id": split, "content_group_id": split, "leakage_group_id": split}]
        self.config = {"task": "dr_grading", "kind": "multiclass", "labels": ["dr_grade"], "num_classes": 5,
            "data": {"datasets": ["DDR"], "protocol": "strict_cross_domain", "source_isolated": True,
                     "image_cache_dir": "/nonexistent/cloud/cache"}}
        self.config_path = self.root / "config.json"
        self.publish()

    def tearDown(self):
        self.temp.cleanup()

    def publish(self):
        index = {"format_version": 1, "all_checks_passed": True, "record_bytes_preserved": True,
                 "base_protocols": ["strict_cross_domain"], "combinations": [{"splits": {}}]}
        isolated = self.data / "prepared/source_isolated"
        for split, rows in self.rows.items():
            payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
            path = self.leaves / f"{split}.jsonl"
            path.write_bytes(payload)
            index["combinations"][0]["splits"][split] = {
                "path": path.relative_to(isolated).as_posix(), "bytes": len(payload),
                "records": len(rows), "sha256": hashlib.sha256(payload).hexdigest()}
        index_path = isolated / "isolation_manifest.json"
        index_path.write_text(json.dumps(index), encoding="utf-8")
        self.config["data"]["isolation_manifest_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def test_multi_view_full_hash_checks_originals_not_missing_cloud_cache(self):
        before = deepcopy(self.config)
        report = verify_source_images(self.config_path, self.data, workers=2)
        self.assertTrue(report["all_checks_passed"], report["errors"])
        self.assertEqual(report["unique_files"], 4)
        self.assertEqual(report["verified_files"], 4)
        self.assertEqual(report["splits"]["train"]["image_references"], 2)
        self.assertEqual(report["splits"]["train"]["records"], 1)
        self.assertTrue(report["image_cache_ignored_for_source_routing"])
        self.assertFalse(report["pixels_decoded"])
        self.assertEqual(self.config, before)

    def test_changed_or_missing_original_produces_failed_report(self):
        original = self.data / self.rows["test"][0]["image_paths"][0]
        original.write_bytes(b"changed image content")
        report = verify_source_images(self.config_path, self.data, workers=1)
        self.assertFalse(report["all_checks_passed"])
        self.assertIn("SHA-256 differs", str(report["errors"]))
        original.unlink()
        report = verify_source_images(self.config_path, self.data, workers=1)
        self.assertFalse(report["all_checks_passed"])
        self.assertIn("FileNotFoundError", str(report["errors"]))

    def test_misaligned_hashes_and_contamination_are_rejected(self):
        self.rows["test"][0]["image_sha256"] = []
        self.publish()
        report = verify_source_images(self.config_path, self.data)
        self.assertIn("aligned lists", str(report["errors"]))
        self.rows["test"][0]["dataset"] = "APTOS2019"
        self.publish()
        report = verify_source_images(self.config_path, self.data)
        self.assertFalse(report["all_checks_passed"])
        self.assertIn("contamination", str(report["errors"]))

    def test_manifest_tampering_and_source_path_escape_are_rejected(self):
        path = self.leaves / "test.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        report = verify_source_images(self.config_path, self.data)
        self.assertIn("audited isolation index", str(report["errors"]))
        self.rows["test"][0]["image_paths"] = ["../outside.bytes"]
        self.publish()
        report = verify_source_images(self.config_path, self.data)
        self.assertFalse(report["all_checks_passed"])
        self.assertIn("leaves data_root", str(report["errors"]))

    def test_file_mutated_during_stream_is_rejected(self):
        path = self.data / self.rows["test"][0]["image_paths"][0]
        expected = self.rows["test"][0]["image_sha256"][0]
        original_open = Path.open

        class ChangingReader:
            def __init__(self, handle):
                self.handle, self.changed = handle, False
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.handle.__exit__(*args)
            def fileno(self):
                return self.handle.fileno()
            def read(self, size):
                value = self.handle.read(size)
                if not self.changed:
                    self.changed = True
                    with original_open(path, "ab") as write:
                        write.write(b"modified during hash")
                return value

        def changing_open(p, mode="r", *args, **kwargs):
            handle = original_open(p, mode, *args, **kwargs)
            return ChangingReader(handle) if p == path and mode == "rb" else handle

        with patch.object(Path, "open", changing_open):
            result = _hash_file("DDR/test_0.bytes", path, expected)
        self.assertFalse(result["passed"])
        self.assertIn("changed during", result["error"])

    def test_cli_writes_failure_report_exits_nonzero_and_never_overwrites_data(self):
        (self.data / self.rows["test"][0]["image_paths"][0]).unlink()
        output = self.root / "reports/verification.json"
        command = [sys.executable, str(PROJECT / "scripts/verify_isolated_source_images.py"),
                   "--config", str(self.config_path), "--data-root", str(self.data),
                   "--output", str(output), "--workers", "2"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["all_checks_passed"])
        untouched = self.data / self.rows["train"][0]["image_paths"][0]
        before = untouched.read_bytes()
        with self.assertRaises(SystemExit):
            main(["--config", str(self.config_path), "--data-root", str(self.data), "--output", str(untouched)])
        self.assertEqual(untouched.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
