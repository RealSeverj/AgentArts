"""Pure local backup guards; no real weights, data, network or GPU."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts import verify_remaining_experts_local_20261114 as verifier


def reports_zip(path, *, wrong_sha=False, extra=None):
    payload = b'{"fixture":true}\n'
    manifest = {
        "contains_weights": False, "contains_raw_images_or_masks": False,
        "source_files": [{"archive_name": "safe/report.json", "bytes": len(payload),
                          "sha256": "0" * 64 if wrong_sha else hashlib.sha256(payload).hexdigest()}],
        "generated_files": ["bounded_profiles.json"]}
    with zipfile.ZipFile(path, "x") as archive:
        archive.writestr("safe/report.json", payload)
        archive.writestr("bounded_profiles.json", '{"probe_only":true}')
        archive.writestr("reports_bundle_manifest.json", json.dumps(manifest))
        if extra:
            archive.writestr(extra, "unexpected")


class LocalVerificationGates(unittest.TestCase):
    def test_safe_names_reject_windows_paths_traversal_raw_and_weight_files(self):
        for name in ("../outside.json", "/absolute.log", "C:/windows.json", "x\\y.json",
                     "model.pt", "raw.png", "script.exe", "folder/./test.json"):
            with self.assertRaises(ValueError, msg=name):
                verifier.safe_member(name)
        self.assertEqual(verifier.safe_member("code/scripts/example.py"), "code/scripts/example.py")

    def test_crc_member_hash_and_exclusive_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports_zip(root / "reports.zip")
            result = verifier.verify_and_extract_reports(root / "reports.zip", root / "reports")
            self.assertTrue(result["crc_passed"])
            self.assertTrue(result["source_member_hashes_passed"])
            original = (root / "reports/safe/report.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "already exists"):
                verifier.verify_and_extract_reports(root / "reports.zip", root / "reports")
            self.assertEqual((root / "reports/safe/report.json").read_bytes(), original)

    def test_wrong_member_sha_or_unlisted_member_never_extracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports_zip(root / "wrong.zip", wrong_sha=True)
            with self.assertRaisesRegex(ValueError, "member SHA"):
                verifier.verify_and_extract_reports(root / "wrong.zip", root / "reports")
            self.assertFalse((root / "reports").exists())
            reports_zip(root / "extra.zip", extra="unlisted.json")
            with self.assertRaisesRegex(ValueError, "Unlisted"):
                verifier.verify_and_extract_reports(root / "extra.zip", root / "reports")
            self.assertFalse((root / "reports").exists())

    def test_auto_expected_file_hash_and_cloud_partial_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "weights.pt"
            data.write_bytes(b"fixture bytes only")
            expected = {"bytes": data.stat().st_size, "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
            self.assertTrue(verifier.verify_file(data, expected)["whole_file_hash_passed"])
            with self.assertRaisesRegex(ValueError, "SHA/size mismatch"):
                verifier.verify_file(data, {**expected, "sha256": "0" * 64})
            for name in ("backup_export_complete.json", "export_manifest.json", "export_status.json"):
                (root / name).write_text('{"status":"failed_partial_export","all_checks_passed":false}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete/partial"):
                verifier.verify_backup(root)
            self.assertFalse((root / "reports").exists())
            self.assertFalse((root / "backup_complete.json").exists())


if __name__ == "__main__":
    unittest.main()
