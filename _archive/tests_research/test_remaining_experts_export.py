"""Pure export/receipt fixtures. Bytes are not real checkpoints or results."""
from copy import deepcopy
import errno
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import export_remaining_experts_20261114 as exporter
from scripts import run_remaining_experts_20261114 as suite
from tests.test_remaining_experts_coordinator import configuration, history


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class ExportGates(unittest.TestCase):
    def test_hardlink_snapshot_preserves_source_and_verifies_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source.pt", root / "export.pt"
            source.write_bytes(b"fixture-owned-checkpoint")
            expected = {"sha256": suite.sha256(source), "bytes": source.stat().st_size}
            report = exporter.publish_weight(source, target, expected)
            self.assertTrue(os.path.samefile(source, target))
            self.assertEqual(report["publish_mode"], "hardlink_to_completed_best_snapshot")
            self.assertEqual(source.read_bytes(), b"fixture-owned-checkpoint")
            self.assertEqual(report["sha256"], expected["sha256"])

    def test_cross_filesystem_copy2_and_reject_wrong_sha_or_existing_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source.pt", root / "copy.pt"
            source.write_bytes(b"source-stays-present")
            expected = {"sha256": suite.sha256(source), "bytes": source.stat().st_size}
            with patch.object(exporter.os, "link", side_effect=OSError(errno.EXDEV, "fixture cross filesystem")):
                report = exporter.publish_weight(source, target, expected)
            self.assertEqual(report["publish_mode"], "cross_filesystem_copy2")
            self.assertFalse(os.path.samefile(source, target))
            self.assertEqual(target.read_bytes(), source.read_bytes())
            with self.assertRaisesRegex(ValueError, "already exists"):
                exporter.publish_weight(source, target, expected)
            wrong = {**expected, "sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                exporter.publish_weight(source, root / "must_not_exist.pt", wrong)
            self.assertFalse((root / "must_not_exist.pt").exists())

    def test_completion_gate_requires_both_success_and_exact_summary_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            operational, remaining = root / "op", root / "suite"
            operational.mkdir()
            remaining.mkdir()
            summary = {"all_checks_passed": True, "experts": {"multidisease": {}, "disc_refuge": {}}}
            write_json(remaining / "suite_summary.json", summary)
            completed = {"status": "complete", "all_checks_passed": True}
            write_json(remaining / "suite_status.json", completed)
            continuation = {**completed, "remaining_suite_summary": str(remaining / "suite_summary.json"),
                            "remaining_suite_summary_sha256": suite.sha256(remaining / "suite_summary.json")}
            write_json(operational / "continuation_status.json", continuation)
            exporter.require_completions(operational, remaining)
            write_json(operational / "continuation_status.json", {**continuation, "status": "waiting"})
            with self.assertRaisesRegex(ValueError, "not successfully complete"):
                exporter.require_completions(operational, remaining)
            write_json(operational / "continuation_status.json", {**continuation, "remaining_suite_summary_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "exact completed suite"):
                exporter.require_completions(operational, remaining)

    def test_archive_allowlist_rejects_weights_images_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = exporter.ReportAllowlist()
            for name in ("fake.pt", "image.png"):
                path = root / name
                path.write_bytes(b"do-not-include")
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    reports.add(path, name)
            report = root / "report.json"
            write_json(report, {"fixture": True})
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                reports.add(report, "../report.json")
            reports.add(report, "safe/report.json")
            result = reports.write(root / "reports.zip", {"bounded_profiles.json": {"probe_only": True}})
            with zipfile.ZipFile(root / "reports.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"safe/report.json", "bounded_profiles.json", "reports_bundle_manifest.json"})
            self.assertFalse(result["contains_weights"])

    def test_failure_is_partial_without_success_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(export_root=root, project_root=root, quality_operational_root=root / "missing",
                                   suite_root=root / "missing_suite")
            with self.assertRaises(FileNotFoundError):
                exporter.export(args)
            status = suite.load_json(root / "export_status.json")
            self.assertEqual(status["status"], "failed_partial_export")
            self.assertFalse(status["all_checks_passed"])
            self.assertFalse((root / "backup_export_complete.json").exists())

    def test_receipt_joins_full_cohort_validation_best_and_weight_sha(self):
        spec = suite.SPECS[0]
        config = configuration(spec)
        config["model"].update(mean=[0.1, 0.2, 0.3], std=[1, 1, 1])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            cfg, audit_path, receipt_path, report_path = (root / name for name in
                                                         ("config.json", "audit.json", "receipt.json", "test.json"))
            write_json(cfg, config)
            write_json(run / "config.resolved.json", config)
            (run / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history(config, spec)), encoding="utf-8")
            manifests = {}
            for split in ("train", "val", "test"):
                manifests[split] = root / (split + ".jsonl")
                manifests[split].write_text("fixture-manifest-" + split, encoding="utf-8")
            write_json(run / "data_identity.json", {"manifests_sha256": {
                split: suite.sha256(manifests[split]) for split in ("train", "val")}})
            write_json(run / "runtime.json", {"world_size": 2, "smoke_test": False,
                                               "train_records": spec[5], "val_records": spec[6]})
            best = run / "best.pt"
            best.write_bytes(b"not-real-checkpoint-pure-fixture")
            info = {"path": str(best), "completed_epoch": 7, "cpu_load_and_finite_tensors_passed": True,
                    "sha256": suite.sha256(best), "bytes": best.stat().st_size}
            audit = {"all_checks_passed": True, "task": spec[1], "source": spec[3], "complete_training_epochs": 30,
                     "best_completed_epoch": 7, "best_metric_name": spec[8], "best_validation_metric": 0.9,
                     "checkpoints": {"best": info}, "manifest_checks": {
                         split: {"sha256": suite.sha256(manifests[split])} for split in ("train", "val")}}
            write_json(audit_path, audit)
            metrics = {"n": spec[7], spec[8]: 0.8}
            report = {"split": "test", "config": config, "checkpoint": str(best),
                      "evaluated_dataset_count": spec[7], "metrics": metrics}
            write_json(report_path, report)
            receipt = {"all_checks_passed": True, "task": spec[1], "source": spec[3], "split": "test",
                       "complete_test_cohort": True, "checkpoint": info,
                       "training_performed_in_evaluation": False, "optimizer_created_in_evaluation": False,
                       "test_used_for_best_checkpoint_selection": False,
                       "report": str(report_path), "report_sha256": suite.sha256(report_path), "metrics": metrics,
                       "manifest": {"path": str(manifests["test"]), "records": spec[7],
                                    "sha256": suite.sha256(manifests["test"])}}
            write_json(receipt_path, receipt)
            with patch("scripts.dataset_routing.manifest_path", side_effect=lambda c, r, s: manifests[s]):
                verified = exporter.verify_expert(spec, run, audit_path, receipt_path, report_path, cfg, root)
                self.assertEqual(verified["best_completed_epoch"], 7)
                self.assertEqual(verified["test_metrics"]["n"], 16249)
                wrong = deepcopy(receipt)
                wrong["checkpoint"]["sha256"] = "0" * 64
                write_json(receipt_path, wrong)
                with self.assertRaisesRegex(ValueError, "receipt does not identify"):
                    exporter.verify_expert(spec, run, audit_path, receipt_path, report_path, cfg, root)
                write_json(receipt_path, receipt)
                best.write_bytes(b"corrupted-after-audit")
                with self.assertRaisesRegex(ValueError, "current checkpoint differs"):
                    exporter.verify_expert(spec, run, audit_path, receipt_path, report_path, cfg, root)


if __name__ == "__main__":
    unittest.main()
