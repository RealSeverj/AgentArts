"""Export completed cloud experts as immutable weight snapshots and small reports.

No training, network connection, source-data copying, shutdown, or cleanup is
performed. The completed CPU checkpoint audits are reused only after matching
their whole-file SHA256 to the current source and exported snapshot.
"""
from __future__ import annotations

import argparse
import errno
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts import run_remaining_experts_20261114 as suite


FAST_CONFIGS = {
    "quality": "image_quality_vitb_eyeq_v100_fast_20261114.json",
    "multidisease": "multidisease_vitb_rfmid_v100_fast_20261114.json",
    "disc_refuge": "disc_cup_unet_refuge_v100_fast_20261114.json",
}
CODE_FILES = (
    "scripts/run_remaining_experts_20261114.py", "scripts/continue_after_quality_20261114.py",
    "scripts/export_remaining_experts_20261114.py", "tests/test_remaining_experts_coordinator.py",
    "tests/test_quality_continuation.py", "tests/test_remaining_experts_export.py",
    "requirements-training.txt", "requirements-tools.txt",
)
PROFILE_FILES = ("quality_probe.json", "multidisease_probe.json", "disc_probe.json",
                 "quality.log", "multidisease.log", "disc.log", "profile_entry.py")


def stat_signature(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def stable_hash(path):
    path = Path(path)
    suite.require(path.is_file() and not path.is_symlink(), "Expected ordinary file: " + str(path))
    before = stat_signature(path)
    digest = suite.sha256(path)
    suite.require(stat_signature(path) == before, "File changed while hashing: " + str(path))
    return {"path": str(path), "bytes": before[2], "sha256": digest}


def publish_weight(source, target, expected):
    """Exclusive hardlink; cross-filesystem copy2 touches only our reserved file."""
    source, target = Path(source), Path(target)
    suite.require(not target.exists(), "Export weight already exists: " + str(target))
    verified = stable_hash(source)
    suite.require(verified["sha256"] == expected["sha256"] and verified["bytes"] == expected["bytes"],
                  "Source checkpoint no longer matches its completed audit")
    before = stat_signature(source)
    try:
        os.link(source, target)
        mode = "hardlink_to_completed_best_snapshot"
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        # O_EXCL reserves a new file. copy2 never targets an existing user file.
        with target.open("xb"):
            pass
        shutil.copy2(source, target)
        mode = "cross_filesystem_copy2"
    exported = stable_hash(target)
    suite.require(stat_signature(source) == before, "Source checkpoint changed during export")
    suite.require(exported["sha256"] == verified["sha256"] and exported["bytes"] == verified["bytes"],
                  "Exported checkpoint differs from its source")
    return {**exported, "filename": target.name, "source_path": str(source), "publish_mode": mode,
            "source_and_export_whole_sha256_verified": True,
            "cpu_finite_audit_reused_from_same_sha256": True,
            "original_checkpoint_preserved": True}


def require_completions(operational, suite_root):
    continuation = suite.load_json(operational / "continuation_status.json")
    status = suite.load_json(suite_root / "suite_status.json")
    summary_path = suite_root / "suite_summary.json"
    summary = suite.load_json(summary_path)
    for name, record in (("continuation", continuation), ("suite status", status)):
        suite.require(record.get("status") == "complete" and record.get("all_checks_passed") is True,
                      name + " is not successfully complete; no backup completion can be claimed")
    suite.require(summary.get("all_checks_passed") is True and
                  set(summary.get("experts", {})) == {"multidisease", "disc_refuge"},
                  "Remaining suite summary is incomplete or has unexpected experts")
    suite.require(Path(continuation["remaining_suite_summary"]).resolve() == summary_path.resolve() and
                  continuation.get("remaining_suite_summary_sha256") == suite.sha256(summary_path),
                  "Continuation does not identify this exact completed suite summary")
    return continuation, status, summary


def verify_expert(spec, run, audit_path, receipt_path, report_path, config_path, project):
    key = spec[0]
    config = suite.load_json(run / "config.resolved.json")
    suite.validate_config(config, spec)
    suite.require(suite.load_json(config_path) == config, key + ": reproducibility config differs from actual run")
    records, best_record, best_score = suite.validate_history(run, config, spec)
    audit, receipt, report = (suite.load_json(path) for path in (audit_path, receipt_path, report_path))
    suite.require(audit.get("all_checks_passed") is True and audit.get("task") == spec[1] and
                  audit.get("source") == spec[3] and audit.get("complete_training_epochs") == spec[4],
                  key + ": completed checkpoint audit is missing or inconsistent")
    suite.require(audit.get("best_completed_epoch") == best_record["epoch"] and
                  audit.get("best_metric_name") == spec[8] and
                  math.isclose(float(audit["best_validation_metric"]), best_score, abs_tol=1e-12),
                  key + ": best checkpoint was not selected by the complete validation history")
    best_info = audit["checkpoints"]["best"]
    expected_best = (run / "best.pt").resolve()
    suite.require(Path(best_info["path"]).resolve() == expected_best and
                  best_info.get("completed_epoch") == best_record["epoch"] and
                  best_info.get("cpu_load_and_finite_tensors_passed") is True,
                  key + ": CPU finite-tensor checkpoint audit is not tied to the selected best")
    suite.require(receipt.get("all_checks_passed") is True and receipt.get("task") == spec[1] and
                  receipt.get("source") == spec[3] and receipt.get("split") == "test" and
                  receipt.get("complete_test_cohort") is True and receipt.get("checkpoint") == best_info,
                  key + ": evaluation receipt does not identify the audited best checkpoint")
    suite.require(receipt.get("training_performed_in_evaluation") is False and
                  receipt.get("optimizer_created_in_evaluation") is False and
                  receipt.get("test_used_for_best_checkpoint_selection") is False,
                  key + ": test evaluation/selection safeguards differ")
    suite.require(Path(receipt["report"]).resolve() == report_path.resolve() and
                  receipt["report_sha256"] == suite.sha256(report_path), key + ": test report SHA differs")
    suite.require(report.get("split") == "test" and report.get("config") == config and
                  Path(report["checkpoint"]).resolve() == expected_best and
                  report.get("evaluated_dataset_count") == spec[7] and
                  suite.observed_count(report["metrics"]) == spec[7] and
                  report["metrics"] == receipt.get("metrics"), key + ": incomplete or inconsistent test report")
    suite.metric(report["metrics"], spec[8])
    from scripts.dataset_routing import manifest_path
    test_manifest = manifest_path(config, project / "data", "test").resolve()
    suite.require(Path(receipt["manifest"]["path"]).resolve() == test_manifest and
                  receipt["manifest"]["records"] == spec[7] and
                  receipt["manifest"]["sha256"] == suite.sha256(test_manifest), key + ": test manifest differs")
    identity = suite.load_json(run / "data_identity.json")
    runtime = suite.load_json(run / "runtime.json")
    suite.require(runtime.get("world_size") == 2 and runtime.get("smoke_test") is False and
                  runtime.get("train_records") == spec[5] and runtime.get("val_records") == spec[6],
                  key + ": formal full training runtime differs")
    for split in ("train", "val"):
        manifest = manifest_path(config, project / "data", split).resolve()
        digest = suite.sha256(manifest)
        suite.require(identity["manifests_sha256"][split] == digest and
                      audit["manifest_checks"][split]["sha256"] == digest,
                      key + ": train/validation identity differs")
    current = stable_hash(expected_best)
    suite.require(current["sha256"] == best_info["sha256"] and current["bytes"] == best_info["bytes"],
                  key + ": current checkpoint differs from CPU-audited and evaluated snapshot")
    return {"key": key, "task": spec[1], "source": spec[3], "train_val_test": list(spec[5:8]),
            "completed_training_epochs": len(records), "best_completed_epoch": best_record["epoch"],
            "best_selection": {"split": "val", "metric": spec[8], "value": best_score,
                               "test_used_for_selection": False},
            "architecture": config["model"]["name"], "labels": config["labels"],
            "num_classes": config.get("num_classes"), "input_size": config["model"]["img_size"],
            "preprocessing": config.get("preprocessing", {"mode": "stretch"}),
            "normalization": {"mean": config["model"]["mean"], "std": config["model"]["std"]},
            "test_metrics": report["metrics"], "checkpoint_expected": best_info,
            "receipt_sha256": suite.sha256(receipt_path), "test_report_sha256": suite.sha256(report_path),
            "test_manifest_sha256": receipt["manifest"]["sha256"], "all_checks_passed": True}


class ReportAllowlist:
    def __init__(self):
        self.entries = []
        self.names = set()

    def add(self, path, archive_name, required=True):
        path = Path(path)
        if not path.exists() and not required:
            return
        suite.require(path.is_file() and not path.is_symlink(), "Required report/code file missing: " + str(path))
        name = str(archive_name).replace("\\", "/")
        suite.require(not name.startswith("/") and ".." not in Path(name).parts and name not in self.names,
                      "Unsafe or duplicate archive member")
        suite.require(path.suffix.lower() in {".json", ".jsonl", ".log", ".py", ".txt", ".stderr"},
                      "Weights, raw images and masks are forbidden in the small reports archive")
        self.names.add(name)
        self.entries.append((path, name))

    def write(self, target, generated):
        target = Path(target)
        suite.require(not target.exists(), "Reports archive already exists")
        records = []
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, name in self.entries:
                before = stat_signature(path)
                record = stable_hash(path)
                archive.write(path, name)
                suite.require(stat_signature(path) == before, "Report changed while archived: " + str(path))
                records.append({**record, "archive_name": name})
            for name, payload in generated.items():
                suite.require(name not in self.names, "Generated report collides with a file")
                archive.writestr(name, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            archive.writestr("reports_bundle_manifest.json", json.dumps({
                "created_utc": suite.utc(), "contains_weights": False,
                "contains_raw_images_or_masks": False, "contains_private_signed_urls": False,
                "source_files": records, "generated_files": sorted(generated)}, ensure_ascii=False, indent=2) + "\n")
        return {**stable_hash(target), "filename": target.name, "file_count": len(records) + len(generated) + 1,
                "contains_weights": False, "contains_raw_images_or_masks": False}


def bounded_profiles(profile_root):
    if profile_root is None:
        return {"probe_only": True, "included": False, "reason": "No optional --profile-root was supplied"}
    profiles = {}
    for key, log_name in (("quality", "quality.log"), ("multidisease", "multidisease.log"),
                          ("disc_refuge", "disc.log")):
        path = profile_root / log_name
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(event, dict) and any(name in event for name in
                                               ("profile_wall_seconds", "peak_allocated_mib", "peak_reserved_mib",
                                                "reserved_mib")):
                events.append(event)
        profiles[key] = {"probe_only": True, "log_path": str(path), "parsed_memory_timing_events": events,
                         "declared_max_train_steps": 4, "declared_max_val_steps": 2,
                         "formal_initialization_referenced_probe": False}
    return {"probe_only": True, "included": True, "profiles": profiles,
            "warning": "Bounded speed/memory probes only. Their metrics/weights are not formal results or initialization."}


def export(args):
    root, project, operational, suite_root = args.export_root, args.project_root, args.quality_operational_root, args.suite_root
    state = {"created_utc": suite.utc(), "status": "verifying", "all_checks_passed": False,
             "export_root": str(root), "models": {}, "dr_exported": False}
    suite.write_json(root / "export_status.json", state)
    try:
        continuation, status, summary = require_completions(operational, suite_root)
        files = ReportAllowlist()
        for name in ("continuation_status.json", "quality_checkpoint_metadata.json", "quality_evaluation_receipt.json",
                     "quality_cpu_audit.log", "quality_test_evaluation.log", "remaining_two_experts.log"):
            files.add(operational / name, "continuation/" + name)
        for name in ("suite_status.json", "suite_summary.json"):
            files.add(suite_root / name, "remaining_suite/" + name)
        for spec in suite.SPECS:
            key = spec[0]
            if key == "quality":
                run = args.quality_run
                audit_path = operational / "quality_checkpoint_metadata.json"
                receipt_path = operational / "quality_evaluation_receipt.json"
                report_path = operational / "quality_evaluation_best_test/evaluation_test.json"
            else:
                expert_root = suite_root / key
                run = expert_root / "run"
                audit_path = run / "checkpoint_metadata.json"
                receipt_path = expert_root / "evaluation_receipt.json"
                report_path = expert_root / "evaluation_best_test/evaluation_test.json"
                for name in ("config.requested.json", "preflight.json", "preflight.log", "training.log",
                             "evaluation.log", "training_summary.json"):
                    files.add(expert_root / name, key + "/" + name)
                files.add(audit_path, key + "/run/checkpoint_metadata.json")
                files.add(receipt_path, key + "/evaluation_receipt.json")
                suite.require(summary["experts"][key].get("status") == "complete" and
                              summary["experts"][key].get("evaluation_receipt") == suite.load_json(receipt_path),
                              key + ": summary and actual test receipt differ")
            config_path = project / "configs/experiments" / FAST_CONFIGS[key]
            verified = verify_expert(spec, run, audit_path, receipt_path, report_path, config_path, project)
            if key == "quality":
                suite.require(continuation.get("quality_evaluation_receipt") == suite.load_json(receipt_path),
                              "Quality continuation and actual receipt differ")
            exported = publish_weight(run / "best.pt", root / (key + "_best.pt"), verified.pop("checkpoint_expected"))
            verified["weights"] = exported
            state["models"][key] = verified
            state["status"] = "exporting"
            suite.write_json(root / "export_status.json", state)
            for name in ("config.resolved.json", "history.jsonl", "runtime.json", "data_identity.json"):
                files.add(run / name, key + "/run/" + name)
            files.add(report_path, key + "/evaluation_best_test/evaluation_test.json")
            files.add(config_path, "code/configs/experiments/" + config_path.name)
        for name, expected_source in (("quality_original_sha256.json", "EyeQ"),
                                      ("multidisease_original_sha256.json", "RFMiD")):
            path = args.source_check_root / name
            report = suite.load_json(path)
            suite.require(report.get("all_checks_passed") is True and report.get("declared_source") == expected_source and
                          report.get("verified_files") == report.get("unique_files") and report.get("unique_files", 0) > 0,
                          "Original-image SHA report is incomplete: " + name)
            files.add(path, "source_checks/" + name)
        preflight = suite.load_json(args.source_check_root / "quality_preflight.json")
        suite.require(preflight.get("all_checks_passed") is True, "Quality successful preflight is missing")
        for name in ("quality_preflight.json", "quality_training.log", "local_plan.json", "source_verification.log", "continuation.log"):
            files.add(args.source_check_root / name, "source_checks/" + name)
        files.add(args.source_check_root / "quality_preflight.stderr", "source_checks/quality_preflight.stderr", required=False)
        for relative in CODE_FILES:
            files.add(project / relative, "code/" + relative)
        profiles = bounded_profiles(args.profile_root)
        if args.profile_root is not None:
            for name in PROFILE_FILES:
                files.add(args.profile_root / name, "bounded_profile_PROBE_ONLY/" + name)
        interpretation = {
            "quality": {"output": "3-class softmax", "class_order": {"0": "Good", "1": "Usable", "2": "Reject"}},
            "multidisease": {"output": "5 independent sigmoids", "label_order": suite.SPECS[1][9],
                             "unknown_targets_are_masked": True},
            "disc_refuge": {"output": "2 sigmoid masks", "channel_order": {"0": "optic_cup", "1": "optic_disc"},
                            "input_size": 512, "metric": "Per-channel global-pixel Dice aggregated over the entire test cohort; macro average across cup/disc",
                            "evaluation_space": "Whole image resized to 512, not native-resolution/sliding-window evaluation",
                            "source_provenance": "Previously audited REFUGE/REFUGE2 mixed public mirror and unchanged isolated manifests; not claimed to be a pure official REFUGE version"},
            "dr": "DDR DR weights and results already backed up separately; deliberately not duplicated here",
            "source_code": "Fast configs, coordinators, fixture tests and requirements included. Attach the separately retained main training source snapshot for complete reproduction.",
            "backup_scope": "Best checkpoint of each of the three completed new experts, small reports and reproducibility additions. Raw datasets/masks and last checkpoints remain at their original paths.",
        }
        bundle = files.write(root / "reports_bundle.zip", {"bounded_profiles.json": profiles,
                                                          "output_interpretation.json": interpretation})
        manifest = {"created_utc": suite.utc(), "all_checks_passed": True, "status": "complete",
                    "models": state["models"], "reports_bundle": bundle, "output_interpretation": interpretation,
                    "bounded_profiles_are_probe_only": True, "dr_included": False,
                    "whole_file_sha256_and_sizes_verified": True, "original_paths_preserved": True,
                    "quality_run": str(args.quality_run), "quality_operational_root": str(operational),
                    "remaining_suite_root": str(suite_root), "cloud_only_export": True,
                    "local_download_completed": False}
        suite.write_json(root / "export_manifest.json", manifest)
        state.update(status="complete", completed_utc=suite.utc(), all_checks_passed=True, reports_bundle=bundle)
        suite.write_json(root / "export_status.json", state)
        # This completion marker is written only after every expected output is present and verified.
        suite.write_json(root / "backup_export_complete.json", {
            "created_utc": suite.utc(), "all_checks_passed": True, "status": "complete",
            "export_manifest": stable_hash(root / "export_manifest.json"),
            "weights": {key: item["weights"] for key, item in state["models"].items()},
            "reports_bundle": bundle, "local_download_completed": False,
            "scope": "Verified cloud export ready to download; not a claim of local backup or server shutdown."})
        print(json.dumps({"event": "backup_export_complete", "export_root": str(root),
                          "all_checks_passed": True}, ensure_ascii=False), flush=True)
    except BaseException as error:
        state.update(status="failed_partial_export", failed_utc=suite.utc(), all_checks_passed=False,
                     error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                     partial_files_preserved=True)
        suite.write_json(root / "export_status.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("quality-run", "quality-operational-root", "suite-root", "project-root", "source-check-root", "export-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--profile-root", type=Path)
    args = parser.parse_args()
    suite.require(sys.platform.startswith("linux"), "This exporter runs only on the authorized Linux cloud host")
    for name in ("quality_run", "quality_operational_root", "suite_root", "project_root", "source_check_root", "profile_root", "export_root"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    for name in ("quality_run", "quality_operational_root", "suite_root", "project_root", "source_check_root"):
        suite.require(getattr(args, name).is_dir(), "Input directory missing: " + str(getattr(args, name)))
    suite.require(args.profile_root is None or args.profile_root.is_dir(), "Optional profile directory missing")
    source_roots = [args.quality_run, args.quality_operational_root, args.suite_root,
                    args.project_root, args.source_check_root]
    if args.profile_root:
        source_roots.append(args.profile_root)
    suite.require(all(args.export_root != source and not args.export_root.is_relative_to(source) for source in source_roots),
                  "Export root must be outside all existing source/input directories")
    suite.require(not args.export_root.exists(), "Export root must be entirely new; partial/complete exports are never overwritten")
    args.export_root.mkdir(parents=True)
    sys.path.insert(0, str(args.project_root))
    export(args)


if __name__ == "__main__":
    main()
