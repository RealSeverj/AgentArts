"""Wait for one exact quality job, audit/test it, then run two new experts.

This ordinary cloud process never signals a training process, shuts down a
machine, resumes a run or adopts another task. All new outputs live in new
directories; the already running quality run is read-only to this coordinator.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts import run_remaining_experts_20261114 as suite


def proc_identity(pid, proc_root=Path("/proc")):
    """Read Linux stat starttime (field22), correctly skipping comm parentheses."""
    directory = proc_root / str(pid)
    try:
        stat = (directory / "stat").read_text(encoding="utf-8")
        _, separator, tail = stat.rpartition(")")
        suite.require(bool(separator), "Malformed process stat")
        fields = tail.split()
        suite.require(len(fields) >= 20, "Incomplete process stat")
        return {"pid": pid, "state": fields[0], "starttime_ticks": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def option_value(command, option):
    for index, item in enumerate(command):
        if item == option and index + 1 < len(command):
            return command[index + 1]
        if item.startswith(option + "="):
            return item.split("=", 1)[1]
    raise ValueError("Known training process lacks " + option)


def verify_training_command(pid, project, config, run, proc_root=Path("/proc")):
    directory = proc_root / str(pid)
    command = [part.decode("utf-8") for part in (directory / "cmdline").read_bytes().split(b"\0") if part]
    suite.require("training.train" in command, "PID does not identify the expected training module")
    suite.require(any("torchrun" in Path(part).name for part in command[:3]), "PID is not torchrun")
    cwd = (directory / "cwd").resolve(strict=True)
    suite.require(cwd == project, "Known training process cwd differs from project root")
    def resolved(value):
        value = Path(value)
        return (value if value.is_absolute() else cwd / value).resolve()
    suite.require(resolved(option_value(command, "--config")) == config,
                  "Known training PID is using a different configuration")
    suite.require(resolved(option_value(command, "--output")) == run,
                  "Known training PID is writing a different run")
    suite.require(option_value(command, "--nproc_per_node") == "2", "Expected two-GPU quality torchrun")
    return command


class Continuation:
    def __init__(self, args):
        self.args = args
        self.root = args.operational_root
        self.project = args.project_root
        self.python = args.runtime_root / "venv/bin/python"
        self.state = {"created_utc": suite.utc(), "status": "preparing", "training_pid": args.training_pid,
                      "quality_run": str(args.quality_run), "quality_config": str(args.quality_config),
                      "suite_root": str(args.suite_root), "commands": [],
                      "signals_sent_to_training_job": False, "server_shutdown_performed": False}
        self.environment = os.environ.copy()
        self.environment.update(RETINA_RUNTIME_ROOT=str(args.runtime_root),
                                RETINA_DATA_ROOT=str(self.project / "data"),
                                TORCH_HOME=str(args.runtime_root / "cache/torch"),
                                HF_HOME=str(args.runtime_root / "cache/huggingface"),
                                OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0,1", NPROC_PER_NODE="2",
                                PYTHONUNBUFFERED="1")
        self.environment.pop("TRAIN_WORKERS", None)

    def save(self):
        self.state["updated_utc"] = suite.utc()
        suite.write_json(self.root / "continuation_status.json", self.state)

    def event(self, name, **data):
        print(json.dumps({"event": name, "utc": suite.utc(), **data}, ensure_ascii=False), flush=True)

    def command(self, command, name):
        log = self.root / (name + ".log")
        record = {"stage": name, "command": [str(part) for part in command], "log": str(log),
                  "started_utc": suite.utc()}
        self.state["status"] = name
        self.state["commands"].append(record)
        self.save()
        self.event("stage_started", **record)
        started = time.monotonic()
        with log.open("xb") as handle:
            process = subprocess.Popen(record["command"], cwd=self.project, env=self.environment,
                                       stdout=handle, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            self.save()
            code = process.wait()
        record.update(completed_utc=suite.utc(), elapsed_seconds=round(time.monotonic() - started, 3), exit_code=code)
        self.save()
        suite.require(code == 0, f"{name} failed with exit {code}; inspect {log}")
        self.event("stage_completed", **record)
        return record

    def wait_quality(self):
        self.state["status"] = "waiting_for_exact_quality_process"
        self.state["poll_interval_seconds"] = 10
        self.state["wait_timeout_seconds"] = 4 * 60 * 60
        initial = proc_identity(self.args.training_pid)
        if initial is None:
            self.state["initial_process"] = None
            self.state["wait_exit_reason"] = "process_already_absent_at_coordinator_start"
            self.save()
            return
        # Validate cmdline only for this exact process, before waiting for it.
        self.state["training_command"] = verify_training_command(
            self.args.training_pid, self.project, self.args.quality_config, self.args.quality_run)
        checked = proc_identity(self.args.training_pid)
        suite.require(checked is not None and checked["starttime_ticks"] == initial["starttime_ticks"],
                      "Training PID identity changed during initial command verification")
        self.state["initial_process"] = initial
        self.save()
        self.event("waiting_quality", process=initial, timeout_seconds=4 * 60 * 60)
        started = time.monotonic()
        last_notice = started
        while True:
            current = proc_identity(self.args.training_pid)
            self.state["last_observed_process"] = current
            self.state["wait_elapsed_seconds"] = round(time.monotonic() - started, 3)
            if current is None:
                self.state["wait_exit_reason"] = "known_process_absent"
                break
            if current["starttime_ticks"] != initial["starttime_ticks"]:
                # The expected job ended; never wait for or inspect a reused PID's command line.
                self.state["wait_exit_reason"] = "pid_reused_expected_process_ended"
                break
            if current["state"] in ("Z", "X", "x"):
                self.state["wait_exit_reason"] = "known_process_exited_not_running"
                break
            self.save()
            if time.monotonic() - started >= 4 * 60 * 60:
                raise TimeoutError("Waited four hours; training process is left untouched")
            if time.monotonic() - last_notice >= 300:
                self.event("quality_still_running", process=current,
                           elapsed_seconds=self.state["wait_elapsed_seconds"])
                last_notice = time.monotonic()
            time.sleep(10)
        self.state["quality_process_finished_utc"] = suite.utc()
        self.save()
        self.event("known_quality_process_finished", reason=self.state["wait_exit_reason"])

    def run(self):
        spec = suite.SPECS[0]
        config, checks = suite.prepare_config(self.args.quality_config, spec, self.project / "data")
        self.wait_quality()
        # A successful exit cannot be inferred from /proc alone. Require complete
        # ordered training history before any evaluation or subsequent training.
        records, best_record, best_score = suite.validate_history(self.args.quality_run, config, spec)
        self.state["quality_history"] = {"complete_epochs": len(records), "best_epoch": best_record["epoch"],
                                         "best_validation_metric": best_score, "metric_name": spec[8]}
        self.save()
        audit_path = self.root / "quality_checkpoint_metadata.json"
        self.command([self.python, "scripts/run_remaining_experts_20261114.py",
                      "--project-root", self.project, "--runtime-root", self.args.runtime_root,
                      "--quality-config", self.args.quality_config,
                      "--audit-existing-run", self.args.quality_run, "--audit-task", "quality",
                      "--audit-output", audit_path], "quality_cpu_audit")
        audit = suite.load_json(audit_path)
        suite.require(audit.get("all_checks_passed") is True, "Quality checkpoint audit did not pass")
        best_info = audit["checkpoints"]["best"]
        best = Path(best_info["path"]).resolve()
        suite.require(best == self.args.quality_run / "best.pt", "Quality audit identified an unexpected best path")
        suite.require(suite.sha256(best) == best_info["sha256"], "Quality checkpoint changed before evaluation")
        evaluation = self.root / "quality_evaluation_best_test"
        execution = self.command(["bash", "scripts/launch_v100.sh", self.args.quality_config, evaluation,
                                  "--eval-only", "--checkpoint", best, "--split", "test",
                                  "--eval-use-source-images"], "quality_test_evaluation")
        report_path = evaluation / "evaluation_test.json"
        report = suite.load_json(report_path)
        suite.require(report.get("split") == "test" and report.get("config") == config,
                      "Quality evaluation config/split differs")
        suite.require(Path(report["checkpoint"]).resolve() == best,
                      "Quality evaluation loaded a different checkpoint")
        suite.require(report.get("evaluated_dataset_count") == 16249 and
                      suite.observed_count(report["metrics"]) == 16249,
                      "Quality evaluation did not cover all 16,249 held-out samples")
        suite.metric(report["metrics"], spec[8])
        suite.require(suite.sha256(best) == best_info["sha256"], "Quality checkpoint changed during evaluation")
        # Rehash the fixed test manifest after inference, not just the checkpoint.
        suite.require(suite.sha256(Path(checks["test"]["path"])) == checks["test"]["sha256"],
                      "Quality test manifest changed during evaluation")
        receipt = {"created_utc": suite.utc(), "all_checks_passed": True,
                   "task": spec[1], "source": spec[3], "checkpoint": best_info,
                   "split": "test", "manifest": checks["test"], "complete_test_cohort": True,
                   "metrics": report["metrics"], "report": str(report_path),
                   "report_sha256": suite.sha256(report_path), "execution": execution,
                   "world_size": 2, "amp": config["training"]["amp"],
                   "workers_per_rank": config["training"]["workers"],
                   "training_performed_in_evaluation": False, "optimizer_created_in_evaluation": False,
                   "best_selected_by_validation_only": True,
                   "test_used_for_best_checkpoint_selection": False}
        suite.write_json(self.root / "quality_evaluation_receipt.json", receipt)
        self.state.update(status="quality_complete", quality_checkpoint_audit=audit,
                          quality_evaluation_receipt=receipt)
        self.save()
        self.event("quality_test_complete", best_epoch=best_info["completed_epoch"], metrics=report["metrics"])
        self.command([self.python, "scripts/run_remaining_experts_20261114.py",
                      "--project-root", self.project, "--runtime-root", self.args.runtime_root,
                      "--suite-root", self.args.suite_root, "--start-from", "multidisease",
                      "--multidisease-config", self.args.multidisease_config,
                      "--segmentation-config", self.args.segmentation_config], "remaining_two_experts")
        summary = suite.load_json(self.args.suite_root / "suite_summary.json")
        suite.require(summary.get("all_checks_passed") is True and
                      set(summary.get("experts", {})) == {"multidisease", "disc_refuge"},
                      "Remaining-two-expert completion summary differs")
        self.state.update(status="complete", completed_utc=suite.utc(), all_checks_passed=True,
                          remaining_suite_summary=str(self.args.suite_root / "suite_summary.json"),
                          remaining_suite_summary_sha256=suite.sha256(self.args.suite_root / "suite_summary.json"))
        self.save()
        self.event("continuation_complete", operational_root=str(self.root), suite_root=str(self.args.suite_root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--quality-config", type=Path, required=True)
    parser.add_argument("--quality-run", type=Path, required=True)
    parser.add_argument("--operational-root", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--multidisease-config", type=Path, required=True)
    parser.add_argument("--segmentation-config", type=Path, required=True)
    args = parser.parse_args()
    suite.require(sys.platform.startswith("linux"), "This continuation is cloud Linux only")
    suite.require(args.training_pid > 1 and args.training_pid != os.getpid(), "Invalid exact training PID")
    args.project_root = args.project_root.resolve()
    args.runtime_root = args.runtime_root.resolve()
    for field in ("quality_config", "multidisease_config", "segmentation_config"):
        value = getattr(args, field)
        value = (value if value.is_absolute() else args.project_root / value).resolve()
        suite.require(value.is_file(), "Configuration missing: " + str(value))
        setattr(args, field, value)
    args.quality_run = args.quality_run.resolve()
    suite.require(args.quality_run.is_dir(), "Existing quality run missing")
    suite.require((args.runtime_root / "venv/bin/python").is_file(), "Cloud virtualenv missing")
    for field in ("operational_root", "suite_root"):
        value = getattr(args, field).resolve()
        suite.require(value != args.project_root and not value.is_relative_to(args.project_root / "data"),
                      "New output must not target the project or raw data")
        suite.require(not value.exists() or (value.is_dir() and not any(value.iterdir())),
                      "New output directory already has files: " + str(value))
        setattr(args, field, value)
    suite.require(args.operational_root != args.suite_root and
                  not args.operational_root.is_relative_to(args.suite_root) and
                  not args.suite_root.is_relative_to(args.operational_root) and
                  not args.operational_root.is_relative_to(args.quality_run) and
                  not args.suite_root.is_relative_to(args.quality_run),
                  "Continuation cannot write inside the already running quality run")
    args.operational_root.mkdir(parents=True, exist_ok=True)
    with (args.operational_root / "continuation.lock").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "created_utc": suite.utc()}) + "\n")
    sys.path.insert(0, str(args.project_root))
    continuation = Continuation(args)
    continuation.save()
    try:
        continuation.run()
    except BaseException as error:
        continuation.state.update(status="failed", failed_utc=suite.utc(), all_checks_passed=False,
                                  error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        continuation.save()
        raise


if __name__ == "__main__":
    main()
