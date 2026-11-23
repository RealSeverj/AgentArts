"""Run the three explicitly authorized, source-isolated cloud experiments.

This coordinator never trains DR, resumes a run, shuts down a server, deletes
data, or includes weights/raw data in its small reports archive. Run under the
cloud virtualenv; all model computation is delegated to the existing launcher.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile


SPECS = (
    ("quality", "image_quality", "multiclass", "EyeQ", 30, 10664, 1875, 16249,
     "macro_f1", ["quality_grade"], "vit_base_patch14_dinov2.lvd142m", 392),
    ("multidisease", "multidisease", "multilabel", "RFMiD", 30, 1878, 634, 615,
     "macro_auprc", ["DR", "AMD", "MYOPIC_FUNDUS", "OPTIC_DISC_CUPPING", "ABNORMAL"],
     "vit_base_patch14_dinov2.lvd142m", 392),
    ("disc_refuge", "disc_cup_segmentation", "segmentation", "REFUGE", 50, 400, 400, 400,
     "macro_dice", ["optic_cup", "optic_disc"], "unet", 512),
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metric(metrics, name):
    result = metrics
    for component in name.split("."):
        result = result[component]
    require(result is not None and math.isfinite(float(result)), "Undefined metric: " + name)
    return float(result)


def observed_count(metrics):
    return metrics.get("n", metrics.get("samples"))


def expected_stage(config, epoch_zero_based):
    if config["kind"] == "segmentation":
        return "full"
    training = config["training"]
    if epoch_zero_based < int(training.get("freeze_backbone_epochs", 0)):
        return "head_only"
    n = int(training.get("unfreeze_last_n_blocks", 0))
    if n and epoch_zero_based < int(training.get("full_finetune_epoch", 0)):
        return f"last_{n}_blocks"
    return "full"


def validate_config(config, spec):
    key, task, kind, source, epochs, _, _, _, best, labels, architecture, size = spec
    require(config.get("task") == task and config.get("kind") == kind,
            f"{key}: unexpected task/kind")
    data = config.get("data", {})
    require(data.get("source_isolated") is True and data.get("datasets") == [source],
            f"{key}: requires exactly the isolated source {source}")
    require(not data.get("external_dataset") and not data.get("external_datasets"),
            f"{key}: external evaluation is outside this suite")
    require(config.get("labels") == labels, f"{key}: output label order changed")
    require(config.get("initialization") is None, f"{key}: DR/other expert weights must not be imported")
    require(config["model"].get("name") == architecture and config["model"].get("img_size") == size,
            f"{key}: architecture/input size changed")
    require(not config["model"].get("checkpoint"), f"{key}: unexpected local pretrained checkpoint")
    training = config["training"]
    require(training.get("epochs") == epochs, f"{key}: expected {epochs} complete epochs")
    require(training.get("best_metric") == best and training.get("best_mode") == "max",
            f"{key}: best checkpoint must use validation {best}")
    require(training.get("amp") == "fp16", f"{key}: V100 requires this prepared FP16 recipe")
    effective = int(training["batch_size"]) * int(training["grad_accum_steps"]) * 2
    require(effective == (16 if kind == "segmentation" else 32),
            f"{key}: unexpected global effective batch")
    if kind != "segmentation":
        require(training.get("freeze_backbone_epochs") == 5 and
                training.get("unfreeze_last_n_blocks") == 4 and
                training.get("full_finetune_epoch") == 30,
                f"{key}: expected head5/last4-blocks25 schedule")


def validate_history(run, config, spec):
    epochs, val_count, best_name = spec[4], spec[6], spec[8]
    path = run / "history.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(records) == epochs, f"{spec[0]}: incomplete or duplicated epoch history")
    require([record["epoch"] for record in records] == list(range(1, epochs + 1)),
            f"{spec[0]}: non-contiguous epoch history")
    best_record, best_score = None, -math.inf
    for index, record in enumerate(records):
        require(record["stage"] == expected_stage(config, index), f"Wrong trainable stage at epoch {index + 1}")
        require(observed_count(record["val"]) == val_count, "Validation did not cover the complete cohort")
        require(record.get("best_metric_name") == best_name, "History selection metric changed")
        score = metric(record["val"], best_name)
        improved = score > best_score
        require(record.get("improved") is improved, "History improvement flag is inconsistent")
        if improved:
            best_record, best_score = record, score
        require(math.isclose(float(record["best_metric"]), best_score, abs_tol=1e-12, rel_tol=1e-12),
                "History running best is inconsistent")
    return records, best_record, best_score


def audit_checkpoints(run, config, spec, manifest_checks, output_path=None):
    import torch
    torch.set_num_threads(1)
    records, best_record, best_score = validate_history(run, config, spec)
    require(load_json(run / "config.resolved.json") == config, "Resolved training configuration changed")
    runtime = load_json(run / "runtime.json")
    require(runtime.get("world_size") == 2 and runtime.get("smoke_test") is False,
            "Expected a full two-GPU training run")
    require(runtime.get("train_records") == spec[5] and runtime.get("val_records") == spec[6],
            "Runtime source sample counts changed")
    identity = load_json(run / "data_identity.json")
    for split in ("train", "val"):
        require(identity["manifests_sha256"][split] == manifest_checks[split]["sha256"],
                "Training manifest identity changed")
    output = {}
    for name, expected_epoch in (("best", best_record["epoch"]), ("last", spec[4])):
        path = run / (name + ".pt")
        require(path.is_file(), "Missing checkpoint: " + str(path))
        before = path.stat()
        digest = sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        require(checkpoint.get("config") == config, name + ": checkpoint configuration differs")
        require(checkpoint.get("data_identity") == identity, name + ": checkpoint data identity differs")
        require(checkpoint.get("world_size") == 2, name + ": checkpoint GPU world size differs")
        require(int(checkpoint["epoch"]) + 1 == expected_epoch, name + ": checkpoint epoch differs")
        require(math.isclose(float(checkpoint["best_metric"]), best_score, abs_tol=1e-12, rel_tol=1e-12),
                name + ": checkpoint best metric differs")
        model = checkpoint.get("model")
        require(isinstance(model, dict) and bool(model), name + ": empty model state")
        for tensor_name, tensor in model.items():
            require(torch.is_tensor(tensor) and tensor.device.type == "cpu", "Model state is not CPU tensors")
            if torch.is_floating_point(tensor):
                require(bool(torch.isfinite(tensor).all()), name + ": non-finite model tensor " + tensor_name)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                (after.st_size, after.st_mtime_ns, after.st_ino), "Checkpoint changed during audit")
        output[name] = {"path": str(path), "bytes": before.st_size, "sha256": digest,
                        "completed_epoch": expected_epoch, "stored_epoch_zero_based": int(checkpoint["epoch"]),
                        "model_tensor_count": len(model), "model_elements": sum(t.numel() for t in model.values()),
                        "cpu_load_and_finite_tensors_passed": True}
        del checkpoint, model
        gc.collect()
    report = {"created_utc": utc(), "task": spec[1], "source": spec[3], "all_checks_passed": True,
              "complete_training_epochs": len(records), "best_metric_name": spec[8],
              "best_validation_metric": best_score, "best_completed_epoch": best_record["epoch"],
              "best_validation_metrics": best_record["val"], "last_validation_metrics": records[-1]["val"],
              "checkpoints": output, "runtime": runtime, "manifest_checks": manifest_checks,
              "checkpoint_scope": "Owned local training outputs; loaded on CPU only for audit."}
    output_path = Path(output_path) if output_path else run / "checkpoint_metadata.json"
    require(not output_path.exists(), "Audit output already exists: " + str(output_path))
    write_json(output_path, report)
    return report


def prepare_config(config_path, spec, data_root):
    from scripts.dataset_routing import manifest_path, validate_training_route, validate_row
    config = load_json(config_path)
    validate_config(config, spec)
    validate_training_route(config, data_root)
    checks = {}
    for split, expected_count in zip(("train", "val", "test"), spec[5:8]):
        path = manifest_path(config, data_root, split)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        require(len(rows) == expected_count, f"{spec[0]}: {split} sample count differs")
        require(len({row["sample_id"] for row in rows}) == expected_count, "Duplicate sample IDs in manifest")
        for row in rows:
            validate_row(config, row, split)
        checks[split] = {"path": str(path), "records": len(rows), "sha256": sha256(path)}
    return config, checks


class Coordinator:
    def __init__(self, project, runtime_root, data_root, suite_root, configs, specs=SPECS):
        self.project, self.runtime_root, self.data_root, self.root = project, runtime_root, data_root, suite_root
        self.configs = configs
        self.specs = specs
        self.environment = os.environ.copy()
        self.environment.update(RETINA_RUNTIME_ROOT=str(runtime_root), RETINA_DATA_ROOT=str(data_root),
                                TORCH_HOME=str(runtime_root / "cache/torch"),
                                HF_HOME=str(runtime_root / "cache/huggingface"),
                                OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0,1", NPROC_PER_NODE="2",
                                PYTHONUNBUFFERED="1")
        self.environment.pop("TRAIN_WORKERS", None)
        self.python = runtime_root / "venv/bin/python"
        self.state = {"created_utc": utc(), "status": "preparing", "current_expert": None,
                      "scope": [spec[1] for spec in specs], "project": str(project),
                      "runtime_root": str(runtime_root), "data_root": str(data_root),
                      "suite_root": str(suite_root), "experts": {}, "commands": [],
                      "dr_baseline_cancelled": True, "server_shutdown_authorized_by_this_suite": False}

    def save(self):
        self.state["updated_utc"] = utc()
        write_json(self.root / "suite_status.json", self.state)

    def bundle(self):
        # Explicit report allowlist: no source images, masks, signed URLs, credentials or model files.
        names = {"suite_status.json", "suite_summary.json", "config.requested.json", "preflight.json",
                 "training_summary.json", "checkpoint_metadata.json", "evaluation_receipt.json",
                 "evaluation_test.json", "history.jsonl", "config.resolved.json", "runtime.json",
                 "data_identity.json", "training.log", "evaluation.log", "preflight.log"}
        paths = sorted(path for path in self.root.rglob("*") if path.is_file() and path.name in names)
        manifest = {"created_utc": utc(), "contains_weights": False, "contains_source_images_or_masks": False,
                    "files": [{"path": str(path.relative_to(self.root)).replace("\\", "/"),
                               "bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths]}
        temporary = self.root / "reports_bundle.zip.tmp"
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, str(path.relative_to(self.root)).replace("\\", "/"))
            archive.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, self.root / "reports_bundle.zip")
        write_json(self.root / "reports_bundle_receipt.json", {
            "created_utc": utc(), "path": str(self.root / "reports_bundle.zip"),
            "bytes": (self.root / "reports_bundle.zip").stat().st_size,
            "sha256": sha256(self.root / "reports_bundle.zip"), "contains_weights": False,
            "file_count": len(paths) + 1})

    def command(self, command, log, expert, stage):
        record = {"expert": expert, "stage": stage, "command": [str(value) for value in command],
                  "cwd": str(self.project), "log": str(log), "started_utc": utc()}
        self.state["commands"].append(record)
        self.state["current_expert"] = expert
        self.state["status"] = stage
        self.save()
        print(json.dumps({"event": "stage_started", **record}, ensure_ascii=False), flush=True)
        started = time.monotonic()
        with log.open("xb") as handle:
            process = subprocess.Popen(record["command"], cwd=self.project, env=self.environment,
                                       stdout=handle, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            self.save()
            code = process.wait()
        record.update(completed_utc=utc(), elapsed_seconds=round(time.monotonic() - started, 3), exit_code=code)
        self.save()
        require(code == 0, f"{expert} {stage} failed with exit {code}; inspect {log}")
        print(json.dumps({"event": "stage_completed", **record}, ensure_ascii=False), flush=True)
        return record

    def run(self):
        prepared = []
        # Gate the entire requested suite before spending GPU time on its first model.
        for spec, config_path in zip(self.specs, self.configs):
            config, checks = prepare_config(config_path, spec, self.data_root)
            prepared.append((spec, config_path, config, checks))
        self.state["status"] = "validated"
        self.save()
        for spec, config_path, config, checks in prepared:
            key = spec[0]
            expert_root = self.root / key
            expert_root.mkdir()
            run = expert_root / "run"
            evaluation = expert_root / "evaluation_best_test"
            self.state["experts"][key] = {"status": "preflight", "task": spec[1], "source": spec[3],
                                           "config_path": str(config_path), "config_sha256": sha256(config_path),
                                           "run": str(run), "evaluation": str(evaluation),
                                           "manifest_checks": checks}
            write_json(expert_root / "config.requested.json", config)
            self.command([self.python, "scripts/cloud_preflight.py", "--config", config_path,
                          "--data-root", self.data_root], expert_root / "preflight.log", key, "preflight")
            preflight = load_json(expert_root / "preflight.log")
            require(preflight.get("all_checks_passed") is True, "Preflight did not pass")
            write_json(expert_root / "preflight.json", preflight)
            self.command(["bash", "scripts/launch_v100.sh", config_path, run],
                         expert_root / "training.log", key, "training")
            audit = audit_checkpoints(run, config, spec, checks)
            training_summary = {"created_utc": utc(), "training_completed": True,
                                "test_evaluation_completed": False, "checkpoint_audit": audit}
            write_json(expert_root / "training_summary.json", training_summary)
            expert_state = self.state["experts"][key]
            expert_state.update(status="trained", checkpoint_metadata=audit)
            self.save()
            self.bundle()
            best = Path(audit["checkpoints"]["best"]["path"])
            require(sha256(best) == audit["checkpoints"]["best"]["sha256"], "Best checkpoint changed before evaluation")
            evaluation_command = ["bash", "scripts/launch_v100.sh", config_path, evaluation,
                                  "--eval-only", "--checkpoint", best, "--split", "test"]
            if spec[2] != "segmentation":
                evaluation_command.append("--eval-use-source-images")
            evaluation_record = self.command(
                evaluation_command,
                expert_root / "evaluation.log", key, "test_evaluation")
            report_path = evaluation / "evaluation_test.json"
            report = load_json(report_path)
            require(report.get("split") == "test" and report.get("config") == config,
                    "Test evaluation configuration/split differs")
            require(Path(report["checkpoint"]).resolve() == best.resolve(), "Test loaded an unexpected checkpoint")
            require(report.get("evaluated_dataset_count") == spec[7] and
                    observed_count(report["metrics"]) == spec[7], "Test evaluation was incomplete")
            metric(report["metrics"], spec[8])
            require(sha256(best) == audit["checkpoints"]["best"]["sha256"], "Best checkpoint changed during evaluation")
            receipt = {"created_utc": utc(), "all_checks_passed": True, "task": spec[1], "source": spec[3],
                       "checkpoint": audit["checkpoints"]["best"], "split": "test",
                       "manifest": checks["test"], "complete_test_cohort": True, "metrics": report["metrics"],
                       "report": str(report_path), "report_sha256": sha256(report_path),
                       "execution": evaluation_record, "world_size": 2,
                       "amp": config["training"]["amp"], "workers_per_rank": config["training"]["workers"],
                       "training_performed_in_evaluation": False, "optimizer_created_in_evaluation": False,
                       "test_used_for_best_checkpoint_selection": False,
                       "segmentation_metric_space": "whole image resized to 512" if spec[2] == "segmentation" else None}
            write_json(expert_root / "evaluation_receipt.json", receipt)
            expert_state.update(status="complete", evaluation_receipt=receipt)
            self.save()
            self.bundle()
        self.state.update(status="complete", current_expert=None, completed_utc=utc(), all_checks_passed=True)
        self.save()
        write_json(self.root / "suite_summary.json", {
            "created_utc": utc(), "all_checks_passed": True, "experts": self.state["experts"],
            "source_isolation_preserved": True, "dr_baseline_run": False,
            "server_shutdown_performed": False, "model_weights_remain_separate_from_reports_bundle": True,
            "limitations": ["One training seed per task; test metrics are not a clinical validation.",
                            "REFUGE refers to the previously audited local mirror and fixed manifests.",
                            "U-Net metrics use whole images resized to 512, not native resolution."]})
        self.bundle()
        print(json.dumps({"event": "suite_complete", "suite_root": str(self.root)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runtime-root", type=Path, default=Path("/mnt/retina"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--suite-root", type=Path, help="New/empty output directory; never resumed")
    parser.add_argument("--quality-config", type=Path)
    parser.add_argument("--multidisease-config", type=Path)
    parser.add_argument("--segmentation-config", type=Path)
    parser.add_argument("--start-from", choices=[spec[0] for spec in SPECS], default="quality",
                        help="Run only this expert and later ones; earlier outputs are not adopted or touched")
    parser.add_argument("--audit-existing-run", type=Path, help="Audit an already completed owned run without training")
    parser.add_argument("--audit-task", choices=[spec[0] for spec in SPECS])
    parser.add_argument("--audit-output", type=Path, help="New JSON path for the standalone CPU audit")
    args = parser.parse_args()
    project = args.project_root.resolve()
    runtime_root = args.runtime_root.resolve()
    data_root = (args.data_root or project / "data").resolve()
    require(sys.platform.startswith("linux"), "This coordinator is for the authorized Linux cloud host only")
    require((runtime_root / "venv/bin/python").is_file(), "Cloud virtualenv Python missing")
    require((project / "scripts/launch_v100.sh").is_file(), "Cloud project launcher missing")
    sys.path.insert(0, str(project))
    config_options = dict(zip([spec[0] for spec in SPECS],
                              (args.quality_config, args.multidisease_config, args.segmentation_config)))
    if args.audit_existing_run:
        require(args.audit_task is not None and args.audit_output is not None,
                "Standalone audit requires --audit-task and --audit-output")
        require(args.suite_root is None, "Standalone audit does not start a suite")
        spec = next(spec for spec in SPECS if spec[0] == args.audit_task)
        config_path = config_options[args.audit_task]
        require(config_path is not None, "Specify the corresponding expert --*-config for audit")
        config_path = config_path if config_path.is_absolute() else project / config_path
        config, checks = prepare_config(config_path.resolve(), spec, data_root)
        output = args.audit_output.resolve()
        require(output.parent.is_dir() and not output.exists(), "Audit output parent must exist and file must be new")
        report = audit_checkpoints(args.audit_existing_run.resolve(), config, spec, checks, output_path=output)
        print(json.dumps({"event": "standalone_audit_complete", "output": str(output),
                          "best_completed_epoch": report["best_completed_epoch"],
                          "best_validation_metric": report["best_validation_metric"]}), flush=True)
        return
    require(args.audit_task is None and args.audit_output is None, "Audit options require --audit-existing-run")
    require(args.suite_root is not None, "Training requires --suite-root")
    suite_root = args.suite_root.resolve()
    require(suite_root != project and not suite_root.is_relative_to(data_root), "Output cannot be the project or raw data")
    if suite_root.exists():
        require(suite_root.is_dir() and not any(suite_root.iterdir()), "Suite output must be new/empty")
    suite_root.mkdir(parents=True, exist_ok=True)
    with (suite_root / "coordinator.lock").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "created_utc": utc()}) + "\n")
    specs = SPECS[next(i for i, spec in enumerate(SPECS) if spec[0] == args.start_from):]
    selected_configs = [config_options[spec[0]] for spec in specs]
    require(all(path is not None for path in selected_configs), "Specify --*-config for all selected experts")
    configs = [path if path.is_absolute() else project / path for path in selected_configs]
    configs = [path.resolve() for path in configs]
    coordinator = Coordinator(project, runtime_root, data_root, suite_root, configs, specs=specs)
    coordinator.save()
    try:
        coordinator.run()
    except BaseException as error:
        coordinator.state.update(status="failed", failed_utc=utc(), all_checks_passed=False,
                                 error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        coordinator.save()
        try:
            coordinator.bundle()
        except Exception as bundle_error:
            print("Failed to export failure bundle: " + str(bundle_error), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
