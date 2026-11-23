"""Verify downloaded owned expert checkpoints on CPU, then mark local backup complete.

No training, inference, networking, deletion, or moving is performed. This
standalone verifier needs Python/PyTorch and only the downloaded backup folder.
Existing result files and an existing reports directory are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile


LABELS = {
    "quality": ["quality_grade"],
    "multidisease": ["DR", "AMD", "MYOPIC_FUNDUS", "OPTIC_DISC_CUPPING", "ABNORMAL"],
    "disc_refuge": ["optic_cup", "optic_disc"],
}
TEST_COUNTS = {"quality": 16249, "multidisease": 615, "disc_refuge": 400}
ALLOWED_REPORT_SUFFIXES = {".json", ".jsonl", ".log", ".py", ".txt", ".stderr"}
OUTPUT_NAMES = ("local_backup_verification.json", "backup_complete.json", "本次训练结果.txt")


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def signature(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def hash_stream(handle):
    digest, size = hashlib.sha256(), 0
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        size += len(block)
        digest.update(block)
    return size, digest.hexdigest()


def verify_file(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Expected ordinary local file: " + str(path))
    before = signature(path)
    with path.open("rb") as handle:
        size, digest = hash_stream(handle)
    require(signature(path) == before, "Local file changed while hashing: " + path.name)
    require(size == expected["bytes"] and digest == expected["sha256"], "Local SHA/size mismatch: " + path.name)
    return {"filename": path.name, "path": str(path), "bytes": size, "sha256": digest,
            "whole_file_hash_passed": True}


def safe_member(name):
    require(isinstance(name, str) and bool(name) and "\\" not in name and ":" not in name and "\0" not in name,
            "Unsafe archive member name")
    parts = PurePosixPath(name).parts
    require(not name.startswith("/") and parts and all(part not in (".", "..") for part in parts),
            "Absolute/traversing archive member")
    require(str(PurePosixPath(name)) == name and not name.endswith("/"), "Noncanonical archive member")
    require(PurePosixPath(name).suffix.lower() in ALLOWED_REPORT_SUFFIXES,
            "Archive contains forbidden weights, images, masks or executable files: " + name)
    return name


def verify_and_extract_reports(path, target):
    """CRC + full member hashes before exclusive extraction into a new directory."""
    path, target = Path(path), Path(target)
    require(not target.exists(), "Reports directory already exists; refusing to overwrite")
    before = signature(path)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [safe_member(info.filename) for info in infos]
        require(len(set(name.casefold() for name in names)) == len(names), "Duplicate/case-colliding archive names")
        require(all(not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16) for info in infos),
                "Archive directories/symlinks are forbidden")
        require(sum(info.file_size for info in infos) <= 2 * 1024**3, "Small reports archive is unexpectedly large")
        require("reports_bundle_manifest.json" in names, "Report bundle member manifest missing")
        require(archive.testzip() is None, "Reports archive CRC check failed")
        manifest = json.loads(archive.read("reports_bundle_manifest.json").decode("utf-8"))
        require(manifest.get("contains_weights") is False and manifest.get("contains_raw_images_or_masks") is False,
                "Report manifest scope differs")
        entries = manifest["source_files"]
        source_names = [safe_member(item["archive_name"]) for item in entries]
        generated_names = [safe_member(name) for name in manifest["generated_files"]]
        expected_names = source_names + generated_names + ["reports_bundle_manifest.json"]
        require(len(set(name.casefold() for name in expected_names)) == len(expected_names) and
                set(expected_names) == set(names), "Unlisted/duplicated archive members")
        for item in entries:
            require(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None, "Invalid declared member SHA")
            with archive.open(item["archive_name"]) as handle:
                size, digest = hash_stream(handle)
            require(size == item["bytes"] and digest == item["sha256"],
                    "Archive member SHA/size mismatch: " + item["archive_name"])
        require(signature(path) == before, "Archive changed during verification")
        target.mkdir()
        for info in infos:
            destination = target.joinpath(*PurePosixPath(info.filename).parts)
            require(destination.resolve().is_relative_to(target.resolve()), "Extraction escaped reports directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
        require(signature(path) == before, "Archive changed during extraction")
    return {"crc_passed": True, "safe_names_passed": True, "source_member_hashes_passed": True,
            "source_member_count": len(entries), "file_count": len(infos), "extracted_to": str(target)}


def model_paths(reports, key):
    audit = (reports / "continuation/quality_checkpoint_metadata.json" if key == "quality" else
             reports / key / "run/checkpoint_metadata.json")
    receipt = (reports / "continuation/quality_evaluation_receipt.json" if key == "quality" else
               reports / key / "evaluation_receipt.json")
    return audit, receipt, reports / key / "evaluation_best_test/evaluation_test.json"


def verify_checkpoint(path, key, model, reports):
    import torch
    torch.set_num_threads(1)
    before = signature(path)
    audit_path, receipt_path, report_path = model_paths(reports, key)
    audit, receipt, report = (read_json(item) for item in (audit_path, receipt_path, report_path))
    config = read_json(reports / key / "run/config.resolved.json")
    identity = read_json(reports / key / "run/data_identity.json")
    best_info = audit["checkpoints"]["best"]
    require(audit.get("all_checks_passed") is True and receipt.get("all_checks_passed") is True and
            best_info["sha256"] == model["weights"]["sha256"] and
            best_info["bytes"] == model["weights"]["bytes"] and receipt["checkpoint"] == best_info,
            key + ": audit/receipt/weight identity differs")
    require(config["labels"] == model["labels"] == LABELS[key] and
            config["model"]["img_size"] == model["input_size"] and
            config["model"]["name"] == model["architecture"], key + ": model configuration differs")
    require(model["best_selection"]["split"] == "val" and
            model["best_selection"]["test_used_for_selection"] is False and
            receipt["test_used_for_best_checkpoint_selection"] is False,
            key + ": best selection is not validation-only")
    require(receipt["report_sha256"] == model["test_report_sha256"], key + ": test report hash identity differs")
    verify_file(report_path, {"bytes": report_path.stat().st_size, "sha256": receipt["report_sha256"]})
    require(report["config"] == config and report["split"] == "test" and
            report["evaluated_dataset_count"] == TEST_COUNTS[key] and
            report["metrics"].get("n", report["metrics"].get("samples")) == TEST_COUNTS[key] and
            report["metrics"] == receipt["metrics"] == model["test_metrics"], key + ": incomplete/different test metrics")
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    require(checkpoint["config"] == config and checkpoint["data_identity"] == identity,
            key + ": CPU checkpoint config/data identity differs")
    require(int(checkpoint["epoch"]) + 1 == model["best_completed_epoch"] == best_info["completed_epoch"] and
            checkpoint.get("world_size") == 2, key + ": CPU checkpoint epoch/world size differs")
    require(math.isclose(float(checkpoint["best_metric"]), float(model["best_selection"]["value"]), abs_tol=1e-12),
            key + ": CPU checkpoint selection metric differs")
    tensors = checkpoint["model"]
    require(isinstance(tensors, dict) and tensors, key + ": model state is empty")
    elements = 0
    for name, tensor in tensors.items():
        require(torch.is_tensor(tensor) and tensor.device.type == "cpu", key + ": non-CPU model tensor")
        if torch.is_floating_point(tensor):
            require(bool(torch.isfinite(tensor).all()), key + ": nonfinite tensor " + name)
        elements += tensor.numel()
    require(elements == best_info["model_elements"] and len(tensors) == best_info["model_tensor_count"],
            key + ": tensor element/count differs from cloud audit")
    if key in ("quality", "multidisease"):
        outputs = 3 if key == "quality" else 5
        require(tuple(tensors["head.1.weight"].shape) == (outputs, 768) and
                tuple(tensors["head.1.bias"].shape) == (outputs,), key + ": classifier output count differs")
        require(tuple(tensors["backbone.pos_embed"].shape) == (1, 785, 768), key + ": ViT-B392 position tensor differs")
        head_shape = list(tensors["head.1.weight"].shape)
    else:
        require(tuple(tensors["head.weight"].shape) == (2, 32, 1, 1) and
                tuple(tensors["head.bias"].shape) == (2,), "U-Net cup/disc head differs")
        head_shape = list(tensors["head.weight"].shape)
    thresholds = config.get("evaluation", {})
    if key in ("multidisease", "disc_refuge"):
        configured = thresholds.get("thresholds", [thresholds.get("threshold", 0.5)] * len(LABELS[key]))
        require(all(value == 0.5 for value in configured), key + ": threshold is not the fixed 0.5")
    require(signature(path) == before, key + ": weight file changed during CPU audit")
    record = {"all_checks_passed": True, "cpu_mmap_load_passed": True, "floating_tensors_finite": True,
              "model_elements": elements, "model_tensor_count": len(tensors), "head_shape": head_shape,
              "completed_epoch": int(checkpoint["epoch"]) + 1, "torch_local_version": torch.__version__,
              "test_records": TEST_COUNTS[key], "formal_local_training": False, "local_inference_performed": False}
    del tensors, checkpoint
    gc.collect()
    return record


def score(value, percent=False):
    return "未定义" if value is None else (f"{100 * value:.2f}%" if percent else f"{value:.4f}")


def result_text(models, dr=None):
    q, multi, seg = (models[key] for key in ("quality", "multidisease", "disc_refuge"))
    qm, mm, sm = q["test_metrics"], multi["test_metrics"], seg["test_metrics"]
    lines = ["本次三个模型训练结果及本地备份", "",
             "三份最佳权重、配置、完整测试报告和训练记录已核对 SHA-256，并在本地完成 CPU 加载与浮点张量有限性检查。",
             "最佳轮次由各自验证集选定。三个任务的数据清单、标签及输出目录保持独立。", "",
             f"一、质控：EyeQ，ViT-B/14，392，训练{q['completed_training_epochs']}轮，最佳第{q['best_completed_epoch']}轮。",
             "类别顺序：0=Good，1=Usable，2=Reject；softmax/argmax。",
             f"独立测试 {qm['n']} 张：准确率 {score(qm['accuracy'], True)}，宏 F1 {score(qm['macro_f1'])}。", "",
             f"二、多病种：RFMiD，ViT-B/14，392，训练{multi['completed_training_epochs']}轮，最佳第{multi['best_completed_epoch']}轮。",
             f"独立测试 {mm['n']} 张：宏 AP {score(mm['macro_auprc'])}，宏 AUC {score(mm['macro_auroc'])}。",
             "五个独立 sigmoid；固定阈值0.5，未使用测试集调阈值。AP为average precision。",
             "标签\tAP\tAUC\t敏感度/召回率\t特异度"]
    for label in LABELS["multidisease"]:
        values = mm["per_label"][label]
        lines.append("\t".join([label, score(values["auprc"]), score(values["auroc"]),
                                score(values["sensitivity"], True), score(values["specificity"], True)]))
    recall = mm["per_label"]["OPTIC_DISC_CUPPING"]["sensitivity"]
    lines.extend([f"短板：视盘凹陷召回率为 {score(recall, True)}，漏检仍需重点改善；高AUC不能替代该阈值下的召回率。", "",
                  f"三、盘杯分割：REFUGE镜像，U-Net，512，训练{seg['completed_training_epochs']}轮，最佳第{seg['best_completed_epoch']}轮。",
                  "通道顺序：0=optic_cup（视杯），1=optic_disc（视盘）；固定阈值0.5。",
                  f"独立测试 {sm['samples']} 张，宏 Dice {score(sm['macro_dice'])}。"])
    for label in LABELS["disc_refuge"]:
        values = sm["per_label"][label]
        lines.append(f"{label}：Dice {score(values['dice'])}，IoU {score(values['iou'])}。")
    lines.extend(["分割指标在整图缩放到512×512后计算：每通道先累计完整测试集的TP/FP/FN像素再算Dice/IoU，宏Dice为两通道均值。不是原分辨率指标，也不是逐图Dice均值。",
                  "数据来源为此前已审计的REFUGE/REFUGE2混合公开镜像及固定清单，不宣称为纯官方REFUGE版本。", "",
                  "备份范围：三个新模型的最佳权重；last权重及原始图片/掩码未重复复制。DR模型此前已单独备份。",
                  "小报告包含本轮配置、协调脚本、测试、依赖清单及训练/测试记录。主训练源码另见training_source_snapshot.zip。",
                  "性能探针资料均标注PROBE_ONLY，不是正式模型结果或初始化来源。",
                  "本脚本只验证本地文件，未进行本地训练、重新推理、云关机或智能体接入。"])
    if dr and dr.get("included"):
        lines.extend(["", f"此前完成的DDR结果（仅引用已有报告，未重新训练或测试）：QWK {score(dr['qwk'])}，准确率 {score(dr['accuracy'], True)}。"])
    return "\n".join(lines) + "\n"


def optional_dr(path):
    if path is None:
        return {"included": False}
    try:
        report = read_json(path)
        require(report.get("all_checks_passed") is True and report.get("source") == "DDR" and
                report.get("split") == "test", "Not a verified DDR test statistics report")
        return {"included": True, "path": str(path), "qwk": report["metrics"]["qwk"],
                "accuracy": report["metrics"]["accuracy"], "scope": "Previously completed report only; no new weight audit/inference."}
    except Exception as error:
        return {"included": False, "optional_warning": str(error)}


def verify_backup(root, dr_report=None):
    root = Path(root).resolve()
    require(root.is_dir(), "Local backup directory is missing")
    require(not (root / "reports").exists() and all(not (root / name).exists() for name in OUTPUT_NAMES),
            "Reports or completion outputs already exist; refusing to overwrite")
    marker = read_json(root / "backup_export_complete.json")
    manifest = read_json(root / "export_manifest.json")
    status = read_json(root / "export_status.json")
    require(marker.get("status") == manifest.get("status") == status.get("status") == "complete" and
            marker.get("all_checks_passed") is True and manifest.get("all_checks_passed") is True and
            status.get("all_checks_passed") is True, "Cloud export is incomplete/partial")
    manifest_file = verify_file(root / "export_manifest.json", marker["export_manifest"])
    require(set(manifest["models"]) == set(LABELS) and marker["weights"] == {
        key: model["weights"] for key, model in manifest["models"].items()} and
        status["models"] == manifest["models"] and marker["reports_bundle"] == manifest["reports_bundle"],
        "Cloud marker/manifest/status identities differ")
    verified = {}
    for key, model in manifest["models"].items():
        require(model["weights"]["filename"] == key + "_best.pt", "Unexpected flat weight filename")
        verified[key] = verify_file(root / (key + "_best.pt"), model["weights"])
    bundle = verify_file(root / "reports_bundle.zip", manifest["reports_bundle"])
    pending = read_json(root / "backup_pending.json")
    snapshot = verify_file(root / "training_source_snapshot.zip", pending["source_snapshot"])
    with zipfile.ZipFile(root / "training_source_snapshot.zip") as archive:
        require(archive.testzip() is None, "Main training source snapshot CRC failed")
    archive_report = verify_and_extract_reports(root / "reports_bundle.zip", root / "reports")
    for key, model in manifest["models"].items():
        verified[key]["cpu_audit"] = verify_checkpoint(root / (key + "_best.pt"), key, model, root / "reports")
    dr = optional_dr(dr_report)
    text = result_text(manifest["models"], dr)
    with (root / "本次训练结果.txt").open("x", encoding="utf-8-sig") as handle:
        handle.write(text)
    result = {"created_utc": utc(), "all_checks_passed": True, "local_backup_complete": True,
              "backup_root": str(root), "export_manifest": manifest_file, "models": verified,
              "reports_bundle": bundle, "reports_archive_checks": archive_report,
              "training_source_snapshot": {**snapshot, "crc_passed": True},
              "optional_prior_dr_report": dr, "formal_local_training": False, "local_inference_performed": False,
              "cloud_shutdown_performed_by_verifier": False,
              "optional_artifacts": {"preview_png_json": "Optional and may be added later; not required for three verified weights.",
                                     "training_curves": "Optional and may be added later; not required for three verified weights."}}
    write_new_json(root / "local_backup_verification.json", result)
    write_new_json(root / "backup_complete.json", {
        "created_utc": utc(), "all_checks_passed": True, "local_backup_complete": True,
        "best_three_models_present_and_verified": True, "models": verified,
        "reports_bundle": bundle, "training_source_snapshot": snapshot,
        "last_checkpoints_duplicated": False, "raw_images_or_masks_duplicated": False,
        "dr_weights_preserved_in_separate_backup": True, "cloud_shutdown_claimed": False,
        "agent_integration_claimed": False, "verification_report": "local_backup_verification.json"})
    print(json.dumps({"event": "local_backup_verified", "backup_root": str(root), "all_checks_passed": True}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--dr-report", type=Path)
    args = parser.parse_args()
    try:
        verify_backup(args.backup_root, args.dr_report)
    except Exception as error:
        root = args.backup_root.resolve()
        failure = root / "local_backup_verification.failed.json"
        if root.is_dir() and not failure.exists() and not (root / "backup_complete.json").exists():
            write_new_json(failure, {"created_utc": utc(), "all_checks_passed": False,
                                    "local_backup_complete": False, "error_type": type(error).__name__,
                                    "error": str(error), "partial_extracted_outputs_preserved": True})
        raise


if __name__ == "__main__":
    main()
