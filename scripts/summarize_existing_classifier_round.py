"""Build verified Chinese reports for the locked DDR and EyeQ research rounds."""
import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import research_classifier_views as views


RULES = {
    "original_argmax": "原图单视图",
    "historical_locked": "原图与历史阈值",
    "two_view_argmax": "原图与水平翻转图取均值",
    "two_view_safety_prior": "双视图与预设等级修正",
    "residual_two_view": "双视图与残差分类头",
    "residual_two_view_safety_prior": "双视图、残差分类头与等级修正",
    "two_view_reject": "双视图与验证集拒收阈值",
    "residual_two_view_reject": "双视图、残差拒收头与验证集阈值",
}
COHORTS = {"dr": {"train": "DDR 训练", "val": "DDR 验证", "test": "DDR 原始测试",
                  "external_test": "IDRiD 历史跨源回归"},
           "quality": {"train": "EyeQ 训练", "val": "EyeQ 验证", "test": "EyeQ 原始测试"}}
LABELS = {"accuracy": "准确率", "macro_f1": "宏 F1", "qwk": "二次加权 Kappa",
          "grade1_to0_rate": "1 级降为 0 级率", "grade3_under_rate": "3 级降为 0–2 级率",
          "severe_to01_rate": "3–4 级降为 0–1 级率", "low_to_severe_rate": "0–1 级升为 3–4 级率",
          "bad_passed_rate": "拒收图放行率", "gradable_rejected_rate": "可判读图误退率"}


def read(path):
    return json.loads(path.read_bytes())


def check_file(path, expected):
    if views.sha(path) != expected:
        raise ValueError(f"Receipt does not match {path}")


def frac(count, total):
    return f"{count}/{total}（{100 * count / total:.2f}%）" if total else "无分母"


def comparison(task, items):
    if task == "dr":
        lines = ["| 方案 | 准确率 | 宏 F1 | QWK | 1→0 | 3→0–2 | 3–4→0–1 | 0–1→3–4 |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
    else:
        lines = ["| 方案 | 准确率 | 宏 F1 | 拒收图漏放 | 可判读图误退 | 拒收类 AP |",
                 "|---|---:|---:|---:|---:|---:|"]
    for name, m in items:
        cells = [name, f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}"]
        if task == "dr":
            n = [sum(row) for row in m["confusion"]]
            cells += [f"{m['qwk']:.4f}", frac(m["grade1_to0"], n[1]), frac(m["grade3_under"], n[3]),
                      frac(m["severe_to01"], sum(n[3:])), frac(m["low_to_severe"], sum(n[:2]))]
        else:
            cells += [frac(m["bad_passed"], m["bad_images"]), frac(m["gradable_rejected"], m["gradable_images"]),
                      f"{m['reject_ap']:.4f}" if "reject_ap" in m else "—"]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def delta_table(report):
    lines = ["| 指标差值（入选减历史归档） | 点估计 | 95% 区间 |", "|---|---:|---:|"]
    for name, metric in report["paired_delta_ci95"]["metrics"].items():
        lo, hi = metric["ci95"]
        lines.append(f"| {LABELS[name]} | {metric['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    return lines


def build(task):
    out = views.REPORTS / views.DIRS[task]
    complete = read(out / "evaluation_complete.json")
    check_file(out / "policy_lock.json", complete["policy_lock_sha256"])
    lock = read(out / "policy_lock.json")
    for name, key in (("experiment_protocol.json", "protocol_sha256"), ("validation_candidates.json", "validation_candidates_sha256"),
                      ("validation_candidate_predictions.csv", "validation_scores_sha256"), ("trained_residual_head.npz", "residual_head_sha256")):
        check_file(out / name, lock[key])
    validation = read(out / "validation_candidates.json")
    rule = lock["rule"]
    val = validation["results"]
    reports = {}
    for split, expected in complete["cohort_reports"].items():
        check_file(out / f"{split}_report.json", expected)
        reports[split] = read(out / f"{split}_report.json")
        check_file(out / f"{split}_paired_predictions.csv", reports[split]["score_export_sha256"])
        if not (datetime.fromisoformat(lock["locked_utc"]) <= datetime.fromisoformat(read(out / f"{split}_view_extraction.json")["start_utc"])):
            raise ValueError("Heldout extraction starts before validation lock")
    receipts = {k: read(out / f"{k}_view_extraction.json") for k in COHORTS[task]}
    name = "模型 1 · 糖尿病视网膜病变五级分级" if task == "dr" else "模型 2 · 眼底图像质量"
    lines = [f"# {name}", "", f"**验证集入选方案为{RULES[rule]}。** " + (
        "输出依次为 0、1、2、3、4 级。" if task == "dr" else
        f"拒收概率达到 {lock['threshold']:.8f} 判为 Reject，其他图像在 Good、Usable 中选择得分较高者。"),
        "", "本报告用于离线研究。主干和原分类头沿用已归档的监督权重。" + (
            "既有 DINOv2 ViT-L/14 先在 APTOS2019 上接受监督训练，再在 DDR 上训练。输入沿用 336×336 的眼底裁切与等比补边。"
            if task == "dr" else "既有 DINOv2 ViT-B/14 以 EyeQ 接受监督训练。输入沿用 392×392 的整图缩放。") +
            "双视图方案将原图与水平翻转图分别归一化推理，对两次类别概率取平均。训练集用于另行拟合残差头，验证集用于选择决策规则，锁定后逐图复测原始留出。", ""]
    supplement=out/"ordinal_refinement"/"实验结果.md"
    if task=="dr" and supplement.is_file():
        lines += ["首轮规则保留原模型。二次探索加入了可转诊最低等级保护和只向上纠正一级的决策约束，另在查看本轮留出预测前锁定。两轮完整结果均已归档。[二次验证探索与留出回归](ordinal_refinement/实验结果.md)", ""]
    for split, report in reports.items():
        old, new = report["archived_locked_policy"], report["selected"]
        if task == "dr":
            lines.append(f"{COHORTS[task][split]}上，宏 F1 由 {old['macro_f1']:.4f} 变为 {new['macro_f1']:.4f}，QWK 由 {old['qwk']:.4f} 变为 {new['qwk']:.4f}。1 级降为 0 级由 {old['grade1_to0']} 变为 {new['grade1_to0']} 张，3 级降为 0–2 级由 {old['grade3_under']} 变为 {new['grade3_under']} 张。")
        else:
            lines.append(f"{COHORTS[task][split]}上，相对历史已锁定阈值，拒收图漏放由 {old['bad_passed']} 变为 {new['bad_passed']} 张，可判读图误退由 {old['gradable_rejected']} 变为 {new['gradable_rejected']} 张。")
        lines.append("")
    lines += ["## 验证集选择", "", f"锁定时间为 {lock['locked_utc']}（UTC）。" + (
        "先限制 QWK 和宏 F1 的下降、重度漏分与低等级升为重度的数量，再比较 1 级降为 0 级及 3 级降为较轻级别的比例。完整范围及并列规则保存在 experiment_protocol.json。"
        if task == "dr" else "以 Good 和 Usable 合计误退率不超过 5% 为验证集约束，优先减少 Reject 漏放，再减少误退。"), ""]
    lines += comparison(task, [(RULES[r] + ("（入选）" if r == rule else "") + ("" if v["eligible"] else "（超出约束）"), v["metrics"])
                               for r, v in val.items()])
    if task == "quality":
        reference = read(views.ARCHIVE / "eyeq/val_metrics.json")["validation_locked"]
        lines += ["", f"历史单视图原验证预测在原阈值下漏放 {reference['bad_passed']}/{reference['bad_images']} 张，误退 {reference['usable_rejected']}/{reference['usable_images']} 张。上表第一行为本机批量 4 重算；两次推理的浮点分数存在细小差异。"]
    for split, report in reports.items():
        lines += ["", f"## {COHORTS[task][split]}", "",
                  f"共有 {report['selected']['samples']} 张图，按样本标识与历史归档逐条配对。"
                  f"单视图复算与归档的类别最大值决策相差 {report['reference_replay_argmax_changed']} 张，单类分数最大绝对差为 {report['reference_replay_max_abs_score_error']:.6f}。", ""]
        items = ([("历史三类最大概率", report["archived_argmax"]), ("历史拒收阈值 0.5", report["archived_threshold_0_5"])] if task == "quality" else []) + [
            ("历史归档，原工作点", report["archived_locked_policy"]), ("同批复算，原工作点", report["replayed_original"]),
            ("本次入选", report["selected"])]
        lines += comparison(task, items)
        if task == "dr":
            graded = report["original_grading_metrics"]
            lines += ["", "沿用原训练报告的样本级和内容组加权算法，历史预测均已复算核对。", "",
                      "| 汇总口径 | 方案 | 有效权重 | 准确率 | 宏 F1 | QWK | 平均绝对等级误差 |", "|---|---|---:|---:|---:|---:|---:|"]
            for unit, unit_name in (("sample", "逐图"), ("content_weighted", "每内容组权重为 1")):
                for key, label in (("historical", "历史归档"), ("selected", "本次入选")):
                    m = graded[key][unit]
                    lines.append(f"| {unit_name} | {label} | {m['weight_sum']:.0f} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['qwk']:.4f} | {m['mean_absolute_grade_error']:.4f} |")
        units = "患者" if task == "quality" else "图像内容组"
        lines += ["", f"在 {report['paired_delta_ci95']['groups']} 个{units}内保持配对，对组有放回抽样 1,000 次。区间针对已锁定的模型和阈值；下表每项使用其原始 0–1 比例标度，0.01 对应率差 1 个百分点。", ""] + delta_table(report)
        if task == "quality":
            lines += ["", "误退率的 5% 上限仅用于选择验证集阈值，测试集按锁定阈值直接计数。"]
    lines += ["", "## 数据与复现", "", "按原始清单校验 SHA-256，并检查样本标识、原图 SHA-256 及可用分组的交集。"
              + ("DDR 和 IDRiD 的释出清单没有患者标识，分组取图像内容组，无法据此排除未标识的同一患者。IDRiD 已有历史跨源评测，本次用于固定回归。"
                 if task == "dr" else "EyeQ 的各清单均有患者标识，开发与留出的已知患者交集为零。本轮只有 EyeQ 同源历史留出。"),
              "", "| 数据 | 图像数 | 设备 | 单批图像数 | 双视图提取（秒） |", "|---|---:|---|---:|---:|"]
    for k, r in receipts.items():
        if r["records"] != views.SPLITS[task][k][1] or r["manifest_sha256"] != views.SPLITS[task][k][2]:
            raise ValueError("Data receipt mismatch")
        lines.append(f"| {COHORTS[task][k]} | {r['records']} | {r['device']} | {r['batch']} | {r['seconds']:.1f} |")
    seconds = sum(r["seconds"] for r in receipts.values())
    image_check = read(out / "original_image_check.json")
    if image_check["policy_lock_sha256"] != complete["policy_lock_sha256"]:
        raise ValueError("Single image replay uses a different lock")
    lines += ["", f"独立原图入口按类别各取验证集清单中的第一张进行逐张复算，{image_check['matched_decisions']}/{len(image_check['samples'])} 张决定与批量结果一致，最大类别分数差为 {image_check['max_abs_score_error']:.6f}。",
              "", f"{len(receipts)} 个完整清单的双视图实测累计用时为 {seconds / 60:.1f} 分钟。开发集在本地提取，留出集在云端以同一原始权重与原图处理规则复测。",
              f"权重位于 `{views.WEIGHTS[task][0].relative_to(views.PROJECT).as_posix()}`。权重 SHA-256 为 `{lock['checkpoint_sha256']}`。"
              f" 决策锁 SHA-256 为 `{complete['policy_lock_sha256']}`。",
              "", "验证集每个候选的分数和决定见 `validation_candidate_predictions.csv`；各留出的逐图配对预测、混淆矩阵、差值区间见同目录的 `*_paired_predictions.csv` 和 `*_report.json`。", "",
              "使用本项目训练环境，从项目根目录运行以下命令可复核缓存、训练记录及归档结果。已锁定的选择会沿用同一份收据。", "",
              "```powershell", f"python scripts/research_classifier_views.py {task} development",
              f"python scripts/optimize_existing_classifiers.py {task} select", f"python scripts/research_classifier_views.py {task} heldout",
              f"python scripts/optimize_existing_classifiers.py {task} evaluate", f"python scripts/optimize_existing_classifiers.py {task} verify-images",
              f"python scripts/summarize_existing_classifier_round.py {task}", "```",
              "", "单张原图的离线入口使用相同权重、几何处理与验证集决策锁。", "",
              "```powershell", f'python scripts/optimize_existing_classifiers.py {task} predict --image "原图路径"', "```", ""]
    (out / "实验结果.md").write_text("\n".join(lines), encoding="utf-8")
    script_sources = ["scripts/research_classifier_views.py", "scripts/optimize_existing_classifiers.py",
                      "scripts/summarize_existing_classifier_round.py", "scripts/evaluate_eyeq_operating_point.py",
                      "scripts/manifest_dataset.py", "training/classification.py", "training/fundus_preprocessing.py", "training/metrics.py"]
    views.write_json(out / "artifact_manifest.json", {
        "task": task, "use": "offline_research_only", "versions": {
            "python": sys.version, **{pkg: version(pkg) for pkg in ("torch","torchvision","numpy","timm","Pillow")}},
        "source_hashes": {s: views.sha(views.PROJECT / s) for s in script_sources},
        "files": {f.name: views.sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name != "artifact_manifest.json" and f.suffix != ".log"}})
    print(out / "实验结果.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=views.DIRS)
    build(parser.parse_args().task)
