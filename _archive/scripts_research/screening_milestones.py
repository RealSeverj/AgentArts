"""Print validation-only milestones for the Model 5 candidates from their local history.json files."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


DISC = ("OPTIC_DISC_CUPPING", "ODE", "ODP")


def summarize(path, total=30):
    if path.is_dir() or path.suffix.lower() != ".json":
        path = path / "history.json"
    name = path.parent.name if path.name == "history.json" else path.stem
    if not path.exists():
        return [f"{name}　尚无本地历史文件。"]
    history = json.loads(path.read_text(encoding="utf-8-sig"))
    if not history:
        return [f"{name}　尚无已完成轮次。"]
    if len(history) > total or [r["epoch"] for r in history] != list(range(1,len(history)+1)):
        raise ValueError(f"{path} 的 epoch 记录不连续或超出目标轮数")
    lines = [f"{name}　已完成 {len(history)}/{total} 轮"]
    milestones = {10, 20, 30} & set(range(1, total+1))
    for idx, row in enumerate(history):
        e = row["epoch"]
        if e not in milestones and e != len(history):
            continue
        v = row["val"]
        scores = [v["macro_ap"], *(v["per_label"][k]["auprc"] for k in DISC)]
        if not all(isinstance(s, (int,float)) and math.isfinite(s) and 0 <= s <= 1 for s in scores):
            raise ValueError(f"{path} 第{e}轮验证 AP 异常")
        records = history[:idx+1]
        recent = [r["elapsed_seconds"] for r in records if r["stage"] == row["stage"]][-5:]
        if any(not math.isfinite(t) or t <= 0 for t in recent):
            raise ValueError(f"{path} 第{e}轮训练耗时异常")
        pace = statistics.median(recent)
        best = history[row["best_epoch"]-1]
        lines.append(f"第{e}轮　宏AP {scores[0]:.4f}，视盘凹陷 {scores[1]:.4f}，"
                     f"视盘水肿 {scores[2]:.4f}，视盘苍白 {scores[3]:.4f}。"
                     f"截至当轮最佳宏AP {best['val']['macro_ap']:.4f}（第{best['epoch']}轮）。"
                     f"同阶段最近{len(recent)}轮中位 {pace:.1f} 秒/轮，"
                     f"该轮结束时预计还需 {(total-e)*pace/60:.1f} 分钟。")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path, nargs="+", help="一个或多个候选运行目录或历史JSON")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    for path in args.candidates:
        print("\n".join(summarize(path,args.epochs)))
    print("耗时含训练和验证。剩余时长按当前阶段速度外推，另需预留检查点写盘、评估及归档时间。")


if __name__ == "__main__":
    main()
