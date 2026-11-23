"""Plot audited held-out DR confusion matrices, with fixed row normalization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    for row, (split, title) in enumerate((("test", "Internal test"), ("external_test", "External IDRiD"))):
        for col, (name, label) in enumerate((("baseline", "Head-only baseline"), ("finetuned", "Full fine-tuning"))):
            result = json.loads((args.report_dir / name / f"evaluation_{split}.json").read_text(encoding="utf-8"))["metrics"]
            counts = np.asarray(result["confusion_matrix"], dtype=np.float64)
            supports = counts.sum(axis=1, keepdims=True)
            percentages = np.divide(counts * 100, supports, out=np.zeros_like(counts), where=supports > 0)
            ax = axes[row, col]
            panel = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
            ax.set_title(f"{title} | {label}\nN={result['n']:,} | QWK={result['qwk']:.4f} | Accuracy={result['accuracy']:.1%}", fontsize=11, pad=12)
            ax.set_xticks(range(5), labels=range(5))
            ax.set_yticks(range(5), labels=range(5))
            ax.set_xlabel("Predicted grade")
            ax.set_ylabel("True grade")
            ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, f"{int(counts[i, j]):,}\n{percentages[i, j]:.1f}%", ha="center", va="center", fontsize=9,
                            color="white" if percentages[i, j] > 55 else "#182433")
    fig.suptitle("DR grading: complete held-out evaluation", fontsize=16)
    fig.colorbar(panel, ax=axes.ravel().tolist(), shrink=.8, pad=.02, label="Percentage within each true grade")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(json.dumps({"png": str(args.output), "svg": str(args.output.with_suffix('.svg'))}))


if __name__ == "__main__":
    main()
