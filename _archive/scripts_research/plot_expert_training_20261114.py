"""Render the three completed, source-isolated expert validation histories.

Reads JSONL/configuration files only. No test results, model loading, or training.
Use --reports-root ROOT for ROOT/{quality,multidisease,disc_refuge}/run/history.jsonl,
or supply all three explicit --*-history paths. PNG/SVG outputs must be new files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SEED = 20261113
SPECS = (
    {"key": "quality", "source": "EyeQ", "task": "image_quality", "kind": "multiclass",
     "model": "vit_base_patch14_dinov2.lvd142m", "model_label": "ViT-B/14, 392 px",
     "metric": "macro_f1", "metric_label": "Validation macro-F1", "count": 1875,
     "epochs": 30, "labels": ["quality_grade"], "color": "#2467a8"},
    {"key": "multidisease", "source": "RFMiD", "task": "multidisease", "kind": "multilabel",
     "model": "vit_base_patch14_dinov2.lvd142m", "model_label": "ViT-B/14, 392 px",
     "metric": "macro_auprc", "metric_label": "Validation macro-AUPRC", "count": 634,
     "epochs": 30, "labels": ["DR", "AMD", "MYOPIC_FUNDUS", "OPTIC_DISC_CUPPING", "ABNORMAL"],
     "color": "#7b4ca2"},
    {"key": "disc_refuge", "source": "REFUGE", "task": "disc_cup_segmentation", "kind": "segmentation",
     "model": "unet", "model_label": "U-Net, 512 px",
     "metric": "macro_dice", "metric_label": "Validation macro-Dice", "count": 400,
     "epochs": 50, "labels": ["optic_cup", "optic_disc"], "color": "#147b67"},
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"{name} must be finite")
    return float(value)


def reject_nonfinite(value):
    raise ValueError(f"Non-standard JSON numeric constant: {value}")


def json_loads(payload):
    return json.loads(payload, parse_constant=reject_nonfinite)


def read_history(path, spec):
    """Require complete histories and cross-check their recorded selection/stages."""
    path = Path(path).resolve()
    payload = path.read_bytes()
    records = [json_loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
    name = spec["key"]
    require(len(records) == spec["epochs"], f"{name}: expected {spec['epochs']} completed epochs")
    values, running_best = [], None
    for expected_epoch, record in enumerate(records, 1):
        require(isinstance(record, dict) and type(record.get("epoch")) is int
                and record["epoch"] == expected_epoch,
                f"{name}: epochs must be exactly 1..{spec['epochs']} in order")
        expected_stage = ("full" if spec["kind"] == "segmentation" else
                          "head_only" if expected_epoch <= 5 else "last_4_blocks")
        require(record.get("stage") == expected_stage,
                f"{name} epoch {expected_epoch}: expected stage {expected_stage}")
        val = record.get("val")
        require(isinstance(val, dict), f"{name} epoch {expected_epoch}: missing validation metrics")
        count_field = "samples" if spec["kind"] == "segmentation" else "n"
        require(type(val.get(count_field)) is int and val[count_field] == spec["count"],
                f"{name} epoch {expected_epoch}: incomplete/wrong validation count")
        score = finite(val.get(spec["metric"]), f"{name} epoch {expected_epoch} {spec['metric']}")
        require(0.0 <= score <= 1.0, f"{name}: validation metric must lie in [0,1]")
        finite(record.get("train_loss"), f"{name} epoch {expected_epoch} training loss")
        finite(val.get("loss"), f"{name} epoch {expected_epoch} validation loss")
        require(record.get("best_metric_name") == spec["metric"], f"{name}: unexpected selection metric")
        improved = running_best is None or score > running_best
        running_best = score if improved else running_best
        require(type(record.get("improved")) is bool and record["improved"] == improved,
                f"{name} epoch {expected_epoch}: improved flag disagrees with first-maximum selection")
        require(math.isclose(finite(record.get("best_metric"), f"{name} recorded best"),
                             running_best, rel_tol=0, abs_tol=1e-12),
                f"{name} epoch {expected_epoch}: recorded best differs from running maximum")
        if spec["kind"] != "segmentation":
            by_source = val.get("by_dataset", {})
            require(isinstance(by_source, dict) and set(by_source) == {spec["source"]},
                    f"{name}: validation must contain only {spec['source']}")
            source = by_source[spec["source"]]
            require(source.get("n") == spec["count"]
                    and math.isclose(finite(source.get(spec["metric"]), f"{name} source metric"),
                                     score, rel_tol=0, abs_tol=1e-12),
                    f"{name}: source-specific validation metric/count differs")
        else:
            require(val.get("evaluation_space") == "whole_image_resized_to_model_img_size",
                    "REFUGE: expected whole-image resized validation")
            channels = val.get("per_label", {})
            require(set(channels) == set(spec["labels"]), "REFUGE: wrong cup/disc validation channels")
            dice = []
            for label in spec["labels"]:
                require(channels[label].get("known") == spec["count"],
                        f"REFUGE: incomplete {label} validation supervision")
                dice.append(finite(channels[label].get("dice"), f"REFUGE {label} Dice"))
            require(math.isclose(sum(dice) / len(dice), score, rel_tol=0, abs_tol=1e-12),
                    "REFUGE: macro-Dice disagrees with cup/disc metrics")
        values.append(score)

    # Resolved configurations accompany the formal run exports. Requiring them
    # also establishes REFUGE provenance, which structural histories omit.
    config_path = path.parent / "config.resolved.json"
    config_payload = config_path.read_bytes()
    config = json_loads(config_payload.decode("utf-8-sig"))
    require(config.get("seed") == SEED and config.get("task") == spec["task"]
            and config.get("kind") == spec["kind"] and config.get("labels") == spec["labels"],
            f"{name}: resolved task/labels/seed differ from this experiment")
    data, model, training = (config.get(key, {}) for key in ("data", "model", "training"))
    require(data.get("source_isolated") is True and data.get("datasets") == [spec["source"]]
            and data.get("protocol") == "strict_cross_domain",
            f"{name}: resolved configuration must isolate {spec['source']}")
    require(model.get("name") == spec["model"]
            and model.get("img_size") == (512 if spec["kind"] == "segmentation" else 392),
            f"{name}: unexpected resolved architecture or image size")
    require(training.get("epochs") == spec["epochs"] and training.get("best_metric") == spec["metric"]
            and training.get("best_mode") == "max", f"{name}: unexpected epoch/selection configuration")
    if spec["kind"] != "segmentation":
        require(training.get("freeze_backbone_epochs") == 5
                and training.get("unfreeze_last_n_blocks") == 4
                and training.get("full_finetune_epoch") == 30,
                f"{name}: unexpected backbone freeze schedule")
    best_index = values.index(max(values))
    return {"spec": spec, "values": values, "best_epoch": best_index + 1,
            "best_value": values[best_index], "history": str(path),
            "history_sha256": hashlib.sha256(payload).hexdigest(),
            "config": str(config_path), "config_sha256": hashlib.sha256(config_payload).hexdigest()}


def plot_histories(paths, prefix):
    histories = [read_history(paths[spec["key"]], spec) for spec in SPECS]
    prefix = Path(prefix).resolve()
    png, svg = Path(str(prefix) + ".png"), Path(str(prefix) + ".svg")
    protected = {Path(item[key]).resolve() for item in histories for key in ("history", "config")}
    require(not ({png, svg} & protected), "Outputs cannot overwrite input histories/configurations")
    if png.exists() or svg.exists():
        raise FileExistsError("Output exists; choose a new --output-prefix")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10.5,
                         "svg.fonttype": "none", "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 6.3), sharey=False)
    fig.subplots_adjust(left=.065, right=.985, top=.72, bottom=.26, wspace=.31)
    for ax, item in zip(axes, histories):
        spec, values = item["spec"], item["values"]
        epochs = list(range(1, spec["epochs"] + 1))
        ax.plot(epochs, values, color=spec["color"], linewidth=2, marker=".", markersize=4, zorder=3)
        ax.scatter([item["best_epoch"]], [item["best_value"]], color=spec["color"],
                   edgecolors="white", linewidths=1.2, s=85, zorder=5)
        if spec["kind"] != "segmentation":
            ax.axvline(6, color="#64748b", linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
        low, high = min(values), max(values)
        padding = max(.025, (high - low) * .18)
        ax.set_ylim(max(0, low - padding), min(1, high + padding))
        ax.set_xlim(1, spec["epochs"])
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30] if spec["epochs"] == 30 else [1, 10, 20, 30, 40, 50])
        ax.set_xlabel("Completed training epoch")
        ax.set_ylabel(spec["metric_label"])
        ax.set_title(f"{spec['source']} | {spec['model_label']}\n{spec['metric_label']} (N={spec['count']:,})", pad=45)
        ax.text(.0, 1.055, f"Selected: epoch {item['best_epoch']}  |  {item['best_value']:.4f}",
                transform=ax.transAxes, fontsize=10.3, color=spec["color"], fontweight="semibold")
        ax.grid(axis="y", color="#e2e8f0", linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for name in ("left", "bottom"):
            ax.spines[name].set_color("#94a3b8")
        ax.tick_params(colors="#334155", labelsize=9)
    fig.suptitle("Expert training: validation learning curves", y=.965, fontsize=19, fontweight="semibold")
    fig.text(.5, .905, "Independent tasks and metrics; each panel has its own y-axis. Values are not a cross-task ranking.",
             ha="center", color="#475569", fontsize=10.5)
    fig.legend(handles=[Line2D([0], [0], color="#475569", linewidth=2, label="Complete validation history"),
                        Line2D([0], [0], marker="o", linestyle="none", color="#475569", label="First maximum: selected epoch"),
                        Line2D([0], [0], color="#64748b", linestyle="--", label="ViT: last 4 blocks + norm unfrozen at epoch 6")],
               loc="lower center", bbox_to_anchor=(.5, .155), ncol=3, frameon=False, fontsize=9.5)
    fig.text(.5, .123, f"Single seed ({SEED}). Checkpoints selected only by the named validation metric; no test scores are plotted or used to select points.",
             ha="center", color="#475569", fontsize=9)
    fig.text(.5, .09, "REFUGE macro-Dice: mean of cup/disc Dice, each calculated from cohort-wide pixel totals on whole images resized to 512 px.",
             ha="center", color="#475569", fontsize=9)
    fig.text(.5, .055, "These curves do not estimate retraining variability, confidence intervals, or performance on unseen sources.",
             ha="center", color="#64748b", fontsize=8.8)
    provenance = [{key: item[key] for key in ("history", "history_sha256", "config", "config_sha256", "best_epoch", "best_value")}
                  | {"source": item["spec"]["source"], "metric": item["spec"]["metric"]} for item in histories]
    metadata = {"Title": "Source-isolated expert validation learning curves",
                "Description": json.dumps({"seed": SEED, "split": "val", "inputs": provenance}, ensure_ascii=False)}
    try:
        # Exclusive creation makes the no-overwrite rule hold even after the
        # earlier existence check; no result file can be replaced silently.
        with png.open("xb") as handle:
            fig.savefig(handle, format="png", dpi=220, bbox_inches="tight", pad_inches=.2, metadata=metadata)
        with svg.open("xb") as handle:
            fig.savefig(handle, format="svg", bbox_inches="tight", pad_inches=.2, metadata=metadata)
    finally:
        plt.close(fig)
    return {"all_checks_passed": True, "png": str(png), "svg": str(svg), "seed": SEED,
            "selection_split": "val", "inputs": provenance,
            "output_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (png, svg)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path)
    for spec in SPECS:
        parser.add_argument("--" + spec["key"].replace("_", "-") + "-history", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    explicit = {spec["key"]: getattr(args, spec["key"] + "_history") for spec in SPECS}
    if args.reports_root:
        if any(explicit.values()):
            parser.error("Use --reports-root or all three explicit history paths, not both")
        paths = {spec["key"]: args.reports_root / spec["key"] / "run/history.jsonl" for spec in SPECS}
    else:
        if not all(explicit.values()):
            parser.error("Supply --reports-root or all three --quality-history/--multidisease-history/--disc-refuge-history paths")
        paths = explicit
    try:
        print(json.dumps(plot_histories(paths, args.output_prefix), ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Validation plotting failed: {error}\n")


if __name__ == "__main__":
    main()
