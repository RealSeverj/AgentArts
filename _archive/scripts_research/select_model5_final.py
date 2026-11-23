"""Extend the archived four-way validation selection with optional completed geometry controls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import screening_ensemble as ensemble
from scripts.compare_model5_regression import read_lock, require
from scripts.screening_evaluation import load_predictions, read_json, write_json
from scripts.select_model5_candidate import BASELINE, LABELS, VALIDATION
from training import fundus_preprocessing as fundus
from training.screening_letterbox import PLAN_SHA, SETTINGS


RELEASE = "c985927d5c9d106eab9c4617715b95752d93b2dea44be05ff2800aa8bc6ddb8c"
TRAIN = "47700b7e2667bdfcde1ade59620016c198b040b15ce9462a87f063a6610ab783"
VIT_B = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
VIT_L = "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"
C = "global_vitb_letterbox"
D = "global_vitl_letterbox"
E = "global_vitb_letterbox_518"
F = "global_vitl_letterbox_dr_transfer"
DR_CHECKPOINT = "a85acf367a34361986739e93030bc5c51a390ddd33c63d49da883abf1562d4ab"
DR_OVERLAY = "6bcbc3f8dbb5e3b383a6dadc2cb733022273d3386015699b8c239ebe0124bbf0"
DR_PARENT = "80c13752f5655290b2349ebc1dce951c83ef7aa7aa27d17572d09fbd0078fe05"
MODE = "global_fundus_crop_letterbox_v1"
GEOMETRY = "dc9eba4e907231ecfe2319a7ea6d1658b6e10b9d2142dde667b0eba6e7199459"
GEOMETRY_518 = "681f4af47fabd66367e60bd70713435f4ce421269432fe76fe4139d98539aefb"
REQUIRED_FILES = ("complete.json", "run.json", "val_predictions.meta.json", "val_predictions.jsonl", "thresholds.json")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_scores(run, scores, meta, rows):
    lock = read_lock(run)
    operating = {}
    for i, label in enumerate(LABELS):
        report = scores["per_label"][label]
        threshold = lock["per_label"][label]
        require(threshold["threshold"] == report["threshold"] and threshold["status"] == report["goal_status"]
                and all(threshold[k] == v for k, v in ensemble.GOAL.items()),
                f"Working point differs from the frozen validation policy: {run}/{label}")
        y, p, _ = ensemble.known_arrays(rows, i)
        calls = p >= threshold["threshold"]
        positive = int(y.sum())
        negative = len(y) - positive
        tp = int((calls & (y == 1)).sum())
        fp = int((calls & (y == 0)).sum())
        recall = tp / positive
        fpr = fp / negative
        met = recall >= ensemble.GOAL["min_recall"] and fpr <= ensemble.GOAL["max_fpr"]
        require((positive,negative,tp,fp) == (threshold["validation_positive"],threshold["validation_negative"],
                threshold["validation_tp"],threshold["validation_fp"]) and (threshold["status"] == "met") == met,
                f"Working-point counts differ: {run}/{label}")
        operating[label] = dict(ap=report["ap"], threshold=threshold["threshold"], positive=positive, negative=negative,
                                tp=tp, fn=positive-tp, fp=fp, tn=negative-fp, recall=recall, fpr=fpr, met=met)
    require(lock["checkpoint_sha256"] == meta["checkpoint_sha256"], "Wrong threshold checkpoint")
    return dict(macro_ap=scores["macro_ap"], checkpoint_sha256=lock["checkpoint_sha256"],
                validation_working_point=dict(min_recall=ensemble.GOAL["min_recall"], max_fpr=ensemble.GOAL["max_fpr"],
                    labels_meeting_both=sum(r["met"] for r in operating.values()), labels_evaluated=len(operating),
                    per_label=operating),
                val_predictions_sha256=lock["val_predictions_sha256"], thresholds_sha256=file_sha(run/"thresholds.json"))


def check_training(name, run, release_sha, val_meta, geometry_sha=None):
    spec = read_json(run / "run.json")
    complete = read_json(run / "complete.json")
    ident = spec["identity"]
    require(complete["epochs"] == 30 and complete["checkpoint_sha256"] == val_meta["checkpoint_sha256"]
            and 1 <= complete["best_epoch"] == val_meta["epoch"] <= 30, f"Incomplete or changed checkpoint: {name}")
    require(ident["task"] == "grouped" and spec["labels"] == LABELS and ident["release_sha256"] == release_sha,
            f"Wrong training release: {name}")
    settings = ident["settings"]
    require(all(settings[k] == v for k,v in SETTINGS.items() if k not in ("batch_size","accum"))
            and settings["batch_size"] * settings["accum"] == 32, f"Optimization schedule differs: {name}")
    expected_weights = VIT_L if name in ("global_vitl", D, F) else VIT_B
    require(ident["pretrained_sha256"] == expected_weights, f"Wrong initialization: {name}")
    if name == "baseline":
        require(complete["checkpoint_sha256"] == BASELINE and ident["plan_sha256"] == PLAN_SHA,
                "Unexpected historical baseline")
    else:
        kind = "large" if name in ("global_vitl", D, F) else "base"
        mode = {"global_vitl":"global", "dual_vitb":"global_disc_roi", C:MODE, D:MODE, E:MODE, F:MODE}[name]
        image_size = 518 if name == E else 392
        require(ident["input_mode"] == mode and ident["backbone"] == f"vit_{kind}_patch14_dinov2.lvd142m"
                and ident["image_size"] == image_size, f"Candidate identity differs: {name}")
    if name in (C,D,E,F):
        recipe = ident["geometry_recipe_sha256"]
        prep = spec["transform"]
        require(ident["locator_sha256"] is None and val_meta["input_mode"] == MODE
                and recipe == val_meta["geometry_recipe_sha256"] == geometry_sha
                and prep == ident["transform"] == val_meta["transform"]
                and prep["preprocessing"] == fundus.preprocessing_spec({"preprocessing":{"mode":fundus.MODE}})
                and prep["augmentation"]["resize_before_augmentation"] is True
                and prep["model"]["img_size"] == image_size and prep["model"]["name"] == ident["backbone"],
                f"Candidate geometry or validation metadata differs: {name}")
        if name == C:
            require(ident["plan_sha256"] == PLAN_SHA and ident["controlled_settings"] == SETTINGS
                    and all(settings[k] == v for k,v in SETTINGS.items()), "Candidate C controls differ")
        elif name == D:
            require(ident["candidate"] == val_meta["candidate"] == "vitl_global_letterbox",
                    "Candidate D identity differs")
        elif name == E:
            controls = ident["controlled_settings"]
            require(ident["candidate"] == val_meta["candidate"] == "vitb_global_letterbox_518"
                    and ident["plan_sha256"] == PLAN_SHA
                    and all(controls[k] == settings[k] for k in SETTINGS)
                    and settings["batch_size"] in (4,8), "Candidate E identity or settings differ")
        else:
            require(ident["candidate"] == val_meta["candidate"] == "vitl_global_letterbox_dr_transfer",
                    "Candidate F identity differs")
            expected_transfer = dict(source_checkpoint_sha256=DR_CHECKPOINT, parent_checkpoint_sha256=DR_PARENT, delta_sha256=DR_OVERLAY,
                    base_sha256=VIT_L, source_epoch=24, source_geometry=336, blocks=[20,21,22,23],
                    final_norm=True, tensor_count=58, include_pos_embed=False, include_source_head=False,
                    supervised_source_history=["APTOS2019","DDR"], semantics="absolute fp32 tensor replacements")
            transfer = ident["transfer"]
            require(transfer == val_meta["transfer"] and all(transfer.get(k) == v for k,v in expected_transfer.items()),
                    "Candidate F supervised initialization differs")
    return complete


def check_four(args):
    # Rebuild the four candidates' validation scores and the ensemble recipe in memory.
    raw = (args.ensemble_run / "selection.json").read_bytes()
    receipt = json.loads(raw)
    predictions, prov, reference, manifest, labels = ensemble.read_split(args, "val", ensemble.MODELS)
    require(manifest == VALIDATION and labels == LABELS and prov == receipt["validation_predictions"]
            and receipt["validation_manifest_sha256"] == VALIDATION and receipt["selection_split"] == "val",
            "Four-way validation inputs differ from the receipt")
    contract = ensemble.encoded(ensemble.recipe(prov, labels)).encode()
    recipe_hash = hashlib.sha256(contract).hexdigest()
    require((args.ensemble_run / "ensemble_recipe.json").read_bytes() == contract
            and recipe_hash == receipt["ensemble_recipe_sha256"], "Ensemble recipe differs")
    predictions[ensemble.BLEND] = ensemble.fuse(predictions, reference, recipe_hash)
    stored, blend_meta, _ = load_predictions(args.ensemble_run, "val")
    require(stored == predictions[ensemble.BLEND] and blend_meta["checkpoint_sha256"] == recipe_hash,
            "Archived ensemble predictions differ from the fixed recipe")
    order = [*ensemble.MODELS, ensemble.BLEND]
    require(receipt["tie_priority"] == order and receipt["selection_metric"] ==
            "11-label macro noninterpolated average precision", "Unrecognized four-way selection policy")
    paths = [args.baseline, args.global_run, args.dual_run, args.ensemble_run]
    results = {}
    for name, path in zip(order, paths):
        scores = ensemble.validation_summary(predictions[name], labels)
        meta = read_json(path / "val_predictions.meta.json")
        results[name] = frozen_scores(path, scores, meta, predictions[name])
        require(math.isclose(scores["macro_ap"], receipt["validation"][name]["macro_ap"], abs_tol=1e-12)
                and all(math.isclose(scores["per_label"][k]["ap"],receipt["validation"][name]["per_label"][k]["ap"],
                                     abs_tol=1e-12) for k in LABELS), f"Published validation AP differs: {name}")
        if name != ensemble.BLEND:
            check_training(name, path, RELEASE, meta)
    winner = max(order, key=lambda n:results[n]["macro_ap"])
    require(winner == receipt["winner"] and (winner == ensemble.BLEND) == receipt["ensemble_selected"]
            and results[ensemble.BLEND]["thresholds_sha256"] == receipt["ensemble_threshold_lock_sha256"],
            "Four-way result or threshold lock differs")
    return results, reference, winner, hashlib.sha256(raw).hexdigest()


def geometry_candidate(name, run, reference, results, geometry_sha):
    if run is None:
        return {"status": "not_supplied"}
    missing = [n for n in REQUIRED_FILES if not (run/n).is_file()]
    if missing:
        return {"status":"incomplete","missing":missing}
    rows, meta, _ = load_predictions(run, "val")
    complete = check_training(name, run, RELEASE, meta, geometry_sha)
    by_id = {r["sample_id"]:r for r in rows}
    require(len(by_id) == len(reference), f"Incomplete validation set: {name}")
    for src in reference:
        row = by_id[src["sample_id"]]
        require(all(row[k] == src[k] for k in ensemble.FIELDS)
                and row.get("image_sha256",src["image_sha256"]) == src["image_sha256"],
                f"Validation eye, label or mask differs: {name}")
    scores = ensemble.validation_summary(rows, LABELS)
    summary = frozen_scores(run, scores, meta, rows)
    require(summary["checkpoint_sha256"] not in {r["checkpoint_sha256"] for r in results.values()}
            and math.isclose(scores["macro_ap"], complete["best_validation_macro_ap"], abs_tol=1e-6),
            f"Checkpoint identity or best validation AP differs: {name}")
    results[name] = summary
    receipt = {"status":"eligible","geometry_recipe_sha256":geometry_sha}
    if name == F:
        receipt["transfer"] = meta["transfer"]
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline","global-run","dual-run","ensemble-run","release","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--candidate-c",type=Path,help="Omit if unavailable at the decision deadline")
    p.add_argument("--candidate-d",type=Path,help="Optional ViT-L global geometry control")
    p.add_argument("--candidate-e",type=Path,help="Optional ViT-B 518px global geometry control")
    p.add_argument("--candidate-f",type=Path,help="Optional ViT-L 392px control initialized from DR blocks 20-23 and final norm")
    args = p.parse_args()
    require(file_sha(args.release/"audit.json") == RELEASE and file_sha(args.release/"grouped/train.jsonl") == TRAIN,
            "Expected the archived, unchanged training release")
    results, reference, prior, receipt_hash = check_four(args)
    geometry = {}
    candidates = [(C,args.candidate_c,GEOMETRY),(D,args.candidate_d,GEOMETRY),(E,args.candidate_e,GEOMETRY_518)]
    if args.candidate_f is not None:
        candidates.append((F,args.candidate_f,GEOMETRY))
    for name, path, policy in candidates:
        geometry[name] = geometry_candidate(name, path, reference, results, policy)
    winner = max(results, key=lambda n:results[n]["macro_ap"])
    doc = dict(format="model5_final_validation_selection_v1", selection_split="val", selected=winner,
               selection_metric="validation_11_label_macro_ap", tie_priority=list(results), results=results,
               validation_manifest_sha256=VALIDATION, training_manifest_sha256=TRAIN, release_sha256=RELEASE,
               prior_winner=prior, four_way_selection_sha256=receipt_hash, geometry_candidates=geometry)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output, doc, lock=True)
    print(json.dumps({"selected":winner,"geometry_candidates":geometry,
                      "validation_macro_ap":{n:r["macro_ap"] for n,r in results.items()},
                      "validation_labels_meeting_both":{n:r["validation_working_point"]["labels_meeting_both"]
                                                        for n,r in results.items()}},indent=2))


if __name__ == "__main__":
    main()
