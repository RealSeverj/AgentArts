"""Select Model 5 using validation AP; score the fixed probability ensemble on held-out splits only if selected."""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.screening_evaluation import (choose_threshold, known_arrays, load_predictions, lock_thresholds,
                                          metrics, read_json, report, write_file, write_json)

MODELS = ("baseline", "global_vitl", "dual_vitb")
BLEND = "mean_probability"
GOAL = dict(min_recall=0.9, max_fpr=0.1)
FIELDS = ("sample_id", "split", "dataset", "patient_id", "content_group_id", "image_path", "labels", "mask")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(doc):
    return json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def read_split(args, split, names):
    runs = dict(zip(MODELS, [args.baseline, args.global_run, args.dual_run]))
    expected = read_json(args.release / "audit.json")
    spec = expected["manifests"][f"grouped/{split}"]
    raw = (args.release / "grouped" / f"{split}.jsonl").read_bytes()
    manifest_hash = sha(raw)
    if manifest_hash != spec["sha256"]:
        raise ValueError(f"Unverified {split} manifest")
    reference = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ids = [(r["dataset"], r["sample_id"]) for r in reference]
    labels = expected["label_order"]["grouped"]
    required_count = {"val":634, "test":615, "external_test":848}[split]
    if (len(reference) != spec["records"] or len(set(ids)) != required_count or len(reference) != required_count
            or len(labels) != 11 or len(set(labels)) != 11):
        raise ValueError("Expected the full audited Model 5 cohort and 11 labels")
    predictions, provenance = {}, {}
    for name in names:
        rows, meta, digest = load_predictions(runs[name], split)
        if meta["manifest_sha256"] != manifest_hash or meta["labels"] != labels:
            raise ValueError(f"{name}/{split}: manifest digest or label order differs")
        locator = meta.get("locator_sha256")
        roi_hash = meta.get("roi_manifest_sha256")
        expected_locator = "bb25840047eedca5b115c3a81c9f7b8e3e73a8a0729cc6c67b9f55b1c38ef5e7"
        if ((name == "dual_vitb" and (locator != expected_locator or not roi_hash))
                or (name != "dual_vitb" and (locator is not None or roi_hash is not None))):
            raise ValueError(f"{name}/{split}: unexpected optic-disc view configuration")
        index = {(r["dataset"],r["sample_id"]):r for r in rows}
        if set(index) != set(ids):
            raise ValueError(f"{name}/{split}: sample IDs differ from the source manifest")
        aligned = [index[k] for k in ids]
        for pred, src in zip(aligned,reference):
            if (any(pred[field] != src[field] for field in FIELDS)
                    or pred.get("image_sha256",src["image_sha256"]) != src["image_sha256"]):
                raise ValueError(f"{name}/{split}: source, target, mask or image identity differs: {src['sample_id']}")
        predictions[name] = aligned
        provenance[name] = {k:meta.get(k) for k in ("checkpoint_sha256", "locator_sha256", "roi_manifest_sha256")}
        provenance[name]["predictions_sha256"] = digest
    if len(set(p["checkpoint_sha256"] for p in provenance.values())) != len(names):
        raise ValueError("Expected distinct candidate checkpoints")
    return predictions, provenance, reference, manifest_hash, labels


def recipe(provenance, labels):
    return dict(version=1, operation="mean of two post-sigmoid probability vectors in float64",
                labels=labels, members=[dict(role=n, weight=0.5,
                                            checkpoint_sha256=provenance[n]["checkpoint_sha256"],
                                            locator_sha256=provenance[n]["locator_sha256"])
                                       for n in MODELS[1:]],
                full_image_member="global_vitl", full_and_optic_disc_member="dual_vitb",
                scores="per-label probabilities; finite inverse-logit exported for compatibility only")


def fuse(predictions, reference, recipe_hash):
    result = []
    for left, right, src in zip(predictions["global_vitl"], predictions["dual_vitb"], reference):
        p = (np.asarray(left["probabilities"],dtype=np.float64) + np.asarray(right["probabilities"],dtype=np.float64))/2
        finite_input = np.clip(p, np.nextafter(0.,1.), np.nextafter(1.,0.))
        logit = np.log(finite_input) - np.log1p(-finite_input)
        row = {k:src[k] for k in FIELDS}
        row.update(checkpoint_sha256=recipe_hash, image_sha256=src["image_sha256"], probabilities=p.tolist(),
                   logits=logit.tolist())
        result.append(row)
    return result


def write_predictions(out, split, rows, recipe_hash, labels, manifest_hash, provenance):
    data = "".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in rows)
    write_file(out/f"{split}_predictions.jsonl",data,lock=True)
    write_json(out/f"{split}_predictions.meta.json",
               dict(split=split, labels=labels, checkpoint_sha256=recipe_hash,
                    checkpoint_identity="file SHA-256 of ensemble_recipe.json",
                    manifest_sha256=manifest_hash, members={n:provenance[n] for n in MODELS[1:]}), lock=True)


def validation_summary(rows, labels):
    values = {}
    for i,label in enumerate(labels):
        y,p,_=known_arrays(rows,i)
        goal = choose_threshold(y,p,GOAL)
        point = metrics(y,p,goal["threshold"])
        if point["ap"] is None:
            raise ValueError(f"Validation lacks supervision for {label}")
        values[label] = dict(ap=point["ap"], recall=point["sensitivity"], fp=point["fp"],
                             threshold=goal["threshold"], goal_status=goal["status"])
    return dict(macro_ap=float(np.mean([v["ap"] for v in values.values()])),per_label=values)


def select(args):
    predictions, prov, src, source_hash, labels = read_split(args, "val", MODELS)
    contract = encoded(recipe(prov,labels))
    identity = sha(contract.encode("utf-8"))
    args.output.mkdir(parents=True,exist_ok=True)
    write_file(args.output/"ensemble_recipe.json",contract,lock=True)
    predictions[BLEND] = fuse(predictions,src,identity)
    summaries = {name:validation_summary(rows,labels) for name,rows in predictions.items()}
    # The insertion order implements the predeclared tie priority.
    winner = max((*MODELS,BLEND),key=lambda k:summaries[k]["macro_ap"])
    write_predictions(args.output,"val",predictions[BLEND],identity,labels,source_hash,prov)
    lock_thresholds(Namespace(run=args.output,policy=None,**GOAL))
    decision = dict(selection_split="val", selection_metric="11-label macro noninterpolated average precision",
                    tie_priority=[*MODELS,BLEND], winner=winner,
                    ensemble_selected=(winner==BLEND), validation_manifest_sha256=source_hash,
                    validation_predictions=prov, ensemble_recipe_sha256=identity,
                    ensemble_threshold_lock_sha256=sha((args.output/"thresholds.json").read_bytes()),
                    validation=summaries)
    write_json(args.output/"selection.json",decision,lock=True)
    print(json.dumps({"winner":winner,"val_macro_ap":{k:v["macro_ap"] for k,v in summaries.items()}},indent=2))


def evaluate(args):
    receipt = read_json(args.output/"selection.json")
    if not receipt["ensemble_selected"] or receipt["winner"] != BLEND:
        print(f"Selected {receipt['winner']} on validation; held-out ensemble evaluation skipped.")
        return
    # Recheck selection using validation only, including the frozen input hashes.
    select(args)
    identity = receipt["ensemble_recipe_sha256"]
    if sha((args.output/"thresholds.json").read_bytes()) != receipt["ensemble_threshold_lock_sha256"]:
        raise ValueError("Threshold lock changed after selection")
    for split in ("test","external_test"):
        members,prov,src,source_hash,labels=read_split(args,split,MODELS[1:])
        contract = encoded(recipe(prov,labels))
        if sha(contract.encode("utf-8")) != identity:
            raise ValueError(f"{split}: one or both ensemble members differ from validation")
        rows = fuse(members,src,identity)
        write_predictions(args.output,split,rows,identity,labels,source_hash,prov)
    report(Namespace(run=args.output,split=("val","test","external_test"),bootstrap=args.bootstrap,seed=20261114))


def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=["select","evaluate"])
    cli.add_argument("--baseline",type=Path,required=True)
    cli.add_argument("--global-run",type=Path,required=True)
    cli.add_argument("--dual-run",type=Path,required=True)
    cli.add_argument("--release",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--bootstrap",type=int,default=1000)
    args=cli.parse_args()
    if args.output.resolve() in [p.resolve() for p in (args.baseline,args.global_run,args.dual_run)]:
        cli.error("Use a separate ensemble output directory")
    if args.action=="select":
        select(args)
    else:
        evaluate(args)


if __name__=="__main__":
    main()
