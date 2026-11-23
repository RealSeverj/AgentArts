"""Model 5 whole-image ViT-B geometry control; run as python -m training.screening_letterbox."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from training import fundus_preprocessing as fundus
from training import screening_letterbox_cache as geometry_cache


PLAN_SHA = "bc3a23642dad050cbf9e81a092c88a029ae6c4055b057fca858d74ffed939cbd"
SETTINGS = dict(variant="partial", blocks=4, head_only_epochs=5, epochs=30, precision="fp16",
                seed=20261114, head_lr=0.001, backbone_lr=0.00001, batch_size=16, accum=2)


def transform_config(baseline):
    config = deepcopy(baseline.TRANSFORM_CONFIG)
    config["kind"] = "multilabel"
    config["preprocessing"] = fundus.preprocessing_spec({"preprocessing":{"mode":fundus.MODE}})
    config["augmentation"]["resize_before_augmentation"] = True
    fundus.validate_preprocessing_config(config)
    return config


@contextmanager
def use_transform(baseline, config, dataset_type=None):
    # The historical single-process entry point constructs both its Dataset and
    # run.json from this configuration, before workers start. Scope the binding
    # to this separate CLI, retaining the original training/evaluation loops.
    original = baseline.TRANSFORM_CONFIG
    original_dataset = baseline.ScreeningDataset
    baseline.TRANSFORM_CONFIG = config
    if dataset_type is not None:
        baseline.ScreeningDataset = dataset_type
    try:
        yield
    finally:
        baseline.TRANSFORM_CONFIG = original
        baseline.ScreeningDataset = original_dataset


def fixed_run(args):
    import torch
    from training import classification
    from training import screening_training as baseline
    from training import screening_letterbox_dataset as cache_dataset

    if not torch.cuda.is_available():
        raise ValueError("This full-data, fp16 comparison requires CUDA")
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    plan_sha = baseline.sha256(args.config)
    if plan_sha != PLAN_SHA or baseline.sha256(args.weights) != baseline.OFFICIAL_SHA:
        raise ValueError("Plan or initial ViT-B weights differ from the archived Model 5 baseline")
    audit_path = args.release / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    if (audit.get("status") != "verified" or audit["label_order"]["grouped"] != baseline.LABELS["grouped"]
            or any(audit["manifests"][f"grouped/{s}"]["records"] != n for s,n in
                   (("train",1878),("val",634),("test",615),("external_test",848)))):
        raise ValueError("Expected the complete annotated Model 5 release")
    for key,value in SETTINGS.items():
        setattr(args,key,value)
    args.device = "cuda"
    config = transform_config(baseline)
    binding = geometry_cache.recipe_sha()
    identity = dict(task="grouped", input_mode="global_fundus_crop_letterbox_v1", image_size=392,
                    backbone=baseline.BACKBONE, locator_sha256=None, plan_sha256=plan_sha,
                    pretrained_sha256=baseline.OFFICIAL_SHA, release_sha256=baseline.sha256(audit_path),
                    transform=config, controlled_settings=SETTINGS,geometry_recipe_sha256=binding,
                    code_sha256={p.name:baseline.sha256(p) for p in map(Path, [
                        __file__, baseline.__file__, classification.__file__, fundus.__file__,
                        geometry_cache.__file__, cache_dataset.__file__])})
    args.output.mkdir(parents=True,exist_ok=True)
    if args.action == "predict" and args.split != "val":
        locked = json.loads((args.output/"thresholds.json").read_text(encoding="utf-8-sig"))
        if (locked["checkpoint_sha256"] != baseline.sha256(args.output/"best.pt")
                or locked["labels"] != baseline.LABELS["grouped"]):
            raise ValueError("Held-out prediction requires the matching validation threshold lock")
    baseline.seed_all(args.seed)
    cached = cache_dataset.CachedGeometryDataset if args.letterbox_cache_dir is not None else None
    with use_transform(baseline, config, cached):
        model = baseline.ScreeningModel("grouped",args.weights).to(args.device)
        if args.action == "train":
            baseline.train(args,audit,"grouped",model,identity)
            split = "val"
        else:
            split = args.split
            baseline.export(model,args,audit,"grouped",split,args.output/"best.pt",identity)
        meta_path = args.output / f"{split}_predictions.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        meta.update(transform=config, input_mode=identity["input_mode"],geometry_recipe_sha256=binding,
                    geometry_cache_index_sha256=(baseline.sha256(args.letterbox_cache_dir/(split+".json"))
                                                 if cached else None))
        baseline.write_json(meta_path,meta)


def edge_dark_fraction(image, threshold):
    mask = np.asarray(image).max(axis=2) <= threshold
    h,w=mask.shape
    total=0
    for c in fundus._components(mask):
        l,t,r,b=c["bbox"]
        if l==0 or t==0 or r==w or b==h:
            total+=c["area"]
    return total/mask.size


def measure(args):
    spec = fundus.preprocessing_spec({"preprocessing":{"mode":fundus.MODE}})
    src = args.release/"grouped/train.jsonl"
    raw = src.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    release = json.loads((args.release/"audit.json").read_text(encoding="utf-8-sig"))
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows)!=1878 or release["manifests"]["grouped/train"]["sha256"]!=digest:
        raise ValueError("The geometry audit must use the complete audited RFMiD training manifest")
    if not 1 <= args.sample_size <= len(rows):
        raise ValueError("Requested sample size exceeds the cohort")
    selected = np.rint(np.linspace(0,len(rows)-1,args.sample_size)).astype(int)
    items = []
    data_root = args.data_root.resolve()
    for i in selected:
        row = rows[i]
        image_path = (data_root/row["image_path"]).resolve()
        if not image_path.is_relative_to(data_root) or row["dataset"]!="RFMiD" or row["split"]!="train":
            raise ValueError("Unexpected image path or source for the geometry audit")
        source_bytes = image_path.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest()!=row["image_sha256"]:
            raise ValueError(f"Training image digest mismatch: {row['sample_id']}")
        with Image.open(io.BytesIO(source_bytes)) as src_image:
            image=src_image.convert("RGB")
        width,height=image.size
        before=image.resize((392,392),Image.Resampling.BILINEAR)
        after,meta=fundus.apply_preprocessing(image,392,spec,return_metadata=True)
        l,t,r,b=meta["crop_bbox"]
        rw,rh=meta["resized_size"]
        items.append(dict(sample_id=row["sample_id"],width=width,height=height,native_aspect_ratio=width/height,
                          stretch_axis_scale_ratio=max(width,height)/min(width,height),
                          source_area_cropped=1-(r-l)*(b-t)/(width*height),cropped=meta["cropped"],
                          fallback=meta["fallback"], letterbox_padding_fraction=1-rw*rh/(392*392),
                          stretch_edge_dark_fraction=edge_dark_fraction(before,spec["black_threshold"]),
                          letterbox_edge_dark_fraction=edge_dark_fraction(after,spec["black_threshold"])))
    keys=("native_aspect_ratio","stretch_axis_scale_ratio","source_area_cropped","letterbox_padding_fraction",
          "stretch_edge_dark_fraction","letterbox_edge_dark_fraction")
    summary={}
    for k in keys:
        v=[r[k] for r in items]
        summary[k]=dict(mean=float(np.mean(v)),median=float(np.median(v)),min=min(v),max=max(v),
                        p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)))
    result=dict(training_manifest_sha256=digest,total_train_records=len(rows),sampled=len(items),
                sampling="evenly spaced rounded indices in the existing train manifest, endpoints included",
                edge_dark_definition="fraction of output pixels whose maximum RGB channel is <=8 and are "
                                     "8-connected to the border, including any newly added black padding",
                pixel_space=[392,392],preprocessing=spec,cropped=sum(r["cropped"] for r in items),
                fallbacks=sum(r["fallback"] for r in items),nonsquare=sum(r["width"]!=r["height"] for r in items),
                summary=summary,rows=items)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    actions=cli.add_subparsers(dest="action",required=True)
    for action in ("train","predict","geometry"):
        p=actions.add_parser(action)
        p.add_argument("--release",type=Path,default=Path("data/prepared/screening_v1"))
        p.add_argument("--data-root",type=Path,default=Path("."))
        if action=="geometry":
            p.add_argument("--sample-size",type=int,default=128)
            p.add_argument("--output",type=Path,required=True)
            continue
        p.add_argument("--weights",type=Path,required=True)
        p.add_argument("--config",type=Path,default=Path("configs/grouped.plan.json"))
        p.add_argument("--output",type=Path,default=Path("runs/grouped_vitb_letterbox"))
        storage = p.add_mutually_exclusive_group()
        storage.add_argument("--rgb-cache-dir",type=Path)
        storage.add_argument("--letterbox-cache-dir",type=Path,help="Verified 392px deterministic RGB geometry cache")
        p.add_argument("--workers",type=int,default=9)
        if action=="train":
            p.add_argument("--resume",action="store_true")
        else:
            p.add_argument("--split",choices=["val","test","external_test"],required=True)
    args=cli.parse_args()
    if args.action=="geometry":
        measure(args)
    else:
        fixed_run(args)


if __name__=="__main__":
    main()
