"""Validation-selected Model 6 inference optimization with fixed image/structural candidates."""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import screening_evaluation as evaluation
from training import classification, fundus_preprocessing, structural
from training import screening_training as baseline

DRIVE_SHA="378fc1a1e15e4f67e016bacaa49fd1b81e016c973043919e3a0a290edadc1aca"
UNET_SHA="bb25840047eedca5b115c3a81c9f7b8e3e73a8a0729cc6c67b9f55b1c38ef5e7"
ORDER=("historical","flip_tta","structural","hybrid")
POLICY=dict(selection="PAPILA validation noninterpolated AP, highest wins, exact ties in listed order",
            candidates=list(ORDER), known_train=692, min_recall=0.90,max_fpr=0.10,
            flip_tta="arithmetic mean of full image and horizontal flip probabilities, fp32 CPU",
            structural="fixed U-Net at 512, probability >= 0.5, largest predicted disc, "
                       "largest cup in intersecting disc pixels within the selected disc bounding box; "
                       "standardized vertical CDR, horizontal CDR, cup/disc area ratio and localization-missing flag",
            fit="training-only median imputation and scaling, class-balanced logistic loss, "
                "L2 penalty 1.0 (unnormalized sum of weighted losses, intercept unpenalized)",
            hybrid="0.5*flip_tta+0.5*structural", disease_checkpoint_sha256=DRIVE_SHA,
            segmentation_checkpoint_sha256=UNET_SHA, original_glaucoma_train_source=["REFUGE","PAPILA"])


def sha(x):
    return hashlib.sha256(x).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def canonical(x):
    return (json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n").encode()


def source(args,split):
    audit=read(args.release/"audit.json")
    spec=audit["manifests"][f"glaucoma/{split}"]
    data=(args.release/"glaucoma"/f"{split}.jsonl").read_bytes()
    rows=[json.loads(line) for line in data.splitlines() if line.strip()]
    if (audit["label_order"]["glaucoma"]!=["glaucoma"] or len(rows)!=spec["records"] or sha(data)!=spec["sha256"]
            or len({r["sample_id"] for r in rows})!=len(rows) or any(r["split"]!=split for r in rows)):
        raise ValueError(f"Invalid cohort provenance for {split}")
    return rows,sha(data)


def masks_to_features(cup,disc):
    components=fundus_preprocessing._components(disc)
    if not components:
        return [None,None,None,1]
    main=max(components,key=lambda c:c["area"])
    l,t,r,b=main["bbox"]
    if main["area"]<16:
        return [None,None,None,1]
    inside=cup[t:b,l:r]&disc[t:b,l:r]
    cups=fundus_preprocessing._components(inside)
    if not cups:
        return [0.,0.,0.,0]
    chosen=max(cups,key=lambda c:c["area"])
    x0,y0,x1,y1=chosen["bbox"]
    return [(y1-y0)/(b-t),(x1-x0)/(r-l),min(1.,chosen["area"]/main["area"]),0]


def networks(args):
    if baseline.sha256(args.historical/"best.pt")!=DRIVE_SHA or baseline.sha256(args.unet)!=UNET_SHA:
        raise ValueError("Archived checkpoint SHA differs")
    if baseline.sha256(args.pretrained)!=baseline.OFFICIAL_SHA:
        raise ValueError("Official ViT-B initial weights differ")
    dc=torch.load(args.historical/"best.pt",map_location="cpu",weights_only=True,mmap=True)
    if dc["labels"]!=["glaucoma"] or dc["epoch"]!=12:
        raise ValueError("Expected the archived independent glaucoma model")
    clf=baseline.ScreeningModel("glaucoma",args.pretrained)
    clf.load_state_dict(dc["model"],strict=True)
    seg=torch.load(args.unet,map_location="cpu",weights_only=False,mmap=True)
    cfg=seg["config"]
    if cfg["labels"]!=["optic_cup","optic_disc"] or cfg["model"]["img_size"]!=512:
        raise ValueError("Unexpected segmentation channels or geometry")
    unet=structural.build_model(cfg,pretrained=False)
    unet.load_state_dict(seg["model"],strict=True)
    return clf.eval().requires_grad_(False),unet.eval().requires_grad_(False).to(memory_format=torch.channels_last),cfg


def generate(args,models,split):
    dest=args.output/"features"/f"{split}.jsonl"
    rows,mhash=source(args,split)
    if dest.exists():
        m=read(dest.with_suffix(".meta.json"))
        if m["source_manifest_sha256"]!=mhash or m["file_sha256"]!=baseline.sha256(dest):
            raise ValueError("Cached image features have changed")
        return [json.loads(l) for l in dest.read_bytes().splitlines()]
    dest.parent.mkdir(parents=True,exist_ok=True)
    clf,unet,seg_cfg=models
    tfm=classification.ClassificationTransform(baseline.TRANSFORM_CONFIG,training=False)
    output=[]
    start=time.monotonic()
    with torch.inference_mode():
        for first in range(0,len(rows),args.batch):
            batch=rows[first:first+args.batch]
            sx,cx=[],[]
            for row in batch:
                original=(args.data_root/row["image_path"]).read_bytes()
                if sha(original)!=row["image_sha256"]:
                    raise ValueError(f"Image SHA mismatch: {row['sample_id']}")
                with Image.open(io.BytesIO(original)) as image:
                    im=image.convert("RGB")
                sx.append(structural._tensor(im.resize((512,512),Image.Resampling.BILINEAR),seg_cfg))
                if split!="train":
                    cx.append(tfm(im))
            raw=unet(torch.stack(sx).contiguous(memory_format=torch.channels_last))
            if not torch.isfinite(raw).all():
                raise ValueError("Nonfinite U-Net output")
            masks=(raw>=0).numpy()
            views=None
            if cx:
                x=torch.stack(cx)
                preds=clf(torch.cat((x,x.flip((-1,))),0)).float().sigmoid()
                if not torch.isfinite(preds).all():raise ValueError("Nonfinite glaucoma output")
                views=preds.reshape(2,len(batch)).T.tolist()
            for i,row in enumerate(batch):
                output.append(dict(sample_id=row["sample_id"],image_sha256=row["image_sha256"],
                                   cdr_features=masks_to_features(masks[i,0],masks[i,1]),
                                   native_flip_probabilities=views[i] if views else None))
            if len(output)%100==0 or len(output)==len(rows):
                print(f"Computed full-cohort image features {split} {len(output)}/{len(rows)}",flush=True)
    raw_text="".join(json.dumps(row,allow_nan=False)+"\n" for row in output)
    evaluation.write_file(dest,raw_text,lock=True)
    evaluation.write_json(dest.with_suffix(".meta.json"),dict(source_manifest_sha256=mhash,file_sha256=baseline.sha256(dest),
                          samples=len(output),runtime_seconds=time.monotonic()-start,device="cpu",dtype="fp32",threads=args.threads,
                          segmentation_checkpoint_sha256=UNET_SHA,disease_checkpoint_sha256=DRIVE_SHA),lock=True)
    return output


def fit_structure(rows, features):
    idx=[i for i,r in enumerate(rows) if r["mask"]==[True]]
    if len(idx)!=692 or any(r["sample_id"]!=f["sample_id"] for r,f in zip(rows,features)):
        raise ValueError("Wrong disease training cohort")
    y=np.array([rows[i]["labels"][0] for i in idx],dtype=float)
    if list(np.bincount(y.astype(int)))!=[592,100]:
        raise ValueError("Unexpected disease support")
    x=np.array([features[i]["cdr_features"] for i in idx],float)
    med=np.nanmedian(x,axis=0)
    x=np.where(np.isfinite(x),x,med)
    mean,std=x.mean(0),x.std(0)
    std=np.where(std>1e-8,std,1.0)
    x=np.column_stack((np.ones(len(y)),(x-mean)/std))
    weights=np.where(y>0.5,len(y)/(2*y.sum()),len(y)/(2*(len(y)-y.sum())))
    reg=np.diag([0]+[1]*(x.shape[1]-1))
    beta=np.zeros(x.shape[1])
    for _ in range(100):
        p=1/(1+np.exp(-np.clip(x@beta,-40,40)))
        grad=x.T@(weights*(p-y))+reg@beta
        hess=(x.T*(weights*p*(1-p)))@x+reg
        step=np.linalg.solve(hess,grad)
        beta-=step
        if np.max(np.abs(step))<1e-10:break
    if np.max(np.abs(grad))>1e-6 or not np.isfinite(beta).all():raise ValueError("Logistic fit did not converge")
    return dict(median=med.tolist(),mean=mean.tolist(),std=std.tolist(),coefficients=beta.tolist(),
                feature_names=["vCDR","hCDR","cup_disc_area_ratio","disc_missing"],known_training_images=len(y),
                training_sources=dict(Counter(rows[i]["dataset"] for i in idx)),positive=int(y.sum()))


def scores(f,head):
    x=np.asarray(f["cdr_features"],dtype=float)
    x=(np.where(np.isfinite(x),x,head["median"])-head["mean"])/head["std"]
    coef=np.array(head["coefficients"]); sp=1/(1+np.exp(-np.clip(coef[0]+x@coef[1:],-40,40)))
    a,b=f["native_flip_probabilities"]
    return dict(flip_tta=(a+b)/2,structural=float(sp),hybrid=((a+b)/2+sp)/2)


def predictions(args,name,split,head):
    src,mhash=source(args,split)
    features=[json.loads(l) for l in (args.output/"features"/f"{split}.jsonl").read_bytes().splitlines()]
    recipe=dict(candidate=name,policy=POLICY,head_sha256=sha(canonical(head)) if name!="flip_tta" else None)
    run=args.output/name;run.mkdir(exist_ok=True)
    evaluation.write_file(run/"inference_recipe.json",canonical(recipe).decode(),lock=True)
    identity=sha(canonical(recipe));records=[]
    for row,feature in zip(src,features):
        if row["image_sha256"]!=feature["image_sha256"] or row["sample_id"]!=feature["sample_id"]:
            raise ValueError("Misaligned features")
        rec={k:row[k] for k in ("sample_id","image_path","dataset","split","image_sha256","labels","mask","patient_id","content_group_id")}
        prob=scores(feature,head)[name]
        clipped=np.clip(prob,np.nextafter(0.,1.),np.nextafter(1.,0.))
        records.append(dict(rec,probabilities=[prob],logits=[float(np.log(clipped)-np.log1p(-clipped))],checkpoint_sha256=identity))
    if len(src)!=len(records) or len(records)!=len(features): raise ValueError("Incomplete predictions")
    evaluation.write_file(run/f"{split}_predictions.jsonl","".join(json.dumps(r,allow_nan=False)+"\n" for r in records),lock=True)
    evaluation.write_json(run/f"{split}_predictions.meta.json",dict(labels=["glaucoma"],split=split,manifest_sha256=mhash,
                          checkpoint_sha256=identity,identity_kind="inference_recipe_sha256"),lock=True)
    return records,identity


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    evaluation.write_json(args.output/"selection_policy.json",POLICY,lock=True)
    models=networks(args)
    tr=generate(args,models,"train");v=generate(args,models,"val")
    train_rows,train_sha=source(args,"train")
    head=fit_structure(train_rows,tr)
    head.update(data_sha256=train_sha,features_sha256=baseline.sha256(args.output/"features/train.jsonl"),locator_sha256=UNET_SHA)
    evaluation.write_file(args.output/"structural_head.json",canonical(head).decode(),lock=True)
    baseline_rows,baseline_meta,_=evaluation.load_predictions(args.historical,"val")
    if baseline_meta["checkpoint_sha256"]!=DRIVE_SHA: raise ValueError("Baseline validation identity mismatch")
    val_source,val_sha=source(args,"val")
    if baseline_meta["manifest_sha256"]!=val_sha: raise ValueError("Baseline validation manifest mismatch")
    old={r["sample_id"]:r for r in baseline_rows}
    if len(old)!=len(val_source) or any(r["sample_id"] not in old or any(old[r["sample_id"]][k]!=r[k] for k in
                 ("image_path","labels","mask","content_group_id","patient_id")) for r in val_source):
        raise ValueError("Historical and current validation reference differ")
    val_stats={}
    for name in ORDER:
        rs,ident=(baseline_rows,DRIVE_SHA) if name=="historical" else predictions(args,name,"val",head)
        p=np.array([r["probabilities"][0] for r in rs if r["mask"][0]])
        y=np.array([r["labels"][0] for r in rs if r["mask"][0]],int)
        loc=args.historical if name=="historical" else args.output/name
        if name!="historical":evaluation.lock_thresholds(Namespace(run=loc,policy=None,min_recall=.9,max_fpr=.1))
        lock=read(loc/"thresholds.json")
        t=lock["per_label"]["glaucoma"]["threshold"]
        val_stats[name]=dict(**evaluation.metrics(y,p,t),threshold=t,checkpoint_sha256=ident,
                             val_predictions_sha256=baseline.sha256(loc/"val_predictions.jsonl"),
                             thresholds_sha256=baseline.sha256(loc/"thresholds.json"))
    winner=max(ORDER,key=lambda n:val_stats[n]["ap"])
    receipt=dict(selected=winner,validation=val_stats,policy_sha256=baseline.sha256(args.output/"selection_policy.json"),
                 labels=["glaucoma"],validation_rows=len(v),known=62,positive=13,positive_patients=7)
    evaluation.write_json(args.output/"selection.json",receipt,lock=True)
    print("Validation selection",json.dumps({k:round(x["ap"],6) for k,x in val_stats.items()}),winner,flush=True)
    if winner!="historical":
        for split in ("test","external_test"):
            generate(args,models,split); predictions(args,winner,split,head)
        evaluation.report(Namespace(run=args.output/winner,split=("val","test","external_test"),bootstrap=1000,seed=20261114))


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("--historical",required=True,type=Path)
    p.add_argument("--unet",required=True,type=Path)
    p.add_argument("--pretrained",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--data-root",type=Path,default=Path("."))
    p.add_argument("--release",type=Path,default=Path("data/prepared/screening_v1"))
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    run(args)


if __name__=="__main__":
    main()
