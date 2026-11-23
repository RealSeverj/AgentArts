"""Extract two frozen image views for the preregistered DR and EyeQ experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from training.classification import build_model, ClassificationTransform

REPORTS=PROJECT/"02_模型训练与评估"/"08_其余五模型优化_20261122"
ARCHIVE=PROJECT/"02_模型训练与评估"/"05_本轮训练结果_20261122"/"existing_models"
CACHE=PROJECT/"runs"/"existing_classifier_views_1122"
DATA=PROJECT/"data"
ACTIVE=PROJECT/"02_模型训练与评估"/"01_在用模型"
WEIGHTS={"dr":(ACTIVE/"01_糖网五级分级"/"best.pt","a85acf367a34361986739e93030bc5c51a390ddd33c63d49da883abf1562d4ab"),
         "quality":(ACTIVE/"02_眼底图像质控"/"quality_best.pt","fa5ee22b483b09c8dec9050455b5e6010d51c2ecced17d0a55982657bdd66f8f")}
SPLITS={"dr":{
 "train":("dr_grading/DDR",6041,"d37f79b24f72fc2fa91bfeafd76df29512765163780205cad0fae8b669e3e33d"),
 "val":("dr_grading/DDR",2479,"b081161f8fe2876254b1fcb316babab84ffd1ea54dd20918c59b2457d176a2fc"),
 "test":("dr_grading/DDR",3898,"7344a49e3925e5a0d97f960b1299ab57077b5bab61ae3e75ad85624ea04f82c5"),
 "external_test":("dr_grading/IDRiD",506,"3b7f861e45fe0f3610c3173e623a9ee15c2c464e6d1b85a4b43c9fe4f2adc194")},
 "quality":{
 "train":("image_quality/EyeQ",10664,"b333523b2ce5c415bf12a733161f5fcbce1d5f0be55310cf8c20fc128a2082d7"),
 "val":("image_quality/EyeQ",1875,"cdf5ce20022f6af08609313f7c38add1cb75b25808451e51c0276c111527de29"),
 "test":("image_quality/EyeQ",16249,"f7e1436214957d925cc085c4992f09b7bf1ec05081cec232836a13fe124b93f3")}}
TARGET={"dr":"dr_grade","quality":"quality_grade"}
DIRS={"dr":"model1_dr","quality":"model2_quality"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""): h.update(block)
    return h.hexdigest()


def write_json(path,value,lock=False):
    data=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    path.parent.mkdir(parents=True,exist_ok=True)
    if lock and path.exists():
        if path.read_text(encoding="utf8")!=data: raise ValueError(f"Conflicting immutable receipt {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent,mode="w",encoding="utf8",delete=False) as f:
        f.write(data);tmp=f.name
    os.replace(tmp,path)


def protocol(task):
    return dict(version=1,checkpoint_sha256=WEIGHTS[task][1],splits=SPLITS[task],
        data_use="Tune using train and val only; lock the checkpoint/decision rule before predicting test/external_test.",
        features="Frozen supervised backbone; original and deterministic horizontal reflection after existing geometry. Same RGB, normalization and FP16 as base. No manual labels as inputs.",
        train="Train a residual head only on the original training split and its two views. Seed 20261122, 8 fixed epochs, AdamW lr 0.0002, wd 0.01, batch 128, dropout=0.1. Frozen weights are never updated.",
        candidates=(["original_argmax","two_view_argmax","two_view_safety_prior","residual_two_view","residual_two_view_safety_prior"] if task=="dr" else ["historical_locked","two_view_reject","residual_two_view_reject"]),
        selection=("QWK >= baseline-0.005 and macro-F1 >= baseline-0.010. No extra grade3/4->0/1 on validation. Extra grade0/1->3/4 <= ceil(0.2% of grade0/1). Among eligible candidates minimize mean of grade1->0 and grade3->0/1/2 rates; then maximize QWK, macro-F1. First listed wins exact ties. Safety multipliers on calibrated or standard grade probabilities [1,1.3,1,1.2,1]." if task=="dr" else
                   "Retain the archived validation-locked reject threshold as the incumbent. For new scores minimize bad passed at <=5% false retakes among gradable images, then fewer false retakes and highest threshold. Rank eligible candidates by missed rejects, false retakes, prefer incumbent on ties."),
        adaptation=("Five-output additive logit residual. Per-view train-only class weights proportional to (N/n_grade)^0.35 normalized to mean one; objective weighted cross entropy + 0.5 * mean cumulative Brier loss + 0.02 * mean squared residual." if task=="dr" else
                    "One-logit reject residual on archived reject-vs-gradable log odds. Train-only BCE with positive weight sqrt(gradable/reject); additive 0.02 * mean squared residual."),
        validation="Patient if known, otherwise content group; 1000 paired fixed-model group bootstrap draws. Heldouts are historical model regression cohorts. No threshold retuning on heldout.",
        output="Deidentified sample keys and hashed groups only; auxiliary candidate uses the same pinned backbone and existing preprocessing. No production registry changes.")


def rows_for(task,split):
    key,n,digest=SPLITS[task][split]; path=DATA/"prepared/source_isolated/strict_cross_domain/tasks"/key/(split+".jsonl")
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest: raise ValueError(f"Manifest fingerprint differs: {task}/{split}")
    rows=[json.loads(s) for s in raw.splitlines() if s.strip()]
    if len(rows)!=n or len({r['sample_id'] for r in rows})!=n: raise ValueError("Coverage mismatch")
    for r in rows:
        if len(r['image_paths'])!=1 or len(r['image_sha256'])!=1 or not r["label_mask"][TARGET[task]]:
            raise ValueError("Expected a fully labelled single-eye classification record")
    return rows


def group_key(row):
    source=row.get("patient_id") or row.get("content_group_id") or row["image_sha256"][0]
    return hashlib.sha256(source.encode()).hexdigest()


def read_pixels(row,transform):
    rel=Path(row["image_paths"][0])
    if rel.is_absolute() or ".." in rel.parts: raise ValueError("Unexpected source image path")
    raw=(DATA/rel).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row["image_sha256"][0]: raise ValueError("Source image bytes differ from manifest")
    with Image.open(io.BytesIO(raw)) as im: return transform(im.convert("RGB"))


def cache_path(task,split):
    return CACHE/task/(split+".npz")


def run(task,phase,batch,workers):
    output=REPORTS/DIRS[task];output.mkdir(parents=True,exist_ok=True)
    write_json(output/"experiment_protocol.json",protocol(task),lock=True)
    if phase=="heldout" and not (output/"policy_lock.json").is_file(): raise ValueError("Heldout extraction requires policy lock")
    splits=("val","train") if phase=="development" else tuple(k for k in SPLITS[task] if k not in ("train","val"))
    ckpath,digest=WEIGHTS[task]
    if sha(ckpath)!=digest: raise ValueError("Unexpected production checkpoint")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    checkpoint=torch.load(ckpath,map_location="cpu",weights_only=False,mmap=True);cfg=checkpoint["config"]
    model=build_model(cfg,pretrained=False);model.load_state_dict(checkpoint["model"],strict=True);del checkpoint
    model.cuda().eval().requires_grad_(False); transform=ClassificationTransform(cfg,training=False)
    for split in splits:
        rows=rows_for(task,split);where=cache_path(task,split);meta_path=where.with_suffix(".meta.json")
        if where.is_file() and meta_path.is_file():
            meta=json.loads(meta_path.read_bytes())
            if (meta["sha256"]==sha(where) and meta["checkpoint_sha256"]==digest
                    and meta["manifest_sha256"]==SPLITS[task][split][2] and meta["views"]==2):
                print(f"{task} {split} verified feature cache",flush=True);continue
            raise ValueError("Existing feature cache failed identity verification")
        ts=now();start=time.monotonic();n=len(rows)
        feats=np.empty((n,2,model.backbone.num_features),dtype=np.float16)
        logits=np.empty((n,2,cfg["num_classes"]),dtype=np.float32)
        print(f"{ts} {task} {split} {n} rows, batch={batch}, device={torch.cuda.get_device_name()}",flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool, torch.inference_mode():
            def enqueue(i): return [pool.submit(read_pixels,r,transform) for r in rows[i:i+batch]]
            futures=enqueue(0)
            for i in range(0,n,batch):
                imgs=torch.stack([f.result() for f in futures]); futures=enqueue(i+batch)
                x=imgs.to("cuda")
                for v in (0,1):
                    with torch.autocast("cuda",dtype=torch.float16):
                        h=model.backbone(x if v==0 else torch.flip(x,dims=(-1,)))
                        z=model.head(h).float()
                    if not torch.isfinite(z).all(): raise ValueError("Nonfinite archived inference scores")
                    feats[i:i+len(x),v]=h.half().cpu().numpy(); logits[i:i+len(x),v]=z.cpu().numpy()
                if (i+len(x))%400 < batch or i+len(x)==n:
                    print(f"{task} {split} {i+len(x)}/{n} elapsed_seconds={int(time.monotonic()-start)}",flush=True)
        where.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=where.parent,delete=False) as file:
            np.savez(file,features=feats,logits=logits,y=np.array([r["labels"][TARGET[task]] for r in rows],dtype=np.int8),
                     sample_id=np.array([r["sample_id"] for r in rows]),group=np.array([group_key(r) for r in rows]),
                     image_sha=np.array([r["image_sha256"][0] for r in rows]))
            tmp=file.name
        os.replace(tmp,where)
        meta=dict(task=task,split=split,views=2,records=n,manifest_sha256=SPLITS[task][split][2],checkpoint_sha256=digest,
                  sha256=sha(where),features_dtype="float16",score_dtype="float32 from fp16 autocast",
                  start_utc=ts,end_utc=now(),seconds=time.monotonic()-start,device=torch.cuda.get_device_name(),
                  batch=batch,workers=workers,torch=str(torch.__version__))
        write_json(meta_path,meta);write_json(output/(split+"_view_extraction.json"),meta)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("task",choices=DIRS);p.add_argument("phase",choices=("development","heldout"))
    p.add_argument("--batch",type=int,default=4);p.add_argument("--workers",type=int,default=4)
    args=p.parse_args();run(args.task,args.phase,args.batch,args.workers)
