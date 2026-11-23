"""Fit a five-output classifier on frozen fundus features and compare fixed validation candidates."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.classification import ClassificationTransform, build_model
from training.metrics import binary_metrics
from training.screening_letterbox_cache import image_path, recipe_sha
from scripts.screening_evaluation import choose_threshold

LABELS = ["DR", "AMD", "MYOPIC_FUNDUS", "OPTIC_DISC_CUPPING", "ABNORMAL"]
OLD = "f5352ccb0834b41a2d9e6ed6d43c80b7b7b2e87dcfa1c0b3690f96eddafca640"
BACKBONE = "c64af5c437417efc6d9b3bd96072db1216f6de6e7798927a3aa74b00822e29a7"
COUNT = {"train":1878,"val":634,"test":615,"external_test":848}
GOAL = {"min_recall":0.9,"max_fpr":0.1}
ARMS = ["original", "original_flip_mean", "fundus_linear", "equal_mean"]
REGULARIZATION = 0.01


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    tmp = path.with_suffix(path.suffix+".part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")
    os.replace(tmp,path)


def check(ok, description):
    if not ok:
        raise ValueError(description)


def prepare(args):
    cohorts = {}
    signatures = {}
    for split in COUNT:
        source = "RFMiD2" if split == "external_test" else "RFMiD"
        path = args.source / source / (split+".jsonl")
        grouped = args.release / "grouped" / (split+".jsonl")
        current = {r["sample_id"]:r for r in map(json.loads,grouped.read_text(encoding="utf-8-sig").splitlines())}
        rows = []
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            r = json.loads(raw)
            g = current[r["sample_id"]]
            check(r["split"] == g["split"] == split and r["dataset"] == g["dataset"] == source, "Split mismatch")
            check(r["image_sha256"] == [g["image_sha256"]] and g["image_path"]=="data/"+r["image_path"],"Image mismatch")
            check(r["content_group_id"] == g["content_group_id"] and r["leakage_group_id"] == g["leakage_group_id"], "Group mismatch")
            row = {k:g[k] for k in ("sample_id","image_path","image_sha256","content_group_id","leakage_group_id")}
            row["target"] = [r["labels"][k] for k in LABELS]
            row["mask"] = [bool(r["label_mask"][k]) for k in LABELS]
            rows.append(row)
        check(len(rows)==len({r["sample_id"] for r in rows})==len(current)==COUNT[split],"Incomplete source")
        check(all(y in (0,1) for r in rows for y,m in zip(r["target"],r["mask"]) if m),"Non-binary label")
        identity = {r[k] for r in rows for k in ("image_sha256","leakage_group_id","content_group_id","sample_id")}
        check(all(identity.isdisjoint(prev) for prev in cohorts.values()),"Overlap between cohorts")
        cohorts[split] = identity
        dest=args.manifests/(split+".json")
        save(dest,rows)
        signatures[split] = {"original_sha256":sha(path),"grouped_sha256":sha(grouped),"rows_sha256":sha(dest),"count":len(rows)}
    save(args.manifests/"index.json",{"labels":LABELS,"splits":signatures,"selection":"validation_five_label_macro_ap","model3_sha256":OLD,"transfer_parent_sha256":BACKBONE})
    print("Verified original model 3 manifests and source/model 5 split identity",COUNT,flush=True)


class Images(Dataset):
    def __init__(self, args, split, rows, config, geometric):
        self.rows=rows
        self.transform=ClassificationTransform(config,training=False,input_preprocessed=geometric)
        self.root=args.root
        self.sources=None
        if geometric:
            policy=recipe_sha()
            index=json.loads((args.geometry/(split+".json")).read_bytes())
            rows_sha=json.loads((args.manifests/"index.json").read_bytes())["splits"][split]["grouped_sha256"]
            check(index["source_manifest_sha256"]==rows_sha and index["recipe_sha256"]==policy,"Geometry contract mismatch")
            refs={r["sample_id"]:r for r in index["records"]}
            self.sources=[]
            for r in rows:
                item=refs[r["sample_id"]]
                check(r["image_sha256"]==item["image_sha256"],"Invalid source geometry")
                self.sources.append((image_path(args.geometry,policy,r["image_sha256"]),item["png_sha256"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row=self.rows[i]
        if self.sources is None:
            path,expected=self.root/row["image_path"],row["image_sha256"]
        else:
            path,expected=self.sources[i]
        raw=path.read_bytes()
        check(hashlib.sha256(raw).hexdigest()==expected,"Evaluation image fingerprint mismatch")
        with Image.open(io.BytesIO(raw)) as im:
            return self.transform(im.convert("RGB"))


def read_rows(args,split):
    path=args.manifests/(split+".json")
    index=json.loads((args.manifests/"index.json").read_bytes())
    check(index["labels"]==LABELS and index["splits"][split]["rows_sha256"]==sha(path),"Manifest changed")
    return json.loads(path.read_bytes())


@torch.inference_mode()
def extract(args,split,rows,model,config,geometric):
    dataset=Images(args,split,rows,config,geometric)
    loader=DataLoader(dataset,batch_size=args.batch,shuffle=False,num_workers=args.workers,pin_memory=True)
    outputs=[]
    for images in loader:
        images=images.cuda(non_blocking=True)
        with torch.autocast("cuda",enabled=geometric,dtype=torch.float16):
            a=model(images).float()
            b=model(images.flip(-1)).float()
        if geometric:
            out=(a+b)*0.5
        else:
            out=torch.stack([a.sigmoid(),(a.sigmoid()+b.sigmoid())*0.5], dim=1)
        outputs.append(out.cpu())
    result=torch.cat(outputs)
    check(len(result)==len(rows) and torch.isfinite(result).all(),"Invalid representation")
    print(split,"features" if geometric else "original+flip",tuple(result.shape),flush=True)
    return result


def new_parent(args):
    import timm
    check(sha(args.parent)==BACKBONE,"Transfer parent differs from the validation-selected D checkpoint")
    weights=torch.load(args.parent,map_location="cpu",mmap=True,weights_only=True)
    identity=weights["identity"]
    check(identity["candidate"]=="vitl_global_letterbox" and identity["geometry_recipe_sha256"]==recipe_sha(),"Unexpected input")
    net=timm.create_model(identity["backbone"],pretrained=False,num_classes=0,img_size=392)
    net.load_state_dict({k.removeprefix("backbone."):v for k,v in weights["model"].items() if k.startswith("backbone.")},strict=True)
    return net.cuda().eval().requires_grad_(False),identity["transform"]


def original(args):
    check(sha(args.legacy)==OLD,"Legacy checkpoint hash mismatch")
    weights=torch.load(args.legacy,map_location="cpu",mmap=True,weights_only=False)
    cfg=weights["config"]
    check(cfg["labels"]==LABELS,"Legacy label contract mismatch")
    net=build_model(cfg,pretrained=False)
    net.load_state_dict(weights["model"],strict=True)
    return net.cuda().eval().requires_grad_(False),cfg


def fit(x,rows,args):
    x=x.double()
    y=torch.tensor([r["target"] for r in rows],dtype=x.dtype)
    check(all(all(r["mask"]) for r in rows),"Training targets must be known")
    mean=x.mean(dim=0);scale=x.std(dim=0).clamp_min(1e-4)
    x=(x-mean)/scale
    w=torch.zeros(x.shape[1],len(LABELS),dtype=x.dtype,requires_grad=True)
    b=torch.zeros(len(LABELS),dtype=x.dtype,requires_grad=True)
    opt=torch.optim.LBFGS([w,b],max_iter=120,tolerance_grad=1e-8,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(x@w+b,y) + REGULARIZATION*w.square().sum()/len(LABELS)/2
        loss.backward()
        return loss
    opt.step(closure)
    state={"mean":mean,"std":scale,"weight":w.detach(),"bias":b.detach(),"labels":LABELS,
           "parent_sha256":BACKBONE,"regularization":REGULARIZATION,"features":"mean(original,horizontal_flip)"}
    path=args.out/"linear_head.pt"
    torch.save(state,path.with_suffix(".tmp"))
    os.replace(path.with_suffix(".tmp"),path)
    return state


def probe(x,state):
    return (((x.double()-state["mean"])/state["std"])@state["weight"]+state["bias"]).sigmoid().numpy()


def targets(rows,i):
    mask=np.array([r["mask"][i] for r in rows],bool)
    return mask,np.array([r["target"][i] for r in rows if r["mask"][i]],int)


def summarize(p,rows,locks=None):
    known={}
    for i,label in enumerate(LABELS):
        mask,y=targets(rows,i)
        if not len(y) or not y.sum() or y.sum()==len(y):
            continue
        scores=p[mask,i]
        known[label]=binary_metrics(y,scores,threshold=locks[label]["threshold"] if locks else 0.5)
        known[label]["working_point_met"]=known[label]["sensitivity"]>=0.9 and known[label]["specificity"]>=0.9
    return {"macro_ap":float(np.mean([v["auprc"] for v in known.values()])),
            "macro_f1":float(np.mean([v["f1"] for v in known.values()])),
            "per_label":known,"working_points_met":sum(v["working_point_met"] for v in known.values()),"labeled_outputs":len(known)}


def pretrain(args):
    from filelock import FileLock
    receipt_path=args.out/"transfer_fit.json"
    score_path=args.out/"transfer_val.npy"
    with FileLock(str(args.out/"transfer_fit.lock"),timeout=1800):
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_bytes())
            check(receipt["parent"]==BACKBONE and receipt["manifest_index"]==sha(args.manifests/"index.json")
                  and receipt["head"]==sha(args.out/"linear_head.pt") and receipt["val_scores"]==sha(score_path),
                  "Cached supervised transfer fit is inconsistent")
            return np.load(score_path,allow_pickle=False)
        train,val=(read_rows(args,s) for s in ("train","val"))
        net,cfg=new_parent(args)
        x=extract(args,"train",train,net,cfg,True)
        state=fit(x,train,args)
        transfer=probe(extract(args,"val",val,net,cfg,True),state)
        np.save(score_path,transfer,allow_pickle=False)
        save(receipt_path,{"parent":BACKBONE,"manifest_index":sha(args.manifests/"index.json"),
             "head":sha(args.out/"linear_head.pt"),"val_scores":sha(score_path),"regularization":REGULARIZATION})
        return transfer


def select(args):
    val=read_rows(args,"val")
    transfer=pretrain(args)
    torch.cuda.empty_cache()
    model,cfg=original(args)
    old=extract(args,"val",val,model,cfg,False).numpy()
    sets=[old[:,0],old[:,1],transfer,(old[:,1]+transfer)*0.5]
    table={}
    for name,p in zip(ARMS,sets):
        thresholds={}
        for i,key in enumerate(LABELS):
            mask,y=targets(val,i)
            thresholds[key]=choose_threshold(y,p[mask,i],GOAL)
        table[name]={"metrics":summarize(p,val,thresholds),"thresholds":thresholds}
    winner=max(ARMS,key=lambda k:table[k]["metrics"]["macro_ap"])
    np.savez_compressed(args.out/"validation_scores.npz",**dict(zip(ARMS,sets)))
    save(args.out/"validation_lock.json",{"labels":LABELS,"selection":"validation_five_label_macro_ap","selected":winner,
         "original_sha256":OLD,"parent_sha256":BACKBONE,"head_sha256":sha(args.out/"linear_head.pt"),
         "manifest_index_sha256":sha(args.manifests/"index.json"),"validation_scores_sha256":sha(args.out/"validation_scores.npz"),
         "candidates":table,"validation_images":len(val),"train_images":COUNT["train"],"tie_preference":ARMS})
    print("VALIDATION_LOCK",winner,{k:round(table[k]["metrics"]["macro_ap"],5) for k in ARMS},flush=True)


def paired_ap(rows,old,new):
    keys=[r["content_group_id"] for r in rows]
    groups,inv=np.unique(keys,return_inverse=True)
    rng=np.random.default_rng(20261122)
    series=[]
    for _ in range(1000):
        counts=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups))[inv]
        diffs=[]
        for i in range(len(LABELS)):
            mask,y=targets(rows,i)
            if not len(y):
                continue
            w=counts[mask]
            if not w[y==1].sum() or not w[y==0].sum():
                break
            diffs.append(binary_metrics(y,new[mask,i],w)["auprc"]-binary_metrics(y,old[mask,i],w)["auprc"])
        else:
            series.append(float(np.mean(diffs)))
    return {"draws":1000,"valid_draws":len(series),"unit":"content_group","delta_ci95":np.quantile(series,[.025,.975]).tolist()}


def evaluate(args):
    lock=json.loads((args.out/"validation_lock.json").read_bytes())
    check(lock["labels"]==LABELS and lock["manifest_index_sha256"]==sha(args.manifests/"index.json") and lock["head_sha256"]==sha(args.out/"linear_head.pt") and lock["validation_scores_sha256"]==sha(args.out/"validation_scores.npz"),"Selection identity changed")
    key=lock["selected"]
    state=torch.load(args.out/"linear_head.pt",weights_only=True,map_location="cpu")
    old,cfg=original(args)
    parent,pcfg=new_parent(args) if key in ("fundus_linear","equal_mean") else (None,None)
    result={}
    for split in ("test","external_test"):
        rows=read_rows(args,split)
        legacy=extract(args,split,rows,old,cfg,False).numpy()
        old_scores=legacy[:,0]
        if parent is None:
            selected=legacy[:,int(key=="original_flip_mean")]
        else:
            p=probe(extract(args,split,rows,parent,pcfg,True),state)
            selected=p if key=="fundus_linear" else (p+legacy[:,1])*0.5
        result[split]={"images":len(rows),"legacy_at_0_5":summarize(old_scores,rows),"selected_at_0_5":summarize(selected,rows),
            "original":summarize(old_scores,rows,lock["candidates"]["original"]["thresholds"]),
            "selected":summarize(selected,rows,lock["candidates"][key]["thresholds"]),
            "paired_macro_ap_delta":paired_ap(rows,old_scores,selected)}
        np.savez_compressed(args.out/(split+"_paired_scores.npz"),sample_ids=np.array([r["sample_id"] for r in rows]),original=old_scores,selected=selected)
        print(split,"macro AP",result[split]["original"]["macro_ap"],result[split]["selected"]["macro_ap"],flush=True)
    save(args.out/"heldout_report.json",{"winner":key,"validation_lock_sha256":sha(args.out/"validation_lock.json"),"splits":result})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["prepare","pretrain","select","evaluate"])
    for opt in ("source","release","manifests","geometry","root","legacy","parent","out"):
        ap.add_argument("--"+opt,type=Path,default=Path("."))
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--batch",type=int,default=12)
    args=ap.parse_args()
    if args.action=="prepare":
        args.manifests.mkdir(parents=True,exist_ok=True)
        return prepare(args)
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(20261122)
    {"pretrain":pretrain,"select":select,"evaluate":evaluate}[args.action](args)


if __name__=="__main__":
    main()

