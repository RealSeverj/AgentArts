"""Fit research-only DR/EyeQ head and decision candidates on fixed development splits."""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import research_classifier_views as views
from scripts.evaluate_eyeq_operating_point import fit_threshold
from training.metrics import binary_metrics, multiclass_metrics


def read(path):
    return json.loads(path.read_bytes())


def softmax(x):
    x=x-x.max(axis=-1,keepdims=True)
    p=np.exp(x)
    return p/p.sum(axis=-1,keepdims=True)


def arrays(task,split):
    p=views.cache_path(task,split);meta=read(p.with_suffix(".meta.json"))
    if (meta["sha256"]!=views.sha(p) or meta["manifest_sha256"]!=views.SPLITS[task][split][2]
            or meta["checkpoint_sha256"]!=views.WEIGHTS[task][1]):
        raise ValueError("Feature-cache provenance changed")
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}


def overlap(left,right):
    return {k:len(set(left[k]) & set(right[k])) for k in ("sample_id","image_sha","group")}


def fit_head(task,data):
    torch.manual_seed(20261122)
    feats=torch.from_numpy(data["features"]).reshape(-1,data["features"].shape[-1])
    base=torch.from_numpy(data["logits"]).reshape(-1,data["logits"].shape[-1])
    y=torch.from_numpy(data["y"].astype(np.int64)).repeat_interleave(2)
    nclass=base.shape[-1]; counts=torch.bincount(y,minlength=nclass).float()
    output=nclass if task=="dr" else 1
    h=torch.nn.Linear(feats.shape[1],output)
    torch.nn.init.zeros_(h.weight); torch.nn.init.zeros_(h.bias)
    dropout=torch.nn.Dropout(.1); opt=torch.optim.AdamW(h.parameters(),lr=.0002,weight_decay=.01)
    grade_weight=(len(y)/counts)**.35; grade_weight/=grade_weight.mean()
    reject_weight=torch.sqrt(counts[:2].sum()/counts[2])
    history=[]
    for epoch in range(8):
        order=torch.randperm(len(y)); loss_sum=0.
        for idx in order.split(128):
            z=base[idx]; target=y[idx]; delta=h(dropout(feats[idx].float()))
            if task=="dr":
                logits=z+delta
                loss=F.cross_entropy(logits,target,weight=grade_weight)
                cdf=logits.softmax(1).cumsum(1)[:,:-1]
                truth=target[:,None]<=torch.arange(nclass-1)[None,:]
                loss=loss+.5*F.mse_loss(cdf,truth.float())
            else:
                margin=z[:,2]-torch.logsumexp(z[:,:2],dim=1)+delta[:,0]
                loss=F.binary_cross_entropy_with_logits(margin,(target==2).float(),pos_weight=reject_weight)
            loss=loss+.02*delta.square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            loss_sum+=float(loss.detach())*len(idx)
        history.append(dict(epoch=epoch+1,train_loss=loss_sum/len(y)))
    return h.weight.detach().numpy(),h.bias.detach().numpy(),history


def transfer_p(task,data,w,b):
    z=data["logits"]
    d=data["features"].astype(np.float32)@w.T+b
    if task=="dr": return softmax(z+d).mean(1)
    # Changing only the reject logit preserves the relative Good/Usable conditional scores.
    z=z.copy();z[:,:,2]+=d[:,:,0];return softmax(z).mean(1)


def probs(task,data,rule,w,b):
    original=softmax(data["logits"])
    if rule in ("original_argmax","historical_locked"): return original[:,0]
    if rule in ("two_view_argmax","two_view_safety_prior","two_view_reject"):
        score=original.mean(1)
    else: score=transfer_p(task,data,w,b)
    if rule.endswith("safety_prior"):
        score=score*np.array([1,1.3,1,1.2,1],dtype=np.float32)
        score=score/score.sum(1,keepdims=True)
    return score


def calls(task,p,t=None):
    if task=="dr" or t is None: return p.argmax(1)
    return np.where(p[:,2]>=t,2,p[:,:2].argmax(1))


def metrics(task,y,pred,weights=None):
    k=5 if task=="dr" else 3
    cm=np.bincount(y.astype(int)*k+pred,minlength=k*k,weights=weights).reshape(k,k)
    pos=cm.sum(1);count=pos.sum()
    if not count: raise ValueError("Empty cohort")
    safe=lambda a,b:float(a/b) if b else None
    cls_f1=np.divide(2*np.diag(cm),pos+cm.sum(0),out=np.zeros(k),where=(pos+cm.sum(0))>0)
    out=dict(accuracy=float(np.trace(cm)/count),macro_f1=float(cls_f1.mean()),samples=int(count),
             confusion=cm.astype(int).tolist(),per_grade_recall=[safe(cm[i,i],n) for i,n in enumerate(pos)])
    if task=="dr":
        distances=(np.arange(k)[:,None]-np.arange(k)[None,:])**2
        chance=(np.outer(pos,cm.sum(0))/count*distances).sum()
        out.update(qwk=float(1-(cm*distances).sum()/chance) if chance else None,
            grade1_to0=int(cm[1,0]),grade1_to0_rate=safe(cm[1,0],pos[1]),
            grade3_under=int(cm[3,:3].sum()),grade3_under_rate=safe(cm[3,:3].sum(),pos[3]),
            severe_to01=int(cm[3:,:2].sum()),severe_to01_rate=safe(cm[3:,:2].sum(),pos[3:].sum()),
            low_to_severe=int(cm[:2,3:].sum()),low_to_severe_rate=safe(cm[:2,3:].sum(),pos[:2].sum()))
    else:
        bad=pos[2]; good=pos[:2].sum(); fn=cm[2,:2].sum();fp=cm[:2,2].sum()
        out.update(bad_images=int(bad),gradable_images=int(good),bad_passed=int(fn),bad_passed_rate=safe(fn,bad),
                   gradable_rejected=int(fp),gradable_rejected_rate=safe(fp,good))
    return out


def historical(task,split,data):
    if task=="quality":
        base=views.ARCHIVE/"eyeq";pred=base/(split+"_predictions.jsonl")
        meta=read(base/(split+"_predictions.meta.json"))
        require(meta["predictions_sha256"]==views.sha(pred) and meta["checkpoint_sha256"]==views.WEIGHTS[task][1],"Historical EyeQ data mismatch")
    else:
        base=(views.PROJECT/"cloud_backup/20261114_DDR_crop_letterbox/reports/evaluation/best"
              if split=="test" else views.ARCHIVE/"dr_idrid")
        pred=base/("predictions_"+split+".jsonl")
        meta=read(base/("predictions_"+split+".meta.json"))
        require(meta["jsonl_sha256"]==views.sha(pred),"Historical DR scores changed")
    require(meta["manifest_sha256"]==views.SPLITS[task][split][2],"Historical cohort mismatch")
    by_id={r["sample_id"]:r for r in (json.loads(s) for s in pred.read_bytes().splitlines() if s)}
    require(len(by_id)==len(data["y"]) and set(by_id)==set(data["sample_id"]),"Historical eye alignment failed")
    rows=[by_id[s] for s in data["sample_id"]]
    y=[r["grade" if task=="quality" else "target"] for r in rows]
    require(np.array_equal(y,data["y"]),"Historical labels differ")
    return np.array([r["probabilities"] for r in rows]),views.sha(pred)


def require(ok,message):
    if not ok: raise ValueError(message)


def paired(task,y,old,new,groups):
    names=(("accuracy","macro_f1","qwk","grade1_to0_rate","grade3_under_rate","severe_to01_rate","low_to_severe_rate")
           if task=="dr" else ("accuracy","macro_f1","bad_passed_rate","gradable_rejected_rate"))
    rng=np.random.default_rng(20261122); ids,inv=np.unique(groups,return_inverse=True); vals={k:[] for k in names}
    for _ in range(1000):
        w=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids))[inv]
        a,b=(metrics(task,y,p,w) for p in (old,new))
        for k in names:
            if a[k] is not None and b[k] is not None: vals[k].append(b[k]-a[k])
    a,b=metrics(task,y,old),metrics(task,y,new)
    return dict(groups=len(ids),draws=1000,convention="selected minus archived comparator; fixed model and validation threshold",
                metrics={k:dict(old=a[k],selected=b[k],delta=b[k]-a[k] if a[k] is not None and b[k] is not None else None,
                                ci95=np.quantile(s,[.025,.975]).tolist() if len(s)>1 else None,valid_draws=len(s)) for k,s in vals.items()})


def save_head(path,weight,bias):
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        np.savez(f,weight=weight,bias=bias);tmp=f.name
    os.replace(tmp,path)


def select(task,out):
    if (out/"policy_lock.json").is_file():
        print(f"{task}: policy already locked",flush=True);return
    train=arrays(task,"train");val=arrays(task,"val")
    overlap_dev=overlap(train,val);require(not any(overlap_dev.values()),"Development cohort overlap")
    require(read(out/"experiment_protocol.json")==json.loads(json.dumps(views.protocol(task))),"Preregistration differs")
    w,b,history=fit_head(task,train);head=out/"trained_residual_head.npz";save_head(head,w,b)
    views.write_json(out/"head_training_history.json",history)
    rules=views.protocol(task)["candidates"]; y=val["y"]; results={}
    prior=read(views.ARCHIVE/"eyeq/threshold.json") if task=="quality" else None
    for rule in rules:
        pred=probs(task,val,rule,w,b);cut=None
        if task=="quality":
            cut=float(prior["threshold"]) if rule=="historical_locked" else float(fit_threshold(y,pred[:,2]))
        m=metrics(task,y,calls(task,pred,cut))
        if task=="quality":m["reject_ap"]=binary_metrics((y==2).astype(int),pred[:,2])["auprc"]
        results[rule]=dict(threshold=cut,metrics=m)
    ref=results[rules[0]]["metrics"]
    for rule,v in results.items():
        m=v["metrics"]
        if task=="dr":
            eligible=(m["qwk"]>=ref["qwk"]-.005 and m["macro_f1"]>=ref["macro_f1"]-.010
                      and m["severe_to01"]<=ref["severe_to01"]
                      and m["low_to_severe"]<=ref["low_to_severe"]+math.ceil(.002*int((y<2).sum())))
            rank=[(m["grade1_to0_rate"]+m["grade3_under_rate"])/2,-m["qwk"],-m["macro_f1"]]
        else:
            eligible=m["gradable_rejected_rate"]<=.05
            rank=[m["bad_passed"],m["gradable_rejected"]]
        v.update(eligible=eligible,rank=rank)
    chosen=min((r for r in rules if results[r]["eligible"]),key=lambda r:results[r]["rank"])
    views.write_json(out/"validation_candidates.json",dict(samples=len(y),cross_split_intersections=overlap_dev,results=results))
    validation_file=out/"validation_candidate_predictions.csv"
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8-sig",newline="",dir=out,delete=False) as stream:
        writer=csv.writer(stream)
        columns=["sample_id","group_hash","target"]
        for rule in rules:
            columns.extend([rule+"_decision",*[rule+"_score_"+str(i) for i in range(val["logits"].shape[-1])]])
        writer.writerow(columns)
        scored=[probs(task,val,r,w,b) for r in rules]
        decisions=[calls(task,p,results[r]["threshold"]) for r,p in zip(rules,scored)]
        for i,sid in enumerate(val["sample_id"]):
            row=[sid,val["group"][i],int(y[i])]
            for sc,dc in zip(scored,decisions): row.extend([int(dc[i]),*sc[i].tolist()])
            writer.writerow(row)
        tmp=stream.name
    os.replace(tmp,validation_file)
    views.write_json(out/"policy_lock.json",dict(task=task,locked_utc=views.now(),rule=chosen,threshold=results[chosen]["threshold"],
      class_order=([0,1,2,3,4] if task=="dr" else ["Good","Usable","Reject"]),checkpoint_sha256=views.WEIGHTS[task][1],
      residual_head_sha256=views.sha(head),protocol_sha256=views.sha(out/"experiment_protocol.json"),
      validation_manifest_sha256=views.SPLITS[task]["val"][2],validation_candidates_sha256=views.sha(out/"validation_candidates.json"),
      validation_scores_sha256=views.sha(validation_file),
      training_cache_sha256=views.sha(views.cache_path(task,"train")),validation_cache_sha256=views.sha(views.cache_path(task,"val")),
      historical_quality_threshold_sha256=views.sha(views.ARCHIVE/"eyeq/threshold.json") if task=="quality" else None),lock=True)
    print(f"{task}: {chosen} locked on validation; threshold={results[chosen]['threshold']}",flush=True)


def read_locked(task,out):
    lock=read(out/"policy_lock.json")
    require(lock["task"]==task and lock["checkpoint_sha256"]==views.WEIGHTS[task][1] and
            lock["residual_head_sha256"]==views.sha(out/"trained_residual_head.npz") and
            lock["protocol_sha256"]==views.sha(out/"experiment_protocol.json") and
            lock["validation_candidates_sha256"]==views.sha(out/"validation_candidates.json"),"Decision lock changed")
    with np.load(out/"trained_residual_head.npz",allow_pickle=False) as h: return lock,h["weight"],h["bias"]


def original_grading_metrics(split, data, archived_scores, selected_scores):
    manifest=views.rows_for("dr",split)
    require([r["sample_id"] for r in manifest]==data["sample_id"].tolist(),"Evaluation weights misaligned")
    weights=np.array([r["content_group_evaluation_weight"] for r in manifest])
    original=(views.PROJECT/"cloud_backup/20261114_DDR_crop_letterbox/reports/evaluation/best/evaluation_test.json"
              if split=="test" else views.ARCHIVE/"dr_idrid/evaluation_external_test.json")
    published=read(original)["metrics"]
    results={}
    for name,score in (("historical",archived_scores),("selected",selected_scores)):
        results[name]={"sample":multiclass_metrics(data["y"],score,5),
                       "content_weighted":multiclass_metrics(data["y"],score,5,weights)}
    for unit,reference in (("sample",published),("content_weighted",published["content_weighted"])):
        for key in ("accuracy","qwk","macro_f1","mean_absolute_grade_error"):
            require(np.isclose(results["historical"][unit][key],reference[key],rtol=0,atol=1e-8),
                    f"Historical grading metric differs: {split}/{unit}/{key}")
    return dict(original_report_sha256=views.sha(original),**results)


def evaluate(task,out):
    lock,w,b=read_locked(task,out); train,val=(arrays(task,k) for k in ("train","val"))
    splits=[s for s in views.SPLITS[task] if s not in ("train","val")];reports={}
    for split in splits:
        data=arrays(task,split); y=data["y"]; ov={k:overlap(old,data) for k,old in (("train",train),("val",val))}
        require(not any(any(v.values()) for v in ov.values()),"Held-out source overlaps development records")
        original=softmax(data["logits"][:,0]);p=probs(task,data,lock["rule"],w,b)
        oldp,oldhash=historical(task,split,data)
        prior=float(read(views.ARCHIVE/"eyeq/threshold.json")["threshold"]) if task=="quality" else None
        old_call=calls(task,oldp,prior);pred=calls(task,p,lock["threshold"])
        csvname=out/(split+"_paired_predictions.csv")
        with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8-sig",newline="",dir=out,delete=False) as stream:
            writer=csv.writer(stream);writer.writerow(["sample_id","group_hash","target","archived_argmax","archived_policy","selected_grade",
              *["archived_score_"+str(i) for i in range(p.shape[1])],*["selected_score_"+str(i) for i in range(p.shape[1])]])
            for idx,sid in enumerate(data["sample_id"]):
                writer.writerow([sid,data["group"][idx],y[idx],oldp[idx].argmax(),old_call[idx],pred[idx],
                                 *oldp[idx].tolist(),*p[idx].tolist()])
            tmp=stream.name
        os.replace(tmp,csvname)
        a,bm=metrics(task,y,old_call),metrics(task,y,pred)
        if task=="quality":
            for m,pp in ((a,oldp),(bm,p)): m["reject_ap"]=binary_metrics((y==2).astype(int),pp[:,2])["auprc"]
        reports[split]=dict(manifest_sha256=views.SPLITS[task][split][2],
          archived_reference_predictions_sha256=oldhash,score_export_sha256=views.sha(csvname),
          known_development_intersections=ov,archived_argmax=metrics(task,y,oldp.argmax(1)),archived_locked_policy=a,
          replayed_original=metrics(task,y,calls(task,original,prior)),selected=bm,paired_delta_ci95=paired(task,y,old_call,pred,data["group"]),
          reference_replay_argmax_changed=int(np.sum(oldp.argmax(1)!=original.argmax(1))),
          reference_replay_max_abs_score_error=float(np.max(abs(oldp-original))))
        if task=="dr":
            reports[split]["original_grading_metrics"]=original_grading_metrics(split,data,oldp,p)
        else:
            reports[split]["archived_threshold_0_5"]=metrics(task,y,calls(task,oldp,.5))
        views.write_json(out/(split+"_report.json"),reports[split]);print(f"{task} {split}: {len(y)} paired records evaluated",flush=True)
    views.write_json(out/"evaluation_complete.json",dict(completed_utc=views.now(),task=task,rule=lock["rule"],
        policy_lock_sha256=views.sha(out/"policy_lock.json"),cohort_reports={k:views.sha(out/(k+"_report.json")) for k in splits}))


def original_view_reader(task):
    from training.classification import build_model,ClassificationTransform
    path,digest=views.WEIGHTS[task]
    require(views.sha(path)==digest,"Wrong original backbone")
    old=torch.load(path,weights_only=False,mmap=True,map_location="cpu");cfg=old["config"]
    net=build_model(cfg,pretrained=False);net.load_state_dict(old["model"],strict=True);del old
    net.cuda().requires_grad_(False).eval()
    transform=ClassificationTransform(cfg,False)
    def extract(image_path):
        with Image.open(image_path) as im: x=transform(im.convert("RGB"))[None].cuda()
        f,z=[],[]
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
            for v in (x,x.flip((-1,))):
                h=net.backbone(v);z.append(net.head(h).float().cpu().numpy());f.append(h.half().cpu().numpy())
        return dict(features=np.stack(f,axis=1),logits=np.stack(z,axis=1))
    return extract


def reviewer(task,out):
    lock,w,b=read_locked(task,out)
    extract=original_view_reader(task);policy_sha=views.sha(out/"policy_lock.json")
    def classify(image_path):
        p=probs(task,extract(image_path),lock["rule"],w,b)
        return dict(task=task,use="offline_research",rule=lock["rule"],grade=int(calls(task,p,lock["threshold"])[0]),
                    class_order=lock["class_order"],decision_scores=p[0].tolist(),reject_threshold=lock["threshold"],
                    base_weights_sha256=lock["checkpoint_sha256"],policy_sha256=policy_sha)
    return classify


def verify_images(task,out):
    lock,_,_=read_locked(task,out); rule=lock["rule"]
    score_file=out/"validation_candidate_predictions.csv"
    require(views.sha(score_file)==lock["validation_scores_sha256"],"Validation scores changed")
    with score_file.open(encoding="utf-8-sig",newline="") as stream:
        scores={r["sample_id"]:r for r in csv.DictReader(stream)}
    samples={}
    for row in views.rows_for(task,"val"):
        target=row["labels"][views.TARGET[task]]
        samples.setdefault(target,row)
    classify=reviewer(task,out);checks=[]
    for target,row in sorted(samples.items()):
        path=views.DATA/row["image_paths"][0];require(views.sha(path)==row["image_sha256"][0],"Source image changed")
        prediction=classify(path); archived=scores[row["sample_id"]]
        ref=[float(archived[rule+"_score_"+str(i)]) for i in range(len(lock["class_order"]))]
        checks.append(dict(sample_id=row["sample_id"],target=target,original_image_grade=prediction["grade"],
                       batch_grade=int(archived[rule+"_decision"]),max_abs_score_error=float(np.max(abs(np.array(ref)-prediction["decision_scores"])))))
    views.write_json(out/"original_image_check.json",dict(task=task,selection_rule=rule,
        design="First validation sample in source-manifest order for each true class; offline original-image entrypoint, batch size one.",
        policy_lock_sha256=views.sha(out/"policy_lock.json"),samples=checks,
        matched_decisions=sum(r["batch_grade"]==r["original_image_grade"] for r in checks),
        max_abs_score_error=max(r["max_abs_score_error"] for r in checks)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("task",choices=views.DIRS)
    p.add_argument("action",choices=("select","evaluate","predict","verify-images"));p.add_argument("--image",type=Path)
    a=p.parse_args();torch.set_num_threads(2);out=views.REPORTS/views.DIRS[a.task]
    if a.action=="select": select(a.task,out)
    elif a.action=="evaluate": evaluate(a.task,out)
    elif a.action=="verify-images": verify_images(a.task,out)
    else:
        if not a.image: p.error("predict requires --image")
        print(json.dumps(reviewer(a.task,out)(a.image),ensure_ascii=False))


if __name__=="__main__":main()
