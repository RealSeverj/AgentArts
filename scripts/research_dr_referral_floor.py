"""Second-stage DDR validation study with a preregistered ordinal referral floor."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import optimize_existing_classifiers as base
from scripts import research_classifier_views as views

ROOT=views.REPORTS/views.DIRS["dr"]
OUT=ROOT/"ordinal_refinement"


def candidates():
    yield {"name":"original","mode":"original","grade1":1.0,"grade3":1.0}
    yield {"name":"floor_best_two_view","mode":"floor","grade1":1.3,"grade3":1.2}
    for a in (1.0,1.3,1.5,2.0,3.0):
        for b in (1.0,1.2,1.5,2.0):
            yield {"name":f"up_one_{a:g}_{b:g}","mode":"up_one","grade1":a,"grade3":b}


def protocol():
    return dict(stage="secondary exploratory validation, requested after initial validation; no heldout results inspected",
        registered_utc=views.now(),previous_lock_sha256=views.sha(ROOT/"policy_lock.json"),
        weights_sha256=views.WEIGHTS["dr"][1],development_manifest=views.SPLITS["dr"]["val"][2],
        inference="Score = arithmetic mean of original and horizontal reflection class probabilities. Multiply grade1 and grade3 by prespecified scalars, then renormalize. The original decision is the unmodified single-view argmax.",
        decision="floor mode: if original grade>=2 keep a minimum of 2, if original<=1 cap at 2. up_one mode: permit only original 0->1 or original 2->3 and only when the adjusted candidate recommends that exact grade; otherwise keep original.",
        selection="Use initial QWK floor -0.005, macro F1 floor -0.010, no increase in severe->0/1, at most ceil(0.002*nlow) additional low->severe. Also cap additional true grade0 called >=1 at ceil(0.01*n0) and true grade2 called >=3 at ceil(0.02*n2). Minimize the mean of grade1->0 and grade3->below3 rates; both individual counts must not worsen. Break ties by QWK, macro F1 and prespecified candidate order. A new winner requires strictly lower mean than original.",
        candidates=list(candidates()))


def decide(logits,rule):
    raw=base.softmax(logits)
    original=raw[:,0].argmax(1)
    if rule["mode"]=="original": return raw[:,0],original
    scores=raw.mean(1)*np.array([1,rule["grade1"],1,rule["grade3"],1],dtype=np.float32)
    scores/=scores.sum(1,keepdims=True)
    candidate=scores.argmax(1)
    if rule["mode"]=="floor":
        result=np.where(original>=2,np.maximum(2,candidate),np.minimum(2,candidate))
    elif rule["mode"]=="up_one":
        allowed=((original==0)&(candidate==1))|((original==2)&(candidate==3))
        result=np.where(allowed,candidate,original)
    else: raise ValueError("Unknown ordinal mode")
    return scores,result


def select():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"policy_lock.json").exists(): return
    if not (OUT/"experiment_protocol.json").exists(): views.write_json(OUT/"experiment_protocol.json",protocol(),lock=True)
    spec=base.read(OUT/"experiment_protocol.json")
    base.require(spec["candidates"]==list(candidates()) and spec["previous_lock_sha256"]==views.sha(ROOT/"policy_lock.json"),"Scope changed")
    val=base.arrays("dr","val");y=val["y"];cnt=np.bincount(y,minlength=5)
    outcomes={};scores={};decisions={}
    for rule in spec["candidates"]:
        key=rule["name"];scores[key],decisions[key]=decide(val["logits"],rule)
        m=base.metrics("dr",y,decisions[key]);cm=np.array(m["confusion"])
        m["grade0_up"]=int(cm[0,1:].sum());m["grade2_up"]=int(cm[2,3:].sum())
        if key=="original": ref=m
        eligible=(m["qwk"]>=ref["qwk"]-.005 and m["macro_f1"]>=ref["macro_f1"]-.010
                  and m["severe_to01"]<=ref["severe_to01"] and m["low_to_severe"]<=ref["low_to_severe"]+math.ceil(.002*sum(cnt[:2]))
                  and m["grade0_up"]<=ref["grade0_up"]+math.ceil(.01*cnt[0]) and m["grade2_up"]<=ref["grade2_up"]+math.ceil(.02*cnt[2])
                  and m["grade1_to0"]<=ref["grade1_to0"] and m["grade3_under"]<=ref["grade3_under"])
        rank=[(m["grade1_to0_rate"]+m["grade3_under_rate"])/2,-m["qwk"],-m["macro_f1"]]
        outcomes[key]=dict(rule=rule,metrics=m,eligible=eligible,rank=rank)
    best=min((k for k,v in outcomes.items() if v["eligible"]),key=lambda k:outcomes[k]["rank"])
    if outcomes[best]["rank"][0]>=outcomes["original"]["rank"][0]: best="original"
    views.write_json(OUT/"validation_candidates.json",outcomes)
    save_predictions(OUT/"validation_predictions.csv",val,scores[best],decisions[best],decisions["original"])
    views.write_json(OUT/"policy_lock.json",dict(locked_utc=views.now(),selected=outcomes[best]["rule"],
        eligible_improvement=best!="original",protocol_sha256=views.sha(OUT/"experiment_protocol.json"),
        results_sha256=views.sha(OUT/"validation_candidates.json"),val_scores_sha256=views.sha(OUT/"validation_predictions.csv")),lock=True)
    print(best,outcomes[best]["metrics"],flush=True)


def save_predictions(path,data,scores,decisions,reference):
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8-sig",newline="",dir=path.parent,delete=False) as stream:
        w=csv.writer(stream);w.writerow(["sample_id","group_hash","target","reference_grade","selected_grade",*[f"adjusted_score_{i}" for i in range(5)]])
        for i,sid in enumerate(data["sample_id"]): w.writerow([sid,data["group"][i],int(data["y"][i]),int(reference[i]),int(decisions[i]),*scores[i]])
        tmp=stream.name
    os.replace(tmp,path)


def locked():
    lock=base.read(OUT/"policy_lock.json")
    for name,key in (("experiment_protocol.json","protocol_sha256"),("validation_candidates.json","results_sha256"),
                     ("validation_predictions.csv","val_scores_sha256")):
        base.require(views.sha(OUT/name)==lock[key],"Ordinal selection fingerprint mismatch")
    base.require(lock["eligible_improvement"],"No validation improvement qualified for heldout")
    return lock


def evaluate():
    lock=locked();rule=lock["selected"];hashes={}
    for split in ("test","external_test"):
        previous=base.read(ROOT/f"{split}_report.json");extract=base.read(ROOT/f"{split}_view_extraction.json")
        base.require(datetime.fromisoformat(lock["locked_utc"])<=datetime.fromisoformat(extract["start_utc"]),"Ordinal decision locked too late")
        base.require(not any(any(g.values()) for g in previous["known_development_intersections"].values()),"Development overlap")
        data=base.arrays("dr",split); old,oldhash=base.historical("dr",split,data)
        base.require(oldhash==previous["archived_reference_predictions_sha256"],"Archived comparator mismatch")
        score,decision=decide(data["logits"],rule);original=old.argmax(1);y=data["y"]
        fname=OUT/(split+"_paired_predictions.csv");save_predictions(fname,data,score,decision,original)
        summary=dict(split=split,manifest_sha256=views.SPLITS["dr"][split][2],archived_reference_predictions_sha256=oldhash,
            score_export_sha256=views.sha(fname),known_development_intersections=previous["known_development_intersections"],
            archived=base.metrics("dr",y,original),selected=base.metrics("dr",y,decision),
            paired_delta_ci95=base.paired("dr",y,original,decision,data["group"]),
            original_grading_metrics=base.original_grading_metrics(split,data,old,np.eye(5,dtype=np.float32)[decision]))
        views.write_json(OUT/(split+"_report.json"),summary)
        hashes[split]=views.sha(OUT/(split+"_report.json"))
    views.write_json(OUT/"evaluation_complete.json",dict(completed_utc=views.now(),policy_lock_sha256=views.sha(OUT/"policy_lock.json"),cohort_reports=hashes))


def predict_source(path,reader,rule):
    scores,grade=decide(reader(path)["logits"],rule)
    return dict(use="offline_research",grade=int(grade[0]),class_order=list(range(5)),adjusted_scores=scores[0].tolist(),
                selected=rule,policy_sha256=views.sha(OUT/"policy_lock.json"),
                semantics="Decision includes the one-step ordinal constraint; adjusted scores alone do not determine the reported grade.")


def verify_images():
    rule=locked()["selected"];reader=base.original_view_reader("dr")
    with (OUT/"validation_predictions.csv").open(encoding="utf-8-sig",newline="") as stream:
        exported={r["sample_id"]:r for r in csv.DictReader(stream)}
    examples={}
    for r in views.rows_for("dr","val"): examples.setdefault(r["labels"]["dr_grade"],r)
    matches=[]
    for target,r in sorted(examples.items()):
        path=views.DATA/r["image_paths"][0];base.require(views.sha(path)==r["image_sha256"][0],"Source changed")
        prediction=predict_source(path,reader,rule);ref=exported[r["sample_id"]]
        expected=np.array([float(ref[f"adjusted_score_{i}"]) for i in range(5)])
        matches.append(dict(sample_id=r["sample_id"],target=target,original_image_grade=prediction["grade"],
                       batch_grade=int(ref["selected_grade"]),max_abs_score_error=float(np.max(abs(expected-prediction["adjusted_scores"])))))
    views.write_json(OUT/"original_image_check.json",dict(policy_lock_sha256=views.sha(OUT/"policy_lock.json"),samples=matches,
        matched_decisions=sum(r["original_image_grade"]==r["batch_grade"] for r in matches),
        max_abs_score_error=max(r["max_abs_score_error"] for r in matches)))


def report():
    from scripts import summarize_existing_classifier_round as tables
    lock=locked();complete=base.read(OUT/"evaluation_complete.json")
    base.require(complete["policy_lock_sha256"]==views.sha(OUT/"policy_lock.json"),"Selection fingerprint mismatch")
    val=base.read(OUT/"validation_candidates.json");k=lock["selected"]["name"]
    good=val[k]["metrics"];old=val["original"]["metrics"]
    ok=sum(v["eligible"] for v in val.values());image=base.read(OUT/"original_image_check.json")
    text=["# 糖网分级二次验证探索", "",
          "首轮验证的双视图方案较原模型增加一例重度降为 0–1 级。在读取本轮留出预测之前，第二阶段登记了保留可转诊最低等级的双视图规则及只向上纠正一级的版本。两轮共享 DDR 验证集，属于二次验证探索。",
          "", f"验证时比较 {len(val)} 个预设规则，{ok} 个满足约束，入选 `{k}`。决策锁定时间为 {lock['locked_utc']}（UTC）。",
          "", "原图单视图首先给出分级。再计算原图和水平翻转图两个视图的平均概率，对 1 级和 3 级使用锁定倍率。只向上纠正一级的模式仅允许 0→1、2→3。可转诊保护模式下，原判 2–4 级的图最低保持 2 级，原判 0–1 级最高可改到 2 级。实际使用的模式、倍率见决策锁。",
          "", f"入选模式为 `{lock['selected']['mode']}`，1 级倍率 {lock['selected']['grade1']:g}，3 级倍率 {lock['selected']['grade3']:g}。",
          "", "## DDR 验证选择", ""] + tables.comparison("dr",[("原模型",old),("二次探索入选",good)])
    for split,digest in complete["cohort_reports"].items():
        tables.check_file(OUT/(split+"_report.json"),digest);r=base.read(OUT/(split+"_report.json"))
        tables.check_file(OUT/(split+"_paired_predictions.csv"),r["score_export_sha256"])
        a,b=r["archived"],r["selected"]
        text+=["",f"## {tables.COHORTS['dr'][split]}", "",
               f"1 级降为 0 级由 {a['grade1_to0']} 变为 {b['grade1_to0']} 张，3 级降为 0–2 级由 {a['grade3_under']} 变为 {b['grade3_under']} 张，宏 F1 由 {a['macro_f1']:.4f} 变为 {b['macro_f1']:.4f}。", ""]
        text+=tables.comparison("dr",[("历史原模型",a),("二次探索入选",b)])+[""]+tables.delta_table(r)
        ca,cb=(np.array(m["confusion"]) for m in (a,b))
        text+=["", f"真实 0 级被升高由 {ca[0,1:].sum()} 变为 {cb[0,1:].sum()} 张；真实 2 级被升为 3–4 级由 {ca[2,3:].sum()} 变为 {cb[2,3:].sum()} 张。"]
    text+=["", f"原图入口按五个真实级别各取一张验证图，与缓存批量结果一致 {image['matched_decisions']}/5 张，最大调整分数差为 {image['max_abs_score_error']:.6f}。",
           "", "差值区间采用 1,000 次图像内容组配对自助抽样。DDR、IDRiD 缺少患者标识，IDRiD 已有固定历史回归。本次仅作离线研究。", "",
           "单张原图离线命令为以下形式。", "", "```bash",'python scripts/research_dr_referral_floor.py predict --image "原图路径"',"```",""]
    (OUT/"实验结果.md").write_text("\n".join(text),encoding="utf-8")
    views.write_json(OUT/"artifact_manifest.json",dict(code_sha256=views.sha(Path(__file__)),files={
        p.name:views.sha(p) for p in OUT.iterdir() if p.is_file() and p.name!="artifact_manifest.json"}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["select","evaluate","verify-images","report","predict"])
    p.add_argument("--image",type=Path);a=p.parse_args()
    if a.action=="select": select()
    elif a.action=="evaluate": evaluate()
    elif a.action=="verify-images": verify_images()
    elif a.action=="report": report()
    else:
        if not a.image: p.error("predict requires --image")
        print(json.dumps(predict_source(a.image,base.original_view_reader("dr"),locked()["selected"]),ensure_ascii=False))
