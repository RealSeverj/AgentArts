"""Build the local Model 5 offline evaluation report from completed or partial archives."""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1] / "02_模型训练与评估"
BASE = ROOT / "05_本轮训练结果_20261122" / "grouped_partial"
ROUND = ROOT / "07_模型5云端优化结果_20261122"
BASE_SHA = "d518f2eac265cbab14763322c4b49a94eb9917a6880ef14ee256624a86f20a00"
VAL_SHA = "9f21c8e0cc337d6cf820f44329452fd3987fcf72487f1add227bbc8051fb379a"
ARMS = {
    "baseline": ("历史 ViT-B 全图", None),
    "global_vitl": ("ViT-L 全图", "large_full"),
    "dual_vitb": ("ViT-B 全图与视盘局部", "base_disc"),
    "global_vitb_letterbox": ("ViT-B 392 裁边补边", "geometry_base"),
    "global_vitl_letterbox": ("ViT-L 392 裁边补边", "geometry_large"),
    "global_vitb_letterbox_518": ("ViT-B 518 裁边补边", "geometry_518"),
    "global_vitl_letterbox_dr_transfer": ("ViT-L 392 糖网权重迁移", "geometry_retina"),
    "mean_probability": ("两模型等权概率平均", "ensemble"),
}
LABELS = ["RVO","LS","DN","AMD","CSR","OPTIC_DISC_CUPPING","ODE","ODP","MYOPIC_FUNDUS","TSLN","ABNORMAL"]
ZH = ["视网膜静脉阻塞","激光瘢痕","玻璃膜疣","年龄相关性黄斑变性","中心性浆液性脉络膜视网膜病变",
      "视盘凹陷","视盘水肿","视盘苍白","近视性眼底","豹纹状眼底","总体异常"]
COHORT = {"test": ("RFMiD",615,"5ede17f3f694600ee8eb46688546c2806044f05929dd3e49209fc84f80abe1b4"),
          "external_test": ("RFMiD2",848,"26b81c4be057558a6c590c35623c928da21f7a021ac08b441d50ec80d19bb103")}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else None


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(predicate, message):
    if not predicate:
        raise ValueError(message)


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def duration(seconds):
    if seconds is None:
        return "—"
    s=round(seconds)
    return f"{s//3600}时{s//60%60:02}分{s%60:02}秒"


def bounds(metric):
    interval=metric["paired_delta_ci95"]["interval"]
    return interval if interval is not None else [None,None]


def atom(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        temp=tmp.name
    os.replace(temp,path)


def run_dirs(args):
    dirs={name: (args.baseline if d is None else args.round/d) for name,(_,d) in ARMS.items()}
    # 518px's final directory can be named differently on different hosts.
    for meta in args.round.glob("*/run.json"):
        identity=read(meta).get("identity",{})
        if identity.get("image_size")==518 and "vit_base" in identity.get("backbone",""):
            dirs["global_vitb_letterbox_518"]=meta.parent
    return dirs


def candidate(path, name, selection):
    h=read(path/"history.json") or []
    check(isinstance(h,list) and [r["epoch"] for r in h]==list(range(1,len(h)+1)),
          f"异常的轮次记录 {path}")
    c=read(path/"complete.json")
    run=read(path/"run.json") or {}
    lock=read(path/"thresholds.json")
    n=len(h)
    expected=selection.get("results",{}).get(name,{})
    evidence=None
    missing=[]
    if c:
        planned=run.get("identity",{}).get("settings",{}).get("epochs")
        check(c["epochs"]==30 and 1 <= c["best_epoch"] <= 30 and (planned is None or c["epochs"]==planned),
              f"异常的完成回执 {path}")
        if expected:
            check(c["checkpoint_sha256"]==expected["checkpoint_sha256"],f"选择回执与权重不符 {path}")
        if n==c["epochs"]:
            ap=h[c["best_epoch"]-1]["val"]["macro_ap"]
            check(math.isclose(ap,c["best_validation_macro_ap"],abs_tol=1e-6),f"最佳验证 AP 不一致 {path}")
        if name=="baseline":
            check(c["checkpoint_sha256"]==BASE_SHA,"历史基线 SHA 不符")
        evidence=c["checkpoint_sha256"]
    if name=="mean_probability":
        if expected: evidence=expected["checkpoint_sha256"]
        wt=path/"ensemble_recipe.json"
    else:
        wt=path/"best.pt"
    wt_ok=False
    if evidence and wt.is_file():
        check(sha(wt)==evidence,f"归档文件 SHA 不符 {wt}")
        wt_ok=True
    if lock:
        if evidence: check(lock["checkpoint_sha256"]==evidence,f"阈值对应的权重不符 {path}")
        check(lock["labels"]==LABELS and lock["val_manifest_sha256"]==VAL_SHA,f"阈值验证集不符 {path}")
        check(all((v["min_recall"],v["max_fpr"])==(.9,.1) for v in lock["per_label"].values()),f"阈值目标不符 {path}")
    if expected and lock:
        check(expected["thresholds_sha256"]==sha(path/"thresholds.json"),f"选择后阈值有变更 {path}")
    for f in (("thresholds.json",) if name=="mean_probability" else ("history.json","complete.json","run.json","thresholds.json")):
        if not (path/f).is_file(): missing.append(f)
    if not wt_ok: missing.append(wt.name)
    if name=="mean_probability": state="融合配方已核验" if wt_ok else "融合配方未核验"
    elif c and not missing and n==c["epochs"]: state=f"{n} 轮完成，权重已核验"
    elif c: state="有完成回执，归档待补"
    elif h: state=f"已见 {n} 轮，缺完成回执"
    else: state="未见训练回执"
    decision=selection.get("geometry_candidates",{}).get(name,{})
    if decision.get("status")=="incomplete":
        state+="；决策时产物未齐"
    elif decision.get("status")=="not_supplied":
        state+="；决策时未送选"
    t=[r.get("elapsed_seconds") for r in h]
    elapsed=sum(t) if t and all(v is not None for v in t) else None
    met=(sum(r["status"]=="met" for r in lock["per_label"].values()) if lock else None)
    if expected:
        ap=expected["macro_ap"]
        op=expected["validation_working_point"]
        check((op["min_recall"],op["max_fpr"])==(.9,.1),f"选择阈值目标不符 {path}")
        if met is not None: check(met==op["labels_meeting_both"],f"工作点计数不符 {path}")
        met=op["labels_meeting_both"]
    else:
        ap=c["best_validation_macro_ap"] if c else None
    return dict(history=h,elapsed=elapsed,n=n,ap=ap,met=met,evidence=evidence,weight=wt,verified=wt_ok,
                status=state,missing=missing,epoch=c["best_epoch"] if c else None)


def pair_input(args, selection_file, selection):
    if not selection: return None
    if args.paired:
        paths=[args.paired]
    else:
        paths=sorted({*args.round.rglob("*paired*.json"),*args.round.rglob("*regression*.json"),*args.round.rglob("*回归*.json")})
    found=[]
    for p in paths:
        r=read(p)
        if not isinstance(r,dict) or not {"winner","score_identities","splits","selection_sha256"}<=r.keys(): continue
        if r["selection_sha256"]!=sha(selection_file):
            if args.paired: raise ValueError("指定的配对回归不属于本次最终选择")
            continue
        found.append((p,r))
    check(len({sha(p) for p,_ in found})<=1,"存在多个不同的配对回归，请用 --paired 明确文件")
    if not found: return None
    _,p=found[0]
    winner=selection["selected"]
    check(p["winner"]==winner and p["score_identities"]["selected"]["checkpoint_sha256"]==selection["results"][winner]["checkpoint_sha256"]
          and p["score_identities"]["baseline"]["checkpoint_sha256"]==BASE_SHA,"回归使用的权重与选择不符")
    for role,name in (("baseline","baseline"),("selected",winner)):
        entry=selection["results"][name]
        check(p["score_identities"][role]["val_predictions_sha256"]==entry["val_predictions_sha256"]
              and p["threshold_files_sha256"][role]==entry["thresholds_sha256"],"回归使用的阈值或验证预测已变更")
    for split,(src,n,manifest) in COHORT.items():
        v=p["splits"][split]
        check((v["dataset"],v["images"],v["manifest_sha256"])==(src,n,manifest)
              and v["macro_ap"]["comparable_10"]["labels"]==LABELS[:10],"配对回归测试来源或可比标签集合不符")
    return p


CSV_FIELDS=["数据集","图像数","标签","名称","标注状态","阳性数","阴性数","历史阈值","入选阈值",
            "历史AP","入选AP","AP差","AP差95%下限","AP差95%上限","历史敏感度","入选敏感度","敏感度差",
            "敏感度差95%下限","敏感度差95%上限","历史误报率","入选误报率","误报率差","误报率差95%下限",
            "误报率差95%上限","历史TP","入选TP","历史FN","入选FN","历史FP","入选FP","历史TN","入选TN","入选同时达标"]


def regression(paired):
    csvbuf=io.StringIO(newline="")
    writer=csv.writer(csvbuf)
    writer.writerow(CSV_FIELDS)
    if not paired: return csvbuf.getvalue(), ["尚无与最终选择回执匹配的同图配对回归。逐标签文件目前仅含列名。"]
    lines=["下表只汇总内外测均有真值的十项。差值为入选方案减历史基线。","",
           "| 队列 | 图像 / 关联组 | 历史宏 AP | 入选宏 AP | 差值 | 差值 95% 区间 | 入选同时达标 |",
           "|---|---:|---:|---:|---:|---|---:|"]
    for split,(src,n,_) in COHORT.items():
        v=paired["splits"][split]
        met=0
        for i,label in enumerate(LABELS):
            if label not in v["per_label"]:
                check(split=="external_test" and label=="ABNORMAL","回归缺少有标注的类别")
                writer.writerow([src,n,label,ZH[i],"全队列标签未知"]+[""]*(len(CSV_FIELDS)-5))
                continue
            row=v["per_label"][label]
            ap,sens,spec,tp,fp=[row[k] for k in ("ap","sensitivity","specificity","tp","fp")]
            pos,neg=row["positive"],row["negative"]
            rates=[fp[k]/neg if neg else None for k in ("baseline","selected")]
            ok=(sens["selected"] is not None and rates[1] is not None and sens["selected"]>=.9 and rates[1]<=.1)
            if i<10: met+=ok
            l,u=bounds(spec)
            data=[src,n,label,ZH[i],"已知",pos,neg,row["thresholds"]["baseline"],row["thresholds"]["selected"],
                  ap["baseline"],ap["selected"],ap["delta"],*bounds(ap),sens["baseline"],sens["selected"],sens["delta"],
                  *bounds(sens),*rates,-spec["delta"] if spec["delta"] is not None else None,
                  -u if u is not None else None,-l if l is not None else None,
                  tp["baseline"],tp["selected"],pos-tp["baseline"],pos-tp["selected"],fp["baseline"],fp["selected"],
                  neg-fp["baseline"],neg-fp["selected"],"是" if ok else "否"]
            check(len(data)==len(CSV_FIELDS),"CSV 行形状错误")
            writer.writerow([round(x,8) if isinstance(x,float) else x for x in data])
        m=v["macro_ap"]["comparable_10"]; lo,hi=bounds(m)
        lines.append(f"| {src} | {n} / {v['groups']} | {fmt(m['baseline'])} | {fmt(m['selected'])} | {fmt(m['delta'])} | [{fmt(lo)}, {fmt(hi)}] | {met}/10 |")
    lines.extend(["",f"95% 区间来自 {paired['paired_bootstrap_draws']:,} 次患者或内容关联组配对重采样，条件为模型和验证阈值固定。"
                  "RFMiD2 的总体异常标签全部未知。RFMiD 内测的总体异常单列在逐标签文件中。"])
    inner=paired["splits"]["test"]
    all_labels=inner["macro_ap"]["all_11"]
    low,high=bounds(all_labels)
    met_labels={role:[ZH[LABELS.index(label)] for label,row in inner['per_label'].items()
                      if row['sensitivity'][role]>=.9 and row['fp'][role]/row['negative']<=.1]
                for role in ('baseline','selected')}
    lines.extend(['',f"RFMiD 全部 11 项宏 AP 由 {fmt(all_labels['baseline'])} 变为 {fmt(all_labels['selected'])}，"
                  f"差值 {fmt(all_labels['delta'])}，95% 区间 [{fmt(low)}, {fmt(high)}]。固定验证阈值后，同时达标数由 "
                  f"{len(met_labels['baseline'])}/11 变为 {len(met_labels['selected'])}/11。入选方案达标项为{'、'.join(met_labels['selected']) or '无'}。"])
    return csvbuf.getvalue(),lines


def plot(root, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.4,4.6),layout="constrained")
    for name,row in stats.items():
        if row["history"]:
            ax.plot([r["epoch"] for r in row["history"]],[r["val"]["macro_ap"] for r in row["history"]],
                    label=name,ls="--" if name=="baseline" else "-")
    ax.set(xlabel="Completed epoch",ylabel="RFMiD validation macro AP (11 labels)",xlim=(1,30))
    ax.grid(axis="y",alpha=.25);ax.legend(fontsize=8)
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=170);plt.close(fig)
    atom(root/"验证宏AP曲线.png",buf.getvalue())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round",type=Path,default=ROUND);p.add_argument("--baseline",type=Path,default=BASE)
    p.add_argument("--selection",type=Path);p.add_argument("--paired",type=Path);p.add_argument("--output",type=Path)
    p.add_argument("--plot",action="store_true")
    p.add_argument("--started-utc",help="Verified ISO-8601 run window start with timezone")
    p.add_argument("--finished-utc",help="Verified ISO-8601 run window end with timezone")
    args=p.parse_args()
    output=args.output or args.round
    selection_file=args.selection or args.round/"final_selection.json"
    if not args.selection and not selection_file.exists():
        matches=list(args.round.rglob("final_selection.json"))
        check(len(matches)<=1,"找到多份最终选择，请明示 --selection")
        if matches: selection_file=matches[0]
    selection=read(selection_file) or {}
    if selection:
        check(selection.get("format")=="model5_final_validation_selection_v1" and selection.get("selection_split")=="val"
              and selection["validation_manifest_sha256"]==VAL_SHA and selection["selection_metric"]=="validation_11_label_macro_ap",
              "最终选择不是已核对的验证集协议")
        win=selection["selected"];results=selection["results"]
        check(win in ARMS and set(results)<=set(ARMS) and results["baseline"]["checkpoint_sha256"]==BASE_SHA
              and results[win]["macro_ap"]==max(v["macro_ap"] for v in results.values()),"最终选择回执候选或指标不符")
    elif args.selection: p.error("指定的最终选择文件不存在")
    dirs=run_dirs(args)
    stats={}
    for name,path in dirs.items():
        if (name in ("global_vitb_letterbox_518","global_vitl_letterbox_dr_transfer") and not path.exists()
                and name not in selection.get("results",{}) and
                selection.get("geometry_candidates",{}).get(name,{}).get("status","not_supplied")=="not_supplied"):
            continue
        stats[name]=candidate(path,name,selection)
    paired=pair_input(args,selection_file,selection)
    if args.paired and not paired: p.error("指定的配对回归未通过完整性核对")
    csv_text,comparison=regression(paired)
    lines=["# 模型 5 眼底多异常训练与回归结果",""]
    if selection:
        s=stats[win]
        lines.append(f"最终验证集选择了 **{ARMS[win][0]}**，11 项宏平均 AP 为 **{fmt(s['ap'])}**。")
        lines.extend(["",f"选择回执记录的{'融合配方' if win=='mean_probability' else '权重'} SHA-256 为 `{results[win]['checkpoint_sha256']}`。"])
        if s["verified"]:
            lines.append(f"已计算本地文件 `{s['weight']}` 的 SHA-256 并核对一致。")
        else: lines.append("该文件尚未完成本地归档校验。")
        if win=="mean_probability" and s["verified"]:
            recipe=read(s["weight"])
            for m in recipe["members"]:
                ms=stats[m["role"]]
                check(m["checkpoint_sha256"]==results[m["role"]]["checkpoint_sha256"],"融合成员与最终回执不一致")
                lines.append(f"成员 `{m['role']}` 占 {m['weight']:.0%}，权重 SHA-256 为 `{m['checkpoint_sha256']}`，本地{'已核验' if ms['verified'] else '待回收核验'}。")
    else: lines.append("本地归档尚无最终选择回执，当前无法确认本轮入选权重。")
    lines.extend(["","本报告用于离线研究。候选仅在固定的 634 张 RFMiD 验证图上比较。每个标签在验证集锁定阈值，"
                  "同时达标要求敏感度 ≥90%、误报率 ≤10%。RFMiD 和 RFMiD2 均用于过往项目评估，本轮外测属于历史留出集回归。",
                  "","## 候选与计时","","| 候选 | 本地归档状态 | 已见轮次 | 最佳轮 | 验证宏 AP | 验证同时达标 | 各轮训练与验证合计 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for name,s in stats.items():
        lines.append(f"| {ARMS[name][0]} | {s['status']} | {s['n'] if s['history'] else '—'} | {s['epoch'] or '—'} | {fmt(s['ap'])} | "
                     f"{str(s['met'])+'/11' if s['met'] is not None else '—'} | {duration(s['elapsed'])} |")
    lines.extend(["","逐轮时间来自 history.json，覆盖实际归档的训练和验证轮次。并行候选的时长有重叠；"
                  "写盘、传输及等权融合的耗时另计。"])
    if args.started_utc or args.finished_utc:
        if not args.started_utc or not args.finished_utc: p.error("必须同时提供已核实的起止时间")
        start,end=(datetime.fromisoformat(v.replace("Z","+00:00")) for v in (args.started_utc,args.finished_utc))
        check(start.tzinfo and end.tzinfo and end>=start,"计时窗口或时区无效")
        lines.append(f"已核实的运行窗口为 {start.isoformat()} 至 {end.isoformat()}，墙钟用时 {duration((end-start).total_seconds())}。")
    else: lines.append("归档未提供已核实的统一起止时间，整轮墙钟耗时待补。")
    if args.plot and any(s["history"] for s in stats.values()):
        lines.extend(["","![各候选逐轮验证宏平均AP](验证宏AP曲线.png)"])
    lines.extend(["","## 历史留出集配对回归","",*comparison,"","逐标签支持数、固定阈值、混淆计数与指标差值见《逐标签回归.csv》。"])
    pending=[f"{ARMS[n][0]}缺少 {'、'.join(s['missing'])}" for n,s in stats.items() if s["missing"]]
    if pending:
        lines.extend(["","## 归档待补","",*["- "+x for x in pending]])
    output.mkdir(parents=True,exist_ok=True)
    if args.plot and any(s["history"] for s in stats.values()): plot(output,stats)
    atom(output/"逐标签回归.csv",csv_text.encode("utf-8-sig"))
    atom(output/"训练结果汇总.md",("\n".join(lines)+"\n").encode("utf-8"))
    print(json.dumps(dict(selection=selection.get("selected"),paired=bool(paired),output=str(output)),ensure_ascii=False))


if __name__=="__main__":
    main()
