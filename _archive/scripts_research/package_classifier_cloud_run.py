"""Pack the two locked local experiments and required public holdout photos without image decoding."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs"/"cloud_transfer_m12_1122"
EVAL=ROOT/"02_模型训练与评估"
RESULT=EVAL/"08_其余五模型优化_20261122"
BASE=EVAL/"05_本轮训练结果_20261122"/"existing_models"
MROOT=ROOT/"data/prepared/source_isolated/strict_cross_domain/tasks"
SPLITS={"eyeq":("image_quality/EyeQ",["train","val","test"],"quality_grade"),
        "ddr":("dr_grading/DDR",["train","val","test"],"dr_grade"),"idrid":("dr_grading/IDRiD",["external_test"],"dr_grade")}
CODE=["scripts/research_classifier_views.py","scripts/optimize_existing_classifiers.py","scripts/summarize_existing_classifier_round.py",
      "scripts/research_dr_referral_floor.py","scripts/evaluate_eyeq_operating_point.py","scripts/screening_compare_legacy.py",
      "scripts/screening_evaluation.py","scripts/manifest_dataset.py","training/classification.py","training/fundus_preprocessing.py",
      "training/metrics.py","training/__init__.py"]


def checksum(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def archive(name,paths):
    target=OUT/(name+".zip")
    paths=sorted(set(paths))
    if not (target.exists() and target.with_suffix(".json").exists()):
        with tempfile.NamedTemporaryFile(dir=OUT,delete=False) as f: tmp=Path(f.name)
        with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as z:
            for p in paths:
                if not p.is_relative_to(ROOT) or not p.is_file(): raise ValueError(f"Missing expected bundle input {p}")
                z.write(p,p.relative_to(ROOT).as_posix())
        os.replace(tmp,target)
        meta={"file":target.name,"bytes":target.stat().st_size,"sha256":checksum(target),"members":len(paths)}
        target.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf8")
    print(target, target.stat().st_size,flush=True)


def rows(source,split):
    return [json.loads(s) for s in (MROOT/SPLITS[source][0]/(split+".jsonl")).read_bytes().splitlines()]


def development():
    paths=[ROOT/f for f in CODE]
    for source,(folder,splits,label) in SPLITS.items():
        paths.extend(MROOT/folder/(s+".jsonl") for s in splits)
        if "val" in splits:
            sample={}
            for r in rows(source,"val"):
                sample.setdefault(r["labels"][label],ROOT/"data"/r["image_paths"][0])
            paths.extend(sample.values())
    for name in ("model1_dr","model2_quality"):
        paths.extend(p for p in (RESULT/name).rglob("*") if p.is_file() and p.suffix in (".csv",".json",".npz") and
                     ("_view_extraction" not in p.name or p.name.startswith(("train_","val_"))))
    paths.extend(p for p in (ROOT/"runs/existing_classifier_views_1122").rglob("*") if p.is_file() and p.name.startswith(("train.","val.")))
    for name in ("test_predictions.jsonl","test_predictions.meta.json","threshold.json","val_metrics.json"):
        paths.append(BASE/"eyeq"/name)
    for split,base in (("test",ROOT/"cloud_backup/20261114_DDR_crop_letterbox/reports/evaluation/best"),("external_test",BASE/"dr_idrid")):
        paths.extend(base/n for n in (f"predictions_{split}.jsonl",f"predictions_{split}.meta.json",f"evaluation_{split}.json"))
    archive("m12_dev_locked",paths)


def raw(source):
    part=[];size=0;index=1;limit=512*1024**2
    for r in rows(source,SPLITS[source][1][-1]):
        path=ROOT/"data"/r["image_paths"][0];n=path.stat().st_size
        if size+n>limit and part:
            archive(f"m12_{source}_raw_{index:02d}",part);part=[];size=0;index+=1
        size+=n;part.append(path)
    if part: archive(f"m12_{source}_raw_{index:02d}",part)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("part",choices=("dev","eyeq","ddr","idrid","code"))
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.part=="dev": development()
    elif a.part=="code": archive("m12_final_code",[ROOT/f for f in CODE]+[ROOT/"scripts/launch_locked_classifiers_cloud.py"])
    else: raw(a.part)
