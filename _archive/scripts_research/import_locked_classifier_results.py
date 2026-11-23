"""Verify a cloud result receipt and import its files without rerunning the classifiers."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
BASE=PurePosixPath("02_模型训练与评估/08_其余五模型优化_20261122")
TASKS={"quality":"model2_quality","dr":"model1_dr"}


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def require(ok,msg):
    if not ok: raise ValueError(msg)


def verify_tree(tree):
    read=lambda p:json.loads((tree/p).read_bytes())
    done=read("evaluation_complete.json")
    require(sha(tree/"policy_lock.json")==done["policy_lock_sha256"],"Policy lock differs from evaluation receipt")
    for split,digest in done["cohort_reports"].items():
        p=tree/f"{split}_report.json";require(sha(p)==digest,"Cohort report differs")
        require(sha(tree/f"{split}_paired_predictions.csv")==read(p.name)["score_export_sha256"],"Exported scores differ")
    require((tree/"实验结果.md").is_file() and (tree/"original_image_check.json").is_file(),"Missing report or raw-image verification")


def verify_and_import(task,archive,receipt,check_only):
    spec=json.loads(receipt.read_bytes());prefix=BASE/TASKS[task];dest=ROOT.joinpath(*prefix.parts)
    require(archive.name==spec["name"]==f"m12_{task}_results.tar.gz" and archive.stat().st_size==spec["bytes"] and
            sha(archive)==spec["sha256"],"Whole archive identity differs")
    files=spec["members"];pairs=[]
    with tempfile.TemporaryDirectory(prefix="m12_import_",dir=dest.parent) as scratch:
        root=Path(scratch)
        with tarfile.open(archive,"r:gz") as package:
            seen=set()
            for member in package:
                n=PurePosixPath(member.name)
                require(n.as_posix()==member.name and not n.is_absolute() and ".." not in n.parts and
                        ":" not in member.name and "\\" not in member.name and member.isfile() and member.name in files and member.name not in seen,
                        "Unsafe or unregistered archive member")
                seen.add(member.name)
                if n.is_relative_to(prefix): rel=n.relative_to(prefix)
                else:
                    require(n.parent==PurePosixPath("scripts") and n.suffix==".py","Unexpected code member")
                    rel=PurePosixPath("cloud_source")/n.name
                require(rel.parts and files[member.name]["bytes"]==member.size,"Wrong entry length")
                dst=root.joinpath(*rel.parts);dst.parent.mkdir(exist_ok=True,parents=True)
                with package.extractfile(member) as src,dst.open("wb") as f:shutil.copyfileobj(src,f)
                require(sha(dst)==files[member.name]["sha256"],"Archive member fingerprint differs")
                pairs.append((dst,dest.joinpath(*rel.parts)))
            require(seen==set(files),"Incomplete tar members")
        verify_tree(root)
        if task=="dr":verify_tree(root/"ordinal_refinement")
        for src,dst in pairs:
            if dst.exists():require(sha(dst)==sha(src),f"Existing local artifact has different bytes: {dst}")
        if not check_only:
            for src,dst in pairs:
                if dst.exists():continue
                dst.parent.mkdir(exist_ok=True,parents=True);os.replace(src,dst)
            path=dest/"cloud_archive_receipt.json"
            if path.exists():require(sha(path)==sha(receipt),"Local cloud receipt differs")
            else:shutil.copyfile(receipt,path)
    print(json.dumps({"task":task,"members":len(pairs),"archive_sha256":spec["sha256"],"imported":not check_only,"destination":str(dest)},ensure_ascii=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("task",choices=TASKS);p.add_argument("archive",type=Path)
    p.add_argument("receipt",type=Path);p.add_argument("--check-only",action="store_true");args=p.parse_args()
    verify_and_import(args.task,args.archive,args.receipt,args.check_only)
