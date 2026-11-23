"""Linux launcher for independent EyeQ GPU1 and DDR GPU0 locked historical regression."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

TRANSFER=Path("/dev/shm/other_model_runs/m12_transfer")
PROJECT=TRANSFER.parent/"m12_project"
RESULT=Path("02_模型训练与评估/08_其余五模型优化_20261122")
ACTIVE=Path("02_模型训练与评估/01_在用模型")
TASKS={
    "quality":("model2_quality",1,"quality_best.pt",ACTIVE/"02_眼底图像质控"/"quality_best.pt",571555732,
               "fa5ee22b483b09c8dec9050455b5e6010d51c2ecced17d0a55982657bdd66f8f",["m12_eyeq_raw_01"]),
    "dr":("model1_dr",0,"dr_best.pt",ACTIVE/"01_糖网五级分级"/"best.pt",1617636556,
          "a85acf367a34361986739e93030bc5c51a390ddd33c63d49da883abf1562d4ab",["m12_idrid_raw_01",*[f"m12_ddr_raw_{n:02d}" for n in range(1,11)]])}


def log(event,**args):
    print(json.dumps(dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),event=event,**args),ensure_ascii=False),flush=True)


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""): h.update(b)
    return h.hexdigest()


def write_json(path,value):
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,delete=False) as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.write("\n");tmp=f.name
    os.replace(tmp,path)


def atomic_extract(zip_file,path,project):
    with zipfile.ZipFile(zip_file) as z:
        for member in z.infolist():
            dest=(project/member.filename).resolve()
            if member.is_dir() or not dest.is_relative_to(project.resolve()) or (member.external_attr>>16)&0o170000==0o120000:
                raise ValueError(f"Unsafe ZIP member {member.filename}")
            dest.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=dest.parent,delete=False) as dst:
                tmp=Path(dst.name)
                try:
                    with z.open(member) as src: shutil.copyfileobj(src,dst,1<<20)
                except BaseException:
                    tmp.unlink(missing_ok=True);raise
            os.replace(tmp,dest)
    return path


def until(call,deadline,what):
    log("waiting",asset=what)
    while time.time()<deadline:
        value=call()
        if value: return value
        time.sleep(4)
    raise TimeoutError(f"Deadline reached waiting for {what}")


def bundle(name,deadline,state):
    marker=state/(name+".done.json");zpath=TRANSFER/(name+".zip");receipt=TRANSFER/(name+".json")
    def available():
        if not receipt.is_file(): return False
        try: spec=json.loads(receipt.read_bytes())
        except (OSError,ValueError): return False
        if spec["file"]!=zpath.name: raise ValueError(f"Wrong zip identity {receipt}")
        if marker.exists() and json.loads(marker.read_bytes())==spec: return spec
        return spec if zpath.is_file() and zpath.stat().st_size==spec["bytes"] else False
    spec=until(available,deadline,name)
    with (state/(name+".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if marker.exists() and json.loads(marker.read_bytes())==spec: return
        if digest(zpath)!=spec["sha256"]: raise ValueError(f"Transfer SHA-256 differs for {zpath}")
        atomic_extract(zpath,marker,PROJECT);write_json(marker,spec);log("unpacked",asset=name,bytes=spec["bytes"])


def dependencies(env,state):
    versions={"timm":"1.0.22","numpy":"1.26.4","Pillow":"11.3.0"}
    private=PROJECT/".m12_runtime";private.mkdir(exist_ok=True)
    env["PYTHONPATH"]=str(private)+os.pathsep+str(PROJECT)+os.pathsep+env.get("PYTHONPATH","")
    program=('from importlib.metadata import distributions; import json; '
             'print(json.dumps({d.metadata["Name"].lower():d.version for d in reversed(list(distributions())) if d.metadata["Name"]}))')
    with (state/"runtime.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        installed=json.loads(subprocess.check_output([sys.executable,"-c",program],env=env))
        need=[f"{p}=={v}" for p,v in versions.items() if installed.get(p.lower())!=v]
        if need:
            log("install_local_dependencies",packages=need)
            subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check","--no-deps","--target",str(private),*need],env=env,check=True)
    check='import torch,torchvision,timm,numpy,PIL; assert torch.cuda.is_available(); print(torch.__version__,torchvision.__version__,timm.__version__,numpy.__version__,PIL.__version__,torch.cuda.get_device_name())'
    subprocess.run([sys.executable,"-c",check],env=env,check=True)


def archive(task,folder):
    artifacts=TRANSFER/"artifacts";artifacts.mkdir(exist_ok=True);target=artifacts/f"m12_{task}_results.tar.gz"
    out=PROJECT/RESULT/folder
    if not all((out/f).is_file() for f in ("evaluation_complete.json","artifact_manifest.json","实验结果.md","original_image_check.json")):
        raise ValueError("Incomplete research result")
    if task=="dr" and not (out/"ordinal_refinement/artifact_manifest.json").is_file(): raise ValueError("Missing ordinal results")
    files=[f for f in out.rglob("*") if f.is_file()]
    files += [PROJECT/"scripts"/f for f in ("research_classifier_views.py","optimize_existing_classifiers.py","research_dr_referral_floor.py","summarize_existing_classifier_round.py")]
    members={p.relative_to(PROJECT).as_posix():{"bytes":p.stat().st_size,"sha256":digest(p)} for p in sorted(files)}
    tmp=target.with_suffix(".partial")
    with tarfile.open(tmp,"w:gz",compresslevel=1) as tar:
        for name in members: tar.add(PROJECT/name,arcname=name,recursive=False)
    os.replace(tmp,target);write_json(artifacts/f"m12_{task}_results.json",{"name":target.name,"bytes":target.stat().st_size,
        "sha256":digest(target),"members":members});log("completed",task=task,path=str(target),bytes=target.stat().st_size)


def run(task,args):
    folder,gpu,remote_name,relative,size,sha,required=TASKS[task]
    PROJECT.mkdir(parents=True,exist_ok=True);state=PROJECT/".m12_state";state.mkdir(exist_ok=True)
    with (state/(task+".run.lock")).open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.time()+args.deadline_minutes*60
        for name in ["m12_dev_locked","m12_final_code",*required]: bundle(name,deadline,state)
        source=TRANSFER/remote_name
        until(lambda: source.is_file() and source.stat().st_size==size,deadline,remote_name)
        if digest(source)!=sha: raise ValueError("Uploaded checkpoint differs")
        target=PROJECT/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if target.resolve()!=source.resolve() and digest(target)!=sha: raise ValueError("A different checkpoint occupies model path")
        else: target.symlink_to(source)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
                                         PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1")
        dependencies(env,state)
        commands=[["research_classifier_views.py",task,"heldout","--batch",str(args.batch),"--workers",str(args.workers)],
                  ["optimize_existing_classifiers.py",task,"evaluate"],["optimize_existing_classifiers.py",task,"verify-images"]]
        if task=="dr":
            commands += [["research_dr_referral_floor.py",a] for a in ("evaluate","verify-images","report")]
        commands += [["summarize_existing_classifier_round.py",task]]
        out=PROJECT/RESULT/folder
        with (out/"cloud_execution.log").open("a",encoding="utf-8") as record:
            for script,*argv in commands:
                log("command",task=task,args=[script,*argv])
                record.write(json.dumps({"utc":time.time(),"command":[script,*argv]})+"\n");record.flush()
                subprocess.run([sys.executable,"-u","-B",str(PROJECT/"scripts"/script),*argv],cwd=PROJECT,env=env,
                               stdout=record,stderr=subprocess.STDOUT,check=True)
        archive(task,folder)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("task",choices=["all","quality","dr"])
    p.add_argument("--batch",type=int,default=8);p.add_argument("--workers",type=int,default=3)
    p.add_argument("--deadline-minutes",type=int,default=210)
    args=p.parse_args()
    if args.task=="all":
        jobs=[subprocess.Popen([sys.executable,"-u",__file__,t,*sys.argv[2:]]) for t in ("quality","dr")]
        codes=[j.wait() for j in jobs];sys.exit(int(any(codes)))
    run(args.task,args)
