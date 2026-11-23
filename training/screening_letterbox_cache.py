"""Build content- and recipe-bound lossless caches for the 392px RFMiD geometry control."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tarfile
import tempfile
import time

import numpy as np
from PIL import Image, __version__ as pillow
from training import fundus_preprocessing as fundus


def sha(data):
    return hashlib.sha256(data).hexdigest()


def recipe():
    return dict(size=[392,392], preprocessing=fundus.preprocessing_spec({"preprocessing":{"mode":fundus.MODE}}),
                pixels="RGB uint8 after deterministic geometry and before training augmentation",
                pillow=pillow,numpy=np.__version__,geometry_code_sha256=sha(Path(fundus.__file__).read_bytes()))


def recipe_sha():
    return sha(json.dumps(recipe(),sort_keys=True,separators=(",",":")).encode())


def image_path(cache, policy, source_sha):
    if len(source_sha)!=64 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("Invalid source image fingerprint")
    return cache/"images"/policy/source_sha[:2]/(source_sha+".png")


def atomic(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix=".tmp",delete=False) as stream:
        stream.write(raw)
        temp=Path(stream.name)
    try:
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def geometry(path,source_sha):
    raw=path.read_bytes()
    if sha(raw)!=source_sha:
        raise ValueError(f"Source image fingerprint differs: {path}")
    with Image.open(io.BytesIO(raw)) as im:
        return fundus.apply_preprocessing(im,392,{"mode":fundus.MODE})


def make_image(job):
    root,cache,policy,src,sid,source_sha=job
    path=(root/src).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source path escapes root")
    result=geometry(path,source_sha)
    buf=io.BytesIO()
    result.save(buf,format="PNG",compress_level=1)
    raw=buf.getvalue()
    dst=image_path(cache,policy,source_sha)
    atomic(dst,raw)
    return dict(sample_id=sid,image_sha256=source_sha,png_sha256=sha(raw),rgb_sha256=sha(result.tobytes()),bytes=len(raw))


def build(args):
    root=args.data_root.resolve()
    out=args.cache.resolve()
    policy=recipe_sha()
    audit=json.loads((args.release/"audit.json").read_bytes())
    for split in args.splits:
        start=time.monotonic()
        key="grouped/"+split
        source=(args.release/(key+".jsonl")).read_bytes()
        if sha(source)!=audit["manifests"][key]["sha256"]:
            raise ValueError("Wrong Model 5 source manifest")
        src=[json.loads(line) for line in source.splitlines() if line]
        jobs=[(root,out,policy,r["image_path"],r["sample_id"],r["image_sha256"]) for r in src]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            records=[]
            for record in pool.map(make_image,jobs,chunksize=4):
                records.append(record)
                if len(records)%256==0:
                    print(f"{split} {len(records)}/{len(src)}",flush=True)
        # Reopen prespecified seeded samples and compare decoded cache bytes with the
        # deterministic transform applied afresh to each original, not pixels in memory.
        checked=random.Random(20261114).sample(range(len(src)),min(16,len(src)))
        for i in checked:
            rec=records[i]; source_sha=rec["image_sha256"]
            original=geometry(root/src[i]["image_path"],source_sha).tobytes()
            raw=image_path(out,policy,source_sha).read_bytes()
            with Image.open(io.BytesIO(raw)) as cached:
                if cached.mode!="RGB" or cached.size!=(392,392) or cached.tobytes()!=original:
                    raise ValueError("Lossless cache pixel check failed")
            if sha(raw)!=rec["png_sha256"] or sha(original)!=rec["rgb_sha256"]:
                raise ValueError("Lossless cache digest check failed")
        manifest=dict(split=split,source_manifest_sha256=sha(source),recipe=recipe(),recipe_sha256=policy,
                      records=records,verified_ids=[src[i]["sample_id"] for i in checked])
        atomic(out/(split+".json"),(json.dumps(manifest,indent=2,allow_nan=False)+"\n").encode())
        print(f"{split} complete n={len(records)} bytes={sum(r['bytes'] for r in records)} "
              f"seconds={time.monotonic()-start:.1f} verified={len(checked)}",flush=True)


def package(args):
    policy=recipe_sha()
    files=set()
    for split in args.splits:
        index=args.cache/(split+".json")
        spec=json.loads(index.read_bytes())
        if spec["recipe_sha256"]!=policy:
            raise ValueError("Geometry/version mismatch")
        files.add(index)
        for rec in spec["records"]:
            path=image_path(args.cache,policy,rec["image_sha256"])
            if sha(path.read_bytes())!=rec["png_sha256"]:
                raise ValueError("Refusing to archive a corrupt cache")
            files.add(path)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix(".tmp")
    try:
        with tarfile.open(tmp,"w") as tar:
            for path in sorted(files):
                tar.add(path,arcname=(Path("letterbox_cache")/path.relative_to(args.cache)).as_posix(),recursive=False)
        os.replace(tmp,args.output)
    finally:
        tmp.unlink(missing_ok=True)
    h=hashlib.sha256()
    with args.output.open("rb") as file:
        for chunk in iter(lambda:file.read(4*1024**2),b""):
            h.update(chunk)
    print(f"{args.output} {args.output.stat().st_size} bytes sha256={h.hexdigest()}",flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    commands=cli.add_subparsers(dest="action",required=True)
    for action in ("build","package"):
        p=commands.add_parser(action)
        p.add_argument("--cache",required=True,type=Path)
        p.add_argument("--splits",nargs="+",choices=["train","val","test","external_test"],required=True)
        if action=="build":
            p.add_argument("--release",type=Path,default=Path("data/prepared/screening_v1"))
            p.add_argument("--data-root",type=Path,default=Path("."))
            p.add_argument("--workers",type=int,default=6)
        else:
            p.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    build(args) if args.action=="build" else package(args)


if __name__=="__main__":
    main()
