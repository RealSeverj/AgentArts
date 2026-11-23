"""Build and verify the distinct 518px lossless cache without modifying the published 392px recipe."""
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
import time

from PIL import Image
from training import fundus_preprocessing as fundus
from training import screening_letterbox_cache as storage

SIZE = 518


def recipe():
    spec = storage.recipe()
    spec["size"] = [SIZE,SIZE]
    return spec


def recipe_sha():
    return storage.sha(json.dumps(recipe(),sort_keys=True,separators=(",",":")).encode())


def render(path, digest):
    raw = path.read_bytes()
    if storage.sha(raw) != digest:
        raise ValueError(f"Original image hash mismatch: {path}")
    with Image.open(io.BytesIO(raw)) as image:
        return fundus.apply_preprocessing(image,SIZE,{"mode":fundus.MODE})


def build_one(job):
    data_root, root, policy, name, sample_id, digest = job
    src = (data_root/name).resolve()
    if not src.is_relative_to(data_root):
        raise ValueError("Image path escapes the dataset root")
    image = render(src,digest)
    stream = io.BytesIO()
    image.save(stream,format="PNG",compress_level=1)
    png = stream.getvalue()
    storage.atomic(storage.image_path(root,policy,digest),png)
    return dict(sample_id=sample_id,image_sha256=digest,png_sha256=storage.sha(png),
                rgb_sha256=storage.sha(image.tobytes()),bytes=len(png))


def build(args):
    policy = recipe_sha()
    data_root = args.data_root.resolve()
    audit = json.loads((args.release/"audit.json").read_bytes())
    for split in args.splits:
        start = time.monotonic()
        raw = (args.release/"grouped"/f"{split}.jsonl").read_bytes()
        expected = audit["manifests"][f"grouped/{split}"]
        if storage.sha(raw)!=expected["sha256"]:
            raise ValueError("Source split digest mismatch")
        rows = [json.loads(r) for r in raw.splitlines() if r.strip()]
        if len(rows)!=expected["records"] or len(set(r["sample_id"] for r in rows))!=len(rows):
            raise ValueError("Expected the complete split")
        jobs = [(data_root,args.cache.resolve(),policy,r["image_path"],r["sample_id"],r["image_sha256"]) for r in rows]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            records = []
            for rec in pool.map(build_one,jobs,chunksize=4):
                records.append(rec)
                if len(records)%256==0:
                    print(f"{split} {len(records)}/{len(rows)}",flush=True)
        selected = random.Random(20261114).sample(range(len(rows)),min(16,len(rows)))
        for i in selected:
            rec = records[i]
            original = render(data_root/rows[i]["image_path"],rec["image_sha256"]).tobytes()
            path = storage.image_path(args.cache,policy,rec["image_sha256"])
            png = path.read_bytes()
            with Image.open(io.BytesIO(png)) as cached:
                if (cached.size!=(SIZE,SIZE) or cached.mode!="RGB" or cached.tobytes()!=original
                        or storage.sha(png)!=rec["png_sha256"] or storage.sha(original)!=rec["rgb_sha256"]):
                    raise ValueError("Original and cached 518px RGB differ")
        index = dict(split=split,source_manifest_sha256=expected["sha256"],recipe=recipe(),
                     recipe_sha256=policy,records=records,verified_ids=[rows[i]["sample_id"] for i in selected])
        storage.atomic(args.cache/f"{split}.json",(json.dumps(index,indent=2)+"\n").encode())
        print(f"{split} n={len(records)} png_bytes={sum(r['bytes'] for r in records)} "
              f"elapsed={time.monotonic()-start:.1f}s recipe_sha256={policy}",flush=True)


def package(args):
    files = set()
    for split in args.splits:
        index = args.cache/f"{split}.json"
        spec = json.loads(index.read_bytes())
        if spec["recipe_sha256"]!=recipe_sha():
            raise ValueError("Wrong 518px recipe")
        files.add(index)
        for rec in spec["records"]:
            path = storage.image_path(args.cache,spec["recipe_sha256"],rec["image_sha256"])
            if storage.sha(path.read_bytes())!=rec["png_sha256"]:
                raise ValueError(f"Cache hash mismatch: {path}")
            files.add(path)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    try:
        with tarfile.open(temporary,"w") as archive:
            for p in sorted(files):
                archive.add(p,arcname=(Path("letterbox_518_cache")/p.relative_to(args.cache)).as_posix(),recursive=False)
        os.replace(temporary,args.output)
    finally:
        temporary.unlink(missing_ok=True)
    digest=hashlib.sha256()
    with args.output.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b""):
            digest.update(chunk)
    print(f"{args.output} {args.output.stat().st_size} bytes sha256={digest.hexdigest()}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="action",required=True)
    for mode in ("build","package"):
        p=commands.add_parser(mode)
        p.add_argument("--cache",type=Path,default=Path("letterbox_518_cache"))
        p.add_argument("--splits",nargs="+",choices=["train","val","test","external_test"],required=True)
        if mode=="build":
            p.add_argument("--data-root",type=Path,default=Path("."))
            p.add_argument("--release",type=Path,default=Path("data/prepared/screening_v1"))
            p.add_argument("--workers",type=int,default=6)
        else:
            p.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.action=="build":
        build(args)
    else:
        package(args)


if __name__=="__main__":
    main()
