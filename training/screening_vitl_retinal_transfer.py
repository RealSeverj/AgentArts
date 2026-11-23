"""Model 5 F: official 392px ViT-L plus only the archived DR tail and final normalization."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from training import screening_vitl_letterbox as control
from training import screening_optimization as experiment
from training import screening_training as baseline

DONOR = "a85acf367a34361986739e93030bc5c51a390ddd33c63d49da883abf1562d4ab"
APTOS = "80c13752f5655290b2349ebc1dce951c83ef7aa7aa27d17572d09fbd0078fe05"
OFFICIAL = experiment.MODELS["vitl"][1]
DELTA = "6bcbc3f8dbb5e3b383a6dadc2cb733022273d3386015699b8c239ebe0124bbf0"
MIGRATED = ("norm1.weight","norm1.bias","attn.qkv.weight","attn.qkv.bias",
            "attn.proj.weight","attn.proj.bias","ls1.gamma","norm2.weight","norm2.bias",
            "mlp.fc1.weight","mlp.fc1.bias","mlp.fc2.weight","mlp.fc2.bias","ls2.gamma")
KEYS = tuple(sorted([f"blocks.{i}.{key}" for i in range(20,24) for key in MIGRATED]+["norm.weight","norm.bias"]))
CANDIDATE = "vitl_global_letterbox_dr_transfer"


def provenance(digest):
    return dict(source_checkpoint_sha256=DONOR, parent_checkpoint_sha256=APTOS, delta_sha256=digest,
                base_sha256=OFFICIAL, source_epoch=24, source_geometry=336, blocks=[20,21,22,23],
                final_norm=True, tensor_count=len(KEYS), include_pos_embed=False, include_source_head=False,
                supervised_source_history=["APTOS2019","DDR"], semantics="absolute fp32 tensor replacements")


def read_donor(path):
    if baseline.sha256(path)!=DONOR:
        raise ValueError("DR donor checksum mismatch")
    parent=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
    cfg=parent["config"]
    lineage=parent["initialization_lineage"]
    if (parent["epoch"]!=23 or cfg["model"]["name"]!=experiment.MODELS["vitl"][0]
            or cfg["model"]["img_size"]!=336 or cfg["task"]!="dr_grading" or cfg["kind"]!="multiclass"
            or cfg["num_classes"]!=5 or cfg["data"]["datasets"]!=["DDR"]
            or lineage["checkpoint_sha256"]!=APTOS
            or lineage["supervised_training_source_history"]!=["APTOS2019","DDR"]):
        raise ValueError("DR architecture, sources or lineage differ from expected")
    state=parent["model"]
    names={k.removeprefix("backbone.") for k in state if
           k.startswith(tuple(f"backbone.blocks.{i}." for i in range(20,24))+("backbone.norm.",))}
    if names!=set(KEYS) or len(names)!=58:
        raise ValueError("Incomplete or unexpected final-block tensors in donor")
    return {k:state["backbone."+k].detach().cpu().contiguous() for k in KEYS}


def extract(args):
    tensors=read_donor(args.source)
    if any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError("Invalid DR parameters")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix(".pending")
    try:
        save_file(tensors,str(tmp),metadata={"source_checkpoint_sha256":DONOR})
        actual=baseline.sha256(tmp)
        if actual!=DELTA:
            raise ValueError(f"Delta archive serialization differs from pinned output: {actual}")
        if args.output.exists() and baseline.sha256(args.output)!=actual:
            raise FileExistsError("Existing output contains a different artifact")
        os.replace(tmp,args.output)
    finally:
        tmp.unlink(missing_ok=True)
    print(json.dumps({"path":str(args.output),"bytes":args.output.stat().st_size,**provenance(actual)}))


def delta_payload(path):
    if baseline.sha256(path)!=DELTA:
        raise ValueError("The transfer tensor pack does not match the pinned fingerprint")
    payload=load_file(str(path),device="cpu")
    if set(payload)!=set(KEYS):
        raise ValueError("Transfer must contain exactly the approved 58 backbone tensors")
    return payload


def apply_tail(model,payload):
    destination=model.backbone.state_dict()
    if len(model.backbone.blocks)!=24 or tuple(destination["pos_embed"].shape)!=(1,785,1024):
        raise ValueError("Expected the official ViT-L resized directly to 392px by the D constructor")
    with torch.no_grad():
        for name in KEYS:
            target,value=destination[name],payload[name]
            if target.shape!=value.shape or target.dtype!=value.dtype or value.dtype!=torch.float32:
                raise ValueError(f"Transfer shape or dtype mismatch: {name}")
            target.copy_(value)


def tensor_hashes(state):
    return {k:hashlib.sha256(t.detach().cpu().contiguous().numpy()).hexdigest() for k,t in state.items()}


def verify(args):
    if baseline.sha256(args.weights)!=OFFICIAL:
        raise ValueError("Wrong official ViT-L initialization")
    torch.manual_seed(20261114)
    model=experiment.GroupedModel("vitl",args.weights,False)
    before=tensor_hashes(model.state_dict())
    payload=delta_payload(args.delta)
    source=read_donor(args.source)
    if any(not torch.equal(payload[k],source[k]) for k in KEYS):
        raise ValueError("Extracted tensors differ from DR donor")
    apply_tail(model,payload)
    after=tensor_hashes(model.state_dict())
    changed=sorted(k for k in before if before[k]!=after[k])
    if set(changed)!={"backbone."+k for k in KEYS}:
        raise ValueError(f"Unexpected changed tensors ({len(changed)})")
    if any(not torch.equal(model.backbone.state_dict()[k],source[k]) for k in KEYS):
        raise ValueError("Loaded tensor differs from its donor")
    shapes=[list(h.weight.shape) for h in model.heads]
    if shapes!=[[2,1024],[3,1024],[3,1024],[2,1024],[1,1024]]:
        raise ValueError("Wrong fresh Model 5 head")
    details=dict(transfer=provenance(DELTA), verified_changed_tensors=changed,
                 unchanged_state_entries=len(after)-len(changed),fresh_head_shapes=shapes,
                 official_392_position_sha256=after["backbone.pos_embed"],
                 fresh_head_sha256={k:v for k,v in after.items() if k.startswith("heads.")},
                 original_tensor_count=len(before),donor_values_exact=True,training_performed=False)
    if args.output:
        baseline.write_json(args.output,details)
    print(f"CPU exact donor/tensor check passed: {len(changed)} changed, {len(after)-len(changed)} unchanged, heads={shapes}")


def run(args):
    if args.workers<0 or (args.action=="train" and args.split!="val"):
        raise ValueError("Invalid worker count or training split")
    if not torch.cuda.is_available():
        raise ValueError("CUDA required for the fp16 training contract")
    args.accum=32//args.batch_size
    args.seed,args.epochs,args.head_only_epochs,args.blocks=20261114,30,5,4
    args.head_lr,args.backbone_lr,args.variant,args.precision=0.001,0.00001,"partial","fp16"
    args.device,args.roi_dir,args.rgb_cache_dir="cuda",None,None
    if args.max_gpu_gib is not None:
        fraction=args.max_gpu_gib*1024**3/torch.cuda.get_device_properties(0).total_memory
        if not math.isfinite(fraction) or not 0<fraction<=1:
            raise ValueError("CUDA allocator cap outside device capacity")
        torch.cuda.set_per_process_memory_fraction(fraction,device=0)
    audit,cfg,signature=control.build_contract(args)
    signature.update(candidate=CANDIDATE,transfer=provenance(DELTA))
    for p in (Path(__file__),Path(control.__file__)):
        signature["code_sha256"][p.name]=baseline.sha256(p)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.action=="predict" and args.split!="val":
        lock=json.loads((args.output/"thresholds.json").read_bytes())
        if lock["checkpoint_sha256"]!=baseline.sha256(args.output/"best.pt") or lock["labels"]!=experiment.LABELS:
            raise ValueError("Held-out prediction requires validation lock")
    payload=delta_payload(args.delta)
    baseline.seed_all(args.seed)
    with control.active_views(cfg):
        # Identical to D up to this point, including fresh heads and the single
        # interpolation of the official positional embedding to the 28x28 grid.
        model=experiment.GroupedModel("vitl",args.weights,False)
        apply_tail(model,payload)
        del payload
        model.to(args.device)
        if args.action=="train":
            experiment.train(args,audit,model,signature)
            split="val"
        else:
            split=args.split
            experiment.export(args,audit,model,signature,split)
        path=args.output/f"{split}_predictions.meta.json"
        meta=json.loads(path.read_bytes())
        meta.update(transform=cfg,input_mode=signature["input_mode"],candidate=CANDIDATE,transfer=signature["transfer"],
                    geometry_recipe_sha256=signature["geometry_recipe_sha256"],
                    geometry_cache_index_sha256=baseline.sha256(args.letterbox_cache_dir/(split+".json")))
        baseline.write_json(path,meta)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    commands=cli.add_subparsers(dest="action",required=True)
    ext=commands.add_parser("extract")
    ext.add_argument("--source",type=Path,required=True)
    ext.add_argument("--output",type=Path,required=True)
    val=commands.add_parser("verify")
    for name in ("source","weights","delta"):
        val.add_argument("--"+name,type=Path,required=True)
    val.add_argument("--output",type=Path)
    for action in ("train","predict"):
        p=commands.add_parser(action)
        for name in ("weights","delta"):
            p.add_argument("--"+name,type=Path,required=True)
        p.add_argument("--release",type=Path,default=Path("data/prepared/screening_v1"))
        p.add_argument("--data-root",type=Path,default=Path("."))
        p.add_argument("--letterbox-cache-dir",type=Path,default=Path("letterbox_cache"))
        p.add_argument("--output",type=Path,default=Path("runs/geometry_retina"))
        p.add_argument("--batch-size",type=int,default=8,choices=[1,2,4,8,16,32])
        p.add_argument("--workers",type=int,default=4)
        p.add_argument("--max-gpu-gib",type=float)
        p.add_argument("--split",choices=["val","test","external_test"],default="val")
        p.add_argument("--resume",action="store_true")
    args=cli.parse_args()
    if args.action in ("verify","extract"):
        torch.set_num_threads(2)
        (verify if args.action=="verify" else extract)(args)
    else:
        run(args)


if __name__=="__main__":
    main()
