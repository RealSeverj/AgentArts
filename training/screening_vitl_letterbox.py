"""Model 5 ViT-L whole-image control using the same deterministic 392px crop/letterbox cache as candidate C."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch

from training import classification, fundus_preprocessing, screening_letterbox, screening_letterbox_cache
from training import screening_letterbox_dataset, screening_optimization as experiment
from training import screening_training as baseline


class LargeLetterboxViews(screening_letterbox_dataset.CachedGeometryDataset):
    def __init__(self, args, audit, split):
        super().__init__(args, audit, "grouped", split)
        # The shared exporter distinguishes whole-image and local-view datasets.
        self.locations = None
        self.roi_sha = None


@contextmanager
def active_views(config):
    original = experiment.GroupedViews
    # Scoped to the independent training process; checkpoints and other CLI
    # entry points retain their original constructors and input contracts.
    with screening_letterbox.use_transform(baseline, config):
        try:
            experiment.GroupedViews = LargeLetterboxViews
            yield
        finally:
            experiment.GroupedViews = original


def build_contract(args):
    name, digest = experiment.MODELS["vitl"]
    audit_file = args.release / "audit.json"
    audit = json.loads(audit_file.read_bytes())
    if (audit.get("status") != "verified" or audit["label_order"]["grouped"] != experiment.LABELS
            or any(audit["manifests"]["grouped/"+k]["records"]!=n for k,n in
                   (("train",1878),("val",634),("test",615),("external_test",848)))):
        raise ValueError("Expected the complete 11-label data release")
    if baseline.sha256(args.weights)!=digest:
        raise ValueError("Initial weights differ from the original DINOv2 ViT-L/14")
    config = screening_letterbox.transform_config(baseline)
    config["model"]["name"] = name
    policy = screening_letterbox_cache.recipe_sha()
    files = [__file__, experiment.__file__, baseline.__file__, classification.__file__, fundus_preprocessing.__file__,
             screening_letterbox.__file__, screening_letterbox_cache.__file__, screening_letterbox_dataset.__file__]
    signature = dict(task="grouped", backbone=name, pretrained_sha256=digest,
                     release_sha256=baseline.sha256(audit_file), image_size=392,
                     input_mode="global_fundus_crop_letterbox_v1", locator_sha256=None,
                     geometry_recipe_sha256=policy, transform=config, candidate="vitl_global_letterbox",
                     code_sha256={p.name:baseline.sha256(p) for p in map(Path,files)})
    return audit,config,signature


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=["train","predict"])
    cli.add_argument("--weights",type=Path,required=True)
    cli.add_argument("--release",type=Path,default=Path("data/prepared/screening_v1"))
    cli.add_argument("--data-root",type=Path,default=Path("."))
    cli.add_argument("--letterbox-cache-dir",type=Path,default=Path("letterbox_cache"))
    cli.add_argument("--output",type=Path,default=Path("runs/grouped_vitl_letterbox"))
    cli.add_argument("--batch-size",type=int,choices=[1,2,4,8,16,32],default=8)
    cli.add_argument("--workers",type=int,default=4)
    cli.add_argument("--max-gpu-gib",type=float,help="Per-process PyTorch CUDA allocator cap in GiB")
    cli.add_argument("--split",choices=["val","test","external_test"],default="val")
    cli.add_argument("--resume",action="store_true")
    args=cli.parse_args()
    if args.workers<0 or (args.action=="train" and args.split!="val"):
        cli.error("Invalid worker count or training split")
    if not torch.cuda.is_available():
        cli.error("CUDA required for this 30-epoch fp16 run")
    # These settings match the original ViT-L whole-image comparison.
    args.accum=32//args.batch_size
    args.seed, args.epochs, args.head_only_epochs, args.blocks = 20261114,30,5,4
    args.head_lr, args.backbone_lr, args.variant, args.precision = 0.001,0.00001,"partial","fp16"
    args.device, args.roi_dir, args.rgb_cache_dir = "cuda",None,None
    total = torch.cuda.get_device_properties(0).total_memory
    if args.max_gpu_gib is not None:
        fraction = args.max_gpu_gib * 1024**3 / total
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            cli.error("Allocator cap must be positive and within the visible GPU capacity")
        torch.cuda.set_per_process_memory_fraction(fraction,device=0)
    free,_=torch.cuda.mem_get_info()
    print(f"CUDA {torch.cuda.get_device_name(0)} free={free/(1024**3):.2f}/{total/(1024**3):.2f} GiB, "
          f"allocator cap={args.max_gpu_gib} GiB; microbatch={args.batch_size}, accum={args.accum}",flush=True)
    audit,config,signature=build_contract(args)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.action=="predict" and args.split!="val":
        lock=json.loads((args.output/"thresholds.json").read_bytes())
        if lock["checkpoint_sha256"]!=baseline.sha256(args.output/"best.pt") or lock["labels"]!=experiment.LABELS:
            raise ValueError("Held-out scores require the matching validation threshold lock")
    baseline.seed_all(args.seed)
    with active_views(config):
        model=experiment.GroupedModel("vitl",args.weights,False).to(args.device)
        if args.action=="train":
            experiment.train(args,audit,model,signature)
            split="val"
        else:
            split=args.split
            experiment.export(args,audit,model,signature,split)
        path=args.output/f"{split}_predictions.meta.json"
        meta=json.loads(path.read_bytes())
        meta.update(input_mode=signature["input_mode"], candidate=signature["candidate"], transform=config,
                    geometry_recipe_sha256=signature["geometry_recipe_sha256"],
                    geometry_cache_index_sha256=baseline.sha256(args.letterbox_cache_dir/(split+".json")))
        baseline.write_json(path,meta)


if __name__=="__main__":
    main()
