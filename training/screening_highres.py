"""Model 5 candidate E: native DINOv2 ViT-B/14 518px full-image, with the fixed crop/letterbox policy."""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import timm
import torch
from torch import nn
from PIL import Image
from timm.models.vision_transformer import checkpoint_filter_fn

from training import classification, fundus_preprocessing, screening_518_cache as high_cache
from training import screening_letterbox as candidate_c
from training import screening_letterbox_cache as storage
from training import screening_training as baseline


class HighResolutionDataset(baseline.ScreeningDataset):
    def __init__(self, args, audit, task, split):
        super().__init__(args,audit,task,split)
        self.geometry_dir = args.letterbox_cache_dir
        spec = json.loads((self.geometry_dir/f"{split}.json").read_bytes())
        self.recipe = high_cache.recipe_sha()
        if (spec["split"]!=split or spec["source_manifest_sha256"]!=self.manifest_sha
                or spec["recipe"]!=high_cache.recipe() or spec["recipe_sha256"]!=self.recipe
                or len(spec["records"])!=len(self.rows)):
            raise ValueError(f"Inconsistent 518px geometry cache for {split}")
        self.cache_rows = spec["records"]
        for original,cached in zip(self.rows,self.cache_rows):
            if any(original[k]!=cached[k] for k in ("sample_id","image_sha256")):
                raise ValueError("518px cache identity or image ordering differs")
        self.transform=classification.ClassificationTransform(
            baseline.TRANSFORM_CONFIG,training=(split=="train"),input_preprocessed=True)

    def source_rgb(self,index):
        row=self.cache_rows[index]
        png=storage.image_path(self.geometry_dir,self.recipe,row["image_sha256"]).read_bytes()
        if storage.sha(png)!=row["png_sha256"]:
            raise ValueError(f"Corrupt geometry artifact: {row['sample_id']}")
        with Image.open(io.BytesIO(png)) as im:
            if im.size!=(high_cache.SIZE, high_cache.SIZE) or im.mode!="RGB":
                raise ValueError("Expected cached 518x518 RGB")
            return im.copy()


class HighResolutionModel(baseline.ScreeningModel):
    def __init__(self,weights):
        nn.Module.__init__(self)
        self.backbone=timm.create_model(baseline.BACKBONE,pretrained=False,img_size=high_cache.SIZE,num_classes=0)
        state=torch.load(weights,map_location="cpu",weights_only=True)
        self.backbone.load_state_dict(checkpoint_filter_fn(state,self.backbone),strict=True)
        self.dropout=nn.Dropout(0.1)
        self.heads=nn.ModuleList(nn.Linear(self.backbone.num_features,n) for n in (2,3,3,2,1))


def contract(args):
    audit_file=args.release/"audit.json"
    audit=json.loads(audit_file.read_bytes())
    if (audit.get("status")!="verified" or audit["label_order"]["grouped"]!=baseline.LABELS["grouped"]
            or any(audit["manifests"]["grouped/"+k]["records"]!=n for k,n in
                   (("train",1878),("val",634),("test",615),("external_test",848)))):
        raise ValueError("Expected the complete Model 5 release")
    if baseline.sha256(args.config)!=candidate_c.PLAN_SHA or baseline.sha256(args.weights)!=baseline.OFFICIAL_SHA:
        raise ValueError("Training plan or ViT-B official initialization differs from candidate C")
    policy=high_cache.recipe_sha()
    cfg=candidate_c.transform_config(baseline)
    cfg["model"]["img_size"]=high_cache.SIZE
    settings=dict(candidate_c.SETTINGS,batch_size=args.batch_size,accum=32//args.batch_size)
    for k,v in settings.items():
        setattr(args,k,v)
    files=(__file__,baseline.__file__,classification.__file__,fundus_preprocessing.__file__,candidate_c.__file__,
           storage.__file__,high_cache.__file__)
    identity=dict(task="grouped",candidate="vitb_global_letterbox_518",backbone=baseline.BACKBONE,
                  image_size=high_cache.SIZE,input_mode="global_fundus_crop_letterbox_v1",locator_sha256=None,
                  transform=cfg,geometry_recipe_sha256=policy,plan_sha256=candidate_c.PLAN_SHA,
                  pretrained_sha256=baseline.OFFICIAL_SHA,release_sha256=baseline.sha256(audit_file),
                  controlled_settings=settings,code_sha256={p.name:baseline.sha256(p) for p in map(Path,files)})
    return audit,cfg,identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["train","predict"])
    p.add_argument("--weights",required=True,type=Path)
    p.add_argument("--release",type=Path,default=Path("data/prepared/screening_v1"))
    p.add_argument("--data-root",type=Path,default=Path("."))
    p.add_argument("--config",type=Path,default=Path("configs/grouped.plan.json"))
    p.add_argument("--letterbox-cache-dir",type=Path,default=Path("letterbox_518_cache"))
    p.add_argument("--output",type=Path,default=Path("runs/grouped_vitb_letterbox_518"))
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--batch-size",type=int,choices=[4,8],default=4)
    p.add_argument("--max-gpu-gib",type=float)
    p.add_argument("--split",choices=["val","test","external_test"],default="val")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.workers<0 or (args.action=="train" and args.split!="val"):
        p.error("Invalid worker count or training split")
    if not torch.cuda.is_available():
        p.error("This fp16 training and predictive evaluation require CUDA")
    args.device,args.rgb_cache_dir="cuda",None
    if args.max_gpu_gib is not None:
        total=torch.cuda.get_device_properties(0).total_memory
        fraction=args.max_gpu_gib*1024**3/total
        if not math.isfinite(fraction) or not 0<fraction<=1:
            p.error("CUDA allocator cap must fall within visible device capacity")
        torch.cuda.set_per_process_memory_fraction(fraction,device=0)
    audit,cfg,identity=contract(args)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.action=="predict" and args.split!="val":
        lock=json.loads((args.output/"thresholds.json").read_bytes())
        if lock["labels"]!=baseline.LABELS["grouped"] or lock["checkpoint_sha256"]!=baseline.sha256(args.output/"best.pt"):
            raise ValueError("Fixed validation thresholds for this checkpoint are required")
    baseline.seed_all(args.seed)
    with candidate_c.use_transform(baseline,cfg,HighResolutionDataset):
        model=HighResolutionModel(args.weights).to(args.device)
        if args.action=="train":
            baseline.train(args,audit,"grouped",model,identity)
            split="val"
        else:
            split=args.split
            baseline.export(model,args,audit,"grouped",split,args.output/"best.pt",identity)
        path=args.output/f"{split}_predictions.meta.json"
        meta=json.loads(path.read_bytes())
        meta.update(transform=cfg,input_mode=identity["input_mode"],candidate=identity["candidate"],
                    geometry_recipe_sha256=identity["geometry_recipe_sha256"],
                    geometry_cache_index_sha256=baseline.sha256(args.letterbox_cache_dir/(split+".json")))
        baseline.write_json(path,meta)


if __name__=="__main__":
    main()
