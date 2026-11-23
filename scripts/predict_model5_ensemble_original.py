"""Score original fundus photographs with the fingerprint-locked Model 5 A/B ensemble."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import timm
import torch
from torch import nn

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts import prepare_screening_optic_disc_rois as optic
from training import screening_optimization as architecture
from training import screening_training as base_engine
from training.classification import ClassificationTransform
from training.screening_optimization import DISC_GROUP,GROUP_WIDTHS,GroupedModel,LABELS,MODELS,ROI_POLICY_SHA
from training.screening_training import sha256
from training.structural import _tensor
from training.train import amp_context

DEFAULT_RUN=PROJECT/"02_模型训练与评估/07_模型5云端优化结果_20261122"
DEFAULT_LOCATOR=PROJECT/"02_模型训练与评估/01_在用模型/04_视盘视杯分割/disc_refuge_best.pt"
FINAL="348a6752a51cd0f2424d6fcdfa85ea1139f3594b3fb8b4ec656f7d297a753b3e"
MEMBERS=(("global_vitl","large_full","vitl",False,"4e70ad8472711a718e49784d8f5ab753ef00e97b5771379eed8c0706a391d4a2"),
         ("dual_vitb","base_disc","vitb",True,"7f492c02278ee95983b582e5a36b4ea17708c7df04111ead173a14039425db2b"))


def require(ok,msg):
    if not ok: raise ValueError(msg)


def pinned(path,expected):
    require(sha256(path)==expected,f"SHA-256 differs for {path}")
    return json.loads(path.read_bytes())


class RestoredGroupedModel(GroupedModel):
    """Reuse A/B's training forward with its complete supervised checkpoint."""
    def __init__(self,arch,roi):
        nn.Module.__init__(self)
        self.backbone=timm.create_model(MODELS[arch][0],pretrained=False,img_size=392,num_classes=0)
        self.use_roi=roi;self.dropout=nn.Dropout(.1)
        dim=self.backbone.num_features
        self.heads=nn.ModuleList(nn.Linear(2*dim if i==DISC_GROUP and roi else dim,w) for i,w in enumerate(GROUP_WIDTHS))


class Ensemble:
    def __init__(self,run,locator,device):
        self.run=run;self.device=torch.device(device)
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        self.final=pinned(run/"final_selection.json",FINAL)
        self.selection=pinned(run/"ensemble/selection.json",self.final["four_way_selection_sha256"])
        require(self.final["selected"]==self.selection["winner"]=="mean_probability" and self.selection["ensemble_selected"],"Wrong selected winner")
        self.recipe_sha=self.selection["ensemble_recipe_sha256"];self.threshold_sha=self.selection["ensemble_threshold_lock_sha256"]
        self.recipe=pinned(run/"ensemble/ensemble_recipe.json",self.recipe_sha)
        self.lock=pinned(run/"ensemble/thresholds.json",self.threshold_sha)
        require(self.recipe["operation"]=="mean of two post-sigmoid probability vectors in float64" and
                self.lock["checkpoint_sha256"]==self.recipe_sha and self.recipe["labels"]==self.lock["labels"]==LABELS,
                "Ensemble transform, thresholds or label order differ")
        require(optic.POLICY_SHA==ROI_POLICY_SHA and sha256(Path(optic.__file__))=="115058d5859e3f620fc40c9b12aad4090e1c1d1158c8d28e0dd507b41b58610e",
                "Frozen optic-disc crop policy differs")
        members={m["role"]:m for m in self.recipe["members"]}
        self.arms={}
        for role,dirname,arch,roi,runsha in MEMBERS:
            member=members[role];where=run/dirname;ck=where/"best.pt";cfg=pinned(where/"run.json",runsha)
            require(member["weight"]==.5 and sha256(ck)==member["checkpoint_sha256"] and
                    member["locator_sha256"]==(optic.LOCATOR if roi else None),"Member weights or locator identity differ")
            done=json.loads((where/"complete.json").read_bytes())
            state=torch.load(ck,map_location="cpu",weights_only=True,mmap=True)
            ident=state["identity"]
            resume_id=dict(ident,settings=state["settings"],code_sha256=sha256(Path(architecture.__file__)),
                           baseline_code_sha256=sha256(Path(base_engine.__file__)))
            require(resume_id==cfg["identity"],"Checkpoint settings, data signature or archived engine revision differ")
            require(state["labels"]==cfg["labels"]==LABELS and state["epoch"]==done["best_epoch"]
                    and state["best_validation_macro_ap"]==done["best_validation_macro_ap"]
                    and done["checkpoint_sha256"]==member["checkpoint_sha256"] and ident["image_size"]==392
                    and ident["input_mode"]==("global_disc_roi" if roi else "global") and ident["backbone"]==MODELS[arch][0]
                    and (not roi or ident["roi_policy_sha256"]==optic.POLICY_SHA),"Incorrect checkpoint identity")
            net=RestoredGroupedModel(arch,roi);net.load_state_dict(state["model"],strict=True)
            precision=state["settings"]["precision"] if self.device.type=="cuda" else "none"
            self.arms[role]=(net.to(self.device).eval().requires_grad_(False),ClassificationTransform(cfg["transform"],False),precision)
            del state
        self.locator,self.locator_cfg=optic.load_locator(locator,self.device)

    def score(self,path):
        raw=path.read_bytes()
        with Image.open(io.BytesIO(raw)) as native: image=native.convert("RGB")
        result=self.score_image(image)
        result["image_sha256"]=hashlib.sha256(raw).hexdigest()
        return result

    @torch.inference_mode()
    def score_image(self,image):
        """Score a decoded RGB original, using the same inputs as the file-based replay."""
        locator_input=_tensor(image.resize((512,512),Image.Resampling.BILINEAR),self.locator_cfg)[None].to(self.device)
        location=self.locator(locator_input)
        require(tuple(location.shape)==(1,2,512,512),"Wrong locator geometry")
        if torch.isfinite(location).all().item():
            mask=location[0,1].sigmoid().ge(optic.POLICY["disc_threshold"]).cpu().numpy()
            bbox,reason,_=optic.disc_window(mask,image.size)
        else: bbox,reason=None,"nonfinite_logits"
        members={}
        for role,_,_,roi,_ in MEMBERS:
            net,transform,precision=self.arms[role];whole=transform(image)
            x=torch.stack((whole,transform(image.crop(bbox)) if bbox else whole)) if roi else whole
            with amp_context(self.device,precision): logits=net(x.unsqueeze(0).to(self.device)).float()[0]
            require(logits.shape==(11,) and torch.isfinite(logits).all().item(),"Invalid classification logits")
            members[role]=logits.sigmoid().cpu().tolist()
        p=(np.asarray(members["global_vitl"],dtype=np.float64)+np.asarray(members["dual_vitb"],dtype=np.float64))/2
        t=[self.lock["per_label"][k]["threshold"] for k in LABELS]
        return dict(use="offline_research",candidate="mean_probability",labels=LABELS,
                    source_size_wh=list(image.size),member_probabilities=members,probabilities=p.tolist(),thresholds=t,
                    decisions=[None if v is None else int(x>=v) for x,v in zip(p,t)],
                    bbox_xyxy=bbox,locator_fallback=reason,locator_sha256=optic.LOCATOR,roi_policy_sha256=optic.POLICY_SHA,
                    recipe_sha256=self.recipe_sha,threshold_sha256=self.threshold_sha,final_selection_sha256=FINAL)

    def compare(self,result,manifest,coordinates):
        expected_hash=self.lock["val_manifest_sha256"]
        require(sha256(manifest)==expected_hash==self.selection["validation_manifest_sha256"],"Wrong validation manifest")
        sha=result["image_sha256"]
        def match(path,field,value):
            rows=[r for r in map(json.loads,path.read_bytes().splitlines()) if r[field]==value]
            require(len(rows)==1,f"Expected one matching row in {path}")
            return rows[0]
        sid=match(manifest,"image_sha256",sha)["sample_id"]
        coord=match(coordinates,"sample_id",sid)
        provenance=self.selection["validation_predictions"]
        require(sha256(coordinates)==provenance["dual_vitb"]["roi_manifest_sha256"] and coord["image_sha256"]==sha
                and coord["locator_sha256"]==optic.LOCATOR and coord["policy_sha256"]==optic.POLICY_SHA,"Incorrect reference ROI")
        errors={}
        for role,dirname,*_ in MEMBERS:
            path=self.run/dirname/"val_predictions.jsonl";require(sha256(path)==provenance[role]["predictions_sha256"],"Wrong reference member scores")
            original=match(path,"sample_id",sid)
            errors[role]=float(np.max(abs(np.asarray(original["probabilities"])-result["member_probabilities"][role])))
        path=self.run/"ensemble/val_predictions.jsonl";require(sha256(path)==self.lock["val_predictions_sha256"],"Wrong reference ensemble scores")
        ref=match(path,"sample_id",sid);require(ref["checkpoint_sha256"]==self.recipe_sha and ref["image_sha256"]==sha,"Reference image differs")
        calls=[None if t is None else int(p>=t) for p,t in zip(ref["probabilities"],result["thresholds"])]
        return dict(sample_id=sid,reference_roi=coord["bbox_xyxy"],same_roi=coord["bbox_xyxy"]==result["bbox_xyxy"],
                    max_abs_member_score_error=errors,max_abs_ensemble_score_error=float(np.max(abs(np.asarray(ref["probabilities"])-result["probabilities"]))),
                    changed_threshold_decisions=[k for k,a,b in zip(LABELS,calls,result["decisions"]) if a!=b])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("image",type=Path)
    parser.add_argument("--run",type=Path,default=DEFAULT_RUN);parser.add_argument("--locator",type=Path,default=DEFAULT_LOCATOR)
    parser.add_argument("--device",choices=["cuda","cpu"],default="cuda");parser.add_argument("--compare-val",action="store_true")
    parser.add_argument("--val-manifest",type=Path,default=PROJECT/"data/prepared/screening_v1/grouped/val.jsonl")
    parser.add_argument("--val-roi",type=Path,default=PROJECT/"data/prepared/screening_optic_disc_rois_v1/grouped/val.jsonl")
    parser.add_argument("--require-match",action="store_true",help="Require identical ROI and flags and bounded probability drift")
    parser.add_argument("--score-atol",type=float,default=.005);args=parser.parse_args()
    if args.require_match and not args.compare_val: parser.error("--require-match requires --compare-val")
    model=Ensemble(args.run,args.locator,args.device);result=model.score(args.image)
    if args.compare_val: result["comparison"]=model.compare(result,args.val_manifest,args.val_roi)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    if args.require_match:
        c=result["comparison"]
        sys.exit(0 if c["same_roi"] and not c["changed_threshold_decisions"] and c["max_abs_ensemble_score_error"]<=args.score_atol else 2)
