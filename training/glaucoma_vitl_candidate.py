"""Full-cohort Model 6 ViT-L/14 candidate with frozen validation-only selection."""
import argparse
from argparse import Namespace
import json
import math
from pathlib import Path

import timm
import torch
from timm.models.vision_transformer import checkpoint_filter_fn
from torch import nn

from scripts import screening_evaluation as ev
from training import classification, fundus_preprocessing, screening_letterbox, screening_letterbox_cache as cache
from training import screening_letterbox_dataset as datasets, screening_training as base


MODEL='vit_large_patch14_dinov2.lvd142m'
WEIGHTS='d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428'
HISTORICAL='378fc1a1e15e4f67e016bacaa49fd1b81e016c973043919e3a0a290edadc1aca'
CANDIDATE='glaucoma_vitl392_letterbox'
POLICY=dict(candidate=CANDIDATE, initialization='official DINOv2 ViT-L/14, all pretrained tensors except new 1-logit glaucoma head',
            geometry='fundus_crop_letterbox_v1, 392px, lossless RGB before stochastic augmentation',
            objective='all 692 known training eyes of the 738 release entries, weighted BCE, same schedule as historical',
            selection='strictly greater noninterpolated PAPILA validation AP than historical frozen model, tie retains historical',
            seed=20261114, epochs=30, head_only_epochs=5, trainable_blocks=4, effective_batch=32,
            head_lr=.001, backbone_lr=.00001, min_recall=.9, max_fpr=.1,
            historical_sha256=HISTORICAL, starting_sha256=WEIGHTS)


class LargeGlaucoma(base.ScreeningModel):
    def __init__(self, weights):
        nn.Module.__init__(self)
        self.backbone=timm.create_model(MODEL,pretrained=False,img_size=392,num_classes=0)
        raw=torch.load(weights,map_location='cpu',weights_only=True,mmap=True)
        self.backbone.load_state_dict(checkpoint_filter_fn(raw,self.backbone),strict=True)
        self.dropout=nn.Dropout(.1)
        self.heads=nn.ModuleList([nn.Linear(self.backbone.num_features,1)])


def identity(args):
    audit_file=args.release/'audit.json'
    audit=ev.read_json(audit_file)
    if audit['status']!='verified' or audit['label_order']['glaucoma']!=['glaucoma']:
        raise ValueError('Wrong audit release or glaucoma label order')
    for split,n in [('train',738),('val',72),('test',78),('external_test',400)]:
        if audit['manifests']['glaucoma/'+split]['records']!=n:raise ValueError('Unexpected release count')
    if base.sha256(args.weights)!=WEIGHTS or cache.recipe_sha()!='dc9eba4e907231ecfe2319a7ea6d1658b6e10b9d2142dde667b0eba6e7199459':
        raise ValueError('Initial weights or 392px geometry recipe differs')
    config=screening_letterbox.transform_config(base)
    config['model']['name']=MODEL
    paths=[Path(p) for p in (__file__,base.__file__,classification.__file__,fundus_preprocessing.__file__,cache.__file__,
                              screening_letterbox.__file__,datasets.__file__)]
    ident=dict(candidate=CANDIDATE,task='glaucoma',backbone=MODEL,labels=['glaucoma'],pretrained_sha256=WEIGHTS,
               release_sha256=base.sha256(audit_file),policy_sha256=cache.sha(json.dumps(POLICY,sort_keys=True).encode()),
               code_sha256={p.name:base.sha256(p) for p in paths},transform=config,image_size=392,
               input_mode='global_fundus_crop_letterbox_v1',geometry_recipe_sha256=cache.recipe_sha(),
               train_manifest_sha256=audit['manifests']['glaucoma/train']['sha256'],
               val_manifest_sha256=audit['manifests']['glaucoma/val']['sha256'])
    return audit,config,ident


def select(args):
    candidate,historical=args.output,args.historical_val
    if ev.read_json(candidate/'selection_policy.json')!=POLICY:raise ValueError('Selection specification changed')
    done=ev.read_json(candidate/'complete.json'); run=ev.read_json(candidate/'run.json')
    if (done['epochs']!=30 or done['checkpoint_sha256']!=base.sha256(candidate/'best.pt') or run['identity']['candidate']!=CANDIDATE
            or run['identity']['policy_sha256']!=cache.sha(json.dumps(POLICY,sort_keys=True).encode())):
        raise ValueError('Incomplete or unrelated candidate training')
    ranks={}; ids=None
    for name,folder,expected in [('historical',historical,HISTORICAL),(CANDIDATE,candidate,done['checkpoint_sha256'])]:
        rows,meta,digest=ev.load_predictions(folder,'val')
        aligned={r['sample_id']:[r[k] for k in ('labels','mask','patient_id','content_group_id','dataset','image_path')] for r in rows}
        if (meta['manifest_sha256']!=run['identity']['val_manifest_sha256'] or meta['labels']!=['glaucoma'] or meta['checkpoint_sha256']!=expected
                or len(rows)!=72 or {r['dataset'] for r in rows}!={'PAPILA'} or (ids is not None and ids!=aligned)):
            raise ValueError('Predictions are not paired across the original PAPILA validation images')
        ids=aligned
        y,p,g=ev.known_arrays(rows,0)
        if (len(y),int(y.sum()),len(set(g[y==1])))!=(62,13,7):raise ValueError('Unexpected PAPILA validation class support')
        if name==CANDIDATE:ev.lock_thresholds(Namespace(run=folder,policy=None,min_recall=.9,max_fpr=.1))
        lock=ev.read_json(folder/'thresholds.json'); operating=lock['per_label']['glaucoma']
        if (lock['checkpoint_sha256']!=expected or lock['val_predictions_sha256']!=digest or operating['min_recall']!=.9 or operating['max_fpr']!=.1):
            raise ValueError('Threshold lock mismatch')
        ranks[name]=dict(metrics=ev.metrics(y,p,operating['threshold']),threshold=operating['threshold'],
                         checkpoint_sha256=expected,thresholds_sha256=base.sha256(folder/'thresholds.json'),val_predictions_sha256=digest)
    if not math.isclose(ranks[CANDIDATE]['metrics']['ap'],done['best_validation_macro_ap'],abs_tol=1e-10) or not math.isclose(ranks['historical']['metrics']['ap'],.74459937949619,abs_tol=1e-10):
        raise ValueError('Validation AP differs from frozen reference')
    winner=CANDIDATE if ranks[CANDIDATE]['metrics']['ap']>ranks['historical']['metrics']['ap'] else 'historical'
    receipt=dict(selected=winner, validation=ranks, policy_sha256=base.sha256(candidate/'selection_policy.json'))
    ev.write_json(candidate/'final_selection.json',receipt,lock=True)
    print(json.dumps(receipt),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','select','predict']);p.add_argument('--release',type=Path,default=Path('model6_vitl392/release'))
    p.add_argument('--weights',type=Path,default=Path('cloud_upload/pretrained/dinov2_vitl14_pretrain.pth'))
    p.add_argument('--output',type=Path,default=Path('model6_vitl392/run'));p.add_argument('--historical-val',type=Path,default=Path('model6_vitl392/historical_val'))
    p.add_argument('--letterbox-cache-dir',type=Path,default=Path('model6_vitl392/cache'));p.add_argument('--data-root',type=Path,default=Path('.'))
    p.add_argument('--split',choices=['val','test','external_test'],default='val');p.add_argument('--batch-size',type=int,choices=[2,4,8,16],default=8)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.action=='select':select(args);return
    if not torch.cuda.is_available() or args.workers<0 or (args.action=='train' and args.split!='val'):p.error('Expected CUDA and training uses only development records')
    if args.action=='predict' and args.split!='val':
        evidence=ev.read_json(args.output/'final_selection.json')
        if (evidence['selected']!=CANDIDATE or evidence['validation'][CANDIDATE]['checkpoint_sha256']!=base.sha256(args.output/'best.pt')
                or evidence['validation'][CANDIDATE]['thresholds_sha256']!=base.sha256(args.output/'thresholds.json')
                or evidence['policy_sha256']!=base.sha256(args.output/'selection_policy.json')):
            raise ValueError('No valid validation selection authorizes held-out evaluation for this model')
    args.variant,args.precision,args.device='partial','fp16','cuda'
    args.seed,args.blocks,args.head_only_epochs,args.epochs=20261114,4,5,30
    args.accum,args.head_lr,args.backbone_lr,args.rgb_cache_dir=32//args.batch_size,.001,.00001,None
    args.config=args.output/'selection_policy.json'
    args.output.mkdir(parents=True,exist_ok=True)
    ev.write_json(args.config,POLICY,lock=True)
    audit,config,token=identity(args)
    base.seed_all(args.seed)
    with screening_letterbox.use_transform(base,config,datasets.CachedGeometryDataset):
        model=LargeGlaucoma(args.weights).to(args.device)
        if args.action=='train':base.train(args,audit,'glaucoma',model,token)
        else:base.export(model,args,audit,'glaucoma',args.split,args.output/'best.pt',token)
    split=args.split if args.action=='predict' else 'val'
    dest=args.output/f'{split}_predictions.meta.json'
    meta=ev.read_json(dest)
    meta.update(candidate=CANDIDATE,geometry_recipe_sha256=token['geometry_recipe_sha256'],transform=config,
                geometry_cache_index_sha256=base.sha256(args.letterbox_cache_dir/(split+'.json')))
    base.write_json(dest,meta)


if __name__=='__main__':main()
