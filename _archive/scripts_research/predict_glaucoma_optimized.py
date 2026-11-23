"""Apply the validation-selected glaucoma recipe to one original image on CPU."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import optimize_glaucoma_screening as opt
from training import classification, screening_training as disease, structural


def predict(record, pretrained, image):
    spec = opt.read(record)
    selected = spec['candidate']
    if selected not in opt.ORDER:
        raise ValueError('Unknown recipe')
    rule = Path(spec['threshold_file'])
    lock = opt.read(rule)
    if disease.sha256(rule)!=spec['threshold_file_sha256'] or lock['checkpoint_sha256']!=spec['inference_recipe_sha256']:
        raise ValueError('Changed threshold identity')
    if selected!='historical' and disease.sha256(Path(spec['inference_recipe_file'])) != spec['inference_recipe_sha256']:
        raise ValueError('Changed inference recipe')
    threshold = lock['per_label']['glaucoma']['threshold']
    with Image.open(image) as original:
        rgb=original.convert('RGB')
    components = {}
    with torch.inference_mode():
        if selected in ('historical','hybrid','flip_tta'):
            weight=Path(spec['disease_model_file'])
            if disease.sha256(weight)!=opt.DRIVE_SHA or disease.sha256(pretrained)!=disease.OFFICIAL_SHA:
                raise ValueError('Changed glaucoma model identity')
            net=disease.ScreeningModel('glaucoma',pretrained).eval().requires_grad_(False)
            net.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True,mmap=True)['model'],strict=True)
            image_tensor=classification.ClassificationTransform(disease.TRANSFORM_CONFIG,training=False)(rgb).unsqueeze(0)
            if selected!='historical':
                image_tensor=torch.cat((image_tensor,image_tensor.flip((-1,))),dim=0)
            raw=net(image_tensor).float().sigmoid().flatten().tolist()
            components['image_probability']=sum(raw)/len(raw)
        if selected in ('hybrid','structural'):
            weight,head_path=Path(spec['segmentation_model_file']),Path(spec['structural_head'])
            if disease.sha256(weight)!=opt.UNET_SHA or disease.sha256(head_path)!=spec['structural_head_sha256']:
                raise ValueError('Changed cup-disc model identity')
            state=torch.load(weight,map_location='cpu',weights_only=False,mmap=True)
            seg=structural.build_model(state['config'],pretrained=False).eval().requires_grad_(False).to(memory_format=torch.channels_last)
            seg.load_state_dict(state['model'],strict=True)
            sx=structural._tensor(rgb.resize((512,512),Image.Resampling.BILINEAR),state['config'])
            mask=(seg(sx.unsqueeze(0).contiguous(memory_format=torch.channels_last))[0]>=0).numpy()
            features=opt.masks_to_features(mask[0],mask[1]); head=opt.read(head_path)
            x=np.asarray(features,dtype=float)
            x=(np.where(np.isfinite(x),x,head['median'])-head['mean'])/head['std']
            coefficients=np.asarray(head['coefficients'])
            components['structural_probability']=float(1/(1+np.exp(-np.clip(coefficients[0]+x@coefficients[1:],-40,40))))
    probability=sum(components.values())/len(components)
    return dict(candidate=selected,label='glaucoma',probability=probability,threshold=threshold,
                screen_positive=probability>=threshold,components=components,recipe_sha256=spec['inference_recipe_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True,help='Final model manifest from the summarized round')
    p.add_argument('--pretrained',type=Path,default=Path('cloud_upload/pretrained/dinov2_vitb14_pretrain.pth'))
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    print(json.dumps(predict(args.model,args.pretrained,args.image),ensure_ascii=False,allow_nan=False))
