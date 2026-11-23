"""Supplemental local-feature audit, not a clinical identity classifier.

Reads source photographs and a fixed candidate list, writes audit reports only.
Camera border is masked; green channel CLAHE exposes vascular detail. SIFT +
ratio test and RANSAC homography are evidence to review, not proof of absence.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]

def features(path):
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
    scale = min(1., 1000 / max(rgb.shape[:2]))
    rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    green = rgb[:, :, 1]
    mask = np.uint8(green > 14) * 255
    mask = cv2.erode(mask, np.ones((51, 51), np.uint8))
    gray = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(green)
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=.008, edgeThreshold=15)
    k, d = sift.detectAndCompute(gray, mask)
    return rgb, k, d

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidates', type=Path, default=ROOT/'cloud_upload/aptos_vitl_20261113/leakage_audit/candidate_12.json')
    parser.add_argument('--output-prefix', default='sift_12')
    args=parser.parse_args()
    cv2.setNumThreads(1)
    source=json.loads(args.candidates.read_text(encoding='utf-8'))
    pairs=source['pairs']
    paths=sorted(set(p['test_image'] for p in pairs) | set(p['other_image'] for p in pairs))
    with ThreadPoolExecutor(max_workers=6) as pool:
        loaded=dict(zip(paths, pool.map(lambda p:features(ROOT/'data'/p), paths)))
    records=[]
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    for i,p in enumerate(pairs,1):
        rgb1,k1,d1=loaded[p['test_image']]
        rgb2,k2,d2=loaded[p['other_image']]
        raw=matcher.knnMatch(d1,d2,k=2) if d1 is not None and d2 is not None else []
        result={k:p[k] for k in ['test_image','other_image']}
        result.update(pair_id=p.get('audit_pair_id',i), keypoints=[len(k1),len(k2)], thresholds={})
        for ratio in [.70,.75,.80]:
            matches=[m for m,n in raw if m.distance < ratio*n.distance]
            inliers=[]
            homography=None
            if len(matches)>=4:
                src=np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                dst=np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                h,mask=cv2.findHomography(src,dst,cv2.RANSAC,3.,maxIters=5000,confidence=.999)
                if h is not None and mask is not None:
                    homography=h.tolist()
                    inliers=[m for m,keep in zip(matches,mask.ravel()) if keep]
            result['thresholds'][str(ratio)]={'ratio_matches':len(matches), 'homography_inliers':len(inliers), 'homography':homography}
            if ratio==.75:
                draw=cv2.drawMatches(cv2.cvtColor(rgb1,cv2.COLOR_RGB2BGR),k1,cv2.cvtColor(rgb2,cv2.COLOR_RGB2BGR),k2,inliers,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.putText(draw, f"Pair {result['pair_id']} | SIFT ratio .75 {len(matches)} matches, {len(inliers)} RANSAC inliers | exploratory, not identity proof",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA)
                dest=args.candidates.parent/f"{args.output_prefix}_pair_{result['pair_id']:02d}.jpg"
                dest.write_bytes(cv2.imencode('.jpg',draw,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tobytes())
                result['inlier_sheet']=str(dest)
        records.append(result)
    out=args.candidates.parent/f'{args.output_prefix}.json'
    out.write_text(json.dumps({'method':'SIFT on 1000px maximum-dimension green-channel CLAHE, mask eroded 51px to remove camera rim; BF L2 ratio tests; RANSAC homography reprojection 3px. Descriptive supplemental evidence, no identity decision threshold.', 'opencv_version':cv2.__version__, 'pairs':records, 'limitations':['Sparse texture or blur can produce false negatives.','Homography can yield incidental local matches between different retinas.','Not a patient identifier and does not establish clinical independence.']},indent=2),encoding='utf-8')
    print(json.dumps([{'pair':r['pair_id'],'test':Path(r['test_image']).stem,'other':Path(r['other_image']).stem,'matches/inliers':{q:[v['ratio_matches'],v['homography_inliers']] for q,v in r['thresholds'].items()}} for r in records]))

if __name__=='__main__':
    main()
