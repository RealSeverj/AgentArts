"""Prepare the glaucoma release's lossless 392px geometry and transfer index."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
import random
import tarfile

from PIL import Image

from training import screening_letterbox_cache as cache


def build(args):
    root=args.data_root.resolve()
    policy=cache.recipe_sha()
    audit=json.loads((args.release/'audit.json').read_bytes())
    if policy!='dc9eba4e907231ecfe2319a7ea6d1658b6e10b9d2142dde667b0eba6e7199459':
        raise ValueError('Expected the original 392px preprocessing recipe, Pillow 11.3.0 and NumPy 1.26.4')
    paths=set()
    for split in args.splits:
        key='glaucoma/'+split
        raw=(args.release/(key+'.jsonl')).read_bytes()
        if cache.sha(raw)!=audit['manifests'][key]['sha256']:raise ValueError('Source manifest SHA mismatch')
        cohort=[json.loads(line) for line in raw.splitlines()]
        rows=[r for r in cohort if split!='train' or any(r['mask'])]
        jobs=[(root,args.cache,policy,r['image_path'],r['sample_id'],r['image_sha256']) for r in rows]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            records=[]
            for item in pool.map(cache.make_image,jobs,chunksize=4):
                records.append(item)
                if len(records)%200==0:print(f'{split} {len(records)}/{len(rows)}',flush=True)
        check=random.Random(20261114).sample(range(len(rows)),min(16,len(rows)))
        for i in check:
            rgb=cache.geometry(root/rows[i]['image_path'],rows[i]['image_sha256'])
            original=rgb.tobytes()
            png=cache.image_path(args.cache,policy,rows[i]['image_sha256']).read_bytes()
            with Image.open(io.BytesIO(png)) as encoded:
                if encoded.tobytes()!=original or encoded.size!=(392,392) or encoded.mode!='RGB':raise ValueError('Decoded pixels differ')
        spec=dict(split=split,source_manifest_sha256=cache.sha(raw),recipe=cache.recipe(),recipe_sha256=policy,records=records,
                  full_manifest_images=len(cohort),verified_ids=[rows[i]['sample_id'] for i in check])
        index=args.cache/(split+'.json')
        cache.atomic(index,(json.dumps(spec,indent=2)+'\n').encode())
        paths.add(index)
        for r in records:
            path=cache.image_path(args.cache,policy,r['image_sha256'])
            if cache.sha(path.read_bytes())!=r['png_sha256']:raise ValueError('PNG digest mismatch')
            paths.add(path)
        print(f'{split}: {len(rows)} records, {len(check)} independently decoded comparisons',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.partial')
    with tarfile.open(temp,'w') as tar:
        for p in sorted(paths):
            tar.add(p,arcname=(Path('model6_vitl392/cache')/p.relative_to(args.cache)).as_posix(),recursive=False)
    temp.replace(args.output)
    print(f'{args.output} bytes={args.output.stat().st_size} sha256={cache.sha(args.output.read_bytes())}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('.'));p.add_argument('--release',type=Path,default=Path('data/prepared/screening_v1'))
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--splits',nargs='+',required=True,choices=['train','val','test','external_test'])
    build(p.parse_args())
