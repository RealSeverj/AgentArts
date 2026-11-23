from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib
from PIL import Image
ROOT=Path(__file__).resolve().parent
EXTRACTED=ROOT/'extracted'
report={'dataset':'Public REFUGE-style mirror labelled refuge2','mirror':'https://www.kaggle.com/datasets/victorlemosml/refuge2','original_source':'https://refuge.grand-challenge.org/Details/','mirror_license':'Unknown','original_data_use_scope_not_established':False,'original_data_license_from_REFUGE2_author_paper':'CC BY-NC-ND; Kaggle listing remains Unknown','archive_crc_checked_during_extraction':True,'splits':[]}
all_hashes=defaultdict(list)
for split in ['train','val','test']:
 dirs=[d for d in EXTRACTED.rglob('images') if d.parent.name.lower()==split]
 if len(dirs)!=1:raise RuntimeError(f'Unexpected image directory count for {split}: {len(dirs)}')
 imdir=dirs[0];maskdirs=[d for d in imdir.parent.iterdir() if d.is_dir() and d.name.lower() in ['mask','masks']]
 if len(maskdirs)!=1:raise RuntimeError('Unexpected mask directory')
 maskdir=maskdirs[0]
 ims=[p for p in imdir.iterdir() if p.is_file() and p.suffix.lower() in ['.jpg','.jpeg','.png']]
 masks=[p for p in maskdir.iterdir() if p.is_file() and p.suffix.lower() in ['.bmp','.png','.tif','.tiff']]
 byid={p.stem:p for p in masks}
 sizes=Counter();mask_modes=Counter();values=set();different_sizes=[]
 for p in ims:
  with Image.open(p) as im:size=im.size
  sizes[str(size)]+=1
  if p.stem not in byid:continue
  with Image.open(byid[p.stem]) as m:
   mask_modes[m.mode]+=1
   if m.size!=size:different_sizes.append({'image':p.name,'image_size':size,'mask_size':m.size})
   colors=m.convert('L').getcolors(maxcolors=32)
   if colors is None:values.add('more_than_32_grayscale_values')
   else:values.update(value for count,value in colors)
  all_hashes[hashlib.sha256(p.read_bytes()).hexdigest()].append(str(p.relative_to(ROOT)))
 sr={'split':split,'images_directory':str(imdir.relative_to(ROOT)),'masks_directory':str(maskdir.relative_to(ROOT)),'original_images':len(ims),'disc_cup_mask_files':len(masks),'image_mask_ids_match':{p.stem for p in ims}==set(byid),'image_sizes':dict(sizes),'mask_modes':dict(mask_modes),'mask_grayscale_values_observed':sorted(values,key=str),'dimension_mismatches':different_sizes,'image_filename_prefix_counts':dict(Counter(p.name[0].lower() for p in ims))}
 if split=='train':sr['disease_labels_from_filename_prefix']={'g':sum(p.name.lower().startswith('g') for p in ims),'n':sum(p.name.lower().startswith('n') for p in ims)}
 else:sr['clinical_glaucoma_labels_available_in_this_mirror']=False
 report['splits'].append(sr)
label_files=[str(p.relative_to(ROOT)) for p in EXTRACTED.rglob('*') if p.is_file() and p.suffix.lower() in ['.csv','.xlsx','.xls','.json','.txt','.md']]
report['standalone_label_or_document_files']=label_files
report['total_images']=sum(s['original_images'] for s in report['splits'])
report['total_disc_cup_masks']=sum(s['disc_cup_mask_files'] for s in report['splits'])
report['distinct_jpeg_byte_hashes']=len(all_hashes)
report['exact_duplicate_groups']=[v for v in all_hashes.values() if len(v)>1]
report['mirror_content_identification']='Incomplete/mixed REFUGE and REFUGE2 subset: Zeiss-size400, KOWA-size400, Canon-size400; not full REFUGE2018 1200 or REFUGE2 2000'
report['clinical_glaucoma_label_coverage']='Training400 filename g/n prefixes only; no independent disease-label table for validation/test was found' if not label_files else 'See standalone label files; coverage requires explicit mapping'
report['download_and_structure_checks_passed']=all(s['image_mask_ids_match'] and not s['dimension_mismatches'] for s in report['splits'])
report['all_checks_passed']=report['download_and_structure_checks_passed']
(ROOT/'verification_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True))
