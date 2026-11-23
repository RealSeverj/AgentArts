from pathlib import Path
from collections import Counter,defaultdict
import csv,json,hashlib,re
from PIL import Image
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'extracted'
def rows(path):
 with path.open(encoding='utf-8-sig',newline='') as f:
  r=csv.DictReader(f);out=list(r)
  return r.fieldnames,out
report={'dataset':'IDRiD','source':'Third-party Zenodo mirror https://zenodo.org/records/17219542; not downloaded from IEEE publisher','original_source':'https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid','license':'CC BY 4.0; license files included in source archives','mirror_published_md5_verified':True,'extraction_crc_verified_by_7zip':True,'segmentation':[],'grading':[],'localization':[]}
for split,n in [('a. Training Set',54),('b. Testing Set',27)]:
 imdir=BASE/'A. Segmentation'/'1. Original Images'/split
 ims=list(imdir.glob('*.jpg'));ids={p.stem for p in ims};maskdir=BASE/'A. Segmentation'/'2. All Segmentation Groundtruths'/split
 counts={};mismatches=[]
 for d in maskdir.iterdir():
  if not d.is_dir():continue
  masks=list(d.glob('*.tif'));counts[d.name]=len(masks)
  for m in masks:
   image_id=re.match(r'(IDRiD_\d+)',m.stem).group(1)
   assert image_id in ids
   with Image.open(imdir/(image_id+'.jpg')) as im:sz=im.size
   with Image.open(m) as im:msz=im.size
   if sz!=msz:mismatches.append(m.name)
 assert len(ims)==n and not mismatches
 report['segmentation'].append({'split':split,'original_images':len(ims),'mask_counts':counts,'all_mask_ids_have_images':True,'all_mask_dimensions_match':True})
grade_hashes=defaultdict(list)
for split,n,cs in [('a. Training Set',413,'a. IDRiD_Disease Grading_Training Labels.csv'),('b. Testing Set',103,'b. IDRiD_Disease Grading_Testing Labels.csv')]:
 p=BASE/'B. Disease Grading'/'2. Groundtruths'/cs
 fields,rr=rows(p);ims=list((BASE/'B. Disease Grading'/'1. Original Images'/split).glob('*.jpg'))
 assert len(rr)==n and len(ims)==n and {x['Image name'] for x in rr}=={p.stem for p in ims}
 sizes=Counter()
 for p in ims:
  grade_hashes[hashlib.sha256(p.read_bytes()).hexdigest()].append(str(p.relative_to(ROOT)))
  with Image.open(p) as im:sizes[str(im.size)]+=1
 assert all(x['Retinopathy grade'] in {'0','1','2','3','4'} and x['Risk of macular edema '] in {'0','1','2'} for x in rr)
 report['grading'].append({'split':split,'original_image_records':len(ims),'label_rows':len(rr),'image_label_ids_match':True,'image_sizes':dict(sizes),'dr_grade_counts':dict(Counter(x['Retinopathy grade'] for x in rr)),'dme_risk_counts':dict(Counter(x['Risk of macular edema '] for x in rr)),'raw_csv_headers':fields})
for split,n in [('a. Training Set',413),('b. Testing Set',103)]:
 ims=list((BASE/'C. Localization'/'1. Original Images'/split).glob('*.jpg'))
 same=all(p.read_bytes()==(BASE/'B. Disease Grading'/'1. Original Images'/split/p.name).read_bytes() for p in ims)
 assert len(ims)==n
 report['localization'].append({'split':split,'image_records':len(ims),'images_byte_identical_to_grading_by_same_name':same})
loc=[]
for p in (BASE/'C. Localization'/'2. Groundtruths').rglob('*.csv'):
 fields,rr=rows(p);valid=[r for r in rr if (r.get('Image No') or '').strip()]
 split='a. Training Set' if 'Training' in p.name else 'b. Testing Set'
 ids={x.stem for x in (BASE/'C. Localization'/'1. Original Images'/split).glob('*.jpg')}
 assert {r['Image No'].strip() for r in valid}==ids
 loc.append({'csv':str(p.relative_to(ROOT)),'raw_rows_including_blank_filler':len(rr),'nonempty_coordinate_rows':len(valid),'image_ids_match':True})
report['localization_coordinate_tables']=loc
report['grading_image_records']=sum(s['original_image_records'] for s in report['grading'])
report['grading_distinct_jpeg_byte_hashes']=len(grade_hashes)
report['segmentation_image_records']=81
report['segmentation_is_same_source_subset_not_independent_external_test']=True
report['localization_images_are_additional_copies_not_independent_samples']=all(s['images_byte_identical_to_grading_by_same_name'] for s in report['localization'])
report['duplicate_groups_file']='grading_exact_duplicate_groups.json'
report['independent_mirror_crosscheck']='independent_mirror_crosscheck/comparison_report.json'
report['independent_mirror_labels_crosscheck']='independent_mirror_crosscheck/label_comparison_report.json'
report['download_and_structure_checks_passed']=True
report['data_quality_status']='requires_cleaning_before_training'
report['data_quality_warnings']=[
 'The acquired mirror grading package contains 516 image records but 510 distinct complete-file SHA256 hashes; six exact duplicate groups are preserved.',
 'Some duplicate image groups have conflicting DR or DME labels; one exact image occurs in training IDRiD_118 and test IDRiD_064 with DR labels 0 and 3.',
 'The cross-split image pair and both conflicting labels were independently verified in a second public Kaggle mirror. Neither check independently obtained the IEEE original, so this is evidence about available mirrors, not a claim about author intent or provenance.',
 'Only three segmentation JPEG files are byte-identical to grading files in this audit. Other source-image relationships require additional image-level matching; dataset collections must not be treated as independent external sets.',
 'Missing lesion-mask files are not automatically negative labels; missingness semantics were not independently confirmed.',
 'CSV tables contain trailing empty columns/rows and a DME header with trailing whitespace; preserve originals and normalize only in downstream preprocessing.'
]
(ROOT/'verification_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'download_and_structure_checks_passed':True,'grading_records':report['grading_image_records'],'grading_distinct_jpeg_bytes':len(grade_hashes),'segmentation_images':81,'data_quality_status':report['data_quality_status'],'localization_copies_match_grading':report['localization_images_are_additional_copies_not_independent_samples']},ensure_ascii=True))
