"""Assemble evidence for the unchanged, already evaluated APTOS partitions.

Reads provenance, immutable original fingerprints and new audit artifacts only.
Does not move, relabel, repartition, train, select checkpoints or tune thresholds.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_training_sets import UnionFind
RUN=ROOT/'cloud_upload/aptos_vitl_20261113'
OUT=RUN/'leakage_audit'
DATA=ROOT/'data'
BASE=DATA/'prepared/source_isolated'

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def lines(path): return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]

def main():
    fresh=read(OUT/'fingerprint_recheck.json')
    oldscreen=read(OUT/'cached_pair_screen.json')
    candidates=read(OUT/'candidate_12.json')
    sift=read(OUT/'sift_12.json')
    iso=read(BASE/'isolation_manifest.json')
    config=read(RUN/'config.resolved.json')
    identity=read(RUN/'data_identity.json')
    runtime=read(RUN/'runtime.json')
    checkpoint=read(RUN/'checkpoint_metadata.json')
    combo=next(x for x in iso['combinations'] if x['base_protocol']=='strict_cross_domain' and x['task']=='dr_grading' and x['source']=='APTOS2019')
    evidence={}
    allrows=[]
    for split,n in [('train',2775),('val',359),('test',354)]:
        leaf=BASE/combo['splits'][split]['path']
        rows=lines(leaf)
        assert len(rows)==n and all(r['dataset']=='APTOS2019' and r['split']==split and r['task']=='dr_grading' for r in rows)
        digest=sha(leaf)
        assert digest==combo['splits'][split]['sha256']
        raw_source=DATA/f'prepared/tasks/dr_grading/{split}.jsonl'
        source_bytes=b''.join(b for b in raw_source.read_bytes().splitlines(keepends=True) if b.strip() and json.loads(b)['dataset']=='APTOS2019')
        assert leaf.read_bytes()==source_bytes
        evidence[split]={'path':str(leaf),'records':n,'sha256':digest,'same_raw_record_stream_as_original_global_manifest':True,'original_global_manifest':str(raw_source),'sources':dict(Counter(r['dataset'] for r in rows)), 'patient_id_known':sum(bool(r.get('patient_id')) for r in rows)}
        allrows+=rows
    for split in ['train','val']:
        assert identity['manifests_sha256'][split]==evidence[split]['sha256']
        assert all(c['manifests_sha256'][split]==evidence[split]['sha256'] for c in checkpoint['checkpoints'].values())
    assert runtime['train_records']==2775 and runtime['val_records']==359
    assert config['data']==identity['experiment']['data']
    assert sha(BASE/'isolation_manifest.json')==config['data']['isolation_manifest_sha256']
    assert config['data']['datasets']==['APTOS2019'] and config['data']['protocol']=='strict_cross_domain' and config['data']['source_isolated']
    assert config['data']['external_dataset']=='IDRiD'
    evaluation_evidence=[]
    for name in ['best','head_best']:
        meta=read(RUN/f'evaluation/{name}/predictions_test.meta.json')
        pred=lines(RUN/f'evaluation/{name}/predictions_test.jsonl')
        assert meta['manifest_sha256']==evidence['test']['sha256'] and meta['sample_count']==354 and meta['source_counts']=={'APTOS2019':354}
        assert {r['sample_id'] for r in pred}=={r['sample_id'] for r in allrows if r['split']=='test'}
        assert sha(RUN/f'evaluation/{name}/predictions_test.jsonl')==meta['jsonl_sha256']
        evaluation_evidence.append({'checkpoint':name,'metadata':str(RUN/f'evaluation/{name}/predictions_test.meta.json'),'sample_count':354,'manifest_hash_and_exact_membership_verified':True})
    dbpath=DATA/'prepared/audit/image_fingerprints.sqlite'
    db=sqlite3.connect('file:'+dbpath.resolve().as_posix()+'?mode=ro&immutable=1',uri=True)
    db.row_factory=sqlite3.Row
    global_ids=[dict(r) for r in db.execute('SELECT path,sha256,phash FROM images ORDER BY path')]
    cache=read(DATA/'prepared/audit/image_identity_cache.json')
    signature=hashlib.sha256(('identity-policy-v1|'+'|'.join(str(r['path'])+str(r['sha256'])+str(r['phash']) for r in global_ids)).encode()).hexdigest()
    assert cache['signature']==signature
    u=UnionFind(len(global_ids))
    for i,g in enumerate(cache['groups']): u.union(i,g)
    v=UnionFind(len(global_ids))
    for i in range(len(global_ids)): v.union(i,u.find(i))
    bypath={r['path']:i for i,r in enumerate(global_ids)}
    content_mismatches=[]; leakage_mismatches=[]
    for row in allrows:
        expected='content_'+hashlib.sha256(json.dumps([u.find(bypath[p]) for p in row['image_paths']]).encode()).hexdigest()[:20]
        if expected!=row['content_group_id']: content_mismatches.append(row['sample_id'])
        if 'group_'+str(v.find(bypath[row['image_paths'][0]])).zfill(6)!=row['leakage_group_id']: leakage_mismatches.append(row['sample_id'])
    assert not content_mismatches and not leakage_mismatches
    def cross_groups(field):
        groups=defaultdict(set)
        for row in allrows: groups[row[field]].add(row['split'])
        return sum(len(s)>1 for s in groups.values())
    assert cross_groups('content_group_id')==0 and cross_groups('leakage_group_id')==0
    # Independently reproduce the ORIGINAL strict composite gate for all pairs
    # across train/val/test. This is a fixed policy, not a new detection threshold.
    fp={r['path']:dict(r) for r in db.execute("SELECT * FROM images WHERE path LIKE 'APTOS2019/%'")}
    db.close()
    bysplit={s:[r for r in allrows if r['split']==s] for s in ['train','val','test']}
    near_counts={}; hash_candidates=[]; strict_pairs=[]
    minimums={'phash':64,'dhash':64}
    for left,right in [('test','train'),('test','val'),('train','val')]:
        counter={'compared_pairs':0,'phash_le_2_dhash_le_4':0,'strict_composite_match':0}
        for arow in bysplit[left]:
            a=fp[arow['image_path']]
            for brow in bysplit[right]:
                b=fp[brow['image_path']]
                counter['compared_pairs']+=1
                pd=(int(a['phash'],16)^int(b['phash'],16)).bit_count()
                dd=(int(a['dhash'],16)^int(b['dhash'],16)).bit_count()
                minimums['phash']=min(minimums['phash'],pd); minimums['dhash']=min(minimums['dhash'],dd)
                if pd>2 or dd>4: continue
                counter['phash_le_2_dhash_le_4']+=1
                aa=np.frombuffer(a['thumb'],dtype=np.uint8).astype(np.float32)
                bb=np.frombuffer(b['thumb'],dtype=np.uint8).astype(np.float32)
                rmse=float(np.sqrt(np.mean((aa-bb)**2))/255)
                ag=aa.reshape(-1,3).mean(1); bg=bb.reshape(-1,3).mean(1)
                ag-=ag.mean(); bg-=bg.mean()
                den=float(np.linalg.norm(ag)*np.linalg.norm(bg))
                corr=float(ag.dot(bg)/den) if den else 0.
                aspect=abs((a['width']/a['height'])/(b['width']/b['height'])-1)
                item={'image_a':arow['image_path'],'split_a':left,'image_b':brow['image_path'],'split_b':right,'phash_distance':pd,'dhash_distance':dd,'thumbnail_rgb_rmse_0_1':rmse,'thumbnail_gray_correlation':corr,'relative_aspect_ratio_delta':aspect}
                hash_candidates.append(item)
                if not a['error'] and not b['error'] and a['gray_std']>=5 and b['gray_std']>=5 and aspect<=.025 and rmse<=.02 and corr>=.9997:
                    strict_pairs.append(item); counter['strict_composite_match']+=1
        near_counts[left+'_vs_'+right]=counter
    assert not strict_pairs
    notes=[
        'Different superior branching and inferior temporal branch geometry are visible.',
        'Disc vascular pattern and superior/inferior branch trajectories differ.',
        'Inferior vascular separation and superior arch geometry differ.',
        'Inferior branch junction and disc vascular origin differ.',
        'Superior cross-field branches and inferior arch branching differ.',
        'Inferior junction positions and superior vessel topology differ.',
        'Superior arch and disc branching configuration differ.',
        'Similar coarse field/appearance and photometric differences. Fine branch correspondence was not reliably established; keep as unresolved same-eye risk, not a confirmed duplicate.',
        'Train image has conspicuous perimacular lesions and different branch geometry; direct same-image transform is not supported.',
        'Inferior trunk junction and disc vessel configurations differ.',
        'Inferior branch junction positions and superior trajectories differ.',
        'Inferior arch approach and vertical branch arrangement differ.',
    ]
    review=[]
    for p,s,note in zip(candidates['pairs'],sift['pairs'],notes):
        review.append({**p,'review_status':'unresolved_not_confirmed' if p['audit_pair_id']==8 else 'visual_nonmatch_evidence_not_clinical_identity_proof','visual_observation':note,'sift_ratio_075':s['thresholds']['0.75'],'sift_inlier_sheet':s['inlier_sheet']})
    assert sum(x['other_split']=='train' for x in candidates['pairs'])==12
    untouched=[{'path':r['path'],'expected_sha256':r['sha256'],'current_sha256':sha(r['path'])} for r in fresh['input_files']]
    assert all(r['expected_sha256']==r['current_sha256'] for r in untouched)
    report={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Audit unchanged downstream fine-tuning split membership and known duplicate-screen coverage, not pretraining membership or guaranteed clinical-patient independence.',
        'decision':'Do not claim that APTOS test354 is proven free of every near duplicate. No exact file/full decoded-pixel cross-split duplicate was found; no pair passed the original strict composite near-match rule. No same-eye duplicate was confirmed by this limited visual review. Existing split and metrics remain unchanged.',
        'user_direction':{'reported_instruction':'有些相似没关系，直接用于训练','action':'Continue with existing fixed datasets as authorized; document uncertainty without a new launch gate, threshold change or split change.'},
        'split_manifest_evidence':evidence,
        'runtime_chain':{'config':str(RUN/'config.resolved.json'),'data_identity':str(RUN/'data_identity.json'),'runtime':str(RUN/'runtime.json'),'checkpoint_metadata':str(RUN/'checkpoint_metadata.json'),'isolation_index':str(BASE/'isolation_manifest.json'),'index_sha256':sha(BASE/'isolation_manifest.json'),'runtime_train_val':[runtime['train_records'],runtime['val_records']],'checkpoint_train_val_manifest_hashes_match':True,'dataset_filter':['APTOS2019'],'protocol':'strict_cross_domain','external_source':'IDRiD','evaluation_evidence':evaluation_evidence},
        'global_fingerprint_group_chain':{'database':str(dbpath),'global_images':len(global_ids),'identity_cache':str(DATA/'prepared/audit/image_identity_cache.json'),'cache_signature_matches_database':True,'cache_reconstruction_method':'Reproduce UnionFind cache loading order in build_image_groups, then image-only initialization of build_leakage_groups. All APTOS content and leakage roots match; all APTOS patient IDs are unknown.','content_group_id_mismatches':content_mismatches,'leakage_group_id_mismatches':leakage_mismatches,'cross_split_content_groups':cross_groups('content_group_id'),'cross_split_leakage_groups':cross_groups('leakage_group_id'),'builder':str(ROOT/'scripts/build_training_sets.py'),'isolated_builder':str(ROOT/'scripts/isolate_training_datasets.py'),'fingerprint_builder':str(ROOT/'scripts/image_fingerprints.py'),'original_near_edges_touching_retained_APTOS':oldscreen['original_near_edges_touching_APTOS_retained'],'original_cross_split_near_edges':oldscreen['original_cross_split_edges']},
        'fresh_all_image_recheck':fresh,
        'fixed_original_near_rule':{'phash_hamming_max':2,'dhash_hamming_max':4,'relative_aspect_ratio_delta_max':.025,'rgb32_rmse_max':.02,'gray32_rgb_channel_mean_correlation_min':.9997,'gray_std_min':5,'candidate_index':'4 x16-bit pHash bands; pHash <=2 necessarily shares at least one band. Fresh audit compares every APTOS cross-split pair directly.','all_conditions_required':True,'cross_split_pair_counts':near_counts,'strict_composite_matches':strict_pairs,'minimum_hash_distances_all_cross_split':minimums,'note':'A zero count for this deliberately conservative rule is not proof of no same-eye or transformed near duplicates.'},
        'candidate_review':{'test_vs_train_val_hash_only_candidates':12,'unique_test_images':len({p['test_image'] for p in candidates['pairs']}),'visually_inspected_pairs':12,'confirmed_same_eye_or_same_image':0,'unresolved_pairs':[8],'additional_train_vs_val_hash_only_candidates_not_visually_reviewed':near_counts['train_vs_val']['phash_le_2_dhash_le_4'],'pairs':review,'supplemental_local_feature_method':sift['method'],'sift_report':str(OUT/'sift_12.json'),'scope':'Triage of these 12 fixed test-related candidates only, not an exhaustive clinical identity assessment. No SIFT decision threshold was used to clear the dataset.'},
        'broader_screen':{'phash_hamming_max':8,'dhash_hamming_max':12,'test_vs_train_val_candidates':oldscreen['counts']['broad'],'full_visual_review_completed':False,'artifact':str(OUT/'cached_pair_screen.json'),'note':'These are similarity candidates, not established duplicates. No additional full review or model adjustment is made after user acceptance of the fixed split.'},
        'inputs_still_unchanged':untouched,
        'limitations':['Patient IDs are absent for all 3488 retained APTOS images; same-patient independence cannot be proven.','Perceptual hash screening is not invariant to arbitrary crop, rotation, resizing, color processing or blur; near duplicates may evade the fixed gates.','12 narrow test-related candidates were reviewed; broader 4346 test-related candidates and 9 additional narrow train-versus-validation candidates were not all visually reviewed. An unresolved risk remains; no confirmed contamination rate is claimed.','Full decoded RGB SHA detects identical decoded pixels only. It does not detect a different photograph of the same eye.','The existing test has already been evaluated; any subsequent exclusion analysis would be a sensitivity analysis, never a new blind holdout.','Public pretrained DINOv2 corpus membership is not audited. This report concerns downstream train/validation/test data.'],
        'actions_not_taken':['No source image, label, manifest, fingerprint database or original evaluation report changed.','No re-splitting, retraining, checkpoint selection or threshold tuning performed.','No server operation performed.'],
    }
    dest=OUT/'audit_summary.json'
    dest.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'cross_split_hash_candidates.json').write_text(json.dumps({'fixed_rule_hash_gate':'pHash <=2 AND dHash <=4; all cross train/val/test pairs','counts':near_counts,'pairs':hash_candidates},indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'report':str(dest),'fresh_images':fresh['images_recomputed'],'pair_counts':near_counts,'confirmed_same_eye':0,'unresolved_pair_ids':[8]},ensure_ascii=False))

if __name__=='__main__': main()
