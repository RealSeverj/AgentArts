"""One fixed weight per row, four rows x three complete cohorts; no training."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_dr_evaluation import load_manifest, load_predictions, read_json, sha256_file, summarize_predictions, assert_report_matches
from scripts.summarize_aptos_vitl import checked_matrix
from scripts.report_dr_safety_metrics import bootstrap_metrics, prioritized_errors, actual_stage

COHORTS={'APTOS2019':('test',354),'DDR':('test',3898),'IDRiD':('external_test',506)}
MODEL_SPECS={'aptos_head':('aptos_vitl_20261113','head_best','APTOS2019'),
             'aptos_partial':('aptos_vitl_20261113','best','APTOS2019'),
             'ddr_head':('ddr_from_aptos_20261113','head_best','DDR'),
             'ddr_partial':('ddr_from_aptos_20261113','best','DDR')}

def require(ok,message):
    if not ok: raise ValueError(message)

def normalized(config):
    c=deepcopy(config); c.pop('notes',None)
    c['training']={k:v for k,v in c.get('training',{}).items() if k not in ('workers','log_every')}
    return c

def build_report(cloud_root,data_root,output_root,replicates=5000,seed=20261113):
    cloud_root,data_root,output_root=map(Path,(cloud_root,data_root,output_root))
    inputs=set()
    def read(p): inputs.add(p); return read_json(p)
    index_path=data_root/'prepared/source_isolated/isolation_manifest.json'
    index=read(index_path); index_sha=sha256_file(index_path)
    require(index.get('all_checks_passed') is True and index.get('record_bytes_preserved') is True,'Isolation index was not successfully audited')
    manifests={}; manifest_evidence={}
    for source,(split,count) in COHORTS.items():
        candidates=[x for x in index['combinations'] if (x['base_protocol'],x['task'],x['source'])==('strict_cross_domain','dr_grading',source)]
        require(len(candidates)==1,'Ambiguous source-isolated cohort')
        info=candidates[0]['splits'][split]
        path=data_root/'prepared/source_isolated'/info['path']
        require(info['path']==f'strict_cross_domain/tasks/dr_grading/{source}/{split}.jsonl'
                and info['records']==count and info['sha256']==sha256_file(path)
                and info['bytes']==path.stat().st_size,'Cohort bytes/count/hash differ from isolation index')
        inputs.add(path)
        manifest=load_manifest(path,split,count)
        require({r['dataset'] for r in manifest.values()}=={source},'Manifest source is mixed')
        require(sum(bool(l.strip()) for l in path.read_text(encoding='utf-8-sig').splitlines())==count,'Extra unknown-label manifest rows')
        manifests[source]=manifest
        manifest_evidence[source]={'source':source,'split':split,'n':count,'manifest_path':str(path.resolve()),'manifest_sha256':info['sha256'],
                                   'target_support':[sum(r['labels']['dr_grade']==g for r in manifest.values()) for g in range(5)]}
    rows={}; predictions={source:{} for source in COHORTS}
    for model_id,(run_name,folder,training_source) in MODEL_SPECS.items():
        run=cloud_root/run_name
        metadata=read(run/'checkpoint_metadata.json')
        config=read(run/'config.resolved.json')
        identity=read(run/'data_identity.json')
        require(metadata.get('all_checks_passed') is True,'Checkpoint metadata failed its audit')
        entry=metadata['checkpoints'][folder+'.pt']
        require(config.get('task')=='dr_grading' and config.get('kind')=='multiclass' and config.get('labels')==['dr_grade']
                and config.get('num_classes')==5 and config['data'].get('datasets')==[training_source]
                and config['data'].get('source_isolated') is True and config['data'].get('protocol')=='strict_cross_domain'
                and config['data'].get('isolation_manifest_sha256')==index_sha,'Original checkpoint configuration mismatch')
        require(identity['experiment']==normalized(config),'Resolved config contradicts saved original data identity')
        require(entry['manifests_sha256']==identity['manifests_sha256'],'Checkpoint train/val hash identity differs')
        history=['APTOS2019','DDR'] if training_source=='DDR' else ['APTOS2019']
        if training_source=='DDR':
            lineage=read(run/'initialization_lineage.json')
            require(entry.get('initialization_lineage')==lineage and lineage['supervised_training_source_history']==history,'Transfer supervision history differs')
        stage,stage_basis=actual_stage(run,entry['completed_epoch'],config,entry)
        for name in ('history.jsonl','head_history.jsonl'):
            if (run/name).is_file(): inputs.add(run/name)
        rows[model_id]={'checkpoint_name':folder+'.pt','checkpoint_completed_epoch':entry['completed_epoch'],
            'checkpoint_sha256':entry['sha256'],'model_state_sha256':entry.get('model_state_sha256'),
            'actual_stage':stage,'stage_basis':stage_basis,'model_name':config['model']['name'],
            'current_run_training_sources':[training_source],'training_history_sources':history,
            'single_fixed_weight_across_all_cohorts':True,'cohorts':{}}
        checkpoint_paths=set()
        for source,(split,count) in COHORTS.items():
            cross=source not in (training_source,'IDRiD')
            directory=output_root/model_id/source if cross else run/'evaluation'/folder
            report_path=directory/f'evaluation_{split}.json'
            pred_path=directory/f'predictions_{split}.jsonl'
            meta_path=directory/f'predictions_{split}.meta.json'
            report,meta=read(report_path),read(meta_path)
            checkpoint_paths.add(report.get('checkpoint'))
            require(report.get('config')==config,'Evaluation original model configuration differs from fixed weight run')
            require(str(report.get('checkpoint','')).replace('\\','/').endswith('/'+folder+'.pt'),'Checkpoint filename/role differs')
            require(meta.get('checkpoint')==report.get('checkpoint') and meta.get('checkpoint_completed_epoch')==entry['completed_epoch'],'Export checkpoint epoch/path differs')
            require(meta.get('split')==split and meta.get('manifest_sha256')==manifest_evidence[source]['manifest_sha256']
                    and meta.get('jsonl_sha256')==sha256_file(pred_path) and meta.get('sample_count')==count
                    and meta.get('expected_dataset_count')==count and meta.get('unique_sample_ids')==count
                    and meta.get('source_counts')=={source:count} and meta.get('all_targets_known') is True
                    and meta.get('complete_dataset_evaluated') is True and meta.get('class_order')==list(range(5))
                    and meta.get('target_label')=='dr_grade','Prediction metadata SHA/source/count/label contract differs')
            if cross:
                require(report.get('original_config')==config and report.get('original_data_identity')==identity,'Cross-cohort exporter altered original config/data identity')
                require(report.get('checkpoint_sha256')==entry['sha256'] and meta.get('checkpoint_sha256')==entry['sha256']
                        and report.get('checkpoint_completed_epoch')==entry['completed_epoch'],'Cross-cohort checkpoint actual SHA/epoch differs')
                require(report.get('evaluation_mode')=='explicit_cross_cohort' and all(report.get(k) is False for k in ('training_performed','optimizer_loaded','initialization_applied','model_weights_updated')),
                        'Cross-cohort evaluation updated or initialized model weights')
                require(report.get('cross_cohort_authorization',{}).get('original_config_mutated') is False
                        and report.get('cross_cohort_authorization',{}).get('original_data_identity_replaced') is False,'Cross-cohort identity flags differ')
                cohort_config=report['evaluation_cohort_config']
                require(cohort_config['model']==config['model'] and cohort_config['data'].get('datasets')==[source]
                        and cohort_config['data'].get('isolation_manifest_sha256')==index_sha
                        and report.get('manifest_sha256')==manifest_evidence[source]['manifest_sha256']
                        and report['cohort'].get('source')==source,'Cross-cohort evaluation routing differs')
                require(report.get('supervision_history',{}).get('actual_supervised_source_history')==history,'Cross-cohort supervision history differs')
            elif meta.get('checkpoint_sha256') is not None:
                require(meta['checkpoint_sha256']==entry['sha256'],'Standard export checkpoint SHA differs')
            require(report.get('evaluated_dataset_count')==count,'Incomplete evaluation report')
            checked_matrix(report.get('metrics'),count,str(report_path))
            exported=load_predictions(pred_path,manifests[source]); inputs.add(pred_path)
            metrics=summarize_predictions(sorted(manifests[source]),manifests[source],exported)
            assert_report_matches(report,split,metrics,{source:metrics},str(report_path))
            matrix=np.asarray(metrics['confusion_matrix'],dtype=np.int64)
            normal=int(matrix[0].sum())
            errors=prioritized_errors(matrix)
            rows[model_id]['cohorts'][source]={'source':source,'split':split,'n':count,'metrics':metrics,'prioritized_errors':errors,
                'normal_grade0_overcalls':{'true0_denominator':normal,'true0_predicted2':int(matrix[0,2]),'true0_predicted_ge2':int(matrix[0,2:].sum()),
                    'true0_predicted_ge2_rate':float(matrix[0,2:].sum()/normal) if normal else None},
                'evaluation_path':str(report_path.resolve()),'prediction_path':str(pred_path.resolve()),
                'sample_ids_targets_source_content_argmax_verified':True,
                'weight_identity_basis':'Cross exporter verified actual checkpoint file SHA before loading' if cross else 'Original exporter checkpoint path/epoch matched original checkpoint_metadata file/model-state hashes',
                'cross_cohort_export':cross}
            predictions[source][model_id]=exported
        require(len(checkpoint_paths)==1,'One row references different checkpoint paths across cohorts')
        rows[model_id]['checkpoint']=next(iter(checkpoint_paths))
    bootstrap={}
    for offset,source in enumerate(COHORTS):
        print(json.dumps({'event':'unified_bootstrap','source':source,'models':4,'replicates':replicates,'seed':seed+offset}),flush=True)
        bootstrap[source]=bootstrap_metrics(manifests[source],predictions[source],replicates,seed+offset)
        bootstrap[source]['method']='Nonparametric whole-group percentile bootstrap; the same cohort group draws are shared by all four fixed models.'
        for model_id in rows:
            rows[model_id]['cohorts'][source]['uncertainty']=bootstrap[source]['models'][model_id]
    ddr=rows['ddr_partial']['cohorts']
    dominant=all(ddr[s]['prioritized_errors']['true_3_or_4_to_0_or_1']['count']<=rows[m]['cohorts'][s]['prioritized_errors']['true_3_or_4_to_0_or_1']['count'] for s in COHORTS for m in rows)
    return {'format_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'all_checks_passed':True,
        'purpose':'Descriptive comparison of four already fixed weights on the same three complete cohorts; one immutable weight per row. No training, fusion, threshold fitting, changing splits or choosing new epochs.',
        'grade_order':list(range(5)),'confusion_matrix_axes':{'rows':'true grade','columns':'predicted grade'},
        'cohorts':manifest_evidence,'models':rows,'bootstrap':bootstrap,
        'bootstrap_policy':'All 12 intervals recomputed using report_dr_safety_metrics.bootstrap_metrics; 5000 whole-group resamples per cohort by default. Seed=base_seed+cohort offset in APTOS2019,DDR,IDRiD order; all four models share each cohort draw. No old CI reuse.',
        'findings':{'ddr_partial_not_worse_on_prioritized_error_count_all_cohorts':dominant,
            'no_model_is_best_across_all_cohorts_and_error_dimensions':True,
            'ddr_partial_IDRiD_tradeoff':{'qwk':ddr['IDRiD']['metrics']['qwk'],
                'true3_or_4_to_0_or_1':ddr['IDRiD']['prioritized_errors']['true_3_or_4_to_0_or_1'],
                'true0_predicted2':ddr['IDRiD']['normal_grade0_overcalls']['true0_predicted2'],
                'true0_denominator':ddr['IDRiD']['normal_grade0_overcalls']['true0_denominator'],
                'interpretation_zh':'严重3/4→0/1为0/152，但正常0级有99/167张判为2级；特定严重低估次数少不等于整体分级好。'},
            'recommendation_zh':'若仅在现有权重中优先降低3/4级误判0/1的次数，DDR迁移partial是该特定维度的候选。其IDRiD QWK仅约0.592，正常0级99/167张判为2级；严重低估0/152不等于整体好，不能称全面最佳或临床可用。APTOS e5在DDR上的表现较弱，先前仅凭APTOS/IDRiD两队列将其推荐为统一主模型过早，应纠正。',
            'recommendation_scope':'Post-hoc descriptive tradeoff among existing fixed checkpoints, not an unbiased final clinical model selection or new prospective validation.'},
        'limitations':['These cohorts and test results are already observed; this analysis is descriptive and cannot create a new blind holdout.',
            'APTOS test patient identities are unknown and transformed/same-eye near-duplicate independence is not established; prior audit limitations remain.',
            'Declared whole-group bootstrap does not include retraining/model-selection uncertainty, unknown patient correlation or prevalence/domain shifts.',
            'Zero observed errors or a [0,0] bootstrap interval do not establish zero population risk. This prioritized pattern does not cover all harmful errors.',
            'DDR transfer weights inherit APTOS supervision; neither is an independent DDR-only model.',
            'No single fused/cherry-picked row is formed from the best value of different models. No claim of clinical deployment readiness is made.'],
        'input_files':[{'path':str(p.resolve()),'sha256':sha256_file(p)} for p in sorted(inputs)]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cloud-root',type=Path,default=ROOT/'cloud_upload')
    p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--evaluation-root',type=Path,default=ROOT/'cloud_upload/unified_dr_evaluation_20261113')
    p.add_argument('--replicates',type=int,default=5000); p.add_argument('--seed',type=int,default=20261113)
    p.add_argument('--output',type=Path)
    a=p.parse_args(); started=time.perf_counter()
    report=build_report(a.cloud_root,a.data_root,a.evaluation_root,a.replicates,a.seed)
    dest=(a.output or a.evaluation_root/'unified_summary.json').resolve()
    require(dest not in {Path(i['path']) for i in report['input_files']},'Output cannot overwrite original inputs')
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+'.tmp'); tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8'); os.replace(tmp,dest)
    print(json.dumps({'all_checks_passed':True,'output':str(dest),'elapsed_seconds':round(time.perf_counter()-started,2)}))

if __name__=='__main__': main()
