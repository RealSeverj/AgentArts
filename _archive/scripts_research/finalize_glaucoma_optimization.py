"""Verify the full-data ViT-L challenger and publish a combined, paired Model 6 report."""
import argparse
import json
from pathlib import Path

from scripts import screening_evaluation as ev
from scripts import summarize_glaucoma_optimization as stats
CANDIDATE='glaucoma_vitl392_letterbox'
HISTORICAL='378fc1a1e15e4f67e016bacaa49fd1b81e016c973043919e3a0a290edadc1aca'
WEIGHTS='d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428'


def finalize(args):
    out,new,old=args.output,args.candidate,args.historical
    select=ev.read_json(new/'final_selection.json'); freeze=ev.read_json(new/'selection_policy.json')
    complete=ev.read_json(new/'complete.json'); run=ev.read_json(new/'run.json'); history=ev.read_json(new/'history.json')
    if (freeze['candidate']!=CANDIDATE or freeze['starting_sha256']!=WEIGHTS or freeze['historical_sha256']!=HISTORICAL
            or select['policy_sha256']!=stats.sha(new/'selection_policy.json') or run['identity']['candidate']!=CANDIDATE):
        raise ValueError('Independent selection protocol mismatch')
    if complete['epochs']!=30 or [r['epoch'] for r in history]!=list(range(1,31)):
        raise ValueError('Incomplete training')
    ranks=select['validation']
    for name,folder in [('historical',old),(CANDIDATE,new)]:
        rec=ranks[name]; rows,meta,digest=ev.load_predictions(folder,'val'); y,p,_=ev.known_arrays(rows,0)
        lock=ev.read_json(folder/'thresholds.json')
        if (rec['checkpoint_sha256']!=meta['checkpoint_sha256'] or lock['checkpoint_sha256']!=meta['checkpoint_sha256']
                or rec['thresholds_sha256']!=stats.sha(folder/'thresholds.json') or rec['val_predictions_sha256']!=digest
                or lock['val_predictions_sha256']!=digest or rec['metrics']!=ev.metrics(y,p,rec['threshold'])):
            raise ValueError('Validation reranking disagrees with receipt')
    winner=CANDIDATE if ranks[CANDIDATE]['metrics']['ap']>ranks['historical']['metrics']['ap'] else 'historical'
    if (winner!=select['selected'] or complete['checkpoint_sha256']!=ranks[CANDIDATE]['checkpoint_sha256']
            or ranks['historical']['checkpoint_sha256']!=HISTORICAL):raise ValueError('Wrong final winner')
    chosen=old if winner=='historical' else new
    identity=ranks[winner]['checkpoint_sha256']
    if stats.sha(chosen/'best.pt')!=identity:raise ValueError('Selected deployment weight hash mismatch')
    if (new/'best.pt').exists() and stats.sha(new/'best.pt')!=complete['checkpoint_sha256']:raise ValueError('Archived candidate weight hash mismatch')
    phase1=ev.read_json(out/'selection.json')
    if phase1['selected']!='historical':raise ValueError('Unexpected first-stage selection')
    archive=out/'阶段一归档';archive.mkdir(exist_ok=True)
    for name in ['最终模型清单.json','逐来源配对回归.json','模型6优化报告.md']:
        previous=archive/name
        if not previous.exists():ev.write_file(previous,(out/name).read_text(encoding='utf-8'),lock=True)
    if winner=='historical':
        reference=ev.read_json(archive/'逐来源配对回归.json')
        if reference['selection']!='historical' or reference['draws']!=1000:raise ValueError('Invalid historical paired reference')
    else:
        reference=ev.read_json(new/'paired_glaucoma_vitl.json')
        if (reference['selected']!=winner or reference['selection_sha256']!=stats.sha(new/'final_selection.json') or reference['draws']!=1000):
            raise ValueError('Invalid cloud paired inference report')
    pairs=reference['splits']
    for s,x in pairs.items():
        if (x['selected_predictions_sha256']!=stats.sha(chosen/f'{s}_predictions.jsonl') or x['historical_predictions_sha256']!=stats.sha(old/f'{s}_predictions.jsonl')
                or x['source_manifest_sha256']!=stats.sha(args.release/'glaucoma'/f'{s}.jsonl')):
            raise ValueError('Paired source fingerprints differ from locally archived files')
    digest_inputs=dict(phase1_selection_sha256=stats.sha(out/'selection.json'), final_selection_sha256=stats.sha(new/'final_selection.json'),
                       candidate_checkpoint_sha256=complete['checkpoint_sha256'],selected_checkpoint_sha256=identity,
                       historical_checkpoint_sha256=HISTORICAL,initial_vitl_sha256=WEIGHTS)
    ev.write_json(out/'逐来源配对回归.json',dict(**digest_inputs,selected=winner,draws=1000,seed=20261114,splits=pairs))
    if winner=='historical':
        spec=ev.read_json(archive/'最终模型清单.json')
    else:
        spec=dict(candidate=winner,checkpoint_file=str((chosen/'best.pt').resolve()),checkpoint_sha256=identity,
                  backbone_initializer_sha256=WEIGHTS,model_class='training.glaucoma_vitl_candidate.LargeGlaucoma',
                  threshold_file=str((chosen/'thresholds.json').resolve()),threshold_file_sha256=stats.sha(chosen/'thresholds.json'),
                  input_transform=run['transform'],geometry_recipe_sha256=run['identity']['geometry_recipe_sha256'],
                  threshold=ranks[winner]['threshold'],inference_dtype='fp16, CUDA')
    spec.update(**digest_inputs,best_epoch=complete['best_epoch'] if winner==CANDIDATE else 12)
    if (new/'best.pt').is_file():
        spec.update(candidate_training_checkpoint_file=str((new/'best.pt').resolve()),
                    candidate_training_checkpoint_bytes=(new/'best.pt').stat().st_size)
    ev.write_json(out/'最终模型清单.json',spec)
    title='ViT-L/14 全图 392 裁边补边' if winner==CANDIDATE else '历史 ViT-B/14 单图模型'
    lines=['# 模型6青光眼辅助筛查优化报告','',f"两阶段验证选型后保留 **{title}**，青光眼判阳阈值为 **{ranks[winner]['threshold']:.8f}**。",'',
           '## 开发与选择','','| 方案 | PAPILA 验证 AP | AUROC | 验证漏检/13 | 验证误报/49 |','|---|---:|---:|---:|---:|']
    for k,score in phase1['validation'].items():
        lines.append(f"| {stats.DISPLAY[k]} | {score['ap']:.4f} | {score['auroc']:.4f} | {score['fn']} | {score['fp']} |")
    nc=ranks[CANDIDATE]['metrics']
    lines.append(f"| ViT-L/14 392 裁边补边 | {nc['ap']:.4f} | {nc['auroc']:.4f} | {nc['fn']} | {nc['fp']} |")
    lines+=['',f"新增 ViT-L 从官方 DINOv2 权重初始化。全部 738 条训练记录均经过标签检查，692 张有标签眼图进入监督训练，另有 46 张诊断不确定或无二分类标签。先用 5 轮训练新头，再用 25 轮训练末 4 块与新头。云端共完成 30 轮，逐轮训练与验证合计 {sum(r['elapsed_seconds'] for r in history)/60:.1f} 分钟，依据验证 AP 留下第 {complete['best_epoch']} 轮。",'',
            '开发验证集共 72 张，其中 62 张有二分类标签，13 张阳性来自 7 位患者。新增候选必须在同一验证清单上严格超过历史模型。每个模型各自在验证集锁定阈值，使召回率至少达到 90%，然后尽量减少误报。','',
            '## 固定阈值复评','','| 数据来源 | 有标签/全部 | 阳性/阴性 | 历史 AP → 最终 AP | AP 差值 [配对95%区间] | 召回率 / 特异度 | 漏检 / 误报 | 达标 |','|---|---:|---:|---:|---:|---:|---:|---|']
    for split,x in pairs.items():
        a,b=x['historical'],x['selected'];d=x['paired']['ap'];lo,hi=d['ci95']
        lines.append(f"| {stats.SPLITS[split]} | {x['known']}/{x['images']} | {x['positive']}/{x['negative']} | {a['ap']:.4f} → {b['ap']:.4f} | {d['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {b['sensitivity']:.1%} / {b['specificity']:.1%} | {b['fn']} / {b['fp']} | {'是' if x['met_goal'] else '否'} |")
    if winner=='historical':
        lines+=['','历史模型的验证 AP 最高，表中两套留出结果均复用其已归档的逐图预测。配对差值为零来自同一模型、同一阈值和同一批眼图。新增 ViT-L 于验证选型后止步，唯一的模型比较来自开发验证集。']
    lines+=['','达标要求召回率至少 90%、误报率至多 10%。各模型使用自己的验证冻结阈值。区间基于 1,000 次配对抽样，PAPILA 按患者、FIVES 按图像内容组重采样，反映既定模型和阈值下的样本不确定性。PAPILA 内测仅 8 名阳性患者。FIVES 缺少患者 ID，只含青光眼和正常各 200 张，该构成会影响 AP。REFUGE 训练部分使用镜像文件名前缀标签。','',
            '完整混淆矩阵、AUROC 和四项配对差值区间在 `逐来源配对回归.json`。权重与阈值指纹在 `最终模型清单.json`。第一阶段报告保存在 `阶段一归档`。']
    ev.write_file(out/'模型6优化报告.md','\n'.join(lines)+'\n')
    print(json.dumps(dict(winner=winner,epoch=complete['best_epoch'],validated_ap=nc['ap'],selected_test=pairs['test']['selected'],selected_external=pairs['external_test']['selected']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--historical',type=Path,required=True)
    p.add_argument('--release',type=Path,default=Path('data/prepared/screening_v1'))
    finalize(p.parse_args())
