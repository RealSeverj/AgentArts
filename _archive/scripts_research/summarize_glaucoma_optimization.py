"""Write a paired, frozen-threshold regression report for the selected glaucoma recipe."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import platform

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import screening_evaluation as ev


DISPLAY = {"historical": "历史单图模型", "flip_tta": "原图与水平翻转均分", "structural": "预测杯盘形态分类器", "hybrid": "翻转双视图与杯盘形态等权融合"}
SPLITS = {"val": "PAPILA 验证", "test": "PAPILA 内部测试", "external_test": "FIVES 外部测试"}
RATES = ("ap", "auroc", "sensitivity", "specificity")


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(data)
    return digest.hexdigest()


def metric_text(number):
    return f"{number:.4f}" if number is not None else "未定义"


def paired(split, historical, selected, release, draws, seed):
    source = release / "glaucoma" / f"{split}.jsonl"
    ref = [json.loads(line) for line in source.read_bytes().splitlines()]
    data = []
    for path in (historical, selected):
        rows, meta, file_sha = ev.load_predictions(path, split)
        lock = ev.read_json(path / "thresholds.json")
        if (meta["manifest_sha256"] != sha(source) or meta["checkpoint_sha256"] != lock["checkpoint_sha256"]
                or lock["val_predictions_sha256"] != sha(path / "val_predictions.jsonl")):
            raise ValueError(f"Invalid locked prediction provenance {path}/{split}")
        by_id = {r["sample_id"]: r for r in rows}
        fields = ("dataset", "split", "labels", "mask", "patient_id", "content_group_id", "image_path")
        if len(by_id) != len(ref) or any(any(r[k] != by_id[r['sample_id']][k] for k in fields) for r in ref):
            raise ValueError(f"Reference cohort mismatch {path}/{split}")
        ordered = [by_id[r["sample_id"]] for r in ref]
        for r, pred in zip(ref, ordered):
            if pred.get("image_sha256", r["image_sha256"]) != r["image_sha256"]:
                raise ValueError(f"Image digest differs for {r['sample_id']}")
        y, p, g = ev.known_arrays(ordered, 0)
        t = lock["per_label"]["glaucoma"]["threshold"]
        data.append((y, p, g, t, file_sha))
    y, p0, g, t0, h0 = data[0]
    y1, p1, g1, t1, h1 = data[1]
    if not np.array_equal(y1, y) or not np.array_equal(g1, g):
        raise ValueError("Paired identities mismatch")
    n_groups = int(g.max()+1)
    before, after = ev.metrics(y, p0, t0), ev.metrics(y, p1, t1)
    rng = np.random.default_rng(seed)
    diffs = {k: [] for k in RATES}
    for _ in range(draws):
        w = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)[g]
        old, new = ev.metrics(y, p0, t0, w), ev.metrics(y, p1, t1, w)
        for k in diffs:
            if old[k] is not None and new[k] is not None:
                diffs[k].append(new[k] - old[k])
    intervals = {k: dict(delta=after[k]-before[k], ci95=np.quantile(ds,[.025,.975]).tolist(), valid_draws=len(ds)) for k,ds in diffs.items()}
    missing_patients = sum(r['mask'][0] and r.get('patient_id') in ('',None) for r in ref)
    return dict(images=len(ref), known=len(y), positive=int(y.sum()), negative=int((y==0).sum()),
                unknown=len(ref)-len(y), resampling_groups=n_groups, positive_groups=len(set(g[y==1])),
                known_images_missing_patient_id=missing_patients, source_manifest_sha256=sha(source),
                historical_predictions_sha256=h0, selected_predictions_sha256=h1,
                historical_threshold=t0, selected_threshold=t1, historical=before, selected=after,
                paired=intervals, met_goal=(after['sensitivity']>=.9 and after['specificity']>=.9))


def run(args):
    out = args.output
    selection = ev.read_json(out / "selection.json")
    policy = ev.read_json(out / "selection_policy.json")
    if sha(out / "selection_policy.json") != selection['policy_sha256']:
        raise ValueError("Frozen selection policy SHA mismatch")
    winner = max(policy['candidates'], key=lambda n: selection['validation'][n]['ap'])
    if winner != selection['selected']:
        raise ValueError("Winner differs from frozen rule")
    for name, score in selection['validation'].items():
        folder = args.historical if name=='historical' else out/name
        if sha(folder/'thresholds.json') != score['thresholds_sha256'] or sha(folder/'val_predictions.jsonl') != score['val_predictions_sha256']:
            raise ValueError(f"Validation receipt mismatch {name}")
    final = args.historical if winner=='historical' else out/winner
    pairs = {s: paired(s,args.historical,final,args.release,args.draws,args.seed) for s in SPLITS}
    valrows,_,_=ev.load_predictions(args.historical,'val')
    views={r['sample_id']:r for r in (json.loads(x) for x in (out/'features/val.jsonl').read_bytes().splitlines())}
    labelled=[r for r in valrows if r['mask'][0]]
    y=np.asarray([r['labels'][0] for r in labelled])
    cpu=np.asarray([views[r['sample_id']]['native_flip_probabilities'][0] for r in labelled])
    old=np.asarray([r['probabilities'][0] for r in labelled])
    old_threshold=pairs['val']['historical_threshold']
    native_check=dict(cpu_fp32_native=ev.metrics(y,cpu,old_threshold), historical_fp16_native=ev.metrics(y,old,old_threshold),
                      known=len(y), median_absolute_probability_difference=float(np.median(np.abs(cpu-old))),
                      maximum_absolute_probability_difference=float(np.max(np.abs(cpu-old))),
                      differing_calls_at_historical_threshold=int(np.count_nonzero((cpu>=old_threshold)!=(old>=old_threshold))))
    spec = dict(candidate=winner, selection_sha256=sha(out/'selection.json'), threshold_file=str((final/'thresholds.json').resolve()),
                threshold_file_sha256=sha(final/'thresholds.json'), threshold=pairs['val']['selected_threshold'],
                inference_recipe_sha256=selection['validation'][winner]['checkpoint_sha256'])
    cli=Path(__file__).with_name('predict_glaucoma_optimized.py').resolve()
    trainer=Path(__file__).with_name('optimize_glaucoma_screening.py').resolve()
    spec.update(inference_entry_point=str(cli), inference_entry_point_sha256=sha(cli),
                optimization_code_sha256=sha(trainer), python_version=platform.python_version(),
                fitted_cpu_dtype='fp32', historical_archived_inference_dtype='fp16')
    if winner in ('historical','flip_tta','hybrid'):
        spec.update(disease_model_file=str((args.historical/'best.pt').resolve()), disease_model_sha256=sha(args.historical/'best.pt'))
        if spec['disease_model_sha256']!=policy['disease_checkpoint_sha256']:raise ValueError('Glaucoma weights changed')
    if winner in ('structural','hybrid'):
        spec.update(segmentation_model_file=str(args.unet.resolve()), segmentation_model_sha256=sha(args.unet),
                    structural_head=str((out/'structural_head.json').resolve()), structural_head_sha256=sha(out/'structural_head.json'))
        if spec['segmentation_model_sha256']!=policy['segmentation_checkpoint_sha256']:raise ValueError('U-Net weights changed')
    if winner!='historical':
        recipe = final/'inference_recipe.json'
        if sha(recipe)!=spec['inference_recipe_sha256']:raise ValueError('Recipe digest differs')
        if winner in ('hybrid','structural') and ev.read_json(recipe)['head_sha256']!=spec['structural_head_sha256']:raise ValueError('Head digest differs')
        spec['inference_recipe_file']=str(recipe.resolve())
    ev.write_json(out/'最终模型清单.json',spec,lock=True)
    ev.write_json(out/'逐来源配对回归.json',dict(selection=winner, draws=args.draws, seed=args.seed, native_validation_precision_check=native_check, paired_delta='selected minus historical',
                  interval='percentile, resampled in matched patient groups where known, else matched content groups; conditional on fixed model and thresholds',splits=pairs))
    lines=['# 模型6青光眼辅助筛查优化', '', f"按 PAPILA 验证集 AP 选中**{DISPLAY[winner]}**。锁定判阳阈值为 {spec['threshold']:.8f}。", '',
           '## 验证选型', '', '| 方案 | AP | AUROC | 漏检/阳性 | 误报/阴性 |', '|---|---:|---:|---:|---:|']
    val=pairs['val']
    for name in policy['candidates']:
        r=selection['validation'][name]; lines.append(f"| {DISPLAY[name]} | {metric_text(r['ap'])} | {metric_text(r['auroc'])} | {r['fn']}/{val['positive']} | {r['fp']}/{val['negative']} |")
    lines += ['', '验证集中 13 张阳性眼图来自 7 名患者。并列时按上表顺序选取。各自工作点在验证集先满足召回率 90%，再尽量减少误报。', '',
              f"同一疾病权重的单图 CPU FP32 验证 AP 为 {native_check['cpu_fp32_native']['ap']:.4f}。历史 GPU FP16 单图验证 AP 为 {native_check['historical_fp16_native']['ap']:.4f}。沿用历史阈值时，两种精度在 {len(y)} 张有标签眼图中有 {native_check['differing_calls_at_historical_threshold']} 张判定不同。", '',
              '## 固定阈值复评', '', '| 来源 | 有标签/全部 | 阳性/阴性 | 旧 AP → 本轮 AP | AP 配对差值 [95% 区间] | 旧漏检 → 本轮漏检 | 旧误报 → 本轮误报 | 本轮召回率 / 特异度 | 同时达标 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for s, v in pairs.items():
        a,b,d=v['historical'],v['selected'],v['paired']['ap']; lo,hi=d['ci95']
        lines.append(f"| {SPLITS[s]} | {v['known']}/{v['images']} | {v['positive']}/{v['negative']} | {a['ap']:.4f} → {b['ap']:.4f} | {d['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {a['fn']} → {b['fn']} | {a['fp']} → {b['fp']} | {b['sensitivity']:.1%} / {b['specificity']:.1%} | {'是' if v['met_goal'] else '否'} |")
    known=ev.read_json(out/'structural_head.json'); head=known['training_sources']
    lines += ['', f"杯盘模型使用 REFUGE 训练的冻结 U-Net 从图像预测视盘与视杯，再在 {head['REFUGE']} 张 REFUGE、{head['PAPILA']} 张 PAPILA 有标签训练眼图上拟合逻辑回归，其中阳性 {known['positive']} 张。使用预测垂直及水平杯盘比、面积比、视盘定位缺失标志。REFUGE 青光眼标签来自镜像训练文件名前缀。水平翻转方案对原图与翻转图的青光眼概率取平均，融合方案对翻转方案与杯盘方案的概率再取平均。", '',
              '配对差值为本轮减历史。PAPILA 按患者重采样，FIVES 缺少患者 ID，按图像内容组重采样。区间来自 1,000 次配对抽样，条件为两套模型与阈值固定。验证集也参与了选型。PAPILA 内部测试只有 8 名阳性患者，疑似病例保留在清单中并从二分类评分排除。FIVES 外测只含青光眼和正常各 200 张，外测 AP 受此构成影响。两套留出集均沿用各自在验证集确定的阈值。', '',
              '同时达标要求召回率至少 90%、误报率至多 10%。逐来源混淆、AUROC、召回率、特异度及配对差值区间在 `逐来源配对回归.json`，最终推理配方和权重指纹在 `最终模型清单.json`。']
    ev.write_file(out/'模型6优化报告.md','\n'.join(lines)+'\n')
    print(json.dumps({s:dict(ap=v['selected']['ap'],delta=v['paired']['ap'],fn=v['selected']['fn'],fp=v['selected']['fp'],met=v['met_goal']) for s,v in pairs.items()},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--historical',type=Path,required=True)
    p.add_argument('--unet',type=Path,required=True)
    p.add_argument('--release',type=Path,default=Path('data/prepared/screening_v1'))
    p.add_argument('--draws',type=int,default=1000);p.add_argument('--seed',type=int,default=20261114)
    run(p.parse_args())
