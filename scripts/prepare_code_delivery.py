"""Assemble a source-only handoff; never copy datasets, checkpoints or credentials."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8", newline="\n")


def write_json(path, obj):
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


PATH_TOOL = '''"""Generate NEW experiment configs by adapting paths only; does not train."""
import argparse
import json
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root', default='/mnt/retina')
    p.add_argument('--aptos-head-checkpoint', help='Previously backed-up APTOS epoch5 head_best.pt; pinned SHA stays unchanged')
    p.add_argument('--output-dir', type=Path, default=PROJECT / 'configs/runtime')
    args = p.parse_args()
    root = PurePosixPath(args.runtime_root)
    if not root.is_absolute() or '..' in root.parts:
        p.error('--runtime-root must be an absolute Linux path')
    jobs = {
        'dr_aptos.json': 'configs/experiments/dr_vitl_aptos_20261113_fast.json',
        'dr_ddr_transfer.json': 'configs/experiments/dr_vitl_ddr_from_aptos_head_20261113.json',
        'quality_vitb_eyeq.json': 'configs/planned_four_expert/quality_vitb_eyeq.json',
        'multidisease_vitb_rfmid.json': 'configs/planned_four_expert/multidisease_vitb_rfmid.json',
        'disc_cup_unet_refuge.json': 'configs/planned_four_expert/disc_cup_unet_refuge.json',
        'disc_cup_unet_papila.json': 'configs/planned_four_expert/disc_cup_unet_papila.json',
    }
    targets = [args.output_dir / name for name in jobs]
    if any(x.exists() for x in targets):
        p.error('Output config exists; choose a new --output-dir')
    payloads = {}
    for name, source in jobs.items():
        cfg = json.loads((PROJECT / source).read_text(encoding='utf-8-sig'))
        if name.startswith('dr_'):
            dataset = cfg['data']['datasets'][0].lower()
            cfg['data']['image_cache_dir'] = str(root / 'cache/source_isolated_336' / dataset)
        if 'initialization' in cfg:
            parent = args.aptos_head_checkpoint or str(root / 'imported/aptos_head_best.pt')
            if not PurePosixPath(parent).is_absolute() or '..' in PurePosixPath(parent).parts:
                p.error('--aptos-head-checkpoint must be an absolute Linux path')
            cfg['initialization']['checkpoint'] = parent
        payloads[name] = cfg
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in payloads.items():
        with (args.output_dir / name).open('x', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.write('\\n')
    print(json.dumps({'created': [str(x) for x in targets], 'training_started': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
'''


README = r'''
眼底模型训练代码交付包（2026-11-14）

1. 包含什么
training/：统一训练入口，分类、多标签、分割、指标、图像缓存和检查点保存。
scripts/：数据准备、来源隔离、缓存、预检、双卡启动、评估、混淆矩阵和bootstrap CI。
configs/experiments/：既有DR实验配置，保留原文件内容。
configs/planned_four_expert/：EyeQ ViT-B、RFMiD ViT-B、REFUGE/PAPILA U-Net待训练模板。
records/：两次实际DR运行的配置与运行环境；DDR的初始化来源另有记录。
tests/：现有unittest测试；不需要pytest。
训练参数说明.txt、TRAINING_PARAMETERS.json：实际参数和未训练模板参数，分开标记。
SHA256SUMS.json：本包各文件大小和SHA-256。

本包是代码交付，不包含原图、掩膜、数据清单、模型权重、虚拟环境、云端登录信息。
原始数据和已固定划分：G:华为ictdata（即 G:/华为ict/data）。
已选DDR第24轮权重：G:/华为ict/cloud_backup/20261113_DDR_from_APTOS/best.pt。
DDR初始化所需APTOS第5轮：G:/华为ict/cloud_backup/20261113_APTOS_ViTL/head_best.pt。
请将需要的数据和权重另行上传；保持data内部相对路径不变，勿重新划分来凑指标。
源数据只读使用，代码包的输出目录与data分开。

2. 已完成与待训练
已完成：DINOv2 ViT-L/14，APTOS 30轮；再从APTOS第5轮分类头权重初始化DDR，训练30轮。
当前保留的DDR候选是第24轮；历史训练来源为APTOS→DDR，不称为独立DDR-only模型。
待训练：EyeQ质控ViT-B、RFMiD多病种ViT-B、盘杯U-Net。
B模板仅替换既有S模板的模型名称；批量、学习率和微调策略尚待实测，不是已验证最优参数。
U-Net的REFUGE和PAPILA是两个独立实验选项，不自动混合。
本包没有启动任何训练，也没有实现screen_fundus服务或智果智能体接入。

3. Linux环境（对应原V100运行环境）
推荐按已验证环境重建：Ubuntu22.04、Python3.10；PyTorch2.6.0+cu118。
先确认数据盘已挂载；脚本不会挂盘或格式化磁盘。
在解压后的项目根目录运行：

export RETINA_RUNTIME_ROOT=/mnt/retina
export RETINA_DATA_ROOT=/mnt/retina/project/data
bash scripts/setup_cloud.sh
source "$RETINA_RUNTIME_ROOT/venv/bin/activate"
nvidia-smi

setup_cloud.sh显式安装torch2.6.0、torchvision0.21.0（cu118）及requirements-training.txt。
requirements-tools.txt仅给画图/数据准备等可选工具补依赖；基础训练不要求安装全部工具。
requirements-tools.txt不是原云环境的完整锁文件；records/environment/pip_freeze.txt保留原环境快照。
V100原实验使用FP16。下面每次启动均显式指定配置，不使用launch_v100.sh的旧S模型默认值。

4. 新实验的路径适配（不改原配置、不训练）
将原固定数据目录复制至RETINA_DATA_ROOT；必须含：
  prepared/source_isolated/isolation_manifest.json
  prepared/source_isolated/strict_cross_domain/tasks/<task>/<source>/{train,val,test,external_test}.jsonl
  对应清单引用的原图、掩膜。
代码通过单来源声明、全局索引和清单SHA校验，遇到混入来源或清单变更即报错。
可复制现有完整data目录，或在原项目使用scripts/package_cloud_training.py打任务数据包。
不要运行build_training_sets.py等准备脚本覆盖当前已固定的划分。

python tools/prepare_paths.py --runtime-root /mnt/retina --aptos-head-checkpoint /mnt/retina/imported/aptos_head_best.pt

生成configs/runtime/。DR缓存路径随runtime-root改变；DDR父权重SHA仍固定为历史APTOS第5轮。
该目录只适合新实验。原检查点的精确resume必须沿用原配置、卡数、训练计划和数据/缓存身份。
重建缓存或更换绝对路径不等同于旧运行的逐位精确恢复。

5. 从公开预训练权重重新跑APTOS（显式操作才启动）
python scripts/build_classification_cache.py --config configs/runtime/dr_aptos.json --data-root "$RETINA_DATA_ROOT" --cache-root "$RETINA_RUNTIME_ROOT/cache/source_isolated_336/aptos2019" --workers 8
python scripts/cloud_preflight.py --config configs/runtime/dr_aptos.json --data-root "$RETINA_DATA_ROOT"
bash scripts/launch_v100.sh configs/runtime/dr_aptos.json "$RETINA_RUNTIME_ROOT/runs/dr_aptos_new"

首次会获取配置中指定的DINOv2-L公开预训练权重。模型名称不是DINOv3。
缓存仅绑定本来源train/val、336×336、resize-before-augmentation。

6. 重跑历史APTOS→DDR迁移方案
上传原备份的APTOS第5轮head_best.pt到/mnt/retina/imported/aptos_head_best.pt。
预期SHA256：80c13752f5655290b2349ebc1dce951c83ef7aa7aa27d17572d09fbd0078fe05。
不要将新重训的APTOS权重冒充这份固定父权重；新父权重属于新实验，须同步记录真实SHA及来源。

python scripts/build_classification_cache.py --config configs/runtime/dr_ddr_transfer.json --data-root "$RETINA_DATA_ROOT" --cache-root "$RETINA_RUNTIME_ROOT/cache/source_isolated_336/ddr" --workers 8
python scripts/cloud_preflight.py --config configs/runtime/dr_ddr_transfer.json --data-root "$RETINA_DATA_ROOT"
bash scripts/launch_v100.sh configs/runtime/dr_ddr_transfer.json "$RETINA_RUNTIME_ROOT/runs/dr_ddr_transfer_new"

仅导入模型参数，重置优化器、调度器、AMP scaler、轮数、最优值和随机状态；第一步更新前先跑完整DDR验证集。
每轮使用验证集QWK选best.pt；不能用测试指标挑轮数或调阈值。
新的DDR独立预训练对照配置仅作为历史候选保留，本交付不自动启动它。

7. 其余三个模型（待训练模板）
显式选择configs/runtime/quality_vitb_eyeq.json、multidisease_vitb_rfmid.json、disc_cup_unet_refuge.json或disc_cup_unet_papila.json。
先运行cloud_preflight，再选一个配置单独运行launch_v100.sh，输出使用新目录。
质控是三分类softmax；多病种为5个sigmoid，未知标签掩码；盘杯为两通道分割。
使用原图而不建分类缓存时，不能把DR缓存后的速度当作B模板速度。
独立训练的不同任务不共享分类头、不拼接标签，也不同时挤进GPU启动全部实验。

8. 评估已保存DDR第24轮（不需要重训或重新下载预训练权重）
将原best.pt上传到/mnt/retina/imported/ddr_epoch24.pt。保持原清单与数据不变。
python -m training.train --config records/DDR/config.resolved.json --data-root "$RETINA_DATA_ROOT" --output "$RETINA_RUNTIME_ROOT/evaluation/ddr_epoch24" --eval-only --checkpoint /mnt/retina/imported/ddr_epoch24.pt --split test --eval-use-source-images --save-predictions --device cuda --workers 6

这里使用原记录配置，以便检查点身份校验；--eval-use-source-images只在完成身份校验后绕过旧缓存。
输出完整测试指标、逐样本预测及元数据。QWK不是准确率。
scripts/evaluate_dr_cohort.py支持显式固定checkpoint SHA的只读跨数据集测试，详见--help。
scripts/report_dr_safety_metrics.py提供真实3/4级→预测0/1级的数量、分母和bootstrap CI。
summary/画图/审计脚本可能默认寻找历史cloud_upload报告；这些报告不在代码包里，请显式传入输入路径。
历史备份校验工具、prepare_aptos_vitl_experiment.py等不是新训练的启动步骤。
CLOUD_TRAINING.md保留历史方案和旧默认值，本文件的显式入口为此次交付入口。

9. 本地验证
python -m unittest discover -s tests -v
该命令测试程序与小样本逻辑，不代表已在新云机器上完成全量训练或速度/显存验证。
包装检查结果见delivery_verification.json；没有为了验证交付而重新训练完整模型。
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'delivery/retina_training_code_20261114')
    args = parser.parse_args()
    dest = args.destination.resolve()
    archive_path = dest.with_suffix('.zip')
    if dest.exists() or archive_path.exists():
        parser.error('Destination already exists; choose a new destination')
    dest.mkdir(parents=True)
    copied = []

    def copy(src, relative):
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        copied.append({'path': str(relative).replace('\\', '/'), 'source_path': src.relative_to(ROOT).as_posix(),
                       'sha256': hashlib.sha256(src.read_bytes()).hexdigest()})

    for directory, suffixes in [('training', {'.py'}), ('scripts', {'.py', '.sh'}),
                                ('configs', {'.json'}), ('tests', {'.py'})]:
        for p in sorted((ROOT / directory).rglob('*')):
            if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts:
                copy(p, p.relative_to(ROOT))
    for name in ['requirements-training.txt', 'CLOUD_TRAINING.md', 'TRAINING_VERIFICATION.json']:
        copy(ROOT / name, Path(name))
    for source, label in [('cloud_backup/20261113_APTOS_ViTL/reports', 'APTOS'),
                          ('cloud_backup/20261113_DDR_from_APTOS/bundle_contents/run', 'DDR')]:
        for name in ['config.resolved.json', 'runtime.json', 'initialization_lineage.json', 'speed_transition_epoch7.json']:
            p = ROOT / source / name
            if p.is_file():
                copy(p, Path('records') / label / name)
    freeze = ROOT / 'cloud_backup/20261113_DDR_from_APTOS/bundle_contents'
    freezes = list(freeze.rglob('*freeze*'))
    freezes = [p for p in freezes if p.is_file()]
    if len(freezes) == 1:
        copy(freezes[0], Path('records/environment/pip_freeze.txt'))

    templates = {
        'quality_vitb_eyeq.json': 'image_quality_eyeq_eyeq.json',
        'multidisease_vitb_rfmid.json': 'multidisease_rfmid.json',
        'disc_cup_unet_refuge.json': 'disc_cup_segmentation_refuge.json',
        'disc_cup_unet_papila.json': 'disc_cup_segmentation_papila.json',
    }
    for name, base in templates.items():
        cfg = json.loads((ROOT / 'configs/source_isolated/strict_cross_domain' / base).read_text(encoding='utf-8-sig'))
        if 'vitb' in name:
            cfg['model']['name'] = 'vit_base_patch14_dinov2.lvd142m'
        cfg['notes'] = ('UNTRAINED DRAFT TEMPLATE. Existing source-isolated labels, splits, loss weights and hyperparameters retained. '
                        'ViT-B templates change only the backbone name from the original ViT-S template. '
                        'Batch size, learning rate and tuning stages require validation before a full run. '
                        'REFUGE and PAPILA are separate choices; never auto-concatenate. '
                        + cfg.get('notes', ''))
        write_json(dest / 'configs/planned_four_expert' / name, cfg)

    actual = {}
    for source, label, counts in [('records/APTOS/config.resolved.json', 'APTOS', [2775, 359, 354]),
                                  ('records/DDR/config.resolved.json', 'DDR_from_APTOS', [6041, 2479, 3898])]:
        cfg = json.loads((dest / source).read_text(encoding='utf-8-sig'))
        actual[label] = {'status': 'completed', 'config': source, 'model': cfg['model'],
                         'training': cfg['training'], 'augmentation': cfg['augmentation'], 'loss': cfg['loss'],
                         'seed': cfg['seed'], 'train_val_test_counts': counts, 'world_size': 2,
                         'effective_batch_size': 32, 'optimizer': 'AdamW', 'optimizer_betas': [0.9, 0.999],
                         'optimizer_eps': 1e-8, 'scheduler': '2-epoch warmup then epoch-level cosine, minimum factor 0.01',
                         'stages': {'epochs_1_to_5': 'head_only', 'epochs_6_to_30': 'last_4_blocks_plus_norm_plus_head'},
                         'initialization': cfg.get('initialization', 'public DINOv2-L pretrained weights')}
    actual['APTOS']['workers_actual'] = 'epochs1-7: 4 per rank; resumed from completed epoch7: 6 per rank'
    actual['DDR_from_APTOS']['workers_actual'] = '6 per rank'
    drafts = {name: json.loads((dest / 'configs/planned_four_expert' / name).read_text(encoding='utf-8'))
              for name in templates}
    write_json(dest / 'TRAINING_PARAMETERS.json', {'actual_completed_runs': actual, 'untrained_draft_templates': drafts})
    write_text(dest / '训练参数说明.txt', '''
实际DR参数（APTOS与DDR使用同一套结构/调度，数据与类别权重分开）
模型：DINOv2 ViT-L/14，vit_large_patch14_dinov2.lvd142m；输入336×336；5类0–4级。
参数总量：303,560,709；后4块+norm+head微调时可训练50,400,261（约16.6%）。
初始化：APTOS从公开DINOv2预训练权重；DDR仅加载已固定的APTOS第5轮head_best.pt模型参数。
训练：30轮；第1–5轮只训分类头；第6–30轮微调最后4块+norm+分类头；本轮没有全量解冻。
优化器：AdamW，betas=(0.9,0.999)，eps=1e-8，weight_decay=0.05。
基础学习率：主干1e-5，分类头1e-3；前2轮warmup，随后逐轮cosine，最低比例0.01。
批量：每卡16×2卡×累积1步=有效32；AMP FP16；梯度裁剪1.0；不启用梯度检查点。
损失：带训练集专属类别权重的CrossEntropy，label_smoothing=0.05。
分类：5维logits经softmax，argmax选级别；evaluation.threshold=0.5不用于DR五级argmax。
增强：水平翻转0.5，旋转±10度，color_jitter=0.1；先缩到336再增强；dropout=0.1。
归一化：mean=[0.485,0.456,0.406]；std=[0.229,0.224,0.225]；seed=20261113。
数据：APTOS train/val/test=2775/359/354；DDR=6041/2479/3898。每次训练只读本来源清单。
类别计数：APTOS [1426,270,735,136,208]；DDR [3070,302,2118,110,441]。
类别权重：平方根逆频率后均值归一化到1，精确值在JSON和原配置内。
workers：DDR每卡6；APTOS第1–7轮每卡4，第8轮起每卡6（原配置目标为6）。
选模：仅按本来源验证集QWK最大保存best.pt；测试集不参与调参。
当前选择：DDR第24轮best.pt。APTOS→DDR是顺序迁移，不能称作从未接触APTOS的模型。

未训练模板（不是实际实验结论）
EyeQ质控：ViT-B/14，392×392，3类；30轮；每卡4、累积4，双卡有效32；LR1e-4；首轮头，第2轮起全量。
RFMiD多病种：ViT-B/14，392×392，5个sigmoid；其余批量/轮数/LR同质控；带未知标签掩码的加权BCE。
U-Net盘杯：512×512，base_channels=32，2通道，随机初始化；50轮；每卡2、累积4，双卡有效16；LR3e-4；BCE+Dice。
三个模板均未正式训练，也未做新服务器速度/显存验证；模板不代表调参完成。
数据源、类别顺序、loss权重始终按各自配置，REFUGE/PAPILA分别跑，禁止自动混合。
完整参数见TRAINING_PARAMETERS.json；实际运行依据records中的配置和runtime。
''')
    write_text(dest / 'README.txt', README.replace(r'G:\u534e\u4e3aict\data', 'G:/华为ict/data'))
    write_text(dest / 'tools/prepare_paths.py', PATH_TOOL)
    write_text(dest / 'requirements-tools.txt', '''
# Optional helpers. Core training versions are in requirements-training.txt.
# Install only the tools needed for your workflow; this is not the historic cloud lockfile.
matplotlib>=3.8,<4
openpyxl>=3.1,<4
tifffile>=2023.4,<2025
requests>=2.32,<3
# Optional SIFT candidate audit with the training environment's numpy<2:
opencv-python-headless>=4.8,<4.12
''')
    write_text(dest / 'data/README.txt', '数据不随代码包重复打包。使用原项目G:/华为ict/data，或将所需原图/掩膜及固定的source_isolated清单复制到此目录。')
    write_json(dest / 'SOURCE_SNAPSHOT.json', {'created_local': datetime.now().astimezone().isoformat(),
                'code_source': 'current project source snapshot', 'copied_original_files': copied,
                'original_files_modified': False, 'dataset_included': False, 'checkpoints_included': False,
                'training_started': False})
    print(json.dumps({'staging': str(dest), 'source_files_copied': len(copied),
                      'files': sum(p.is_file() for p in dest.rglob('*')),
                      'archive_pending_verification': str(archive_path)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
