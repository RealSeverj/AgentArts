"""Prepare the explicitly authorized APTOS-only ViT-L experiment, without training."""
from pathlib import Path
import json

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / 'configs/experiments/dr_vitl_aptos_20261113_fast.json'


def main():
    original = PROJECT / 'configs/source_isolated/strict_cross_domain/dr_grading_aptos2019.json'
    config = json.loads(original.read_text(encoding='utf-8-sig'))
    config['data']['external_dataset'] = 'IDRiD'
    config['data']['image_cache_dir'] = '/mnt/retina/cache/source_isolated_336/aptos2019'
    config['model'].update(name='vit_large_patch14_dinov2.lvd142m', img_size=336,
                           grad_checkpointing=False, dropout=0.1)
    config['model']['kwargs'] = {'pretrained_cfg_overlay': {
        'hf_hub_id': None,
        'url': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'}}
    config['augmentation']['resize_before_augmentation'] = True
    config['training'].update(epochs=30, batch_size=16, grad_accum_steps=1, workers=6,
        amp='fp16', lr=1e-5, backbone_lr=1e-5, head_lr=1e-3,
        freeze_backbone_epochs=5, unfreeze_last_n_blocks=4, full_finetune_epoch=30,
        warmup_epochs=2, log_every=25)
    config['notes'] = (
        'APTOS-only DINOv2-L/14 at 336. Epochs1-5 head only; epochs6-30 last4 '
        'transformer blocks and head only; no full-backbone unfreezing. New '
        'initialization from public pretrained weights, never the old mixed-source '
        'checkpoint. Select best by APTOS validation QWK. Lock best checkpoint '
        'before internal and IDRiD external tests. Never tune on test results. '
        'IDRiD was already evaluated in earlier project experiments, so it is '
        'not a never-observed prospective holdout. DDR is deferred by user.')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        if json.loads(OUTPUT.read_text(encoding='utf-8')) != config:
            raise ValueError('Refusing to overwrite a different experiment')
    else:
        OUTPUT.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    plan = {'status': 'preparing', 'active_source': 'APTOS2019', 'deferred_source': 'DDR',
        'train_records': 2775, 'val_records': 359, 'test_records': 354,
        'external_source': 'IDRiD', 'external_records': 506,
        'config': OUTPUT.relative_to(PROJECT).as_posix(),
        'server_output': '/mnt/retina/runs/dr_vitl_aptos_20261113_fast',
        'speed_changes': '16/card x 2 cards x 1 accumulation = same global32; disable activation recomputation after memory probe; 6 workers/card; verified source-only 336 cache.',
        'selection': 'Best APTOS validation QWK across all 30 epochs. Preserve head-only best before unfreezing.',
        'previous_model': '/mnt/retina/runs/dr_baseline/best.pt retained as rollback/reference',
        'comparison_note': 'Compare fixed checkpoints on identical test records; mixed-source validation QWK cannot be directly compared with APTOS-only validation QWK.',
        'data_limitations': 'APTOS patient IDs unavailable; known image/content/group isolation does not establish full patient-level independence.',
        'other_experts': 'Quality ViT-B; multidisease ViT-B; disc/cup U-Net. No other expert training started.'}
    (PROJECT / 'cloud_upload/DR_VITL_EXPERIMENT_PLAN.json').write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(OUTPUT)


if __name__ == '__main__':
    main()
