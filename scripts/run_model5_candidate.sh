#!/usr/bin/env bash
# Run one Model 5 candidate on an already provisioned two-GPU host.
set -euo pipefail
cd "$(dirname "$0")/.."
project="$PWD"
export PYTHONPATH="$project/deps:$project${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
arm=${1:?Choose large_full or base_disc}
action=${2:?Choose train or evaluate}
out_root=${MODEL5_RESULTS_DIR:-"$project/runs/model5_four_hours"}
roi_dir=${MODEL5_ROI_DIR:-"$project/data/prepared/screening_optic_disc_rois_v1/grouped"}
case "$arm" in
  large_full) backbone=vitl; gpu=0; view=() ;;
  base_disc) backbone=vitb; gpu=1; view=(--roi-dir "$roi_dir") ;;
  *) echo 'Unknown candidate' >&2; exit 2 ;;
esac
case "$action" in train|evaluate) ;; *) echo 'Unknown action' >&2; exit 2 ;; esac
export CUDA_VISIBLE_DEVICES="$gpu"
run="$out_root/$arm"
mkdir -p "$run"
exec 9>"$run/launch.lock"
flock -n 9 || { echo "Candidate is already active: $arm" >&2; exit 1; }
deadline=${MODEL5_DEADLINE_UTC:?Set the same absolute Unix end time for both candidates}
[[ "$deadline" =~ ^[0-9]+$ ]] || { echo 'Deadline must be a Unix timestamp' >&2; exit 2; }
bounded() {
  local remaining=$(( deadline - $(date -u +%s) - 10 ))
  (( remaining > 0 )) || { echo 'Shared time window has ended' >&2; return 124; }
  timeout --kill-after=10s "${remaining}s" "$@"
}
common=(--backbone "$backbone" --weights "cloud_upload/pretrained/dinov2_${backbone}14_pretrain.pth"
        --release data/prepared/screening_v1 --data-root . --output "$run" "${view[@]}"
        --precision fp16 --batch-size 8 --accum 4 --workers 6)
if [[ -n "${MODEL5_RGB_CACHE:-}" ]]; then common+=(--rgb-cache-dir "$MODEL5_RGB_CACHE"); fi
runner=(python3 -u -m training.screening_optimization)
if [[ "$action" == train ]]; then
  if [[ ! -f "$run/complete.json" ]]; then
    resume=()
    if [[ -f "$run/last.pt" ]]; then resume=(--resume); fi
    bounded "${runner[@]}" train "${common[@]}" --epochs 30 --head-only-epochs 5 --variant partial \
      --blocks 4 --head-lr 0.001 --backbone-lr 0.00001 --seed 20261114 "${resume[@]}"
  fi
  bounded python3 scripts/screening_evaluation.py lock --run "$run" --min-recall 0.90 --max-fpr 0.10
else
  [[ -f "$run/complete.json" && -f "$run/thresholds.json" ]] || { echo 'Finish training and lock validation thresholds first' >&2; exit 2; }
  for split in test external_test; do
    bounded "${runner[@]}" predict "${common[@]}" --split "$split"
  done
  bounded python3 scripts/screening_evaluation.py report --run "$run" --split val test external_test
fi
