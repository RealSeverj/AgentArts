#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RETINA_RUNTIME_ROOT:-/mnt/retina}"
source "$RUNTIME_ROOT/venv/bin/activate"
export HF_HOME="$RUNTIME_ROOT/cache/huggingface"
export TORCH_HOME="$RUNTIME_ROOT/cache/torch"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
CONFIG_PATH="${1:-configs/source_isolated/strict_cross_domain/dr_grading_ddr.json}"
OUTPUT_DIR="${2:-$RUNTIME_ROOT/runs/dr_ddr_isolated}"
if (( $# >= 1 )); then shift; fi
if (( $# >= 1 )); then shift; fi
WORKER_ARGS=()
if [[ -n "${TRAIN_WORKERS:-}" ]]; then
  WORKER_ARGS=(--workers "$TRAIN_WORKERS")
fi
cd "$PROJECT_DIR"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m training.train --config "$CONFIG_PATH" \
  --data-root "${RETINA_DATA_ROOT:-$PROJECT_DIR/data}" \
  --output "$OUTPUT_DIR" --device cuda "${WORKER_ARGS[@]}" "$@"
