#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
runtime="${1:-cu118}"
case "$runtime" in
  cpu) index="https://download.pytorch.org/whl/cpu" ;;
  cu118) index="https://download.pytorch.org/whl/cu118" ;;
  *) echo "Usage: bash deployment/inference/install_runtime.sh cpu|cu118" >&2; exit 2 ;;
esac
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url "$index"
.venv/bin/python -m pip install -r requirements-inference.txt
.venv/bin/python scripts/check_inference_environment.py --device "${runtime/cu118/cuda}"
