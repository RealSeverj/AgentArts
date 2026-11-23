#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RETINA_RUNTIME_ROOT:-/mnt/retina}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$RUNTIME_ROOT/cache/huggingface" "$RUNTIME_ROOT/cache/torch" "$RUNTIME_ROOT/runs"
"$PYTHON_BIN" -m venv "$RUNTIME_ROOT/venv"
source "$RUNTIME_ROOT/venv/bin/activate"
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-cache-dir -r "$PROJECT_DIR/requirements-training.txt"
python - <<'PY'
import torch
print('torch:', torch.__version__, 'CUDA wheel:', torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable: check NVIDIA driver and the selected Python environment.')
print('GPU count:', torch.cuda.device_count(), 'wheel architectures:', torch.cuda.get_arch_list())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
    x = torch.ones(8, device=f'cuda:{i}')
    print('CUDA operation:', float((x * x).sum()))
PY
printf '%s\n' "Environment prepared in $RUNTIME_ROOT/venv" "Next: bash scripts/launch_v100.sh"
