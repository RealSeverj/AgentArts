#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${RETINA_API_KEY:?Set RETINA_API_KEY to a random secret of at least 32 bytes}"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec .venv/bin/python -m inference serve --host 127.0.0.1 --port "${RETINA_PORT:-8000}"
