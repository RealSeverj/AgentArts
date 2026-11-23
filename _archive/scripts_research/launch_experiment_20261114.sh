#!/usr/bin/env bash
# Explicitly launch one prepared experiment; never start a suite automatically.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RETINA_RUNTIME_ROOT:-/mnt/retina}"
export RETINA_RUNTIME_ROOT="$RUNTIME_ROOT"
export RETINA_DATA_ROOT="${RETINA_DATA_ROOT:-$PROJECT_DIR/data}"
export TORCH_HOME="$RUNTIME_ROOT/cache/torch"
export HF_HOME="$RUNTIME_ROOT/cache/huggingface"
case "${1:-}" in
  dr) CONFIG="configs/experiments/dr_vitl_ddr_crop_letterbox_20261114.json"; RUN="dr_ddr_crop_letterbox_20261114" ;;
  quality) CONFIG="configs/experiments/image_quality_vitb_eyeq_20261114.json"; RUN="quality_vitb_eyeq_20261114" ;;
  multidisease) CONFIG="configs/experiments/multidisease_vitb_rfmid_20261114.json"; RUN="multidisease_vitb_rfmid_20261114" ;;
  disc_refuge) CONFIG="configs/experiments/disc_cup_unet_refuge_20261114.json"; RUN="disc_cup_unet_refuge_20261114" ;;
  disc_papila) CONFIG="configs/experiments/disc_cup_unet_papila_20261114.json"; RUN="disc_cup_unet_papila_20261114" ;;
  *) printf '%s\n' 'Usage: bash scripts/launch_experiment_20261114.sh {dr|quality|multidisease|disc_refuge|disc_papila} [NEW_OUTPUT_DIRECTORY]'; exit 2 ;;
esac
source "$RUNTIME_ROOT/venv/bin/activate"
cd "$PROJECT_DIR"
OUTPUT="${2:-$RUNTIME_ROOT/runs/$RUN}"
python - "$OUTPUT" "$CONFIG" <<'PY'
from pathlib import Path
import hashlib, json, sys
out = Path(sys.argv[1])
if out.exists() and any(out.iterdir()):
    raise SystemExit('Use a new/empty output directory; this launcher does not resume or overwrite old runs.')
config = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8-sig'))
init = config.get('initialization')
if init:
    parent = Path(init['checkpoint'])
    if not parent.is_file():
        raise SystemExit(f'Missing fixed APTOS parent: {parent}')
    with parent.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
    if digest is None:
        hasher = hashlib.sha256()
        with parent.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''): hasher.update(chunk)
        digest = hasher.hexdigest()
    if digest != init['checkpoint_sha256']:
        raise SystemExit('APTOS parent SHA differs from the pinned historical checkpoint.')
PY
if [[ "${1}" == "dr" ]]; then
  CACHE_ROOT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"]["image_cache_dir"])' "$CONFIG")"
  python scripts/build_classification_cache.py --config "$CONFIG" --data-root "$RETINA_DATA_ROOT" --cache-root "$CACHE_ROOT" --workers 8
fi
python scripts/cloud_preflight.py --config "$CONFIG" --data-root "$RETINA_DATA_ROOT"
exec bash scripts/launch_v100.sh "$CONFIG" "$OUTPUT"
