#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CFG="$ROOT_DIR/configs/swingait/swingait3D_B1122C2_CCPG_G2_stopgrad_shallow.yaml"
TMP_CFG="$ROOT_DIR/configs/swingait/.tmp_g2_stopgrad_shallow.yaml"

GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
RESTORE_HINT="${RESTORE_HINT:-0}"
SAVE_NAME_SUFFIX="${SAVE_NAME_SUFFIX:-}"

python - "$BASE_CFG" "$TMP_CFG" "$RESTORE_HINT" "$SAVE_NAME_SUFFIX" <<'PY'
import re
import sys
from pathlib import Path

base_cfg = Path(sys.argv[1])
tmp_cfg = Path(sys.argv[2])
restore_hint = sys.argv[3]
save_name_suffix = sys.argv[4]

text = base_cfg.read_text(encoding='utf-8')

replacement = restore_hint
if restore_hint != '0':
    replacement = f'"{restore_hint}"'

text = re.sub(r'(?m)^  restore_hint: .*$', f'  restore_hint: {replacement}', text, count=1)

if save_name_suffix:
    text = re.sub(
        r'(?m)^(\s*save_name:\s*)(\S+)$',
        lambda m: f"{m.group(1)}{m.group(2)}_{save_name_suffix}",
        text,
        count=2,
    )

tmp_cfg.write_text(text, encoding='utf-8')
print(tmp_cfg)
PY

cleanup() {
  rm -f "$TMP_CFG"
}
trap cleanup EXIT

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$GPUS" python -m torch.distributed.launch --nproc_per_node="$NPROC" opengait/main.py --cfgs "$TMP_CFG" --phase train