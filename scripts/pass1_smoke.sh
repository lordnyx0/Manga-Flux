#!/usr/bin/env bash
set -euo pipefail

python run_pass1_local.py \
  --page-image data/dummy_manga_test.png \
  --output-mask outputs/smoke/masks/page_001_text.png \
  --style-reference data/dummy_manga_test.png \
  --page-num 1 \
  --chapter-id smoke \
  --metadata-output metadata

echo "[OK] Pass1 smoke metadata generated."
