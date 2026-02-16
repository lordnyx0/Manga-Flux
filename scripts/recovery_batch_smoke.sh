#!/usr/bin/env bash
set -euo pipefail

TMP_DIR="outputs/smoke/pages_bw"
META_DIR="metadata"
MASK_DIR="outputs/smoke/masks"
OUT_DIR="outputs/smoke/pass2"

mkdir -p "$TMP_DIR" "$META_DIR" "$MASK_DIR" "$OUT_DIR"

# Cria lote sintético de 3 páginas para smoke local.
cp data/dummy_manga_test.png "$TMP_DIR/page_001.png"
cp data/dummy_manga_test.png "$TMP_DIR/page_002.png"
cp data/dummy_manga_test.png "$TMP_DIR/page_003.png"

python run_two_pass_batch_local.py \
  --input-dir "$TMP_DIR" \
  --style-reference data/dummy_manga_test.png \
  --metadata-output "$META_DIR" \
  --masks-output "$MASK_DIR" \
  --pass2-output "$OUT_DIR" \
  --chapter-id smoke-batch \
  --start-page 1 \
  --prompt-template "smoke batch page={page_num} file={filename}" \
  --engine flux

python scripts/validate_two_pass_outputs.py --metadata-dir "$META_DIR" --pass2-dir "$OUT_DIR" --expected-pages 3

echo "[OK] Batch smoke complete for 3 pages (Pass1->Pass2)."
