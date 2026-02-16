#!/usr/bin/env bash
set -euo pipefail

# Cria uma fonte local /workspace/manga a partir do histórico deste repositório,
# útil quando clone externo do GitHub está bloqueado.

SRC_COMMIT="${1:-1082117}"
TARGET_DIR="${2:-/workspace/manga}"

mkdir -p "$TARGET_DIR"

# Exporta apenas blocos relevantes para migração Pass1/Pass2.
git archive "$SRC_COMMIT" \
  README.md \
  config \
  core \
  scripts \
  | tar -x -C "$TARGET_DIR"

echo "[OK] Fonte local criada em: $TARGET_DIR (commit $SRC_COMMIT)"
