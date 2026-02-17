#!/usr/bin/env bash
set -euo pipefail

# Instala dependências mínimas para execução do Pass1 portado sem fallback
# e baixa modelo YOLO bootstrap.

python -m pip install --quiet numpy pillow opencv-python-headless
python -m pip install --quiet --force-reinstall torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install --quiet ultralytics transformers scikit-learn scikit-image insightface onnxruntime

apt-get update -y
apt-get install -y libgl1 libglib2.0-0

python scripts/download_pass1_models.py
python scripts/pass1_dependency_report.py

echo "[OK] Pass1 runtime setup completed."
