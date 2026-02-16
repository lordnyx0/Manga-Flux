"""
Manga-Flux v1.0 - Configurações do Sistema
Configurações otimizadas para o pipeline Flux Klein 9B + SAM 2.1 + YOLO.
"""

import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# ============================================================================
# HARDWARE E PERFORMANCE
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Otimizações para 12GB VRAM
OFFLOAD_TO_CPU = True              # Carrega componentes para CPU quando inativos
ENABLE_VAE_SLICING = True          # Slicing para processamento de imagem em alta res

# ============================================================================
# PASS 1: ANÁLISE (YOLO + SAM 2.1)
# ============================================================================
YOLO_MODEL_ID = "deepghs/manga109_yolo"
YOLO_CONFIDENCE = 0.3
YOLO_TEXT_CONFIDENCE = 0.20

SAM2_ENABLED = True
SAM2_MODEL_SIZE = "tiny"
SAM2_DEVICE = "cpu"
SAM2_FALLBACK_TO_BBOX = True

ZBUFFER_ENABLED = True
ZBUFFER_WEIGHT_Y = 0.5
ZBUFFER_WEIGHT_AREA = 0.5

# ============================================================================
# DIRETÓRIOS E CACHE
# ============================================================================
DATA_DIR = Path("./data")
CACHE_DIR = DATA_DIR / "cache"
MASKS_DIR = DATA_DIR / "masks"

for dir_path in [CACHE_DIR, MASKS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
