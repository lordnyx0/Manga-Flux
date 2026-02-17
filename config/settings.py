"""
Manga-Flux v1.0 - Configurações do Sistema
Configurações otimizadas para o pipeline Flux Klein 9B + SAM 2.1 + YOLO.
"""

import os
from pathlib import Path

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - portability fallback
    class _TorchCudaShim:
        @staticmethod
        def is_available() -> bool:
            return False

    class _TorchShim:
        bfloat16 = "bfloat16"
        float32 = "float32"
        cuda = _TorchCudaShim()

    torch = _TorchShim()  # type: ignore

# ============================================================================
# HARDWARE E PERFORMANCE
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Otimizações para 12GB VRAM
OFFLOAD_TO_CPU = True
ENABLE_VAE_SLICING = True

# ============================================================================
# PASS 1: ANÁLISE (YOLO + SAM 2.1)
# ============================================================================
YOLO_MODEL_ID = "deepghs/manga109_yolo"
YOLO_CONFIDENCE = 0.3
YOLO_TEXT_CONFIDENCE = 0.20
DETECTION_IOU_THRESHOLD = 0.45

SAM2_ENABLED = True
SAM2_MODEL_SIZE = "tiny"
SAM2_DEVICE = "cpu"
SAM2_FALLBACK_TO_BBOX = True

ZBUFFER_ENABLED = True
ZBUFFER_WEIGHT_Y = 0.5
ZBUFFER_WEIGHT_AREA = 0.5
ZBUFFER_WEIGHT_TYPE = 0.0

CONTEXT_INFLATION_FACTOR = 1.5
MASK_MIN_AREA_RATIO = 0.0001
MASK_MAX_COMPONENTS = 256

# ============================================================================
# PASS 1: IDENTIDADE / PALETA (compatibilidade com porta do /manga)
# ============================================================================
INSIGHTFACE_MODEL = "buffalo_l"
FACE_DETECTION_SIZE = (640, 640)
EMBEDDING_DIM = 768
EMBEDDING_CACHE_FORMAT = "pt"
MAX_CACHED_EMBEDDINGS = 2048

PALETTE_REGIONS = ["hair", "skin", "eyes", "clothes"]
PALETTE_COLORS_PER_REGION = 5
PALETTE_DRIFT_THRESHOLD = 12.0
TEMPORAL_SMOOTHING = 0.35
COLOR_SPACE = "lab"

VERBOSE = os.getenv("MANGA_FLUX_VERBOSE", "0") == "1"

# ============================================================================
# DIRETÓRIOS E CACHE
# ============================================================================
DATA_DIR = Path("./data")
CACHE_DIR = DATA_DIR / "cache"
MASKS_DIR = DATA_DIR / "masks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

for dir_path in [CACHE_DIR, MASKS_DIR, EMBEDDINGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
