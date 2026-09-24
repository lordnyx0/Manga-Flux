"""Segmentação de cenas por capítulo (para janelamento do resolvedor).

Princípio: janela = 1-2 cenas, nunca N páginas fixas. Dentro da cena a
continuidade visual é densa (mesmo lugar/temporoupa); a fronteira é onde o
handoff de IDs importa.

Sinais (baratos, sem VLM):
1. Cortes duros: mudança cor↔P&B, mudança de dimensão (spread/página).
2. Queda de similaridade CLIP entre páginas vizinhas (página inteira em
   baixa resolução) abaixo de media - k*desvio.
3. Teto/mínimo de tamanho para não gerar micro-cenas nem janelas gigantes.

Saída: scenes.json — [{"pages": [idx...], "reason": "..."}] (índices 1-based
na ordem dos arquivos).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def page_signature_clip(page_path: str | Path, processor, model, device: str) -> np.ndarray:
    """Embedding CLIP da página inteira (256px, barato)."""
    import torch

    img = Image.open(page_path).convert("RGB").resize((256, 256), Image.LANCZOS)
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        feat = out.image_embeds if hasattr(out, "image_embeds") else out[0]
        feat = feat.float()
        if feat.dim() == 3:
            feat = feat.mean(dim=1)  # mean-pool dos patches p/ vetor global
    v = feat[0].detach().cpu().numpy().astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def is_color_page(page_path: str | Path) -> bool:
    import cv2

    img = cv2.imread(str(page_path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) > 12.0


def segment_pages(
    page_files: list[str | Path],
    max_window: int = 10,
    min_scene: int = 2,
    k: float = 1.0,
) -> list[dict[str, Any]]:
    """Segmenta em cenas. Retorna [{"pages": [...], "reason": ...}]."""
    from transformers import CLIPImageProcessor, CLIPModel

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    n = len(page_files)
    embs = [page_signature_clip(p, processor, model, device) for p in page_files]
    colors = [is_color_page(p) for p in page_files]
    sizes = [Image.open(p).size for p in page_files]

    # Cortes duros primeiro.
    hard = set()
    for i in range(1, n):
        if colors[i] != colors[i - 1] or sizes[i] != sizes[i - 1]:
            hard.add(i)  # corte ANTES da página i (0-based) -> cena nova

    sims = [float(embs[i] @ embs[i - 1]) for i in range(1, n)]
    mu, sd = float(np.mean(sims)), float(np.std(sims)) + 1e-9
    soft = {i for i, s in enumerate(sims, start=1) if s < mu - k * sd}

    cuts = sorted(hard | soft)
    scenes: list[dict[str, Any]] = []
    start = 0
    for c in cuts + [n]:
        scenes.append({"pages": list(range(start + 1, c + 1)),
                       "reason": "hard" if c in hard else ("soft" if c in cuts else "end")})
        start = c

    # Normaliza: funde micro-cenas (<min) e quebra gigantes (>max).
    norm: list[dict[str, Any]] = []
    for s in scenes:
        if norm and len(s["pages"]) < min_scene:
            norm[-1]["pages"].extend(s["pages"])
            norm[-1]["reason"] += "+merged"
        else:
            norm.append(s)
    final: list[dict[str, Any]] = []
    for s in norm:
        while len(s["pages"]) > max_window:
            final.append({"pages": s["pages"][:max_window], "reason": s["reason"] + "+split"})
            s = {"pages": s["pages"][max_window:], "reason": s["reason"] + "+cont"}
        final.append(s)
    return final


def save_scenes(output_path: str | Path, scenes: list[dict[str, Any]]) -> str:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)
