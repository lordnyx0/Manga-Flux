"""Colorização por painel (padrão global anti-bleed/anti-inserção).

Evidência (DOCS/QWEN_MIGRATION.md, pág-004 cap.1): página inteira com todos
os IDs mistura identidades (bleed) e inventa gente da referência; painel
isolado com 1 ID não tem com o que trocar nem onde inserir.

Regras (valem para qualquer mangá):
- 1 geração por painel que contém marcas; painéis sem marcas vão no fluxo
  genérico (sem IDs no prompt).
- Micro-painel (<15% da área ou lado <600px): gera com margem de contexto
  (12%, mín. 48px) e recompõe só o miolo — painel nu deriva composição.
- 1 crop por ID (o builder já garante); style cover como <image2>.
- Composite por colagem exata no bbox do frame (bordas pretas escondem seams).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

MIN_PANEL_FRACTION = 0.15
MIN_PANEL_SIDE = 600
CTX_MARGIN_RATIO = 0.12
CTX_MARGIN_MIN = 48


def assign_marks_to_panels(
    marks: list[dict],
    frames: list[tuple[int, int, int, int]],
) -> dict[int, list[dict]]:
    """Mapeia marcas -> índice do frame que contém seu centro (-1 = fora)."""
    out: dict[int, list[dict]] = {}
    for m in marks:
        x1, y1, x2, y2 = m["bbox"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        pi = -1
        for i, (fx1, fy1, fx2, fy2) in enumerate(frames):
            if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
                pi = i
                break
        out.setdefault(pi, []).append(m)
    return out


def colorize_page_panels(
    page_path: str | Path,
    page_marks: list[dict],
    assignments: list[dict],
    dramatis_cast: list[dict],
    registry,
    style_cover_path: str | None,
    engine,
    out_path: str | Path,
    seed_base: int = 0,
    steps: int = 25,
) -> dict[str, Any]:
    """Coloriza uma página painel a painel e recompõe. Retorna resumo.

    - Painel com IDs: payload do dramatis restrito ao painel.
    - Painel sem marcas: payload genérico (sem IDs, sem crops).
    - Sem frames detectados: página inteira de uma vez.
    """
    import cv2

    from core.detection.yolo_detector import YOLODetector
    from core.generation.orchestrator import prepare_cast_payload

    page_path = str(page_path)
    full = Image.open(page_path).convert("RGB")
    FW, FH = full.size
    page_area = FW * FH

    img = cv2.imread(page_path)
    det = YOLODetector(conf_threshold=0.3)
    frames = sorted(
        [tuple(d.bbox) for d in det.detect(img) if d.class_id == 2],
        key=lambda b: (b[1], b[0]),
    )
    by_mark = {m["mark"]: m for m in page_marks}
    canvas = full.copy()
    jobs: list[dict[str, Any]] = []

    if not frames:
        jobs.append({"bbox": (0, 0, FW, FH), "marks": page_marks,
                     "assign": assignments, "panel": -1})
    else:
        pmap = assign_marks_to_panels(page_marks, frames)
        for pi, fr in enumerate(frames):
            pm = pmap.get(pi, [])
            pa = [a for a in assignments if a["mark"] in {m["mark"] for m in pm}]
            jobs.append({"bbox": fr, "marks": pm, "assign": pa, "panel": pi})
        # Marcas fora de qualquer frame: gera junto num job de fallback.
        orphans = pmap.get(-1, [])
        if orphans:
            oa = [a for a in assignments if a["mark"] in {m["mark"] for m in orphans}]
            jobs.append({"bbox": (0, 0, FW, FH), "marks": orphans,
                         "assign": oa, "panel": -2})

    for ji, job in enumerate(jobs):
        fx1, fy1, fx2, fy2 = job["bbox"]
        pw, ph = fx2 - fx1, fy2 - fy1
        tiny = (pw * ph < page_area * MIN_PANEL_FRACTION
                or min(pw, ph) < MIN_PANEL_SIDE)
        if tiny:
            mx = max(CTX_MARGIN_MIN, int(pw * CTX_MARGIN_RATIO))
            my = max(CTX_MARGIN_MIN, int(ph * CTX_MARGIN_RATIO))
            ex1, ey1 = max(0, fx1 - mx), max(0, fy1 - my)
            ex2, ey2 = min(FW, fx2 + mx), min(FH, fy2 + my)
        else:
            ex1, ey1, ex2, ey2 = fx1, fy1, fx2, fy2
        crop_path = Path(str(out_path) + f".job{ji}.png")
        full.crop((ex1, ey1, ex2, ey2)).save(crop_path)
        shifted = [{"mark": m["mark"],
                    "bbox": [m["bbox"][0] - ex1, m["bbox"][1] - ey1,
                             m["bbox"][2] - ex1, m["bbox"][3] - ey1]}
                   for m in job["marks"]]
        payload = prepare_cast_payload(
            str(crop_path), shifted, job["assign"], dramatis_cast,
            registry, style_cover_path=style_cover_path)
        panel_img, stats = engine.generate(
            payload, seed=seed_base + ji, strength=1.0,
            options={"steps": steps})
        ow, oh = panel_img.size
        inner = panel_img.crop((
            int((fx1 - ex1) / (ex2 - ex1) * ow),
            int((fy1 - ey1) / (ey2 - ey1) * oh),
            int((fx2 - ex1) / (ex2 - ex1) * ow),
            int((fy2 - ey1) / (ey2 - ey1) * oh),
        )).resize((fx2 - fx1, fy2 - fy1), Image.LANCZOS)
        canvas.paste(inner, (fx1, fy1))
        try:
            crop_path.unlink()
        except OSError:
            pass
        job["stats"] = stats

    canvas.save(out_path)
    return {"out": str(out_path),
            "jobs": [{"panel": j["panel"], "bbox": j["bbox"],
                      "ids": sorted({a["person_id"] for a in j["assign"]
                                     if a["person_id"] != "EXTRA"})}
                     for j in jobs]}
