"""Resolução de elenco chapter-wide (re-identificação com contexto temporal).

Estratégia (decisão de 2026-09-21):
- YOLO propõe candidatos (bom em achar, ruim em distinguir).
- VLM forte decide identidade vendo TODAS as páginas do capítulo de uma vez
  (set-of-marks: bboxes numerados desenhados na página + downscale ~768px).
- Chamada única quando couber no contexto (16-32k); janela deslizante com
  carry-over como fallback para capítulos longos.
- VLM NUNCA escreve prompt de colorização — só responde identidade em JSON
  travado (com retry em resposta malformada). Prompt final continua
  determinístico no orchestrator.
- Saída cacheável: dramatis_personae.json por capítulo.

Nada aqui roda inferência pesada local (só monta payloads + chama o
VLM via HTTP + valida). Seguro para implementar sem GPU.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

logger = logging.getLogger("CastResolver")

# Lado maior da página enviada ao VLM (encoder tem custo fixo/imagem).
VLM_PAGE_MAX_SIDE = 768
# Teto de páginas por chamada única (16-32k ctx). Acima disso, janelas.
SINGLE_CALL_MAX_PAGES = 12
# Sobreposição entre janelas no modo fallback.
WINDOW_OVERLAP = 1


def draw_numbered_marks(
    page_image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    max_side: int = VLM_PAGE_MAX_SIDE,
) -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
    """Desenha bboxes numerados (set-of-marks) e reduz para o VLM.

    Retorna (imagem_marcada, boxes_reescalados). Números começam em 1
    por página; o mapeamento número->grupo é mantido pelo chamador.
    """
    w, h = page_image.size
    scale = min(1.0, max_side / max(w, h))
    small = page_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(small)
    scaled = []
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        sx1, sy1, sx2, sy2 = (int(v * scale) for v in (x1, y1, x2, y2))
        draw.rectangle([sx1, sy1, sx2, sy2], outline=(255, 0, 0), width=max(2, int(3 * scale) or 2))
        draw.text((sx1 + 2, sy1 + 2), str(i), fill=(255, 255, 255),
                  stroke_width=2, stroke_fill=(255, 0, 0))
        scaled.append((sx1, sy1, sx2, sy2))
    _ = scaled
    return small, boxes


CAST_SYSTEM_PROMPT = """You are a manga character tracker. You receive FULL manga pages in READING ORDER with numbered RED boxes (white halo digits) marking detected characters, plus optional ANCHOR portraits of already-known characters (labeled [Anchor ID]).

RULES:
- Same hair AND same outfit across consecutive pages = SAME person. Keep their ID, and cite the previous page in "why" (continuity counts as a cue).
- Anchors show ONE example look each — verify EVERY match with a SECOND independent cue (garment detail, hair length, facial hair, panel continuity). Resemblance alone is not enough.
- A look never seen before (new hair/outfit/face) = NEW person with a fresh id (P6, P7, ...) added to "cast". Expect 3-6 distinct people in a chapter, not 2.
- If more than two thirds of marks would share one ID, you are over-merging: split by outfit/hair differences first.
- Never recycle an ID for a different-looking person.
- A person can appear multiple times per page (different marks, same ID).
- Background figures in tiny panels: assign "EXTRA" (do not create IDs for them).

CONFIDENCE (use the whole range honestly):
- 0.95-1.0 only when anchor match AND a second cue agree.
- 0.7-0.9 when cues agree but no anchor exists.
- 0.5-0.7 when guessing between similar looks. Never emit 1.0 for a guess.

OUTPUT ONLY raw JSON, no markdown, no explanation. Every "why" must cite TWO independent cues in your own words (never copy these examples word for word):
{"cast": [{"id": "P1", "description": "...", "first_seen_page": 1}], "assignments": [{"page": 1, "mark": 2, "person_id": "P1", "confidence": 0.8, "why": "short brown hair like p.2 mark 1 plus red vest trim"}]}
"""


def build_cast_user_prompt(
    page_files: list[str],
    anchors: list[dict[str, Any]] | None = None,
    carry_over: dict[str, Any] | None = None,
    n_anchor_images: int = 0,
    sent_numbers: list[int] | None = None,
    mark_counts: dict[int, int] | None = None,
) -> str:
    """Texto que acompanha as imagens marcadas + âncoras."""
    lines = []
    if n_anchor_images:
        sent = sent_numbers or list(range(1, len(page_files) + 1))
        lines.append(
            f"You receive {n_anchor_images + len(sent)} images in order: "
            f"the first {n_anchor_images} are ANCHOR portraits (labeled [Anchor ID], "
            f"NOT pages, never assign marks to them), followed by chapter pages "
            f"{', '.join(str(n) for n in sent)} (labeled [Page i of {len(page_files)}]). "
            f"Use ONLY these page numbers in assignments. Red numbered boxes are character candidates."
        )
    if mark_counts:
        lines.append("Marks per page (assign EVERY one, none may be skipped):")
        for n in (sent_numbers or sorted(mark_counts)):
            lines.append(f"- Page {n}: marks 1-{mark_counts[n]}")
    else:
        lines.append(
            f"Pages 1-{len(page_files)} are the chapter in order. "
            f"Red numbered boxes are character candidates."
        )
    if anchors:
        lines.append("Known anchors (portraits attached as [Anchor ID] images — do NOT reassign these looks to new IDs):")
        for a in anchors:
            lines.append(f"- {a['id']}: {a.get('description', '')}")
    if carry_over and carry_over.get("cast"):
        lines.append("Previous window cast (KEEP these IDs consistent):")
        for c in carry_over["cast"]:
            lines.append(f"- {c['id']}: {c.get('description', '')}")
    lines.append("Assign every numbered mark. Respond ONLY the JSON.")
    return "\n".join(lines)


def parse_cast_response(
    content: str,
    expected_marks: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Parse + validação estrutural e de cobertura.

    Args:
        content: resposta crua do VLM.
        expected_marks: {nº_da_página: nº_de_marcas} (páginas 1-based na
            ordem enviada). Quando fornecido, exige cobertura exata:
            cada marca aparece uma única vez, e nenhuma página sem
            marcas recebe assignments (anti-alucinação).

    Levanta ValueError se inválido (o chamador faz retry).
    """
    import re as _re

    # Thinking models (ex. Qwen3.5) emitem <think>...</think> antes do final.
    # O trail é ouro para análise — o chamador o salva em .raw.txt — mas o
    # parse usa só o que vem depois.
    no_think = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL)
    cleaned = no_think.replace("```json", "").replace("```", "").strip()
    # Tolerância extra: extrai do primeiro { ao último }.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"Sem objeto JSON na resposta: {content[:200]}")
    data = json.loads(cleaned[start:end + 1])
    if not isinstance(data, dict) or "assignments" not in data or "cast" not in data:
        raise ValueError("Resposta sem chaves 'cast'/'assignments'")
    for a in data["assignments"]:
        if not all(k in a for k in ("page", "mark", "person_id")):
            raise ValueError(f"Assignment incompleto: {a}")

    if expected_marks is not None:
        seen: dict[int, set[int]] = {}
        for a in data["assignments"]:
            try:
                pg, mk = int(a["page"]), int(a["mark"])
            except Exception:
                raise ValueError(f"page/mark não-inteiros: {a}")
            if pg not in expected_marks:
                raise ValueError(f"Assignment para página inexistente: {pg}")
            if not 1 <= mk <= expected_marks[pg]:
                raise ValueError(
                    f"Marca {mk} fora do intervalo na página {pg} "
                    f"(esperado 1..{expected_marks[pg]})"
                )
            if mk in seen.setdefault(pg, set()):
                raise ValueError(f"Marca duplicada: página {pg} marca {mk}")
            seen[pg].add(mk)
        for pg, total in expected_marks.items():
            got = len(seen.get(pg, set()))
            if got != total:
                raise ValueError(
                    f"Cobertura incompleta na página {pg}: {got}/{total} marcas"
                )
    return data


def save_dramatis(output_path: str | Path, data: dict[str, Any]) -> str:
    """Persiste o dramatis (JSON legível/editável)."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def detect_tiled(
    page_path: str | Path,
    detector,
    tiles: tuple[int, int] = (2, 2),
    overlap: float = 0.2,
    iou_merge: float = 0.4,
) -> list:
    """Detecção em tiles para páginas onde o full-page falha (ex. capas).

    Divide em grade `tiles` com sobreposição, detecta por tile com offset
    de volta à resolução original e funde por NMS simples (mesma classe,
    IoU > `iou_merge` mantém maior confiança). Retorna DetectionResults.
    """
    import cv2 as _cv2

    img = _cv2.imread(str(page_path))
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {page_path}")
    h, w = img.shape[:2]
    cols, rows = tiles
    step_x = w / (cols - (cols - 1) * (1 - overlap) if cols > 1 else 1)
    step_y = h / (rows - (rows - 1) * (1 - overlap) if rows > 1 else 1)
    win_w = w / cols + (w - w / cols) * overlap if cols > 1 else w
    win_h = h / rows + (h - h / rows) * overlap if rows > 1 else h

    from core.detection.yolo_detector import YOLODetector as _YD

    all_dets: list = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(min(c * (w - win_w) / max(cols - 1, 1), w - win_w)) if cols > 1 else 0
            y0 = int(min(r * (h - win_h) / max(rows - 1, 1), h - win_h)) if rows > 1 else 0
            x1, y1 = int(x0 + win_w), int(y0 + win_h)
            tile = img[y0:y1, x0:x1]
            for d in detector.detect(tile):
                bx1, by1, bx2, by2 = d.bbox
                d.bbox = (bx1 + x0, by1 + y0, bx2 + x0, by2 + y0)
                all_dets.append(d)

    all_dets.sort(key=lambda d: d.confidence, reverse=True)
    kept: list = []
    for cand in all_dets:
        dup = False
        for k in kept:
            if k.class_id == cand.class_id and _YD._bbox_iou(k.bbox, cand.bbox) >= iou_merge:
                dup = True
                break
        if not dup:
            kept.append(cand)
    return kept


def resolve_chapter(
    pages: list[tuple[str, list[tuple[int, int, int, int]]]],
    anchors: list[dict[str, Any]] | None = None,
    output_path: str | Path | None = None,
    debug_dir: str | Path | None = None,
    max_retries: int = 3,
    timeout: float = 1200.0,
    temperature: float = 0.1,
    max_tokens: int = 32768,
) -> dict[str, Any]:
    """Resolve o elenco do capítulo em chamada única (quando couber).

    Args:
        pages: [(caminho_da_página, [bboxes body/face na resolução original])].
        anchors: retratos conhecidos [{id, description, image?}] onde image
            é PIL ou caminho — enviada como [Anchor ID] antes das páginas.
        output_path: destino do dramatis_personae.json (opcional; a resposta
            crua vai para o lado, <stem>.raw.txt, para análise do raciocínio).
        debug_dir: salva as páginas marcadas enviadas ao VLM (opcional).
        max_retries: tentativas em resposta malformada/falha.
        timeout: timeout por chamada VLM (s).

    Retorna o dramatis validado. Levanta RuntimeError após esgotar retries.
    """
    from PIL import Image as _Image

    from core.identity.vlm_service import VLMService

    if len(pages) > SINGLE_CALL_MAX_PAGES:
        raise ValueError(
            f"{len(pages)} páginas > teto de chamada única ({SINGLE_CALL_MAX_PAGES}). "
            "Use janelas com carry-over (a implementar no batch)."
        )

    images: list[_Image.Image] = []
    labels: list[str] = []
    for a in anchors or []:
        if a.get("image") is not None:
            ai = a["image"]
            ai = _Image.open(ai).convert("RGB") if isinstance(ai, (str, Path)) else ai.convert("RGB")
            images.append(ai)
            labels.append(f"[Anchor {a['id']}]")

    # Páginas sem marcas não são enviadas: só gastam contexto e o modelo
    # as conta na numeração (ex. capa-âncora virava "page 1" fantasma).
    # A cobertura continua indexada pela ordem original dos arquivos.
    marked: list[_Image.Image] = []
    marked_numbers: list[int] = []
    for idx, (page_path, boxes) in enumerate(pages):
        if not boxes:
            continue
        img = _Image.open(page_path).convert("RGB")
        m, _ = draw_numbered_marks(img, boxes)
        marked.append(m)
        marked_numbers.append(idx + 1)
        if debug_dir:
            d = Path(debug_dir)
            d.mkdir(parents=True, exist_ok=True)
            m.save(d / f"marked_{idx + 1:03d}.png")

    images.extend(marked)
    n_anchors = len(images) - len(marked)
    total = len(pages)
    labels.extend(f"[Page {n} of {total}]" for n in marked_numbers)

    # Mapa marca->bbox para o payload Qwen-Image ("marca 2" sozinha é órfã).
    # {nº página (ordem original): [{mark, bbox}]}. Páginas sem marcas: [].
    marks_map: dict[int, list[dict[str, Any]]] = {
        i + 1: [{"mark": m + 1, "bbox": list(b)} for m, b in enumerate(boxes)]
        for i, (_, boxes) in enumerate(pages)
    }

    # Cobertura exata esperada: {nº página 1-based: nº de marcas}.
    expected = {i + 1: len(boxes) for i, (_, boxes) in enumerate(pages)}

    base_prompt = build_cast_user_prompt(
        [p for p, _ in pages],
        anchors=anchors,
        n_anchor_images=n_anchors,
        sent_numbers=marked_numbers,
        mark_counts={n: expected[n] for n in marked_numbers},
    )
    vlm = VLMService()
    last_error: str = "sem resposta"
    for attempt in range(1, max_retries + 1):
        # Feedback do erro anterior: o modelo corrige em vez de repetir.
        user_prompt = base_prompt
        if attempt > 1:
            user_prompt += (
                f"\nYour previous answer was REJECTED: {last_error}. "
                f"Fix exactly that and respond ONLY the corrected JSON."
            )
        resp = vlm.ask_cast_identities(
            images,
            system_prompt=CAST_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            timeout=timeout,
            image_labels=labels,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if resp is None:
            last_error = f"tentativa {attempt}: VLM sem resposta"
            logger.warning(last_error)
            continue
        content = resp.get("content", "")
        thinking = resp.get("reasoning", "")
        logger.info(
            f"tentativa {attempt}: finish={resp.get('finish_reason')} "
            f"content={len(content)}ch think={len(thinking)}ch"
        )
        if output_path:
            base = Path(str(output_path))
            base.with_suffix(f".attempt{attempt}.raw.txt").write_text(
                content, encoding="utf-8"
            )
            if thinking:
                base.with_suffix(f".attempt{attempt}.think.txt").write_text(
                    thinking, encoding="utf-8"
                )
        if not content:
            last_error = (
                f"tentativa {attempt}: conteúdo vazio "
                f"(finish={resp.get('finish_reason')}, think={len(thinking)}ch)"
            )
            logger.warning(last_error)
            continue
        try:
            data = parse_cast_response(content, expected_marks=expected)
        except ValueError as e:
            last_error = f"tentativa {attempt}: resposta inválida ({e}): {content[:200]}"
            logger.warning(last_error)
            continue
        # Flag anti-colapso: raciocínios idênticos em massa indicam
        # carimbo (template-matching), não atribuição real. Não rejeita —
        # apenas sinaliza para o reviewer / eval comparativo.
        whys = [str(a.get("why", "")) for a in data["assignments"]]
        top_freq = 0.0
        if whys:
            from collections import Counter as _Counter

            top_freq = max(_Counter(whys).values()) / len(whys)
        data["meta"] = {
            "pages": len(pages),
            "anchors": len(anchors or []),
            "attempt": attempt,
            "top_why_freq": round(top_freq, 3),
            "collapse_warning": bool(top_freq > 0.7 and len(data["assignments"]) >= 10),
        }
        if output_path:
            save_dramatis(output_path, data)
            Path(str(output_path)).with_suffix(".raw.txt").write_text(
                content, encoding="utf-8"
            )
            marks_path = Path(str(output_path)).with_name("marks.json")
            marks_path.write_text(
                json.dumps(marks_map, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        data["marks_map"] = marks_map
        return data

    raise RuntimeError(f"Resolução de elenco falhou: {last_error}")
