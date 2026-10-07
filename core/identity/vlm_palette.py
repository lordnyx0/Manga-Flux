"""Qwen aponta, heurística escreve (registro de cor com autoridade VLM).

Divisão de trabalho (decisão de 2026-09-26):
- QWEN (juiz): diz a cor com PALAVRAS por região, diz se a região está
  visível/mensurável e se o recorte presta (corpo inteiro vs close só-rosto).
  Palavras, nunca hexadecimal (VLM arredonda/inventa hex).
- HEURÍSTICA (régua): mede o hexadecimal dentro da região apontada, com
  filtros de sombra e nota de confiança (`character_registry`).
- ACORDO: palavra x número concordam -> grava no prontuário; discordam
  ou recorte ruim -> segura para revisão humana, nunca grava lixo.

Nada aqui exige GPU/servidor para importar ou testar: `ask_vlm_palette`
é o único ponto com HTTP e retorna None offline (fallback preservado).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger("VLMPalette")

REGIONS = ("hair", "skin", "eyes", "clothes")

VLM_COLOR_SYSTEM = """You are a manga character color registrar. You NEVER invent hexadecimal codes.
Analyze the character crop and respond ONLY raw JSON, no markdown, no explanation:
{"hair": {"color": "<1-3 english words or null>", "usable": true/false},
 "skin": {"color": "<1-3 english words or null>", "usable": true/false},
 "eyes": {"color": "<1-3 english words or null>", "usable": true/false},
 "clothes": {"color": "<1-3 english words or null>", "usable": true/false},
 "crop_ok": true/false,
 "note": "<≤10 words: why any region is unusable>"}
RULES:
- "color" describes the TRUE surface color ignoring shadow/night/screentone (e.g. "dark purple", not "gray shadow").
- "usable": false when the region is not visible, cropped out, or covered by shadow/text.
- "crop_ok": false when this is a face-only close-up or otherwise useless for body palette (hair/skin ok, clothes not).
- null color + usable:false is always better than a guess."""


def parse_vlm_palette_response(content: str) -> dict[str, Any]:
    """Parse + validação do JSON do juiz. Levanta ValueError se inválido."""
    no_think = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    cleaned = no_think.replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"Sem objeto JSON: {content[:160]}")
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("Resposta não é objeto")
    if not any(r in data for r in REGIONS):
        raise ValueError("Sem nenhuma região (hair/skin/eyes/clothes)")
    out: dict[str, Any] = {"crop_ok": bool(data.get("crop_ok", True)),
                           "note": str(data.get("note", ""))[:200]}
    for r in REGIONS:
        node = data.get(r, {})
        if not isinstance(node, dict):
            raise ValueError(f"Região '{r}' ausente ou malformada")
        color = node.get("color")
        color = str(color).strip().lower() if color else None
        out[r] = {"color": color, "usable": bool(node.get("usable", False))}
    return out


def ask_vlm_palette(vlm_service, image: Union[str, Path, Any],
                    timeout: float = 300.0) -> dict[str, Any] | None:
    """Pergunta ao Qwen as cores em palavras. None = offline/falha."""
    try:
        from core.identity.vlm_service import VLMService  # noqa: F401
        b64 = vlm_service._encode_image_to_base64(image)
        model = vlm_service._get_active_model()
        import requests
        resp = requests.post(
            vlm_service.base_url + "/chat/completions",
            headers={"Content-Type": "application/json"},
            json={"model": model,
                  "temperature": 0.0,
                  "max_tokens": 1024,
                  "messages": [
                      {"role": "system", "content": VLM_COLOR_SYSTEM},
                      {"role": "user", "content": [
                          {"type": "text", "text": "Register this character's true colors."},
                          {"type": "image_url", "image_url": {
                              "url": "data:image/png;base64," + b64}}]}]},
            timeout=timeout)
        if resp.status_code != 200:
            logger.warning(f"VLM status {resp.status_code}")
            return None
        msg = resp.json()["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        return parse_vlm_palette_response(content)
    except Exception as e:  # noqa: BLE001 - offline-safe por desenho
        logger.warning(f"Juiz indisponível: {e}")
        return None


# Palavras -> RGB aproximado (só para checagem grosseira palavra x número).
BASE_RGB = {
    "black": (30, 30, 30), "white": (240, 240, 240), "gray": (140, 140, 140),
    "grey": (140, 140, 140), "red": (200, 40, 40), "orange": (220, 130, 40),
    "yellow": (220, 200, 60), "green": (50, 160, 80), "cyan": (60, 180, 200),
    "blue": (70, 120, 210), "indigo": (70, 70, 160), "purple": (130, 70, 170),
    "violet": (130, 70, 170), "pink": (230, 150, 180), "brown": (120, 80, 50),
    "blonde": (220, 200, 130), "peach": (240, 200, 170), "skin": (235, 200, 175),
    "beige": (225, 205, 175), "gold": (210, 170, 60), "silver": (190, 190, 200),
    "teal": (50, 150, 150),
}


def word_to_rgb(word: str) -> tuple[int, int, int] | None:
    """'dark purple' -> RGB aproximado (modificadores light/dark/pale/deep)."""
    w = word.lower()
    base = next((BASE_RGB[k] for k in BASE_RGB if k in w), None)
    if base is None:
        return None
    r, g, b = base
    if any(m in w for m in ("light", "pale", "pastel")):
        r, g, b = (r + 255) // 2, (g + 255) // 2, (b + 255) // 2
    if any(m in w for m in ("dark", "deep", "navy")):
        r, g, b = int(r * 0.55), int(g * 0.55), int(b * 0.55)
    return (r, g, b)


def word_hex_agree(word: str | None, hexcode: str | None,
                   tol: float = 90.0) -> bool:
    """Palavra e número falam da mesma cor? Tolerância larga (palavra é grosseira)."""
    if not word or not hexcode:
        return False
    ref = word_to_rgb(word)
    if ref is None:
        return False
    h = (int(hexcode[1:3], 16), int(hexcode[3:5], 16), int(hexcode[5:7], 16))
    return float(sum((a - b) ** 2 for a, b in zip(ref, h)) ** 0.5) <= tol


def adjudicate(vlm: dict[str, Any],
               measured: dict[str, str | None]) -> dict[str, dict[str, Any]]:
    """Acordo por região: confirmed | held | conflict."""
    out: dict[str, dict[str, Any]] = {}
    crop_ok = bool(vlm.get("crop_ok", True))
    for r in REGIONS:
        node = vlm.get(r, {})
        word, usable = node.get("color"), bool(node.get("usable", False))
        hexcode = measured.get(r)
        if not crop_ok and r == "clothes":
            out[r] = {"verdict": "held", "reason": "crop face-only",
                      "vlm_word": word, "measured_hex": hexcode}
        elif not usable or word is None:
            out[r] = {"verdict": "held", "reason": "VLM marcou inutilizável",
                      "vlm_word": word, "measured_hex": hexcode}
        elif hexcode is None:
            out[r] = {"verdict": "held", "reason": "sem medição",
                      "vlm_word": word, "measured_hex": hexcode}
        elif word_hex_agree(word, hexcode):
            out[r] = {"verdict": "confirmed", "reason": "palavra=número",
                      "vlm_word": word, "measured_hex": hexcode}
        else:
            out[r] = {"verdict": "conflict", "reason": f"'{word}' x {hexcode}",
                      "vlm_word": word, "measured_hex": hexcode}
    return out


def register_character_vlm_first(registry, person_id: str,
                                 colorized_path: Union[str, Path],
                                 body_bbox: tuple[int, int, int, int],
                                 page_num: int, vlm_service=None,
                                 description: str = "",
                                 input_size: tuple[int, int] | None = None,
                                 ) -> dict[str, Any]:
    """Entrada única do E2E: tenta juiz+VLM, cai para heurística pura offline.

    Retorna o entry do ledger (com `region_verdicts` quando o juiz participou,
    ou `vlm_skipped: True` no fallback).
    """
    vlm = ask_vlm_palette(vlm_service, colorized_path) if vlm_service else None
    if vlm is None:
        entry = registry.register_from_colorized(
            person_id, colorized_path, body_bbox, page_num,
            description=description, input_size=input_size)
        entry["vlm_skipped"] = True
        registry.save()
        return entry
    return registry.register_from_vlm(
        person_id, colorized_path, body_bbox, page_num, vlm,
        description=description, input_size=input_size)
