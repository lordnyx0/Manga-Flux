"""Ledger de personagens por capítulo (memória de cor entre páginas).

Problema que resolve: um estreante colorizado pela primeira vez ganha uma
cor inventada pelo modelo; sem registro, a próxima aparição inventa outra.
Aqui a invenção acontece UMA vez (na estreia) e depois é lida do ledger
— via prompt textual (hex) + crop colorido de referência visual.

Formato (JSON legível/editável à mão):
{
  "chapter": "chapter_001",
  "characters": {
    "P3": {"description": "...", "status": "confirmed|provisional",
           "first_seen_page": 3,
           "palette": {"hair": "#6B4A2F", "skin": null, "clothes": null},
           "ref_crop": "registry_crops/P3.png"}
  }
}

Extração de paleta é amostragem estatística simples (PIL, CPU): cabelo =
faixa superior do bbox, roupa = faixa média, pele = melhor esforço no
terço superior-central. Resultado é `provisional` até confirmação
(reaparição com sim alto ou curadoria humana).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


def _median_hex(crop: Image.Image) -> str | None:
    if crop.size[0] < 2 or crop.size[1] < 2:
        return None
    st = ImageStat.Stat(crop.convert("RGB"))
    r, g, b = (int(v) for v in st.median[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def scale_bbox(
    bbox: tuple[int, int, int, int],
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Reescala bbox entre resoluções (Qwen varia o tamanho da saída)."""
    fw, fh = max(from_size[0], 1), max(from_size[1], 1)
    tw, th = to_size
    x1, y1, x2, y2 = bbox
    return (
        int(x1 * tw / fw), int(y1 * th / fh),
        int(x2 * tw / fw), int(y2 * th / fh),
    )


def extract_palette_from_colorized(
    colorized_path: str | Path,
    body_bbox: tuple[int, int, int, int],
    input_size: tuple[int, int] | None = None,
) -> dict[str, str | None]:
    """Amostra paleta aproximada de um corpo na imagem JÁ colorida.

    Args:
        body_bbox: bbox na resolução da PÁGINA DE ENTRADA.
        input_size: (w, h) da entrada — obrigatório, pois a saída do Qwen
            varia de tamanho (ex. 2500×3556 → 864×1216). Sem isso a
            amostragem cai na região errada.
    """
    img = Image.open(colorized_path).convert("RGB")
    w, h = img.size
    if input_size is not None:
        body_bbox = scale_bbox(tuple(body_bbox), input_size, (w, h))
    x1, y1, x2, y2 = (max(0, int(v)) for v in body_bbox)
    x1, y1 = min(x1, w - 1), min(y1, h - 1)
    x2, y2 = min(max(x2, x1 + 1), w), min(max(y2, y1 + 1), h)
    bw, bh = x2 - x1, y2 - y1
    hair = img.crop((x1, y1, x2, y1 + int(bh * 0.15)))
    clothes = img.crop((x1, y1 + int(bh * 0.35), x2, y1 + int(bh * 0.75)))
    skin = img.crop((
        x1 + int(bw * 0.35), y1 + int(bh * 0.15),
        x1 + int(bw * 0.65), y1 + int(bh * 0.30),
    ))
    return {
        "hair": _median_hex(hair),
        "skin": _median_hex(skin),
        "clothes": _median_hex(clothes),
    }


class CharacterRegistry:
    """CRUD do ledger + bloco de prompt."""

    def __init__(self, path: str | Path, chapter: str = "default"):
        self.path = Path(path)
        self.chapter = chapter
        self.data: dict[str, Any] = {"chapter": chapter, "characters": {}}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                self.data.setdefault("characters", {})
            except Exception:
                pass

    def save(self) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(self.path)

    def get(self, person_id: str) -> dict[str, Any] | None:
        return self.data["characters"].get(person_id)

    def seed(self, person_id: str, description: str = "",
             first_seen_page: int | None = None,
             ref_crop: str | None = None) -> dict[str, Any]:
        chars = self.data["characters"]
        entry = chars.setdefault(person_id, {
            "description": description,
            "status": "provisional",
            "first_seen_page": first_seen_page,
            "palette": {"hair": None, "eyes": None, "skin": None, "clothes": None},
            "ref_crop": ref_crop,
        })
        if description and not entry.get("description"):
            entry["description"] = description
        if ref_crop and not entry.get("ref_crop"):
            entry["ref_crop"] = ref_crop
        return entry

    def register_from_colorized(
        self,
        person_id: str,
        colorized_path: str | Path,
        body_bbox: tuple[int, int, int, int],
        page_num: int,
        description: str = "",
        save_crop: bool = True,
        input_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Registra/atualiza personagem a partir da saída colorida.

        `input_size` = (w, h) da página de entrada — sem ele, bboxes não
        são reescalados para a saída do Qwen (tamanhos variam!) e a
        amostragem/crop cai na região errada.
        """
        entry = self.seed(person_id, description, first_seen_page=page_num)
        fresh = extract_palette_from_colorized(
            colorized_path, body_bbox, input_size=input_size
        )
        # Merge sem destruir: amostragem ruim (bbox fora de escala, crop
        # degenerado) retorna None e NÃO pode apagar valor bom já registrado
        # (ex. escrita concorrente entre E2E e curadoria manual).
        merged = dict(entry.get("palette") or {})
        for k, v in fresh.items():
            if v is not None:
                merged[k] = v
        entry["palette"] = merged
        if save_crop:
            try:
                img = Image.open(colorized_path).convert("RGB")
                box = body_bbox
                if input_size is not None:
                    box = scale_bbox(tuple(body_bbox), input_size, img.size)
                x1, y1, x2, y2 = (max(0, int(v)) for v in box)
                crop = img.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))
                dest = self.path.parent / "registry_crops" / f"{person_id}.png"
                dest.parent.mkdir(parents=True, exist_ok=True)
                crop.save(dest)
                entry["ref_crop"] = str(dest)
            except Exception:
                pass
        self.save()
        return entry

    def confirm(self, person_id: str) -> None:
        entry = self.data["characters"].get(person_id)
        if entry:
            entry["status"] = "confirmed"
            self.save()

    def prompt_block(self) -> str:
        """Bloco textual do ledger para o prompt Qwen (só confirmados c/ paleta)."""
        parts = []
        for pid, e in self.data["characters"].items():
            pal = e.get("palette") or {}
            known = {k: v for k, v in pal.items() if v}
            if e.get("status") == "confirmed" and known:
                desc = e.get("description", pid)
                cols = ", ".join(f"{k} {v}" for k, v in known.items())
                parts.append(f"{pid} ({desc}) must keep: {cols}")
        if not parts:
            return ""
        return "Registered colors (HIGHEST priority, never contradict): " + "; ".join(parts) + "."
