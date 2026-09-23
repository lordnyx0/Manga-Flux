import os
import json
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional

class LayerRole(str, Enum):
    TEXT_MASK = "text_mask"
    PERSON_MASK = "person"
    BACKGROUND_MASK = "background"

class PromptBuilder:
    """Responsável por construir o conditioning textual a partir dos metadados extraídos pelo Pass1."""
    
    def __init__(self, metadata: dict):
        self.metadata = metadata

    def build_global_prompt(self, base_style_prompt: str = "colorMangaKlein, vibrant colors, anime style, highly detailed shading, masterpiece") -> str:
        # Extrai infos de cena detectadas no Passo 1
        scene_type = self.metadata.get("scene_type", "unknown")
        
        prompt_parts = [base_style_prompt]
        if scene_type != "unknown" and scene_type != "present":
            prompt_parts.append(f"{scene_type} style environment")
            
        return ", ".join(prompt_parts)

class MaskBinder:
    """
    Responsável por determinar o propósito das máscaras. Na Fase B inicial,
    o objetivo primário é resgatar/isolar balões de texto das áreas ativas.
    """
    
    def __init__(self, text_mask_path: str):
        self.text_mask_path = text_mask_path
    
    def get_text_preservation_mask(self):
        text_mask_path_str = self.text_mask_path
        # Garante que só aceitamos strings/Path — qualquer outro tipo (numpy array,
        # dict, None) é tratado como ausência de máscara.
        if not isinstance(text_mask_path_str, (str, Path)):
            return None
        if not text_mask_path_str:
            return None
        if not os.path.exists(text_mask_path_str):
            return None

        from PIL import Image
        # Masks are typically L mode (grayscale), black=background, white=mask
        mask = Image.open(text_mask_path_str).convert("L")
        return mask

class StyleBinder:
    """
    Responsável por gerenciar a injecão de embeds/estilos globais de referência.
    Na Fase B Inicial, extraímos a paleta cromática da referência e injetamos
    semanticamente como texto no prompt (contornando limites de VRAM/IP-Adapter).
    """
    
    def __init__(self, style_reference_path: str):
        self.style_reference_path = style_reference_path
        
    def get_global_style_image(self):
        if not self.style_reference_path or not os.path.exists(self.style_reference_path):
            return None
        from PIL import Image
        return Image.open(self.style_reference_path).convert("RGB")

    def get_style_prompt(self) -> str:
        """Extrai paleta de cores dominante da referência e converte em texto (Fluxo A)."""
        if not self.style_reference_path or not os.path.exists(self.style_reference_path):
            return ""
        try:
            from PIL import Image
            from core.identity.palette_manager import PaletteExtractor, generate_prompt_from_palette
            img = Image.open(self.style_reference_path).convert("RGB")
            extractor = PaletteExtractor()
            palette = extractor.extract(img)
            desc = generate_prompt_from_palette(palette)
            if desc:
                return f"color reference palette: {desc}"
        except Exception:
            pass
        return ""

def _position_of(bbox, img_width: int) -> str:
    x1, _, x2, _ = bbox
    cx = (x1 + x2) / 2.0 / max(img_width, 1)
    if cx < 0.35:
        return "on the LEFT"
    if cx > 0.65:
        return "on the RIGHT"
    return "in the CENTER"


def prepare_cast_payload(
    page_path: str,
    page_marks: list[dict],
    assignments: list[dict],
    dramatis_cast: list[dict],
    registry,
    style_cover_path: str | None = None,
    max_crops: int = 8,
) -> dict:
    """Monta o payload Qwen-Image a partir do dramatis validado.

    Args:
        page_path: página P&B a colorir.
        page_marks: [{"mark": N, "bbox": [x1,y1,x2,y2]}] (marks.json).
        assignments: [{"mark": N, "person_id": ...}] desta página.
        dramatis_cast: [{"id": ..., "description": ...}].
        registry: CharacterRegistry (ledger; pode estar vazio na 1ª vez).
        style_cover_path: capa colorida como <image2> (opcional).
        max_crops: teto de crops (image_3..).

    Retorna dict pronto para QwenEngine.generate (prompt, qwen_prompt,
    style_image, ref_crops, base_image_path).
    """
    from PIL import Image as _Image

    descs = {c["id"]: c.get("description", c["id"]) for c in dramatis_cast}
    by_mark = {m["mark"]: m for m in page_marks}
    img = _Image.open(page_path).convert("RGB")
    w, _ = img.size

    lines: list[str] = []
    ref_crops: list = []
    seen_ids: list[str] = []
    for a in sorted(assignments, key=lambda x: x["mark"]):
        pid = a["person_id"]
        if pid == "EXTRA":
            continue
        m = by_mark.get(a["mark"])
        if not m:
            continue
        pos = _position_of(tuple(m["bbox"]), w)
        lines.append(f"character {pos} (mark {a['mark']}) = {pid} ({descs.get(pid, pid)})")
        if pid not in seen_ids:
            seen_ids.append(pid)
            # Crop de referência: ledger (colorido registrado) > âncora >
            # crop da própria página.
            crop = None
            reg = registry.get(pid) if registry else None
            rp = (reg or {}).get("ref_crop")
            if rp and Path(rp).exists():
                crop = _Image.open(rp).convert("RGB")
            if crop is None:
                x1, y1, x2, y2 = (max(0, int(v)) for v in m["bbox"])
                crop = img.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))
            if len(ref_crops) < max_crops:
                ref_crops.append(crop)

    prompt = (
        "Colorize the ENTIRE black-and-white manga page in <image1> preserving "
        "exact lineart, panel layout and screentones: every person (named or "
        "not), animal, object, clothing, background and sky gets full color. "
        "Only speech bubbles and SFX text stay white. "
        "Use the color identity from <image2>. "
    )
    if lines:
        prompt += "Cast on this page: " + "; ".join(lines) + ". "
    if len(seen_ids) > 1:
        prompt += (
            "These are DIFFERENT people — never swap their colors: "
            + ", ".join(f"{pid} is NOT {other}" for pid in seen_ids for other in seen_ids if other != pid)
            + ". "
        )
    reg_block = registry.prompt_block() if registry else ""
    if reg_block:
        prompt += reg_block + " "
    prompt += (
        "The same person keeps the SAME hair/skin/clothes colors in EVERY "
        "panel, including night, shadow and rain scenes — shading, screentone "
        "or darkness never changes identity colors. "
        "Flat anime colors, no photorealism, leave speech bubbles white."
    )

    style_image = None
    if style_cover_path and Path(style_cover_path).exists():
        style_image = _Image.open(style_cover_path).convert("RGB")

    return {
        "prompt": prompt,
        "qwen_prompt": prompt,
        "style_image": style_image,
        "ref_crops": ref_crops,
        "base_image_path": str(page_path),
    }


class Pass2Orchestrator:
    """
    Orquestra a preparação do payload para a Engine (Agnóstica).
    Lê o JSON, aciona os Binders, e gera um dict limpo para qualquer Engine rodar.
    """
    
    def __init__(self, meta_json_path: str, masks_dir: str, style_ref_path: str):
        self.meta_json_path = meta_json_path
        self.masks_dir = masks_dir
        self.style_ref_path = style_ref_path
        self.metadata = {}

        if os.path.exists(meta_json_path):
            with open(meta_json_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

        self.prompt_builder = PromptBuilder(self.metadata)
        self.style_binder = StyleBinder(style_ref_path)

        # Registro global de personagens extraído da capa pelo VLM (describe_all_characters).
        # Quando presente, habilita geração de prompt modular via Gemma.
        self.vlm_character_registry: Optional[list] = None

        # Inicializar o serviço de busca vetorial FAISS se o estilo existir
        self.faiss_service = None
        self.reference_characters_count = 0
        
        if style_ref_path and os.path.exists(style_ref_path):
            try:
                from core.pass1_analyzer import Pass1Analyzer
                from core.identity.faiss_service import FaissService

                # Executa Pass1Analyzer na imagem de estilo (como página -1) para detectar personagens
                analyzer = Pass1Analyzer()
                style_result = analyzer.analyze_page(style_ref_path, page_num=-1)

                self.faiss_service = FaissService()
                for det in style_result.get("detections", []):
                    if det.get("class_name") in ("body", "face") and det.get("body_embedding"):
                        # FIX: usa body_embedding (embedding real 1D) ao invés de embedding (aninhado)
                        emb = det["body_embedding"]
                        if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                            emb = emb[0]  # desaninha [[...]] -> [...]
                        self.faiss_service.add_reference_character(emb, det)
                        self.reference_characters_count += 1

                # Captura o registry VLM gerado durante a análise da capa (describe_all_characters).
                # Isso ativa o Caminho 1 (prompt modular via Gemma) em prepare_generation_payload().
                captured_registry = getattr(analyzer, "_vlm_global_characters", None)
                if captured_registry:
                    self.vlm_character_registry = captured_registry
                    print(
                        f"[Pass2Orchestrator] Registry VLM capturado da capa: "
                        f"{len(self.vlm_character_registry)} personagem(ns). "
                        f"Caminho 1 (prompt modular Gemma) ativado."
                    )
                else:
                    print(
                        "[Pass2Orchestrator] Registry VLM não disponível (VLM offline ou sem personagens). "
                        "Usando Caminho 2 (FAISS/paleta/base)."
                    )

                print(f"[Pass2Orchestrator] Inicializado: {self.reference_characters_count} personagens indexados a partir da referência {style_ref_path}")
            except Exception as e:
                print(f"[Pass2Orchestrator] Falha ao analisar e indexar imagem de referência: {e}. Fallback global ativo.")
                self.faiss_service = None
        
    def prepare_generation_payload(self) -> dict:
        """
        Retorna um dicionário puro que não depende da arquitetura (Flux, SDXL, Qwen).
        """
        # Garantir que builders tenham acesso aos metadados atualizados
        self.prompt_builder.metadata = self.metadata
        
        text_mask_path = self.metadata.get("text_mask")
        
        # Tratar dicionários ou strings de caminhos dependendo da extração do Pass1
        if isinstance(text_mask_path, dict) and "path" in text_mask_path:
            text_mask_path = text_mask_path["path"]
            
        mask_binder = MaskBinder(text_mask_path)
        
        # Obter prompt básico de cena
        base_style_prompt = "colorMangaKlein, vibrant colors, anime style, highly detailed shading, masterpiece"
        base_prompt = self.prompt_builder.build_global_prompt(base_style_prompt=base_style_prompt)
        
        # Fluxo A vs Fluxo B
        matched_character_descs = []

        def _is_neutral_color(desc: str) -> bool:
            """Retorna True se o descritor de cor só tem tons neutros (manga P&B)."""
            neutral_words = {"white", "black", "gray", "light gray"}
            tokens = set(desc.lower().replace(",", " ").split())
            return tokens.issubset(neutral_words | {"", "hair", "clothes", "skin", "eyes",
                                                     "accessories", "clothes details"})

        if self.faiss_service is not None and self.reference_characters_count > 0:
            # Fluxo A: Mapeamento local fino por similaridade de embeddings (corpo/rosto)
            # FIX: threshold reduzido de 0.5 para 0.35 — embeddings P&B vs cor têm menor cosseno
            detections = self.metadata.get("detections", [])
            for det in detections:
                if det.get("class_name") in ("body", "face"):
                    # FIX: usa body_embedding para busca (mesmo espaço que a referência usa)
                    emb = det.get("body_embedding") or det.get("embedding")
                    if not emb:
                        continue
                    if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                        emb = emb[0]  # desaninha [[...]] -> [...]

                    matched_ref, sim = self.faiss_service.search(emb, threshold=0.35)
                    if matched_ref:
                        # Prioritiza descrição rica do VLM caso exista, com fallback para paleta clássica
                        desc = matched_ref.get("vlm_description") or matched_ref.get("palette_string")
                        if not desc:
                            palette_dict = matched_ref.get("palette")
                            if palette_dict:
                                try:
                                    from core.identity.palette_manager import CharacterPalette, generate_prompt_from_palette
                                    palette_obj = CharacterPalette.from_dict(palette_dict)
                                    desc = generate_prompt_from_palette(palette_obj)
                                except Exception as e:
                                    print(f"[Pass2Orchestrator] Erro ao instanciar paleta do match: {e}")

                        # FIX: pula descritores neutros (só cinza/branco — imagem P&B)
                        if desc and not _is_neutral_color(desc):
                            x1 = det["bbox"][0]
                            img_size = self.metadata.get("image_size", [1024, 1024])
                            img_w = img_size[0] if isinstance(img_size, list) else 1024
                            position = "in the center"
                            if x1 < img_w * 0.35:
                                position = "on the left"
                            elif x1 > img_w * 0.65:
                                position = "on the right"
                            matched_character_descs.append(
                                f"character {position} with {desc}"
                            )
                            print(f"[Pass2Orchestrator] Fluxo A match (sim={sim:.3f}): character {position} with {desc}")
                        else:
                            print(f"[Pass2Orchestrator] Fluxo A: match encontrado (sim={sim:.3f}) mas descrição é neutra ou vazia, saltando.")
                    else:
                        # FIX: usa palette_string do metadata da página como contexto adicional
                        # (pode ser "white hair, white clothes" para P&B — útil para manter coerência)
                        ps = det.get("palette_string")
                        if ps and not _is_neutral_color(ps):
                            matched_character_descs.append(f"character with {ps}")

        # ── Caminho 1: Prompt Modular via VLM (Gemma) ────────────────────────────
        # Ativado quando o registro global de personagens da capa está disponível.
        # O Gemma analisa a página P&B + registry e gera um prompt estruturado em
        # seções [Layout]/[Character Design]/[Color Mapping]/[Lighting]/[Background]/[Rendering].
        final_prompt = None

        page_image_path = self.metadata.get("page_image")
        if self.vlm_character_registry and page_image_path:
            try:
                # Recupera prompt_hint das runtime_options se fornecidas para resolver ambiguidades cromáticas de forma genérica
                prompt_hint = None
                if hasattr(self, "runtime_options") and isinstance(self.runtime_options, dict):
                    prompt_hint = self.runtime_options.get("prompt_hint")

                from core.identity.vlm_service import VLMService
                vlm = VLMService()
                modular_prompt = vlm.generate_modular_flux_prompt(
                    bw_page_image=page_image_path,
                    character_registry=self.vlm_character_registry,
                    prompt_hint=prompt_hint,
                )
                if modular_prompt:
                    final_prompt = modular_prompt
                    print(
                        f"[Pass2Orchestrator] Prompt Modular VLM (Gemma) gerado "
                        f"({len(modular_prompt)} chars)."
                    )
                else:
                    print("[Pass2Orchestrator] VLM retornou prompt vazio — usando fallback.")
            except Exception as e:
                print(f"[Pass2Orchestrator] Falha ao gerar prompt modular via VLM: {e} — usando fallback.")

        # ── Caminho 2: Prompt Flat (fallback FAISS / paleta / base) ───────────────
        if final_prompt is None:
            if matched_character_descs:
                final_prompt = f"{base_prompt}. Character color details: {', and '.join(matched_character_descs)}"
                print(f"[Pass2Orchestrator] Injeção Semântica Local (Fluxo A): {final_prompt}")
            else:
                style_prompt = self.style_binder.get_style_prompt()
                if style_prompt:
                    final_prompt = f"{base_prompt}, {style_prompt}"
                    print(f"[Pass2Orchestrator] Injeção Semântica Global (Fallback Fluxo A): {final_prompt}")
                else:
                    final_prompt = base_prompt
                    print(f"[Pass2Orchestrator] Colorização Sem Referência (Fluxo B): {final_prompt}")
            
        payload = {
            "prompt": final_prompt,
            "style_image": self.style_binder.get_global_style_image(),
            "text_preservation_mask": mask_binder.get_text_preservation_mask(),
            "base_image_path": self.metadata.get("page_image")
        }
        return payload

    # ── Qwen-Image-2.1 Edit (determinístico, sem Gemma por página) ────────

    QWEN_EDIT_TEMPLATE = (
        "Colorize the black-and-white manga page in <image1> preserving "
        "exact lineart, panel layout and screentones. "
        "Use the color identity from <image2>.{crops}{palette} "
        "Flat anime colors, no photorealism, leave speech bubbles white."
    )

    def position_of(
        self, bbox: tuple[int, int, int, int], img_width: int
    ) -> str:
        """Posição horizontal do bbox (para grounding espacial no prompt)."""
        x1, _, x2, _ = bbox
        cx = (x1 + x2) / 2.0 / max(img_width, 1)
        if cx < 0.35:
            return "on the LEFT"
        if cx > 0.65:
            return "on the RIGHT"
        return "in the CENTER"

    def build_qwen_edit_prompt(
        self,
        matched_character_descs: list,
        style_prompt: str = "",
        num_crops: int = 0,
    ) -> str:
        """Prompt determinístico com tags <imageN> obrigatórias.

        - <image1> = página P&B, <image2> = style_ref (sempre presentes).
        - <image3..N> citados só se crops reais forem enviados (num_crops).
        - Sem trigger Flux (colorMangaKlein) e sem seções livres do Gemma.
        """
        crops_txt = ""
        if num_crops > 0:
            last = 2 + num_crops
            crops_txt = (
                f" Character reference crops in <image3>-<image{last}> "
                "show the same characters; match their hair, skin, eyes and outfits exactly."
            )
        palette_txt = ""
        if matched_character_descs:
            palette_txt = " Character color details: " + "; ".join(matched_character_descs) + "."
        elif style_prompt:
            palette_txt = f" {style_prompt}."
        return self.QWEN_EDIT_TEMPLATE.format(crops=crops_txt, palette=palette_txt)

    def prepare_qwen_edit_payload(
        self,
        faiss_threshold: float = 0.35,
        max_ref_crops: int = 8,
        ref_crops_dir: str | None = None,
    ) -> dict:
        """Payload agnóstico para a QwenEngine (sem nenhuma chamada VLM).

        Retorna as mesmas chaves do payload Flux + `qwen_prompt` e
        `ref_crops` (lista de PIL crops da style_ref para image_3..N).
        Fail-safe: 0 detecções ou 0 matches → prompt global só com
        image_1+image_2, nunca aborta.
        """
        from pathlib import Path as _Path

        self.prompt_builder.metadata = self.metadata
        text_mask_path = self.metadata.get("text_mask")
        if isinstance(text_mask_path, dict) and "path" in text_mask_path:
            text_mask_path = text_mask_path["path"]
        mask_binder = MaskBinder(text_mask_path)

        # Override manual para teste A/B: engine carrega do diretório.
        if ref_crops_dir and _Path(ref_crops_dir).is_dir():
            style_prompt = self.style_binder.get_style_prompt()
            manual = sorted(
                [p for p in _Path(ref_crops_dir).iterdir()
                 if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
            )[:max_ref_crops]
            final_prompt = self.build_qwen_edit_prompt([], style_prompt, len(manual))
            return {
                "prompt": final_prompt,
                "qwen_prompt": final_prompt,
                "style_image": self.style_binder.get_global_style_image(),
                "ref_crops": [str(p) for p in manual],
                "text_preservation_mask": mask_binder.get_text_preservation_mask(),
                "base_image_path": self.metadata.get("page_image"),
            }

        matched_descs: list[str] = []
        ref_crops: list = []
        try:
            if self.faiss_service is not None and self.reference_characters_count > 0:
                style_img = self.style_binder.get_global_style_image()
                for det in self.metadata.get("detections", []):
                    if det.get("class_name") not in ("body", "face"):
                        continue
                    emb = det.get("body_embedding") or det.get("embedding")
                    if not emb:
                        continue
                    if isinstance(emb, list) and emb and isinstance(emb[0], list):
                        emb = emb[0]
                    matched_ref, sim = self.faiss_service.search(emb, threshold=faiss_threshold)
                    if not matched_ref or sim < faiss_threshold:
                        continue
                    desc = matched_ref.get("vlm_description") or matched_ref.get("palette_string")
                    if not desc:
                        palette_dict = matched_ref.get("palette")
                        if palette_dict:
                            try:
                                from core.identity.palette_manager import (
                                    CharacterPalette, generate_prompt_from_palette,
                                )
                                desc = generate_prompt_from_palette(CharacterPalette.from_dict(palette_dict))
                            except Exception:
                                desc = None
                    if desc:
                        matched_descs.append(f"character with {desc} (match sim={sim:.2f})")
                    # Crop visual da style_ref via bbox da referência (PIL, barato).
                    if len(ref_crops) < max_ref_crops and style_img is not None:
                        bbox = matched_ref.get("bbox")
                        try:
                            if bbox and len(bbox) == 4:
                                x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
                                if x2 > x1 and y2 > y1:
                                    crop = style_img.crop((x1, y1, x2, y2))
                                    if crop.size[0] >= 4 and crop.size[1] >= 4:
                                        ref_crops.append(crop)
                        except Exception:
                            continue
        except Exception as e:
            print(f"[Pass2Orchestrator] Qwen match falhou ({e}); usando fallback global.")

        style_prompt = "" if matched_descs else self.style_binder.get_style_prompt()
        final_prompt = self.build_qwen_edit_prompt(matched_descs, style_prompt, len(ref_crops))
        return {
            "prompt": final_prompt,
            "qwen_prompt": final_prompt,
            "style_image": self.style_binder.get_global_style_image(),
            "ref_crops": ref_crops,
            "text_preservation_mask": mask_binder.get_text_preservation_mask(),
            "base_image_path": self.metadata.get("page_image"),
        }
