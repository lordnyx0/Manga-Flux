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
