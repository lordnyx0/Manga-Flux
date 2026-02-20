import os
import json
from enum import Enum
from pathlib import Path

class LayerRole(str, Enum):
    TEXT_MASK = "text_mask"
    PERSON_MASK = "person"
    BACKGROUND_MASK = "background"

class PromptBuilder:
    """Responsável por construir o conditioning textual a partir dos metadados extraídos pelo Pass1."""
    
    def __init__(self, metadata: dict):
        self.metadata = metadata

    def build_global_prompt(self, base_style_prompt: str = "A highly detailed manga panel, high quality, vibrant colors") -> str:
        # Extrai infos de cena ou personagens globais, se houver
        scene_info = self.metadata.get("scene_analysis", {})
        time_of_day = scene_info.get("time_of_day", "unknown")
        scene_type = scene_info.get("dominant_type", "unknown")
        
        # Constrói conditioning básico baseado na cena detectada
        prompt_parts = [base_style_prompt]
        if time_of_day != "unknown":
            prompt_parts.append(f"{time_of_day} lighting")
        if scene_type != "unknown":
            prompt_parts.append(f"{scene_type} environment")
            
        return ", ".join(prompt_parts)

class MaskBinder:
    """
    Responsável por determinar o propósito das máscaras. Na Fase B inicial,
    o objetivo primário é resgatar/isolar balões de texto das áreas ativas.
    """
    
    def __init__(self, text_mask_path: str):
        self.text_mask_path = text_mask_path
    
    def get_text_preservation_mask(self):
        """Retorna a máscara combinada de todos os balões/textos para preservação."""
        if not self.text_mask_path or not os.path.exists(self.text_mask_path):
            return None
            
        from PIL import Image
        # Masks are typically L mode (grayscale), black=background, white=mask
        mask = Image.open(self.text_mask_path).convert("L")
        return mask

class StyleBinder:
    """
    Responsável por gerenciar a injecão de embeds/estidos globais de referência.
    Na Fase B Inicial, operamos através de prompt / IP-Adapter em nível global (sem malha espacial precisa).
    """
    
    def __init__(self, style_reference_path: str):
        self.style_reference_path = style_reference_path
        
    def get_global_style_image(self):
        if not self.style_reference_path or not os.path.exists(self.style_reference_path):
            return None
        from PIL import Image
        return Image.open(self.style_reference_path).convert("RGB")

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
        
    def prepare_generation_payload(self) -> dict:
        """
        Retorna um dicionário puro que não depende da arquitetura (Flux, SDXL, Qwen).
        """
        # Ensure builders have access to the potentially injected metadata
        self.prompt_builder.metadata = self.metadata
        
        # Instantiate MaskBinder dynamically here so it catches the injected metadata correctly
        text_mask_path = self.metadata.get("text_mask")
        
        # Handle dict or path strings depending on Pass1 extraction
        if isinstance(text_mask_path, dict) and "path" in text_mask_path:
            text_mask_path = text_mask_path["path"]
            
        mask_binder = MaskBinder(text_mask_path)
        
        payload = {
            "prompt": self.prompt_builder.build_global_prompt(),
            "style_image": self.style_binder.get_global_style_image(),
            "text_preservation_mask": mask_binder.get_text_preservation_mask(),
            "base_image_path": self.metadata.get("page_image")
        }
        return payload
