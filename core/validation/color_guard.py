import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List, Optional
import logging

from core.identity.palette_manager import PaletteExtractor, CharacterPalette, ColorRegion

logger = logging.getLogger("ColorGuard")

class ColorGuard:
    """
    Fase C.5: Validador de Consistência Cromática (CIE-LAB Centroid Checker).
    Mede a distância euclidiana perceptualmente uniforme (Delta E) entre as regiões de cor
    dos personagens colorizados e seus perfis cromáticos de referência.
    """
    
    def __init__(self, drift_threshold: float = 20.0):
        self.drift_threshold = drift_threshold
        self.extractor = PaletteExtractor()
        
    def calculate_drift(self, ref_palette: CharacterPalette, gen_palette: CharacterPalette) -> Dict[str, float]:
        """
        Calcula a variação Delta E entre a paleta de referência e a gerada para cada região comum.
        """
        drift = {}
        for region_name, gen_region in gen_palette.regions.items():
            if region_name not in ref_palette.regions:
                continue
            
            ref_color = ref_palette.regions[region_name].dominant_color
            gen_color = gen_region.dominant_color
            
            # Calcula Delta E no espaço perceptual CIE-LAB
            delta_e = self.extractor.calculate_delta_e(ref_color, gen_color)
            drift[region_name] = float(delta_e)
            
        return drift

    def validate_page_colors(
        self, 
        colorized_image_path: str, 
        page_metadata: Dict, 
        faiss_service_instance = None
    ) -> Dict[str, Any]:
        """
        Analisa a imagem colorizada e compara as cores de cada personagem com a sua referência.
        """
        if not os.path.exists(colorized_image_path):
            return {"status": "error", "error": "Imagem colorizada não encontrada."}
            
        # Carrega a imagem colorizada
        img_color = Image.open(colorized_image_path).convert("RGB")
        img_color_np = np.array(img_color)
        
        detections = page_metadata.get("detections", [])
        char_diagnostics = []
        global_verdict = "ACCEPTABLE"
        
        for det in detections:
            if det.get("class_name") not in ("body", "face"):
                continue
                
            char_id = det.get("char_id", "unknown")
            bbox = det.get("bbox")
            
            if not bbox or not det.get("embedding"):
                continue
                
            # Recuperar a paleta de referência
            ref_palette_dict = None
            
            # Método 1: Tenta buscar pelo FaissService se fornecido
            if faiss_service_instance is not None:
                matched_ref, _ = faiss_service_instance.search(det["embedding"], threshold=0.5)
                if matched_ref:
                    ref_palette_dict = matched_ref.get("palette")
            
            # Método 2: Fallback para a paleta salva no próprio metadado se foi inserida anteriormente
            if not ref_palette_dict:
                ref_palette_dict = det.get("palette")
                
            if not ref_palette_dict:
                logger.debug(f"Nenhuma paleta de referência encontrada para {char_id}, ignorando.")
                continue
                
            try:
                # Recorta o personagem da imagem colorizada gerada
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_color.width, x2), min(img_color.height, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                char_crop = img_color_np[y1:y2, x1:x2]
                char_crop_pil = Image.fromarray(char_crop)
                
                # Extrai a paleta de cores gerada no crop
                gen_palette = self.extractor.extract(char_crop_pil)
                ref_palette = CharacterPalette.from_dict(ref_palette_dict)
                
                # Mede a distância Delta E
                drifts = self.calculate_drift(ref_palette, gen_palette)
                
                has_violation = False
                violations_regions = []
                
                for region, score in drifts.items():
                    if score > self.drift_threshold:
                        has_violation = True
                        violations_regions.append(region)
                        global_verdict = "COLOR_DRIFT_WARNING"
                        
                char_diagnostics.append({
                    "char_id": char_id,
                    "bbox": bbox,
                    "drifts": drifts,
                    "has_violation": has_violation,
                    "violation_regions": violations_regions,
                    "verdict": "DRIFT_VIOLATION" if has_violation else "OK"
                })
                
            except Exception as e:
                logger.error(f"Erro ao processar validação cromática do personagem {char_id}: {e}")
                
        return {
            "status": "success",
            "verdict": global_verdict,
            "character_diagnostics": char_diagnostics,
            "validated_characters_count": len(char_diagnostics)
        }
