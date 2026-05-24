import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger("StructureGuard")

class StructureGuard:
    """
    Fase C: Validador de Estrutura e Alucinações de Traço.
    Compara a lineart da imagem P&B original contra a lineart da imagem colorida final
    usando Adaptive Canny, Distance Transform e fatiamento por painéis.
    """
    
    def __init__(self, distance_tolerance: int = 10, dice_threshold: float = 0.75):
        self.distance_tolerance = distance_tolerance
        self.dice_threshold = dice_threshold
        
    def create_edge_map(self, image_gray: np.ndarray, is_colorized: bool = False) -> np.ndarray:
        """
        Isola a lineart do mangá minimizando alucinações de sombreamento.
        """
        # Filtro bilateral para suavizar tons sem destruir bordas fortes
        if is_colorized:
            blurred = cv2.bilateralFilter(image_gray, d=15, sigmaColor=75, sigmaSpace=75)
        else:
            blurred = cv2.bilateralFilter(image_gray, d=9, sigmaColor=50, sigmaSpace=50)

        # Threshold adaptativo local
        block_size = 21 if is_colorized else 15
        c_off = 8 if is_colorized else 5
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, c_off
        )
        
        # Limpeza morfológica
        kernel_clean = np.ones((3, 3), np.uint8)
        if is_colorized:
            edges = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
        else:
            edges = thresh
            
        # Dilatação leve para aumentar tolerância espacial
        kernel_dil = np.ones((2, 2), np.uint8)
        return cv2.dilate(edges, kernel_dil, iterations=1)

    def calculate_dice_coefficient(self, edges_orig: np.ndarray, edges_color: np.ndarray) -> float:
        """
        Calcula o coeficiente de similaridade Dice entre dois mapas de bordas binários.
        """
        _, b1 = cv2.threshold(edges_orig, 127, 1, cv2.THRESH_BINARY)
        _, b2 = cv2.threshold(edges_color, 127, 1, cv2.THRESH_BINARY)
        intersection = np.sum(b1 & b2)
        total = np.sum(b1) + np.sum(b2)
        if total == 0:
            return 1.0
        return float(2.0 * intersection / total)

    def calculate_edge_anomaly_dt(
        self, edges_orig: np.ndarray, edges_color: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula anomalias de traço (linhas adicionadas/perdidas) via Distance Transform.
        """
        _, orig_bin = cv2.threshold(edges_orig, 127, 255, cv2.THRESH_BINARY)
        _, color_bin = cv2.threshold(edges_color, 127, 255, cv2.THRESH_BINARY)
        
        orig_inv = cv2.bitwise_not(orig_bin)
        color_inv = cv2.bitwise_not(color_bin)
        
        dist_orig = cv2.distanceTransform(orig_inv, cv2.DIST_L2, 3)
        dist_color = cv2.distanceTransform(color_inv, cv2.DIST_L2, 3)
        
        added_lines = np.zeros_like(orig_bin)
        added_lines[(color_bin == 255) & (dist_orig > self.distance_tolerance)] = 255
        
        lost_lines = np.zeros_like(orig_bin)
        lost_lines[(orig_bin == 255) & (dist_color > self.distance_tolerance)] = 255
        
        kernel = np.ones((3, 3), np.uint8)
        added_lines = cv2.morphologyEx(added_lines, cv2.MORPH_OPEN, kernel)
        lost_lines = cv2.morphologyEx(lost_lines, cv2.MORPH_OPEN, kernel)
        
        return added_lines, lost_lines

    def extract_panels(self, img_gray: np.ndarray, target_shape: Tuple[int, int] = None) -> List[Tuple[int, int, int, int]]:
        """
        Extrai as coordenadas dos quadros (painéis) do mangá.
        """
        orig_h, orig_w = img_gray.shape[:2]
        _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        kernel = np.ones((3, 3), np.uint8)
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        panels = []
        img_area = orig_h * orig_w
        
        scale_x = 1.0
        scale_y = 1.0
        if target_shape is not None:
            target_h, target_w = target_shape[:2]
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area > img_area * 0.02 and area < img_area * 0.95:
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                scaled_w = int(w * scale_x)
                scaled_h = int(h * scale_y)
                panels.append((scaled_x, scaled_y, scaled_w, scaled_h))
                
        panels.sort(key=lambda b: (b[1], b[0]))
        return panels

    def validate_page(self, orig_path: str, color_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Executa a validação estrutural ponta a ponta.
        """
        if not os.path.exists(orig_path) or not os.path.exists(color_path):
            return {"status": "error", "error": "Arquivos de entrada não encontrados."}
            
        img_orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
        img_color_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
        
        if img_orig is None or img_color_bgr is None:
            return {"status": "error", "error": "Falha ao carregar imagens via OpenCV."}
            
        img_color_gray = cv2.cvtColor(img_color_bgr, cv2.COLOR_BGR2GRAY)
        
        # Alinha dimensões caso difiram
        if img_orig.shape != img_color_gray.shape:
            img_orig = cv2.resize(img_orig, (img_color_gray.shape[1], img_color_gray.shape[0]))
            
        # 1. Mapa de Bordas
        edges_orig = self.create_edge_map(img_orig, is_colorized=False)
        edges_color = self.create_edge_map(img_color_gray, is_colorized=True)
        
        # 2. Coeficiente Dice
        dice_score = self.calculate_dice_coefficient(edges_orig, edges_color)
        
        # 3. Anomalias pontuais via Distance Transform
        added_lines, lost_lines = self.calculate_edge_anomaly_dt(edges_orig, edges_color)
        
        # 4. Fatiamento de painéis
        panels = self.extract_panels(img_orig, target_shape=img_color_gray.shape)
        
        panel_diagnostics = []
        combined_error_mask = cv2.bitwise_or(added_lines, lost_lines)
        
        # Kernel para expandir zona de inpaint
        inpaint_kernel = np.ones((25, 25), np.uint8)
        final_inpaint_mask = cv2.dilate(combined_error_mask, inpaint_kernel, iterations=2)
        
        for i, (x, y, w, h) in enumerate(panels):
            panel_area = w * h
            panel_mask = final_inpaint_mask[y:y+h, x:x+w]
            failed_pixels = cv2.countNonZero(panel_mask)
            failure_ratio = (failed_pixels / panel_area) if panel_area > 0 else 0.0
            
            verdict = "OK"
            if failure_ratio > 0.30:
                verdict = "CRITICAL"
            elif failure_ratio > 0.10:
                verdict = "MICRO_INPAINT"
                
            panel_diagnostics.append({
                "panel_index": i,
                "bbox": [x, y, x+w, y+h],
                "failure_ratio": float(failure_ratio),
                "verdict": verdict
            })
            
        verdict = "ACCEPTABLE" if dice_score >= self.dice_threshold else "CRITICAL_FAILURE"
        
        result = {
            "status": "success",
            "dice_score": dice_score,
            "verdict": verdict,
            "panels": panel_diagnostics,
            "detected_panels_count": len(panels)
        }
        
        # Salvar overlays para visualização se diretório for fornecido
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            basename = os.path.basename(color_path)
            
            # Criar overlay colorido
            overlay = img_color_bgr.copy()
            kernel_vis = np.ones((3, 3), np.uint8)
            added_vis = cv2.dilate(added_lines, kernel_vis, iterations=1)
            lost_vis = cv2.dilate(lost_lines, kernel_vis, iterations=1)
            
            overlay[added_vis == 255] = [0, 0, 255]  # Vermelho: Linhas adicionadas
            overlay[lost_vis == 255] = [255, 0, 0]   # Azul: Linhas perdidas
            
            for p in panel_diagnostics:
                x1, y1, x2, y2 = p["bbox"]
                color = (0, 255, 0)
                if p["verdict"] == "CRITICAL":
                    color = (0, 0, 255)
                elif p["verdict"] == "MICRO_INPAINT":
                    color = (0, 255, 255)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 4)
                
            overlay_path = os.path.join(output_dir, f"struct_overlay_{basename}")
            mask_path = os.path.join(output_dir, f"struct_trigger_mask_{basename}")
            
            cv2.imwrite(overlay_path, overlay)
            cv2.imwrite(mask_path, final_inpaint_mask)
            
            result["overlay_path"] = overlay_path
            result["trigger_mask_path"] = mask_path
            
        return result
