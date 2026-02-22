import cv2
import numpy as np
import os
import sys

# Ensure we can import core modules
sys.path.append(r'C:\Users\Nyx\Desktop\MANGACOLOR\Manga-Flux')
from core.detection.yolo_detector import YOLODetector

def get_yolo_objects(img, detector):
    """
    Returns panels (class 2) and characters (class 0, 1) from the image.
    """
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    detections = detector.detect(img_color)
    
    panels = []
    bodies = []
    faces = []
    
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        box = [x1, y1, x2 - x1, y2 - y1]
        
        if d.class_id == 2: # Frame
            panels.append(box)
        elif d.class_id == 0: # Body
            bodies.append(box)
        elif d.class_id == 1: # Face
            faces.append(box)
            
    # Sort panels top-to-bottom
    panels.sort(key=lambda b: (b[1], b[0]))
    return panels, bodies, faces

def count_objects_in_crop(crop, detector):
    """
    Runs YOLO purely on a cropped panel to find semantic objects inside it.
    """
    if crop.size == 0:
        return 0, 0
        
    if len(crop.shape) == 2 or (len(crop.shape) == 3 and crop.shape[2] == 1):
        img_color = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    else:
        img_color = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
    detections = detector.detect(img_color)
    
    bodies = 0
    faces = 0
    for d in detections:
        if d.class_id == 0:
            bodies += 1
        elif d.class_id == 1:
            faces += 1
            
    return bodies, faces

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

def scale_box(box, scale_x, scale_y):
    x, y, w, h = box
    return (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y))

def analyze_semantics(orig_path, color_path, output_dir, detector):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(color_path)
    
    img_orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    img_color_bgra = cv2.imread(color_path, cv2.IMREAD_COLOR)
    img_color_gray = cv2.cvtColor(img_color_bgra, cv2.COLOR_BGR2GRAY)
    
    orig_h, orig_w = img_orig.shape[:2]
    col_h, col_w = img_color_gray.shape[:2]
    
    scale_x = col_w / orig_w
    scale_y = col_h / orig_h

    print("--- 1. Extracting Base Topology ---")
    orig_panels, _, _ = get_yolo_objects(img_orig, detector)
    color_panels, _, _ = get_yolo_objects(img_color_bgra, detector)
    
    overlay = img_color_bgra.copy()
    
    print("\n--- 2. Semantic Hallucination Graphing ---")
    
    if len(orig_panels) == 0:
        print("No panels found in original. Skipping.")
        return

    for i, orig_box in enumerate(orig_panels):
        scaled_orig_box = scale_box(orig_box, scale_x, scale_y)
        sx, sy, sw, sh = scaled_orig_box
        
        # Match topology first (Assume phase_c_topology logic)
        best_iou = 0.0
        best_match_idx = -1
        for j, color_box in enumerate(color_panels):
            iou = calculate_iou(scaled_orig_box, color_box)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = j
                
        # If topology completely failed, we skip semantic check
        if best_iou < 0.60:
            print(f"Panel {i} {scaled_orig_box}: SKIPPED SEMANTICS (Failed Topology IoU: {best_iou:.2f})")
            cv2.rectangle(overlay, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2, lineType=cv2.LINE_4)
            continue
            
        print(f"Panel {i} {scaled_orig_box}: Topology OK. Running semantic check inside crop...")
        # Draw base green box for surviving panel
        cx, cy, cw, ch = color_panels[best_match_idx]
        cv2.rectangle(overlay, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 2)
        
        # Extract the pure pixel crops for this isolated panel
        crop_orig = img_orig[orig_box[1]:orig_box[1]+orig_box[3], orig_box[0]:orig_box[0]+orig_box[2]]
        crop_color = img_color_bgra[cy:cy+ch, cx:cx+cw]
        
        # Run YOLO Semantics exclusively inside the Panel Context
        orig_bodies, orig_faces = count_objects_in_crop(crop_orig, detector)
        col_bodies, col_faces = count_objects_in_crop(crop_color, detector)
        
        # --- NEW: GEOMETRIC COHERENCE (EDGE BASED) ---
        # Ensure grayscale
        def to_gray(img):
            if img is None or img.size == 0: return None
            if len(img.shape) == 3 and img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img

        crop_orig_gray = to_gray(crop_orig)
        crop_color_gray = to_gray(crop_color)

        if crop_orig_gray is None or crop_color_gray is None:
            print(f"  -> SKIPPING GEOMETRY: Empty crop.")
            continue

        # Resize original to match colorized for pixel-wise edge comparison
        crop_orig_resized = cv2.resize(crop_orig_gray, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
        
        # Edge Detection
        blur_c = cv2.bilateralFilter(crop_color_gray, d=9, sigmaColor=75, sigmaSpace=75)
        edges_c = cv2.Canny(blur_c, 50, 150)
        
        blur_o = cv2.GaussianBlur(crop_orig_resized, (3, 3), 0)
        v = np.median(blur_o)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edges_o = cv2.Canny(blur_o, lower, upper)
        
        # Thicken for tolerance
        kernel_dil = np.ones((3, 3), np.uint8)
        edges_o_dil = cv2.dilate(edges_o, kernel_dil, iterations=1)
        
        # Hallucinated Lines Detector
        dist_orig = cv2.distanceTransform(cv2.bitwise_not(edges_o_dil), cv2.DIST_L2, 3)
        added_lines = np.zeros_like(edges_c)
        added_lines[(edges_c == 255) & (dist_orig > 6)] = 255
        
        total_col_edges = cv2.countNonZero(edges_c)
        hallucinated_pixels = cv2.countNonZero(added_lines)
        hallucination_ratio = (hallucinated_pixels / max(1, total_col_edges)) * 100
        
        print(f"  -> Original Panel Semantics : {orig_bodies} bodies, {orig_faces} faces")
        print(f"  -> Colorized Panel Semantics: {col_bodies} bodies, {col_faces} faces")
        print(f"  -> Geometric Hallucination Index: {hallucination_ratio:.2f}% (Threshold: 20%)")
        
        # Formulate Discrepancy Score
        body_diff = col_bodies - orig_bodies
        
        verdict = "SEMANTICS OK"
        verdict_color = (0, 255, 0)
        alert_box_color = None
        
        # 1. Check for Semantic Count Mismatch (YOLO)
        if body_diff > 0:
            if body_diff >= 2:
                verdict = f"SEVERE HALLUCINATION (+{body_diff} Bodies)"
                alert_box_color = (0, 165, 255) # Orange
            else:
                verdict = f"MINOR HALLUCINATION (+1 Body)"
                alert_box_color = (0, 255, 255) # Yellow
        elif body_diff < 0:
            if orig_bodies > 0 and col_bodies == 0:
                verdict = "TOTAL COHERENCE LOSS (0 Bodies)"
                alert_box_color = (0, 0, 255) # Red
            elif abs(body_diff) >= 2:
                verdict = f"SEVERE ERASURE ({body_diff} Bodies)"
                alert_box_color = (0, 165, 255)
            else:
                verdict = f"MINOR ERASURE (-1 Body)"
                alert_box_color = (0, 255, 255)

        # 2. Check for Geometric Hallucination (Added Structural Noise)
        # If the character count is the same, but 20%+ of the lines are completely new/hallucinated
        if alert_box_color is None and hallucination_ratio > 20.0:
            if hallucination_ratio > 40.0:
                verdict = f"GEOMETRIC DISORDER ({hallucination_ratio:.1f}%)"
                alert_box_color = (0, 165, 255) # Orange
            else:
                verdict = f"STRUCTURAL NOISE ({hallucination_ratio:.1f}%)"
                alert_box_color = (0, 255, 255) # Yellow
                
        # Draw overlay results if altered
        if alert_box_color is not None:
            # Draw an inner thick boundary to signal semantic warning inside the topological green box
            cv2.rectangle(overlay, (cx+4, cy+4), (cx+cw-4, cy+ch-4), alert_box_color, 4)
            verdict_color = alert_box_color
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"P{i} [S]: {verdict}"
        (text_w, text_h), baseline = cv2.getTextSize(text, font, 0.7, 2)
        cv2.rectangle(overlay, (cx, cy + ch - text_h - baseline - 10), (cx + text_w, cy + ch), (0,0,0), -1)
        cv2.putText(overlay, text, (cx, cy + ch - baseline - 5), font, 0.7, verdict_color, 2)

    out_path = os.path.join(output_dir, f"semantic_overlay_{basename}")
    cv2.imwrite(out_path, overlay)
    print(f"\nSaved Semantic Analysis to: {out_path}")

if __name__ == "__main__":
    import glob
    
    print("Loading YOLO Detector (Manga109) for Semantic Tracing...")
    detector = YOLODetector()
    
    INPUT_DIR = r"C:\Users\Nyx\Desktop\MANGACOLOR\Manga-Flux\Doukutsu_Ou_kara_Hajimeru_Rakuen_Life\chapters\26_2\inputs"
    COLOR_DIR = r"C:\Users\Nyx\Desktop\MANGACOLOR\Manga-Flux\outputs\batch_test_run"
    OUTPUT_DIR = r"C:\Users\Nyx\Desktop\MANGACOLOR\Manga-Flux\outputs\tests"
    
    color_files = glob.glob(os.path.join(COLOR_DIR, "*_colorized.png"))
    
    for color_path in color_files:
        basename = os.path.basename(color_path)
        page_name = basename.replace("_colorized.png", ".jpg")
        orig_path = os.path.join(INPUT_DIR, page_name)
        
        if os.path.exists(orig_path):
            print(f"\n{'='*50}\nSemantic Scan: {basename}\n{'='*50}")
            analyze_semantics(orig_path, color_path, OUTPUT_DIR, detector)
        else:
            print(f"\nSkipping {basename}: Original {page_name} not found.")
            
    with open(os.path.join(OUTPUT_DIR, 'batch_semantics_yolo_log.txt'), 'w', encoding='utf-8') as f:
        f.write("Semantic Batch completed.")
