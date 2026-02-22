import cv2
import numpy as np
import os
import sys

# Ensure we can import core modules
sys.path.append(r'C:\Users\Nyx\Desktop\MANGACOLOR\Manga-Flux')
from core.detection.yolo_detector import YOLODetector

def get_yolo_panels(img, detector):
    """
    Extracts topological structure (panels) using YOLO Manga109.
    Returns a list of [x, y, w, h] bounding boxes.
    """
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    detections = detector.detect(img_color)
    panels = []
    for d in detections:
        if d.class_id == 2: # Frame
            x1, y1, x2, y2 = d.bbox
            panels.append([x1, y1, x2 - x1, y2 - y1])
            
    # Sort top-to-bottom, left-to-right
    panels.sort(key=lambda b: (b[1], b[0]))
    return panels

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

def calculate_edge_anomaly_dt(edges_orig, edges_color, distance_tolerance=4):
    dist_color = cv2.distanceTransform(cv2.bitwise_not(edges_color), cv2.DIST_L2, 3)
    lost_lines = np.zeros_like(edges_orig)
    lost_lines[(edges_orig == 255) & (dist_color > distance_tolerance)] = 255
    
    dist_orig = cv2.distanceTransform(cv2.bitwise_not(edges_orig), cv2.DIST_L2, 3)
    added_lines = np.zeros_like(edges_color)
    added_lines[(edges_color == 255) & (dist_orig > distance_tolerance)] = 255

    return added_lines, lost_lines

def analyze_topology(orig_path, color_path, output_dir, detector):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(color_path)
    
    img_orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    img_color_bgra = cv2.imread(color_path, cv2.IMREAD_COLOR)
    img_color_gray = cv2.cvtColor(img_color_bgra, cv2.COLOR_BGR2GRAY)
    
    orig_h, orig_w = img_orig.shape[:2]
    col_h, col_w = img_color_gray.shape[:2]
    
    scale_x = col_w / orig_w
    scale_y = col_h / orig_h

    print("--- 1. Extracting Independent Topological Graphs (YOLO Semantic Method) ---")
    orig_panels = get_yolo_panels(img_orig, detector)
    color_panels = get_yolo_panels(img_color_bgra, detector)
    
    print(f"Original Panels: {len(orig_panels)}")
    print(f"Colorized Panels: {len(color_panels)}")

    overlay = img_color_bgra.copy()
    
    print("\n--- 2. Topological Graph Comparison ---")
    
    if len(orig_panels) == 0:
        print("No panels found in original. Skipping.")
        return

    for i, orig_box in enumerate(orig_panels):
        scaled_orig_box = scale_box(orig_box, scale_x, scale_y)
        sx, sy, sw, sh = scaled_orig_box
        
        best_iou = 0.0
        best_match_idx = -1
        
        for j, color_box in enumerate(color_panels):
            iou = calculate_iou(scaled_orig_box, color_box)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = j
                
        cv2.rectangle(overlay, (sx, sy), (sx+sw, sy+sh), (255, 255, 255), 2, lineType=cv2.LINE_4)
                
        if best_iou < 0.30:
            verdict = "ABSORBED (CRITICAL)"
            color_c = (0, 0, 255) # Red
            print(f"Panel {i} {scaled_orig_box}: {verdict} (Max IoU: {best_iou:.2f})")
            
        elif best_iou < 0.60:
            verdict = "FUSED/DEFORMED (CRITICAL)"
            color_c = (0, 165, 255) # Orange
            print(f"Panel {i} {scaled_orig_box}: {verdict} (Matched Col_{best_match_idx} IoU: {best_iou:.2f})")
            cx, cy, cw, ch = color_panels[best_match_idx]
            cv2.rectangle(overlay, (cx, cy), (cx+cw, cy+ch), color_c, 4)
            
        else:
            verdict = "TOPOLOGY OK"
            color_c = (0, 255, 0) # Green
            print(f"Panel {i} {scaled_orig_box}: {verdict} (Matched Col_{best_match_idx} IoU: {best_iou:.2f})")
            
            # Draw Colorized Box in Green for Topological Success
            cx, cy, cw, ch = color_panels[best_match_idx]
            cv2.rectangle(overlay, (cx, cy), (cx+cw, cy+ch), color_c, 4)
            
            # --- MICRO-FAILURE VERIFICATION ---
            print("  -> Running Micro-Distance Validations...")
            
            # Non-Destructive Local Cropping
            crop_orig = img_orig[orig_box[1]:orig_box[1]+orig_box[3], orig_box[0]:orig_box[0]+orig_box[2]]
            crop_color = img_color_gray[cy:cy+ch, cx:cx+cw]
            
            if crop_orig.size == 0 or crop_color.size == 0:
                print("  -> ERROR: Invalid crop size.")
                continue

            # Resize Original Crop UP to Colorized Crop dimensions for 1:1 pixel comparison
            crop_orig_resized = cv2.resize(crop_orig, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
            
            # Bilateral filter color to kill shading. Canny original dynamically based on crop intensity.
            blur_c = cv2.bilateralFilter(crop_color, d=9, sigmaColor=75, sigmaSpace=75)
            edges_c = cv2.Canny(blur_c, 50, 150)
            
            blur_o = cv2.GaussianBlur(crop_orig_resized, (3, 3), 0)
            v = np.median(blur_o)
            lower = int(max(0, (1.0 - 0.33) * v))
            upper = int(min(255, (1.0 + 0.33) * v))
            edges_o = cv2.Canny(blur_o, lower, upper)
            
            kernel_dil = np.ones((2, 2), np.uint8)
            edges_o = cv2.dilate(edges_o, kernel_dil, iterations=1)
            edges_c = cv2.dilate(edges_c, kernel_dil, iterations=1)
            
            added_lines, lost_lines = calculate_edge_anomaly_dt(edges_o, edges_c, distance_tolerance=4)
            
            total_orig_edge_pixels = cv2.countNonZero(edges_o)
            lost_pixels = cv2.countNonZero(lost_lines)
            lost_ratio = (lost_pixels / max(1, total_orig_edge_pixels)) * 100
            
            print(f"  -> Original Panel Lost/Destroyed Lines: {lost_ratio:.2f}%")
            
            if lost_ratio > 15.0:
                print(f"  -> MICRO-FAILURE DETECTED: Lineart destruction exceeds 15% threshold.")
                verdict = "OK (BUT LOST MICRO-DETAILS)"
                color_c = (0, 255, 255) # Yellow warning
                cv2.rectangle(overlay, (cx, cy), (cx+cw, cy+ch), color_c, 4) 
                
            kernel_vis = np.ones((3, 3), np.uint8)
            lost_lines_vis = cv2.dilate(lost_lines, kernel_vis, iterations=1)
            
            overlay_crop = overlay[cy:cy+ch, cx:cx+cw]
            # Safety check before coloring the array
            if overlay_crop.shape[:2] == lost_lines_vis.shape[:2]:
                overlay_crop[lost_lines_vis == 255] = [255, 0, 0] # BLUE overlay for missing micro-structure
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"P{i}: {verdict}"
        (text_w, text_h), baseline = cv2.getTextSize(text, font, 0.8, 2)
        cv2.rectangle(overlay, (sx, sy - text_h - baseline), (sx + text_w, sy), (0,0,0), -1)
        cv2.putText(overlay, text, (sx, sy - baseline), font, 0.8, color_c, 2)

    out_path = os.path.join(output_dir, f"topology_overlay_{basename}")
    cv2.imwrite(out_path, overlay)
    print(f"\nSaved Topological Analysis to: {out_path}")

if __name__ == "__main__":
    import glob
    
    print("Loading YOLO Detector (Manga109)...")
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
            print(f"\n{'='*50}\nEvaluating: {basename}\n{'='*50}")
            analyze_topology(orig_path, color_path, OUTPUT_DIR, detector)
        else:
            print(f"\nSkipping {basename}: Original {page_name} not found.")

    with open(os.path.join(OUTPUT_DIR, 'batch_topology_yolo_log.txt'), 'w', encoding='utf-8') as f:
        f.write("Batch completed.")
