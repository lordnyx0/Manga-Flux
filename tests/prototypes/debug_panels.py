import cv2
import numpy as np
import os

def tune_panel_extraction(orig_path, output_path):
    img_orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        return

    # Invert so ink is 255
    _, thresh = cv2.threshold(img_orig, 230, 255, cv2.THRESH_BINARY_INV)

    # Find ALL contours of ink
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = []
    # Filter out tiny noise
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 5:
            rects.append([x, y, x+w, y+h])
            
    # Function to merge intersecting or very close rectangles
    def merge_rects(rectangles, distance_threshold=8):
        # distance_threshold should be LESS than the minimum gutter width (e.g., manga gutters are ~15-30px)
        # So we merge rects that are within 8px of each other
        merged = []
        for r in rectangles:
            x1, y1, x2, y2 = r
            r_expanded = [x1 - distance_threshold, y1 - distance_threshold, x2 + distance_threshold, y2 + distance_threshold]
            
            intersected_indices = []
            for i, existing in enumerate(merged):
                ex1, ey1, ex2, ey2 = existing
                # Check for intersection
                if not (r_expanded[2] < ex1 or r_expanded[0] > ex2 or r_expanded[3] < ey1 or r_expanded[1] > ey2):
                    intersected_indices.append(i)
                    
            if not intersected_indices:
                merged.append([x1, y1, x2, y2])
            else:
                # Merge all intersected
                min_x = x1
                min_y = y1
                max_x = x2
                max_y = y2
                for i in sorted(intersected_indices, reverse=True):
                    ex1, ey1, ex2, ey2 = merged.pop(i)
                    min_x = min(min_x, ex1)
                    min_y = min(min_y, ey1)
                    max_x = max(max_x, ex2)
                    max_y = max(max_y, ey2)
                merged.append([min_x, min_y, max_x, max_y])
        return merged

    # Iteratively merge until no more merges happen
    prev_len = len(rects)
    rects = merge_rects(rects)
    while len(rects) < prev_len:
        prev_len = len(rects)
        rects = merge_rects(rects)

    # Convert back to (x, y, w, h)
    panels = []
    img_area = img_orig.shape[0] * img_orig.shape[1]
    for r in rects:
        x1, y1, x2, y2 = r
        w = x2 - x1
        h = y2 - y1
        area = w * h
        # Only keep real panels (larger than 2% of page, smaller than 95%)
        if area > img_area * 0.02 and area < img_area * 0.95:
            panels.append((x1, y1, w, h))

    panels.sort(key=lambda b: (b[1], b[0]))
    
    orig_color = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2BGR)
    print(f"Detected {len(panels)} panels.")
    for i, (x, y, w, h) in enumerate(panels):
        print(f"Panel {i}: X:{x} Y:{y} W:{w} H:{h}")
        cv2.rectangle(orig_color, (x, y), (x+w, y+h), (0, 255, 0), 4)
        cv2.putText(orig_color, f"P{i}", (x+10, y+40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imwrite(output_path, orig_color)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    ORIG_PATH = r"C:\Users\Nyx\Desktop\MANGACOLOR\Manga-Flux\Doukutsu_Ou_kara_Hajimeru_Rakuen_Life\chapters\26_2\inputs\page_002.jpg"
    OUTPUT_DIR = r"C:\Users\Nyx\Desktop\MANGACOLOR\Manga-Flux\outputs\tests\panel_extraction_debug.png"
    tune_panel_extraction(ORIG_PATH, OUTPUT_DIR)
