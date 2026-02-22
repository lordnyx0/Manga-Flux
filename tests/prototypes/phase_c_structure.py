import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def create_edge_map(image_gray, is_colorized=False):
    """
    Isolates manga lineart. Stronger handling for colorized images to avoid 
    detecting painted shading/gradients as edges.
    """
    # 1. Apply Edge-Preserving Blur (Bilateral Filter)
    # This aggressively blurs away soft textures (shading/colors) but keeps strong edges (ink lines) intact.
    if is_colorized:
        blurred = cv2.bilateralFilter(image_gray, d=15, sigmaColor=75, sigmaSpace=75)
    else:
        blurred = cv2.bilateralFilter(image_gray, d=9, sigmaColor=50, sigmaSpace=50)

    # 2. Adaptive Threshold for local line extraction
    # Block size needs to be larger for colorized to ignore local shading transitions
    block_size = 21 if is_colorized else 15
    c_off = 8 if is_colorized else 5
    
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_off)
    
    # 3. Clean up noise (salt and pepper from painting)
    kernel_clean = np.ones((3, 3), np.uint8)
    if is_colorized:
        edges = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
    else:
        edges = thresh
        
    # Thicken slightly for distance transform tolerance
    kernel_dil = np.ones((2, 2), np.uint8)
    return cv2.dilate(edges, kernel_dil, iterations=1)

def calculate_edge_anomaly_dt(edges_orig, edges_color, distance_tolerance=5):
    """
    Uses Distance Transform to find edges that are missing or hallucinated, 
    allowing for minor pixel shifts (tolerance) caused by diffusion.
    """
    # Ensure binary
    _, orig_bin = cv2.threshold(edges_orig, 127, 255, cv2.THRESH_BINARY)
    _, color_bin = cv2.threshold(edges_color, 127, 255, cv2.THRESH_BINARY)
    
    # Invert for distance transform (background must be 0)
    orig_inv = cv2.bitwise_not(orig_bin)
    color_inv = cv2.bitwise_not(color_bin)
    
    # Distance from each pixel to the nearest edge
    dist_orig = cv2.distanceTransform(orig_inv, cv2.DIST_L2, 3)
    dist_color = cv2.distanceTransform(color_inv, cv2.DIST_L2, 3)
    
    # Hallucinated: Edge exists in color_bin, but distance to nearest orig_bin edge > tolerance
    added_lines = np.zeros_like(orig_bin)
    added_lines[(color_bin == 255) & (dist_orig > distance_tolerance)] = 255
    
    # Missing: Edge exists in orig_bin, but distance to nearest color_bin edge > tolerance
    lost_lines = np.zeros_like(orig_bin)
    lost_lines[(orig_bin == 255) & (dist_color > distance_tolerance)] = 255
    
    # Clean up both maps (remove very tiny isolated noise)
    kernel = np.ones((3, 3), np.uint8)
    added_lines = cv2.morphologyEx(added_lines, cv2.MORPH_OPEN, kernel)
    lost_lines = cv2.morphologyEx(lost_lines, cv2.MORPH_OPEN, kernel)
    
    return added_lines, lost_lines

def extract_panels(img_gray, target_shape=None):
    """
    Dynamically finds Manga Panels based on standard box contours in the original B&W image.
    If target_shape (height, width) is provided, scales the coordinates to match the target.
    """
    orig_h, orig_w = img_gray.shape[:2]
    
    # Threshold for dark lines vs white background
    _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # To isolate panels without merging them, we apply a gentle morphological closing
    # to close panel borders without jumping across gutters.
    kernel = np.ones((3, 3), np.uint8)
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find external contours
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
        # Filter too small (noise) or too large (entire page bound)
        if area > img_area * 0.02 and area < img_area * 0.95:
            # Scale coordinates to target shape
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            scaled_w = int(w * scale_x)
            scaled_h = int(h * scale_y)
            panels.append((scaled_x, scaled_y, scaled_w, scaled_h))
            
    # Sort top to bottom, then left to right (rough manga reading order)
    panels.sort(key=lambda b: (b[1], b[0]))
    return panels

def calculate_regional_ssim(img1, img2, window_size=11):
    """
    Calculates a heatmap of SSIM to find texture/shading hallucinations.
    """
    # Compute SSIM map
    score, diff = ssim(img1, img2, full=True, win_size=window_size, data_range=255)
    
    # The diff image contains the actual image differences represented as floating point
    # values in the range [-1, 1]. Invert and normalize it to [0, 255].
    diff = (diff * 255).astype("uint8")
    
    # Lower SSIM = higher difference -> invert for mask where anomalies are white
    diff_inv = cv2.bitwise_not(diff)
    
    # SSIM drops drastically just because colors changed the grayscale luminance.
    # We need a very high threshold to only trigger on absolute structural obliteration.
    # 255 = completely inverted (max anomaly). We only trigger above 220.
    _, ssim_mask = cv2.threshold(diff_inv, 220, 255, cv2.THRESH_BINARY)
    
    # Clean up small noise
    kernel = np.ones((5, 5), np.uint8)
    ssim_mask_clean = cv2.morphologyEx(ssim_mask, cv2.MORPH_OPEN, kernel)
    
    return ssim_mask_clean

def analyze_structure(orig_path, color_path, output_dir):
    """
    Runs the Phase C Structural verification prototype on two images.
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(color_path)
    
    print(f"Loading Original: {orig_path}")
    print(f"Loading Colorized: {color_path}")
    
    # Load original as Grayscale
    img_orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        print("Error loading original image!")
        return
        
    # Load colorized and convert to Grayscale
    img_color_bgra = cv2.imread(color_path, cv2.IMREAD_COLOR)
    if img_color_bgra is None:
        print("Error loading color image!")
        return
    img_color_gray = cv2.cvtColor(img_color_bgra, cv2.COLOR_BGR2GRAY)
    
    # Resize them to match exactly if they don't (just in case)
    if img_orig.shape != img_color_gray.shape:
        print(f"Resizing orig {img_orig.shape} to match color {img_color_gray.shape}")
        img_orig = cv2.resize(img_orig, (img_color_gray.shape[1], img_color_gray.shape[0]))

    print("--- 1. Generating Edge Maps ---")
    edges_orig = create_edge_map(img_orig, is_colorized=False)
    # the colorized image goes through the colorized filter
    edges_color = create_edge_map(img_color_gray, is_colorized=True)
    
    cv2.imwrite(os.path.join(output_dir, f"edges_orig_{basename}"), edges_orig)
    cv2.imwrite(os.path.join(output_dir, f"edges_color_{basename}"), edges_color)
    
    print("--- 2. Calculating Edge Anomaly via Distance Transform ---")
    # Increased tolerance to 10 pixels to allow for line thickening/shifting caused by FLUX
    added_lines, lost_lines = calculate_edge_anomaly_dt(edges_orig, edges_color, distance_tolerance=10)
    
    print("--- 3. Calculating Regional SSIM Map ---")
    # Apply stronger blur before SSIM to ignore diffusion artifact noise and shading differences
    blur_orig = cv2.GaussianBlur(img_orig, (9, 9), 0)
    blur_color = cv2.GaussianBlur(img_color_gray, (9, 9), 0)
    # Larger window size to evaluate macro-structure, not micro-textures
    ssim_mask = calculate_regional_ssim(blur_orig, blur_color, window_size=21)
    
    print("--- 4. Dynamically Extracting Panels (with scaling) ---")
    # Extract panels from the UNRESIZED original image, but scale boxes to the colorized shape
    orig_unscaled = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    panels = extract_panels(orig_unscaled, target_shape=img_color_gray.shape)
    print(f"Detected {len(panels)} structural panels.")
    
    print("--- 5. Assembling Visualization Overlay ---")
    # Create an overlay image using the *colorized* image as the base
    overlay = img_color_bgra.copy()
    
    # Dilate edge masks slightly for visualization
    kernel_vis = np.ones((3, 3), np.uint8)
    added_lines_vis = cv2.dilate(added_lines, kernel_vis, iterations=1)
    lost_lines_vis = cv2.dilate(lost_lines, kernel_vis, iterations=1)
    
    # Filter SSIM mask drastically to avoid noise, require large contiguous blobs
    kernel_ssim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    ssim_mask_filtered = cv2.morphologyEx(ssim_mask, cv2.MORPH_OPEN, kernel_ssim)
    ssim_mask_vis = cv2.dilate(ssim_mask_filtered, kernel_vis, iterations=1)
    
    # Apply tints
    overlay[added_lines_vis == 255] = [0, 0, 255]     # RED: Hallucinated lines
    overlay[lost_lines_vis == 255] = [255, 0, 0]      # BLUE: Lost lines
    
    ssim_only = cv2.bitwise_and(ssim_mask_vis, cv2.bitwise_not(cv2.bitwise_or(added_lines_vis, lost_lines_vis)))
    overlay[ssim_only == 255] = [0, 255, 255]         # YELLOW: SSIM texture anomalies
    
    # Generate Trigger Mask
    combined_error_mask = cv2.bitwise_or(added_lines_vis, cv2.bitwise_or(lost_lines_vis, ssim_mask_vis))
    inpaint_kernel = np.ones((25, 25), np.uint8)
    final_inpaint_mask = cv2.dilate(combined_error_mask, inpaint_kernel, iterations=2)
    
    print(f"\n[PER-PANEL DIAGNOSTIC RESULTS]")
    for i, (x, y, w, h) in enumerate(panels):
        # Draw green bounding box around detected panel on overlay for visualization (thick line so it stays on top)
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 8)
        
        # Calculate failure purely within this panel
        panel_area = w * h
        panel_mask = final_inpaint_mask[y:y+h, x:x+w]
        failed_pixels = cv2.countNonZero(panel_mask)
        failure_ratio = (failed_pixels / panel_area) * 100
        
        print(f"Panel {i} ([{x}:{x+w}, {y}:{y+h}]): Affected Area: {failure_ratio:.2f}%")
        
        # Helper to draw readable text with background
        def draw_text_with_bg(img, text, pos, text_color, bg_color):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(img, (pos[0], pos[1] - text_h - baseline), (pos[0] + text_w, pos[1] + baseline), bg_color, -1)
            cv2.putText(img, text, pos, font, font_scale, text_color, thickness)
            
        if failure_ratio < 10.0:
            print(f"  -> Verdict: ACCEPTABLE. Minor issues, no Inpaint needed.")
            draw_text_with_bg(overlay, f"P{i}: OK ({failure_ratio:.1f}%)", (x+10, y+35), (0, 255, 0), (0, 0, 0))
            # Optional: mask out the trigger mask for acceptable panels so we don't inpaint them
            final_inpaint_mask[y:y+h, x:x+w] = 0
        elif failure_ratio < 30.0:
            print(f"  -> Verdict: MINOR ERRORS. Add this panel to Micro-Regional Inpaint queue.")
            draw_text_with_bg(overlay, f"P{i}: MICRO ({failure_ratio:.1f}%)", (x+10, y+35), (0, 255, 255), (0, 0, 0))
        else:
            print(f"  -> Verdict: CRITICAL FAILURE (Panel Swallowed/Destroyed). Re-roll entire panel bbox.")
            draw_text_with_bg(overlay, f"P{i}: CRITICAL ({failure_ratio:.1f}%)", (x+10, y+35), (0, 0, 255), (0, 0, 0))
            # If critical, force the entire panel to be white in the inpaint mask
            final_inpaint_mask[y:y+h, x:x+w] = 255
            
    # Save the final composite and mask to disk
    out_overlay_path = os.path.join(output_dir, f"analysis_overlay_{basename}")
    cv2.imwrite(out_overlay_path, overlay)
    cv2.imwrite(os.path.join(output_dir, f"inpaint_trigger_mask_{basename}"), final_inpaint_mask)
    print(f"Saved analysis overlay to: {out_overlay_path}")

if __name__ == "__main__":
    ORIG_PATH = r"C:\Users\Nyx\Desktop\MANGACOLOR\Manga-Flux\Doukutsu_Ou_kara_Hajimeru_Rakuen_Life\chapters\26_2\inputs\page_002.jpg"
    COLOR_PATH = r"C:\Users\Nyx\Desktop\MANGACOLOR\Manga-Flux\outputs\batch_test_run\page_002_colorized.png"
    OUTPUT_DIR = r"C:\Users\Nyx\Desktop\MANGACOLOR\Manga-Flux\outputs\tests"
    
    analyze_structure(ORIG_PATH, COLOR_PATH, OUTPUT_DIR)
