# Phase C: Correction and Consistency System (Post-Pass2)

Phase C operates right after generating the colorized image (Pass2) and aims to ensure that the colorization did not destroy the structure of the original art and that characters maintain chromatic consistency across panels.

Due to the limitations of diffusion models (FLUX), hallucinations (such as the invention of halos of light, inexact accessories) and color variations (sudden palette changes or shadow color bleeding) can occur. To solve this without relying purely on slow VLMs, we implemented a **quantitative check** in two steps: **Phase C (Structure)** and **Phase C.5 (Color)**.

---

## 1. Phase C — Structural Validation

The goal of Phase C is to ensure that the form and narrative of the original black-and-white (B&W) manga were faithfully preserved in the color version, without destructive hallucinations.

### 1.1 Structural Comparison Methodology (False Positive Prevention)
Directly comparing B&W versus Color — or even normalized pixels — often fails because of the diffusion nature, which softens traits, alters local contrast, and introduces noise. The right flow is:

**Layer 1: Edge Validation (IoU / Dice) — Core Check**
*   **Resolved problem:** Pure SSIM claims an error if FLUX simply "thickens" or "blurs" an original line, generating false alarms for smooth lighting bounds.
*   **Edge Detector (Lineart):** We extract a binary edge map (lineart) from the generated image and compare it with the original image lineart (Pass1).
*   **Edge-to-Edge Metrics:** We calculate the **Dice Coefficient** or **IoU (Intersection over Union)** on the line mask arrays.
*   **Role:** Accurately detects the appearance of **new objects** (structural hallucinations, accessories) and **deeply distorted shapes**, completely ignoring shading/fill brightness fluctuations.

**Layer 2: Analytical Heatmap — Auxiliary Check**
*   **SSIM Map (Regional SSIM) and Diff Mask:** SSIM acts as a similarity *Heatmap* rather than a global guard by sliding an 11x11 pixel window evaluation after grayscale conversion. Together with an Absolute Difference Map, this maps out the exact contour of the anomaly spotted in Layer 1.

**Layer 3: Perceptual and Semantic Similarity (Lightweight GPU)**
*   **LPIPS:** Computes perceptual distances to throw away false positives deriving from harmless changes.
*   **CLIP (OpenCLIP - Domain Normalized):** 
    *   *Problem resolved:* CLIP falls under domain gaps. Comparing B&W with Color drastically reduces cosine similarity, even if shapes are exactly the same.
    *   *Solution:* Before embedding, **we forcefully convert both images to grayscale RGB** (dropping saturation to zero and duplicating channels). By doing this, the latent space purely evaluates **shape semantics**, voiding chromatic biases. If similarity plummets, total decay happened (rendering a "random panel").

### 1.2 Inpaint Decision Logic: Micro-Regional vs Global
The response trickles down through instances (Crop by Bounding Box from YOLO), not the entire page.

1.  **Localized Error (Halos, FX, Tiny Accessories):**
    *   If metrics (Dice/SSIM Map) highlight divergence strictly in minuscule clusters (e.g., `< 15%` of panel area), we drop a dilated mask squarely on that spot.
    *   FLUX launches an Inpaint **strictly inside this micro-region**, saving render time and massive VRAM footprint.
2.  **Global or Destructive Error (Random/Disconnected Panel):**
    *   If the "Line IoU" deeply plummets across the crop surface, or if CLIP similarity entirely fails visual narrative (missing prompt cohesion), surgical inpainting is useless.
    *   The whole bounding box area gets submitted into a brand-new complete structural Inpaint pipeline.

---

## 2. Phase C.5 — Color Consistency (Color Profile Check)

Phase C ensured shape. Phase C.5 ensures a character doesn't swap from red hair to green hair standing between two panels. The comparison anchors entirely on a baseline reference (a manual character JSON file or a heuristic sampled from the first 3 panels).

### 2.1 Robust Color Evaluation (Noise Prevention)

Extracting the principal Hue from a generic image crop or histogram yields corrupted data under heavy shading or blown-out highlights. Worse, if 40% of the target hair lays in shadow, standard clustering (KMeans) will choose the dark gray shadow-tone as the "hair color".

**Chromatic Cleansing Checks:**

1.  **Brightness and Saturation Filtering:**
    *   Hue deviation math fails inside achromatic zones. Thus, we transfer over to the HSV space and **mask/drop all pixels where:**
        *   Saturation (S) < `0.20`
        *   Value/Brightness (V) isn't inside safety scales (`0.15` to `0.85`).
    *   **Result:** Only "middle" unadulterated color pixels dictate the core hue.

2.  **CIE-LAB Space Clustering (Highly Recommended):**
    *   The **LAB** space isolates the Luminance (`L`) channel and operates chroma values onto opponent poles (`a` and `b`). It maps human ocular perception extremely well.
    *   We execute KMeans clustering **checking solely the `a` and `b` channels**, completely disregarding the `L` axis.
    *   **Result:** Even if the character bounces between direct sunlight and shadow during the panel, the `a/b` parameters remain remarkably whole, absolutely asserting precise dominant color and blocking "gray cluster" takeover.

### 2.2 Correction Triggers
*   The scene extracted `a/b` centroid matches against the *Reference's* `a/b` centroid (which were evaluated filtering via the exact same principles inside Pass1/Pass2 runtime scopes).
*   If the Euclidean Distance spanning over the `(a, b)` bounds breaches tolerable thresholds, it fires a local character chromatic consistency failure.
*   **Correction:** We launch a **directed, color-focused Inpaint** prompt (e.g., "consistent deep blue hair, lacking colored lighting"), strictly bound inside that object's bounded segmentation.

---

## 3. Automated Flow

```text
Pass2 (Generation Concluded)
    │
    ▼
For each panel/character (YOLO Crops):
    │
    ├─► Phase C: Structure Respected?
    │    │ (Edge Extraction -> Dice/IoU Mask -> Grayscale CLIP Filter)
    │    └─ If No: 
    │        ├─ Minor Error (< 15% area) -> Micro-Regional Inpaint (Heatmap Mask)
    │        └─ Critical Error (> 30% area) -> Full BBox Inpaint
    │
    ├─► Phase C.5: Chromatic Identity Maintained?
    │    │ (Character Crop -> Bright Filter S>0.2, 0.15<V<0.85 -> LAB KMeans [A/B])
    │    └─ If No: Micro-Regional Inpaint (Prompt focused into fixing colors)
```

## 4. Architecture Optimization (Max: 12GB VRAM RTX3060)

The heartbeat defining these objective validations keeps up fast compute and mild processing times.
*   Fifty percent pipelines purely on **CPU** side via OpenCV (Lineart Extraction, Edge Maps, Dice Coef/SSIM, KMeans, LAB Space, HSV Filtering).
*   **GPU:** Bare minimum inference required for CLIP embeddings and responsive Inpaint execution — calculating fractions of crop regions over masks.
