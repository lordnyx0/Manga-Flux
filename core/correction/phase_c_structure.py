from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class PhaseCPanelResult:
    panel_index: int
    bbox: tuple[int, int, int, int]
    affected_ratio_pct: float
    verdict: str


def _to_gray_array(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return image


def _extract_lineart(gray: np.ndarray, is_colorized: bool) -> np.ndarray:
    _ = is_colorized
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    return cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)


def _line_metrics(edge_a: np.ndarray, edge_b: np.ndarray) -> dict[str, float]:
    a = edge_a > 0
    b = edge_b > 0

    intersection = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    a_sum = float(a.sum())
    b_sum = float(b.sum())

    iou = intersection / union if union > 0 else 1.0
    dice = (2.0 * intersection) / (a_sum + b_sum) if (a_sum + b_sum) > 0 else 1.0
    return {"line_iou": round(iou, 6), "line_dice": round(dice, 6)}


def _edge_anomaly_with_tolerance(
    edges_orig: np.ndarray,
    edges_color: np.ndarray,
    distance_tolerance: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, orig_bin = cv2.threshold(edges_orig, 127, 255, cv2.THRESH_BINARY)
    _, color_bin = cv2.threshold(edges_color, 127, 255, cv2.THRESH_BINARY)

    orig_inv = cv2.bitwise_not(orig_bin)
    color_inv = cv2.bitwise_not(color_bin)

    dist_orig = cv2.distanceTransform(orig_inv, cv2.DIST_L2, 3)
    dist_color = cv2.distanceTransform(color_inv, cv2.DIST_L2, 3)

    added = np.zeros_like(orig_bin)
    lost = np.zeros_like(orig_bin)
    added[(color_bin == 255) & (dist_orig > distance_tolerance)] = 255
    lost[(orig_bin == 255) & (dist_color > distance_tolerance)] = 255
    return added, lost


def _regional_ssim_mask(orig_gray: np.ndarray, color_gray: np.ndarray, threshold: int = 220) -> np.ndarray:
    img1 = cv2.GaussianBlur(orig_gray, (9, 9), 1.5).astype(np.float32)
    img2 = cv2.GaussianBlur(color_gray, (9, 9), 1.5).astype(np.float32)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    denominator = np.where(denominator == 0, 1e-8, denominator)

    ssim_map = np.clip(numerator / denominator, 0.0, 1.0)
    diff_inv = ((1.0 - ssim_map) * 255.0).astype(np.uint8)

    _, ssim_mask = cv2.threshold(diff_inv, threshold, 255, cv2.THRESH_BINARY)
    ssim_mask = cv2.morphologyEx(ssim_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ssim_mask = cv2.morphologyEx(
        ssim_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    )
    return ssim_mask


def _detect_panels(orig_gray: np.ndarray, target_shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    orig_h, orig_w = orig_gray.shape[:2]
    target_h, target_w = target_shape

    _, threshold = cv2.threshold(orig_gray, 200, 255, cv2.THRESH_BINARY_INV)
    connected = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = orig_h * orig_w
    scale_x = target_w / max(1, orig_w)
    scale_y = target_h / max(1, orig_h)

    panels: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > img_area * 0.02 and area < img_area * 0.95:
            panels.append((int(x * scale_x), int(y * scale_y), max(1, int(w * scale_x)), max(1, int(h * scale_y))))

    if not panels:
        panels = [(0, 0, target_w, target_h)]
    panels.sort(key=lambda b: (b[1], b[0]))
    return panels


def _make_overlay(
    base_image: np.ndarray,
    added_mask: np.ndarray,
    lost_mask: np.ndarray,
    ssim_mask: np.ndarray,
    panels: list[dict[str, Any]],
) -> np.ndarray:
    if len(base_image.shape) == 2:
        overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = base_image.copy()

    overlay[added_mask > 0] = [0, 0, 255]   # red
    overlay[lost_mask > 0] = [255, 0, 0]    # blue

    ssim_only = cv2.bitwise_and(ssim_mask, cv2.bitwise_not(cv2.bitwise_or(added_mask, lost_mask)))
    overlay[ssim_only > 0] = [0, 255, 255]  # yellow

    for panel in panels:
        x, y, w, h = panel["bbox"]
        verdict = panel["verdict"]
        color = (0, 255, 0)
        if verdict == "micro_inpaint":
            color = (0, 255, 255)
        elif verdict == "critical_inpaint":
            color = (0, 0, 255)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

    return overlay


def serialize_phase_c_report(report: dict[str, Any]) -> dict[str, Any]:
    keys_to_remove = {"inpaint_mask", "added_lines_mask", "lost_lines_mask", "ssim_mask"}
    return {k: v for k, v in report.items() if k not in keys_to_remove}


def save_phase_c_artifacts(
    report: dict[str, Any],
    output_dir: str | Path,
    page_num: int,
    overlay_base_image: str | Path | None = None,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"page_{page_num:03d}_phase_c_structure.json"
    mask_path = out / f"page_{page_num:03d}_phase_c_inpaint_mask.png"
    overlay_path = out / f"page_{page_num:03d}_phase_c_overlay.png"

    json_path.write_text(json.dumps(serialize_phase_c_report(report), ensure_ascii=False, indent=2), encoding="utf-8")

    artifacts: dict[str, str] = {"json": str(json_path), "inpaint_mask": str(mask_path)}
    inpaint_mask = report.get("inpaint_mask")
    if isinstance(inpaint_mask, np.ndarray):
        cv2.imwrite(str(mask_path), inpaint_mask)

    if overlay_base_image is not None:
        base = cv2.imread(str(overlay_base_image), cv2.IMREAD_COLOR)
        if base is not None:
            overlay = _make_overlay(
                base_image=base,
                added_mask=np.asarray(report.get("added_lines_mask", np.zeros(base.shape[:2], dtype=np.uint8))),
                lost_mask=np.asarray(report.get("lost_lines_mask", np.zeros(base.shape[:2], dtype=np.uint8))),
                ssim_mask=np.asarray(report.get("ssim_mask", np.zeros(base.shape[:2], dtype=np.uint8))),
                panels=list(report.get("panels", [])),
            )
            cv2.imwrite(str(overlay_path), overlay)
            artifacts["overlay"] = str(overlay_path)

    return artifacts


def run_phase_c_structure_check(
    original_path: str | Path,
    colorized_path: str | Path,
    micro_threshold_pct: float = 15.0,
    critical_threshold_pct: float = 30.0,
    distance_tolerance_px: int = 10,
) -> dict[str, Any]:
    original_gray = _to_gray_array(original_path)
    color_gray = _to_gray_array(colorized_path)

    if original_gray.shape != color_gray.shape:
        original_gray = cv2.resize(original_gray, (color_gray.shape[1], color_gray.shape[0]), interpolation=cv2.INTER_LINEAR)

    edges_original = _extract_lineart(original_gray, is_colorized=False)
    edges_colorized = _extract_lineart(color_gray, is_colorized=True)

    line_metrics = _line_metrics(edges_original, edges_colorized)
    added_lines, lost_lines = _edge_anomaly_with_tolerance(edges_original, edges_colorized, distance_tolerance=distance_tolerance_px)
    ssim_mask = _regional_ssim_mask(original_gray, color_gray)

    kernel_vis = np.ones((3, 3), np.uint8)
    added_vis = cv2.dilate(added_lines, kernel_vis, iterations=1)
    lost_vis = cv2.dilate(lost_lines, kernel_vis, iterations=1)
    ssim_vis = cv2.dilate(ssim_mask, kernel_vis, iterations=1)

    combined_error_mask = cv2.bitwise_or(added_vis, cv2.bitwise_or(lost_vis, ssim_vis))
    inpaint_mask = cv2.dilate(combined_error_mask, np.ones((25, 25), np.uint8), iterations=2)

    panels = _detect_panels(original_gray, target_shape=color_gray.shape)
    panel_results: list[PhaseCPanelResult] = []

    for i, (x, y, w, h) in enumerate(panels):
        panel_mask = inpaint_mask[y : y + h, x : x + w]
        panel_area = max(1, w * h)
        failed_pixels = int(cv2.countNonZero(panel_mask))
        failure_ratio = (failed_pixels / panel_area) * 100.0

        if failure_ratio < micro_threshold_pct:
            verdict = "acceptable"
            inpaint_mask[y : y + h, x : x + w] = 0
        elif failure_ratio < critical_threshold_pct:
            verdict = "micro_inpaint"
        else:
            verdict = "critical_inpaint"
            inpaint_mask[y : y + h, x : x + w] = 255

        panel_results.append(PhaseCPanelResult(i, (x, y, w, h), round(failure_ratio, 3), verdict))

    page_affected_ratio = round(float(cv2.countNonZero(inpaint_mask)) / max(1, inpaint_mask.size) * 100.0, 3)
    panels_payload = [
        {
            "panel_index": p.panel_index,
            "bbox": list(p.bbox),
            "affected_ratio_pct": p.affected_ratio_pct,
            "verdict": p.verdict,
        }
        for p in panel_results
    ]

    return {
        "page_shape": {"height": int(color_gray.shape[0]), "width": int(color_gray.shape[1])},
        "thresholds": {
            "micro_pct": micro_threshold_pct,
            "critical_pct": critical_threshold_pct,
            "distance_tolerance_px": distance_tolerance_px,
        },
        "line_metrics": line_metrics,
        "panel_count": len(panel_results),
        "panels": panels_payload,
        "page_affected_ratio_pct": page_affected_ratio,
        "has_structural_alert": any(p.verdict != "acceptable" for p in panel_results),
        "inpaint_mask": inpaint_mask,
        "added_lines_mask": added_vis,
        "lost_lines_mask": lost_vis,
        "ssim_mask": ssim_vis,
    }
