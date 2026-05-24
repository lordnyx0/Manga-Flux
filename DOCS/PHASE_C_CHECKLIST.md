# Phase C Implementation Checklist

Status tracker for Phase C (Structure + Color Consistency).

## ✅ Done
- [x] Structural check module added (`core/correction/phase_c_structure.py`).
- [x] Per-panel verdict routing (`acceptable`, `micro_inpaint`, `critical_inpaint`).
- [x] Global line overlap metrics (`line_iou`, `line_dice`).
- [x] Regional anomaly support (edge anomaly + SSIM mask).
- [x] Batch CLI integration via `--phase-c-structure`.
- [x] Per-page Phase C JSON artifact (`page_XXX_phase_c_structure.json`).
- [x] Per-page inpaint mask image artifact (`page_XXX_phase_c_inpaint_mask.png`).
- [x] Optional debug overlay artifact (`page_XXX_phase_c_overlay.png`).
- [x] Batch summary now includes `phase_c_structure` and `phase_c_artifacts`.
- [x] Unit tests for identical case, destructive case, and serialization.

## ⏭️ Next (high priority)
- [x] Add optional debug overlay image artifact for manual QA (panel boxes + anomaly colors).
- [ ] Gate optional CLIP semantic check on availability (grayscale-normalized embeddings).
- [ ] Introduce per-class/panel thresholds from config (instead of fixed defaults).
- [ ] Wire `inpaint_mask` directly into Pass2 inpaint retry flow (micro and full-bbox reruns).

## 🔜 Phase C.5 (color consistency)
- [ ] Build reference color profile bootstrap (first N panels or manual character JSON).
- [ ] HSV filtering (S/V gates) and LAB clustering (`a/b`) for robust dominant color.
- [ ] Per-character drift scoring and local color-focused inpaint prompts.
- [ ] Persist per-character color history for chapter-level consistency.

## Validation checklist
- [x] `python -m py_compile run_two_pass_batch_local.py core/correction/phase_c_structure.py tests/test_phase_c_structure.py`
- [x] `PYTHONPATH=. pytest -q tests/test_phase_c_structure.py`
- [x] `python run_two_pass_batch_local.py --help`
