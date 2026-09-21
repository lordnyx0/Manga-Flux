# Manga-Flux: The First Specialist Manga Colorization Engine (v2.0)

Manga-Flux is an advanced headless colorization pipeline via API designed with a **Two-Pass** architecture:

- **Pass1 (Analysis)**: Identification and structural segmentation (Speech bubbles, Faces, Bodies, Panels) using Vision AI (YOLO Manga109).
- **Pass2 (Generation)**: Ultra-high fidelity colorization using the **Qwen-Image-2.1-Uncensored-GGUF** engine via ComfyUI Image-Edit (`TextEncodeQwenImage21`, `cfg=1.0`), guided by metadata, chapter-wide cast resolution and a persistent character registry.

> **Current Status:** (September 2026) Qwen-Image-2.1 is the permanent Pass2 engine. Pass1 and Pass2 are integrated and operational; the legacy FLUX path remains available as fallback (`--engine flux`).
>
> **Known Issues & Updates:**
> * **Hallucinations (Horror Vacui):** The model tends to fill "empty" areas (white sky, bubble backgrounds) instead of preserving blank white. Mitigated via prompt (`preserve empty white backgrounds`) and Phase C compositing.
> * **Structure validation:** `StructureGuard` thresholds were calibrated for the FLUX/`ReferenceLatent` path and currently under-score Qwen outputs — recalibration in progress (see `DOCS/QWEN_MIGRATION.md`).
> * **Conflict Resolution:** Phase C (Decoupled) uses Passive Compositing and Regional Inpainting (guided by Pass1) to correct and mask hallucinations.

## 🌟 Key Features

- **Qwen-Image-2.1 Edit Integration**: Native multimodal edit (`image_1`=B&W page, `image_2`=style reference, `image_3..N`=character crops) instead of img2img tricks — no `ReferenceLatent`, no LoRA required.
- **Chapter-wide Cast Resolution**: A VLM sees all pages at once (set-of-marks + visual anchors) and assigns stable character IDs (`core/identity/cast_resolver.py` → `dramatis_personae.json`), with exact-coverage validation and retry.
- **Character Color Registry**: First-appearance colors are extracted back from the output and persisted per chapter, so newcomers keep the same hair/outfit on reappearance.
- **Smart Resolution Compositing**: Bidirectional scaling ensures your HD manga is not downsized by GPU limits, and colorization is gracefully upscaled for bubble assembly.
- **Text Isolation**: Clean speech bubbles via surgical detection.

## 📦 Required Dependencies

### Base Framework and Modules
- `Python 3.10+`
- `onnxruntime-gpu` (or `onnxruntime` for CPU) - For YOLO inference in Pass1.
- `fastapi`, `uvicorn`, `requests`, `numpy`, `Pillow`

```bash
pip install fastapi uvicorn requests numpy Pillow onnxruntime-gpu
```

### ComfyUI Engine Backend
Manga-Flux works by intercepting a local instance of **ComfyUI** via API. You will need:
1. ComfyUI installed locally, up to date (Qwen-Image 2.1 nodes required: https://github.com/comfyanonymous/ComfyUI)
2. Custom Node GGUF with Qwen-Image 2.1 support (`leejet` fork — the `city96` fork gives `Unknown model architecture!`): `git clone https://github.com/leejet/ComfyUI-GGUF`
3. Start with low-VRAM flags on 12GB GPUs: `python main.py --lowvram`

### Local VLM (Cast Resolution)
Chapter-wide character tracking runs through an OpenAI-compatible server (llama.cpp `llama-server` or LM Studio), selected via env `VLM_MODEL=gemma|qwen3.5` (see `config/settings.py`: `GEMMA_*` / `QWEN35_*` paths, `GEMMA_CTX`, `REASONING_BUDGET`).

## 🧠 Models Used

### YOLO / Pass1 (Manga Analysis)
*   **Manga109 YOLO ONNX**: `data/models/manga109_yolo.onnx`
    *   *Link*: [https://huggingface.co/deepghs/manga109_yolo]

### ComfyUI / Pass2 (Qwen-Image-2.1 Edit)
*   **Diffusion (GGUF):** `qwen-image-2.1-Q4_K_M.gguf` -> Place in `ComfyUI/models/diffusion_models/` (Q4_K_M recommended; never `Q8_0` — tensor shape bug)
    *   *Link*: [https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF]
*   **Text Encoder:** `qwen3vl_8b_int8_convrot.safetensors` (`CLIPLoader type=qwen_image`) -> Place in `ComfyUI/models/text_encoders/` (runs in system RAM, ~9GB)
*   **VAE:** `qwen_image_2.1_vae_bf16.safetensors` -> Place in `ComfyUI/models/vae/`
*   **Legacy FLUX path** (`--engine flux`): `flux-2-klein-4b-Q4_K_M.gguf` + `qwen_3_4b_fp4_flux2.safetensors` + `flux2-vae.safetensors` (+ `ComfyUI_experiments` for `ReferenceLatent`)

---

## 🛠️ Running the Pipeline

### Run realistic local batch (Pass1->Pass2)

```bash
python run_two_pass_batch_local.py \
  --input-dir data/pages_bw \
  --style-reference data/style_ref.png \
  --metadata-output outputs/batch_test_run/metadata \
  --masks-output outputs/batch_test_run/masks \
  --pass2-output outputs/batch_test_run \
  --chapter-id chapter_test \
  --engine qwen \
  --phase-c-structure
```

When `--phase-c-structure` is enabled, each page emits `page_XXX_phase_c_structure.json`, `page_XXX_phase_c_inpaint_mask.png`, and `page_XXX_phase_c_overlay.png` with panel-level structural verdicts and inpaint routing/QA artifacts for Phase C correction.
The report includes lineart overlap metrics (`line_iou`, `line_dice`) and regional anomaly routing (`acceptable`, `micro_inpaint`, `critical_inpaint`).

Qwen-only options via `--pass2-option key=value`: `ref_crops_dir=` (manual reference crops), `faiss_threshold=0.35`, `max_ref_crops=8`, `use_gemma_prompts=1` (legacy modular prompt), `steps`, `cfg`, `resolution`.

## 📄 Contracts and Architecture

- `metadata/README.md` (Pass1 -> Pass2 Contract)
- `DOCS/QWEN_MIGRATION.md` (Qwen engine migration log + cast-resolver experiments)
- `docs/PHASE_B_IMPLEMENTATION.md` (legacy FLUX Flow-Matching Generation Architecture)
- `docs/PHASE_C_CORRECTION.md` (Passive Compositing and Active Inpainting)
- `DOCS/PHASE_C_CHECKLIST.md` (Implementation checklist and next milestones)
- `core/utils/meta_validator.py` (P2 Validator)

## ▶️ Operation

- `docs/OPERATION.md` (Operation guide with batch commands)

## 🔌 API and Chrome Extension

- Local API: `api/server.py`
- Companion extension: `extension/manga-flux-extension`
- Full Guide: `docs/API_EXTENSION.md`
- FAISS Adaptation Analysis: `docs/FAISS_ADAPTATION_MANGA_FLUX.md`
