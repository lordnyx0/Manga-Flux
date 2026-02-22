# Manga-Flux: The First Specialist Manga Colorization Engine (v1.0)

Manga-Flux is an advanced headless colorization pipeline via API designed with a **Two-Pass** architecture:

- **Pass1 (Analysis)**: Identification and structural segmentation (Speech bubbles, Faces, Bodies, Panels) using Vision AI (YOLO Manga109).
- **Pass2 (Generation)**: Ultra-high fidelity colorization using the **FLUX.2-Klein** engine, guided by metadata and directly injecting Lineart into the textual conditioning vector (`ReferenceLatent`) to 100% preserve original traits.

> **Current Status:** (February 2026) The project has reached a historical milestone. Pass1 and Pass2 are integrated and operational. The **ReferenceLatent** architecture proved capable of perfect colorization while preserving lineart without breaking traditional Denoising in Flux.
>
> **Known Issues (Heading to Phase C):** 
> * **Excessive Colors / Hyper-detailing:** The current generation results in highly vibrant colors with unpredicted details.
> * **Hallucinations (Horror Vacui):** The model struggles to compose "empty" areas (white sky, poorly read bubble backgrounds), tending to draw random objects where it should preserve empty white. 
> * **Conflict Resolution:** Phase C (Decoupled) is designed to use Passive Compositing and Regional Inpainting (guided by Pass1) to correct and mask these hallucinations.

## 🌟 Key Features

- **FLUX Flow Matching Integration**: Uses custom `EmptyLatent` + `ReferenceLatent` techniques to bypass img2img colorization limits in FLUX.
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
1. ComfyUI installed locally (https://github.com/comfyanonymous/ComfyUI)
2. Custom Node GGUF (`ComfyUI-GGUF`): `git clone https://github.com/city96/ComfyUI-GGUF`
3. Custom Node ReferenceLatent (`ComfyUI_experiments`): `git clone https://github.com/comfyanonymous/ComfyUI_experiments`

## 🧠 Models Used

### YOLO / Pass1 (Manga Analysis)
*   **Manga109 YOLO ONNX**: `data/models/manga109_yolo.onnx`
    *   *Link*: [To be added]

### ComfyUI / Pass2 (Diffusion Generation)
*   **UNet (Base Model):** `flux-2-klein-9b-Q4_K_M.gguf` -> Place in `ComfyUI/models/unet/`
    *   *Link*: [To be added]
*   **LoRA (Style Injector):** `colorMangaKlein_9B.safetensors` -> Place in `ComfyUI/models/loras/`
    *   *Link*: [To be added]
*   **CLIP (Text Encoder):** `qwen_3_8b_fp4mixed.safetensors` -> Place in `ComfyUI/models/clip/`
    *   *Link*: [To be added]
*   **VAE:** `flux2-vae.safetensors` -> Place in `ComfyUI/models/vae/`
    *   *Link*: [To be added]

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
  --engine flux
```

## 📄 Contracts and Architecture

- `metadata/README.md` (Pass1 -> Pass2 Contract)
- `DOCS/PHASE_B_IMPLEMENTATION.md` (FLUX Flow-Matching Generation Architecture)
- `DOCS/PHASE_C_CORRECTION.md` (Passive Compositing and Active Inpainting)
- `core/utils/meta_validator.py` (P2 Validator)

## ▶️ Operation 

- `DOCS/OPERATION.md` (Operation guide with batch commands)

## 🔌 API and Chrome Extension

- Local API: `api/server.py`
- Companion extension: `extension/manga-flux-extension`
- Full Guide: `DOCS/API_EXTENSION.md`
- FAISS Adaptation Analysis: `DOCS/FAISS_ADAPTACAO_MANGA_FLUX.md`
