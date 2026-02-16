# Manga-Flux: The First Specialist Manga Colorization Engine (v1.0)

Manga-Flux is a high-performance image colorization pipeline designed for manga and doujinshi. It leverages the state-of-the-art **Flux.1-Dev** architecture and a specialized LoRA trained on character-consistent triplets to deliver professional-grade results.

## 🌟 Key Features

- **Flux Specialist**: Built natively for Flux.1-Dev with specialist LoRA integration.
- **Global Coherence**: Optimized for single-pass generation to maintain color consistency across the entire page.
- **VRAM Optimized**: Native support for **NF4 (4-bit)** quantization, requiring only 12GB VRAM.
- **Text Preservation**: Intelligent YOLO-based masking ensures text and speech bubbles remain pristine.
- **Two-Pass Protocol**: Decoupled analysis (Pass 1) and generation (Pass 2) for maximum flexibility.

## 🛠️ Getting Started

### 1. Requirements
Ensure you have Python 3.10+ and a CUDA-capable GPU (12GB+ VRAM recommended).
```bash
pip install torch diffusers transformers accelerate bitsandbytes peft pyyaml
```

### 2. Model Setup
Manga-Flux requires the Flux.1-Dev weights and the specialized Manga Colorizer LoRA.
- Update `configs/flux.yaml` with sua LoRA local.

### 3. Execution
```bash
python run_pass2_local.py --meta path/to/page.meta.json --engine flux
```
