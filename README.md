# Manga-Flux: The First Specialist Manga Colorization Engine (v1.0)

Manga-Flux é um pipeline avançado de colorização headless via API projetado com uma arquitetura **Two-Pass**:

- **Pass1 (Análise)**: Identificação e segmentação estrutural (Balões de texto, Rostos, Corpos, Quadros) usando IA de Visão (YOLO Manga109).
- **Pass2 (Geração)**: Colorização de altíssima fidelidade utilizando a engine **FLUX.2-Klein**, guiado por metadados e injetando a Lineart diretamente no vetor de condicionamento textual (`ReferenceLatent`) para preservar 100% dos traços originais.

> **Status Atual:** (Fevereiro 2026) O projeto alcançou um marco histórico. O Pass1 e o Pass2 estão integrados e operacionais. A arquitetura **ReferenceLatent** provou-se capaz de colorir perfeitamente preservando lineart sem a quebra do Denoise tradicional no Flux.
>
> **Problemas Conhecidos (A Caminho da Fase C):** 
> * **Cores Excessivas/Hiper-detalhamento:** A geração atual resulta em cores muito vibrantes e com detalhes não previstos.
> * **Alucinações (Horror Vacui):** O modelo sofre para compor áreas de "vazio" (céu branco, fundos de balão mal lido), tendendo a desenhar objetos aleatórios onde deveria preservar o branco vazio. 
> * **Resolução de Conflitos:** A Fase C (Desacoplada) está projetada para usar Composição Passiva e Inpaint Regional (guiado pelo Pass1) para corrigir e mascarar essas alucinações.

## 🌟 Recursos Principais

- **FLUX Flow Matching Integration**: Usa técnicas de `EmptyLatent` + `ReferenceLatent` customizadas para saltar limitações de coloração img2img no FLUX.
- **Smart Resolution Compositing**: Escalonamento bidirecional garante que seu mangá em HD não seja reduzido por limites de GPU, e que a colorização seja upscaled graciosamente para a montagem dos balões.
- **Isolamento de Texto**: Balões de fala limpos via detecção cirúrgica.

## 📦 Dependências Necessárias

### Framework e Módulos Base
- `Python 3.10+`
- `onnxruntime-gpu` (ou `onnxruntime` para CPU) - Para inferência do YOLO no Pass1.
- `fastapi`, `uvicorn`, `requests`, `numpy`, `Pillow`

```bash
pip install fastapi uvicorn requests numpy Pillow onnxruntime-gpu
```

### ComfyUI Engine Backend
O Manga-Flux funciona interceptando uma instância local do **ComfyUI** via API. Você precisará:
1. ComfyUI instalado localmente (https://github.com/comfyanonymous/ComfyUI)
2. Custom Node GGUF (`ComfyUI-GGUF`): `git clone https://github.com/city96/ComfyUI-GGUF`
3. Custom Node ReferenceLatent (`ComfyUI_experiments`): `git clone https://github.com/comfyanonymous/ComfyUI_experiments`

## 🧠 Modelos Utilizados

### YOLO / Pass1 (Manga Analysis)
*   **Manga109 YOLO ONNX**: `data/models/manga109_yolo.onnx`
    *   *Link*: [A ser adicionado]

### ComfyUI / Pass2 (Diffusion Generation)
*   **UNet (Base Model):** `flux-2-klein-9b-Q4_K_M.gguf` -> Coloque em `ComfyUI/models/unet/`
    *   *Link*: [A ser adicionado]
*   **LoRA (Style Injector):** `colorMangaKlein_9B.safetensors` -> Coloque em `ComfyUI/models/loras/`
    *   *Link*: [A ser adicionado]
*   **CLIP (Text Encoder):** `qwen_3_8b_fp4mixed.safetensors` -> Coloque em `ComfyUI/models/clip/`
    *   *Link*: [A ser adicionado]
*   **VAE:** `flux2-vae.safetensors` -> Coloque em `ComfyUI/models/vae/`
    *   *Link*: [A ser adicionado]

---

## 🛠️ Executando o Pipeline

### Executar batch real local (Pass1->Pass2)

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

## 📄 Contratos e Arquitetura

- `metadata/README.md` (Contrato Pass1 -> Pass2)
- `DOCS/PHASE_B_IMPLEMENTATION.md` (Arquitetura Geração FLUX Flow-Matching)
- `DOCS/PHASE_C_CORRECTION.md` (Composição Passiva e Inpaint Ativo)
- `core/utils/meta_validator.py` (Validador P2)

## ▶️ Operação 

- `DOCS/OPERATION.md` (Guia operacional com comandos batch)

## 🔌 API e Extensão Chrome

- API local: `api/server.py`
- Companion extension: `extension/manga-flux-extension`
- Guia Completo: `DOCS/API_EXTENSION.md`
- Análise de Adaptação FAISS: `DOCS/FAISS_ADAPTACAO_MANGA_FLUX.md`
