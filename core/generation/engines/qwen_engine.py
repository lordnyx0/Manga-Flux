from __future__ import annotations

"""
Qwen-Image-2.1 Image-Edit engine (Manga-Flux Pass2, experimental).

Diferenças arquiteturais em relação ao FluxEngine (FLUX.2-Klein):
- Sem ReferenceLatent / EmptyFlux2LatentImage / Flux2Scheduler / LoRA colorManga.
- Condicionamento via TextEncodeQwenImage21 (multimodal nativo):
    image_1 = página P&B (alvo da edição)
    image_2 = style_ref / capa (referência visual de cor)
    image_3..N = crops de personagens (FAISS ou --ref-crops manual)
  referenciados no prompt como <image1>, <image2>, ...
- KSampler com cfg=1.0 (caminho oficial Qwen), denoise=1.0 (edição),
  sampler=euler, scheduler=simple. `strength` do caller é ignorado
  por incompatibilidade semântica (denoise img2img vs edição).
- Latent do KSampler vem da saída `latent` do TextEncode (com
  ComfySwitch para custom_size, como no template oficial), nunca de
  VAEEncode direto.
- GGUF requer fork leejet/ComfyUI-GGUF (ModelQwenImage). Q8_0 é
  incompatível (shape mismatch [136] vs [128]) — usar Q4_K_M/Q5_K_M/Q6_K.

Nenhuma inferência acontece aqui sem ComfyUI acessível; a construção
do workflow JSON (_build_comfyui_workflow_json) é pura e testável em CPU.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image

from core.generation.interfaces import ColorizationEngine

# Limite do template oficial: image_1..image_10 no TextEncodeQwenImage21.
# image_1 = P&B, image_2 = style_ref, sobram 8 slots para crops.
MAX_REF_CROPS = 8

DEFAULT_UNET = "qwen-image-2.1-Q4_K_M.gguf"
DEFAULT_CLIP = "qwen3vl_8b_int8_convrot.safetensors"
DEFAULT_VAE = "qwen_image_2.1_vae_bf16.safetensors"


class QwenEngine(ColorizationEngine):
    """Cliente headless ComfyUI para colorização via Qwen-Image-2.1 Edit."""

    def __init__(self, comfy_host: str = "127.0.0.1", comfy_port: int = 8188):
        self.comfy_url = f"http://{comfy_host}:{comfy_port}"

    # ── API pública (mesmo contrato do FluxEngine) ──────────────────────

    def generate(self, payload: dict, seed: int, strength: float = 1.0, options: dict = None):
        options = dict(options or {})
        prompt = payload.get("qwen_prompt") or payload.get("prompt") or "manga panel"
        base_image_path = payload.get("base_image_path")

        if not base_image_path or not os.path.exists(base_image_path):
            raise FileNotFoundError(f"Source image not found: {base_image_path}")

        if strength is not None and abs(float(strength) - 1.0) > 1e-6:
            print(
                f"[QwenEngine] Aviso: strength={strength} ignorado "
                "(edição Qwen usa denoise=1.0 fixo)."
            )

        uploaded_bw = self._upload_image_to_comfy(base_image_path)

        style_image = payload.get("style_image")
        uploaded_style = self._upload_style_image_to_comfy(style_image)

        ref_crops = self._resolve_ref_crops(payload, options)
        uploaded_crops = [self._upload_crop_to_comfy(c, i) for i, c in enumerate(ref_crops)]
        uploaded_crops = [n for n in uploaded_crops if n]

        workflow = self._build_comfyui_workflow_json(
            prompt=prompt,
            bw_image_name=uploaded_bw,
            style_image_name=uploaded_style,
            ref_crop_names=uploaded_crops,
            seed=seed,
            options=options,
        )

        return self._submit_workflow(workflow, base_image_path)

    def unload(self) -> None:
        """ComfyUI gerencia a própria VRAM via RPC; nada a liberar aqui."""
        pass

    # ── Resolução de crops de referência ───────────────────────────────

    def _resolve_ref_crops(self, payload: dict, options: dict) -> list:
        """Retorna até MAX_REF_CROPS crops (PIL | path), com override manual.

        Ordem de precedência:
        1. options["ref_crops_dir"]: diretório com *.png ordenados (teste A/B manual).
        2. payload["ref_crops"]: lista montada pelo orchestrator via FAISS.
        """
        ref_dir = options.get("ref_crops_dir")
        if ref_dir:
            p = Path(ref_dir)
            if p.is_dir():
                files = sorted([x for x in p.iterdir() if x.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
                if files:
                    print(f"[QwenEngine] Usando {len(files)} crops manuais de {p}")
                    return [str(f) for f in files[:MAX_REF_CROPS]]
                print(f"[QwenEngine] Aviso: ref_crops_dir vazio: {p}. Caindo para payload.")
            else:
                print(f"[QwenEngine] Aviso: ref_crops_dir inexistente: {p}. Caindo para payload.")

        crops = payload.get("ref_crops") or []
        if len(crops) > MAX_REF_CROPS:
            print(f"[QwenEngine] Truncando ref_crops {len(crops)} -> {MAX_REF_CROPS} (limite do nó).")
            crops = crops[:MAX_REF_CROPS]
        return list(crops)

    # ── Construção do workflow (pura, sem I/O de rede) ─────────────────

    def _build_comfyui_workflow_json(
        self,
        prompt: str,
        bw_image_name: str,
        style_image_name: str | None,
        ref_crop_names: list[str],
        seed: int,
        options: dict,
    ) -> dict:
        """Monta o workflow API Qwen-Image-2.1 Image-Edit.

        IDs estáveis (SaveImage="19" para reaproveitar o polling):
          1 LoadImage P&B, 2 Scale, 3 UnetLoaderGGUF, 4 CLIPLoader
          (qwen_image), 5 VAELoader, 6 TextEncodeQwenImage21,
          7 QwenImage21Cache, 8 KSampler, 9 EmptyLatentImage,
          10 ComfySwitchNode, 18 VAEDecode, 19 SaveImage,
          20/21 LoadImage+Scale da style_ref,
          30+i / 31+i LoadImage+Scale de cada crop extra.
        """
        opts = options or {}
        unet_name = opts.get("unet_name", DEFAULT_UNET)
        clip_name = opts.get("clip_name", DEFAULT_CLIP)
        vae_name = opts.get("vae_name", DEFAULT_VAE)
        steps = int(opts.get("steps", 25))
        cfg = float(opts.get("cfg", 1.0))
        if abs(cfg - 1.0) > 1e-6:
            print(f"[QwenEngine] Aviso: cfg={cfg} != 1.0 foge do caminho oficial Qwen.")
        sampler = opts.get("sampler_name", "euler")
        scheduler = opts.get("scheduler", "simple")
        resolution = int(opts.get("resolution", 1024))
        custom_size = bool(opts.get("custom_size", False))
        width = int(opts.get("width", 1024))
        height = int(opts.get("height", 1024))

        workflow: dict = {
            "1": {"class_type": "LoadImage", "inputs": {"image": bw_image_name}},
            "2": {
                "class_type": "ImageScaleToTotalPixels",
                "inputs": {
                    "image": ["1", 0],
                    "upscale_method": "lanczos",
                    "megapixels": 1.0,
                    "resolution_steps": 32,
                },
            },
            "3": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet_name}},
            "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "qwen_image"}},
            "5": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
            "6": {
                "class_type": "TextEncodeQwenImage21",
                "inputs": {
                    "prompt": prompt,
                    "negative_prompt": "",
                    "resolution": resolution,
                    "clip": ["4", 0],
                    "vae": ["5", 0],
                    "images.image_1": ["2", 0],
                },
            },
            "7": {
                "class_type": "QwenImage21Cache",
                "inputs": {"model": ["3", 0], "device": "auto", "dtype": "default"},
            },
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["7", 0],
                    "positive": ["6", 0],
                    "negative": ["6", 1],
                    "latent_image": ["10", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": 1.0,
                },
            },
            "9": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
            "10": {
                "class_type": "ComfySwitchNode",
                "inputs": {
                    "on_false": ["6", 2],
                    "on_true": ["9", 0],
                    "switch": custom_size,
                },
            },
            "18": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["5", 0]}},
            "19": {
                "class_type": "SaveImage",
                "inputs": {"images": ["18", 0], "filename_prefix": "manga_qwen_edit"},
            },
        }

        text_encode_inputs = workflow["6"]["inputs"]
        next_id = 30

        if style_image_name:
            workflow["20"] = {"class_type": "LoadImage", "inputs": {"image": style_image_name}}
            workflow["21"] = {
                "class_type": "ImageScaleToTotalPixels",
                "inputs": {
                    "image": ["20", 0],
                    "upscale_method": "lanczos",
                    "megapixels": 1.0,
                    "resolution_steps": 32,
                },
            }
            text_encode_inputs["images.image_2"] = ["21", 0]

        for i, crop_name in enumerate(ref_crop_names or []):
            slot = i + 3  # image_3 .. image_10
            load_id, scale_id = str(next_id), str(next_id + 1)
            workflow[load_id] = {"class_type": "LoadImage", "inputs": {"image": crop_name}}
            workflow[scale_id] = {
                "class_type": "ImageScaleToTotalPixels",
                "inputs": {
                    "image": [load_id, 0],
                    "upscale_method": "lanczos",
                    "megapixels": 1.0,
                    "resolution_steps": 32,
                },
            }
            text_encode_inputs[f"images.image_{slot}"] = [scale_id, 0]
            next_id += 2

        return workflow

    # ── Transporte ComfyUI (igual ao FluxEngine, SaveImage="19") ────────

    def _submit_workflow(self, workflow: dict, base_image_path: str):
        start_time = time.time()
        client_id = str(uuid.uuid4())
        headers = {"Content-Type": "application/json", "User-Agent": "Manga-Flux-Client/1.0"}
        req_data = json.dumps({"prompt": workflow, "client_id": client_id}).encode("utf-8")
        req = urllib.request.Request(f"{self.comfy_url}/prompt", data=req_data, headers=headers)

        try:
            with urllib.request.urlopen(req) as response:
                prompt_id = json.loads(response.read()).get("prompt_id")
                print(f"Queued ComfyUI Payload (qwen): {prompt_id}")
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="ignore")
            print(f"ComfyUI Request Failed (HTTP {he.code}): {err_body}")
            return Image.open(base_image_path).convert("RGB"), {
                "duration_ms": int((time.time() - start_time) * 1000),
                "status": "failed",
                "error": err_body,
                "engine_backend": "comfyui_gguf_qwen",
            }
        except Exception as e:
            print(f"ComfyUI Request Failed: {e}")
            return Image.open(base_image_path).convert("RGB"), {
                "duration_ms": int((time.time() - start_time) * 1000),
                "status": "failed",
                "error": str(e),
                "engine_backend": "comfyui_gguf_qwen",
            }

        print(f"Waiting for ComfyUI generation (Prompt ID: {prompt_id})...")
        output_url = None
        while True:
            try:
                req_h = urllib.request.Request(f"{self.comfy_url}/history/{prompt_id}", headers=headers)
                with urllib.request.urlopen(req_h) as hr:
                    history = json.loads(hr.read())
                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        if "19" in outputs and "images" in outputs["19"] and outputs["19"]["images"]:
                            filename = outputs["19"]["images"][0]["filename"]
                            output_url = f"{self.comfy_url}/view?filename={urllib.parse.quote(filename)}"
                        break
            except Exception as e:
                print(f"Error polling ComfyUI: {e}")
                break
            time.sleep(2)

        if output_url:
            try:
                req_img = urllib.request.Request(output_url, headers=headers)
                with urllib.request.urlopen(req_img) as ir:
                    result_image = Image.open(BytesIO(ir.read())).convert("RGB")
                    print("Generation downloaded successfully!")
            except Exception as e:
                print(f"Failed to download generated image: {e}")
                result_image = Image.open(base_image_path).convert("RGB")
        else:
            print("Failed to retrieve generation output from ComfyUI.")
            result_image = Image.open(base_image_path).convert("RGB")

        run_stats = {
            "duration_ms": int((time.time() - start_time) * 1000),
            "vram_peak_mb": 0,
            "engine_backend": "comfyui_gguf_qwen",
        }
        return result_image, run_stats

    # ── Uploads ─────────────────────────────────────────────────────────

    def _upload_image_to_comfy(self, local_path: str) -> str:
        with open(local_path, "rb") as f:
            return self._post_image_bytes(f.read(), os.path.basename(local_path), uuid.uuid4().hex)

    def _upload_style_image_to_comfy(self, style_image) -> str | None:
        if style_image is None:
            return None
        buf = BytesIO()
        if isinstance(style_image, (str, Path)):
            img = Image.open(style_image).convert("RGB")
        else:
            img = style_image.convert("RGB")
        img.save(buf, format="PNG")
        return self._post_image_bytes(buf.getvalue(), f"style_ref_{uuid.uuid4().hex[:8]}.png", uuid.uuid4().hex)

    def _upload_crop_to_comfy(self, crop, index: int) -> str | None:
        try:
            buf = BytesIO()
            if isinstance(crop, (str, Path)):
                img = Image.open(crop).convert("RGB")
            else:
                img = crop.convert("RGB")
            img.save(buf, format="PNG")
            return self._post_image_bytes(
                buf.getvalue(), f"ref_crop_{index}_{uuid.uuid4().hex[:8]}.png", uuid.uuid4().hex
            )
        except Exception as e:
            print(f"[QwenEngine] Falha no upload do crop {index}: {e}")
            return None

    def _post_image_bytes(self, file_data: bytes, filename: str, boundary: str) -> str:
        data = [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="image"; filename="{filename}"'.encode(),
            b"Content-Type: application/octet-stream",
            b"",
            file_data,
            f"--{boundary}--".encode(),
            b"",
        ]
        body = b"\r\n".join(data)
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "User-Agent": "Manga-Flux-Client/1.0",
        }
        req = urllib.request.Request(f"{self.comfy_url}/upload/image", data=body, headers=headers)
        try:
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read()).get("name", filename)
        except Exception as e:
            print(f"Failed to upload image to ComfyUI API: {e}")
            return filename
