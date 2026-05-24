from __future__ import annotations
import os
import time
import json
import urllib.request
import urllib.parse
from pathlib import Path
from PIL import Image

from core.generation.interfaces import ColorizationEngine
from config.settings import FLUX_MODEL_PATH

class FluxEngine(ColorizationEngine):
    """
    Motor real do Flux 2 / Manga-Flux (Fase B).
    Atua como um cliente "Headless ComfyUI Wrapper" visando usar o arquivo GGUF 
    quantizado para economia brutal de VRAM suportada pela comunidade.
    """

    def __init__(self, comfy_host: str = "127.0.0.1", comfy_port: int = 8188):
        self.comfy_url = f"http://{comfy_host}:{comfy_port}"

    def generate(self, payload: dict, seed: int, strength: float = 1.0, options: dict = None) -> tuple[Image.Image, dict]:
        """
        Gera a imagem acionando uma instância local do ComfyUI via Workflow API.

        Workflow correto do Klein 4B:
          - style_ref (colorida) → VAEEncode → ReferenceLatent  (referência visual de cor)
          - bw_page   (P&B)      → VAEEncode → latent_image      (o que será colorizado)
          - KSampler guiado pelo ReferenceLatent + prompt textual
        """
        prompt = payload.get("prompt", "manga panel")
        base_image_path = payload.get("base_image_path")

        if not base_image_path or not os.path.exists(base_image_path):
            raise FileNotFoundError(f"Source image not found: {base_image_path}")

        # Upload da página P&B (será o latent_image do KSampler)
        uploaded_bw_name = self._upload_image_to_comfy(base_image_path)

        # Upload da style_ref colorida (alimenta o ReferenceLatent)
        style_image = payload.get("style_image")
        uploaded_style_name = self._upload_style_image_to_comfy(style_image)

        # Monta o workflow com os dois encodings corretamente separados
        comfy_workflow = self._build_comfyui_workflow_json(
            prompt=prompt,
            bw_image_name=uploaded_bw_name,
            style_image_name=uploaded_style_name,
            seed=seed,
            strength=strength,
            options=options
        )
        
        start_time = time.time()
        
        # Execute workflow against ComfyUI
        import uuid
        client_id = str(uuid.uuid4())
        
        req_data = json.dumps({
            "prompt": comfy_workflow,
            "client_id": client_id
        }).encode('utf-8')
        
        # Added User-Agent to bypass potential 403 errors from the ComfyUI server
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Manga-Flux-Client/1.0'
        }
        req = urllib.request.Request(f"{self.comfy_url}/prompt", data=req_data, headers=headers)
        
        try:
            with urllib.request.urlopen(req) as response:
                resp_data = json.loads(response.read())
                prompt_id = resp_data.get("prompt_id")
                print(f"Queued ComfyUI Payload: {prompt_id}")
        except urllib.error.HTTPError as he:
            err_body = he.read().decode('utf-8', errors='ignore')
            print(f"ComfyUI Request Failed (HTTP {he.code}): {err_body}")
            run_stats = {"duration_ms": int((time.time() - start_time) * 1000), "status": "failed", "error": err_body}
            return Image.open(base_image_path).convert("RGB"), run_stats
        except Exception as e:
            print(f"ComfyUI Request Failed: {e}")
            run_stats = {"duration_ms": int((time.time() - start_time) * 1000), "status": "failed", "error": str(e)}
            return Image.open(base_image_path).convert("RGB"), run_stats
            
        print(f"Waiting for ComfyUI generation (Prompt ID: {prompt_id})...")
        
        # Polling Loop
        output_image_path = None
        while True:
            try:
                # Need the same user agent here too
                req_history = urllib.request.Request(f"{self.comfy_url}/history/{prompt_id}", headers=headers)
                with urllib.request.urlopen(req_history) as history_response:
                    history = json.loads(history_response.read())
                    if prompt_id in history:
                        # Generation finished!
                        outputs = history[prompt_id].get("outputs", {})
                        # Node 19 is our SaveImage node
                        if "19" in outputs and "images" in outputs["19"]:
                            images_data = outputs["19"]["images"]
                            if images_data:
                                filename = images_data[0]["filename"]
                                # ComfyUI serves generated images at /view?filename=...
                                output_image_path = f"{self.comfy_url}/view?filename={urllib.parse.quote(filename)}"
                        break
            except Exception as e:
                print(f"Error polling ComfyUI: {e}")
                break
                
            time.sleep(2) # Poll every 2 seconds
            
        if output_image_path:
            # Download the resulting image into memory
            try:
                req_img = urllib.request.Request(output_image_path, headers=headers)
                with urllib.request.urlopen(req_img) as img_resp:
                    from io import BytesIO
                    result_image = Image.open(BytesIO(img_resp.read())).convert("RGB")
                    print("Generation downloaded successfully!")
            except Exception as e:
                print(f"Failed to download generated image: {e}")
                result_image = Image.open(base_image_path).convert("RGB")
        else:
            print("Failed to retrieve generation output from ComfyUI.")
            result_image = Image.open(base_image_path).convert("RGB")
        
        end_time = time.time()
        duration_ms = int((end_time - start_time) * 1000)
        
        # We can't query VRAM from another process easily without extensions, stub to 0
        run_stats = {
            "duration_ms": duration_ms,
            "vram_peak_mb": 0,
            "engine_backend": "comfyui_gguf"
        }
        
        return result_image, run_stats

    def _upload_image_to_comfy(self, local_path: str) -> str:
        """
        Faz o POST de um arquivo de imagem para o ComfyUI local via multipart form-data
        e retorna o nome registrado no servidor.
        """
        import uuid
        boundary = uuid.uuid4().hex
        filename = os.path.basename(local_path)

        with open(local_path, "rb") as f:
            file_data = f.read()

        return self._post_image_bytes(file_data, filename, boundary)

    def _upload_style_image_to_comfy(self, style_image) -> str | None:
        """
        Faz o upload da imagem de referência colorida (PIL Image ou None).
        Salva em memória como PNG e envia ao ComfyUI.
        Retorna o nome registrado, ou None se não houver referência.
        """
        if style_image is None:
            return None

        import uuid
        from io import BytesIO

        boundary = uuid.uuid4().hex
        filename = f"style_ref_{uuid.uuid4().hex[:8]}.png"

        buf = BytesIO()
        # Aceita PIL Image ou path string/Path
        if isinstance(style_image, (str, Path)):
            img = Image.open(style_image).convert("RGB")
        else:
            img = style_image.convert("RGB")
        img.save(buf, format="PNG")
        file_data = buf.getvalue()

        return self._post_image_bytes(file_data, filename, boundary)

    def _post_image_bytes(self, file_data: bytes, filename: str, boundary: str) -> str:
        """Helper: faz o POST multipart de bytes de imagem ao ComfyUI."""
        data = []
        data.append(f'--{boundary}'.encode('utf-8'))
        data.append(f'Content-Disposition: form-data; name="image"; filename="{filename}"'.encode('utf-8'))
        data.append(b'Content-Type: application/octet-stream')
        data.append(b'')
        data.append(file_data)
        data.append(f'--{boundary}--'.encode('utf-8'))
        data.append(b'')
        body = b'\r\n'.join(data)

        headers = {
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'Content-Length': str(len(body)),
            'User-Agent': 'Manga-Flux-Client/1.0'
        }

        req = urllib.request.Request(f"{self.comfy_url}/upload/image", data=body, headers=headers)
        try:
            with urllib.request.urlopen(req) as response:
                resp_data = json.loads(response.read())
                return resp_data.get("name", filename)
        except Exception as e:
            print(f"Failed to upload image to ComfyUI API: {e}")
            return filename

    def _build_comfyui_workflow_json(
        self,
        prompt: str,
        bw_image_name: str,
        style_image_name: str | None,
        seed: int,
        strength: float,
        options: dict,
    ) -> dict:
        """
        Monta o workflow correto do Klein 4B para colorização de manga P&B.

        Arquitetura:
          Nós 1-2  : LoadImage + Scale da página P&B
          Nós 20-21: LoadImage + Scale da style_ref colorida (se disponível)
          Nó  3    : UnetLoaderGGUF (Klein 4B)
          Nós 4-5  : CLIPLoader + VAELoader
          Nós 6-7  : CLIPTextEncode positivo/negativo
          Nó  8    : VAEEncode da página P&B  → latent_image do KSampler
          Nó  22   : VAEEncode da style_ref   → ReferenceLatent
          Nós 9-10 : ReferenceLatent (usa style_ref como âncora visual de cor)
          Nó  13   : KSampler  — denoise < 1.0 preserva os traços do manga
          Nós 18-19: VAEDecode + SaveImage
        """
        steps = options.get("num_inference_steps", 28) if options else 28
        cfg   = options.get("guidance_scale", 4.0)     if options else 4.0

        # Decide qual imagem alimenta o ReferenceLatent:
        # Para modelos de edição (edit models) como o Flux Klein, a imagem guia original (source/unedited)
        # deve ser sempre a página P&B para que os traços sejam preservados deterministicamente.
        ref_vae_node = "8"

        workflow: dict = {
            # ── Página P&B (o que será colorizado) ───────────────────────────
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": bw_image_name}
            },
            "2": {
                "class_type": "ImageScaleToTotalPixels",
                "inputs": {
                    "image": ["1", 0],
                    "upscale_method": "lanczos",
                    "megapixels": 1.0,
                    "resolution_steps": 64,
                }
            },

            # ── Modelos ───────────────────────────────────────────────────────
            "3": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": "flux-2-klein-4b-Q4_K_M.gguf"}
            },
            "4": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen_3_4b_fp4_flux2.safetensors",
                    "type": "flux2",
                }
            },
            "5": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "flux2-vae.safetensors"}
            },

            # ── Condicionamento textual ───────────────────────────────────────
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt or "colorMangaKlein, vibrant colors, detailed shading",
                    "clip": ["4", 0],
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "grayscale, monochrome, blurry, low quality, deformed",
                    "clip": ["4", 0],
                }
            },

            # ── VAEEncode da página P&B → latent_image do KSampler ───────────
            "8": {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": ["2", 0],
                    "vae":    ["5", 0],
                }
            },

            # ── ReferenceLatent (âncora visual de cor) ────────────────────────
            "9": {
                "class_type": "ReferenceLatent",
                "inputs": {
                    "conditioning": ["6", 0],
                    "latent":       [ref_vae_node, 0],
                }
            },
            "10": {
                "class_type": "ReferenceLatent",
                "inputs": {
                    "conditioning": ["7", 0],
                    "latent":       [ref_vae_node, 0],
                }
            },

            # ── KSampler: coloriza P&B guiado pela referência ─────────────────
            # denoise < 1.0 preserva a estrutura dos traços originais.
            # Recomendado: 0.85 (preserva traços) a 0.95 (mais cor, menos traços).
            "13": {
                "class_type": "KSampler",
                "inputs": {
                    "seed":          seed,
                    "steps":         steps,
                    "cfg":           cfg,
                    "sampler_name":  "euler",
                    "scheduler":     "beta",
                    "denoise":       strength,
                    "model":         ["3", 0],
                    "positive":      ["9", 0],
                    "negative":      ["10", 0],
                    "latent_image":  ["8", 0],   # ← página P&B, não a style_ref
                }
            },

            # ── Decode e salva ────────────────────────────────────────────────
            "18": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["13", 0],
                    "vae":     ["5", 0],
                }
            },
            "19": {
                "class_type": "SaveImage",
                "inputs": {
                    "images":          ["18", 0],
                    "filename_prefix": "manga_colorized",
                }
            },
        }

        # Adiciona nós da style_ref somente se ela foi enviada
        if style_image_name:
            workflow["20"] = {
                "class_type": "LoadImage",
                "inputs": {"image": style_image_name}
            }
            workflow["21"] = {
                "class_type": "ImageScaleToTotalPixels",
                "inputs": {
                    "image":          ["20", 0],
                    "upscale_method": "lanczos",
                    "megapixels":     1.0,
                    "resolution_steps": 64,
                }
            }
            workflow["22"] = {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": ["21", 0],
                    "vae":    ["5", 0],
                }
            }

        return workflow

    def unload(self) -> None:
        """No memory to unload locally, ComfyUI manages its own VRAM over RPC."""
        pass

