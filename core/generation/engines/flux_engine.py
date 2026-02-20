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
        Espera que o ComfyUI já esteja rodando e com o 'flux-2-klein-9b-Q4_K_M.gguf' disponível.
        """
        prompt = payload.get("prompt", "manga panel")
        base_image_path = payload.get("base_image_path")
        
        if not base_image_path or not os.path.exists(base_image_path):
            raise FileNotFoundError(f"Source image not found: {base_image_path}")
            
        # TODO: Upload the base_image to ComfyUI's /upload/image endpoint 
        # so it can be referenced by the LoadImage node.
        uploaded_image_name = self._upload_image_to_comfy(base_image_path)
        
        # Build the generic ComfyUI JSON Workflow
        comfy_workflow = self._build_comfyui_workflow_json(
            prompt=prompt,
            image_name=uploaded_image_name,
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
                        # Node 10 is our SaveImage node
                        if "10" in outputs and "images" in outputs["10"]:
                            images_data = outputs["10"]["images"]
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
        Faz o POST da imagem para o ComfyUI local via multipart form-data
        e retorna o nome registrado no servidor.
        """
        import uuid
        boundary = uuid.uuid4().hex
        filename = os.path.basename(local_path)
        
        with open(local_path, "rb") as f:
            file_data = f.read()
        
        data = []
        data.append(f'--{boundary}'.encode('utf-8'))
        data.append(f'Content-Disposition: form-data; name="image"; filename="{filename}"'.encode('utf-8'))
        data.append(f'Content-Type: application/octet-stream'.encode('utf-8'))
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

    def _build_comfyui_workflow_json(self, prompt: str, image_name: str, seed: int, strength: float, options: dict) -> dict:
        """
        Monta o nó de Geração (GGUF Loader -> CLIP Text Encode -> Sampler -> VAE Decode) 
        usando a API JSON estática do ComfyUI.
        """
        # Node keys map exactly to the standard ComfyUI API architecture.
        workflow = {
            "1": {
                "inputs": {
                    "unet_name": "flux-2-klein-9b-Q4_K_M.gguf"
                },
                "class_type": "UnetLoaderGGUF"
            },
            "11": {
                "inputs": {
                    "lora_name": "colorMangaKlein_9B.safetensors",
                    "strength_model": 1.0,
                    "model": ["1", 0]
                },
                "class_type": "LoraLoaderModelOnly"
            },
            "2": {
                "inputs": {
                    "clip_name": "qwen_3_8b_fp4mixed.safetensors",
                    "type": "flux2"
                },
                "class_type": "CLIPLoader"
            },
            "3": {
                "inputs": {
                    "vae_name": "flux2-vae.safetensors"
                },
                "class_type": "VAELoader"
            },
            "4": {
                "inputs": {
                    "text": prompt,
                    "clip": ["2", 0]
                },
                "class_type": "CLIPTextEncode"
            },
            "6": {
                "inputs": {
                    "pixels": ["8", 0],
                    "vae": ["3", 0]
                },
                "class_type": "VAEEncode"
            },
            "7": {
                "inputs": {
                    "seed": seed,
                    "steps": options.get("num_inference_steps", 25) if options else 25,
                    "cfg": 1.0, # Flux base cfg 
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 0.55, # Lower strength to prevent hallucinations overriding lines
                    "model": ["11", 0],
                    "positive": ["4", 0],
                    "negative": ["4", 0], # Flux doesn't use negative natively in some setups
                    "latent_image": ["6", 0]
                },
                "class_type": "KSampler"
            },
            "8": {
                "inputs": {
                    "image": image_name
                },
                "class_type": "LoadImage"
            },
            "9": {
                "inputs": {
                    "samples": ["7", 0],
                    "vae": ["3", 0]
                },
                "class_type": "VAEDecode"
            },
            "10": {
                "inputs": {
                    "filename_prefix": "manga_flux",
                    "images": ["9", 0]
                },
                "class_type": "SaveImage"
            }
        }
        return workflow

    def unload(self) -> None:
        """No memory to unload locally, ComfyUI manages its own VRAM over RPC."""
        pass

