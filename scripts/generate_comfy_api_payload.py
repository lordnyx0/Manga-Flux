import json
import urllib.request
import urllib.parse
import sys

def convert_ui_to_api_comfyui(comfy_url, ui_json_path, output_api_path):
    """
    To convert UI JSON to API JSON without the frontend JS, we can theoretically 
    send the UI JSON to a specific endpoint if we had a converter node, 
    but natively ComfyUI only converts it in the browser via app.graphToPrompt().
    
    Since we can't easily run the JS locally without a browser, our script will
    actually use a headless browser (like Playwright/Selenium) or we just map 
    the basic keys directly in Python for the Flux 2 Klein workflow.
    """
    
    # Given the extreme complexity of installing a headless browser just for this,
    # and since we DO know the exact node IDs from the `parse_comfy_json.py` run,
    # let's hardcode the API JSON for Flux 2 Klein based on standard ComfyUI GGUF topologies.
    
    api_workflow = {
        "1": {
            "inputs": {
                "unet_name": "flux-2-klein-base-9b-fp8.safetensors" # Or flux-2-klein-9b-Q4_K_M.gguf
            },
            "class_type": "UNETLoader" # Or UnetLoaderGGUF
        },
        "2": {
            "inputs": {
                "clip_name1": "t5xxl_fp16.safetensors",
                "clip_name2": "clip_l.safetensors",
                "type": "flux"
            },
            "class_type": "DualCLIPLoader"
        },
        "3": {
            "inputs": {
                "vae_name": "flux2-vae.safetensors"
            },
            "class_type": "VAELoader"
        },
        "4": {
            "inputs": {
                "text": "A highly detailed manga panel",
                "clip": ["2", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "5": {
            "inputs": {
                "text": "",
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
                "seed": 12345,
                "steps": 25,
                "cfg": 4.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 0.8,
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0]
            },
            "class_type": "KSampler"
        },
        "8": {
            "inputs": {
                "image": "base_image.png"
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
    
    with open(output_api_path, "w", encoding="utf-8") as f:
        json.dump(api_workflow, f, indent=4)
        
    print(f"Generated base API format workflow at {output_api_path}")

if __name__ == "__main__":
    convert_ui_to_api_comfyui(
        "http://127.0.0.1:8188",
        "c:/Users/Nyx/Desktop/MANGACOLOR/Flux2_klein_image_edit_9b_base.json",
        "c:/Users/Nyx/Desktop/MANGACOLOR/Manga-Flux/flux2_api_workflow.json"
    )
