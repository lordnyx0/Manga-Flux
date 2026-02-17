#!/usr/bin/env python
"""Download modelos necessários para o Manga-Flux."""
import argparse
import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("huggingface_hub não instalado. Instalando...")
    os.system("pip install -q huggingface_hub")
    from huggingface_hub import hf_hub_download, list_repo_files


MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_yolo_manga():
    """Baixa modelo YOLO Manga109 do deepghs."""
    print("[INFO] Baixando YOLO Manga109...")
    repo_id = "deepghs/manga109_yolo"
    
    # Usa versão m_yv11 (média, YOLOv11)
    subfolder = "v2023.12.07_m_yv11"
    
    try:
        # Baixa o modelo ONNX
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subfolder}/model.onnx",
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )
        # Renomeia para o nome esperado
        dest = MODELS_DIR / "manga109_yolo.onnx"
        import shutil
        shutil.move(model_path, dest)
        print(f"[OK] Modelo baixado: {dest}")
        return True
    except Exception as e:
        print(f"[ERRO] Falha ao baixar modelo ONNX: {e}")
        # Tenta formato .pt
        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{subfolder}/model.pt",
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False
            )
            dest = MODELS_DIR / "manga109_yolo.pt"
            import shutil
            shutil.move(model_path, dest)
            print(f"[OK] Modelo baixado: {dest}")
            return True
        except Exception as e2:
            print(f"[ERRO] Também falhou em .pt: {e2}")
            return False


def download_sam2():
    """Baixa SAM 2.1 Tiny."""
    print("[INFO] SAM2 é baixado automaticamente pelo ultralytics.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download modelos Manga-Flux")
    parser.add_argument("--models", choices=["yolo_manga", "sam2", "all"], 
                       default="all", help="Qual modelo baixar")
    args = parser.parse_args()
    
    success = True
    if args.models in ("yolo_manga", "all"):
        success = download_yolo_manga() and success
    if args.models in ("sam2", "all"):
        success = download_sam2() and success
    
    if success:
        print("\n[OK] Todos os modelos baixados com sucesso!")
    else:
        print("\n[AVISO] Alguns modelos falharam. Verifique os erros acima.")


if __name__ == "__main__":
    main()
