import os
import subprocess
import time
import requests
import logging
from pathlib import Path
from typing import Optional

from config.settings import (
    LLAMA_CPP_DIR,
    LLAMA_SERVER_EXE,
    GEMMA_MIN_VRAM_MB,
    GEMMA_CTX,
    REASONING_BUDGET,
    DRAFT_MODEL_PATH,
    active_vlm_config,
)

logger = logging.getLogger("LLAMACppServerManager")
logger.setLevel(logging.INFO)


def _active() -> dict:
    """Modelo/mmproj/porta ativos (VLM_MODEL=gemma|qwen3.5)."""
    return active_vlm_config()

class LLAMACppServerManager:
    """
    Gerencia o ciclo de vida do llama-server.exe em segundo plano.
    Calcula dinamicamente a VRAM livre para decidir a offload de GPU optimal.
    """
    
    _process: Optional[subprocess.Popen] = None

    @staticmethod
    def get_free_vram_mb() -> float:
        """Consulta nvidia-smi para obter a quantidade exata de VRAM livre na GPU em MB."""
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,nounits,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            out = res.stdout.strip()
            if out:
                total, used = map(float, out.split(","))
                return total - used
        except Exception as e:
            logger.warning(f"Falha ao consultar nvidia-smi: {e}. Usando modo de seguranca (CPU).")
        return 0.0

    @classmethod
    def calculate_gpu_layers(cls) -> int:
        """
        Faz as contas de alocação de VRAM baseadas na GPU RTX 3060 12GB:
        - Gemma 2B GGUF Q4_K_XL + mmproj-F32 pesam juntos cerca de 5.1GB no disco, mas apenas ~2.1GB em VRAM.
        - ATENÇÃO: Para evitar o bug crítico do scheduler do llama.cpp (GGML_SCHED_MAX_SPLIT_INPUTS failed),
          o modelo multimodal NUNCA deve ter camadas divididas (partial offload) entre CPU/GPU.
          Ele deve rodar 100% na GPU (-ngl 999) ou 100% na CPU (-ngl 0).
        """
        free_vram = cls.get_free_vram_mb()
        logger.info(f"VRAM Livre detectada: {free_vram:.1f} MB (~{free_vram/1024:.2f} GB)")

        if free_vram > GEMMA_MIN_VRAM_MB:
            logger.info(f"VRAM suficiente (>{GEMMA_MIN_VRAM_MB} MB). Alocando Gemma na GPU (-ngl 999).")
            return 999
        else:
            logger.info(f"VRAM menor que o limite seguro ({GEMMA_MIN_VRAM_MB} MB). Alocando Gemma na CPU (-ngl 0) para evitar o bug de fragmentacao.")
            return 0

    @classmethod
    def start_server(cls) -> bool:
        """
        Inicia o llama-server.exe em segundo plano se já não estiver rodando.
        Retorna True se o servidor estiver ativo e pronto para receber conexões.
        """
        cfg = _active()
        port = cfg["port"]
        # Verifica se a porta já está ocupada (servidor já ativo)
        try:
            res = requests.get(f"http://localhost:{port}/v1/models", timeout=1.0)
            if res.status_code == 200:
                logger.info(f"Servidor llama.cpp já está ativo na porta {port}.")
                return True
        except Exception:
            pass

        if not os.path.exists(LLAMA_SERVER_EXE):
            logger.error(f"Executável do llama-server não encontrado em: {LLAMA_SERVER_EXE}")
            return False

        if not os.path.exists(cfg["model"]) or not os.path.exists(cfg["mmproj"]):
            logger.error(f"Modelo VLM ou mmproj não encontrados: {cfg['model']} | {cfg['mmproj']}")
            return False

        # Faz as contas de alocação de VRAM
        ngl = cls.calculate_gpu_layers()

        cmd = [
            LLAMA_SERVER_EXE,
            "-m", cfg["model"],
            "--mmproj", cfg["mmproj"],
            "--port", str(port),
            "-ngl", str(ngl),
            "-c", str(GEMMA_CTX),
            "-t", "8",
            "--no-mmap",
            "--no-warmup"
        ]
        if REASONING_BUDGET is not None:
            cmd += ["--reasoning-budget", str(REASONING_BUDGET)]
        # Draft só quando o arquivo existe E há VRAM (draft ~1GB extra).
        # Medido em 2026-09-21 (Qwen3.5-4B + 0.8B draft, resolver chapter):
        # 63.4 tok/s COM draft vs 71.7 SEM — thinking imprevisível tem baixa
        # aceitação e o overhead piora. Default OFF; ligue com USE_DRAFT=1
        # apenas se medir ganho no seu workload.
        if (
            DRAFT_MODEL_PATH
            and os.getenv("USE_DRAFT", "0") == "1"
            and os.path.exists(DRAFT_MODEL_PATH)
        ):
            cmd += ["--model-draft", DRAFT_MODEL_PATH]

        logger.info(f"Iniciando llama-server em segundo plano: {' '.join(cmd)}")
        
        try:
            # Roda em segundo plano ocultando console no Windows se necessário
            cls._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            # Aguarda o servidor responder (/v1/models) por ate 60 segundos
            for attempt in range(60):
                time.sleep(1.0)
                try:
                    res = requests.get(f"http://localhost:{port}/v1/models", timeout=1.0)
                    if res.status_code == 200:
                        logger.info("Servidor llama.cpp iniciado com sucesso e pronto para uso!")
                        return True
                except Exception:
                    pass
                logger.info(f"Aguardando inicializacao do llama-server... (tentativa {attempt+1}/60)")
                
            logger.error("Timeout excedido aguardando o servidor responder.")
            cls.stop_server()
            return False
            
        except Exception as e:
            logger.error(f"Erro ao iniciar llama-server: {e}")
            cls._process = None
            return False

    @classmethod
    def stop_server(cls):
        """Finaliza o processo do llama-server.exe para liberar 100% da VRAM da GPU."""
        if cls._process:
            logger.info("Finalizando processo do llama-server para liberar VRAM da GPU...")
            try:
                cls._process.terminate()
                cls._process.wait(timeout=3.0)
            except Exception:
                try:
                    cls._process.kill()
                except Exception:
                    pass
            cls._process = None
            logger.info("llama-server finalizado com sucesso!")
        else:
            # Caso tenha sido iniciado externamente por fora do Python, tenta matar via taskkill
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", "llama-server.exe"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                logger.info("Comando taskkill enviado para garantir liberação total da VRAM.")
            except Exception:
                pass
