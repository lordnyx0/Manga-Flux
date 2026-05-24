import base64
import requests
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Union, Optional
from PIL import Image

logger = logging.getLogger("VLMService")
logger.setLevel(logging.INFO)

from config.settings import VLM_PROVIDER, VLM_PORT
VLM_TIMEOUT_SECONDS = 180.0

class VLMService:
    """
    Serviço VLM local para comunicação com LM Studio ou llama-server (OpenAI-compatible API).
    Analisa imagens de personagens e extrai descrições semânticas de cores altamente detalhadas.
    """
    
    def __init__(self, host: str = "localhost", port: int = VLM_PORT):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/v1"
        
    def _encode_image_to_base64(self, image: Union[str, Path, Image.Image]) -> str:
        """Converte uma imagem (caminho ou PIL) para string Base64 em formato PNG."""
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image, Image.Image):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            raise ValueError("Formato de imagem não suportado. Use caminho (str/Path) ou PIL Image.")

    def _get_active_model(self) -> str:
        """Consulta LM Studio para obter o identificador do modelo carregado no servidor."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=3.0)
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                if models:
                    return models[0].get("id", "local-model")
        except Exception:
            pass
        return "local-model"

    def describe_character_colors(self, character_image: Union[str, Path, Image.Image]) -> Optional[str]:
        """
        Envia a imagem recortada do personagem para o VLM local.
        Retorna uma string de descrição de cor padronizada e semântica.
        
        Retorna None se o servidor estiver offline, o modelo não suportar visão ou ocorrer falha.
        """
        # Se configurado para rodar via llama.cpp local em segundo plano, garante inicializacao
        if VLM_PROVIDER == "llama-cpp":
            try:
                from core.identity.llama_server_manager import LLAMACppServerManager
                if not LLAMACppServerManager.start_server():
                    logger.error("Falha ao inicializar o llama-server em segundo plano.")
                    return None
            except Exception as e:
                logger.error(f"Erro ao gerenciar llama-server: {e}")
                return None

        try:
            base64_image = self._encode_image_to_base64(character_image)
        except Exception as e:
            logger.error(f"Falha ao codificar imagem em base64: {e}")
            return None

        # Detecta o modelo ativo
        model_name = self._get_active_model()
        logger.info(f"Usando modelo local ativo: {model_name}")

        # Configura o Prompt do Sistema e do Usuário para extração de JSON estruturado
        system_prompt = (
            "You are a professional manga and anime character color analyzer VLM.\n"
            "Your task is to analyze the character in the image and extract their exact colors.\n"
            "You MUST respond ONLY with a valid JSON object matching the following schema:\n"
            "{\n"
            "  \"hair\": \"[color description]\",\n"
            "  \"skin\": \"[color description]\",\n"
            "  \"eyes\": \"[color description]\",\n"
            "  \"clothes\": \"[color description]\",\n"
            "  \"accessories\": \"[color description]\"\n"
            "}\n"
            "CRITICAL: Do NOT output any explanation, markdown, code blocks, or preamble outside the JSON object. Output the raw JSON."
        )

        user_prompt = "Analyze the character in this crop and extract their precise color attributes into the requested JSON format."

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,  # Baixa temperatura para manter respostas determinísticas e factuais
            "max_tokens": 1024   # Aumentado para dar espaço ao pensamento (Reasoning) + resposta JSON
        }

        url = f"{self.base_url}/chat/completions"
        
        try:
            logger.info("Enviando requisição ao VLM local (LM Studio/llama.cpp)...")
            response = requests.post(url, headers=headers, json=payload, timeout=VLM_TIMEOUT_SECONDS)
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                
                # Tenta parsear a resposta como JSON para maior precisão estrutural
                try:
                    cleaned_content = content.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(cleaned_content)
                    
                    parts = []
                    for key in ["hair", "skin", "eyes", "clothes", "accessories"]:
                        val = parsed.get(key, "").strip()
                        # Ignora valores vazios, neutros, N/A ou invisíveis
                        if val and val.lower() not in ["n/a", "none", "not visible", "null", "unknown", "not visible in image"]:
                            parts.append(f"{val} {key}")
                    
                    if parts:
                        content = ", ".join(parts)
                    else:
                        content = "N/A"
                except Exception as e:
                    logger.warning(f"Falha ao parsear JSON da resposta do VLM, usando fallback raw: {e}")
                    content = content.replace("`", "").replace("*", "").replace("\n", " ").strip()
                    if content.startswith('"') and content.endswith('"'):
                        content = content[1:-1].strip()
                
                logger.info(f"VLM processou com sucesso e retornou: '{content}'")
                return content
            else:
                logger.warning(f"LM Studio retornou status de erro {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Timeout excedido na requisição ao VLM local.")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Não foi possível conectar ao servidor LM Studio local. Verifique se ele está ativo na porta 1234.")
            return None
        except Exception as e:
            logger.error(f"Erro inesperado na chamada ao VLM local: {e}")
            return None

    def describe_all_characters(self, image_path: Union[str, Path, Image.Image]) -> Optional[list]:
        """
        Analisa a imagem inteira (geralmente style_ref.png) e extrai
        a descrição estruturada de todos os personagens de uma só vez.
        Retorna uma lista de dicionários correspondentes aos personagens encontrados.
        """
        if VLM_PROVIDER == "llama-cpp":
            try:
                from core.identity.llama_server_manager import LLAMACppServerManager
                if not LLAMACppServerManager.start_server():
                    logger.error("Falha ao inicializar o llama-server em segundo plano.")
                    return None
            except Exception as e:
                logger.error(f"Erro ao gerenciar llama-server: {e}")
                return None

        try:
            base64_image = self._encode_image_to_base64(image_path)
        except Exception as e:
            logger.error(f"Falha ao codificar imagem em base64: {e}")
            return None

        model_name = self._get_active_model()
        logger.info(f"Usando modelo local ativo para análise global: {model_name}")

        system_prompt = (
            "You are a professional manga and anime character color analyzer VLM.\n"
            "Analyze the ENTIRE image and identify ALL characters present. For each character:\n"
            "1. Determine their horizontal position in the image (e.g., 'left', 'center', 'right').\n"
            "2. Extract their exact color attributes (hair, skin, eyes, clothes, accessories).\n\n"
            "You MUST respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            "  \"characters\": [\n"
            "    {\n"
            "      \"position\": \"[left / center / right]\",\n"
            "      \"hair\": \"[color description]\",\n"
            "      \"skin\": \"[color description]\",\n"
            "      \"eyes\": \"[color description]\",\n"
            "      \"clothes\": \"[color description]\",\n"
            "      \"accessories\": \"[color description]\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "CRITICAL: Output ONLY the raw JSON object. Do not include markdown code blocks, comments, or explanations."
        )

        user_prompt = "Identify and analyze all characters in the image. Return the description of each character in the requested JSON schema."

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 4096   # Concede o budget completo de 4K solicitado pelo usuário
        }

        url = f"{self.base_url}/chat/completions"
        
        try:
            logger.info("Enviando requisição global de VLM (LM Studio/llama.cpp)...")
            response = requests.post(url, headers=headers, json=payload, timeout=VLM_TIMEOUT_SECONDS)
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                
                try:
                    cleaned_content = content.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(cleaned_content)
                    characters_list = parsed.get("characters", [])
                    logger.info(f"VLM processou globalmente com sucesso e identificou {len(characters_list)} personagens.")
                    return characters_list
                except Exception as e:
                    logger.error(f"Falha ao parsear JSON global do VLM: {e}. Resposta crua: {content}")
                    return None
            else:
                logger.warning(f"Servidor retornou status de erro {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Timeout excedido na requisição global ao VLM.")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Não foi possível conectar ao servidor VLM local na porta 1234.")
            return None
        except Exception as e:
            logger.error(f"Erro inesperado na chamada global ao VLM: {e}")
            return None

    def generate_modular_flux_prompt(
        self,
        bw_page_image: Union[str, Path, Image.Image],
        character_registry: list,
        prompt_hint: Optional[str] = None,
    ) -> Optional[str]:
        """
        Analisa a página P&B usando o registro de personagens da capa como referência
        e gera um prompt modularizado no formato esperado pelo Klein 4B (colorMangaKlein).

        O prompt segue as seções: [Layout], [Character Design], [Color Mapping],
        [Lighting], [Background] e [Rendering], onde palavras próximas são tratadas
        como clusters semânticos pelo mecanismo de atenção do Flux.

        Args:
            bw_page_image: Imagem P&B da página a colorir.
            character_registry: Lista de dicts gerada por describe_all_characters()
                                 contendo hair/skin/eyes/clothes/accessories por personagem.

        Returns:
            String com o prompt modularizado, ou None em caso de falha.
        """
        if VLM_PROVIDER == "llama-cpp":
            try:
                from core.identity.llama_server_manager import LLAMACppServerManager
                if not LLAMACppServerManager.start_server():
                    logger.error("Falha ao inicializar o llama-server em segundo plano.")
                    return None
            except Exception as e:
                logger.error(f"Erro ao gerenciar llama-server: {e}")
                return None

        try:
            base64_image = self._encode_image_to_base64(bw_page_image)
        except Exception as e:
            logger.error(f"Falha ao codificar imagem P&B em base64: {e}")
            return None

        model_name = self._get_active_model()
        logger.info(f"Usando modelo local ativo para geração de prompt modular: {model_name}")

        # Serializa o registro de personagens para injetar no system prompt
        registry_text = json.dumps(character_registry, indent=2, ensure_ascii=False)

        system_prompt = (
            "You are an expert prompt engineer for Flux-based manga colorization models.\n"
            "You will receive:\n"
            "  1. A black-and-white manga page to be colorized.\n"
            "  2. A character registry extracted from the manga cover (JSON below).\n\n"
            "CHARACTER REGISTRY (from cover):\n"
            f"{registry_text}\n\n"
            "YOUR TASK:\n"
            "Analyze the B&W page carefully. Identify which characters from the registry appear,\n"
            "their panel positions (left/right/center/top/bottom), and their expressions.\n"
            "Then generate a colorization prompt in the EXACT modular format shown below.\n\n"
            "REQUIRED OUTPUT FORMAT (copy structure exactly, fill content based on the page):\n"
            "---\n"
            "colorMangaKlein, masterpiece, best quality, anime-style manga colorization, full manga page, highly detailed shading, sharp line art\n\n"
            "[Layout]\n"
            "<Describe the panel composition and which character appears in which panel/side>\n\n"
            "[Character Design]\n"
            "<For each character detected, list a labeled block with bullet points for hair, skin, eyes, outfit, accessories>\n\n"
            "[Color Mapping]\n"
            "<State explicit color fidelity rules: which colors must stay vivid, which must not bleed>\n\n"
            "[Lighting]\n"
            "<Describe lighting style: rim light, contrast, shadows, ambient color>\n\n"
            "[Background]\n"
            "<Describe background colors, textures, atmosphere>\n\n"
            "[Rendering]\n"
            "<Describe rendering quality: fabric detail, line art, eye detail, shadow style>\n"
            "---\n\n"
            "CRITICAL RULES:\n"
            "- Output ONLY the prompt text starting from 'colorMangaKlein'. No preamble, no explanation.\n"
            "- Each section header must appear on its own line inside square brackets.\n"
            "- Keep each section focused. Flux treats nearby words as semantic clusters — be precise.\n"
            "- Do NOT assume character positions (left/right) in the registry match the positions in the B&W page panels! The registry positions are from the cover art, whereas the B&W page layout is completely different. Identify who is who by matching their unique visual traits (hair styles, clothing designs, face marks, headbands, scars, etc.).\n"
            "- If a character cannot be confidently matched to the registry, describe their visible features.\n"
            "- Use the registry colors as ground truth for known characters."
        )

        user_prompt = (
            "Analyze this black-and-white manga page using the character registry provided. "
            "Generate the modular colorization prompt."
        )
        if prompt_hint:
            user_prompt += f"\n\n{prompt_hint}"

        headers = {"Content-Type": "application/json"}

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.2,   # Ligeiramente acima de 0 para criatividade controlada no prompt
            "max_tokens": 4096
        }

        url = f"{self.base_url}/chat/completions"

        try:
            logger.info("Enviando requisição de prompt modular ao VLM (LM Studio/llama.cpp)...")
            response = requests.post(url, headers=headers, json=payload, timeout=VLM_TIMEOUT_SECONDS)

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()

                # Remove markdown fences se o modelo as inserir mesmo com instrução
                content = content.replace("```", "").strip()

                # Valida que o prompt começa com o trigger token correto
                if "colorMangaKlein" not in content:
                    logger.warning(
                        "Resposta do VLM não contém 'colorMangaKlein'. "
                        f"Conteúdo recebido: {content[:200]}"
                    )
                    return None

                # Garante que começa exatamente pelo trigger (remove lixo antes se houver)
                start_idx = content.find("colorMangaKlein")
                content = content[start_idx:]

                logger.info(f"Prompt modular gerado com sucesso ({len(content)} chars).")
                return content
            else:
                logger.warning(
                    f"Servidor retornou status de erro {response.status_code}: {response.text}"
                )
                return None

        except requests.exceptions.Timeout:
            logger.error("Timeout excedido na geração do prompt modular.")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Não foi possível conectar ao servidor VLM local na porta 1234.")
            return None
        except Exception as e:
            logger.error(f"Erro inesperado na geração do prompt modular: {e}")
            return None
