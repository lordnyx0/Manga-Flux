"""
MangaAutoColor Pro - Sistema Híbrido de Identidade

Implementa extração de identidades de personagens usando abordagem híbrida:
1. ArcFace (InsightFace) para identidade facial (quando disponível)
2. CLIP Image Encoder para embedding visual geral (IP-Adapter)
3. Cache de tensores para eficiência no Pass 2

Segue o princípio de imutabilidade: uma vez extraído no Pass 1,
o embedding é congelado e cacheado para todo o capítulo.

Referências:
- InsightFace/ArcFace: github.com/deepinsight/insightface
- IP-Adapter: github.com/tencent-ailab/IP-Adapter
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List, Any
import cv2
from dataclasses import dataclass
import hashlib
import json

from core.identity.vector_store import SQLiteVectorStore

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

try:
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from config.settings import (
    DEVICE, DTYPE, INSIGHTFACE_MODEL, FACE_DETECTION_SIZE,
    EMBEDDINGS_DIR, MAX_CACHED_EMBEDDINGS, VERBOSE
)


@dataclass
class IdentityFeatures:
    """
    Features de identidade de um personagem.
    
    Attributes:
        clip_embedding: Embedding CLIP para IP-Adapter (tensor)
        face_embedding: Embedding facial ArcFace (se disponível)
        face_bbox: Bounding box do rosto detectado
        confidence: Confiança na detecção facial
        method: Método usado ("face+clip", "clip_only", "fallback")
        character_id: ID único do personagem
    """
    clip_embedding: torch.Tensor
    face_embedding: Optional[np.ndarray] = None
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0
    method: str = "unknown"
    character_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Converte para dicionário serializável"""
        return {
            'face_embedding': self.face_embedding.tolist() if self.face_embedding is not None else None,
            'face_bbox': self.face_bbox,
            'confidence': self.confidence,
            'method': self.method,
            'character_id': self.character_id
        }


class HybridIdentitySystem:
    """
    Sistema híbrido de extração e cache de identidades.
    
    O sistema prioriza:
    1. ArcFace + CLIP: Se rosto detectado, usa ambos
    2. CLIP apenas: Se sem rosto, usa apenas CLIP
    3. Cache: Todos embeddings são persistidos no vector store local (SQLite)
    
    Args:
        device: Dispositivo para inferência
        dtype: Tipo de dados (fp16 para RTX 3060)
        use_insightface: Habilitar ArcFace (requer onnxruntime)
        cache_dir: Diretório para cache de embeddings
    """
    
    def __init__(
        self,
        device: str = DEVICE,
        dtype: torch.dtype = DTYPE,
        use_insightface: bool = True,
        cache_dir: Path = EMBEDDINGS_DIR
    ):
        self.device = device
        self.dtype = dtype
        self.use_insightface = use_insightface and INSIGHTFACE_AVAILABLE
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = SQLiteVectorStore(self.cache_dir / "identity_cache.sqlite3")
        
        # Componentes (lazy loading)
        self._face_analyzer = None
        self._clip_processor = None
        self._clip_model = None
        
        # Cache em memória (LRU simples)
        self._memory_cache: Dict[str, IdentityFeatures] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        if VERBOSE:
            print(f"[HybridIdentitySystem] Inicializado (device={device}, "
                  f"insightface={self.use_insightface})")
    
    def _get_face_analyzer(self) -> Optional[Any]:
        """Lazy loading do analisador facial (InsightFace)"""
        if not self.use_insightface:
            return None
        
        if self._face_analyzer is None:
            if not INSIGHTFACE_AVAILABLE:
                print("[HybridIdentitySystem] InsightFace não disponível")
                return None
            
            try:
                self._face_analyzer = FaceAnalysis(
                    name=INSIGHTFACE_MODEL,
                    root=str(self.cache_dir / "insightface"),
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self._face_analyzer.prepare(
                    ctx_id=0 if self.device == "cuda" else -1,
                    det_size=FACE_DETECTION_SIZE
                )
                print("[HybridIdentitySystem] FaceAnalyzer carregado")
            except Exception as e:
                print(f"[HybridIdentitySystem] Erro ao carregar InsightFace: {e}")
                self.use_insightface = False
                return None
        
        return self._face_analyzer
    
    def _get_clip_encoder(self):
        """Lazy loading do encoder CLIP (para IP-Adapter)"""
        if self._clip_model is None:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers não está instalado")
            
            try:
                # CLIP Vision Model para extração de features visuais
                model_name = "openai/clip-vit-large-patch14"
                
                self._clip_processor = CLIPImageProcessor.from_pretrained(model_name)
                self._clip_model = CLIPVisionModelWithProjection.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype
                ).to(self.device)
                
                self._clip_model.eval()
                
                print("[HybridIdentitySystem] CLIP encoder carregado")
            except Exception as e:
                print(f"[HybridIdentitySystem] Erro ao carregar CLIP: {e}")
                raise
        
        return self._clip_processor, self._clip_model
    
    def extract_identity(
        self,
        image: Union[Image.Image, np.ndarray],
        character_hint: Optional[str] = None,
        timeout: float = 5.0
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """
        Extrai embedding de identidade de uma imagem.
        
        Args:
            image: Imagem do personagem (PIL ou numpy)
            character_hint: Nome/dica do personagem (opcional)
            timeout: Timeout para processamento
            
        Returns:
            Tuple de (clip_embedding, face_embedding)
        """
        # Converte para PIL se necessário
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Verifica cache
        cache_key = self._compute_cache_key(image, character_hint)
        
        if cache_key in self._memory_cache:
            self._cache_hits += 1
            cached = self._memory_cache[cache_key]
            return cached.clip_embedding, cached.face_embedding
        
        # Verifica cache persistente (SQLite)
        if self.vector_store.get(cache_key) is not None:
            try:
                features = self._load_from_disk(cache_key)
                self._memory_cache[cache_key] = features
                self._cache_hits += 1
                return features.clip_embedding, features.face_embedding
            except Exception as e:
                print(f"[HybridIdentitySystem] Erro ao carregar cache: {e}")
        
        self._cache_misses += 1
        
        # Extrai features
        features = self._extract_features(image, character_hint)
        features.character_id = cache_key[:16]
        
        # Salva em cache
        self._memory_cache[cache_key] = features
        self._save_to_disk(cache_key, features)
        
        # Limita tamanho do cache em memória
        self._prune_memory_cache()
        
        return features.clip_embedding, features.face_embedding
    
    def _extract_features(
        self,
        image: Image.Image,
        character_hint: Optional[str] = None
    ) -> IdentityFeatures:
        """
        Extrai features usando ArcFace + CLIP ou CLIP apenas.
        
        Args:
            image: Imagem PIL
            character_hint: Dica do personagem
            
        Returns:
            IdentityFeatures
        """
        # Converte para numpy para InsightFace
        image_np = np.array(image)
        if len(image_np.shape) == 2:  # Grayscale
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Tenta extrair embedding facial
        face_embedding = None
        face_bbox = None
        confidence = 0.0
        
        if self.use_insightface:
            face_analyzer = self._get_face_analyzer()
            if face_analyzer is not None:
                try:
                    faces = face_analyzer.get(image_np)
                    
                    if faces and len(faces) > 0:
                        # Pega a face mais proeminente (maior)
                        main_face = max(faces, key=lambda f: 
                            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                        
                        face_embedding = main_face.embedding
                        face_bbox = tuple(map(int, main_face.bbox))
                        confidence = main_face.det_score
                except Exception as e:
                    print(f"[HybridIdentitySystem] Erro na detecção facial: {e}")
        
        # Extrai embedding CLIP (sempre)
        clip_embedding = self._extract_clip_embedding(image)
        
        # Determina método
        if face_embedding is not None:
            method = "face+clip"
        else:
            method = "clip_only"
            confidence = 0.5  # Confiança base para CLIP only
        
        return IdentityFeatures(
            clip_embedding=clip_embedding,
            face_embedding=face_embedding,
            face_bbox=face_bbox,
            confidence=confidence,
            method=method
        )
    
    def _extract_clip_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Extrai embedding CLIP da imagem.
        
        Args:
            image: Imagem PIL
            
        Returns:
            Embedding CLIP normalizado
        """
        processor, model = self._get_clip_encoder()
        
        # Preprocessa
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extrai features
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.image_embeds
            
            # Normaliza
            embedding = F.normalize(embedding, dim=-1)
        
        # Move para CPU para economia de VRAM
        return embedding.cpu().to(self.dtype)
    
    def _compute_cache_key(
        self,
        image: Image.Image,
        character_hint: Optional[str] = None
    ) -> str:
        """
        Computa chave de cache única para a imagem.
        
        Args:
            image: Imagem PIL
            character_hint: Dica do personagem
            
        Returns:
            Hash MD5 da imagem
        """
        # Reduz imagem para computação rápida de hash
        small = image.resize((64, 64))
        img_bytes = small.tobytes()
        
        # Adiciona hint se disponível
        if character_hint:
            img_bytes += character_hint.encode()
        
        return hashlib.md5(img_bytes).hexdigest()
    
    def _save_to_disk(self, cache_key: str, features: IdentityFeatures):
        """Salva features no vector store local (SQLite)."""
        payload = {
            'face_embedding': features.face_embedding.tolist() if features.face_embedding is not None else None,
            'face_bbox': features.face_bbox,
            'confidence': features.confidence,
            'method': features.method,
            'character_id': features.character_id,
        }
        self.vector_store.upsert(
            vector_id=cache_key,
            tensor=features.clip_embedding,
            metadata_json=json.dumps(payload, ensure_ascii=False),
        )
    
    def _load_from_disk(self, cache_key: str) -> IdentityFeatures:
        """Carrega features do SQLite."""
        row = self.vector_store.get(cache_key)
        if row is None:
            raise FileNotFoundError(f"Embedding not found in SQLite cache: {cache_key}")

        clip_embedding, metadata_json = row
        metadata = json.loads(metadata_json)
        face_embedding = metadata.get('face_embedding')
        return IdentityFeatures(
            clip_embedding=clip_embedding.to(self.dtype),
            face_embedding=np.array(face_embedding) if face_embedding is not None else None,
            face_bbox=metadata.get('face_bbox'),
            confidence=metadata.get('confidence', 0.0),
            method=metadata.get('method', 'unknown'),
            character_id=metadata.get('character_id'),
        )
    
    def _prune_memory_cache(self):
        """Limita tamanho do cache em memória (LRU simples)"""
        if len(self._memory_cache) > MAX_CACHED_EMBEDDINGS:
            # Remove 20% mais antigo (simplificação: aleatório)
            keys_to_remove = list(self._memory_cache.keys())[
                :MAX_CACHED_EMBEDDINGS // 5
            ]
            for key in keys_to_remove:
                del self._memory_cache[key]
    
    def find_similar_characters(
        self,
        embedding: torch.Tensor,
        threshold: float = 0.85
    ) -> List[Tuple[str, float]]:
        """
        Encontra personagens similares no cache baseado no embedding CLIP.
        
        Args:
            embedding: Embedding CLIP de consulta
            threshold: Similaridade mínima (cosseno)
            
        Returns:
            Lista de (character_id, similarity) ordenada por similaridade
        """
        results = []
        
        embedding_norm = F.normalize(embedding, dim=-1)
        
        for cache_key, features in self._memory_cache.items():
            cached_emb = features.clip_embedding.to(embedding.device)
            cached_emb_norm = F.normalize(cached_emb, dim=-1)
            
            # Similaridade cosseno
            similarity = torch.dot(embedding_norm.flatten(), cached_emb_norm.flatten())
            similarity = similarity.item()
            
            if similarity >= threshold:
                results.append((features.character_id or cache_key[:16], similarity))
        
        # Ordena por similaridade
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """Retorna estatísticas do cache"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        
        return {
            'memory_cache_size': len(self._memory_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'disk_cache_size': self.vector_store.count()
        }
    
    def clear_memory_cache(self):
        """Limpa cache em memória (libera RAM)"""
        self._memory_cache.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def unload(self):
        """Descarrega modelos da VRAM"""
        self._clip_model = None
        self._clip_processor = None
        self._face_analyzer = None
        self.clear_memory_cache()
        
        if VERBOSE:
            print("[HybridIdentitySystem] Modelos descarregados")
    
    def __del__(self):
        """Destructor"""
        self.unload()


class IdentityCache:
    """Cache persistente de identidades para um capítulo inteiro."""

    def __init__(
        self,
        chapter_id: str,
        cache_dir: Path = EMBEDDINGS_DIR
    ):
        self.chapter_id = chapter_id
        self.cache_dir = Path(cache_dir) / chapter_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = SQLiteVectorStore(self.cache_dir / "chapter_vectors.sqlite3")

        self._character_map: Dict[str, Dict] = {}
        self._next_char_id = 0

        self._load_mapping()

    def _load_mapping(self):
        mapping_file = self.cache_dir / "character_map.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                data = json.load(f)
                self._character_map = data.get('characters', {})
                self._next_char_id = data.get('next_id', 0)

    def _save_mapping(self):
        mapping_file = self.cache_dir / "character_map.json"
        with open(mapping_file, 'w') as f:
            json.dump({
                'characters': self._character_map,
                'next_id': self._next_char_id,
                'chapter_id': self.chapter_id
            }, f, indent=2)

    def register_character(
        self,
        embedding: torch.Tensor,
        detection_bbox: Tuple[int, int, int, int],
        page_num: int
    ) -> str:
        char_id = self._find_similar_character(embedding)

        if char_id is None:
            char_id = f"char_{self._next_char_id:03d}"
            self._next_char_id += 1

            self._character_map[char_id] = {
                'first_seen_page': page_num,
                'embedding_id': char_id,
                'detections': []
            }

            self.vector_store.upsert(
                vector_id=char_id,
                tensor=embedding,
                metadata_json=json.dumps({'created_by': 'IdentityCache'}, ensure_ascii=False),
                chapter_id=self.chapter_id,
            )

        self._character_map[char_id]['detections'].append({
            'page': page_num,
            'bbox': detection_bbox
        })

        self._save_mapping()
        return char_id

    def _find_similar_character(
        self,
        embedding: torch.Tensor,
        threshold: float = 0.90
    ) -> Optional[str]:
        embedding_norm = F.normalize(embedding, dim=-1)

        for char_id, data in self._character_map.items():
            embedding_id = data.get('embedding_id', char_id)
            row = self.vector_store.get(embedding_id)
            if row is None:
                continue

            saved_emb, _ = row
            saved_emb_norm = F.normalize(saved_emb, dim=-1)
            similarity = torch.dot(
                embedding_norm.flatten(),
                saved_emb_norm.flatten()
            ).item()

            if similarity >= threshold:
                return char_id

        return None

    def get_character_embedding(self, char_id: str) -> Optional[torch.Tensor]:
        if char_id not in self._character_map:
            return None

        embedding_id = self._character_map[char_id].get('embedding_id', char_id)
        row = self.vector_store.get(embedding_id)
        if row is None:
            return None
        emb, _ = row
        return emb

    def get_all_characters(self) -> Dict[str, Dict]:
        return self._character_map.copy()

    def get_character_count(self) -> int:
        return len(self._character_map)


# Alias para compatibilidade com código existente
HybridIdentityEncoder = HybridIdentitySystem
