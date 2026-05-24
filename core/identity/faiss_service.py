import numpy as np
import torch
from typing import List, Dict, Optional, Union, Tuple
import logging

logger = logging.getLogger("FaissService")

FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    pass

class FaissService:
    """
    Serviço leve de busca vetorial para pareamento de personagens.
    Usa FAISS (IndexFlatIP) se disponível, com fallback para numpy/cosseno.
    """
    
    def __init__(self, dimension: int = 768, use_faiss: bool = True):
        self.dimension = dimension
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.reference_data = []  # Lista de dicts mapeando índice para info do personagem
        self.faiss_index = None
        self.numpy_embeddings = []  # Lista de arrays numpy normalizados para fallback
        
        if self.use_faiss:
            # IndexFlatIP usa Produto Interno. Para vetores normalizados, é idêntico à Similaridade de Cosseno.
            self.faiss_index = faiss.IndexFlatIP(dimension)
            logger.info("FaissService inicializado usando FAISS (IndexFlatIP)")
        else:
            logger.info("FaissService inicializado usando Numpy Cosine Similarity (Fallback)")
            
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def add_reference_character(self, embedding: Union[np.ndarray, List[float], torch.Tensor, List[List[float]]], metadata: Dict):
        """
        Adiciona um personagem de referência ao índice.
        """
        # Converter embedding para array numpy 1D
        if isinstance(embedding, list):
            # Lidar com lista aninhada [[...]]
            if len(embedding) > 0 and isinstance(embedding[0], list):
                embedding = embedding[0]
            arr = np.array(embedding, dtype=np.float32)
        elif isinstance(embedding, torch.Tensor):
            arr = embedding.detach().cpu().numpy().astype(np.float32).flatten()
        elif isinstance(embedding, np.ndarray):
            arr = embedding.astype(np.float32).flatten()
        else:
            logger.warning("Tipo de embedding inválido para add_reference_character, ignorando.")
            return

        # Garante dimensão correta
        if arr.shape[0] != self.dimension:
            if len(self.reference_data) == 0:
                logger.info(f"Redimensionando índice do FaissService de {self.dimension} para {arr.shape[0]}")
                self.dimension = arr.shape[0]
                if self.use_faiss:
                    self.faiss_index = faiss.IndexFlatIP(self.dimension)
            else:
                logger.error(f"Dimensão do embedding ({arr.shape[0]}) não bate com a do índice ({self.dimension})")
                return
                
        normalized_arr = self._normalize(arr)
        
        if self.use_faiss:
            # FAISS espera matriz 2D (num_vectors, dimension)
            self.faiss_index.add(np.expand_dims(normalized_arr, axis=0))
        else:
            self.numpy_embeddings.append(normalized_arr)
            
        self.reference_data.append(metadata)
        logger.info(f"Personagem de referência adicionado: {metadata.get('char_id', 'unknown')}")

    def search(self, query_embedding: Union[np.ndarray, List[float], torch.Tensor, List[List[float]]], threshold: float = 0.5) -> Tuple[Optional[Dict], float]:
        """
        Busca o personagem de referência mais similar no índice.
        
        Returns:
            Tuple[Optional[Dict], float]: (metadata_do_melhor_match, score_de_similaridade)
        """
        if len(self.reference_data) == 0:
            return None, 0.0
            
        # Converter embedding para array numpy 1D
        if isinstance(query_embedding, list):
            if len(query_embedding) > 0 and isinstance(query_embedding[0], list):
                query_embedding = query_embedding[0]
            arr = np.array(query_embedding, dtype=np.float32)
        elif isinstance(query_embedding, torch.Tensor):
            arr = query_embedding.detach().cpu().numpy().astype(np.float32).flatten()
        elif isinstance(query_embedding, np.ndarray):
            arr = query_embedding.astype(np.float32).flatten()
        else:
            return None, 0.0
            
        if arr.shape[0] != self.dimension:
            logger.error(f"Dimensão do query embedding ({arr.shape[0]}) não bate com a do índice ({self.dimension})")
            return None, 0.0
            
        normalized_arr = self._normalize(arr)
        
        best_idx = -1
        best_sim = -1.0
        
        if self.use_faiss:
            # Query precisa ser 2D para FAISS
            distances, indices = self.faiss_index.search(np.expand_dims(normalized_arr, axis=0), 1)
            best_sim = float(distances[0][0])
            best_idx = int(indices[0][0])
        else:
            # Fallback manual em Numpy
            for idx, ref_emb in enumerate(self.numpy_embeddings):
                sim = float(np.dot(normalized_arr, ref_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx
                    
        if best_idx >= 0 and best_idx < len(self.reference_data) and best_sim >= threshold:
            return self.reference_data[best_idx], best_sim
            
        return None, best_sim
