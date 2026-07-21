"""Módulo de geração de embeddings."""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import CONFIG


class EmbeddingModel:
    """Gerenciador de modelo de embeddings."""
    
    def __init__(self, model_name: str = None, hf_token: str = None):
        """
        Inicializa o modelo de embeddings.
        
        Args:
            model_name: Nome do modelo sentence-transformers
            hf_token: Token Hugging Face (opcional)
        """
        self.model_name = model_name or CONFIG.embedding.model_name
        self.hf_token = hf_token or CONFIG.embedding.hf_token
        
        # Configurar token HF se fornecido
        if self.hf_token:
            import os
            os.environ['HF_TOKEN'] = self.hf_token
        
        # Carregar modelo
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, text: Union[str, List[str]], 
               show_progress: bool = False) -> np.ndarray:
        """
        Gera embeddings para texto(s).
        
        Args:
            text: Texto único ou lista de textos
            show_progress: Mostrar barra de progresso
            
        Returns:
            Array numpy com embeddings
        """
        return self.model.encode(
            text, 
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
    
    def encode_batch(self, texts: List[str], 
                     batch_size: int = 32) -> np.ndarray:
        """
        Gera embeddings em lote.
        
        Args:
            texts: Lista de textos
            batch_size: Tamanho do lote
            
        Returns:
            Array numpy com embeddings
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )