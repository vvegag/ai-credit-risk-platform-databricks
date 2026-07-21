"""Módulo de busca vetorial usando Databricks Vector Search."""
from typing import List, Dict, Any, Optional
from databricks.vector_search.client import VectorSearchClient
from .config import CONFIG
from .embeddings import EmbeddingModel


class VectorSearch:
    """Cliente de busca vetorial."""
    
    def __init__(self, 
                 endpoint_name: str = None,
                 index_name: str = None,
                 embedding_model: EmbeddingModel = None):
        """
        Inicializa cliente de busca vetorial.
        
        Args:
            endpoint_name: Nome do endpoint Vector Search
            index_name: Nome do índice vetorial
            embedding_model: Modelo de embeddings (opcional)
        """
        self.endpoint_name = endpoint_name or CONFIG.vector_search.endpoint_name
        self.index_name = index_name or CONFIG.vector_search.index_name
        
        # Cliente Vector Search
        self.client = VectorSearchClient(disable_notice=True)
        
        # Modelo de embeddings
        self.embedding_model = embedding_model or EmbeddingModel()
    
    def search(self, 
               query: str, 
               top_k: int = 5,
               columns: List[str] = None,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Busca documentos relevantes.
        
        Args:
            query: Texto da consulta
            top_k: Número de resultados
            columns: Colunas a retornar
            filters: Filtros SQL opcionais
            
        Returns:
            Lista de documentos com metadados
        """
        # Gerar embedding da query
        query_embedding = self.embedding_model.encode(query)
        
        # Colunas padrão se não especificadas
        if columns is None:
            columns = [
                "chunk_id", 
                "documento_id", 
                "tipo_documento", 
                "id_cliente", 
                "texto_chunk"
            ]
        
        # Buscar no índice
        try:
            results = self.client.get_index(
                endpoint_name=self.endpoint_name,
                index_name=self.index_name
            ).similarity_search(
                query_vector=query_embedding.tolist(),
                columns=columns,
                num_results=top_k,
                filters=filters
            )
            
            # Extrair e formatar documentos
            if results and 'result' in results:
                docs = results['result']['data_array']
                return self._format_results(docs, columns)
            
            return []
            
        except Exception as e:
            print(f"❌ Erro na busca vetorial: {e}")
            return []
    
    def _format_results(self, 
                        docs: List[List[Any]], 
                        columns: List[str]) -> List[Dict[str, Any]]:
        """
        Formata resultados da busca.
        
        Args:
            docs: Resultados brutos
            columns: Nomes das colunas
            
        Returns:
            Lista de dicionários
        """
        formatted = []
        for doc in docs:
            doc_dict = {col: val for col, val in zip(columns, doc)}
            # Renomear texto_chunk para texto (mais genérico)
            if 'texto_chunk' in doc_dict:
                doc_dict['texto'] = doc_dict.pop('texto_chunk')
            formatted.append(doc_dict)
        
        return formatted