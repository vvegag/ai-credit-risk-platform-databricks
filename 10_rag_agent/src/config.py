"""Configurações centralizadas do projeto RAG."""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class VectorSearchConfig:
    """Configurações do Databricks Vector Search."""
    endpoint_name: str = "credit_risk_vector_endpoint"
    index_name: str = "credit_risk.documentos.credit_docs_vector_index"
    source_table: str = "credit_risk.documentos.embeddings_documentos"
    embedding_dimension: int = 768


@dataclass
class EmbeddingConfig:
    """Configurações do modelo de embeddings."""
    model_name: str = "paraphrase-multilingual-mpnet-base-v2"
    dimension: int = 768
    hf_token: Optional[str] = None

    def __post_init__(self):
        # Carregar HF token de variável de ambiente se não fornecido
        if self.hf_token is None:
            self.hf_token = os.getenv('HF_TOKEN')


@dataclass
class LLMConfig:
    """Configurações do LLM."""
    endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
    temperature: float = 0.1
    max_tokens: int = 1000


@dataclass
class RAGConfig:
    """Configurações gerais do RAG Agent."""
    vector_search: VectorSearchConfig = None
    embedding: EmbeddingConfig = None
    llm: LLMConfig = None
    top_k_documents: int = 5
    
    def __post_init__(self):
        if self.vector_search is None:
            self.vector_search = VectorSearchConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.llm is None:
            self.llm = LLMConfig()


# Instância global de configuração
CONFIG = RAGConfig()