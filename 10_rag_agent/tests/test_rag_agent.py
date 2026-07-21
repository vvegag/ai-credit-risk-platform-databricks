"""Testes básicos do RAG Agent."""
import pytest
import sys
sys.path.append('/Workspace/Users/valdomirovega@hotmail.com/ai-credit-risk-platform-databricks/10_rag_agent')

from src.rag_agent import RAGAgent
from src.vector_search import VectorSearch
from src.embeddings import EmbeddingModel


class TestRAGAgent:
    """Suite de testes do RAG Agent."""
    
    @pytest.fixture
    def agent(self):
        """Fixture: instância do RAG Agent."""
        return RAGAgent()
    
    def test_agent_initialization(self, agent):
        """Testa inicialização do agente."""
        assert agent is not None
        assert agent.vector_search is not None
        assert agent.llm is not None
        assert agent.chain is not None
    
    def test_query_returns_dict(self, agent):
        """Testa se query retorna dicionário."""
        resultado = agent.query("Teste de query")
        assert isinstance(resultado, dict)
        assert 'resposta' in resultado
        assert 'documentos' in resultado
    
    def test_query_with_valid_question(self, agent):
        """Testa query com pergunta válida."""
        resultado = agent.query("Quais empresas têm score alto?")
        assert resultado['resposta'] is not None
        assert len(resultado['resposta']) > 0
    
    def test_empty_query_handling(self, agent):
        """Testa handling de query vazia."""
        resultado = agent.query("")
        assert 'resposta' in resultado


class TestVectorSearch:
    """Suite de testes do Vector Search."""
    
    @pytest.fixture
    def vector_search(self):
        """Fixture: instância do Vector Search."""
        return VectorSearch()
    
    def test_vector_search_initialization(self, vector_search):
        """Testa inicialização."""
        assert vector_search is not None
        assert vector_search.client is not None
        assert vector_search.embedding_model is not None
    
    def test_search_returns_list(self, vector_search):
        """Testa se search retorna lista."""
        resultados = vector_search.search("teste", top_k=3)
        assert isinstance(resultados, list)


class TestEmbeddingModel:
    """Suite de testes do Embedding Model."""
    
    @pytest.fixture
    def embedding_model(self):
        """Fixture: instância do modelo."""
        return EmbeddingModel()
    
    def test_model_initialization(self, embedding_model):
        """Testa inicialização."""
        assert embedding_model is not None
        assert embedding_model.model is not None
        assert embedding_model.dimension == 768
    
    def test_encode_single_text(self, embedding_model):
        """Testa encoding de texto único."""
        embedding = embedding_model.encode("teste")
        assert embedding.shape[0] == 768
    
    def test_encode_batch(self, embedding_model):
        """Testa encoding em lote."""
        texts = ["texto 1", "texto 2", "texto 3"]
        embeddings = embedding_model.encode_batch(texts)
        assert embeddings.shape == (3, 768)


if __name__ == "__main__":
    pytest.main(["-v", __file__])