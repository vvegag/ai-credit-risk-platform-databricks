"""Testes leves pra `10_rag_agent/src/config.py`, Fase G do roadmap técnico.

`config.py` não depende de Spark/dbutils/workspace real — só de `os.environ` e
`dataclasses`. Mas importá-lo como `from src.config import CONFIG` executaria
`10_rag_agent/src/__init__.py`, que importa `RAGAgent`/`VectorSearch`/`EmbeddingModel` e, com
isso, `langchain_databricks`, `databricks.vector_search` e `sentence_transformers` —
dependências pesadas que não existem fora de um ambiente com esse workspace configurado.
Carregamos o módulo direto do arquivo, no mesmo espírito do `load_notebook_functions` de
`tests/conftest.py` (Fase C): isolar só o que precisamos, sem puxar a cadeia de imports
pesada do pacote.
"""
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent.parent / "src" / "config.py"


def _load_config_module():
    spec = importlib.util.spec_from_file_location("rag_config_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vector_search_config_tem_defaults_esperados():
    config = _load_config_module()
    vs = config.VectorSearchConfig()

    assert vs.endpoint_name == "credit_risk_vector_endpoint"
    assert vs.index_name == "credit_risk.documentos.credit_docs_vector_index"
    assert vs.source_table == "credit_risk.documentos.embeddings_documentos"
    assert vs.embedding_dimension == 768


def test_embedding_config_usa_hf_token_da_variavel_de_ambiente(monkeypatch):
    config = _load_config_module()
    monkeypatch.setenv("HF_TOKEN", "token-de-teste")

    embedding = config.EmbeddingConfig()

    assert embedding.hf_token == "token-de-teste"


def test_embedding_config_hf_token_explicito_nao_e_sobrescrito_pelo_ambiente(monkeypatch):
    config = _load_config_module()
    monkeypatch.setenv("HF_TOKEN", "token-do-ambiente")

    embedding = config.EmbeddingConfig(hf_token="token-explicito")

    assert embedding.hf_token == "token-explicito"


def test_embedding_config_sem_variavel_de_ambiente_fica_none(monkeypatch):
    config = _load_config_module()
    monkeypatch.delenv("HF_TOKEN", raising=False)

    embedding = config.EmbeddingConfig()

    assert embedding.hf_token is None


def test_llm_config_tem_defaults_esperados():
    config = _load_config_module()
    llm = config.LLMConfig()

    assert llm.endpoint == "databricks-meta-llama-3-3-70b-instruct"
    assert llm.temperature == 0.1
    assert llm.max_tokens == 1000


def test_rag_config_preenche_sub_configs_quando_nao_informadas():
    config = _load_config_module()
    rag = config.RAGConfig()

    assert isinstance(rag.vector_search, config.VectorSearchConfig)
    assert isinstance(rag.embedding, config.EmbeddingConfig)
    assert isinstance(rag.llm, config.LLMConfig)
    assert rag.top_k_documents == 5


def test_config_global_e_uma_instancia_de_ragconfig_valida():
    config = _load_config_module()

    assert isinstance(config.CONFIG, config.RAGConfig)
    assert isinstance(config.CONFIG.vector_search, config.VectorSearchConfig)
