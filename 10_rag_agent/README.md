# 🤖 RAG Agent - Análise Inteligente de Crédito

> Parte do projeto **AI Credit Risk Platform** - Módulo 10

## 📋 Visão Geral
Sistema RAG (Retrieval-Augmented Generation) para análise inteligente de documentos de crédito empresarial usando Databricks Vector Search e LLMs.

## 🏗️ Arquitetura
```
Pergunta → Vector Search → Contexto → LLM → Resposta Fundamentada
```

## 🔧 Tecnologias
- **LLM**: Databricks Meta Llama 3.3 70B
- **Embeddings**: sentence-transformers (multilíngue)
- **Vector Store**: Databricks Vector Search
- **Framework**: LangChain
- **Storage**: Delta Lake + Unity Catalog

## 📁 Estrutura do Módulo
```
10_rag_agent/
├── src/                      # Código fonte modular
│   ├── config.py            # Configurações centralizadas
│   ├── embeddings.py        # Geração de embeddings
│   ├── vector_search.py     # Busca vetorial
│   └── rag_agent.py         # Agente RAG principal
├── tests/                    # Testes automatizados
│   └── test_rag_agent.py
├── deploy/                   # Scripts de deployment
│   └── model_serving.py     # Deploy Model Serving
├── example_usage.py         # Notebook demonstrativo
└── requirements.txt          # Dependências
```

## 🚀 Instalação
```bash
pip install -r requirements.txt
```

## 💡 Uso Rápido
```python
import sys
sys.path.append('/Workspace/Users/<seu_usuario>/ai-credit-risk-platform-databricks/10_rag_agent')

from src import RAGAgent

# Inicializar agente
agent = RAGAgent()

# Fazer pergunta
resultado = agent.query("Quais empresas têm score > 700?")
print(resultado['resposta'])
```

## 📊 Dados
- **Tabela Bronze**: `credit_risk.bronze.clientes`
- **Documentos**: `/Volumes/credit_risk/documentos/documentos_credito/`
- **Embeddings**: `credit_risk.documentos.embeddings_documentos`
- **Vector Index**: `credit_risk.documentos.credit_docs_vector_index`

## 🧪 Testes
```bash
pytest tests/test_rag_agent.py -v
```

## 🌐 Deploy

### Opção 1: Model Serving
```python
python deploy/model_serving.py
```

### Opção 2: Databricks Apps
```bash
databricks apps deploy credit-risk-rag
```

## 📈 Performance
- Latência média: ~2-3s por query
- 60 documentos vetorizados
- 768 dimensões (mpnet)
- Top-5 retrieval

## 🔐 Segurança
- Autenticação workspace
- Secrets Manager para tokens
- Unity Catalog para governança

## 🔗 Projeto Completo
Este módulo faz parte do **AI Credit Risk Platform**:
- `01_setup/` - Configuração inicial
- `02_ingestion/` - Ingestão de dados
- `03_feature_engineering/` - Feature engineering
- `04_modeling/` - Modelagem ML
- `05_mlops/` - MLOps e monitoramento
- `05_dashboard/` - Dashboards execução
- `06_rag_validation/` - Validação RAG
- `07_monitoring/` - Monitoring
- `08_dashboards/` - Dashboards analíticos
- `09_docs/` - Documentação
- **`10_rag_agent/`** - RAG Agent (você está aqui)

## 👥 Autor
Valdomiro Vega García - Databricks

## 📝 Licença
Proprietário