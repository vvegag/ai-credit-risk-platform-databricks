# Databricks notebook source
# DBTITLE 1,📚 Exemplo de Uso - RAG Agent Modular
# MAGIC %md
# MAGIC # 📚 Exemplo de Uso do RAG Agent Modular
# MAGIC
# MAGIC Este notebook demonstra como usar o código refatorado em módulos Python.
# MAGIC
# MAGIC ## ✅ Vantagens da Estrutura Modular
# MAGIC - Código reutilizável
# MAGIC - Fácil manutenção
# MAGIC - Testável
# MAGIC - Pronto para produção

# COMMAND ----------

# DBTITLE 1,1️⃣ Importar Módulos
import sys
sys.path.append('/Workspace/Users/valdomirovega@hotmail.com/ai-credit-risk-platform-databricks/10_rag_agent')

from src import RAGAgent, CONFIG

print("✅ Módulos importados com sucesso!")
print(f"\n📊 Configurações:")
print(f"   • Endpoint Vector Search: {CONFIG.vector_search.endpoint_name}")
print(f"   • LLM: {CONFIG.llm.endpoint}")
print(f"   • Embedding Model: {CONFIG.embedding.model_name}")

# COMMAND ----------

# DBTITLE 1,2️⃣ Inicializar RAG Agent
# Criar instância do agente
agent = RAGAgent()

print("✅ RAG Agent inicializado!")
print("\n📦 Componentes:")
print("   • Vector Search: conectado")
print("   • LLM: configurado")
print("   • Chain: pronta")

# COMMAND ----------

# DBTITLE 1,3️⃣ Fazer Perguntas
# Pergunta 1: Score de crédito
resultado1 = agent.query(
    "Quais clientes têm score de crédito alto?",
    top_k=5,
    verbose=True
)

# COMMAND ----------

# DBTITLE 1,✅ Próximos Passos
# MAGIC %md
# MAGIC ## 🚀 Deploy em Produção
# MAGIC
# MAGIC ### Opção 1: Model Serving
# MAGIC ```python
# MAGIC %run ./deploy/model_serving.py
# MAGIC ```
# MAGIC
# MAGIC ### Opção 2: Testes
# MAGIC ```bash
# MAGIC pytest tests/test_rag_agent.py -v
# MAGIC ```
# MAGIC
# MAGIC ### Opção 3: Databricks App
# MAGIC ```bash
# MAGIC databricks apps deploy credit-risk-rag
# MAGIC ```

# COMMAND ----------


