# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,🤖 RAG Agent - README
# MAGIC %md
# MAGIC # 🤖 RAG Agent com LangChain - Análise de Crédito
# MAGIC
# MAGIC ## 🎯 Objetivo
# MAGIC Agente conversacional RAG (Retrieval-Augmented Generation) para análise inteligente de documentos de crédito empresarial.
# MAGIC
# MAGIC ## 🏗️ Arquitetura
# MAGIC ```
# MAGIC Pergunta do Usuário
# MAGIC       ↓
# MAGIC 1. Busca Semântica (Vector Search)
# MAGIC       ↓
# MAGIC 2. Recuperação de Documentos Relevantes
# MAGIC       ↓
# MAGIC 3. Construção de Contexto
# MAGIC       ↓
# MAGIC 4. LLM Generation (DBRX)
# MAGIC       ↓
# MAGIC Resposta Fundamentada
# MAGIC ```
# MAGIC
# MAGIC ## 🔌 Componentes
# MAGIC - **LLM**: Databricks DBRX Instruct
# MAGIC - **Vector Store**: Databricks Vector Search
# MAGIC - **Framework**: LangChain
# MAGIC - **Embeddings**: sentence-transformers (multilíngue)
# MAGIC
# MAGIC ## 💡 Casos de Uso
# MAGIC - "Quais clientes têm score de crédito acima de 700?"
# MAGIC - "Me mostre empresas do setor Tecnologia com receita > 30M"
# MAGIC - "Existe algum contrato com valor acima de 5 milhões?"
# MAGIC - "Analise o perfil de risco do Cliente 15"
# MAGIC - "Quais documentos indicam inadimplência?"

# COMMAND ----------

# DBTITLE 1,📦 Instalar Bibliotecas
# MAGIC %pip install sentence-transformers langchain langchain-databricks databricks-vectorsearch mlflow --quiet
# MAGIC dbutils.library.restartPython()
# MAGIC
# MAGIC print("✅ Bibliotecas instaladas:")
# MAGIC print("  • langchain - Framework RAG")
# MAGIC print("  • databricks-vectorsearch - Vector Search")
# MAGIC print("  • mlflow - LLM tracking")

# COMMAND ----------

# DBTITLE 1,🔧 Configuração Inicial
from sentence_transformers import SentenceTransformer
from databricks.vector_search.client import VectorSearchClient
import os

dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
catalog = dbutils.widgets.get("catalog")

print("🔧 Configurando componentes...\n")

# HF_TOKEN já deve estar configurado como variável de ambiente do cluster/secret scope

# Carregar modelo de embeddings
print("🤖 Carregando modelo de embeddings...")
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
print(f"✅ Modelo carregado: {model.get_sentence_embedding_dimension()} dimensões\n")

# Cliente Vector Search
print("🔍 Conectando ao Vector Search...")
vsc = VectorSearchClient(disable_notice=True)

# Configurações
endpoint_name = "credit_risk_vector_endpoint"
index_name = f"{catalog}.documentos.credit_docs_vector_index"

print(f"✅ Vector Search conectado")
print(f"   Endpoint: {endpoint_name}")
print(f"   Índice: {index_name}")

print("\n✅ Configuração completa!")

# COMMAND ----------

# DBTITLE 1,🔍 Função de Busca Vetorial
def buscar_documentos_relevantes(query, top_k=5):
    """
    Busca documentos relevantes usando Vector Search
    """
    try:
        # Gerar embedding da query
        query_embedding = model.encode(query)
        
        # Buscar no índice
        results = vsc.get_index(
            endpoint_name=endpoint_name,
            index_name=index_name
        ).similarity_search(
            query_vector=query_embedding.tolist(),
            columns=["chunk_id", "documento_id", "tipo_documento", "id_cliente", "texto_chunk"],
            num_results=top_k
        )
        
        # Extrair documentos
        if results and 'result' in results:
            docs = results['result']['data_array']
            
            documentos = []
            for doc in docs:
                documentos.append({
                    'chunk_id': doc[0],
                    'documento_id': doc[1],
                    'tipo_documento': doc[2],
                    'id_cliente': doc[3],
                    'texto': doc[4]
                })
            
            return documentos
        
        return []
    
    except Exception as e:
        print(f"❌ Erro na busca: {e}")
        return []

print("✅ Função buscar_documentos_relevantes() criada")

# Testar busca
print("\n🧪 Testando busca...")
test_docs = buscar_documentos_relevantes("empresas de tecnologia", top_k=2)
print(f"\n✅ {len(test_docs)} documentos encontrados")
if test_docs:
    print(f"   Exemplo: {test_docs[0]['documento_id']} - {test_docs[0]['tipo_documento']}")

# COMMAND ----------

# DBTITLE 1,🤖 Configurar LLM (DBRX)
from langchain_databricks import ChatDatabricks
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

print("🤖 Configurando LLM Databricks DBRX...\n")

# Configurar LLM
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    temperature=0.1,
    max_tokens=1000
)

print("✅ LLM configurado: databricks-meta-llama-3-3-70b-instruct")
print("   Temperature: 0.1 (respostas mais precisas)")
print("   Max tokens: 1000")

# Testar LLM
print("\n🧪 Testando LLM...")
test_response = llm.invoke("Diga olá em português de forma profissional.")
print(f"\n📝 Resposta teste: {test_response.content[:100]}...")
print("\n✅ LLM funcionando!")

# COMMAND ----------

# DBTITLE 1,📝 Template do Prompt RAG
# Template para o prompt RAG
rag_template = """Você é um analista de crédito especializado em análise de documentos empresariais.

Sua tarefa é responder a pergunta do usuário baseando-se EXCLUSIVAMENTE nos documentos fornecidos abaixo.

**REGRAS IMPORTANTES:**
1. Use APENAS as informações presentes nos documentos
2. Se a informação não estiver nos documentos, diga "Não encontrei essa informação nos documentos disponíveis"
3. Cite sempre os documentos que você usou (ex: "Segundo o contrato_5...")
4. Seja preciso e objetivo
5. Use dados numéricos quando disponíveis
6. Responda em português brasileiro

**DOCUMENTOS RELEVANTES:**
{context}

**PERGUNTA DO USUÁRIO:**
{question}

**SUA RESPOSTA:**
"""

rag_prompt = PromptTemplate(
    template=rag_template,
    input_variables=["context", "question"]
)

print("✅ Template RAG criado")
print("\n📋 Variáveis do prompt:")
print("   • context: Documentos recuperados")
print("   • question: Pergunta do usuário")

# COMMAND ----------

# DBTITLE 1,🔗 Criar Chain RAG
# Criar chain RAG (sintaxe moderna do LangChain)
rag_chain = rag_prompt | llm

print("✅ Chain RAG criada (usando RunnableSequence)")
print("\n🔄 Fluxo:")
print("   1. Usuário faz pergunta")
print("   2. Sistema busca documentos relevantes")
print("   3. Documentos são formatados como contexto")
print("   4. LLM gera resposta baseada no contexto")
print("   5. Resposta é retornada ao usuário")

# COMMAND ----------

# DBTITLE 1,💬 Função Principal: RAG Agent
def rag_agent(pergunta, top_k=5, verbose=True):
    """
    Agente RAG completo: Busca + Context + Generate
    
    Args:
        pergunta: Pergunta do usuário
        top_k: Número de documentos a recuperar
        verbose: Mostrar logs detalhados
    
    Returns:
        dict com resposta e metadados
    """
    if verbose:
        print("🤖 RAG AGENT INICIADO\n")
        print("="*80)
        print(f"\n❓ Pergunta: {pergunta}\n")
    
    # 1. BUSCA VETORIAL
    if verbose:
        print("🔍 Etapa 1: Buscando documentos relevantes...")
    
    documentos = buscar_documentos_relevantes(pergunta, top_k=top_k)
    
    if not documentos:
        return {
            'resposta': "Não encontrei documentos relevantes para responder sua pergunta.",
            'documentos': [],
            'erro': 'Nenhum documento encontrado'
        }
    
    if verbose:
        print(f"✅ {len(documentos)} documentos recuperados\n")
    
    # 2. CONSTRUIR CONTEXTO
    if verbose:
        print("📄 Etapa 2: Construindo contexto...")
    
    context_parts = []
    for i, doc in enumerate(documentos, 1):
        context_parts.append(
            f"[DOCUMENTO {i}]\n"
            f"ID: {doc['documento_id']}\n"
            f"Tipo: {doc['tipo_documento']}\n"
            f"Cliente: {doc['id_cliente']}\n"
            f"Conteúdo:\n{doc['texto']}\n"
        )
    
    context = "\n" + "-"*80 + "\n".join(context_parts)
    
    if verbose:
        print(f"✅ Contexto montado ({len(context)} caracteres)\n")
    
    # 3. GERAR RESPOSTA
    if verbose:
        print("🤖 Etapa 3: Gerando resposta com LLM...\n")
    
    try:
        resposta = rag_chain.invoke({
            "context": context,
            "question": pergunta
        }).content
        
        if verbose:
            print("="*80)
            print("\n✅ RESPOSTA GERADA:\n")
            print(resposta)
            print("\n" + "="*80)
            print("\n📚 Documentos utilizados:")
            for doc in documentos:
                print(f"   • {doc['documento_id']} ({doc['tipo_documento']}) - Cliente {doc['id_cliente']}")
            print("\n" + "="*80)
        
        return {
            'resposta': resposta,
            'documentos': documentos,
            'num_documentos': len(documentos),
            'pergunta': pergunta
        }
    
    except Exception as e:
        print(f"\n❌ Erro ao gerar resposta: {e}")
        return {
            'resposta': f"Erro ao gerar resposta: {e}",
            'documentos': documentos,
            'erro': str(e)
        }

print("✅ Função rag_agent() criada")
print("\n💡 Uso:")
print("   resultado = rag_agent('Quais empresas têm score acima de 700?')")

# COMMAND ----------

# DBTITLE 1,🧪 Teste 1: Análise de Score de Crédito
# Teste 1: Buscar empresas por score
resultado1 = rag_agent(
    "Quais clientes têm score de crédito alto? Me dê detalhes sobre eles.",
    top_k=5
)

# COMMAND ----------

# DBTITLE 1,🧪 Teste 2: Análise Setorial
# Teste 2: Análise por setor
resultado2 = rag_agent(
    "Mostre informações sobre empresas do setor de Tecnologia. Qual a receita delas?",
    top_k=5
)

# COMMAND ----------

# DBTITLE 1,🧪 Teste 3: Análise de Contratos
# Teste 3: Análise de contratos
resultado3 = rag_agent(
    "Existem contratos de crédito aprovados? Quais os valores e condições?",
    top_k=5
)

# COMMAND ----------

# DBTITLE 1,🧪 Teste 4: Análise Financeira
# Teste 4: Situação financeira
resultado4 = rag_agent(
    "Quais clientes apresentam saldo bancário positivo nos extratos?",
    top_k=5
)

# COMMAND ----------

# DBTITLE 1,💬 Interface Conversacional Interativa
def chat_interface():
    """
    Interface de chat interativa
    """
    print("\n" + "="*80)
    print("🤖 ASSISTENTE RAG DE ANÁLISE DE CRÉDITO")
    print("="*80)
    print("\n💡 Faça perguntas sobre os documentos de crédito!")
    print("\n📚 Exemplos:")
    print("   • Quais clientes têm receita anual acima de 50 milhões?")
    print("   • Me mostre contratos aprovados com prazo de 36 meses")
    print("   • Analise o perfil de risco do Cliente 5")
    print("   • Quais empresas do setor Varejo têm score baixo?")
    print("\n❌ Digite 'sair' para encerrar\n")
    print("="*80 + "\n")
    
    historico = []
    
    while True:
        # Receber pergunta
        pergunta = input("\n🙋 Você: ").strip()
        
        if pergunta.lower() in ['sair', 'exit', 'quit']:
            print("\n👋 Encerrando assistente. Até logo!\n")
            break
        
        if not pergunta:
            print("⚠️  Digite uma pergunta válida.")
            continue
        
        # Processar com RAG
        resultado = rag_agent(pergunta, top_k=5, verbose=False)
        
        # Mostrar resposta
        print(f"\n🤖 Assistente:\n")
        print(resultado['resposta'])
        
        # Mostrar fontes
        if 'documentos' in resultado and resultado['documentos']:
            print(f"\n📚 Fontes consultadas: {resultado['num_documentos']} documentos")
        
        # Adicionar ao histórico
        historico.append({
            'pergunta': pergunta,
            'resposta': resultado['resposta'],
            'num_docs': resultado.get('num_documentos', 0)
        })
    
    return historico

print("✅ Interface chat_interface() criada")
print("\n💡 Para iniciar o chat interativo, execute:")
print("   historico = chat_interface()")

# COMMAND ----------

# DBTITLE 1,✅ Resumo Final e Próximos Passos
print("="*80)
print("\n🎉 RAG AGENT CONFIGURADO E TESTADO!\n")
print("="*80)

print("\n📊 COMPONENTES ATIVOS:\n")
print("  1. ✅ Vector Search")
print("     🔍 Busca semântica em 60 documentos de crédito")

print("\n  2. ✅ LLM")
print("     🤖 Databricks DBRX Instruct (Foundation Model)")

print("\n  3. ✅ Chain RAG")
print("     🔗 LangChain: Retrieval → Context → Generation")

print("\n  4. ✅ Funções Disponíveis")
print("     💬 rag_agent(pergunta, top_k)")
print("     💬 chat_interface() - modo interativo")

print("\n" + "="*80)
print("\n🚀 MODO DE USO:\n")
print("  # Pergunta única")
print("  resultado = rag_agent('Sua pergunta aqui')")
print("\n  # Chat interativo")
print("  historico = chat_interface()")

print("\n" + "="*80)
print("\n🎯 PRÓXIMOS PASSOS SUGERIDOS:\n")
print("  1. 📊 Dashboard de Análise")
print("     → Visualizar métricas de crédito")
print("     → Análise de risco por setor")
print("     → Distribuição de scores")

print("\n  2. 🔄 Pipeline Automatizado")
print("     → Ingestão contínua de novos documentos")
print("     → Atualização automática do Vector Index")
print("     → Monitoramento de qualidade")

print("\n  3. 🤖 Agente Multi-Ferramentas")
print("     → Integrar com SQL para queries estruturadas")
print("     → Adicionar cálculos de risco em tempo real")
print("     → Gerar relatórios automatizados")

print("\n  4. 🌐 Deploy como API")
print("     → Criar endpoint REST com Model Serving")
print("     → Interface web com Databricks Apps")
print("     → Autenticação e controle de acesso")

print("\n" + "="*80)
print("\n💡 EXEMPLO RÁPIDO:\n")
print(">>> resultado = rag_agent('Quais clientes têm receita > 50M?')")
print(">>> print(resultado['resposta'])")
print("\n" + "="*80)

# COMMAND ----------

