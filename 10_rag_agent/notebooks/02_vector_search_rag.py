# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,🔍 Vector Search + RAG - README
# MAGIC %md
# MAGIC # 🔍 Vector Search + RAG para Documentos de Crédito
# MAGIC
# MAGIC ## 🎯 Objetivo
# MAGIC Implementar busca semântica e RAG (Retrieval-Augmented Generation) sobre os documentos PDF de crédito empresarial.
# MAGIC
# MAGIC ## 📊 Pipeline
# MAGIC 1. **Extração de Texto** - Ler PDFs e extrair conteúdo
# MAGIC 2. **Chunking** - Dividir documentos em chunks semânticos
# MAGIC 3. **Embeddings** - Gerar vetores com sentence-transformers
# MAGIC 4. **Vector Search** - Criar índice vetorial no Databricks
# MAGIC 5. **RAG** - Busca semântica + LLM para respostas
# MAGIC
# MAGIC ## 🗂️ Fonte de Dados
# MAGIC - **Documentos**: `/Volumes/credit_risk/documentos/documentos_credito/`
# MAGIC - **Metadados**: `credit_risk.documentos.metadata_documentos`
# MAGIC - **60 PDFs**: 20 contratos + 20 extratos + 20 comprovantes
# MAGIC
# MAGIC ## 🚀 Resultados Esperados
# MAGIC - Tabela Delta com embeddings: `credit_risk.documentos.embeddings_documentos`
# MAGIC - Índice Vector Search: `credit_risk_docs_index`
# MAGIC - Função RAG para consultas: `buscar_documentos(query)`
# MAGIC
# MAGIC ## 💡 Casos de Uso
# MAGIC - "Encontre contratos aprovados com score > 650"
# MAGIC - "Mostre empresas do setor Tecnologia com receita > 20M"
# MAGIC - "Quais clientes tiveram atrasos nos pagamentos?"
# MAGIC - "Explique por que o Cliente X foi aprovado"

# COMMAND ----------

# DBTITLE 1,📦 Instalar Bibliotecas
# MAGIC %pip install pypdf sentence-transformers langchain langchain-community databricks-vectorsearch --quiet
# MAGIC dbutils.library.restartPython()
# MAGIC
# MAGIC print("✅ Bibliotecas instaladas:")
# MAGIC print("  • pypdf - Extração de texto de PDFs")
# MAGIC print("  • sentence-transformers - Geração de embeddings")
# MAGIC print("  • langchain - Framework RAG")
# MAGIC print("  • databricks-vectorsearch - Cliente Vector Search")

# COMMAND ----------

# DBTITLE 1,📊 Carregar Metadados dos Documentos
# Carregar metadados dos documentos gerados
print("📊 Carregando metadados dos documentos...\n")

df_metadata = spark.table("credit_risk.documentos.metadata_documentos")

print(f"✅ {df_metadata.count()} documentos encontrados\n")

# Mostrar distribuição por tipo
print("📋 Distribuição por tipo:")
df_metadata.groupBy("tipo_documento").count().show()

# Converter para Pandas para facilitar iteração
metadata_pd = df_metadata.toPandas()

print(f"\n📁 Localização base: /Volumes/credit_risk/documentos/documentos_credito/")
print(f"\n✅ Metadados carregados com sucesso!")

# COMMAND ----------

# DBTITLE 1,📄 Função: Extrair Texto de PDF
from pypdf import PdfReader
import re  # Biblioteca para operações com expressões regulares (regex)

def extrair_texto_pdf(pdf_path):
    """
    Extrai texto de um arquivo PDF
    """
    try:
        # Ler PDF
        reader = PdfReader(pdf_path)
        
        # Extrair texto de todas as páginas
        texto_completo = ""
        for page in reader.pages:
            texto_completo += page.extract_text() + "\n"
        
        # Limpar texto
        # Remover múltiplas quebras de linha
        texto_limpo = re.sub(r'\n{3,}', '\n\n', texto_completo)
        # Remover espaços múltiplos
        texto_limpo = re.sub(r' {2,}', ' ', texto_limpo)
        
        return texto_limpo.strip()
    
    except Exception as e:
        print(f"❌ Erro ao ler {pdf_path}: {e}")
        return ""

print("✅ Função extrair_texto_pdf() criada")

# Testar com um documento
print("\n🧪 Testando extração...")
teste_path = "/Volumes/credit_risk/documentos/documentos_credito/contratos/contrato_1.pdf"
texto_teste = extrair_texto_pdf(teste_path)
print(f"\n📝 Primeiros 500 caracteres extraídos:")
print(texto_teste[:500] if texto_teste else "Erro na extração")

# COMMAND ----------

# DBTITLE 1,✂️ Função: Chunking de Texto
def criar_chunks(texto, chunk_size=512, overlap=50):
    """
    Divide texto em chunks com overlap
    """
    if not texto:
        return []
    
    palavras = texto.split()
    chunks = []
    
    i = 0
    while i < len(palavras):
        # Pegar chunk de palavras
        chunk_palavras = palavras[i:i + chunk_size]
        chunk_texto = ' '.join(chunk_palavras)
        
        chunks.append(chunk_texto)
        
        # Avançar com overlap
        i += chunk_size - overlap
    
    return chunks

print("✅ Função criar_chunks() criada")

# Testar chunking
print("\n🧪 Testando chunking...")
chunks_teste = criar_chunks(texto_teste, chunk_size=256, overlap=25)
print(f"\n📦 {len(chunks_teste)} chunks criados")
print(f"\n📝 Primeiro chunk (primeiros 300 caracteres):")
print(chunks_teste[0][:300] if chunks_teste else "Erro no chunking")

# COMMAND ----------

# DBTITLE 1,🤖 Carregar Modelo de Embeddings
from sentence_transformers import SentenceTransformer
import torch
import os
# HF_TOKEN já deve estar configurado como variável de ambiente do cluster/secret scope

print("🤖 Carregando modelo de embeddings...\n")

# Usar modelo multilíngue otimizado para português
# Alternativas: 'paraphrase-multilingual-MiniLM-L12-v2' (mais leve)
#               'distiluse-base-multilingual-cased-v2' (mais rápido)
model_name = "paraphrase-multilingual-mpnet-base-v2"

try:
    model = SentenceTransformer(model_name)
    print(f"✅ Modelo carregado: {model_name}")
    print(f"📊 Dimensão dos embeddings: {model.get_sentence_embedding_dimension()}")
    print(f"💻 Device: {model.device}")
    
    # Testar embedding
    print("\n🧪 Testando geração de embedding...")
    texto_exemplo = "Contrato de crédito empresarial aprovado"
    embedding_exemplo = model.encode(texto_exemplo)
    print(f"✅ Embedding gerado: vetor de {len(embedding_exemplo)} dimensões")
    print(f"📊 Primeiros 10 valores: {embedding_exemplo[:10]}")
    
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    print("\n💡 Sugestão: Use um modelo mais leve como 'paraphrase-multilingual-MiniLM-L12-v2'")

# COMMAND ----------

# DBTITLE 1,🔄 Processar Todos os Documentos
from tqdm import tqdm
import numpy as np

print("🔄 PROCESSANDO TODOS OS DOCUMENTOS\n")
print("="*80)

documentos_processados = []
erros = []

for idx, row in tqdm(metadata_pd.iterrows(), total=len(metadata_pd), desc="Processando PDFs"):
    documento_id = row['documento_id']
    caminho = row['caminho_arquivo']
    tipo_doc = row['tipo_documento']
    id_cliente = row['id_cliente']
    
    try:
        # 1. Extrair texto
        texto = extrair_texto_pdf(caminho)
        
        if not texto:
            erros.append({'documento_id': documento_id, 'erro': 'Texto vazio'})
            continue
        
        # 2. Criar chunks
        chunks = criar_chunks(texto, chunk_size=256, overlap=25)
        
        if not chunks:
            erros.append({'documento_id': documento_id, 'erro': 'Chunks vazios'})
            continue
        
        # 3. Gerar embeddings para cada chunk
        for chunk_idx, chunk in enumerate(chunks):
            # Gerar embedding
            embedding = model.encode(chunk, show_progress_bar=False)
            
            # Armazenar
            documentos_processados.append({
                'documento_id': documento_id,
                'id_cliente': id_cliente,
                'tipo_documento': tipo_doc,
                'chunk_id': f"{documento_id}_chunk_{chunk_idx}",
                'chunk_index': chunk_idx,
                'texto_chunk': chunk,
                'texto_completo': texto if chunk_idx == 0 else None,  # Só no primeiro chunk
                'embedding': embedding.tolist(),
                'caminho_arquivo': caminho
            })
    
    except Exception as e:
        erros.append({'documento_id': documento_id, 'erro': str(e)})

print("\n" + "="*80)
print("\n✅ PROCESSAMENTO CONCLUÍDO!\n")
print(f"📊 Estatísticas:")
print(f"  • Documentos processados: {len(set([d['documento_id'] for d in documentos_processados]))}")
print(f"  • Total de chunks: {len(documentos_processados)}")
print(f"  • Erros: {len(erros)}")

if erros:
    print(f"\n⚠️  Documentos com erro:")
    for erro in erros[:5]:
        print(f"  • {erro['documento_id']}: {erro['erro']}")
    if len(erros) > 5:
        print(f"  ... e mais {len(erros) - 5} erros")

print(f"\n📦 Média de chunks por documento: {len(documentos_processados) / len(set([d['documento_id'] for d in documentos_processados])):.1f}")
print("\n" + "="*80)

# COMMAND ----------

# DBTITLE 1,💾 Criar Tabela Delta com Embeddings
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType
import pandas as pd

print("💾 Criando tabela Delta com embeddings...\n")

# Converter para DataFrame Pandas primeiro
df_embeddings_pd = pd.DataFrame(documentos_processados)

# Remover coluna texto_completo dos chunks não-primeiros (None)
df_embeddings_pd['texto_completo'] = df_embeddings_pd['texto_completo'].fillna('')

print(f"📊 DataFrame criado: {len(df_embeddings_pd)} linhas\n")

# Converter para Spark DataFrame
df_embeddings_spark = spark.createDataFrame(df_embeddings_pd)

# Criar tabela Delta
table_name = "credit_risk.documentos.embeddings_documentos"

df_embeddings_spark.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

print(f"✅ Tabela criada: {table_name}")
print(f"📝 {df_embeddings_spark.count()} chunks com embeddings inseridos\n")

# Mostrar schema
print("📋 Schema da tabela:")
df_embeddings_spark.printSchema()

# Mostrar amostra (sem embedding para não poluir)
print("\n📊 Amostra dos dados:")
spark.table(table_name).select(
    'chunk_id', 'documento_id', 'tipo_documento', 'id_cliente', 'chunk_index'
).show(5, truncate=False)

# COMMAND ----------

# DBTITLE 1,🔍 Setup Databricks Vector Search
from databricks.vector_search.client import VectorSearchClient

print("🔍 CONFIGURANDO DATABRICKS VECTOR SEARCH\n")
print("="*80)

# Inicializar cliente
vsc = VectorSearchClient(disable_notice=True)

print("\n✅ Cliente Vector Search inicializado\n")

# Configurações
endpoint_name = "credit_risk_vector_endpoint"
index_name = "credit_risk.documentos.credit_docs_vector_index"
source_table = "credit_risk.documentos.embeddings_documentos"
embedding_dimension = 768  # Dimensão do modelo mpnet
embedding_column = "embedding"
primary_key = "chunk_id"

print("⚙️  Configurações:")
print(f"  • Endpoint: {endpoint_name}")
print(f"  • Índice: {index_name}")
print(f"  • Tabela fonte: {source_table}")
print(f"  • Dimensão embedding: {embedding_dimension}")
print(f"  • Coluna embedding: {embedding_column}")
print(f"  • Chave primária: {primary_key}")

print("\n" + "="*80)
print("\n💡 NOTA: Criação do endpoint e índice requer permissões de administrador.")
print("   Se você não tem permissões, peça ao admin do workspace para criar.\n")
print("   Comando para criar endpoint:")
print(f"   vsc.create_endpoint(name='{endpoint_name}', endpoint_type='STANDARD')\n")
print("   Comando para criar índice:")
print(f"""   vsc.create_delta_sync_index(
       endpoint_name='{endpoint_name}',
       index_name='{index_name}',
       source_table_name='{source_table}',
       pipeline_type='TRIGGERED',
       primary_key='{primary_key}',
       embedding_dimension={embedding_dimension},
       embedding_vector_column='{embedding_column}'
   )""")
print("\n" + "="*80)

# COMMAND ----------

# DBTITLE 1,🧪 Criar Endpoint e Índice (se tiver permissões)
# ATENÇÃO: Esta célula requer permissões de admin
# Se você não tem permissões, pule para a próxima célula

try:
    print("🔨 Tentando criar endpoint...\n")
    
    # Verificar se endpoint já existe
    try:
        existing_endpoint = vsc.get_endpoint(endpoint_name)
        print(f"✅ Endpoint '{endpoint_name}' já existe!")
        print(f"   Status: {existing_endpoint.get('endpoint_status', 'UNKNOWN')}\n")
    except Exception:
        # Criar endpoint
        print(f"📍 Criando endpoint '{endpoint_name}'...")
        vsc.create_endpoint(
            name=endpoint_name,
            endpoint_type="STANDARD"
        )
        print(f"✅ Endpoint criado! Aguardando provisionamento...\n")
    
    # Verificar se índice já existe
    try:
        existing_index = vsc.get_index(endpoint_name, index_name)
        print(f"✅ Índice '{index_name}' já existe!")
        print(f"   Status: {existing_index.get('status', {}).get('state', 'UNKNOWN')}\n")
    except Exception:
        # Criar índice
        print(f"📊 Criando índice vetorial '{index_name}'...")
        vsc.create_delta_sync_index(
            endpoint_name=endpoint_name,
            index_name=index_name,
            source_table_name=source_table,
            pipeline_type="TRIGGERED",
            primary_key=primary_key,
            embedding_dimension=embedding_dimension,
            embedding_vector_column=embedding_column
        )
        print(f"✅ Índice criado! Sincronização iniciada...\n")
    
    print("="*80)
    print("\n🎉 Setup do Vector Search completo!\n")
    print("💡 O índice pode levar alguns minutos para sincronizar completamente.")
    print("   Verifique o status no Databricks UI: Data > Vector Search\n")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ Erro ao criar endpoint/índice: {e}\n")
    print("💡 Possíveis causas:")
    print("   • Você não tem permissões de administrador")
    print("   • O endpoint/índice já existe com configurações diferentes")
    print("   • Limite de recursos atingido\n")
    print("📋 Solução: Peça ao admin do workspace para criar ou ajustar permissões.\n")
    print("="*80)

# COMMAND ----------

# DBTITLE 1,🔧 Habilitar Change Data Feed e Recriar Índice
print("🔧 CORRIGINDO CONFIGURAÇÃO DA TABELA\n")
print("="*80)

# 1. Habilitar Change Data Feed na tabela
print("\n📊 Habilitando Change Data Feed na tabela...\n")

spark.sql(f"""
    ALTER TABLE {source_table}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

print("✅ Change Data Feed habilitado!\n")

# Verificar propriedades da tabela
print("📋 Propriedades da tabela:")
props = spark.sql(f"SHOW TBLPROPERTIES {source_table}").filter("key = 'delta.enableChangeDataFeed'").collect()
for prop in props:
    print(f"  • {prop['key']} = {prop['value']}")

print("\n" + "="*80)
print("\n🔨 RECRIANDO ÍNDICE VETORIAL\n")
print("="*80)

try:
    # Tentar deletar índice anterior se existir
    try:
        print(f"\n🗑️  Deletando índice anterior se existir...")
        vsc.delete_index(endpoint_name=endpoint_name, index_name=index_name)
        print("✅ Índice anterior deletado\n")
        import time
        time.sleep(5)  # Aguardar alguns segundos
    except Exception as e:
        print(f"ℹ️  Índice não existia ou já foi deletado: {e}\n")
    
    # Criar novo índice
    print(f"📊 Criando índice vetorial '{index_name}'...\n")
    
    vsc.create_delta_sync_index(
        endpoint_name=endpoint_name,
        index_name=index_name,
        source_table_name=source_table,
        pipeline_type="TRIGGERED",
        primary_key=primary_key,
        embedding_dimension=embedding_dimension,
        embedding_vector_column=embedding_column
    )
    
    print("✅ Índice criado com sucesso!\n")
    print("="*80)
    print("\n🎉 VECTOR SEARCH INDEX CONFIGURADO!\n")
    print("💡 O índice está sincronizando. Isso pode levar 1-3 minutos.")
    print("   Você pode verificar o progresso em: Data > Vector Search\n")
    
    # Verificar status do índice
    print("📊 Status do índice:")
    index_info = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
    print(f"  • Estado: {index_info.get('status', {}).get('state', 'UNKNOWN')}")
    print(f"  • Endpoint: {index_info.get('endpoint_name', 'N/A')}")
    print(f"  • Tabela fonte: {index_info.get('delta_sync_index_spec', {}).get('source_table', 'N/A')}")
    
    print("\n" + "="*80)
    print("\n✅ Correção completa! Aguarde a sincronização e teste a busca.\n")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ Erro ao criar índice: {e}\n")
    print("💡 Possíveis soluções:")
    print("   1. Aguarde alguns segundos e execute esta célula novamente")
    print("   2. Verifique se você tem permissões de admin")
    print("   3. Verifique se o endpoint está ativo: vsc.get_endpoint(endpoint_name)\n")
    print("="*80)

# COMMAND ----------

# DBTITLE 1,🔎 Função: Busca Semântica
def buscar_documentos(query, top_k=5, filters=None):
    """
    Busca semântica nos documentos usando Vector Search
    
    Args:
        query: Texto da consulta
        top_k: Número de resultados a retornar
        filters: Filtros SQL adicionais (opcional)
    
    Returns:
        Lista de documentos relevantes com scores
    """
    try:
        # Gerar embedding da query
        query_embedding = model.encode(query)
        
        # Buscar no índice vetorial
        results = vsc.get_index(
            endpoint_name=endpoint_name,
            index_name=index_name
        ).similarity_search(
            query_vector=query_embedding.tolist(),
            columns=["chunk_id", "documento_id", "tipo_documento", "id_cliente", "texto_chunk"],
            num_results=top_k,
            filters=filters
        )
        
        return results
    
    except Exception as e:
        print(f"❌ Erro na busca: {e}")
        return None

print("✅ Função buscar_documentos() criada")
print("\n💡 Uso:")
print("   resultados = buscar_documentos('empresas aprovadas setor tecnologia', top_k=5)")
print("   resultados = buscar_documentos('score alto', top_k=3, filters={'tipo_documento': 'contratos'})")

# COMMAND ----------

# DBTITLE 1,🔍 Verificar Status do Índice
# Testar se o índice está pronto fazendo uma busca simples
print("🔍 Verificando sincronização do índice...\n")

try:
    # Tentar busca de teste
    query_teste = "contrato"
    embedding_teste = model.encode(query_teste)
    
    resultado = vsc.get_index(
        endpoint_name=endpoint_name,
        index_name=index_name
    ).similarity_search(
        query_vector=embedding_teste.tolist(),
        columns=["chunk_id"],
        num_results=1
    )
    
    print("✅ ÍNDICE SINCRONIZADO E PRONTO!\n")
    print("🚀 Pode executar a célula 14 para testar buscas semânticas.\n")
    
except Exception as e:
    if "not ready" in str(e).lower():
        print("⏳ AINDA SINCRONIZANDO...\n")
        print("💡 Aguarde mais 1-2 minutos e execute esta célula novamente.\n")
    else:
        print(f"❌ Erro: {e}\n")

# COMMAND ----------

# DBTITLE 1,🔍 Listar Índices e Reconectar
# Listar todos os índices no endpoint
print("🔍 Listando índices no endpoint...\n")

try:
    endpoint = vsc.get_endpoint(endpoint_name)
    print(f"✅ Endpoint encontrado: {endpoint_name}")
    print(f"   Status: {endpoint.get('endpoint_status', 'UNKNOWN')}\n")
    
    # Tentar listar índices
    print("📋 Tentando buscar o índice...\n")
    
    try:
        index = vsc.get_index(
            endpoint_name=endpoint_name,
            index_name=index_name
        )
        print(f"✅ ÍNDICE ENCONTRADO!")
        print(f"   Nome: {index_name}")
        print(f"\n🚀 Índice está pronto! Execute a célula 14 para testar buscas.\n")
    except Exception as e:
        print(f"❌ Índice não encontrado: {e}\n")
        print("💡 O índice pode ter sido deletado. Execute a célula 11 novamente para recriar.\n")
        
except Exception as e:
    print(f"❌ Erro: {e}\n")

# COMMAND ----------

# DBTITLE 1,🧪 Testar Busca Semântica
print("🧪 TESTANDO BUSCA SEMÂNTICA\n")
print("="*80)

# Queries de teste
test_queries = [
    "contratos aprovados com score de crédito alto",
    "empresas do setor de tecnologia",
    "documentos com receita anual elevada",
    "clientes com saldo bancário positivo"
]

for i, query in enumerate(test_queries, 1):
    print(f"\n🔍 Query {i}: '{query}'")
    print("-" * 80)
    
    try:
        resultados = buscar_documentos(query, top_k=3)
        
        if resultados and 'result' in resultados:
            docs = resultados['result']['data_array']
            
            if docs:
                for j, doc in enumerate(docs, 1):
                    print(f"\n  📄 Resultado {j}:")
                    print(f"     Documento: {doc[1]}")
                    print(f"     Tipo: {doc[2]}")
                    print(f"     Cliente: {doc[3]}")
                    print(f"     Trecho: {doc[4][:150]}...")
            else:
                print("  ℹ️  Nenhum resultado encontrado")
        else:
            print("  ⚠️  Busca retornou vazio - verifique se o índice está sincronizado")
    
    except Exception as e:
        print(f"  ❌ Erro: {e}")
        print("  💡 O índice pode ainda estar sincronizando. Aguarde alguns minutos.")
        break

print("\n" + "="*80)
print("\n✅ Testes de busca concluídos!")
print("\n💡 Para queries customizadas, use: buscar_documentos('sua query aqui')")
print("\n" + "="*80)

# COMMAND ----------

# DBTITLE 1,✅ Resumo Final
print("✅" * 40)
print("\n🎉 VECTOR SEARCH + RAG SETUP COMPLETO!\n")
print("✅" * 40)

print("\n📊 COMPONENTES CRIADOS:\n")
print(f"  1. ✅ Tabela de Embeddings")
print(f"     📍 {table_name}")
print(f"     📦 {len(documentos_processados)} chunks vetorizados")
print(f"     🔢 {embedding_dimension} dimensões")

print(f"\n  2. ✅ Modelo de Embeddings")
print(f"     🤖 {model_name}")
print(f"     🌍 Multilíngue (otimizado para PT-BR)")

print(f"\n  3. ✅ Vector Search")
print(f"     📡 Endpoint: {endpoint_name}")
print(f"     📊 Índice: {index_name}")

print(f"\n  4. ✅ Funções RAG")
print(f"     🔍 buscar_documentos(query, top_k, filters)")

print("\n" + "="*80)
print("\n🚀 PRÓXIMOS PASSOS:\n")
print("  1. 🤖 Criar notebook de RAG Agent com LangChain")
print("     → Integrar LLM (DBRX, GPT-4, etc)")
print("     → Criar chain: Busca + Context + Generate")
print("     → Interface conversacional")

print("\n  2. 📊 Dashboard de Análise")
print("     → Visualizar distribuição de embeddings")
print("     → Métricas de qualidade de busca")
print("     → Análise de clusters de documentos")

print("\n  3. 🔄 Pipeline de Atualização")
print("     → Automatizar extração de novos PDFs")
print("     → Incremental sync do Vector Index")
print("     → Monitoramento de qualidade")

print("\n" + "="*80)
print("\n💡 EXEMPLOS DE USO:\n")
print("  # Busca simples")
print("  resultados = buscar_documentos('empresas aprovadas')\n")
print("  # Busca com filtros")
print("  resultados = buscar_documentos('receita alta', filters={'tipo_documento': 'comprovantes'})\n")
print("  # Top 10 mais relevantes")
print("  resultados = buscar_documentos('score crédito', top_k=10)\n")
print("="*80)

# COMMAND ----------

