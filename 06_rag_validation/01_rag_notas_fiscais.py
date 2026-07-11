# Databricks notebook source
# DBTITLE 1,Sistema RAG - Validação de Notas Fiscais
# MAGIC %md
# MAGIC # 📝 Sistema RAG - Validação de Notas Fiscais (Protótipo)
# MAGIC
# MAGIC **Objetivo**: Automatizar a validação de documentos fiscais usando Retrieval Augmented Generation (RAG).
# MAGIC
# MAGIC ⚠️ **Nota de escopo**: este notebook é um **protótipo/demo** — os documentos fiscais são simulados a partir das
# MAGIC próprias faturas (não há OCR real de PDF), materializados como arquivos `.txt` em disco para simular um
# MAGIC documento físico já ingerido, e depois lidos de volta do disco para gerar embeddings. Não há chunking porque
# MAGIC cada nota fiscal já é um texto curto (um chunk natural). O índice de similaridade usa FAISS local. Em produção,
# MAGIC isso seria substituído por parsing real de PDF/OCR e por **Databricks Vector Search** gerenciado.
# MAGIC
# MAGIC ## Use Cases
# MAGIC * Extrair informações de PDFs de notas fiscais
# MAGIC * Validar consistência com dados do sistema
# MAGIC * Detectar divergências e inconsistências
# MAGIC * Reconciliação automática fatura vs nota fiscal
# MAGIC
# MAGIC ## Componentes
# MAGIC * **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
# MAGIC * **Vector Store**: FAISS local (protótipo) — trocar por Databricks Vector Search em produção
# MAGIC * **Validation**: Regras de negócio vetorizadas (Spark/pandas, sem `.apply(axis=1)`)

# COMMAND ----------

# DBTITLE 1,1️⃣ Setup e Instalação de Dependências
import sys

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers já instalado")
except ImportError:
    print("🔧 Instalando sentence-transformers...")
    %pip install sentence-transformers --quiet
    print("✅ Instalado")

try:
    import faiss
    print("✅ faiss já instalado")
except ImportError:
    print("🔧 Instalando faiss-cpu...")
    %pip install faiss-cpu --quiet
    print("✅ Instalado")

import pandas as pd
import numpy as np
from datetime import datetime
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

print("✅ Todas as bibliotecas carregadas")

# COMMAND ----------

# DBTITLE 1,2️⃣ Selecionar Faturas para Simular Documentos Fiscais
# O join/filtro/limit acontece em Spark; só a amostra final (100 linhas) vira pandas
# para geração de texto e embeddings — as duas etapas seguintes exigem processamento local.
df_faturas_spark = (
    spark.table(f"{CATALOG}.silver.faturas_enriquecidas")
    .join(spark.table(f"{CATALOG}.silver.clientes"), "id_cliente")
    .select("id_fatura", "id_cliente", "nome", "cnpj", "valor_total", "data_emissao", "data_vencimento")
    .limit(100)
)
df_faturas = df_faturas_spark.toPandas()

# COMMAND ----------

# DBTITLE 1,3️⃣ Gerar Documentos Fiscais Sintéticos (simula texto extraído de PDF)
# Pequena variação de valor simulando erros de digitação/arredondamento (comum em conciliação real)
np.random.seed(42)
df_docs = df_faturas.copy()
df_docs['valor_nf'] = df_docs['valor_total'] * np.random.choice(
    [1.0, 0.99, 1.01, 1.02], size=len(df_docs), p=[0.7, 0.1, 0.1, 0.1]
)
df_docs['valor_fatura'] = df_docs['valor_total']
df_docs['data_upload'] = datetime.now()

df_docs['texto_nf'] = df_docs.apply(
    lambda r: (
        f"NOTA FISCAL ELETRÔNICA Nº {r['id_fatura']}\n"
        f"Emitente: AI Credit Risk Platform\n"
        f"Cliente: {r['nome']}\n"
        f"CNPJ/CPF: {r['cnpj']}\n"
        f"Data de Emissão: {r['data_emissao']}\n"
        f"Valor Total: R$ {r['valor_nf']:.2f}\n"
        f"Vencimento: {r['data_vencimento']}"
    ),
    axis=1,
)  # geração de texto por linha é aceitável aqui: só roda 1x sobre uma amostra de 100 docs, não em produção/escala

print(f"✅ {len(df_docs)} documentos fiscais sintéticos gerados")
print("\n📊 Amostra de documento:")
print(df_docs.iloc[0]['texto_nf'])

# COMMAND ----------

# DBTITLE 1,3.5️⃣ Materializar Documentos como Arquivos .txt (simula upload físico)
# Cada nota fiscal vira um arquivo .txt real em disco, simulando o texto já extraído
# de um PDF via OCR/parsing. O passo seguinte lê esses arquivos do disco em vez de usar
# a string em memória — mais próximo do fluxo real de ingestão de documentos.
import os

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = "/Workspace" + "/".join(notebook_path.split("/")[:-2])
docs_dir = f"{repo_root}/02_ingestion/sample_data/notas_fiscais_txt"
os.makedirs(docs_dir, exist_ok=True)

df_docs['arquivo_txt'] = df_docs['id_fatura'].apply(lambda id_fatura: f"{docs_dir}/nota_fiscal_{id_fatura}.txt")
for _, row in df_docs.iterrows():
    with open(row['arquivo_txt'], 'w', encoding='utf-8') as f:
        f.write(row['texto_nf'])

print(f"✅ {len(df_docs)} arquivos .txt gravados em {docs_dir}")

# COMMAND ----------

# DBTITLE 1,4️⃣ Ler Documentos do Disco e Criar Embeddings
from sentence_transformers import SentenceTransformer
import faiss

# Lê o texto de volta do arquivo físico (não da coluna em memória) — simula o ponto de
# entrada real de um pipeline RAG, onde o texto vem de um documento já ingerido/parseado.
textos_docs = []
for arquivo in df_docs['arquivo_txt']:
    with open(arquivo, 'r', encoding='utf-8') as f:
        textos_docs.append(f.read())

print("🔄 Carregando modelo de embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("🔄 Gerando embeddings...")
embeddings = model.encode(textos_docs, show_progress_bar=True)
print(f"✅ Embeddings gerados: {embeddings.shape}")

print("🔄 Criando índice FAISS...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype('float32'))
print(f"✅ Índice criado com {index.ntotal} vetores")

# COMMAND ----------

# DBTITLE 1,5️⃣ Validar Documentos vs Faturas (vetorizado, sem .apply(axis=1))
# Regra 1: tolerância de valor (2% aprovado, 2-5% aprovado com ressalva, >5% rejeitado)
perc_diff = ((df_docs['valor_nf'] - df_docs['valor_fatura']).abs() / df_docs['valor_fatura']) * 100
df_docs['perc_diff_valor'] = perc_diff

# Regra 2: campos obrigatórios presentes no texto (checagem vetorizada com str.contains)
campos_obrigatorios = ['NOTA FISCAL', 'Emitente:', 'Cliente:', 'Valor Total:']
campos_presentes = pd.concat(
    [df_docs['texto_nf'].str.contains(campo, regex=False) for campo in campos_obrigatorios], axis=1
)
df_docs['campos_faltando'] = (~campos_presentes).sum(axis=1)

# Status final, vetorizado com np.select
condicoes = [
    (perc_diff <= 2) & (df_docs['campos_faltando'] == 0),
    (perc_diff <= 5) & (df_docs['campos_faltando'] == 0),
]
resultados = ['APROVADO', 'APROVADO_COM_RESSALVA']
df_docs['status'] = np.select(condicoes, resultados, default='REJEITADO')

criticidade_map = {'APROVADO': 'OK', 'APROVADO_COM_RESSALVA': 'BAIXA', 'REJEITADO': 'ALTA'}
df_docs['criticidade'] = df_docs['status'].map(criticidade_map)
df_docs['issues'] = np.where(
    df_docs['status'] == 'APROVADO', 'Sem issues',
    'Divergência de valor (' + df_docs['perc_diff_valor'].round(2).astype(str) + '%) ou campo obrigatório ausente'
)
df_docs['num_issues'] = (df_docs['status'] != 'APROVADO').astype(int) + (df_docs['campos_faltando'])

print("\n" + "="*70)
print("📊 RESULTADO DA VALIDAÇÃO")
print("="*70)
print(f"\n📄 Total de documentos: {len(df_docs)}")
print("\n📊 Status:")
print(df_docs['status'].value_counts())

# COMMAND ----------

# DBTITLE 1,6️⃣ Busca Semântica (Similarity Search)
query = "Encontre documentos fiscais com valores acima de R$ 50.000"
print(f"🔍 Query: {query}")

query_embedding = model.encode([query])
k = 5
distances, indices = index.search(query_embedding.astype('float32'), k)

print(f"\n🎯 Top {k} documentos mais relevantes:\n")
for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
    doc = df_docs.iloc[idx]
    print(f"{i+1}. Fatura {doc['id_fatura']} - Valor: R$ {doc['valor_fatura']:,.2f} | Score: {1 / (1 + dist):.4f} | Status: {doc['status']}")

# COMMAND ----------

# DBTITLE 1,7️⃣ Salvar Resultados no Gold Layer
df_save = df_docs[[
    'id_fatura', 'id_cliente', 'valor_nf', 'valor_fatura',
    'status', 'criticidade', 'issues', 'num_issues', 'data_upload'
]].copy()

spark_df = spark.createDataFrame(df_save)
spark_df.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.gold.validacao_notas_fiscais")

print(f"✅ Tabela salva: {CATALOG}.gold.validacao_notas_fiscais")

# COMMAND ----------

# DBTITLE 1,8️⃣ Dashboard de Validação
print("="*70)
print("📊 DASHBOARD DE VALIDAÇÃO DE NOTAS FISCAIS")
print("="*70)

status_counts = df_docs['status'].value_counts()
total = len(df_docs)
for status, count in status_counts.items():
    print(f"  {status}: {count} ({count/total*100:.1f}%)")

print("\n⚠️ DOCUMENTOS REJEITADOS:")
rejeitados = df_docs[df_docs['status'] == 'REJEITADO'][['id_fatura', 'id_cliente', 'valor_fatura', 'issues']]
print(rejeitados.head(10) if len(rejeitados) > 0 else "  Nenhum documento rejeitado! 🎉")

print("\n" + "="*70)
print("✅ SISTEMA RAG (PROTÓTIPO) CONFIGURADO E FUNCIONAL")
print("="*70)
print("\n🔍 Próximos passos para produção:")
print("  1. Integrar com OCR/parsing real de PDF")
print("  2. Substituir FAISS local por Databricks Vector Search")
print("  3. Workflow de aprovação manual para ressalvas + alertas automáticos")
print("="*70)
