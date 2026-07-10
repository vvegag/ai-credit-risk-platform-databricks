# Databricks notebook source
# DBTITLE 1,Sistema RAG - Validação de Notas Fiscais
# MAGIC %md
# MAGIC # 📝 Sistema RAG - Validação de Notas Fiscais
# MAGIC
# MAGIC **Objetivo**: Automatizar a validação de documentos fiscais usando Retrieval Augmented Generation (RAG).
# MAGIC
# MAGIC ## Use Cases
# MAGIC * Extrair informações de PDFs de notas fiscais
# MAGIC * Validar consistência com dados do sistema
# MAGIC * Detectar fraudes e inconsistências
# MAGIC * Reconciliação automática fatura vs nota fiscal
# MAGIC * Compliance e auditoria
# MAGIC
# MAGIC ## Arquitetura RAG
# MAGIC ```
# MAGIC PDF Upload → OCR/Parse → Embeddings → Vector Index → Similarity Search
# MAGIC                                                 ↓
# MAGIC                                           Validação & Alerta
# MAGIC ```
# MAGIC
# MAGIC ## Componentes
# MAGIC * **Document Parsing**: PDF → Text (PyPDF2, pdfplumber)
# MAGIC * **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
# MAGIC * **Vector Store**: Simulado com FAISS (em produção: Databricks Vector Search)
# MAGIC * **Validation**: Regras de negócio + Similarity matching

# COMMAND ----------

# DBTITLE 1,1️⃣ Setup e Instalação de Dependências
# Instalar bibliotecas necessárias
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
import json
import re
from datetime import datetime
from pyspark.sql import functions as F

print("✅ Todas as bibliotecas carregadas")

# COMMAND ----------

# DBTITLE 1,2️⃣ Gerar Documentos Fiscais Sintéticos
# Criar tabela de documentos fiscais sintéticos
# Em produção, esses documentos viriam de PDFs parseados

# Buscar faturas
df_faturas = spark.sql("""
  SELECT 
    f.fatura_id,
    f.cliente_id,
    c.nome_cliente,
    c.cnpj_cpf,
    f.valor_total,
    f.data_emissao,
    f.data_vencimento,
    m.nome_marca
  FROM workspace.risco_silver.faturas_enriquecidas f
  JOIN workspace.risco_bronze.clientes_raw c ON f.cliente_id = c.cliente_id
  JOIN workspace.risco_bronze.marcas_raw m ON f.marca_id = m.marca_id
  LIMIT 100
""").toPandas()

# Simular texto extraído de notas fiscais (formato de NF-e XML simplificado)
documentos = []

for _, row in df_faturas.iterrows():
    # Simular pequenas variações nos valores (erros de digitação, arredondamentos)
    valor_nf = row['valor_total'] * np.random.choice([1.0, 0.99, 1.01, 1.02], p=[0.7, 0.1, 0.1, 0.1])
    
    texto_nf = f"""
    NOTA FISCAL ELETRÔNICA Nº {row['fatura_id']}
    Emitente: {row['nome_marca']}
    Cliente: {row['nome_cliente']}
    CNPJ/CPF: {row['cnpj_cpf']}
    Data de Emissão: {row['data_emissao']}
    Valor Total: R$ {valor_nf:.2f}
    Descrição: Prestação de serviços conforme contrato
    Vencimento: {row['data_vencimento']}
    """
    
    documentos.append({
        'fatura_id': row['fatura_id'],
        'cliente_id': row['cliente_id'],
        'texto_nf': texto_nf.strip(),
        'valor_nf': valor_nf,
        'valor_fatura': row['valor_total'],
        'data_upload': datetime.now(),
        'status_validacao': None
    })

df_docs = pd.DataFrame(documentos)

print(f"✅ {len(df_docs)} documentos fiscais sintéticos gerados")
print(f"\n📊 Amostra de documento:")
print(df_docs.iloc[0]['texto_nf'])

# COMMAND ----------

# DBTITLE 1,3️⃣ Criar Embeddings dos Documentos
from sentence_transformers import SentenceTransformer
import faiss

# Carregar modelo de embeddings
print("🔄 Carregando modelo de embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Modelo carregado")

# Gerar embeddings
print("🔄 Gerando embeddings...")
textos = df_docs['texto_nf'].tolist()
embeddings = model.encode(textos, show_progress_bar=True)

print(f"✅ Embeddings gerados: {embeddings.shape}")
print(f"  Dimensão do embedding: {embeddings.shape[1]}")

# Criar índice FAISS para busca de similaridade
print("🔄 Criando índice FAISS...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype('float32'))
print(f"✅ Índice criado com {index.ntotal} vetores")

# COMMAND ----------

# DBTITLE 1,4️⃣ Validar Documentos vs Faturas
# Regras de validação
def validar_documento(row):
    """
    Valida se documento fiscal está consistente com fatura do sistema
    """
    issues = []
    
    # 1. Tolerância de valor: 2%
    diff_valor = abs(row['valor_nf'] - row['valor_fatura'])
    perc_diff = (diff_valor / row['valor_fatura']) * 100
    
    if perc_diff > 2:
        issues.append(f"Divergência de valor: NF R$ {row['valor_nf']:.2f} vs Fatura R$ {row['valor_fatura']:.2f} ({perc_diff:.2f}%)")
    
    # 2. Extrair CNPJ/CPF do texto
    cnpj_match = re.search(r'CNPJ/CPF: ([\d\.\/-]+)', row['texto_nf'])
    if not cnpj_match:
        issues.append("CNPJ/CPF não encontrado no documento")
    
    # 3. Verificar campos obrigatórios
    campos_obrigatorios = ['NOTA FISCAL', 'Emitente:', 'Cliente:', 'Valor Total:']
    for campo in campos_obrigatorios:
        if campo not in row['texto_nf']:
            issues.append(f"Campo obrigatório ausente: {campo}")
    
    # Definir status
    if len(issues) == 0:
        status = 'APROVADO'
        criticidade = 'OK'
    elif len(issues) == 1 and 'Divergência de valor' in issues[0] and perc_diff <= 5:
        status = 'APROVADO_COM_RESSALVA'
        criticidade = 'BAIXA'
    else:
        status = 'REJEITADO'
        criticidade = 'ALTA'
    
    return {
        'status': status,
        'criticidade': criticidade,
        'issues': ' | '.join(issues) if issues else 'Sem issues',
        'num_issues': len(issues)
    }

# Aplicar validação
print("🔄 Validando documentos...")
validacoes = df_docs.apply(validar_documento, axis=1, result_type='expand')
df_docs = pd.concat([df_docs, validacoes], axis=1)

print("\n" + "="*70)
print("📊 RESULTADO DA VALIDAÇÃO")
print("="*70)
print(f"\n📄 Total de documentos: {len(df_docs)}")
print(f"\n📊 Status:")
print(df_docs['status'].value_counts())
print(f"\n⚠️ Criticidade:")
print(df_docs['criticidade'].value_counts())
print("\n" + "="*70)

# COMMAND ----------

# DBTITLE 1,5️⃣ Busca Semântica (Similarity Search)
# Exemplo de busca semântica
# Usuário faz uma pergunta em linguagem natural

query = "Encontre documentos fiscais com valores acima de R$ 50.000"

print(f"🔍 Query: {query}")

# Gerar embedding da query
query_embedding = model.encode([query])

# Buscar documentos similares
k = 5  # Top 5 resultados
distances, indices = index.search(query_embedding.astype('float32'), k)

print(f"\n🎯 Top {k} documentos mais relevantes:\n")
for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
    doc = df_docs.iloc[idx]
    print(f"{i+1}. Fatura {doc['fatura_id']} - Valor: R$ {doc['valor_fatura']:,.2f}")
    print(f"   Score de similaridade: {1 / (1 + dist):.4f}")
    print(f"   Status: {doc['status']}")
    print()

# COMMAND ----------

# DBTITLE 1,6️⃣ Salvar Resultados no Gold Layer
# Preparar dados para salvar
df_save = df_docs[[
    'fatura_id', 'cliente_id', 'valor_nf', 'valor_fatura', 
    'status', 'criticidade', 'issues', 'num_issues', 'data_upload'
]].copy()

# Converter para Spark
spark_df = spark.createDataFrame(df_save)
spark_df.write.mode("overwrite").saveAsTable("workspace.risco_rag_documents.validacao_notas_fiscais")

print("✅ Tabela salva: workspace.risco_rag_documents.validacao_notas_fiscais")

# Criar schema se não existir
spark.sql("""
  CREATE SCHEMA IF NOT EXISTS workspace.risco_rag_documents
  COMMENT 'Documentos e validações RAG'
""")

print("\n💾 DADOS SALVOS COM SUCESSO")

# COMMAND ----------

# DBTITLE 1,7️⃣ Dashboard de Validação
print("="*70)
print("📊 DASHBOARD DE VALIDAÇÃO DE NOTAS FISCAIS")
print("="*70)

print("\n📄 RESUMO GERAL:")
print(f"  Total de Documentos: {len(df_docs)}")
print(f"  Aprovados: {len(df_docs[df_docs['status'] == 'APROVADO'])} ({len(df_docs[df_docs['status'] == 'APROVADO'])/len(df_docs)*100:.1f}%)")
print(f"  Aprovados com Ressalva: {len(df_docs[df_docs['status'] == 'APROVADO_COM_RESSALVA'])} ({len(df_docs[df_docs['status'] == 'APROVADO_COM_RESSALVA'])/len(df_docs)*100:.1f}%)")
print(f"  Rejeitados: {len(df_docs[df_docs['status'] == 'REJEITADO'])} ({len(df_docs[df_docs['status'] == 'REJEITADO'])/len(df_docs)*100:.1f}%)")

print("\n⚠️ DOCUMENTOS REJEITADOS:")
rejeitados = df_docs[df_docs['status'] == 'REJEITADO'][['fatura_id', 'cliente_id', 'valor_fatura', 'issues']]
if len(rejeitados) > 0:
    print(rejeitados.head(10))
else:
    print("  Nenhum documento rejeitado! 🎉")

print("\n📊 PRINCIPAIS ISSUES:")
top_issues = df_docs[df_docs['num_issues'] > 0].nlargest(5, 'num_issues')[['fatura_id', 'num_issues', 'issues']]
if len(top_issues) > 0:
    print(top_issues)
else:
    print("  Nenhum issue encontrado!")

print("\n💰 TOTAL FINANCEIRO:")
print(f"  Valor Total Aprovado: R$ {df_docs[df_docs['status'] == 'APROVADO']['valor_fatura'].sum():,.2f}")
print(f"  Valor Total Rejeitado: R$ {df_docs[df_docs['status'] == 'REJEITADO']['valor_fatura'].sum():,.2f}")

print("\n" + "="*70)
print("✅ SISTEMA RAG CONFIGURADO E FUNCIONAL")
print("="*70)
print("\n🔍 Próximos passos:")
print("  1. Integrar com OCR para PDFs reais")
print("  2. Conectar ao Databricks Vector Search (produção)")
print("  3. Adicionar workflow de aprovação manual para ressalvas")
print("  4. Alertas automáticos para documentos rejeitados")
print("="*70)

# COMMAND ----------



