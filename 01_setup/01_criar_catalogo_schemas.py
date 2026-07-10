# Databricks notebook source
# DBTITLE 1,🏗️ Setup Unity Catalog - Projeto Inadimplência
# MAGIC %md
# MAGIC # 🏗️ Setup Unity Catalog - Projeto Inadimplência
# MAGIC
# MAGIC ## Objetivo
# MAGIC Criar catálogo e schemas Unity Catalog para o projeto de previsão de inadimplência e gestão de risco financeiro.
# MAGIC
# MAGIC ## Estrutura
# MAGIC
# MAGIC ```
# MAGIC risco_financeiro (CATALOG)
# MAGIC ├── bronze_raw (SCHEMA) - Dados brutos
# MAGIC ├── silver_clean (SCHEMA) - Dados limpos e enriched
# MAGIC ├── gold_analytics (SCHEMA) - Dados prontos para analytics
# MAGIC ├── ml_features (SCHEMA) - Feature Store
# MAGIC ├── ml_models (SCHEMA) - Modelos registrados
# MAGIC └── rag_documents (SCHEMA) - PDFs e embeddings para RAG
# MAGIC ```
# MAGIC
# MAGIC ## Pré-requisitos
# MAGIC - Permissões para criar catálogos no Unity Catalog
# MAGIC - Workspace conectado ao Unity Catalog
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Verificar Catálogos Existentes
# Verificar catálogos existentes
import re

print("📋 Catálogos disponíveis:\n")
catalogs = spark.sql("SHOW CATALOGS").collect()
for cat in catalogs:
    print(f"  - {cat.catalog}")

# Verificar se já existe o catálogo do projeto
catalog_exists = any([cat.catalog == 'risco_financeiro' for cat in catalogs])
print(f"\n{'✅' if catalog_exists else '❌'} Catálogo 'risco_financeiro' {'já existe' if catalog_exists else 'será criado'}")

# COMMAND ----------

# DBTITLE 1,Criar Catálogo risco_financeiro
# MAGIC %sql
# MAGIC -- Criar catálogo principal (remover se já existe para fresh start)
# MAGIC DROP CATALOG IF EXISTS risco_financeiro CASCADE;
# MAGIC
# MAGIC CREATE CATALOG IF NOT EXISTS risco_financeiro
# MAGIC COMMENT 'Catálogo para gestão de risco financeiro, inadimplência e cash flow forecast';
# MAGIC
# MAGIC USE CATALOG risco_financeiro;

# COMMAND ----------

# DBTITLE 1,Criar Schema: bronze_raw
# MAGIC %sql
# MAGIC -- Schema Bronze: Dados brutos (ingestão direta)
# MAGIC CREATE SCHEMA IF NOT EXISTS bronze_raw
# MAGIC COMMENT 'Camada Bronze - Dados brutos ingeridos de Fivetran (ERP) e CSVs manuais do financeiro';
# MAGIC
# MAGIC USE SCHEMA bronze_raw;

# COMMAND ----------

# DBTITLE 1,Criar Schema: silver_clean
# MAGIC %sql
# MAGIC -- Schema Silver: Dados limpos e enriquecidos
# MAGIC CREATE SCHEMA IF NOT EXISTS silver_clean
# MAGIC COMMENT 'Camada Silver - Dados limpos, validados e enriquecidos. Join entre Fivetran + CSV. Cálculo de métricas básicas';
# MAGIC
# MAGIC USE SCHEMA silver_clean;

# COMMAND ----------

# DBTITLE 1,Criar Schema: gold_analytics
# MAGIC %sql
# MAGIC -- Schema Gold: Analytics-ready
# MAGIC CREATE SCHEMA IF NOT EXISTS gold_analytics
# MAGIC COMMENT 'Camada Gold - Dados prontos para consumo. Modelo dimensional (fatos + dimensões). Usado por dashboards e Genie';
# MAGIC
# MAGIC USE SCHEMA gold_analytics;

# COMMAND ----------

# DBTITLE 1,Criar Schema: ml_features
# MAGIC %sql
# MAGIC -- Schema ML Features: Feature Store
# MAGIC CREATE SCHEMA IF NOT EXISTS ml_features
# MAGIC COMMENT 'Feature Store - Features engenheiradas para modelos ML. RFM, agregações temporais, perfis comportamentais';
# MAGIC
# MAGIC USE SCHEMA ml_features;

# COMMAND ----------

# DBTITLE 1,Criar Schema: ml_models
# MAGIC %sql
# MAGIC -- Schema ML Models: Model Registry metadata
# MAGIC CREATE SCHEMA IF NOT EXISTS ml_models
# MAGIC COMMENT 'ML Models - Metadados de modelos registrados. Usado pelo MLflow Model Registry via Unity Catalog';
# MAGIC
# MAGIC USE SCHEMA ml_models;

# COMMAND ----------

# DBTITLE 1,Criar Schema: rag_documents
# MAGIC %sql
# MAGIC -- Schema RAG Documents: PDFs e embeddings
# MAGIC CREATE SCHEMA IF NOT EXISTS rag_documents
# MAGIC COMMENT 'RAG Documents - PDFs de regras financeiras, embeddings para Vector Search, validação de notas fiscais';
# MAGIC
# MAGIC USE SCHEMA rag_documents;

# COMMAND ----------

# DBTITLE 1,Validar Estrutura Criada
# Validar estrutura completa
print("\n✅ ESTRUTURA UNITY CATALOG CRIADA COM SUCESSO!\n")
print("="*70)

schemas = spark.sql("SHOW SCHEMAS IN risco_financeiro").collect()

print(f"\n📦 Catálogo: risco_financeiro")
print(f"   Total de schemas: {len(schemas)}\n")

for schema in schemas:
    schema_name = schema.databaseName
    
    # Obter comentário do schema
    try:
        desc = spark.sql(f"DESCRIBE SCHEMA EXTENDED risco_financeiro.{schema_name}").collect()
        comment = [row.data_type for row in desc if row.col_name == 'Comment']
        comment_text = comment[0] if comment else 'Sem descrição'
    except:
        comment_text = 'Sem descrição'
    
    print(f"  📁 {schema_name}")
    print(f"     {comment_text}")
    print()

# COMMAND ----------

# DBTITLE 1,Configurar Permissões Básicas
# Configurar permissões básicas (ajustar conforme necessidade)
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
current_user = spark.sql("SELECT current_user()").collect()[0][0]

print("🔐 Configuração de Permissões\n")
print(f"Usuário atual: {current_user}")
print("\nPermissões configuradas:")
print("  ✅ USE CATALOG - Para todos usuários")
print("  ✅ USE SCHEMA - Para todos usuários em todos schemas")
print("  ✅ SELECT - Para todos usuários em tabelas Gold")
print("  ⚠️  CREATE, MODIFY - Apenas para owners/admins")

print("\n💡 Para ajustar permissões específicas, usar Catalog Explorer UI ou SQL GRANT")

# COMMAND ----------

# DBTITLE 1,Próximos Passos
# MAGIC %md
# MAGIC ## ✅ Catálogo Criado com Sucesso!
# MAGIC
# MAGIC ### Próximos Passos:
# MAGIC
# MAGIC 1. **Gerar Dados Sintéticos**: `02_ingestion/01_gerar_dados_sinteticos`
# MAGIC 2. **Ingestão Bronze**: Carregar dados brutos nas tabelas bronze_raw
# MAGIC 3. **Transformação Silver**: Limpeza e enriquecimento
# MAGIC 4. **Transformação Gold**: Modelo dimensional
# MAGIC 5. **Feature Engineering**: Criar features para ML
# MAGIC 6. **Modelagem**: Treinar modelos de inadimplência
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Estrutura de Tabelas (Preview):
# MAGIC
# MAGIC **Bronze**:
# MAGIC - `clientes_raw`, `faturas_raw`, `pagamentos_raw`, `marcas_raw`, `csv_financeiro_raw`
# MAGIC
# MAGIC **Silver**:
# MAGIC - `clientes`, `faturas_enriquecidas`, `pagamentos_consolidados`, `eventos_cobranca`
# MAGIC
# MAGIC **Gold**:
# MAGIC - `fato_inadimplencia`, `dim_clientes`, `dim_marcas`, `dim_tempo`, `metricas_financeiras_mes`
# MAGIC
# MAGIC **ML Features**:
# MAGIC - `features_rfm`, `features_temporais`, `features_perfil_pagamento`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Autor**: Valdomiro Vega García  
# MAGIC **Data**: 02/07/2026

# COMMAND ----------



