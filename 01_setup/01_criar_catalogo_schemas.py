# Databricks notebook source
# DBTITLE 1,🏗️ Setup Unity Catalog - AI Credit Risk Platform
# MAGIC %md
# MAGIC # 🏗️ Setup Unity Catalog - AI Credit Risk Platform
# MAGIC
# MAGIC ## Objetivo
# MAGIC Criar catálogo e schemas Unity Catalog (arquitetura Medallion) para o projeto de previsão de inadimplência e gestão de risco de crédito.
# MAGIC
# MAGIC ## Estrutura
# MAGIC
# MAGIC ```
# MAGIC credit_risk (CATALOG)
# MAGIC ├── bronze (SCHEMA) - Dados brutos (clientes, faturas, pagamentos)
# MAGIC ├── silver (SCHEMA) - Dados limpos e enriquecidos
# MAGIC └── gold   (SCHEMA) - Features de ML, predições, métricas e alertas
# MAGIC ```
# MAGIC
# MAGIC ## Pré-requisitos
# MAGIC - Permissões para criar catálogos no Unity Catalog
# MAGIC - Workspace conectado ao Unity Catalog

# COMMAND ----------

# DBTITLE 1,Widgets de Configuração
dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

# COMMAND ----------

# DBTITLE 1,Verificar Catálogos Existentes
print("📋 Catálogos disponíveis:\n")
catalogs = spark.sql("SHOW CATALOGS").collect()
for cat in catalogs:
    print(f"  - {cat.catalog}")

catalog_exists = any(cat.catalog == CATALOG for cat in catalogs)
print(f"\n{'✅' if catalog_exists else '❌'} Catálogo '{CATALOG}' {'já existe' if catalog_exists else 'será criado'}")

# COMMAND ----------

# DBTITLE 1,Criar Catálogo e Schemas (Bronze/Silver/Gold)
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG} COMMENT 'Plataforma de risco de crédito e inadimplência'")
spark.sql(f"USE CATALOG {CATALOG}")

spark.sql("CREATE SCHEMA IF NOT EXISTS bronze COMMENT 'Camada Bronze - dados brutos (clientes, faturas, pagamentos)'")
spark.sql("CREATE SCHEMA IF NOT EXISTS silver COMMENT 'Camada Silver - dados limpos, tipados e enriquecidos'")
spark.sql("CREATE SCHEMA IF NOT EXISTS gold COMMENT 'Camada Gold - features de ML, predições, métricas e alertas'")

print(f"✅ Catálogo '{CATALOG}' e schemas bronze/silver/gold prontos")

# COMMAND ----------

# DBTITLE 1,Validar Estrutura Criada
schemas = spark.sql(f"SHOW SCHEMAS IN {CATALOG}").collect()
print(f"\n📦 Catálogo: {CATALOG}")
print(f"   Total de schemas: {len(schemas)}\n")
for schema in schemas:
    print(f"  📁 {schema.databaseName}")

# COMMAND ----------

# DBTITLE 1,Próximos Passos
# MAGIC %md
# MAGIC ## ✅ Catálogo Criado com Sucesso!
# MAGIC
# MAGIC ### Próximos Passos:
# MAGIC 1. **Permissões**: `01_setup/02_configurar_permissoes`
# MAGIC 2. **Ingestão Bronze**: `02_ingestion/*`
# MAGIC 3. **Transformação Silver**: `03_feature_engineering/01_transformacao_silver`
# MAGIC 4. **Transformação Gold + Feature Store**: `03_feature_engineering/02_*` a `04_*`
# MAGIC 5. **Modelagem**: `04_modeling/*`
# MAGIC
# MAGIC ### Tabelas atuais no catálogo (referência)
# MAGIC - **Bronze**: `clientes`, `faturas`, `pagamentos`
# MAGIC - **Silver**: `clientes`, `faturas_enriquecidas` (criadas em `03_feature_engineering/01_transformacao_silver`)
# MAGIC - **Gold**: `features_agregadas`, `features_rfm`, `features_ml`, `model_metrics`, `model_alerts`, `model_versions`, `model_predictions`
