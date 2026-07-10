# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,RFM Scoring - Header
# MAGIC %md
# MAGIC # RFM Scoring - Recency, Frequency, Monetary
# MAGIC
# MAGIC **Input**: credit_risk.gold.features_agregadas, credit_risk.bronze.faturas  
# MAGIC **Output**: credit_risk.gold.features_rfm
# MAGIC
# MAGIC **RFM Score** (1-5, sendo 5 o melhor):
# MAGIC - **Recency**: Dias desde última fatura
# MAGIC - **Frequency**: count_faturas_total
# MAGIC - **Monetary**: total_faturado_90d

# COMMAND ----------

# DBTITLE 1,Carregar Dados
from pyspark.sql.functions import *

print("="*60)
print("🔧 FASE 2.2 - RFM SCORING")
print("="*60)
print()

# Carregar features agregadas
df_features = spark.table("credit_risk.gold.features_agregadas")
df_faturas = spark.table("credit_risk.bronze.faturas")

print(f"✅ {df_features.count():,} clientes carregados")

# COMMAND ----------

# DBTITLE 1,Calcular Recency
# Calcular Recency: dias desde última fatura
print("📅 Calculando Recency...")
recency = df_faturas.groupBy("id_cliente").agg(
    datediff(current_date(), max("data_emissao")).alias("recency_dias")
)

# Join
df_rfm = df_features.join(recency, "id_cliente", "left")

print(f"  ✅ Recency calculado para {df_rfm.count():,} clientes")
print()

# COMMAND ----------

# DBTITLE 1,RFM Score e Categorias
# RFM Score (1-5, sendo 5 o melhor)
print("🎯 Calculando RFM Score...")
df_rfm = df_rfm.withColumn(
    "rfm_score",
    when(col("recency_dias") < 30, 5)
    .when(col("recency_dias") < 60, 4)
    .when(col("recency_dias") < 90, 3)
    .when(col("recency_dias") < 180, 2)
    .otherwise(1)
)

# Categoria RFM
df_rfm = df_rfm.withColumn(
    "categoria_rfm",
    when(col("rfm_score") >= 4, "Premium")
    .when(col("rfm_score") == 3, "Regular")
    .otherwise("Em Risco")
)

print("  ✅ RFM Score e categorias criadas")
print()

# COMMAND ----------

# DBTITLE 1,Salvar Tabela RFM
# Salvar
print("💾 Salvando credit_risk.gold.features_rfm...")
df_rfm.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("credit_risk.gold.features_rfm")

count_rfm = df_rfm.count()
print(f"  ✅ Tabela criada!")
print(f"     • {count_rfm:,} clientes")
print()

# COMMAND ----------

# DBTITLE 1,Distribuição RFM
# Distribuição RFM
print("="*60)
print("📊 DISTRIBUIÇÃO DE CATEGORIAS RFM")
print("="*60)
display(spark.table("credit_risk.gold.features_rfm").groupBy("categoria_rfm").count().orderBy("count", ascending=False))

# COMMAND ----------

# DBTITLE 1,Amostra de Dados
# Amostra de dados
print("📋 Amostra de Dados RFM:")
display(spark.table("credit_risk.gold.features_rfm").select([
    "id_cliente", "nome", "porte",
    "recency_dias", "rfm_score", "categoria_rfm",
    "total_faturado_90d", "taxa_pagamento"
]).limit(10))

print()
print("="*60)
print("✅ NOTEBOOK 2 COMPLETO - features_rfm")
print("="*60)
