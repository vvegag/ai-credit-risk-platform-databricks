# Databricks notebook source
# DBTITLE 1,Transformação Gold - Header
# MAGIC %md
# MAGIC # Camada Gold - Features Agregadas
# MAGIC
# MAGIC **Input**: `credit_risk.silver.{clientes, faturas_enriquecidas}`
# MAGIC **Output**: `credit_risk.gold.features_agregadas`
# MAGIC
# MAGIC Agregações temporais (90/180/365 dias), contagens por status e taxas de pagamento/inadimplência por cliente.
# MAGIC 100% PySpark nativo — nenhuma conversão para pandas nesta etapa.

# COMMAND ----------

# DBTITLE 1,Setup
from pyspark.sql.functions import *

dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

print("="*70)
print("🔧 FASE 2.1 - FEATURES AGREGADAS GOLD")
print("="*70)

# COMMAND ----------

# DBTITLE 1,Carregar Silver
df_clientes = spark.table(f"{CATALOG}.silver.clientes")
df_faturas = spark.table(f"{CATALOG}.silver.faturas_enriquecidas")

print(f"  ✅ {df_clientes.count():,} clientes carregados")
print(f"  ✅ {df_faturas.count():,} faturas carregadas")

# COMMAND ----------

# DBTITLE 1,Features Agregadas por Cliente
features = df_faturas.groupBy("id_cliente").agg(
    # Faturamento por período
    sum_(when(datediff(current_date(), col("data_emissao")) <= 90, col("valor_total")).otherwise(0)).alias("total_faturado_90d"),
    sum_(when(datediff(current_date(), col("data_emissao")) <= 180, col("valor_total")).otherwise(0)).alias("total_faturado_180d"),
    sum_(when(datediff(current_date(), col("data_emissao")) <= 365, col("valor_total")).otherwise(0)).alias("total_faturado_365d"),

    # Contagens por status
    count("*").alias("count_faturas_total"),
    sum_(when(col("status") == "Paga", 1).otherwise(0)).alias("count_faturas_pagas"),
    sum_(when(col("status") == "Pendente", 1).otherwise(0)).alias("count_faturas_pendentes"),
    sum_(when(col("status") == "Atrasada", 1).otherwise(0)).alias("count_faturas_atrasadas"),

    # Estatísticas de valores
    avg("valor_total").alias("valor_medio_fatura"),
    stddev("valor_total").alias("desvio_padrao_valores"),
    min("valor_total").alias("valor_minimo_fatura"),
    max("valor_total").alias("valor_maximo_fatura")
)

# Taxas calculadas
features = features.withColumn(
    "taxa_pagamento",
    round((col("count_faturas_pagas") / col("count_faturas_total")) * 100, 2)
).withColumn(
    "taxa_inadimplencia",
    round((col("count_faturas_atrasadas") / col("count_faturas_total")) * 100, 2)
)

print("  ✅ Features agregadas e taxas calculadas")

# COMMAND ----------

# DBTITLE 1,Join com Clientes e Salvar
df_gold = df_clientes.join(features, "id_cliente", "left")

numeric_cols = [
    "total_faturado_90d", "total_faturado_180d", "total_faturado_365d",
    "count_faturas_total", "count_faturas_pagas", "count_faturas_pendentes",
    "count_faturas_atrasadas", "valor_medio_fatura", "desvio_padrao_valores",
    "valor_minimo_fatura", "valor_maximo_fatura", "taxa_pagamento", "taxa_inadimplencia"
]
df_gold = df_gold.fillna(0, subset=numeric_cols)

df_gold.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.gold.features_agregadas")

print(f"  ✅ {CATALOG}.gold.features_agregadas criada")
print(f"     • {df_gold.count():,} clientes | {len(df_gold.columns)} features")

# COMMAND ----------

# DBTITLE 1,Validação
display(spark.table(f"{CATALOG}.gold.features_agregadas").limit(10))

print("="*70)
print("✅ ETAPA 1 COMPLETA - features_agregadas")
print("   Próximo: 03_feature_engineering/03_feature_store_rfm")
print("="*70)
