# Databricks notebook source
# DBTITLE 1,Transformação Silver - Header
# MAGIC %md
# MAGIC # Camada Silver - Limpeza e Enriquecimento
# MAGIC
# MAGIC **Input**: `credit_risk.bronze.{clientes, faturas, pagamentos}`
# MAGIC **Output**: `credit_risk.silver.{clientes, faturas_enriquecidas}`
# MAGIC
# MAGIC Todo o processamento é PySpark nativo (DataFrame API / Spark SQL) — sem `toPandas()`.

# COMMAND ----------

# DBTITLE 1,Setup
from pyspark.sql.functions import *

dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

print("="*60)
print("🧼 TRANSFORMAÇÃO SILVER")
print("="*60)

# COMMAND ----------

# DBTITLE 1,Silver: Clientes
# Tipagem e deduplicação — validação leve, base já vem tratada da geração sintética
df_clientes_bronze = spark.table(f"{CATALOG}.bronze.clientes")

df_clientes_silver = (
    df_clientes_bronze
    .dropDuplicates(["id_cliente"])
    .withColumn("nome", trim(col("nome")))
    .withColumn("setor", trim(col("setor")))
    .withColumn("porte", trim(col("porte")))
    .filter(col("id_cliente").isNotNull())
)

df_clientes_silver.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.silver.clientes")

print(f"✅ {CATALOG}.silver.clientes: {df_clientes_silver.count():,} registros")

# COMMAND ----------

# DBTITLE 1,Silver: Faturas Enriquecidas
# Join faturas + pagamentos, tratamento de nulos/negativos, e colunas derivadas
# usadas pelo forecast de cashflow (04_modeling/04_modelo_forecast_cashflow) e RAG/monitoring.
df_faturas_bronze = spark.table(f"{CATALOG}.bronze.faturas")
df_pagamentos_bronze = spark.table(f"{CATALOG}.bronze.pagamentos")

# Um pagamento por fatura (a base sintética já garante isso, mas protegemos contra duplicatas)
df_pagamentos_agg = df_pagamentos_bronze.groupBy("id_fatura").agg(
    sum("valor").alias("valor_pago_total"),
    first("forma_pagamento").alias("forma_pagamento")
)

df_faturas_enriquecidas = (
    df_faturas_bronze
    .filter(col("valor") >= 0)  # descarta registros inconsistentes
    .withColumnRenamed("valor", "valor_total")
    .join(df_pagamentos_agg, "id_fatura", "left")
    .withColumn("valor_pago_total", coalesce(col("valor_pago_total"), lit(0.0)))
    .withColumn("pago_flag", (col("status") == lit("Paga")).cast("int"))
    .withColumn(
        "valor_em_aberto",
        when(col("pago_flag") == 1, lit(0.0)).otherwise(col("valor_total") - col("valor_pago_total"))
    )
    # dias_atraso já vem calculado no Bronze na geração sintética; mantido aqui para não quebrar
    # o contrato downstream, mas seria recalculado a partir de data_vencimento/data_pagamento
    # em uma ingestão real via datediff(coalesce(data_pagamento, current_date()), data_vencimento).
)

# Particionamento herdado de bronze.faturas (ano_mes_emissao) — mantém data skipping na Silver
df_faturas_enriquecidas.write.mode("overwrite").option("overwriteSchema", "true") \
    .partitionBy("ano_mes_emissao") \
    .saveAsTable(f"{CATALOG}.silver.faturas_enriquecidas")

print(f"✅ {CATALOG}.silver.faturas_enriquecidas: {df_faturas_enriquecidas.count():,} registros")

# COMMAND ----------

# DBTITLE 1,Validação
display(spark.table(f"{CATALOG}.silver.faturas_enriquecidas").limit(10))

print("="*60)
print("✅ CAMADA SILVER CRIADA")
print("   Próximo: 03_feature_engineering/02_transformacao_gold")
print("="*60)
