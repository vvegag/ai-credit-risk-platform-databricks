# Databricks notebook source
# DBTITLE 1,Clustering K-Means - Header
# MAGIC %md
# MAGIC # Perfis Comportamentais - K-Means Clustering
# MAGIC
# MAGIC **Input**: `credit_risk.gold.features_rfm`
# MAGIC **Output**: `credit_risk.gold.features_ml` ⭐ (dataset final para os modelos de `04_modeling`)
# MAGIC
# MAGIC Pipeline PySpark ML nativo: `VectorAssembler` → `StandardScaler` → `KMeans(k=4)`.

# COMMAND ----------

# DBTITLE 1,Setup
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

print("="*70)
print("🤖 FASE 2.3 - K-MEANS CLUSTERING")
print("="*70)

# COMMAND ----------

# DBTITLE 1,Carregar Features RFM
df_rfm_load = spark.table(f"{CATALOG}.gold.features_rfm")
print(f"  ✅ {df_rfm_load.count():,} clientes carregados")

# COMMAND ----------

# DBTITLE 1,Preparar Vetores de Features
feature_cols = [
    "total_faturado_90d",
    "count_faturas_total",
    "taxa_pagamento",
    "taxa_inadimplencia",
    "recency_dias",
    "rfm_score"
]

df_clean = df_rfm_load.fillna(0, subset=feature_cols)

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
df_assembled = assembler.transform(df_clean)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withStd=True,
    withMean=True
)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

print(f"  ✅ {len(feature_cols)} features normalizadas (StandardScaler): {', '.join(feature_cols)}")

# COMMAND ----------

# DBTITLE 1,Treinar K-Means
kmeans = KMeans(k=4, seed=42, featuresCol="features_scaled", predictionCol="cluster")
model = kmeans.fit(df_scaled)
df_clustered = model.transform(df_scaled)

print(f"  ✅ K-Means treinado (k=4) | WSSSE: {model.summary.trainingCost:.2f}")

# COMMAND ----------

# DBTITLE 1,Validar Qualidade do Clustering (Silhouette Score)
# k=4 foi escolhido pelos 4 perfis de negócio que fazem sentido pra ação de cobrança/CS
# (Alto/Médio/Baixo Risco, Premium), não por busca de hiperparâmetro — o silhouette score
# aqui não decide k, só documenta objetivamente quão bem separados os clusters resultantes
# ficaram (varia de -1 a 1; > 0.5 é considerado razoável, > 0.7 forte). Métrica adicional,
# não muda k nem o resultado do clustering acima.
evaluator = ClusteringEvaluator(
    featuresCol="features_scaled", predictionCol="cluster", metricName="silhouette"
)
silhouette_score = evaluator.evaluate(df_clustered)
print(f"  📊 Silhouette Score (k=4): {silhouette_score:.4f}")

# COMMAND ----------

# DBTITLE 1,Mapear Clusters para Perfis Comportamentais
df_perfis = df_clustered.withColumn(
    "perfil_comportamental",
    when(col("cluster") == 0, "Alto Risco")
    .when(col("cluster") == 1, "Médio Risco")
    .when(col("cluster") == 2, "Baixo Risco")
    .otherwise("Premium")
)

# COMMAND ----------

# DBTITLE 1,Salvar Dataset Final
df_final = df_perfis.select(df_rfm_load.columns + ["cluster", "perfil_comportamental"])

df_final.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.gold.features_ml")

print(f"  ✅ {CATALOG}.gold.features_ml criada")
print(f"     • {df_final.count():,} clientes | {len(df_final.columns)} features totais")

# COMMAND ----------

# DBTITLE 1,Distribuição de Perfis
display(spark.table(f"{CATALOG}.gold.features_ml").groupBy("perfil_comportamental").count().orderBy("count", ascending=False))

# COMMAND ----------

# DBTITLE 1,Estatísticas por Perfil
display(
    spark.table(f"{CATALOG}.gold.features_ml").groupBy("perfil_comportamental").agg(
        round(avg("taxa_pagamento"), 2).alias("avg_taxa_pagamento"),
        round(avg("taxa_inadimplencia"), 2).alias("avg_taxa_inadimplencia"),
        round(avg("total_faturado_90d"), 2).alias("avg_faturado_90d"),
        round(avg("rfm_score"), 2).alias("avg_rfm_score")
    )
)

print("="*70)
print("🎉 FASE 2 COMPLETA - features_ml pronta para 04_modeling/01_modelo_classificacao_risco")
print("="*70)
