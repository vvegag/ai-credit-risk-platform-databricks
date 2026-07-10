# Databricks notebook source
# MAGIC %md
# MAGIC # FASE 2 COMPLETA - Feature Engineering Gold Layer 
# MAGIC
# MAGIC **Execução Completa das 3 Etapas :**
# MAGIC 1. Features Agregadas (temporal features)
# MAGIC 2. RFM Scoring (segmentação)
# MAGIC 3. K-Means Clustering (perfis comportamentais)
# MAGIC
# MAGIC **Output Final:** credit_risk.gold.features_ml (dataset para XGBoost)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 ETAPA 1: Features Agregadas

# COMMAND ----------

from pyspark.sql.functions import *

print("="*70)
print("🔧 FASE 2.1 - FEATURES AGREGADAS GOLD")
print("="*70)
print()

# Carregar Bronze
print("📥 Carregando dados Bronze...")
df_clientes = spark.table("credit_risk.bronze.clientes")
df_faturas = spark.table("credit_risk.bronze.faturas")

count_clientes = df_clientes.count()
count_faturas = df_faturas.count()

print(f"  ✅ {count_clientes:,} clientes carregados")
print(f"  ✅ {count_faturas:,} faturas carregadas")
print()

# COMMAND ----------

# Features agregadas por cliente
print("📊 Calculando features agregadas...")
features = df_faturas.groupBy("id_cliente").agg(
    # Faturamento por período
    sum(when(datediff(current_date(), col("data_emissao")) <= 90, col("valor")).otherwise(0)).alias("total_faturado_90d"),
    sum(when(datediff(current_date(), col("data_emissao")) <= 180, col("valor")).otherwise(0)).alias("total_faturado_180d"),
    sum(when(datediff(current_date(), col("data_emissao")) <= 365, col("valor")).otherwise(0)).alias("total_faturado_365d"),
    
    # Contagens por status
    count("*").alias("count_faturas_total"),
    sum(when(col("status") == "Paga", 1).otherwise(0)).alias("count_faturas_pagas"),
    sum(when(col("status") == "Pendente", 1).otherwise(0)).alias("count_faturas_pendentes"),
    sum(when(col("status") == "Atrasada", 1).otherwise(0)).alias("count_faturas_atrasadas"),
    
    # Estatísticas de valores
    avg("valor").alias("valor_medio_fatura"),
    stddev("valor").alias("desvio_padrao_valores"),
    min("valor").alias("valor_minimo_fatura"),
    max("valor").alias("valor_maximo_fatura")
)

print("  ✅ Features agregadas calculadas")
print()

# COMMAND ----------

# Taxas calculadas
print("💯 Calculando taxas...")
features = features.withColumn(
    "taxa_pagamento", 
    round((col("count_faturas_pagas") / col("count_faturas_total")) * 100, 2)
).withColumn(
    "taxa_inadimplencia",
    round((col("count_faturas_atrasadas") / col("count_faturas_total")) * 100, 2)
)

print("  ✅ Taxas calculadas")
print()

# COMMAND ----------

# Join com clientes
print("🔗 Juntando com dados de clientes...")
df_gold = df_clientes.join(features, "id_cliente", "left")

# Preencher nulos
numeric_cols = [
    "total_faturado_90d", "total_faturado_180d", "total_faturado_365d",
    "count_faturas_total", "count_faturas_pagas", "count_faturas_pendentes",
    "count_faturas_atrasadas", "valor_medio_fatura", "desvio_padrao_valores",
    "valor_minimo_fatura", "valor_maximo_fatura", "taxa_pagamento", "taxa_inadimplencia"
]

for col_name in numeric_cols:
    df_gold = df_gold.fillna({col_name: 0})

print("  ✅ Join completo")
print()

# COMMAND ----------

# Salvar
print("💾 Salvando credit_risk.gold.features_agregadas...")
df_gold.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("credit_risk.gold.features_agregadas")

count_gold = df_gold.count()
num_cols = len(df_gold.columns)

print(f"  ✅ Tabela criada!")
print(f"     • {count_gold:,} clientes")
print(f"     • {num_cols} features")
print()

print("="*70)
print("✅ ETAPA 1 COMPLETA - features_agregadas")
print("="*70)
print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 ETAPA 2: RFM Scoring

# COMMAND ----------

print("="*70)
print("🎯 FASE 2.2 - RFM SCORING")
print("="*70)
print()

# Carregar features agregadas
print("📥 Carregando features agregadas...")
df_features = spark.table("credit_risk.gold.features_agregadas")
df_faturas = spark.table("credit_risk.bronze.faturas")

print(f"  ✅ {df_features.count():,} clientes carregados")
print()

# COMMAND ----------

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

# Salvar
print("💾 Salvando credit_risk.gold.features_rfm...")
df_rfm.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("credit_risk.gold.features_rfm")

count_rfm = df_rfm.count()
print(f"  ✅ Tabela criada!")
print(f"     • {count_rfm:,} clientes")
print()

# Distribuição RFM
print("📊 DISTRIBUIÇÃO DE CATEGORIAS RFM:")
df_rfm.groupBy("categoria_rfm").count().orderBy("count", ascending=False).show()
print()

print("="*70)
print("✅ ETAPA 2 COMPLETA - features_rfm")
print("="*70)
print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🤖 ETAPA 3: K-Means Clustering

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

print("="*70)
print("🤖 FASE 2.3 - K-MEANS CLUSTERING")
print("="*70)
print()

# Carregar features RFM
print("📥 Carregando features RFM...")
df_rfm_load = spark.table("credit_risk.gold.features_rfm")

print(f"  ✅ {df_rfm_load.count():,} clientes carregados")
print()

# COMMAND ----------

# Selecionar features para clustering
feature_cols = [
    "total_faturado_90d", 
    "count_faturas_total", 
    "taxa_pagamento", 
    "taxa_inadimplencia",
    "recency_dias",
    "rfm_score"
]

# Preencher nulos
df_clean = df_rfm_load.fillna(0, subset=feature_cols)

print(f"📊 Features para clustering:")
print(f"  ✅ {df_clean.count():,} registros preparados")
print(f"  ✅ {len(feature_cols)} features: {', '.join(feature_cols)}")
print()

# COMMAND ----------

# Vector Assembler
print("🔧 Preparando vetores de features...")
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
df_assembled = assembler.transform(df_clean)

# Standard Scaler (normalização)
scaler = StandardScaler(
    inputCol="features_raw", 
    outputCol="features_scaled",
    withStd=True, 
    withMean=True
)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

print("  ✅ Features normalizadas (StandardScaler)")
print()

# COMMAND ----------

# K-Means com 4 clusters
print("🤖 Treinando K-Means (k=4)...")
kmeans = KMeans(
    k=4, 
    seed=42, 
    featuresCol="features_scaled", 
    predictionCol="cluster"
)

model = kmeans.fit(df_scaled)
df_clustered = model.transform(df_scaled)

print("  ✅ K-Means modelo treinado!")
print(f"  ✅ WSSSE: {model.summary.trainingCost:.2f}")
print()

# COMMAND ----------

# Mapear clusters para perfis comportamentais
print("🎯 Mapeando clusters para perfis...")
df_perfis = df_clustered.withColumn(
    "perfil_comportamental",
    when(col("cluster") == 0, "Alto Risco")
    .when(col("cluster") == 1, "Médio Risco")
    .when(col("cluster") == 2, "Baixo Risco")
    .otherwise("Premium")
)

print("  ✅ Perfis comportamentais mapeados")
print()

# COMMAND ----------

# Selecionar colunas finais
print("💾 Salvando credit_risk.gold.features_ml...")
df_final = df_perfis.select(
    df_rfm_load.columns + ["cluster", "perfil_comportamental"]
)

# Salvar dataset final para ML
df_final.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("credit_risk.gold.features_ml")

count_final = df_final.count()
num_cols = len(df_final.columns)

print(f"  ✅ Tabela criada!")
print(f"     • {count_final:,} clientes")
print(f"     • {num_cols} features totais")
print()

# COMMAND ----------

# Distribuição de Perfis
print("="*70)
print("📊 DISTRIBUIÇÃO DE PERFIS COMPORTAMENTAIS")
print("="*70)
df_final.groupBy("perfil_comportamental").count().orderBy("count", ascending=False).show()
print()

# COMMAND ----------

# Estatísticas por Perfil
print("📊 ESTATÍSTICAS POR PERFIL:")
df_final.groupBy("perfil_comportamental").agg(
    round(avg("taxa_pagamento"), 2).alias("avg_taxa_pagamento"),
    round(avg("taxa_inadimplencia"), 2).alias("avg_taxa_inadimplencia"),
    round(avg("total_faturado_90d"), 2).alias("avg_faturado_90d"),
    round(avg("rfm_score"), 2).alias("avg_rfm_score")
).show()
print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎉 FASE 2 COMPLETA!

# COMMAND ----------

print("="*70)
print("🎉 FASE 2 COMPLETA - FEATURE ENGINEERING GOLD LAYER")
print("="*70)
print()
print("📦 TABELAS GOLD CRIADAS:")
print("   1️⃣ credit_risk.gold.features_agregadas")
print("   2️⃣ credit_risk.gold.features_rfm")
print("   3️⃣ credit_risk.gold.features_ml ⭐ (DATASET FINAL PARA XGBOOST)")
print()
print("🚀 PRÓXIMO: FASE 3 - Modelo XGBoost de Classificação")
print("="*70)

# COMMAND ----------

# Validação Final
print("\n✅ VALIDAÇÃO FINAL:")
print("-" * 70)

# Verificar tabelas
tabelas = ["features_agregadas", "features_rfm", "features_ml"]
for tabela in tabelas:
    count = spark.table(f"credit_risk.gold.{tabela}").count()
    print(f"   • credit_risk.gold.{tabela}: {count:,} registros")

print("-" * 70)
print("✅ Todas as tabelas Gold foram criadas com sucesso!")
