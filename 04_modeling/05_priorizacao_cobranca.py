# Databricks notebook source
# DBTITLE 1,Priorização de Cobrança — Ranking Combinado
# MAGIC %md
# MAGIC # 🎯 Priorização de Cobrança — Ranking Combinado
# MAGIC
# MAGIC ## Objetivo
# MAGIC Combinar as saídas do classificador (`01_modelo_classificacao_risco.py`, probabilidade
# MAGIC de inadimplência) e do regressor (`02_modelo_regressao.py`, valor monetário em risco) num
# MAGIC único ranking de priorização — "quais clientes a equipe de cobrança deve contatar primeiro".
# MAGIC
# MAGIC ## Por que isso importa
# MAGIC Hoje os dois modelos existem, mas cada um responde só metade da pergunta de negócio:
# MAGIC - O classificador diz **quem** provavelmente vai inadimplir (probabilidade)
# MAGIC - O regressor diz **quanto** dinheiro está em risco (valor)
# MAGIC
# MAGIC Um cliente com 90% de probabilidade de inadimplência mas R$ 200 em risco importa menos
# MAGIC pra cobrança do que um cliente com 60% de probabilidade e R$ 50.000 em risco. O score de
# MAGIC priorização (`probabilidade × valor_previsto`) resolve esse trade-off — é a mesma lógica que
# MAGIC times de cobrança usam manualmente para decidir "quais são as primeiras 10, primeiras 5"
# MAGIC contas a acionar primeiro, só que sistematizada e recalculável a cada retreino.
# MAGIC
# MAGIC ## Dependências
# MAGIC Roda depois de `01_modelo_classificacao_risco.py` (lê `gold.model_predictions`) e
# MAGIC `02_modelo_regressao.py` (lê `gold.previsao_valor_inadimplente`).

# COMMAND ----------

# DBTITLE 1,Setup e Imports
dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

from pyspark.sql import functions as F

print("✅ Setup concluído")

# COMMAND ----------

# DBTITLE 1,Carregar Predições dos Dois Modelos
# Junção feita em Spark (não pandas) — ambas as tabelas de origem já são pequenas
# (1 linha por cliente), mas mantém o padrão PySpark-first do projeto para joins/agregações.
df_classificacao = spark.table(f"{CATALOG}.gold.model_predictions").select(
    "id_cliente", "perfil_comportamental", "probabilidade_inadimplencia", "predicao_inadimplente"
)

df_regressao = spark.table(f"{CATALOG}.gold.previsao_valor_inadimplente").select(
    "id_cliente", "valor_previsto", "categoria_risco_monetario"
)

df_priorizacao = df_classificacao.join(df_regressao, on="id_cliente", how="inner")

print(f"✅ Predições carregadas e unidas: {df_priorizacao.count():,} clientes")

# COMMAND ----------

# DBTITLE 1,Calcular Score de Priorização
# score = probabilidade de inadimplência × valor previsto em risco. valor_previsto pode ser
# negativo (erro do regressor) — usamos greatest(..., 0) para não inflar a prioridade de
# quem o modelo já indica ter valor em risco nulo/negativo.
df_priorizacao = df_priorizacao.withColumn(
    "valor_previsto_ajustado", F.greatest(F.col("valor_previsto"), F.lit(0.0))
).withColumn(
    "score_priorizacao", F.col("probabilidade_inadimplencia") * F.col("valor_previsto_ajustado")
)

df_priorizacao = df_priorizacao.withColumn(
    "faixa_prioridade",
    F.when(F.col("score_priorizacao") >= 20000, "Crítica")
     .when(F.col("score_priorizacao") >= 5000, "Alta")
     .when(F.col("score_priorizacao") >= 1000, "Média")
     .otherwise("Baixa")
)

# Ranking explícito (rank 1 = maior prioridade)
from pyspark.sql.window import Window
w = Window.orderBy(F.col("score_priorizacao").desc())
df_priorizacao = df_priorizacao.withColumn("ranking", F.row_number().over(w))

print("✅ Score e ranking de priorização calculados")

# COMMAND ----------

# DBTITLE 1,Salvar Tabela de Priorização
table_name = f"{CATALOG}.gold.priorizacao_cobranca"

df_priorizacao.select(
    "ranking", "id_cliente", "perfil_comportamental",
    "probabilidade_inadimplencia", "valor_previsto", "score_priorizacao",
    "faixa_prioridade", "predicao_inadimplente", "categoria_risco_monetario",
).orderBy("ranking").write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)

print(f"✅ Tabela salva: {table_name}")

# COMMAND ----------

# DBTITLE 1,Sumário Executivo
print("="*70)
print("🎯 SUMÁRIO — PRIORIZAÇÃO DE COBRANÇA")
print("="*70)

print("\n📊 Distribuição por faixa de prioridade:")
display(
    df_priorizacao.groupBy("faixa_prioridade")
    .agg(F.count("*").alias("num_clientes"), F.sum("valor_previsto_ajustado").alias("valor_total_em_risco"))
    .orderBy(F.desc("valor_total_em_risco"))
)

print("\n🔴 TOP 10 — Prioridade máxima de contato:")
top10 = (
    df_priorizacao.orderBy("ranking")
    .select("ranking", "id_cliente", "probabilidade_inadimplencia", "valor_previsto", "score_priorizacao", "faixa_prioridade")
    .limit(10)
    .toPandas()
)
for _, row in top10.iterrows():
    print(
        f"  #{row['ranking']:>3} Cliente {row['id_cliente']}: "
        f"prob={row['probabilidade_inadimplencia']:.1%}, "
        f"valor=R$ {row['valor_previsto']:,.2f}, "
        f"score={row['score_priorizacao']:,.2f} ({row['faixa_prioridade']})"
    )

print("\n" + "="*70)
