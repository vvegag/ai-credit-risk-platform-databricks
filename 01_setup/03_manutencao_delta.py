# Databricks notebook source
# DBTITLE 1,Manutenção Delta - Header
# MAGIC %md
# MAGIC # 🧹 Manutenção Delta - OPTIMIZE / ZORDER / VACUUM
# MAGIC
# MAGIC **Objetivo**: Rotina de manutenção das tabelas Delta mais consultadas, para manter performance
# MAGIC de leitura em produção conforme o volume de dados cresce.
# MAGIC
# MAGIC ⚠️ **Não roda automaticamente** a cada execução do pipeline — é uma rotina separada, pensada para
# MAGIC ser agendada (ex: `databricks.yml` job semanal) ou executada manualmente quando necessário.
# MAGIC
# MAGIC **O que faz**:
# MAGIC - `OPTIMIZE ... ZORDER BY` nas tabelas Gold mais consultadas (colocação física dos dados por
# MAGIC   coluna de filtro/join frequente → menos I/O nas queries)
# MAGIC - `VACUUM` com retenção padrão de 7 dias (remove arquivos de dados órfãos de versões antigas)
# MAGIC
# MAGIC **Recomendação adicional (configurar no cluster/warehouse, não neste notebook)**: habilitar
# MAGIC **Photon** — o workload deste projeto é dominado por agregações SQL/DataFrame (feature
# MAGIC engineering, drift detection), que se beneficiam diretamente do engine vetorizado do Photon.

# COMMAND ----------

# DBTITLE 1,Setup
dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

# COMMAND ----------

# DBTITLE 1,OPTIMIZE + ZORDER nas tabelas Gold mais consultadas
# id_cliente é a coluna de join/filtro mais comum entre features_ml, model_predictions e monitoring
zorder_targets = {
    "gold.features_ml": "id_cliente",
    "gold.model_predictions": "id_cliente",
    "silver.faturas_enriquecidas": "id_cliente, data_vencimento",
}

for table_suffix, zorder_cols in zorder_targets.items():
    full_table = f"{CATALOG}.{table_suffix}"
    try:
        spark.sql(f"OPTIMIZE {full_table} ZORDER BY ({zorder_cols})")
        print(f"✅ OPTIMIZE + ZORDER concluído: {full_table} (por {zorder_cols})")
    except Exception as e:
        print(f"⚠️ Pulei {full_table}: {str(e)[:150]}")

# COMMAND ----------

# DBTITLE 1,VACUUM (retenção padrão de 7 dias)
vacuum_targets = [
    "bronze.clientes", "bronze.faturas", "bronze.pagamentos",
    "silver.clientes", "silver.faturas_enriquecidas",
    "gold.features_agregadas", "gold.features_rfm", "gold.features_ml",
    "gold.model_predictions",
]

for table_suffix in vacuum_targets:
    full_table = f"{CATALOG}.{table_suffix}"
    try:
        spark.sql(f"VACUUM {full_table}")
        print(f"✅ VACUUM concluído: {full_table}")
    except Exception as e:
        print(f"⚠️ Pulei {full_table}: {str(e)[:150]}")

print("\n✅ Manutenção Delta concluída")
