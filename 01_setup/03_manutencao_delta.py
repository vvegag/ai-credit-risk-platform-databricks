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
# MAGIC - `ALTER TABLE ... ADD CONSTRAINT` (`NOT NULL` / `CHECK`) nas tabelas Gold mais críticas
# MAGIC   (`features_ml`, `model_predictions`) — declarativo, idempotente (constraint já existente
# MAGIC   ou violada por dado atual é só logada e pulada, igual ao tratamento de OPTIMIZE/VACUUM acima)
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

# COMMAND ----------

# DBTITLE 1,ALTER TABLE ADD CONSTRAINT (NOT NULL) nas tabelas Gold mais críticas
# id_cliente é a chave de junção usada por praticamente todo o pipeline a partir daqui
# (04_modeling, 05_mlops, 07_monitoring) — nula quebraria join silenciosamente. As demais
# colunas marcadas NOT NULL são as que os modelos consomem direto (sem fallback/imputação
# downstream). Fase F do roadmap técnico: features_ml e model_predictions hoje não têm
# nenhuma constraint declarada.
not_null_targets = {
    "gold.features_ml": ["id_cliente", "perfil_comportamental"],
    "gold.model_predictions": ["id_cliente", "probabilidade_inadimplencia"],
}

for table_suffix, columns in not_null_targets.items():
    full_table = f"{CATALOG}.{table_suffix}"
    for col_name in columns:
        try:
            spark.sql(f"ALTER TABLE {full_table} ALTER COLUMN {col_name} SET NOT NULL")
            print(f"✅ NOT NULL aplicado: {full_table}.{col_name}")
        except Exception as e:
            print(f"⚠️ Pulei NOT NULL em {full_table}.{col_name}: {str(e)[:150]}")

# COMMAND ----------

# DBTITLE 1,ALTER TABLE ADD CONSTRAINT (CHECK) nas tabelas Gold mais críticas
# CHECK constraints sobre os domínios de valor já garantidos hoje só implicitamente pela
# lógica de geração (cluster vem de KMeans k=4 → 0-3, rfm_score vem de aplicar_rfm_score
# → 1-5, probabilidade vem de predict_proba → [0,1], predicao_inadimplente é classe binária
# do XGBoost). Declarar isso como constraint no Unity Catalog passa a rejeitar qualquer
# escrita futura (manual ou de um pipeline alterado) que viole essas faixas, em vez de só
# confiar na lógica de quem gerou os dados.
check_targets = {
    "gold.features_ml": {
        "chk_features_ml_cluster_range": "cluster BETWEEN 0 AND 3",
        "chk_features_ml_rfm_score_range": "rfm_score BETWEEN 1 AND 5",
        "chk_features_ml_perfil_valido": (
            "perfil_comportamental IN ('Alto Risco', 'Médio Risco', 'Baixo Risco', 'Premium')"
        ),
    },
    "gold.model_predictions": {
        "chk_model_predictions_probabilidade_range": "probabilidade_inadimplencia BETWEEN 0 AND 1",
        "chk_model_predictions_predicao_binaria": "predicao_inadimplente IN (0, 1)",
    },
}

for table_suffix, constraints in check_targets.items():
    full_table = f"{CATALOG}.{table_suffix}"
    for constraint_name, condition in constraints.items():
        try:
            spark.sql(f"ALTER TABLE {full_table} ADD CONSTRAINT {constraint_name} CHECK ({condition})")
            print(f"✅ CHECK aplicado: {full_table} ({constraint_name})")
        except Exception as e:
            print(f"⚠️ Pulei CHECK {constraint_name} em {full_table}: {str(e)[:150]}")

print("\n✅ Manutenção Delta concluída")
