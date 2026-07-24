# 🏗️ Architecture — AI Credit Risk Platform

## 📐 Overview

End-to-end Machine Learning system for delinquency prediction and cashflow forecasting, built on
Databricks with a Medallion architecture and a PySpark-native pipeline.

```
┌─────────────────────────────────────────────────────────────────┐
│                       DATA SOURCES                              │
│  Synthetic generator (02_ingestion/01_)  │  Manual CSVs (02_)   │
└───────────┬───────────────────────────────┴───────────┬─────────┘
            ▼                                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  🥉 BRONZE — credit_risk.bronze.*                                │
│  clientes, faturas (partitioned by ano_mes_emissao), pagamentos  │
│  Raw data, no transformation                                     │
└───────────┬───────────────────────────────────────────────────────┘
            │ PySpark DataFrame API (03_feature_engineering/01_)
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  🥈 SILVER — credit_risk.silver.*                                 │
│  clientes, faturas_enriquecidas (joins with payments, dias_atraso,│
│  valor_em_aberto, pago_flag)                                      │
└───────────┬───────────────────────────────────────────────────────┘
            │ Feature engineering (03_feature_engineering/02_–04_)
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  🥇 GOLD — credit_risk.gold.*                                     │
│  features_agregadas → features_rfm → features_ml (final ML table) │
│  model_predictions, model_metrics, model_alerts, model_versions,  │
│  forecast_cashflow, previsao_valor_inadimplente,                  │
│  validacao_notas_fiscais, monitoring_logs                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧱 Components

### 1. Data Ingestion (`02_ingestion/`)
Synthetic Bronze data generation and a CSV Auto Loader path — both PySpark-native (`spark.createDataFrame`,
`readStream`/`writeStream`), no `toPandas()` round-trips.

### 2. Silver Transformation (`03_feature_engineering/01_transformacao_silver.py`)
Cleans and types Bronze data, joins `faturas` with `pagamentos`, and derives `valor_em_aberto` /
`pago_flag`. 100% PySpark DataFrame API.

### 3. Gold / Feature Store (`03_feature_engineering/02_`–`04_`)
- **02_transformacao_gold**: temporal aggregates (90/180/365d), payment/delinquency rates
- **03_feature_store_rfm**: Recency/Frequency/Monetary scoring
- **04_clustering_features_ml**: `VectorAssembler` → `StandardScaler` → `KMeans(k=4)` behavioral profiles

`credit_risk.gold.features_ml` is the single feature table consumed by every model in `04_modeling/`.

### 4. Modeling (`04_modeling/`)
| Notebook | Model | Library |
|---|---|---|
| `01_modelo_classificacao_risco` | Delinquency classifier | XGBoost + SHAP |
| `02_modelo_regressao` | Monetary value at risk | XGBoost Regressor |
| `03_modelo_forecast_cashflow` | 90-day cashflow forecast | Prophet |

### 5. Model Registry (Unity Catalog)
Both `04_modeling/01_modelo_classificacao_risco.py` and `05_mlops/01_mlops_pipeline.py` register
to the **same** Unity Catalog Model Registry entry (`credit_risk.gold.credit_risk_classifier`) —
via `mlflow.xgboost.log_model(..., registered_model_name=...)`, not a legacy workspace registry
and not a loose pickle file. Promotion uses `Champion`/`Challenger` **aliases**
(`MlflowClient.set_registered_model_alias`), not raw version numbers:
- `04_modeling/01_` bootstraps the **first** Champion on a fresh catalog (no-op if one already exists).
- `05_mlops/01_mlops_pipeline.py` owns retraining and promotion: it compares the new model's metrics
  against the *current Champion's* real metrics (pulled from the registry, not hardcoded), and on
  approval promotes the new version to Champion while the previous Champion steps down to Challenger
  — a one-step rollback path, not a full A/B framework.
- Both notebooks score `credit_risk.gold.model_predictions` by loading `models:/.../@Champion`, so the
  predictions table always reflects whichever model is actually promoted, not whatever was last trained
  in memory.

### 6. MLOps (`05_mlops/01_mlops_pipeline.py`) and Monitoring (`07_monitoring/`)
Two complementary layers: a fuller MLOps pipeline that retrains, checks drift, and logs
metrics/alerts/versions to dedicated Gold tables on every run; and a lightweight, standalone
KS-test drift/health-check notebook for quick, ad-hoc inspection. Production metrics in the MLOps
pipeline are computed from the **real** `gold.model_predictions` table (which already carries the
ground-truth label `perfil_real`) — not simulated with `np.random`.

### 7. RAG (`06_rag_validation/` and `10_rag_agent/`)
Two implementations at different maturity levels, kept side by side on purpose:
- `06_rag_validation/01_rag_notas_fiscais.py` — **prototype**: simulated invoice text written to
  `.txt` files, local FAISS index.
- `10_rag_agent/` — **real Databricks Vector Search**: generates actual PDFs (`reportlab`), extracts/
  chunks text (`pypdf`), embeds, and syncs to a managed Vector Search index
  (`credit_risk.documentos.credit_docs_vector_index`) via Change Data Feed on the underlying Delta
  table. Includes a LangChain agent (`src/rag_agent.py`) for conversational retrieval and a Model
  Serving deploy script (`deploy/model_serving.py`).

---

## 🐍 PySpark-first convention

All ETL, joins, aggregations, and filtering over Bronze/Silver/Gold tables must use the Spark
DataFrame API or Spark SQL — not pandas. `.toPandas()` is only acceptable:

1. **Immediately before** a `.fit()` call for a library with no Spark-native equivalent, on an
   already-small, already-aggregated table (one row per client): XGBoost's sklearn API, SHAP's
   `TreeExplainer`, Prophet.
2. For **synthetic data generation** with numpy/random distributions in `02_ingestion` (small,
   one-off, not part of the recurring pipeline).

Anywhere a `.apply(axis=1)` or `.iterrows()` shows up on a Spark-sourced DataFrame is a bug —
replace it with vectorized pandas (`np.select`, `str.contains`, boolean masks) or push the logic
back into Spark. `06_rag_validation/01_rag_notas_fiscais.py` documents a worked example of this
(validation rules implemented with `np.select` instead of row-wise `.apply`).

**Known exceptions to this policy (acknowledged, not fixed)**: `06_rag_validation/01_rag_notas_fiscais.py`
and `10_rag_agent/notebooks/01_gerar_documentos_credito.py` both call `.toPandas()` on
invoice/payment-grain tables (not the one-row-per-client tables the two exceptions above cover) to
generate document text/PDFs row by row. This is acceptable at the synthetic data volume used today
(a few thousand invoices) but would **not** scale to real production invoice volume — at that point
these notebooks would need the document-generation logic rewritten with `mapInPandas`/`pandas_udf`
(keeps the per-row Python logic but runs it distributed across the Spark cluster instead of
collecting everything to the driver first).

`.cache()`/`.persist()` are avoided on Spark DataFrames project-wide: on serverless compute they
compile down to a `PERSIST TABLE` RPC, which raises `NOT_SUPPORTED_WITH_SERVERLESS`
(see `04_modeling/01_modelo_classificacao_risco.py`, `07_monitoring/01_drift_detection.py`, both hit
this in practice and were fixed by dropping the cache call).

---

## ⚡ Performance & Delta maintenance

- **Partitioning**: `bronze.faturas` and `silver.faturas_enriquecidas` are partitioned by
  `ano_mes_emissao` (year-month), matching the temporal filters used across feature engineering and
  monitoring.
- **OPTIMIZE + ZORDER**: `01_setup/03_manutencao_delta.py` runs `OPTIMIZE ... ZORDER BY` on the most
  frequently joined/filtered Gold tables (`id_cliente`), and on `silver.faturas_enriquecidas`
  (`id_cliente, data_vencimento`).
- **VACUUM**: same notebook, default 7-day retention. Wired into `databricks.yml` as the
  `manutencao_delta` task, which runs after every branch of the weekly scheduled Job finishes
  writing — it's a maintenance step, not part of the data-producing critical path, but it does run
  automatically now (previously manual-only).
- **Photon**: recommended (cluster/warehouse setting, not code) — the workload is dominated by
  SQL/DataFrame aggregations (feature engineering, drift detection), which benefit directly from
  Photon's vectorized engine.

---

## 🔐 Governance

- **Catalog**: `credit_risk` (parameterized via a `catalog` widget in every notebook — never
  hardcoded, so the same code runs against any catalog name/workspace)
- **Schemas**: `bronze`, `silver`, `gold`
- **Permissions**: `01_setup/02_configurar_permissoes.py` — data engineers (full access),
  data scientists (read bronze/silver, full gold), business users (read gold only)
- **Auditability**: Delta Time Travel, MLflow experiment tracking, Unity Catalog lineage

### Sensitive data (known gap, documented design — not implemented)

`bronze.clientes.cnpj` is stored as plaintext today (a Brazilian legal-entity ID, same
sensitivity class as an individual's CPF), with no column-level masking, classification tag, or
row-level security, and no LGPD-specific documentation elsewhere in this repo. This wasn't
implemented because Unity Catalog row filters/column masks require workspace-admin privileges
that aren't reliably available on the trial/academic accounts this project has been validated
against — not because it's out of scope for a real deployment. The intended design, if/when a
workspace with the right privileges is available:

1. **Column mask** on `cnpj` (`ALTER TABLE ... ALTER COLUMN cnpj SET MASK ...`) — a SQL function
   that returns the full value to `data_engineers`/`data_scientists` (need it for joins/dedup) and
   a redacted form (e.g. only the first 2 and last 2 digits) to `business_users`.
2. **Column tag** (`SET TAGS ('sensitivity' = 'pii')`) on `cnpj`, so it shows up in Unity Catalog's
   built-in classification/lineage UI instead of being indistinguishable from any other string
   column.
3. Document the masking rule in `DICIONARIO_DADOS.md` next to the column definition, so the data
   contract and the actual enforcement don't drift apart.

---

## 🚀 Roadmap

See the [README](../README.md#-roadmap) for the current split between what's implemented in this
repo and what's documented as future work.
