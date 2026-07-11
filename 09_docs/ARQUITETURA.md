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

### 5. MLOps (`05_mlops/01_mlops_pipeline.py`) and Monitoring (`07_monitoring/`)
Two complementary layers: a fuller MLOps pipeline that retrains, checks drift, and logs
metrics/alerts/versions to dedicated Gold tables on every run; and a lightweight, standalone
KS-test drift/health-check notebook for quick, ad-hoc inspection.

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

DataFrames read more than once in the same run are `.cache()`d and explicitly `.unpersist()`d after
their last use (see `04_modeling/01_modelo_classificacao_risco.py`, `07_monitoring/01_drift_detection.py`).

---

## ⚡ Performance & Delta maintenance

- **Partitioning**: `bronze.faturas` and `silver.faturas_enriquecidas` are partitioned by
  `ano_mes_emissao` (year-month), matching the temporal filters used across feature engineering and
  monitoring.
- **OPTIMIZE + ZORDER**: `01_setup/03_manutencao_delta.py` runs `OPTIMIZE ... ZORDER BY` on the most
  frequently joined/filtered Gold tables (`id_cliente`), and on `silver.faturas_enriquecidas`
  (`id_cliente, data_vencimento`).
- **VACUUM**: same notebook, default 7-day retention. This is a separate, schedulable maintenance
  routine — it does not run automatically as part of the main pipeline.
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

---

## 🚀 Roadmap

See the [README](../README.md#-roadmap) for the current split between what's implemented in this
repo and what's documented as future work.
