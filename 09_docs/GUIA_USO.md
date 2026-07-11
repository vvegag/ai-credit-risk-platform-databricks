# 📖 Usage Guide — AI Credit Risk Platform

See the [README](../README.md#-quick-start) for full setup instructions (Asset Bundle or manual).
This guide covers day-to-day usage once the pipeline has been run at least once.

## 🔍 Querying the Data

```sql
-- Highest-risk invoices still open
SELECT id_fatura, id_cliente, valor_total, dias_atraso, valor_em_aberto
FROM credit_risk.silver.faturas_enriquecidas
WHERE pago_flag = 0 AND dias_atraso > 30
ORDER BY valor_em_aberto DESC;

-- Clients with the highest predicted delinquency probability
SELECT id_cliente, perfil_comportamental, probabilidade_inadimplencia
FROM credit_risk.gold.model_predictions
WHERE predicao_inadimplente = 1
ORDER BY probabilidade_inadimplencia DESC
LIMIT 20;
```

---

## 🔬 Using the Trained Model

### Load from MLflow

```python
import mlflow

run_id = "<run_id_from_mlflow_experiment>"
model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
```

### Batch scoring

```python
CATALOG = "credit_risk"
df = spark.table(f"{CATALOG}.gold.features_ml").toPandas()
# Apply the same feature selection / one-hot encoding used in
# 04_modeling/01_modelo_classificacao_risco.py before predicting.
```

`05_mlops/01_mlops_pipeline.py` already implements a full retraining + validation + versioning
flow — prefer extending that notebook over writing a new scoring script from scratch.

---

## 📊 Dashboards and Genie Space

Dashboard exports live in `08_dashboards/exports/` (`.lvdash.json`) — import them via
**Dashboards → Import** in the Databricks UI, then point the datasets at `credit_risk.gold.*`.

For a **Genie Space** (documented as future work — not built in this repo), point it at
`credit_risk.gold.model_predictions` and `credit_risk.gold.features_ml`. Example questions:
- "Which clients have more than R$100k in `valor_em_aberto`?"
- "What's the delinquency rate by `perfil_comportamental`?"

---

## 🔄 Retraining

`05_mlops/01_mlops_pipeline.py` implements the retraining workflow: load current data, retrain,
compare metrics against the previous version, and decide whether to promote the new model — logging
the decision to `credit_risk.gold.model_versions` and `model_alerts`. Trigger it manually, or wire it
into the scheduled Job defined in `databricks.yml`.

---

## 🛠️ Troubleshooting

**Table not found** — confirm the `catalog` widget matches where you actually ran setup:
```sql
SHOW SCHEMAS IN credit_risk;
```

**Silver/Gold tables empty or stale** — re-run `03_feature_engineering/01_` through `04_` in order;
each one depends on the previous (Bronze → Silver → Gold aggregates → RFM → clustering).

**Model performance looks off** — check the target distribution first
(`SELECT categoria_risco, COUNT(*) FROM credit_risk.bronze.clientes GROUP BY categoria_risco`,
should be roughly 70/20/10 for a fresh synthetic run), then check for drift via
`07_monitoring/01_drift_detection.py`.
