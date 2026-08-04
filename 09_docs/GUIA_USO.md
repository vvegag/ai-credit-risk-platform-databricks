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

## 📈 KS and Decile Table

`04_modeling/01_modelo_classificacao_risco.py` evaluates the classifier with **KS
(Kolmogorov-Smirnov)** and a **decile table**, alongside the AUC-ROC/precision/recall metrics
already there. This isn't a random addition — KS is the market-standard metric for risk
separation in credit and propensity models specifically (used more than AUC-ROC in this
domain), and both showed up as the central ask in a real technical case for a Senior Data
Scientist role at a bank.

### What KS measures

KS is the maximum distance between the cumulative distributions of the model's score for the
two classes (adimplente vs. inadimplente) — computed with `scipy.stats.ks_2samp`, not a manual
CDF implementation. A higher KS means the model separates the two populations further apart at
their best-discriminating threshold. As a rule of thumb in credit risk: KS < 0.20 is weak,
0.20–0.40 reasonable, 0.40–0.60 good, and above ~0.60–0.70 starts being **unusually high** for
a real-world risk/propensity model — worth double-checking rather than celebrating.

### How to read the decile table

The test set is split into 10 groups by predicted score (`pd.qcut`), decile 1 = highest risk.
For a model that's actually separating risk well, the delinquency rate should **decrease
monotonically** from decile 1 to decile 10 — that's the same shape used in real credit scoring
reports to sanity-check a score before deploying it. The notebook prints a warning (not a hard
failure) if it finds inversions, since small per-decile samples can produce 1–2 non-monotonic
steps without indicating a real problem.

### Interpreting a suspiciously high test-set KS

If test-set KS goes above `KS_TESTE_LIMIAR_SUSPEITO` (0.70, defined as a named constant in the
notebook cell, not a magic number), the notebook prints an explicit alert. **This is not a new,
disconnected check** — it's a second lens on the exact same risk the notebook's existing
"Análise de Target Leakage" section (correlation > 0.95 against the target) already looks for.
A test-set KS that's implausibly good is one of the most common real-world signals of target
leakage or a proxy variable that only exists because of/after the outcome you're trying to
predict — the same reasoning used to diagnose it applies regardless of which signal (raw
correlation or KS) surfaced it first. On this project's synthetic data, a high KS can be
expected (synthetic separation tends to be cleaner than real data) — the alert is a prompt to
check, not a claim that something is broken.

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

## 🔀 Switching Databricks Accounts/Workspaces

The project was built to be account-agnostic on purpose — every table reference is parameterized
via the `catalog` widget, there are no hardcoded personal paths, and `databricks.yml` deploys the
whole pipeline as a single Job. This matters in practice if you study/experiment across multiple
Databricks accounts (e.g. a Free Edition workspace and a student/trial workspace with separate
compute limits) and need to move between them when one runs out of credits.

**GitHub is the source of truth, not any single Databricks account.** Nothing important should
live only inside a workspace:

1. Before switching accounts, make sure your latest code is committed and pushed (`git push`).
   Anything uncommitted is the only thing you can actually lose.
2. On the new account: clone/pull the repo into a Databricks Repo, then run:
   ```bash
   databricks bundle deploy -t dev
   databricks bundle run credit_risk_pipeline -t dev
   ```
   This recreates the catalog, regenerates the synthetic data, retrains all 3 models, registers
   them in the new account's Unity Catalog Model Registry, and rebuilds the RAG Vector Search
   index — all from scratch, in one command.
3. Don't try to migrate MLflow experiments, registered models, or Delta tables between accounts —
   it's not worth the effort. Everything here is synthetic and fully reproducible; re-running the
   pipeline on the new account is faster and cleaner than exporting/importing state.
4. Double-check the actual credit/compute policy of each account type before relying on it (Free
   Edition, trial, student/academic access, and Community Edition all have different rules) — the
   assumption that "credits reset every 24h" may only hold for one specific plan.

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
should be roughly 70/20/10 for a fresh synthetic run),  then check for drift via
`07_monitoring/01_drift_detection.py`.
