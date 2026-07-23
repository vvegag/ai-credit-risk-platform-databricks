# 🤖 AI Credit Risk Platform

[![Databricks](https://img.shields.io/badge/Databricks-Lakehouse-FF3621?logo=databricks&logoColor=white)](https://databricks.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Delta Lake](https://img.shields.io/badge/Delta-Lake-00ADD4?logo=delta&logoColor=white)](https://delta.io)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🎯 **End-to-end Machine Learning platform** for credit risk prediction, delinquency forecasting, and automated document validation using **Databricks Lakehouse**.

---

## 📊 Overview

**AI Credit Risk Platform** is a complete MLOps solution built on Databricks that combines:
- **Predictive Analytics**: Identify high-risk customers before delinquency occurs
- **Value Estimation**: Quantify monetary risk for prioritization
- **Cashflow Forecasting**: 12-month ahead predictions for financial planning
- **Document Automation**: RAG-based invoice validation (prototype)
- **Continuous Monitoring**: Drift detection and data quality checks

**Impact**: demonstrates the full pipeline end-to-end on synthetic data — see the [Results](#-results) section for what "end-to-end" means in this repo and what's still a prototype.

---

## 🌟 Key Features

### 🎯 Machine Learning Models

| Model | Type | Purpose |
|-------|------|---------|
| **Risk Classifier** | XGBoost + SHAP | Predict delinquency probability & risk class |
| **Value Regressor** | XGBoost | Estimate monetary value at risk |
| **Cashflow Forecast** | Prophet + external regressor | 90-day cashflow prediction with confidence intervals; an `evento_irregular` regressor isolates non-periodic events (Black Friday, year-end) from Prophet's automatic seasonality, and every run logs a Model Card artifact (assumptions, backtest window, MAE/MAPE) to MLflow for auditability (`04_modeling/03_modelo_forecast_cashflow.py`) |
| **AutoML + LightGBM comparison** | Databricks AutoML, LightGBM | Benchmarks the manual XGBoost classifier against an AutoML baseline and LightGBM on the same features/split (`04_modeling/04_automl_lightgbm_comparacao.py`) |
| **Collection Prioritization Ranking** | Combines Classifier + Regressor | Joins delinquency probability and value-at-risk into a single `probability × value` priority score and Top-N ranking, answering "who should collections contact first" (`04_modeling/05_priorizacao_cobranca.py`) |

Performance numbers depend on the specific synthetic data run — see [Results](#-results) below.

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GOLD LAYER (Business)                    │
│  • Predictions  • Forecasts  • Monitoring Logs  • RAG Docs  │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                   SILVER LAYER (Enriched)                   │
│           • Enriched Invoices  • ML Features                │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                    BRONZE LAYER (Raw)                       │
│          • Clients  • Invoices  • Transactions              │
└─────────────────────────────────────────────────────────────┘
```

**Medallion Architecture** with Delta Lake for ACID transactions and Unity Catalog for governance.

### 🤖 RAG: Two Implementations (Prototype → Production)

**`06_rag_validation`** (prototype):
- Document simulation: invoice text materialized as `.txt` files on disk, read back before embedding
- Embeddings: Sentence-Transformers, indexed with local FAISS
- Business rules: automated validation against contracts (auto-approve, flag, reject)

**`10_rag_agent`** (real Databricks Vector Search):
- Generates real PDF documents (credit contracts, bank statements, income proof) with `reportlab`
- Extracts and chunks text (`pypdf`), embeds with Sentence-Transformers
- Indexes in a real **Databricks Vector Search** endpoint (`credit_risk.documentos.credit_docs_vector_index`), synced from a Delta table via Change Data Feed
- **LangChain agent** for conversational queries over the indexed documents
- Modular Python package (`src/`) with its own tests (`tests/`) and a Model Serving deploy script (`deploy/`)

### 📈 Monitoring & Observability

- **Drift Detection**: Kolmogorov-Smirnov tests for feature distributions
- **Performance Tracking**: Weekly out-of-sample AUC monitoring
- **Data Quality**: Health checks for all Delta tables
- **Alerts**: Automated notifications for drift or performance degradation

---

## 📂 Project Structure

```
ai-credit-risk-platform-databricks/
├── 01_setup/
│   ├── 01_criar_catalogo_schemas.py    # Creates the credit_risk catalog + bronze/silver/gold schemas
│   ├── 02_configurar_permissoes.py     # Unity Catalog GRANTs (data engineers/scientists/business users)
│   └── 03_manutencao_delta.py          # OPTIMIZE / ZORDER / VACUUM maintenance routine
├── 02_ingestion/
│   ├── 01_popular_dados_completos.py   # Synthetic Bronze data generator (clients, invoices, payments)
│   ├── 02_ingestao_csv_manuais.py      # Auto Loader for manual CSVs
│   ├── 04_gerar_pdfs_notas.py          # Sample invoice PDF generator
│   └── sample_data/
│       ├── csvs/                       # Sample CSVs used by the manual-ingestion demo
│       ├── notas_fiscais/              # Sample invoice PDFs (gitignored)
│       └── notas_fiscais_txt/          # Simulated "physical" invoice text used by the RAG prototype
├── 03_feature_engineering/
│   ├── 01_transformacao_silver.py      # Bronze → Silver (cleaning, typing, invoice enrichment)
│   ├── 02_transformacao_gold.py        # Silver → Gold aggregated features (90/180/365d windows)
│   ├── 03_feature_store_rfm.py         # RFM scoring (Recency/Frequency/Monetary)
│   └── 04_clustering_features_ml.py    # K-Means behavioral profiles → final ML feature table
├── 04_modeling/
│   ├── 01_modelo_classificacao_risco.py     # XGBoost Classifier + SHAP + MLflow
│   ├── 02_modelo_regressao.py               # XGBoost Regressor (monetary value at risk)
│   ├── 03_modelo_forecast_cashflow.py       # Prophet cashflow forecast + irregular-event regressor + Model Card
│   ├── 04_automl_lightgbm_comparacao.py     # Databricks AutoML + LightGBM vs. manual XGBoost
│   └── 05_priorizacao_cobranca.py           # Ranks clients by probability × value-at-risk (collection priority)
├── 05_mlops/
│   ├── 01_mlops_pipeline.py            # Retraining, drift checks, metrics/alerts, versioning
│   └── 02_model_serving_endpoint.py    # Real-time Model Serving endpoint synced to the Champion alias
├── 06_rag_validation/
│   └── 01_rag_notas_fiscais.py         # RAG-style invoice validation (prototype, see note below)
├── 07_monitoring/
│   ├── 01_drift_detection.py                     # KS-test data drift + prediction monitoring
│   └── 02_validacao_ab_intervencao_cobranca.py   # A/B statistical test for collection-intervention effect (synthetic demo)
├── 08_dashboards/exports/              # AI/BI dashboard exports (.lvdash.json)
├── 09_docs/                            # Architecture, data dictionary, usage guide
├── 10_rag_agent/                       # RAG with real Databricks Vector Search (see note below)
│   ├── notebooks/                      # Document generation, indexing, LangChain agent
│   ├── src/                            # Importable package: config, embeddings, vector_search, rag_agent
│   ├── deploy/                         # Model serving deployment script
│   └── tests/                          # Unit tests
├── .gitignore
├── databricks.yml                      # Databricks Asset Bundle (job definition, catalog variable)
├── requirements.txt
├── LICENSE
└── README.md
```

> ⚠️ **Two RAG implementations, different maturity**:
> - `06_rag_validation` is a **prototype** — it simulates invoice text from existing records, writes
>   each one as a `.txt` file under `02_ingestion/sample_data/notas_fiscais_txt/` (stand-in for a
>   "physical" document already parsed), reads it back from disk, and indexes it with a **local FAISS**
>   index — not a real PDF/OCR pipeline or managed vector search.
> - `10_rag_agent` is the **real implementation**: it generates actual PDF documents (credit contracts,
>   bank statements, income proof), extracts and chunks their text, embeds them, and indexes them in a
>   real **Databricks Vector Search** endpoint (`credit_risk.documentos.credit_docs_vector_index`), with
>   a LangChain agent for conversational queries. `06_rag_validation` was kept as-is to document the
>   "quick prototype vs. production" progression within the same repo.

---

## 🚀 Quick Start

### Prerequisites

- Databricks Workspace (Azure, AWS, or GCP) with Unity Catalog enabled
- Serverless Compute (or a cluster with ML Runtime 14+)
- [Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/index.html) configured (`databricks configure`), if using the Asset Bundle path

### Option A — Deploy with Databricks Asset Bundle (recommended)

```bash
git clone https://github.com/vvegag/ai-credit-risk-platform-databricks.git
cd ai-credit-risk-platform-databricks
databricks bundle deploy -t dev
databricks bundle run credit_risk_pipeline -t dev
```

This uploads the notebooks and creates a Job that runs the full pipeline end-to-end: setup → ingestion →
feature engineering → **all 3 models in parallel** (classification, regression, cashflow forecast) →
MLOps (retrain/promote the classifier) → monitoring, plus **both RAG paths** (the FAISS prototype and
the real Databricks Vector Search pipeline: PDF generation → chunking/embeddings → indexing → LangChain
agent) running alongside modeling. Uses the `catalog` variable defined in `databricks.yml` (default:
`credit_risk`). See `databricks.yml` for the full task DAG, or to change the catalog name/schedule.

### Option B — Manual setup

1. **Clone into your Workspace**: `Workspace → Git folders → Add Git folder`, paste the repo URL.
2. **Setup**: run `01_setup/01_criar_catalogo_schemas.py`, then `02_configurar_permissoes.py`.
3. **Ingestion**: run `02_ingestion/01_popular_dados_completos.py` (and optionally `02_`, `04_`).
4. **Feature engineering**: run `03_feature_engineering/01_` through `04_` in order.
5. **Modeling**: run `04_modeling/01_` through `04_` (classification, regression, forecast, AutoML/LightGBM comparison), then `05_priorizacao_cobranca.py` (needs `01_` and `02_` already run).
6. **MLOps**: run `05_mlops/01_mlops_pipeline.py`, then `02_model_serving_endpoint.py`.
7. **Monitoring**: run `07_monitoring/01_drift_detection.py`, then `02_validacao_ab_intervencao_cobranca.py` (needs `05_priorizacao_cobranca.py` already run).

All notebooks expose a `catalog` widget (default `credit_risk`) — no hardcoded workspace paths or
personal usernames anywhere in the code.

---

## 📊 Results

> ⚠️ **On synthetic data**: all figures below come from a run against the synthetic dataset
> generated by `02_ingestion/01_popular_dados_completos.py` (controlled distributions, not real
> production data). They illustrate the pipeline working end-to-end — they are **not** a claim of
> production-grade model performance. Re-running the pipeline will produce different numbers.

### Example Model Performance (synthetic data)

- **Classification** (`04_modeling/01_modelo_classificacao_risco.py`): XGBoost + SHAP, tracked via MLflow
- **Regression** (`04_modeling/02_modelo_regressao.py`): monetary value at risk per client
- **Forecast** (`04_modeling/03_modelo_forecast_cashflow.py`): 90-day cashflow prediction with confidence intervals (Prophet), an external regressor for non-periodic events (Black Friday, year-end), and a Model Card artifact logged to MLflow documenting assumptions/backtest/MAPE for auditability
- **Collection priority ranking** (`04_modeling/05_priorizacao_cobranca.py`): joins classifier probability with regressor value-at-risk into one `credit_risk.gold.priorizacao_cobranca` Top-N table
- **A/B intervention validation** (`07_monitoring/02_validacao_ab_intervencao_cobranca.py`): two-proportion Z-test methodology for measuring whether contacting flagged clients reduces delinquency — runs on a clearly-labeled **synthetic** control/treatment simulation (no real intervention log exists in this project), but the statistical test itself is production-ready: swap in a real contact log and it works unchanged

Exact metric values depend on the specific synthetic run — check the MLflow experiment after running
`04_modeling/` yourself, or `credit_risk.gold.model_metrics` after `05_mlops/01_mlops_pipeline.py`.

---

## 🛠️ Tech Stack

### Platform
- **Databricks** (Lakehouse Platform) — cloud-agnostic by design (validated on AWS-backed and
  Azure-backed workspaces), no cloud-specific code paths
- **Unity Catalog** (Governance & Lineage)
- **Delta Lake** (Storage Layer)
- **MLflow** (Experiment Tracking & Model Registry, Model Serving)
- **Serverless Compute**

### Languages & Frameworks
- **Python** (PySpark, Pandas, NumPy)
- **SQL** (Spark SQL, Databricks SQL)
- **XGBoost** (Classification & Regression)
- **Prophet** (Time Series Forecasting)
- **Sentence-Transformers** (Embeddings)
- **FAISS** (Vector Search)

### Visualization
- **Databricks AI/BI Dashboards**
- **Plotly** (Interactive charts)

---

## 📈 Roadmap

### ✅ Implemented in this repo
- [x] Medallion architecture (Bronze → Silver → Gold), 100% PySpark-native ETL
- [x] 3 ML models: classification (XGBoost + SHAP), regression, cashflow forecast (Prophet)
- [x] **Unity Catalog Model Registry** with `Champion`/`Challenger` aliases (`mlflow.xgboost.log_model` +
      `MlflowClient.set_registered_model_alias`) — not a legacy workspace registry, not a loose pickle file
- [x] MLOps pipeline: retraining, drift checks, real production metrics (computed from actual predictions,
      not simulated data), alerts, Champion promotion (`05_mlops/01_mlops_pipeline.py`)
- [x] Closed consumption loop: classifier writes `gold.model_predictions` → MLOps pipeline reads it for
      real metrics/alerts → `07_monitoring/` and dashboards/Genie read the same table
- [x] Standalone KS-test drift detection + health checks (`07_monitoring/01_drift_detection.py`)
- [x] Delta maintenance routine: `OPTIMIZE`/`ZORDER`/`VACUUM` (`01_setup/03_manutencao_delta.py`)
- [x] Databricks Asset Bundle (`databricks.yml`) orchestrating the full pipeline as a scheduled Job
- [x] RAG with real **Databricks Vector Search** (`10_rag_agent/`) — PDF generation, chunking, embeddings,
      managed vector index, LangChain conversational agent
- [x] RAG prototype (`06_rag_validation/`) — local FAISS, kept to document the prototype→production path
- [x] AI/BI executive dashboard with filters (exports in `08_dashboards/`)
- [x] **Databricks AutoML** baseline + **LightGBM** comparison against the manual XGBoost classifier
      (`04_modeling/04_automl_lightgbm_comparacao.py`)
- [x] **Real-time Model Serving** endpoint (`05_mlops/02_model_serving_endpoint.py`) — serves the
      Champion alias directly, scale-to-zero, re-synced automatically after every retraining/promotion
- [x] **Collection prioritization ranking** combining classifier probability × regressor value-at-risk
      into a single Top-N table (`04_modeling/05_priorizacao_cobranca.py`)
- [x] **A/B testing methodology** for collection strategies — two-proportion Z-test with confidence
      interval and significance check (`07_monitoring/02_validacao_ab_intervencao_cobranca.py`). Runs on
      a clearly-labeled **synthetic** control/treatment simulation, since no real intervention log exists
      in this project; the statistical test itself works unchanged against real data

### 🚧 Documented as future work (not built in this repo)
- [ ] **Genie Space** for self-service natural-language analytics
- [ ] **Slack/email alerts** wired to the drift/alerts tables already produced by `05_mlops/01_mlops_pipeline.py`
- [ ] Inference Tables (automatic request/response logging) on the Model Serving endpoint
- [ ] CRM integration (Salesforce), Next Best Action, Lifetime Value (LTV) prediction

---

## 📚 Documentation

- **[Architecture](09_docs/ARQUITETURA.md)** — Medallion design, PySpark-first convention, performance notes
- **[Data Dictionary](09_docs/DICIONARIO_DADOS.md)** — Tables and columns across bronze/silver/gold
- **[Usage Guide](09_docs/GUIA_USO.md)** — How to run and extend the pipeline

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Valdomiro Vega**

- 📧 Email: valdomirovega@hotmail.com
- 💼 GitHub: [@vvegag](https://github.com/vvegag)
- 🔗 LinkedIn: [valdomiro-vega](https://linkedin.com/in/valdomiro-vega)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Databricks** for the incredible Lakehouse platform
- **MLflow** for seamless experiment tracking
- **XGBoost** and **Prophet** teams for robust ML libraries
- **Sentence-Transformers** and **FAISS** for RAG capabilities

---

## ⭐ Star This Repository

If you found this project helpful, please give it a ⭐ on GitHub!

---

**Built with ❤️ using Databricks**
