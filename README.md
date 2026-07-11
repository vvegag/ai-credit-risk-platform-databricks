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
| **Cashflow Forecast** | Prophet | 90-day cashflow prediction with confidence intervals |

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

### 🤖 RAG System for Document Validation

- **Document simulation**: invoice text materialized as `.txt` files on disk (stand-in for an OCR/parsing output), then read back from disk before embedding
- **Embeddings**: Sentence-Transformers for semantic understanding (no chunking — each invoice is already a short, atomic text)
- **Vector Search**: FAISS for fast similarity matching
- **Business Rules**: Automated validation against contracts
- **Workflow**: Auto-approve, flag for review, or reject

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
│   ├── 01_modelo_classificacao_risco.py # XGBoost Classifier + SHAP + MLflow
│   ├── 02_modelo_regressao.py           # XGBoost Regressor (monetary value at risk)
│   └── 03_modelo_forecast_cashflow.py   # Prophet cashflow forecast
├── 05_mlops/
│   └── 01_mlops_pipeline.py            # Retraining, drift checks, metrics/alerts, versioning
├── 06_rag_validation/
│   └── 01_rag_notas_fiscais.py         # RAG-style invoice validation (prototype, see note below)
├── 07_monitoring/
│   └── 01_drift_detection.py           # KS-test data drift + prediction monitoring
├── 08_dashboards/exports/              # AI/BI dashboard exports (.lvdash.json)
├── 09_docs/                            # Architecture, data dictionary, usage guide
├── .gitignore
├── databricks.yml                      # Databricks Asset Bundle (job definition, catalog variable)
├── requirements.txt
├── LICENSE
└── README.md
```

> ⚠️ **RAG note**: `06_rag_validation` is a **prototype** — it simulates invoice text from existing
> records, writes each one as a `.txt` file under `02_ingestion/sample_data/notas_fiscais_txt/`
> (stand-in for a "physical" document already parsed), reads it back from disk, and indexes it with
> a local FAISS index — not a real PDF/OCR pipeline or Databricks Vector Search. See the Roadmap
> below for what a production version would add.

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

This uploads the notebooks and creates a Job that runs the full pipeline end-to-end (setup → ingestion →
feature engineering → modeling → monitoring), using the `catalog` variable defined in `databricks.yml`
(default: `credit_risk`). See `databricks.yml` to change the catalog name or schedule.

### Option B — Manual setup

1. **Clone into your Workspace**: `Workspace → Git folders → Add Git folder`, paste the repo URL.
2. **Setup**: run `01_setup/01_criar_catalogo_schemas.py`, then `02_configurar_permissoes.py`.
3. **Ingestion**: run `02_ingestion/01_popular_dados_completos.py` (and optionally `02_`, `04_`).
4. **Feature engineering**: run `03_feature_engineering/01_` through `04_` in order.
5. **Modeling**: run `04_modeling/01_` through `03_` in order.
6. **MLOps**: run `05_mlops/01_mlops_pipeline.py`.
7. **Monitoring**: run `07_monitoring/01_drift_detection.py`.

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
- **Forecast** (`04_modeling/03_modelo_forecast_cashflow.py`): 90-day cashflow prediction with confidence intervals (Prophet)

Exact metric values depend on the specific synthetic run — check the MLflow experiment after running
`04_modeling/` yourself, or `credit_risk.gold.model_metrics` after `05_mlops/01_mlops_pipeline.py`.

---

## 🛠️ Tech Stack

### Platform
- **Databricks** (Lakehouse Platform)
- **Unity Catalog** (Governance & Lineage)
- **Delta Lake** (Storage Layer)
- **MLflow** (Experiment Tracking & Model Registry)
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
- [x] MLOps pipeline: retraining, drift checks, metrics/alerts tables, model versioning (`05_mlops/01_mlops_pipeline.py`)
- [x] Standalone KS-test drift detection + health checks (`07_monitoring/01_drift_detection.py`)
- [x] Delta maintenance routine: `OPTIMIZE`/`ZORDER`/`VACUUM` (`01_setup/03_manutencao_delta.py`)
- [x] Databricks Asset Bundle (`databricks.yml`) orchestrating the full pipeline as a scheduled Job
- [x] RAG-style invoice validation — **prototype** (see note above), not yet backed by Vector Search
- [x] AI/BI executive dashboard with filters (exports in `08_dashboards/`)

### 🚧 Documented as future work (not built in this repo)
- [ ] **Genie Space** for self-service natural-language analytics
- [ ] **Slack/email alerts** wired to the drift/alerts tables already produced by `05_mlops/01_mlops_pipeline.py`
- [ ] **Real-time Model Serving** endpoint for the registered classifier
- [ ] Production RAG: real PDF/OCR parsing + Databricks Vector Search instead of local FAISS
- [ ] A/B testing for collection strategies
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
