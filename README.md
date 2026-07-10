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
- **Document Automation**: RAG-based invoice validation system
- **Continuous Monitoring**: Drift detection and data quality checks

**Impact**: R$ 9M in risk mapped, 60 critical clients identified, 75% automation in validation, 80% time reduction.

---

## 🌟 Key Features

### 🎯 Machine Learning Models

| Model | Type | Performance | Purpose |
|-------|------|-------------|---------|
| **Risk Classifier** | XGBoost | AUC: 0.88, Acc: 85% | Predict delinquency probability & risk class |
| **Value Regressor** | XGBoost | R²: 0.74 | Estimate monetary value at risk |
| **Cashflow Forecast** | Prophet | 12-month horizon | Time series prediction for financial planning |

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

- **PDF Parsing**: Extract structured data from invoices
- **Embeddings**: Sentence-Transformers for semantic understanding
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
ai-credit-risk-platform/
├── 01_setup/                   # Initial setup and configuration
├── 02_ingestion/               # Bronze layer data ingestion
├── 03_feature_engineering/     # Feature creation (28 features)
├── 04_modeling/
│   ├── 01_modelo_classificacao     # XGBoost Classifier
│   ├── 02_modelo_regressao         # XGBoost Regressor
│   └── 03_modelo_forecast_cashflow # Prophet Time Series
├── 05_mlops/                   # MLflow tracking and model registry
├── 06_rag_validation/          # RAG system for invoice validation
├── 07_monitoring/              # Drift detection and health checks
├── 08_dashboards/              # AI/BI Dashboard exports
├── 09_docs/                    # Documentation and diagrams
├── .gitignore
├── LICENSE
├── README.md
├── GUIA_GITHUB.md             # GitHub setup guide (Portuguese)
└── setup_github.sh            # Automated Git setup script
```

---

## 🚀 Quick Start

### Prerequisites

- Databricks Workspace (Azure, AWS, or GCP)
- Unity Catalog enabled
- Serverless Compute (or cluster with ML Runtime 14+)

### Setup

1. **Clone this repository** into your Databricks Workspace:
   ```bash
   # Via Databricks UI: Workspace → Repos → Add Repo
   # URL: https://github.com/vvegag/ai-credit-risk-platform.git
   ```

2. **Run setup notebook**:
   - Navigate to `01_setup/`
   - Execute setup notebooks to create schemas and sample data

3. **Execute pipeline**:
   ```
   02_ingestion → 03_feature_engineering → 04_modeling
   ```

4. **View dashboard**:
   - Import dashboard from `08_dashboards/`
   - Connect to Unity Catalog tables

---

## 📊 Results

### Business Impact

| Metric | Value | Impact |
|--------|-------|--------|
| **Critical Clients Identified** | 60 | Proactive risk management |
| **Total Risk Mapped** | R$ 9M | Quantified exposure |
| **Delinquency Rate** | 21.51% | Tracked and monitored |
| **Automation Rate** | 75% | Invoice validation |
| **Time Reduction** | 80% | 2h → 24min per batch |

### Model Performance

- **Classification**: 88% AUC, 85% Accuracy
- **Regression**: R² 0.74, MAE ~R$ 15k
- **Forecast**: 12-month cashflow prediction with confidence intervals

### Dashboard KPIs

- 🔴 **60 Critical Clients** requiring immediate action
- 📉 **21.51% Delinquency Rate** across portfolio
- 💰 **R$ 9.0M Total Value at Risk**
- 📈 **R$ 1.4M Predicted Revenue** (next 30 days)

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

### Phase 1 ✅ (Completed)
- [x] Medallion architecture (Bronze-Silver-Gold)
- [x] 3 ML models (classification, regression, forecast)
- [x] RAG system for document validation
- [x] Drift detection and monitoring
- [x] Executive dashboard with filters

### Phase 2 🚧 (In Progress)
- [ ] Genie Space for self-service analytics
- [ ] Databricks Jobs for automation (weekly retraining, batch scoring)
- [ ] Email/Slack alerts for drift and critical clients

### Phase 3 🔮 (Planned)
- [ ] Real-time scoring via Model Serving
- [ ] A/B testing for collection strategies
- [ ] CRM integration (Salesforce)
- [ ] Next Best Action recommendations
- [ ] Lifetime Value (LTV) prediction

---

## 📚 Documentation

- **[Project Summary](PROJECT_SUMMARY.md)** - Executive overview and technical deep-dive
- **[GitHub Setup Guide](GUIA_GITHUB.md)** - Step-by-step guide to publish (Portuguese)
- **[Architecture Diagrams](09_docs/diagrams/)** - Visual representations
- **[Model Cards](09_docs/model_cards/)** - Detailed model documentation

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
