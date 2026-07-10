# 🏗️ Arquitetura do Projeto - Inadimplência e Risco Financeiro

## 📐 Visão Geral

Sistema end-to-end de Machine Learning para previsão de inadimplência, construído em Databricks com arquitetura Medallion e MLOps completo.

```
┌─────────────────────────────────────────────────────────────────┐
│                     FONTES DE DADOS                             │
├─────────────────────────────────────────────────────────────────┤
│  Fivetran (ERP Simulado)    │  CSVs Manuais (Financeiro)       │
│  - Clientes                 │  - Ajustes                        │
│  - Faturas                  │  - Renegociações                  │
│  - Marcas                   │  - Correções Manuais              │
└───────────┬─────────────────┴───────────────┬───────────────────┘
            │                                 │
            ▼                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                 🥉 BRONZE (Raw Data)                            │
│  workspace.risco_bronze.*                                       │
│  - marcas_raw, clientes_raw, faturas_raw                        │
│  - Dados brutos, sem transformação                              │
└───────────┬─────────────────────────────────────────────────────┘
            │
            │ PySpark Transformations
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 🥈 SILVER (Clean & Enriched)                    │
│  workspace.risco_silver.*                                       │
│  - faturas_enriquecidas (26 colunas)                            │
│  - Joins, cálculo de dias_atraso, classificação de risco        │
│  - Métricas agregadas por cliente                               │
└───────────┬─────────────────────────────────────────────────────┘
            │
            │ Feature Engineering
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 🥇 GOLD (Analytics-Ready)                       │
│  workspace.risco_gold.*                                         │
│  - predicoes_inadimplencia                                      │
│  - Dados prontos para dashboards e consumo                      │
└─────────────────────────────────────────────────────────────────┘
            │
            │ Parallel Processing
            │
      ┌─────┴─────┐
      │           │
      ▼           ▼
┌──────────┐  ┌──────────────────┐
│ Features │  │   ML Models      │
│  Store   │  │   & Training     │
│          │  │                  │
│ RFM      │  │  MLflow          │
│ Temporal │  │  Experiments     │
│ Behavior │  │  RandomForest    │
│          │  │  XGBoost         │
└──────────┘  └────────┬─────────┘
                       │
                       │ Model Registry
                       │
                       ▼
              ┌─────────────────┐
              │  Unity Catalog  │
              │  Model Registry │
              │                 │
              │  Champion/      │
              │  Challenger     │
              └────────┬────────┘
                       │
                       │ Deploy
                       │
                       ▼
              ┌─────────────────┐
              │  Model Serving  │
              │  (REST API)     │
              │                 │
              │  Batch          │
              │  Inference      │
              └─────────────────┘
```

---

## 🧱 Componentes

### 1. Data Ingestion Layer

**Tecnologia**: Auto Loader (simulado com batch)
**Cadência**: Diária
**Volume**: ~500 novas faturas/dia

```python
# Bronze ingestion
spark.createDataFrame(df_raw) \
    .write.mode("append") \
    .saveAsTable("workspace.risco_bronze.faturas_raw")
```

### 2. Data Transformation Layer

**Tecnologia**: PySpark SQL + DataFrames
**Cadência**: Após ingestão (triggered)
**Transformações**:
- Limpeza de dados (nulls, duplicatas)
- Joins entre clientes, marcas e faturas
- Cálculos de métricas (dias_atraso, valor_em_aberto)
- Classificação de risco

```python
# Silver transformation
df_silver = df_bronze \
    .join(df_clientes, "codigo_cliente") \
    .withColumn("dias_atraso", datediff(current_date(), col("data_vencimento"))) \
    .withColumn("faixa_atraso", classify_delay_bucket(...))
```

### 3. Feature Store

**Tecnologia**: Unity Catalog Delta Tables
**Features**: 17 features por cliente
**Atualização**: Diária

**Categorias de Features**:
- RFM (Recency, Frequency, Monetary)
- Comportamento de pagamento
- Métricas de atraso
- Valores agregados

### 4. ML Training Pipeline

**Framework**: Scikit-learn + MLflow
**Algoritmo**: Random Forest Classifier
**Métricas Tracked**:
- Accuracy
- Precision, Recall, F1
- ROC-AUC
- Feature Importance

```python
with mlflow.start_run():
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    mlflow.log_metrics({...})
    mlflow.sklearn.log_model(model, "model")
```

### 5. Model Registry & Deployment

**Registry**: Unity Catalog Model Registry
**Aliases**:
- `champion`: Modelo em produção
- `challenger`: Modelo candidato

**Deployment Options**:
1. **Batch Inference**: Spark job semanal
2. **REST API**: Model Serving endpoint (planned)
3. **Embedded**: Python function em notebooks

---

## 🔄 Fluxo de Dados Completo

```
1. INGEST     → Bronze tables (raw data)
2. TRANSFORM  → Silver tables (clean + enriched)
3. AGGREGATE  → Gold tables (analytics-ready)
4. ENGINEER   → Features (ML-ready)
5. TRAIN      → Model (MLflow tracked)
6. REGISTER   → Model Registry (versioned)
7. DEPLOY     → Scoring (batch/API)
8. MONITOR    → Drift detection (planned)
9. RETRAIN    → Automated trigger (planned)
```

---

## 🔐 Governança & Segurança

### Unity Catalog

- **Catalogs**: workspace
- **Schemas**: risco_bronze, risco_silver, risco_gold, risco_ml_features
- **Tables**: Delta format com versionamento
- **Permissions**: Granular por schema/table

### Auditoria

- **Delta Time Travel**: Acesso a versões anteriores
- **MLflow Tracking**: Rastreabilidade de experimentos
- **Lineage**: Databricks Unity Catalog lineage

---

## 📊 Performance & Escalabilidade

### Volumes Atuais
- Clientes: 300
- Faturas: 4,486
- Features: 300 × 17
- Scoring: 300 predições/batch

### Escalabilidade Projetada
- Clientes: 50k+
- Faturas: 1M+
- Features: Delta caching
- Scoring: Distributed PySpark

### Otimizações
- **Z-Ordering**: Por data_vencimento em faturas
- **Partitioning**: Por ano/mês em faturas
- **Caching**: Feature Store em memória
- **Broadcast Joins**: Para tabelas pequenas (marcas)

---

## 🚀 Roadmap Técnico

### Fase 1: MVP ✅
- [x] Arquitetura Medallion
- [x] Feature Engineering
- [x] Modelo de Classificação
- [x] Batch Inference
- [x] MLflow Tracking

### Fase 2: MLOps (Planejado)
- [ ] Model Serving REST API
- [ ] Drift Detection
- [ ] Automated Retraining
- [ ] A/B Testing (Champion vs Challenger)
- [ ] Inference Tables para auditoria

### Fase 3: Avançado (Planejado)
- [ ] Modelo de Regressão (valor não recebido)
- [ ] Forecast de Cash Flow (Prophet/ARIMA)
- [ ] RAG para validação de notas fiscais
- [ ] Alertas automáticos (Slack/Email)

---

**Autor**: Valdomiro Vega García  
**Data**: 02/07/2026
**Versão Arquitetura**: 1.0
