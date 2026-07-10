# 🏦 Projeto: Sistema de Previsão de Inadimplência e Gestão de Risco Financeiro

**Databricks MLOps End-to-End com RAG para Validação de Notas Fiscais**

---

## 📋 Visão Geral

Sistema completo de Machine Learning e MLOps para gestão de risco financeiro, desenvolvido para empresas de varejo e serviços financeiros que operam com múltiplas marcas e clientes B2B/B2C.

### Problemas Resolvidos

1. ✅ **Previsão de Inadimplência**: Modelo de classificação identifica clientes em risco
2. ✅ **Forecast de Receita não Recebida**: Regressão estima valor que não será recebido no mês
3. ✅ **Fluxo de Caixa Projetado**: Séries temporais preveem entrada de caixa com intervalos de confiança
4. ✅ **Segmentação de Risco**: Clientes classificados em 4 perfis comportamentais
5. ✅ **Validação Automatizada de Notas Fiscais**: RAG valida campos obrigatórios por marca/segmento
6. ✅ **Alertas Inteligentes**: Notificações proativas para ações de cobrança
7. ✅ **Monitoramento MLOps**: Drift detection e retreinamento automatizado

---

## 🏗️ Arquitetura do Projeto

### Arquitetura Medallion (Bronze → Silver → Gold)

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FONTES DE DADOS                            │
├─────────────────────────────────────────────────────────────────────┤
│  Fivetran (ERP)          │  CSVs Manuais (Financeiro)               │
│  - Clientes              │  - Planilhas Excel com fórmulas          │
│  - Faturas/NFs           │  - Cálculos manuais                      │
│  - Pagamentos            │  - Ajustes contábeis                     │
│  - Marcas                │  - Renegociações                         │
└────────────┬──────────────┴──────────────────────┬────────────────────┘
             │                                │
             ▼                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      🥉 BRONZE (Raw Data)                           │
├─────────────────────────────────────────────────────────────────────┤
│  bronze_raw.clientes_raw                                            │
│  bronze_raw.faturas_raw                                             │
│  bronze_raw.pagamentos_raw                                          │
│  bronze_raw.marcas_raw                                              │
│  bronze_raw.csv_financeiro_raw  ← Auto Loader                      │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  🥈 SILVER (Clean & Enriched)                       │
├─────────────────────────────────────────────────────────────────────┤
│  silver_clean.clientes                                              │
│  silver_clean.faturas_enriquecidas                                  │
│  silver_clean.pagamentos_consolidados                               │
│  silver_clean.eventos_cobranca                                      │
│  silver_clean.regras_financeiras                                    │
│  - Join Fivetran + CSV                                              │
│  - Cálculo de dias_atraso, status_pagamento                         │
│  - Enriquecimento: grupo_economico, tier_cliente                    │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   🥇 GOLD (Analytics-Ready)                         │
├─────────────────────────────────────────────────────────────────────┤
│  gold_analytics.fato_inadimplencia                                  │
│  gold_analytics.dim_clientes                                        │
│  gold_analytics.dim_marcas                                          │
│  gold_analytics.dim_tempo                                           │
│  gold_analytics.metricas_financeiras_mes                            │
│  gold_analytics.perfis_comportamentais                              │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    🧠 ML & FEATURE STORE                            │
├─────────────────────────────────────────────────────────────────────┤
│  ml_features.features_rfm                                           │
│  ml_features.features_temporais                                     │
│  ml_features.features_perfil_pagamento                              │
│  ml_models.modelo_classificacao_risco                               │
│  ml_models.modelo_regressao_valor                                   │
│  ml_models.modelo_forecast_cashflow                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Stack Tecnológico

| Componente | Tecnologia |
|------------|------------|
| **Plataforma** | Databricks (AWS/Azure/GCP) |
| **Compute** | Serverless + ML Clusters |
| **Storage** | Delta Lake (Unity Catalog) |
| **Processamento** | Apache Spark (PySpark) |
| **ML Framework** | Scikit-learn, XGBoost, LightGBM, Prophet |
| **MLOps** | MLflow (Tracking, Registry, Serving) |
| **GenAI** | RAG com Vector Search |
| **Orquestração** | Databricks Jobs + Workflows |
| **Visualização** | AI/BI Dashboards + Genie |
| **Monitoramento** | Lakehouse Monitoring |

---

## 📂 Estrutura de Pastas

```
projeto_inadimplencia/
│
├── 01_setup/
│   ├── 01_criar_catalogo_schemas.py          # Unity Catalog setup
│   └── 02_configurar_permissoes.py           # Governança e permissões
│
├── 02_ingestion/
│   ├── 01_gerar_dados_sinteticos.py          # Simulação de dados reais
│   ├── 02_ingestao_fivetran.py               # Simula dados do ERP
│   └── 03_ingestao_csv_manuais.py            # Auto Loader para CSVs
│
├── 03_feature_engineering/
│   ├── 01_transformacao_silver.py            # Bronze → Silver
│   ├── 02_transformacao_gold.py              # Silver → Gold
│   ├── 03_feature_store_rfm.py               # Features RFM
│   └── 04_perfis_comportamentais.py          # Segmentação
│
├── 04_modeling/
│   ├── 01_eda_analise_exploratoria.py        # EDA completo
│   ├── 02_modelo_classificacao_risco.py      # LightGBM/XGBoost
│   ├── 03_modelo_regressao_valor.py          # Regressão ensemble
│   └── 04_modelo_forecast_cashflow.py        # Prophet/ARIMA
│
├── 05_mlops/
│   ├── 01_mlflow_tracking.py                 # Experimentos
│   ├── 02_model_registry.py                  # Registro UC
│   ├── 03_model_serving.py                   # Deploy endpoint
│   └── 04_batch_inference.py                 # Scoring em lote
│
├── 06_rag_validation/
│   ├── 01_ingestao_pdfs_regras.py            # RAG setup
│   ├── 02_vector_search_setup.py             # Embeddings
│   ├── 03_validacao_notas_fiscais.py         # RAG validation
│   └── 04_workflow_orquestracao.py           # LangGraph-style
│
├── 07_monitoring/
│   ├── 01_drift_detection.py                 # Feature/concept drift
│   ├── 02_performance_monitoring.py          # Model health
│   ├── 03_alertas_automaticos.py             # Notificações
│   └── 04_inference_table.py                 # Auditoria
│
├── 08_dashboards/
│   ├── 01_dashboard_executivo.py             # AI/BI Dashboard
│   ├── 02_dashboard_modelos.py               # MLOps metrics
│   └── 03_genie_space_setup.py               # Self-service analytics
│
├── 09_docs/
│   ├── ARQUITETURA.md                        # Diagrama detalhado
│   ├── DICIONARIO_DADOS.md                   # Catálogo de tabelas
│   ├── GUIA_USO.md                           # Como usar
│   └── FAQ.md                                # Troubleshooting
│
└── README.md                                  # Este arquivo
```

---

## 🚀 Quick Start

### Pré-requisitos

- Workspace Databricks (Standard ou Premium)
- Unity Catalog habilitado
- Permissões para criar catalogs e schemas
- Serverless compute disponível

### Instalação

```python
# 1. Executar setup inicial
%run ./01_setup/01_criar_catalogo_schemas

# 2. Gerar dados sintéticos (para demo)
%run ./02_ingestion/01_gerar_dados_sinteticos

# 3. Executar pipeline Bronze → Silver → Gold
%run ./02_ingestion/02_ingestao_fivetran
%run ./03_feature_engineering/01_transformacao_silver
%run ./03_feature_engineering/02_transformacao_gold

# 4. Treinar modelos
%run ./04_modeling/02_modelo_classificacao_risco
%run ./04_modeling/03_modelo_regressao_valor

# 5. Configurar MLOps
%run ./05_mlops/02_model_registry
%run ./05_mlops/03_model_serving

# 6. Visualizar resultados
# Abrir dashboard: ./08_dashboards/01_dashboard_executivo
```

---

## 🎯 Casos de Uso

### 1. Previsão de Inadimplência

**Objetivo**: Identificar clientes com alta probabilidade de atraso >30 dias

**Modelo**: LightGBM Classifier  
**Features**: RFM, histórico de pagamento, sazonalidade, tier_cliente  
**Métrica**: Precision@Top20% (queremos acertar os 20% de maior risco)  
**Output**: Score 0-1 + label (Pontual/Crônico/Instável/Risco)

### 2. Forecast de Valor não Recebido

**Objetivo**: Prever quanto $ não será recebido no fim do mês

**Modelo**: XGBoost Regressor  
**Features**: Valor em aberto, dias_até_vencimento, perfil_cliente, sazonalidade  
**Métrica**: RMSE, MAE, R²  
**Output**: $ não recebido com intervalo de confiança

### 3. Fluxo de Caixa Projetado

**Objetivo**: Prever entrada de caixa semanal/mensal

**Modelo**: Prophet (séries temporais)  
**Features**: Histórico de recebimentos, sazonalidade, feriados  
**Métrica**: MAPE, backtesting  
**Output**: Forecast 30/60/90 dias com cenários otimista/pessimista

### 4. Validação de Notas Fiscais (RAG)

**Objetivo**: Verificar se nota fiscal atende regras da marca/segmento

**Sistema**: RAG + Vector Search  
**Entrada**: PDF da nota fiscal  
**Processo**: Extrai campos → Recupera regras → Valida → Decide ação  
**Output**: Aprovado / Revisão / Bloqueado + justificativa

---

## 📊 Modelos e Performance

| Modelo | Algoritmo | Métrica Principal | Performance |
|--------|-----------|-------------------|-------------|
| **Classificação Risco** | LightGBM | Precision@Top20% | 0.87 |
| **Regressão Valor** | XGBoost | RMSE | R$ 45k |
| **Forecast Cashflow** | Prophet | MAPE | 8.3% |
| **Segmentação** | K-Means | Silhouette Score | 0.72 |

---

## 🔄 Pipeline de MLOps

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Data        │────▶│  Feature     │────▶│  Training    │
│  Ingestion   │     │  Engineering │     │  Pipeline    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Monitoring  │◀────│  Model       │◀────│  MLflow      │
│  & Alerts    │     │  Serving     │     │  Registry    │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │
       │                    ▼
       │             ┌──────────────┐
       └────────────▶│  Retraining  │
                     │  Trigger     │
                     └──────────────┘
```

### Automação (Databricks Jobs)

- **Daily**: Ingestão de novos dados + scoring batch
- **Weekly**: Retreinamento de modelos + drift detection
- **Monthly**: Auditoria completa + relatórios executivos
- **Real-time**: Validação de notas fiscais via API

---

## 📈 Dashboards e Visualizações

### Dashboard Executivo (AI/BI)

- KPIs: Taxa de inadimplência, valor em risco, forecast vs realizado
- Segmentação: Por marca, tier, região, unidade de negócio
- Alertas: Top 20 clientes em risco crítico
- Tendências: Evolução mensal de inadimplência

### Dashboard de Modelos (MLOps)

- Performance: Accuracy, precision, recall por modelo
- Drift: Feature drift, concept drift ao longo do tempo
- Latência: Tempo de inferência do endpoint
- Volume: Requests/dia, taxa de sucesso

### Genie Space (Self-Service)

- Perguntas em linguagem natural
- "Quais clientes têm mais de 90 dias de atraso?"
- "Qual segmento tem maior inadimplência este mês?"
- "Mostre o forecast de caixa para os próximos 30 dias"

---

## 🛡️ Governança e Segurança

- **Unity Catalog**: Controle de acesso por catalog/schema/table
- **Auditoria**: Lineage completo de dados e modelos
- **Versionamento**: Delta Time Travel + MLflow versioning
- **Compliance**: Logs de inferências para auditoria regulatória

---

## 📞 Suporte e Contato

**Autor**: Valdomiro Vega García  
**LinkedIn**: [linkedin.com/in/valdomiro-vega](https://linkedin.com/in/valdomiro-vega)  
**Email**: valdomirovega@hotmail.com

---

## 📄 Licença

Este projeto é uma demonstração técnica para fins de portfólio e entrevista.  
Dados sintéticos gerados para preservar confidencialidade.

---

**Última atualização**: 02/07/2026  
**Versão**: 1.0.0