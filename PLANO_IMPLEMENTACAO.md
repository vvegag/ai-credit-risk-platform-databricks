
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🎯 PLANO DE IMPLEMENTAÇÃO COMPLETO                         ║
║   AI Credit Risk Platform - Do MVP ao Produto Completo      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

📊 STATUS ATUAL:    40-50% implementado
🎯 META:            95-100% funcional
⏱️  TEMPO ESTIMADO: 12-16 horas (trabalho focado)
📅 PRAZO SUGERIDO:  3-4 dias (4h/dia)

═══════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────┐
│ 🏗️  FASE 1: FUNDAÇÃO - Dados e Infraestrutura Base          │
│    Prioridade: 🔴 CRÍTICA | Tempo: 2-3h                      │
└──────────────────────────────────────────────────────────────┘

✅ O QUE JÁ EXISTE:
• 01_setup/01_criar_catalogo_schemas.py
• 02_ingestion/01_gerar_dados_sinteticos.py
• 03_feature_engineering/01_transformacao_silver.py

🔨 O QUE CRIAR:

📄 Arquivos de Código:
├─ 01_setup/02_configurar_permissoes.py
│  └─ GRANT/REVOKE no Unity Catalog
│      Permissões: SELECT, MODIFY, CREATE
│      Para: workspace.risco_bronze/silver/gold
│
├─ 02_ingestion/02_ingestao_csv_manuais.py
│  └─ Auto Loader para CSVs
│      Schema inference, bad records handling
│
└─ 02_ingestion/03_popular_dados_completos.py
   └─ Expandir geração sintética:
       • 1000 clientes (vs. ~100 atual)
       • 5000 faturas
       • 3000 pagamentos
       • Distribuição realista (70% adimplentes, 30% risco)

📦 Assets a Criar (pasta: 02_ingestion/sample_data/):
├─ notas_fiscais/ (10 PDFs)
│  ├─ nota_001.pdf  (Cliente ABC - R$ 5.000,00)
│  ├─ nota_002.pdf  (Cliente XYZ - R$ 12.000,00)
│  ├─ ... (8 PDFs mais)
│  └─ [Usar library: reportlab para gerar PDFs]
│
├─ clientes.csv (1000 linhas)
│  └─ Colunas: id, nome, cnpj, setor, porte, receita_anual
│
└─ faturas_manuais.csv (500 linhas)
   └─ Colunas: id_fatura, id_cliente, valor, vencimento, status

🗄️ Tabelas Unity Catalog a Popular:
• workspace.risco_bronze.clientes_raw        (1000 registros)
• workspace.risco_bronze.faturas_raw         (5000 registros)
• workspace.risco_bronze.pagamentos_raw      (3000 registros)
• workspace.risco_bronze.notas_fiscais_pdf   (10 PDFs → volume)
• workspace.risco_silver.faturas_enriquecidas (JOIN + limpeza)

✅ CRITÉRIO DE SUCESSO:
SELECT COUNT(*) FROM workspace.risco_silver.faturas_enriquecidas
→ Deve retornar >= 5000 linhas

═══════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────┐
│ 🔧 FASE 2: FEATURE ENGINEERING - Gold Layer                  │
│    Prioridade: 🟠 ALTA | Tempo: 2h | Dep: FASE 1            │
└──────────────────────────────────────────────────────────────┘

🔨 ARQUIVOS A CRIAR:

📄 03_feature_engineering/02_transformacao_gold.py
   └─ Features agregadas:
       • total_faturado_90d, total_faturado_180d, total_faturado_365d
       • count_faturas_pagas, count_faturas_atrasadas
       • avg_dias_atraso, max_dias_atraso
       • taxa_pagamento_pontual (% on time)
       • valor_medio_fatura, desvio_padrao_valores

📄 03_feature_engineering/03_feature_store_rfm.py
   └─ RFM Scoring:
       • Recency: Dias desde última fatura
       • Frequency: Número de faturas em 12 meses
       • Monetary: Valor total faturado
       • Score final: 1-5 (5 = melhor cliente)

📄 03_feature_engineering/04_perfis_comportamentais.py
   └─ Segmentação:
       • Cluster K-Means (4 clusters)
       • Perfis: "Excelente", "Bom", "Risco Moderado", "Alto Risco"
       • Features: payment_behavior, volume_stability, growth_trend

🗄️ Tabela Final:
• workspace.risco_gold.features_ml
  └─ Colunas (30+):
      id_cliente, rfm_score, cluster_risco,
      total_faturado_90d, taxa_pagamento_pontual,
      avg_dias_atraso, count_faturas_atrasadas,
      setor, porte, receita_anual, ... [+ 20 features]

✅ CRITÉRIO DE SUCESSO:
SELECT * FROM workspace.risco_gold.features_ml LIMIT 5
→ Deve ter 30+ colunas sem nulos críticos

═══════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────┐
│ 🤖 FASE 3: ML - Modelo de Classificação XGBoost             │
│    Prioridade: 🟠 ALTA | Tempo: 2-3h | Dep: FASE 2          │
└──────────────────────────────────────────────────────────────┘

🔨 ARQUIVO A CRIAR:

📄 04_modeling/01_modelo_classificacao_risco.py
   └─ Implementação completa:
   
   1️⃣ EDA (Exploratory Data Analysis)
      • Distribuição da variável target (30% inadimplentes)
      • Correlação entre features
      • Outliers e missing values
      • Feature importance inicial
   
   2️⃣ Preparação de Dados
      • Train/Validation/Test split (60/20/20)
      • Normalização de features numéricas
      • Encoding de categóricas (setor, porte)
      • Balanceamento de classes (SMOTE se necessário)
   
   3️⃣ Treinamento XGBoost
      • Hyperparameter tuning (Optuna ou GridSearch)
      • Parâmetros: max_depth, learning_rate, n_estimators
      • Early stopping (validation set)
      • Cross-validation 5-fold
   
   4️⃣ Avaliação
      • Métricas: Precision, Recall, F1, AUC-ROC, AUC-PR
      • Confusion Matrix
      • Classification Report
      • ROC Curve plot
      • Precision-Recall Curve plot
      • Threshold tuning (maximizar F1 ou Recall)
   
   5️⃣ Interpretabilidade
      • SHAP values (top 10 features)
      • Feature importance (gain, cover, weight)
      • Partial Dependence Plots
      • Exemplo de predição com explicação
   
   6️⃣ MLflow Tracking
      • Log params, metrics, artifacts
      • Salvar modelo serializado
      • Registrar feature list
      • Tag: "classification", "production-candidate"

📊 Gráficos a Gerar (salvos como PNG):
├─ confusion_matrix.png
├─ roc_curve.png
├─ precision_recall_curve.png
├─ feature_importance.png
└─ shap_summary_plot.png

🎯 Target de Performance:
• AUC-ROC: >= 0.85
• Precision: >= 0.80
• Recall: >= 0.75
• F1-Score: >= 0.77

✅ CRITÉRIO DE SUCESSO:
MLflow UI mostra experiment com:
- Múltiplos runs (>= 5)
- Best model com AUC-ROC >= 0.85
- Artifacts: model + plots

═══════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────┐
│ 🚀 FASE 4: MLOps - Tracking, Registry, Serving              │
│    Prioridade: 🟡 MÉDIA | Tempo: 2h | Dep: FASE 3           │
└──────────────────────────────────────────────────────────────┘

🔨 ARQUIVOS A CRIAR:

📄 05_mlops/01_mlflow_tracking.py
   └─ Setup de MLflow:
       • Experiment naming convention
       • Auto-logging configurado
       • Custom metrics logging
       • Artifact logging (plots, datasets)
       • Run tagging (git commit, user, environment)

📄 05_mlops/02_model_registry.py
   └─ Model Registry Workflow:
       • Registrar modelo no Registry
       • Transição: None → Staging → Production
       • Versioning automático
       • Model aliases: "champion", "challenger"
       • Approval workflow (comments, tags)

📄 05_mlops/03_model_serving.py
   └─ Deploy do Modelo:
       • Criar Model Serving Endpoint (via API)
       • Configuração: Serverless, Auto-scaling
       • Schema de input/output (JSON)
       • Teste de inferência (100 requests)
       • Latência e throughput monitoring

📄 05_mlops/04_batch_inference.py
   └─ Batch Scoring Pipeline:
       • Carregar modelo do Registry (production stage)
       • Score toda a base de clientes
       • Salvar em: workspace.risco_gold.scores_clientes
       • Colunas: id_cliente, probabilidade_inadimplencia,
                  classe_risco, score_date, model_version
       • Agendar via Databricks Job (diário às 6am)

✅ CRITÉRIO DE SUCESSO:
• Model Serving endpoint retorna predição em < 200ms
• Batch scoring processa 1000 clientes em < 2min
• Model Registry tem >= 2 versões (Staging + Production)

═══════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────┐
│ 🔍 FASE 5: RAG Completo - Vector Search + Orquestração      │
│    Prioridade: 🟡 MÉDIA | Tempo: 2-3h | Dep: FASE 1         │
└──────────────────────────────────────────────────────────────┘

🔨 ARQUIVOS A CRIAR:

📄 06_rag_validation/02_vector_search_setup.py
   └─ Databricks Vector Search:
       • Criar Vector Search endpoint
       • Index: notas_fiscais_embeddings
       • Embedding model: sentence-transformers
       • Dimensão: 384 (all-MiniLM-L6-v2)
       • Sync automático com Delta table

📄 06_rag_validation/03_validacao_notas_fiscais.py
   └─ Pipeline de Validação:
       • Parse PDF → Extract text
       • Generate embeddings
       • Similarity search (top-3 matches)
       • LLM validation (via Databricks Foundation Models)
       • Score: 0-100 (confiança da validação)
       • Log em: workspace.risco_silver.validacoes_rag

📄 06_rag_validation/04_workflow_orquestracao.py
   └─ Databricks Workflow:
       • Task 1: Ingest novos PDFs
       • Task 2: Extract + Embed
       • Task 3: Validate via RAG
       • Task 4: Alerta se score < 70
       • Trigger: Manual ou via API

📦 Assets Adicionais:
├─ 06_rag_validation/prompts/
│  ├─ validation_prompt.txt
│  └─ extraction_prompt.txt
└─ 06_rag_validation/sample_queries/
   └─ test_queries.json (10 queries de teste)

✅ CRITÉRIO DE SUCESSO:
Query: "Notas fiscais do cliente X acima de R$ 10k"
→ Retorna top-3 matches com score >= 0.85

═══════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────┐
│ 📊 FASE 6: Monitoramento Avançado + Alertas                 │
│    Prioridade: 🟡 MÉDIA | Tempo: 2h | Dep: FASE 4           │
└──────────────────────────────────────────────────────────────┘

🔨 ARQUIVOS A CRIAR:

📄 07_monitoring/02_performance_monitoring.py
   └─ Dashboard de Métricas:
       • Accuracy, Precision, Recall over time
       • Prediction distribution drift
       • Feature drift (KS-test por feature)
       • Model latency (p50, p95, p99)
       • Error rate tracking
       • Salvar em: workspace.risco_gold.model_metrics_history

📄 07_monitoring/03_alertas_automaticos.py
   └─ Sistema de Alertas:
       • Trigger 1: Drift detected (KS > 0.2)
       • Trigger 2: Accuracy drop (< 80%)
       • Trigger 3: High error rate (> 5%)
       • Trigger 4: Cliente crítico previsto (score > 0.9)
       • Canal: Slack webhook + Email (via Databricks Alerts)

📄 07_monitoring/04_inference_table.py
   └─ Logging de Inferências:
       • workspace.risco_gold.inference_log
       • Colunas: timestamp, model_version, input_features,
                  prediction, probability, latency_ms
       • Retenção: 90 dias
       • Indexado por: timestamp, id_cliente

🔔 Configurar Databricks Alerts:
• Alert 1: "Drift Detectado" → Canal #data-alerts
• Alert 2: "Cliente Alto Risco" → Canal #collections-team

✅ CRITÉRIO DE SUCESSO:
Simular drift artificial → Alerta dispara em < 5min

═══════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────┐
│ 📈 FASE 7: Dashboards e Automação                           │
│    Prioridade: 🟢 BAIXA | Tempo: 2h | Dep: FASES 1-6        │
└──────────────────────────────────────────────────────────────┘

🔨 ARQUIVOS A CRIAR:

📄 08_dashboards/01_dashboard_executivo.py
   └─ Criar via Databricks API:
       • KPIs: Total clientes, % inadimplentes, receita em risco
       • Gráficos: Trend inadimplência, Top 10 clientes risco
       • Filtros: Data range, Setor, Porte
       • Export: dashboard_executivo.lvdash.json

📄 08_dashboards/02_dashboard_modelos.py
   └─ Dashboard de ML:
       • Métricas: AUC-ROC, Precision, Recall (histórico)
       • Feature importance atual
       • Drift detection status
       • Predições recentes (últimas 24h)

📄 08_dashboards/03_genie_space_setup.py
   └─ Genie Space Configuration:
       • Conectar tables: features_ml, scores_clientes
       • Sample queries: "Quem são os top 20 clientes em risco?"
       • Permissions: Read-only para business users

🤖 Databricks Jobs a Criar:

├─ Job 1: "Weekly Model Retraining"
│  ├─ Schedule: Domingos 2am
│  ├─ Tasks: 
│  │   1. Feature engineering (Gold)
│  │   2. Train model (04_modeling/01_*.py)
│  │   3. Evaluate & register
│  │   4. Promote to Production (se AUC > threshold)
│  └─ Alerts: Email se falhar
│
├─ Job 2: "Daily Batch Scoring"
│  ├─ Schedule: Diário 6am
│  ├─ Tasks:
│  │   1. Load production model
│  │   2. Score all active clients
│  │   3. Update scores table
│  │   4. Trigger alerts para high-risk
│  └─ SLA: < 30min
│
└─ Job 3: "Hourly Drift Check"
   ├─ Schedule: A cada hora
   ├─ Tasks:
   │   1. Compare distributions (últimas 24h vs. baseline)
   │   2. Run KS-test
   │   3. Log results
   │   4. Alert se drift > threshold
   └─ Timeout: 5min

✅ CRITÉRIO DE SUCESSO:
• 3 Jobs ativos no Databricks Workflows
• Genie Space responde query em < 3s
• Dashboard atualiza automaticamente

═══════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────┐
│ 📚 FASE 8: Documentação Final + Assets Visuais              │
│    Prioridade: 🟢 BAIXA | Tempo: 2h | Dep: FASES 1-7        │
└──────────────────────────────────────────────────────────────┘

📦 ASSETS A CRIAR:

📸 Screenshots (pasta: 09_docs/screenshots/):
├─ dashboard_executivo_overview.png
├─ dashboard_modelos_metrics.png
├─ mlflow_experiments_list.png
├─ model_serving_endpoint.png
├─ genie_space_example_query.png
├─ drift_detection_alert.png
└─ unity_catalog_structure.png

🎨 Diagramas (pasta: 09_docs/diagrams/):
├─ arquitetura_completa.png (draw.io → export)
│  └─ Camadas: Ingestion → Bronze → Silver → Gold → ML → Serving
├─ fluxo_mlops.png
│  └─ Train → Register → Serve → Monitor → Retrain
└─ pipeline_rag.png
   └─ PDF → Parse → Embed → Vector Search → Validate

🎥 Vídeo Demo (Loom 3-5min):
└─ Roteiro:
    1. Intro: Visão geral do projeto (30s)
    2. Dados: Mostrar tabelas Unity Catalog (30s)
    3. ML: Rodar modelo e explicar SHAP (60s)
    4. Dashboard: Navegar por KPIs e filtros (60s)
    5. MLOps: Mostrar Model Registry e Serving (60s)
    6. Conclusão: Roadmap e diferenciais (30s)

📄 Documentos:
├─ 09_docs/FAQ.md
│  └─ 20 perguntas frequentes
│      Ex: "Como reprovar a arquitetura?"
│          "Qual o custo de serverless compute?"
│          "Como adicionar novos features?"
│
├─ 09_docs/CONTRIBUTING.md
│  └─ Guia para colaboradores:
│      • Coding standards (PEP8, docstrings)
│      • Branch strategy (main, develop, feature/*)
│      • PR template
│      • Testing requirements
│
├─ 09_docs/CHANGELOG.md
│  └─ Histórico de versões:
│      v1.0.0 - Initial release (MVP)
│      v1.1.0 - Added classification model
│      v1.2.0 - MLOps pipeline complete
│      v2.0.0 - Production-ready (FASE 8 completa)
│
└─ README.md (ATUALIZAR)
   └─ Adicionar:
       • Badges reais: Build Status, Coverage, License
       • GIFs/Screenshots embedded
       • Link para vídeo demo
       • Atualizar roadmap: ✅ Phase 1, ✅ Phase 2, 🚧 Phase 3

🧪 TESTE FINAL:
1. Criar novo workspace (ou usar outra conta)
2. Git clone do projeto
3. Seguir README step-by-step
4. Validar que tudo funciona
5. Documentar qualquer erro/gap

✅ CRITÉRIO DE SUCESSO:
• README permite setup 0→100 em < 1h (usuário tech)
• Vídeo demo recebe feedback positivo (3+ pessoas)
• Projeto tem 100% das features documentadas

═══════════════════════════════════════════════════════════════

📊 RESUMO EXECUTIVO: 8 FASES
═══════════════════════════════════════════════════════════════

FASE 1: Fundação (2-3h)          → 1000+ records, 10 PDFs, UC setup
FASE 2: Gold Layer (2h)          → 30+ features, RFM, clusters
FASE 3: ML Classificação (3h)    → XGBoost, AUC 0.85+, SHAP
FASE 4: MLOps (2h)               → Registry, Serving, Batch
FASE 5: RAG (3h)                 → Vector Search, validação
FASE 6: Monitoramento (2h)       → Drift, alertas, logging
FASE 7: Dashboards/Jobs (2h)     → 3 dashboards, 3 jobs
FASE 8: Docs/Assets (2h)         → Screenshots, vídeo, FAQ

───────────────────────────────────────────────────────────────
TOTAL: 16-18 horas | 4 dias (4h/dia) | Projeto 95%+ completo
───────────────────────────────────────────────────────────────

🎯 ORDEM RECOMENDADA (Dependências):

DIA 1 (4h): FASE 1 completa → FASE 2 começa
DIA 2 (4h): FASE 2 completa → FASE 3 completa
DIA 3 (4h): FASE 4 completa → FASE 5 completa
DIA 4 (4h): FASE 6 → FASE 7 → FASE 8 (parte)
DIA 5 (2h): FASE 8 completa + testes finais

═══════════════════════════════════════════════════════════════

🚀 PRONTO PARA COMEÇAR?

Comando para começar:
"Iniciar FASE 1" → Vou criar todos os arquivos e dados base!

═══════════════════════════════════════════════════════════════
