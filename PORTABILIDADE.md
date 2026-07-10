
╔═══════════════════════════════════════════════════════════════╗
║         🎯 PORTABILIDADE: LEVAR TUDO PARA OUTRO WORKSPACE     ║
╚═══════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════
✅ O QUE VAI PARA GIT (Portável):
═══════════════════════════════════════════════════════════════

1. ✅ NOTEBOOKS (.ipynb, .py)
   → Vão para Git automaticamente
   → No novo workspace: git clone e pronto!

2. ✅ DASHBOARDS (.lvdash.json)  
   → MOVIDOS para: 05_dashboard/exports/
   → No novo workspace: importar via UI

3. ✅ SCRIPTS (setup, configs, .md)
   → Vão para Git automaticamente

═══════════════════════════════════════════════════════════════
❌ O QUE NÃO VAI (Precisa Recriar):
═══════════════════════════════════════════════════════════════

1. ❌ MLFLOW EXPERIMENTS/RUNS
   • Experimentos ficam no workspace antigo
   • SOLUÇÃO: Re-rodar notebooks de modeling no novo workspace
   • Os NOTEBOOKS já estão no Git → basta executar

2. ❌ UNITY CATALOG TABLES
   • Tables (workspace.risco_bronze.*, etc) não vão
   • SOLUÇÃO: Re-rodar notebooks de ETL (01_setup → 03_bronze)
   • OU exportar DDL (ver abaixo)

3. ❌ JOBS CONFIGURADOS
   • Jobs não vão para Git
   • SOLUÇÃO: Recriar jobs manualmente ou via API

═══════════════════════════════════════════════════════════════
🚀 PLANO DE AÇÃO: NOVO WORKSPACE
═══════════════════════════════════════════════════════════════

WORKSPACE NOVO (Outro Databricks / Outra Conta):

1️⃣ CLONAR GIT (2 min):
   • Git Folders → Clone repo
   • URL: https://github.com/vvegag/ai-credit-risk-platform.git
   • ✅ Todos os notebooks aparecem!

2️⃣ RODAR SETUP (5 min):
   • 01_setup/01_create_schemas.sql
   • Cria: workspace.risco_bronze, risco_silver, risco_gold

3️⃣ RODAR ETL (10-30 min):
   • 02_ingestion → 03_bronze → silver → gold
   • ✅ Recria todas as tables

4️⃣ RODAR MODELING (10 min):
   • 04_modeling/02_modelo_regressao.py
   • 04_modeling/03_modelo_forecast_cashflow.py
   • ✅ Cria novos experiments MLflow no novo workspace

5️⃣ IMPORTAR DASHBOARDS (2 min):
   • No Databricks UI: Dashboards → Import
   • Selecionar: 05_dashboard/exports/*.lvdash.json
   • Ajustar queries (apontar para novo catalog se diferente)

═══════════════════════════════════════════════════════════════
📦 STATUS ATUAL (Workspace Original):
═══════════════════════════════════════════════════════════════

✅ Dashboards movidos para: ai-credit-risk-platform/05_dashboard/exports/
   • Dashboard Executivo - Inadimplência e Risco Financeiro.lvdash.json
   • Dashboard backup

✅ Notebooks de modeling identificados:
   • 04_modeling/02_modelo_regressao.py
   • 04_modeling/03_modelo_forecast_cashflow.py

PRÓXIMO PASSO:
→ Colocar tudo no Git (workflow -TEMP que expliquei antes)
→ No novo workspace: clone + re-run = TUDO recriado!

═══════════════════════════════════════════════════════════════
💡 POR QUE MLflow NÃO VAI?
═══════════════════════════════════════════════════════════════

MLflow experiments são BANCO DE DADOS do workspace, não arquivos.

Mas o CÓDIGO dos models (notebooks) vai! Então:
• Workspace antigo: 50 runs do XGBoost
• Git: Notebook que cria o XGBoost ✅
• Workspace novo: Roda notebook → novos runs criados

Você NÃO perde o código, só o histórico de runs.
Se precisar do histórico, exportar via MLflow API.

═══════════════════════════════════════════════════════════════
🎯 RESUMO ULTRA-SUCINTO:
═══════════════════════════════════════════════════════════════

GIT = Código (notebooks, configs, dashboards .json)
NÃO GIT = Dados gerados (tables, mlflow runs, jobs)

NOVO WORKSPACE:
1. Clone Git
2. Re-run notebooks
3. Tudo recriado!

Código é portável, execuções você refaz.

═══════════════════════════════════════════════════════════════
