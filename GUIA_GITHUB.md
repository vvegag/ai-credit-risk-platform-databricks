# 🚀 GUIA COMPLETO: Renomear Projeto e Publicar no GitHub

## 📋 DECISÃO: Escolher o Nome do Projeto

### Opções Sugeridas:
1. **`ai-credit-risk-platform`** ⭐ (Recomendado)
2. `smart-credit-analytics`
3. `ml-financial-risk-engine`
4. `intelligent-delinquency-predictor`
5. `lakehouse-credit-risk`

**Escolhido**: `ai-credit-risk-platform`

---

## 🔄 PARTE 1: Renomear a Pasta no Databricks

### Opção A: Via Interface (Mais Fácil)
1. No Databricks Workspace, navegue até `/Users/valdomirovega@hotmail.com/`
2. Clique com botão direito em `projeto_inadimplencia`
3. Selecione "Rename"
4. Digite o novo nome: `ai-credit-risk-platform`
5. Confirme

### Opção B: Via CLI (Alternativa)
```bash
# Nota: Databricks não tem comando mv direto para pastas
# É mais fácil usar a interface
```

---

## 📝 PARTE 2: Atualizar Documentos com o Novo Nome

Vamos atualizar todas as referências de "projeto_inadimplencia" para "ai-credit-risk-platform":

### Arquivos a atualizar:
- ✅ `README.md` (principal)
- ✅ `RESUMO_PROJETO_INADIMPLENCIA.md` (renomear para `PROJECT_SUMMARY.md`)
- ✅ `GUIA_GITHUB.md` (este arquivo)
- ✅ Todos os notebooks que mencionam o nome do projeto

---

## 🔧 PARTE 3: Configurar Git Local

### Passo 1: Configurar Credenciais Git no Databricks

1. **Via UI (Recomendado)**:
   - Clique no seu avatar (canto superior direito)
   - Vá em **Settings** → **User Settings** → **Git Integration**
   - Clique em **Add Git Credential**
   - Escolha **GitHub**
   - Preencha:
     - **Git username**: `vvegag`
     - **Token**: Você precisa gerar um Personal Access Token (PAT) no GitHub

2. **Como gerar GitHub Personal Access Token**:
   - Vá em https://github.com/settings/tokens
   - Clique em "Generate new token" → "Generate new token (classic)"
   - Nome: `Databricks Access`
   - Marque os scopes:
     - ✅ `repo` (todos os sub-itens)
     - ✅ `workflow`
   - Clique em "Generate token"
   - **COPIE O TOKEN** (você só verá uma vez!)
   - Cole no Databricks

### Passo 2: Criar Repositório no GitHub

1. Vá em https://github.com/vvegag
2. Clique em **New repository**
3. Preencha:
   - **Repository name**: `ai-credit-risk-platform`
   - **Description**: `End-to-end ML platform for credit risk prediction using Databricks Lakehouse`
   - **Visibility**: Public (ou Private, sua escolha)
   - **NÃO** marque "Initialize with README" (já temos um)
   - **NÃO** adicione .gitignore nem LICENSE ainda
4. Clique em **Create repository**
5. **COPIE a URL**: `https://github.com/vvegag/ai-credit-risk-platform.git`

---

## 📤 PARTE 4: Conectar Databricks ao GitHub

### Método 1: Via Git Folders (Recomendado para Databricks)

1. **No Databricks Workspace**:
   - Clique em **Workspace** no menu lateral
   - Navegue até `/Users/valdomirovega@hotmail.com/`
   - Clique em **⋮** (três pontos) ao lado de `ai-credit-risk-platform`
   - Selecione **Create Git Folder**

2. **Configurar Git Folder**:
   - **Git provider**: GitHub
   - **Repository URL**: `https://github.com/vvegag/ai-credit-risk-platform.git`
   - **Git credential**: Selecione a credencial configurada no Passo 1
   - **Branch**: `main` (ou `master`, dependendo do padrão do GitHub)
   - Clique em **Create Git Folder**

3. **Mover conteúdo**:
   - Databricks criará uma nova pasta Git Folder
   - Você precisará **mover** o conteúdo da pasta antiga para a nova Git Folder
   - Ou simplesmente **clone o repo vazio** e copie seus arquivos para dentro

### Método 2: Via CLI no Notebook (Alternativa)

**IMPORTANTE**: Este método funciona, mas o Método 1 é mais integrado ao Databricks.

```bash
# Executar no notebook com %sh

cd /Workspace/Users/valdomirovega@hotmail.com/ai-credit-risk-platform

# Inicializar Git
git init

# Configurar usuário
git config user.name "vvegag"
git config user.email "valdomirovega@hotmail.com"

# Adicionar todos os arquivos
git add .

# Primeiro commit
git commit -m "Initial commit: AI Credit Risk Platform"

# Adicionar remote
git remote add origin https://github.com/vvegag/ai-credit-risk-platform.git

# Push (você será solicitado a autenticar)
git push -u origin main
```

**Nota sobre autenticação**: Use o Personal Access Token como senha.

---

## 📋 PARTE 5: Preparar Arquivos para GitHub

### Arquivos Essenciais:

1. **`.gitignore`** (ignorar arquivos desnecessários)
```gitignore
# Databricks internals
.databricks/
__pycache__/
*.pyc
*.pyo
*.egg-info/
.ipynb_checkpoints/

# MLflow
mlruns/
mlartifacts/

# Data files (não commitar dados sensíveis)
*.csv
*.parquet
*.json
data/

# Secrets
.env
*.key
*.pem

# OS
.DS_Store
Thumbs.db
```

2. **`LICENSE`** (escolha uma licença)
```text
MIT License

Copyright (c) 2026 Valdomiro Vega

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

3. **`README.md`** atualizado (já temos, mas vamos melhorar)

4. **`.github/workflows/`** (opcional: CI/CD)

---

## 🎯 PARTE 6: Estrutura Final do Repositório

```
ai-credit-risk-platform/
├── .gitignore
├── LICENSE
├── README.md
├── PROJECT_SUMMARY.md          # Resumo executivo
├── ARCHITECTURE.md             # Arquitetura detalhada
│
├── 01_setup/
│   └── notebooks...
│
├── 02_ingestion/
│   └── notebooks...
│
├── 03_feature_engineering/
│   └── notebooks...
│
├── 04_modeling/
│   ├── 01_modelo_classificacao.ipynb
│   ├── 02_modelo_regressao.ipynb
│   └── 03_modelo_forecast_cashflow.ipynb
│
├── 05_mlops/
│   └── notebooks...
│
├── 06_rag_validation/
│   └── 01_rag_notas_fiscais.ipynb
│
├── 07_monitoring/
│   └── 01_drift_detection.ipynb
│
├── 08_dashboards/
│   └── dashboard_exports...
│
├── 09_docs/
│   ├── images/              # Screenshots do dashboard
│   ├── diagrams/            # Diagramas de arquitetura
│   └── presentations/       # Slides de apresentação
│
└── config/
    ├── cluster_config.json  # Configurações de cluster
    └── requirements.txt     # Dependências Python
```

---

## ✅ PARTE 7: Checklist Final Antes do Push

- [ ] Renomear pasta no Databricks: `projeto_inadimplencia` → `ai-credit-risk-platform`
- [ ] Criar repositório no GitHub: `https://github.com/vvegag/ai-credit-risk-platform`
- [ ] Gerar Personal Access Token no GitHub
- [ ] Configurar Git Credentials no Databricks
- [ ] Atualizar README.md com novo nome
- [ ] Renomear `RESUMO_PROJETO_INADIMPLENCIA.md` → `PROJECT_SUMMARY.md`
- [ ] Criar `.gitignore`
- [ ] Adicionar LICENSE
- [ ] Remover dados sensíveis (se houver)
- [ ] Verificar se todos os notebooks funcionam
- [ ] Exportar screenshots do dashboard para `09_docs/images/`
- [ ] Criar Git Folder no Databricks
- [ ] Mover arquivos para Git Folder
- [ ] Commit inicial
- [ ] Push para GitHub
- [ ] Verificar repositório no GitHub
- [ ] Atualizar README com badges e links

---

## 🎨 PARTE 8: Melhorias no README para GitHub

Adicione ao topo do README:

```markdown
# 🤖 AI Credit Risk Platform

[![Databricks](https://img.shields.io/badge/Databricks-Lakehouse-red)](https://databricks.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org)
[![Delta Lake](https://img.shields.io/badge/Delta-Lake-green)](https://delta.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> End-to-end Machine Learning platform for credit risk prediction, delinquency forecasting, and automated document validation using Databricks Lakehouse.

![Dashboard Preview](09_docs/images/dashboard_preview.png)

## 🌟 Features

- 🎯 **Credit Risk Prediction**: XGBoost classifier with 88% AUC
- 💰 **Value at Risk Estimation**: Regression model for monetary risk prediction
- 📈 **Cashflow Forecasting**: Prophet-based time series prediction (12-month horizon)
- 🤖 **RAG System**: Automated invoice validation using embeddings + FAISS
- 📊 **Executive Dashboard**: Interactive AI/BI dashboards with filters
- 🔍 **Drift Detection**: Continuous monitoring with KS tests
- 🏗️ **Medallion Architecture**: Bronze-Silver-Gold data layers
- 🔒 **Unity Catalog**: Governance and lineage tracking

## 🚀 Quick Start

[Instruções de setup...]

## 📊 Results

- **60 critical clients** identified proactively
- **R$ 9M in risk** mapped and monitored
- **21.51% delinquency rate** tracked
- **75% automation** in invoice validation
- **80% time reduction** (2h → 24min per batch)

## 📚 Documentation

- [Project Summary](PROJECT_SUMMARY.md) - Executive overview
- [Architecture](ARCHITECTURE.md) - Technical architecture
- [Model Cards](09_docs/model_cards/) - Model documentation

## 🛠️ Tech Stack

- **Platform**: Databricks (Unity Catalog, Delta Lake, MLflow)
- **Languages**: Python, SQL, PySpark
- **ML Libraries**: XGBoost, Prophet, Scikit-learn, Sentence-Transformers
- **Visualization**: Databricks AI/BI, Plotly

## 📧 Contact

**Valdomiro Vega**  
Email: valdomirovega@hotmail.com  
GitHub: [@vvegag](https://github.com/vvegag)  
LinkedIn: [valdomiro-vega](https://linkedin.com/in/valdomiro-vega)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

---

## 🤔 PARTE 9: FAQ

### P: Preciso commitar dados sensíveis?
**R**: NÃO! Use `.gitignore` para excluir:
- CSVs, Parquets com dados reais
- Credenciais, tokens, keys
- Apenas commit código e documentação

### P: Como atualizar o repositório depois?
**R**: No Databricks Git Folder:
1. Faça suas mudanças nos notebooks
2. Clique em **Git** no topo da pasta
3. **Commit & Push** com mensagem descritiva

### P: E se eu quiser trabalhar localmente?
**R**: Clone o repo:
```bash
git clone https://github.com/vvegag/ai-credit-risk-platform.git
cd ai-credit-risk-platform
```

### P: Como adiciono colaboradores?
**R**: No GitHub:
1. Vá em **Settings** → **Collaborators**
2. Clique em **Add people**
3. Digite o username

---

## 📞 Próximos Passos

1. **Escolher o nome definitivo** do projeto
2. **Renomear a pasta** no Databricks
3. **Criar o repositório** no GitHub
4. **Configurar Git Folder** no Databricks
5. **Fazer o primeiro push**
6. **Adicionar badges e screenshots** ao README
7. **Compartilhar o link** no LinkedIn! 🎉

---

**🎯 Seu repositório ficará assim**:  
`https://github.com/vvegag/ai-credit-risk-platform`

**Pronto para impressionar recrutadores! 🚀**
