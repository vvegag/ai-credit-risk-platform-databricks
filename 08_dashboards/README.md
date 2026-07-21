# 📊 Dashboards - Módulo 08

> Dashboards executivos e analíticos do projeto AI Credit Risk Platform

## 📋 Visão Geral
Esta pasta contém os dashboards Lakeview (Databricks) para visualização e análise de risco de crédito e inadimplência.

## 📁 Estrutura

```
08_dashboards/
├── exports/                              # Exports JSON dos dashboards
│   ├── Dashboard Executivo - Inadimplência e Risco Financeiro.lvdash.json
│   └── Dashboard Risco Financeiro - Inadimplência 2026-07-02T13-14-37.lvdash.json
└── README.md                             # Este arquivo
```

## 🎯 Dashboards Disponíveis

### 1. Dashboard Executivo - Inadimplência e Risco Financeiro
**Arquivo**: `Dashboard Executivo - Inadimplência e Risco Financeiro.lvdash.json`  
**Páginas**: 2  
**Tamanho**: 27KB

**Conteúdo**:
- **Página 1 - Visão Executiva**:
  - KPIs principais (inadimplência, exposição, tickets)
  - Evolução temporal de inadimplência
  - Distribuição por faixa de atraso
  - Top clientes em risco
  
- **Página 2 - Análise Detalhada**:
  - Segmentação por perfil de pagamento
  - Análise por marca/segmento
  - Predições do modelo ML
  - Métricas de performance (precision, recall)

**Fonte de Dados**: 
- `workspace.risco_gold.predicoes_inadimplencia`
- `workspace.risco_silver.faturas_enriquecidas`

---

### 2. Dashboard Risco Financeiro - Inadimplência (Legacy)
**Arquivo**: `Dashboard Risco Financeiro - Inadimplência 2026-07-02T13-14-37.lvdash.json`  
**Páginas**: 1  
**Tamanho**: 512B

**Status**: ⚠️ Versão antiga/minimalista (manter para referência)

---

## 🚀 Como Importar no Databricks

### Via UI (Recomendado)
1. Acesse **Dashboards** no workspace
2. Clique em **Import**
3. Selecione o arquivo `.lvdash.json`
4. Aguarde a importação
5. Configure as conexões de dados se necessário

### Via API
```python
import requests
import json

workspace_url = "https://<workspace>.cloud.databricks.com"
token = dbutils.secrets.get(scope="<scope>", key="<key>")

with open("exports/Dashboard Executivo - Inadimplência e Risco Financeiro.lvdash.json") as f:
    dashboard_json = json.load(f)

response = requests.post(
    f"{workspace_url}/api/2.0/lakeview/dashboards/create",
    headers={"Authorization": f"Bearer {token}"},
    json=dashboard_json
)
```

---

## 📊 Fontes de Dados

Todos os dashboards consomem dados das seguintes camadas:

### Gold Layer
- `workspace.risco_gold.predicoes_inadimplencia` - Predições ML
- `workspace.risco_gold.metricas_inadimplencia` - Agregações

### Silver Layer
- `workspace.risco_silver.faturas_enriquecidas` - Faturas enriquecidas
- `workspace.risco_silver.clientes_enriquecidos` - Perfis de cliente

---

## 🔄 Atualização dos Dashboards

Os dashboards são atualizados automaticamente conforme:
- Novos dados chegam via pipeline de ingestão
- Modelos ML são retreinados
- Novas predições são geradas

**Frequência de refresh**: Depende do schedule configurado no dashboard

---

## 🛠️ Manutenção

### Exportar Dashboard Atualizado
1. Abra o dashboard no Databricks
2. Menu `⋮` → **Export**
3. Salve na pasta `exports/`
4. Commit no Git

### Criar Novo Dashboard
1. Crie no Databricks UI
2. Desenvolva visualizações
3. Export como `.lvdash.json`
4. Adicione à pasta `exports/`
5. Documente neste README

---

## 📈 Métricas Monitoradas

### KPIs Principais
- **Taxa de Inadimplência Global**: % do valor total inadimplente
- **Valor em Risco**: Montante total em aberto
- **Clientes em Risco**: Quantidade com prob > 0.6
- **Ticket Médio Inadimplente**: Valor médio das faturas em atraso

### Segmentações
- Por faixa de atraso (0-30d, 31-60d, 61-90d, 90+d)
- Por perfil de pagamento (Pontual, Crônico, Instável, Risco)
- Por segmento de marca
- Por tier de cliente

---

## 🔗 Projeto Completo

Este módulo faz parte do **AI Credit Risk Platform**:
- `01_setup/` - Configuração inicial
- `02_ingestion/` - Ingestão de dados
- `03_feature_engineering/` - Feature engineering
- `04_modeling/` - Modelagem ML
- `05_mlops/` - MLOps e monitoramento
- `06_rag_validation/` - Validação RAG
- `07_monitoring/` - Monitoring
- **`08_dashboards/`** - Dashboards (você está aqui)
- `09_docs/` - Documentação
- `10_rag_agent/` - RAG Agent

---

## 👥 Autor
Valdomiro Vega García - Databricks

## 📝 Última Atualização
20/07/2026
