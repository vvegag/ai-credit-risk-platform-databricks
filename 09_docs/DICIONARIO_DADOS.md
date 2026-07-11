# 📚 Data Dictionary — AI Credit Risk Platform

Catalog: `credit_risk` (name parameterized via a `catalog` widget in every notebook).

## 🥉 BRONZE (`credit_risk.bronze`)

#### `clientes`
| Column | Type | Description |
|---|---|---|
| id_cliente | int | Client ID (PK) |
| nome | string | Company/client name |
| cnpj | string | Tax ID |
| setor | string | Industry sector |
| porte | string | Small / Medium / Large |
| receita_anual | long | Annual revenue |
| score_risco | int | Initial synthetic risk score |
| categoria_risco | string | Low / Medium / High (synthetic generation label) |
| data_cadastro | string | Registration date |

#### `faturas` (partitioned by `ano_mes_emissao`)
| Column | Type | Description |
|---|---|---|
| id_fatura | int | Invoice ID (PK) |
| id_cliente | int | FK → clientes |
| valor | double | Invoice amount |
| data_emissao | string | Issue date |
| data_vencimento | string | Due date |
| data_pagamento | string | Actual payment date (null if unpaid) |
| status | string | Paga / Pendente / Atrasada |
| dias_atraso | int | Days past due |
| ano_mes_emissao | string | `YYYY-MM` — partition column |

#### `pagamentos`
| Column | Type | Description |
|---|---|---|
| id_pagamento | int | Payment ID (PK) |
| id_fatura | int | FK → faturas |
| id_cliente | int | FK → clientes |
| valor | double | Amount paid |
| data_pagamento | string | Payment date |
| forma_pagamento | string | Boleto / Transferência / Cartão / PIX |

---

## 🥈 SILVER (`credit_risk.silver`)

#### `clientes`
Same columns as Bronze, deduplicated and trimmed.

#### `faturas_enriquecidas` (partitioned by `ano_mes_emissao`)
All `faturas` columns, with `valor` renamed to `valor_total`, plus:

| Column | Type | Description |
|---|---|---|
| valor_pago_total | double | Sum of payments matched to this invoice |
| forma_pagamento | string | Payment method (from `pagamentos`) |
| pago_flag | int | 1 if `status = 'Paga'` |
| valor_em_aberto | double | `valor_total - valor_pago_total` when unpaid, else 0 |

---

## 🥇 GOLD (`credit_risk.gold`)

#### `features_agregadas`
Client columns + temporal aggregates over `faturas_enriquecidas`:

| Column | Type | Description |
|---|---|---|
| total_faturado_90d / 180d / 365d | double | Revenue billed in each trailing window |
| count_faturas_total / pagas / pendentes / atrasadas | long | Invoice counts by status |
| valor_medio_fatura, desvio_padrao_valores, valor_minimo_fatura, valor_maximo_fatura | double | Invoice value statistics |
| taxa_pagamento | double | % of invoices paid |
| taxa_inadimplencia | double | % of invoices past due |

#### `features_rfm`
All `features_agregadas` columns +

| Column | Type | Description |
|---|---|---|
| recency_dias | int | Days since the client's last invoice |
| rfm_score | int | 1 (worst) – 5 (best) |
| categoria_rfm | string | Premium / Regular / Em Risco |

#### `features_ml` ⭐ (final feature table used by every model in `04_modeling/`)
All `features_rfm` columns +

| Column | Type | Description |
|---|---|---|
| cluster | int | K-Means cluster id (k=4) |
| perfil_comportamental | string | Alto Risco / Médio Risco / Baixo Risco / Premium |

#### `model_predictions`
Output of `04_modeling/01_modelo_classificacao_risco.py`.

| Column | Type | Description |
|---|---|---|
| id_cliente | int | Client ID |
| perfil_comportamental | string | Behavioral profile at prediction time |
| probabilidade_inadimplencia | double | Predicted probability [0–1] |
| predicao_inadimplente | int | Predicted class (0/1) |
| perfil_real | int | Ground-truth label used in training |

#### `previsao_valor_inadimplente`
Output of `04_modeling/02_modelo_regressao.py`: `id_cliente`, `valor_em_risco` (actual),
`valor_previsto`, `erro_previsao`, `categoria_risco_monetario`.

#### `forecast_cashflow`
Output of `04_modeling/03_modelo_forecast_cashflow.py`: `data_prevista`, `cashflow_previsto`,
`cashflow_min`, `cashflow_max`, `risco_cashflow`, `dias_futuro`, `janela`.

#### `validacao_notas_fiscais`
Output of `06_rag_validation/01_rag_notas_fiscais.py` (prototype): `id_fatura`, `id_cliente`,
`valor_nf`, `valor_fatura`, `status`, `criticidade`, `issues`, `num_issues`, `data_upload`.

Source text: each row's synthetic invoice text is written to a `.txt` file under
`02_ingestion/sample_data/notas_fiscais_txt/` (stand-in for an already-parsed physical document),
then read back from disk before embedding — no chunking, since each invoice is already a short,
atomic text.

#### `monitoring_logs`
Output of `07_monitoring/01_drift_detection.py`: one row per run with drift/prediction summary
stats (`percentual_drift`, `risco_critico_pct`, `alerta_drift`, `alerta_risco`, `status_geral`, ...).

#### `model_metrics`, `model_alerts`, `model_versions`
Produced by `05_mlops/01_mlops_pipeline.py` — production metrics history, active alerts, and
model version/governance records. See that notebook for the exact schema (it's self-documenting:
each table is created right before first use, with an explicit `CREATE TABLE`/`saveAsTable` call).

---

## 📊 Business Metrics

**Delinquency rate**: `(valor_faturado - valor_pago) / valor_faturado * 100`

**Behavioral profile** (K-Means cluster label, `features_ml.perfil_comportamental`):
- **Baixo Risco**: high payment rate, low delinquency
- **Premium**: high billing volume, moderate delinquency
- **Médio Risco**: intermediate profile
- **Alto Risco**: high delinquency rate, low recent activity

**Classification target** (`04_modeling/01_modelo_classificacao_risco.py`): binary, defined as
`taxa_inadimplencia > 40%`.
