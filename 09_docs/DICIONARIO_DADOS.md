# 📚 Dicionário de Dados - Projeto Inadimplência

## 🗂️ Estrutura de Tabelas

### BRONZE LAYER (workspace.risco_bronze)

#### marcas_raw
Cadastro de marcas/clientes B2B

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| codigo_marca | string | Identificador único da marca |
| nome_marca | string | Nome fantasia da marca |
| segmento | string | Segmento de mercado (Moda, Alimentos, etc) |
| unidade_negocio | string | Unidade de negócio CRMBonus |
| tier | string | Classificação (Tier 1-4) |
| data_cadastro | date | Data de cadastro no sistema |

#### clientes_raw
Cadastro de clientes (PF + PJ)

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| codigo_cliente | string | ID único do cliente |
| nome_cliente | string | Nome/Razão Social |
| tipo_pessoa | string | PF ou PJ |
| codigo_marca | string | FK para marcas_raw |
| data_cadastro | date | Data de cadastro |
| perfil_pagamento | string | Pontual/Cronico/Instavel/Risco |

#### faturas_raw
Faturas emitidas

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| titulo | string | Número da fatura (PK) |
| codigo_cliente | string | FK para clientes |
| codigo_marca | string | FK para marcas |
| data_emissao | date | Data de emissão |
| data_vencimento | date | Data de vencimento |
| data_pagamento | date | Data real de pagamento (NULL se pendente) |
| valor_titulo | double | Valor original faturado |
| valor_pagado | double | Valor efetivamente pago |
| status | string | Pago/Pendente/Parcial |

---

### SILVER LAYER (workspace.risco_silver)

#### faturas_enriquecidas
Faturas com cálculos e enriquecimentos

**Todas as colunas do Bronze +**

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| dias_desde_vencimento | int | Dias desde o vencimento |
| dias_atraso | int | Dias de atraso efetivo |
| valor_em_aberto | double | Diferença não paga |
| faixa_atraso | string | Por Vencer, 01-30d, 31-60d, etc |
| status_enriquecido | string | Recebido/Vencido/Risco Médio/Alto/Perda Provável |
| total_faturado_cliente | double | Total histórico por cliente |
| total_pago_cliente | double | Total pago por cliente |
| num_faturas_cliente | long | Quantidade de faturas por cliente |
| taxa_inadimplencia_cliente | double | % inadimplência do cliente |

---

### GOLD LAYER (workspace.risco_gold)

#### predicoes_inadimplencia
Predições do modelo ML

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| codigo_cliente | string | ID do cliente |
| nome_cliente | string | Nome do cliente |
| tipo_pessoa | string | PF/PJ |
| perfil_pagamento | string | Perfil real |
| total_em_aberto | double | Valor pendente total |
| taxa_inadimplencia | double | % inadimplência histórica |
| inadimplente | int | Label real (0/1) |
| predicao_inadimplente | int | Predição do modelo (0/1) |
| prob_inadimplente | double | Probabilidade [0-1] |
| classe_risco | string | Baixo/Médio/Alto/Crítico |

---

### ML FEATURES (workspace.risco_ml_features)

#### features_clientes
Feature Store para modelos ML

| Feature | Tipo | Descrição |
|---------|------|-----------|
| **RFM** | | |
| recencia_dias | int | Dias desde última fatura |
| frequencia_faturas | long | Número total de faturas |
| monetario_total | double | Valor total faturado |
| valor_medio_fatura | double | Ticket médio |
| **Comportamento** | | |
| num_pagas | long | Faturas pagas |
| num_pendentes | long | Faturas pendentes |
| num_parciais | long | Pagamentos parciais |
| **Atrasos** | | |
| media_dias_atraso | double | Média de dias de atraso |
| max_dias_atraso | int | Maior atraso registrado |
| num_atrasos_30plus | long | Atrasos >30 dias |
| num_atrasos_90plus | long | Atrasos >90 dias |
| **Valores** | | |
| total_em_aberto | double | Valor não recebido |
| total_pago | double | Valor recebido |
| taxa_inadimplencia | double | % inadimplência |
| **Risco** | | |
| num_faturas_risco_alto | long | Faturas em risco alto |
| taxa_pagamento | double | % faturas pagas |
| taxa_atraso_frequente | double | % atrasos frequentes |
| valor_medio_em_aberto | double | Ticket médio em aberto |
| **Target** | | |
| inadimplente | int | Label para ML (0/1) |

---

## 📊 Métricas de Negócio

### Taxa de Inadimplência
```
(valor_faturado - valor_pago) / valor_faturado * 100
```

### Perfil de Risco
- **Pontual**: Paga na data ou até 5 dias após
- **Crônico**: Sempre paga com 20-40 dias de atraso
- **Instável**: Comportamento irregular
- **Risco**: >90 dias de atraso ou múltiplas pendências

### Classificação de Probabilidade
- **Baixo**: prob < 0.3
- **Médio**: 0.3 <= prob < 0.6
- **Alto**: 0.6 <= prob < 0.8
- **Crítico**: prob >= 0.8

---

**Autor**: Valdomiro Vega García  
**Data**: 02/07/2026
