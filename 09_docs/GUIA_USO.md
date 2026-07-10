# 📖 Guia de Uso - Projeto Inadimplência

## 🚀 Início Rápido

### 1. Executar Setup (Uma Vez)

```python
# Criar schemas no Unity Catalog
# Já foi executado - schemas criados em workspace.risco_*
```

### 2. Gerar Dados (Primeira Vez)

Os dados sintéticos já foram gerados:
- 20 marcas
- 300 clientes (200 PJ, 100 PF)
- 4,486 faturas
- R$ 85.4M faturado
- R$ 11.5M inadimplente (13.46%)

### 3. Consultar Dados

```sql
-- Ver faturas enriquecidas
SELECT * FROM workspace.risco_silver.faturas_enriquecidas
WHERE status_enriquecido = 'Risco Alto'
LIMIT 10;

-- Ver predições de risco
SELECT 
    codigo_cliente,
    nome_cliente,
    total_em_aberto,
    prob_inadimplente,
    classe_risco
FROM workspace.risco_gold.predicoes_inadimplencia
WHERE classe_risco = 'Crítico'
ORDER BY total_em_aberto DESC;
```

---

## 🔬 Como Usar o Modelo ML

### Carregar Modelo do MLflow

```python
import mlflow

# Carregar modelo treinado
run_id = "9bc188f497b24fa0ba99b7352361450b"  # ID do run
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# Fazer predição
import pandas as pd
novo_cliente = pd.DataFrame({
    'recencia_dias': [45],
    'frequencia_faturas': [12],
    'monetario_total': [150000],
    # ... outras 14 features
})

probabilidade = model.predict_proba(novo_cliente)[0, 1]
print(f"Probabilidade de inadimplência: {probabilidade:.2%}")
```

### Batch Scoring

```python
# Carregar features
df = spark.table("workspace.risco_ml_features.features_clientes").toPandas()
X = df[feature_cols].fillna(0)

# Predições
df['prob_risco'] = model.predict_proba(X)[:, 1]

# Salvar resultados
spark.createDataFrame(df).write.mode("overwrite") \
    .saveAsTable("workspace.risco_gold.scores_atualizados")
```

---

## 📊 Dashboards e Visualizações

### Dashboard Executivo

Criar em AI/BI Dashboards com tabela `workspace.risco_gold.predicoes_inadimplencia`:

**KPIs Principais:**
- Total em Risco (soma de total_em_aberto onde prob > 0.6)
- Taxa de Inadimplência Média
- Clientes em Risco Crítico (count onde classe_risco = 'Crítico')

**Visualizações:**
1. **Gráfico de Pizza**: Distribuição por classe_risco
2. **Gráfico de Barras**: Top 20 clientes por prob_inadimplente
3. **Tabela Detalhada**: Filtros por perfil, tipo_pessoa, risco

### Genie Space

Criar Genie Space apontando para `workspace.risco_gold.*`:

**Perguntas de Exemplo:**
- "Quais clientes têm mais de R$ 100k em aberto?"
- "Qual a taxa de inadimplência por perfil de pagamento?"
- "Mostre os 10 clientes de maior risco"

---

## 🔄 Pipeline de Retreinamento

### Quando Retreinar?

- **Semanal**: Se houver novos dados significativos
- **Mensal**: Como rotina de manutenção
- **Sob Demanda**: Se performance cair >5%

### Como Retreinar

```python
# 1. Atualizar features
spark.sql("REFRESH TABLE workspace.risco_ml_features.features_clientes")

# 2. Treinar novo modelo
# ... (mesmo código de treinamento)

# 3. Comparar métricas
# Se novo modelo > modelo atual: deploy
# Caso contrário: manter modelo atual
```

---

## 🛠️ Troubleshooting

### Problema: Modelo com performance ruim

**Solução:**
1. Verificar distribuição do target (deve ter ~20-30% inadimplentes)
2. Checar features com valores NULL
3. Avaliar feature importance

### Problema: Predições muito conservadoras/agressivas

**Solução:**
1. Ajustar threshold de classificação
2. Recalibrar probabilidades
3. Retreinar com mais dados

### Problema: Tabelas não encontradas

**Solução:**
```sql
-- Verificar schemas existentes
SHOW SCHEMAS IN workspace LIKE 'risco%';

-- Recriar se necessário
CREATE SCHEMA IF NOT EXISTS workspace.risco_bronze;
```

---

## 📈 Métricas de Sucesso

**Modelo:**
- Accuracy > 0.85
- Precision@Top20% > 0.80 (acertar os 20% de maior risco)
- ROC-AUC > 0.85

**Negócio:**
- Redução de inadimplência >180 dias em 30%
- Aumento de contatos proativos de cobrança em 50%
- Redução de perda total (write-offs) em 20%

---

**Autor**: Valdomiro Vega García  
**Data**: 02/07/2026
