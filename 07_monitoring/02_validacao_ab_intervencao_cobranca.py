# Databricks notebook source
# DBTITLE 1,Validação A/B — Efeito da Intervenção de Cobrança
# MAGIC %md
# MAGIC # 🧪 Validação A/B — Efeito da Intervenção de Cobrança
# MAGIC
# MAGIC ## Objetivo
# MAGIC Validar estatisticamente se contatar clientes sinalizados como alto risco pelo classificador
# MAGIC (`01_modelo_classificacao_risco.py`) reduz a taxa de inadimplência, comparando um grupo
# MAGIC contatado (tratamento) contra um grupo não contatado (controle) entre os clientes de maior
# MAGIC prioridade (`04_modeling/05_priorizacao_cobranca.py`).
# MAGIC
# MAGIC ## ⚠️ IMPORTANTE — Este notebook usa uma SIMULAÇÃO SINTÉTICA
# MAGIC O projeto não tem um log real de quem foi contatado pela cobrança nem o resultado real
# MAGIC dessa intervenção (não existe essa camada de dados aqui). Para demonstrar a metodologia
# MAGIC de validação — a mesma usada para provar que uma campanha realmente funcionou, com grupo
# MAGIC controle vs. tratamento e teste de hipótese, não achismo — este notebook:
# MAGIC 1. Sorteia aleatoriamente (seed fixa) metade dos clientes de alta prioridade como "contatados"
# MAGIC 2. Simula um efeito de redução de inadimplência PARA O GRUPO TRATAMENTO (parâmetro
# MAGIC    `EFEITO_SIMULADO_RELATIVO`, documentado explicitamente abaixo — não é um resultado medido)
# MAGIC 3. Aplica o teste estatístico correto (teste Z de duas proporções) sobre esse cenário simulado
# MAGIC
# MAGIC **Os números de p-value/lift aqui NÃO são um resultado de produção.** O que é reaproveitável
# MAGIC de verdade é o código do teste estatístico — plugue nele um log real de intervenções (ex:
# MAGIC uma tabela `crm.historico_contatos` com `id_cliente`, `foi_contatado`, `inadimpliu_realizado`)
# MAGIC e o resultado passa a ser válido.

# COMMAND ----------

# DBTITLE 1,Setup e Imports
dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

import numpy as np
import pandas as pd
from scipy import stats
from pyspark.sql import functions as F

# Efeito simulado: redução relativa da probabilidade de inadimplência no grupo
# tratamento, usada só para gerar o cenário de demonstração (ver aviso acima).
EFEITO_SIMULADO_RELATIVO = 0.15  # -15% na probabilidade de inadimplência do grupo contatado
SEED = 42

print("✅ Setup concluído")
print(f"⚠️ EFEITO_SIMULADO_RELATIVO = {EFEITO_SIMULADO_RELATIVO:.0%} — parâmetro sintético, não medido")

# COMMAND ----------

# DBTITLE 1,Selecionar População de Alto Risco
# População elegível para a intervenção: clientes de prioridade Crítica/Alta no ranking
# de cobrança (ver 04_modeling/05_priorizacao_cobranca.py).
df_alvo = (
    spark.table(f"{CATALOG}.gold.priorizacao_cobranca")
    .filter(F.col("faixa_prioridade").isin("Crítica", "Alta"))
    .select("id_cliente", "probabilidade_inadimplencia", "valor_previsto", "faixa_prioridade")
    .toPandas()
)

print(f"✅ População elegível para a intervenção: {len(df_alvo):,} clientes")

# COMMAND ----------

# DBTITLE 1,Simular Alocação Controle/Tratamento e Resultado
# Alocação aleatória 50/50 — grupo controle não é contatado, grupo tratamento é.
rng = np.random.default_rng(SEED)
df_alvo["grupo"] = rng.choice(["tratamento", "controle"], size=len(df_alvo), p=[0.5, 0.5])

# Resultado simulado (SINTÉTICO, ver aviso no topo do notebook): sorteia se o cliente
# de fato inadimpliu usando a probabilidade do classificador como base — o grupo
# tratamento recebe a probabilidade reduzida por EFEITO_SIMULADO_RELATIVO.
def _simular_resultado(row):
    prob = row["probabilidade_inadimplencia"]
    if row["grupo"] == "tratamento":
        prob = prob * (1 - EFEITO_SIMULADO_RELATIVO)
    return int(rng.random() < prob)

df_alvo["inadimpliu_simulado"] = df_alvo.apply(_simular_resultado, axis=1)

resumo_grupos = df_alvo.groupby("grupo")["inadimpliu_simulado"].agg(["count", "sum", "mean"])
resumo_grupos.columns = ["n_clientes", "n_inadimplentes", "taxa_inadimplencia"]
print("✅ Simulação de alocação e resultado concluída")
print("\n📊 Resumo por grupo:")
print(resumo_grupos)

# COMMAND ----------

# DBTITLE 1,Teste Estatístico — Duas Proporções (Z-test)
# Teste Z de duas proporções: H0 = taxa de inadimplência é igual entre controle e
# tratamento; H1 = taxa é menor no grupo tratamento (teste unilateral, já que a hipótese
# de negócio é "contato reduz inadimplência", não "contato altera inadimplência").
n_controle = int(resumo_grupos.loc["controle", "n_clientes"])
x_controle = int(resumo_grupos.loc["controle", "n_inadimplentes"])
n_tratamento = int(resumo_grupos.loc["tratamento", "n_clientes"])
x_tratamento = int(resumo_grupos.loc["tratamento", "n_inadimplentes"])

p_controle = x_controle / n_controle
p_tratamento = x_tratamento / n_tratamento
p_pooled = (x_controle + x_tratamento) / (n_controle + n_tratamento)

erro_padrao = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_controle + 1 / n_tratamento))
z_stat = (p_controle - p_tratamento) / erro_padrao if erro_padrao > 0 else 0.0
p_value = 1 - stats.norm.cdf(z_stat)  # unilateral: tratamento < controle

# Intervalo de confiança 95% para a diferença de proporções (não-pooled)
diff = p_controle - p_tratamento
erro_padrao_diff = np.sqrt(
    p_controle * (1 - p_controle) / n_controle + p_tratamento * (1 - p_tratamento) / n_tratamento
)
ic_inferior = diff - 1.96 * erro_padrao_diff
ic_superior = diff + 1.96 * erro_padrao_diff

lift_relativo = (p_controle - p_tratamento) / p_controle if p_controle > 0 else 0.0
significativo = p_value < 0.05

print("="*70)
print("🧪 RESULTADO DO TESTE ESTATÍSTICO (cenário SIMULADO)")
print("="*70)
print(f"\n  Taxa de inadimplência — controle:    {p_controle:.2%} ({x_controle}/{n_controle})")
print(f"  Taxa de inadimplência — tratamento:  {p_tratamento:.2%} ({x_tratamento}/{n_tratamento})")
print(f"  Diferença absoluta:                  {diff:.2%}")
print(f"  Lift relativo (redução):             {lift_relativo:.2%}")
print(f"  IC 95% da diferença:                 [{ic_inferior:.2%}, {ic_superior:.2%}]")
print(f"  Z-statistic:                         {z_stat:.4f}")
print(f"  P-value (unilateral):                {p_value:.4f}")
print(f"\n  {'✅ Estatisticamente significativo (p < 0.05)' if significativo else '⚠️ NÃO significativo (p >= 0.05)'}")

# COMMAND ----------

# DBTITLE 1,Salvar Resultado
df_resultado = pd.DataFrame([{
    "data_analise": pd.Timestamp.now(),
    "n_controle": n_controle,
    "n_tratamento": n_tratamento,
    "taxa_controle": p_controle,
    "taxa_tratamento": p_tratamento,
    "lift_relativo": lift_relativo,
    "ic95_inferior": ic_inferior,
    "ic95_superior": ic_superior,
    "z_stat": z_stat,
    "p_value": p_value,
    "significativo": significativo,
    "efeito_simulado_relativo_input": EFEITO_SIMULADO_RELATIVO,
    "eh_simulacao_sintetica": True,
}])

spark.createDataFrame(df_resultado).write.mode("append").saveAsTable(f"{CATALOG}.gold.validacao_ab_cobranca")

print(f"✅ Resultado salvo em: {CATALOG}.gold.validacao_ab_cobranca")
print("\n💡 Para usar com dados reais: substitua a célula de simulação por uma leitura de um")
print("   log real de intervenções (quem foi contatado, quem realmente inadimpliu) e o resto")
print("   do notebook (teste estatístico, IC, salvamento) já funciona sem alteração.")
