# Databricks notebook source
# DBTITLE 1,Forecast de Cash Flow com Prophet
# MAGIC %md
# MAGIC # 📈 Forecast de Cash Flow - Séries Temporais
# MAGIC
# MAGIC **Objetivo**: Prever o fluxo de caixa futuro com base em padrões históricos de receitas e inadimplência.
# MAGIC
# MAGIC ## Use Cases
# MAGIC * Planejamento financeiro (quanto dinheiro entrará nos próximos 30/60/90 dias)
# MAGIC * Provisionamento de capital de giro
# MAGIC * Detecção de sazonalidade (meses críticos)
# MAGIC * Simulação de cenários (best/worst case)
# MAGIC
# MAGIC ## Arquitetura
# MAGIC ```
# MAGIC Faturas Agregadas por Data → Prophet → Forecast 90 dias → Gold Layer
# MAGIC ```
# MAGIC
# MAGIC ## Métricas
# MAGIC * Receita esperada (soma dos valores vencendo)
# MAGIC * Receita real (valor efetivamente recebido)
# MAGIC * Cash gap (diferença entre esperado e real)

# COMMAND ----------

# DBTITLE 1,1️⃣ Setup e Imports
# Instalar Prophet se necessário
import sys

try:
    from prophet import Prophet
    print("✅ Prophet já instalado")
except ImportError:
    print("🔧 Instalando Prophet...")
    %pip install prophet --quiet
    from prophet import Prophet
    print("✅ Prophet instalado com sucesso")

import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from pyspark.sql import functions as F
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

mlflow.set_experiment(f"/Shared/{CATALOG}_forecast_cashflow")

print("✅ Bibliotecas carregadas")
print(f"📦 MLflow version: {mlflow.__version__}")

# COMMAND ----------

# DBTITLE 1,2️⃣ Criar Série Temporal de Cash Flow
# Agregação por SEMANA (não por dia) feita em Spark. Com ~5.000 faturas sintéticas
# espalhadas em ~1 ano, agregação diária deixa a série muito esparsa/ruidosa (poucas
# faturas vencendo por dia) — o Prophet ajusta mal e o MAPE fica altíssimo (>200%,
# confirmado rodando de verdade). Semanal suaviza o ruído e ainda é granularidade
# realista para planejamento financeiro.
df_cashflow = spark.sql(f"""
  SELECT
    date_trunc('week', data_vencimento) as data,
    SUM(valor_total) as receita_esperada,
    SUM(CASE WHEN dias_atraso <= 0 THEN valor_total ELSE 0 END) as receita_recebida,
    SUM(CASE WHEN dias_atraso > 0 THEN valor_em_aberto ELSE 0 END) as perda_inadimplencia,
    COUNT(*) as num_faturas
  FROM {CATALOG}.silver.faturas_enriquecidas
  GROUP BY date_trunc('week', data_vencimento)
  ORDER BY data
""").toPandas()

# Converter para datetime
df_cashflow['data'] = pd.to_datetime(df_cashflow['data'])

# Cash flow líquido = receita_recebida - perda_inadimplencia
df_cashflow['cashflow_liquido'] = df_cashflow['receita_recebida'] - df_cashflow['perda_inadimplencia']

print(f"📊 Datas disponíveis: {df_cashflow['data'].min()} até {df_cashflow['data'].max()}")
print(f"📊 Total de registros: {len(df_cashflow)}")
print(f"\n💰 Métricas Históricas:")
print(f"  Receita Total Esperada: R$ {df_cashflow['receita_esperada'].sum():,.2f}")
print(f"  Receita Real Recebida:  R$ {df_cashflow['receita_recebida'].sum():,.2f}")
print(f"  Perda Inadimplência:   R$ {df_cashflow['perda_inadimplencia'].sum():,.2f}")
print(f"  Cash Flow Líquido:     R$ {df_cashflow['cashflow_liquido'].sum():,.2f}")

df_cashflow.head(10)

# COMMAND ----------

# DBTITLE 1,3️⃣ Preparar Dados para Prophet
# Prophet requer colunas 'ds' (data) e 'y' (valor)
# Vamos prever o cash flow líquido
df_prophet = df_cashflow[['data', 'cashflow_liquido']].copy()
df_prophet.columns = ['ds', 'y']

# Remover outliers extremos (opcional)
q1 = df_prophet['y'].quantile(0.25)
q3 = df_prophet['y'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 3 * iqr
upper_bound = q3 + 3 * iqr

df_prophet_clean = df_prophet[
    (df_prophet['y'] >= lower_bound) & 
    (df_prophet['y'] <= upper_bound)
].copy()

print(f"✅ Dados preparados para Prophet")
print(f"📊 Registros após limpeza: {len(df_prophet_clean)} (removidos {len(df_prophet) - len(df_prophet_clean)} outliers)")
print(f"\n📊 Estatísticas do Target (Cash Flow Líquido):")
print(df_prophet_clean['y'].describe())

# COMMAND ----------

# DBTITLE 1,4️⃣ Treinar Prophet e Fazer Forecast
# Iniciar MLflow run
with mlflow.start_run(run_name="prophet_cashflow_forecast") as run:
    
    # Configurar Prophet — série agora é semanal, não diária (ver célula anterior)
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,  # não faz sentido com pontos semanais
        yearly_seasonality=False,
        changepoint_prior_scale=0.05  # Sensibilidade a mudanças de tendência
    )

    # Treinar
    print("🔄 Treinando Prophet...")
    model.fit(df_prophet_clean)

    # Fazer forecast para as próximas ~13 semanas (~90 dias), respeitando a frequência
    # semanal da série de treino
    future = model.make_future_dataframe(periods=13, freq='W')
    forecast = model.predict(future)
    
    # Separar histórico de forecast
    forecast_future = forecast[forecast['ds'] > df_prophet_clean['ds'].max()].copy()
    
    # Calcular métricas de acurácia no histórico
    historical_forecast = forecast[forecast['ds'] <= df_prophet_clean['ds'].max()].copy()
    historical_actual = df_prophet_clean.merge(historical_forecast[['ds', 'yhat']], on='ds')
    
    mae = np.mean(np.abs(historical_actual['y'] - historical_actual['yhat']))
    mape = np.mean(np.abs((historical_actual['y'] - historical_actual['yhat']) / historical_actual['y'])) * 100
    
    # Log métricas
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mape", mape)
    mlflow.log_param("forecast_horizon_days", 90)
    
    # Salvar modelo como artefato
    mlflow.prophet.log_model(model, "prophet_model")
    
    run_id = run.info.run_id
    
    print("\n" + "="*60)
    print("📊 RESULTADOS DO FORECAST")
    print("="*60)
    print(f"\n🎯 Métricas de Acurácia (Histórico):")
    print(f"  MAE:  R$ {mae:,.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"\n📅 Período de Forecast:")
    print(f"  Início: {forecast_future['ds'].min().strftime('%Y-%m-%d')}")
    print(f"  Fim:    {forecast_future['ds'].max().strftime('%Y-%m-%d')}")
    print(f"\n💰 Previsão de Cash Flow (próximas ~13 semanas / ~90 dias):")
    print(f"  Total Esperado:  R$ {forecast_future['yhat'].sum():,.2f}")
    print(f"  Média Semanal:   R$ {forecast_future['yhat'].mean():,.2f}")
    print(f"  Limite Inferior: R$ {forecast_future['yhat_lower'].sum():,.2f}")
    print(f"  Limite Superior: R$ {forecast_future['yhat_upper'].sum():,.2f}")
    print(f"\n📦 Run ID: {run_id}")
    print("="*60)

# COMMAND ----------

# DBTITLE 1,5️⃣ Salvar Forecast no Gold Layer
# Preparar dados para salvar
df_forecast_save = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
df_forecast_save.columns = ['data_prevista', 'cashflow_previsto', 'cashflow_min', 'cashflow_max']

# Adicionar categorização de risco — thresholds escalados para total SEMANAL
# (cashflow_previsto agora é soma de 1 semana, não de 1 dia)
df_forecast_save['risco_cashflow'] = pd.cut(
    df_forecast_save['cashflow_previsto'],
    bins=[-np.inf, 0, 350000, 700000, np.inf],
    labels=['Crítico', 'Atenção', 'Normal', 'Excelente']
)

# Adicionar janelas de tempo
df_forecast_save['dias_futuro'] = (df_forecast_save['data_prevista'] - df_prophet_clean['ds'].max()).dt.days
df_forecast_save['janela'] = pd.cut(
    df_forecast_save['dias_futuro'],
    bins=[0, 30, 60, 90],
    labels=['0-30 dias', '31-60 dias', '61-90 dias']
)

# Converter para Spark e salvar (tabela pequena: ~13 linhas, 1 por semana)
spark_df_forecast = spark.createDataFrame(df_forecast_save)
spark_df_forecast.write.mode("overwrite").saveAsTable(f"{CATALOG}.gold.forecast_cashflow")

print(f"✅ Tabela salva: {CATALOG}.gold.forecast_cashflow")
print(f"\n📊 Forecast por Janela de Tempo:")
print(df_forecast_save.groupby('janela')['cashflow_previsto'].sum())
print(f"\n⚠️ Dias com Risco Crítico (cashflow negativo):")
risco_critico = df_forecast_save[df_forecast_save['risco_cashflow'] == 'Crítico']
if len(risco_critico) > 0:
    print(risco_critico[['data_prevista', 'cashflow_previsto', 'risco_cashflow']])
else:
    print("  Nenhum dia com risco crítico identificado! 🎉")

# COMMAND ----------

# DBTITLE 1,6️⃣ Sumário Executivo
print("="*70)
print("📊 SUMÁRIO EXECUTIVO - FORECAST DE CASH FLOW")
print("="*70)

print("\n🎯 MODELO:")
print(f"  Algoritmo: Prophet (Facebook)")
print(f"  Horizon: 90 dias")
print(f"  MAE: R$ {mae:,.2f}")
print(f"  MAPE: {mape:.2f}%")

print("\n💰 PREVISÃO DE CASH FLOW:")
for janela in ['0-30 dias', '31-60 dias', '61-90 dias']:
    valor = df_forecast_save[df_forecast_save['janela'] == janela]['cashflow_previsto'].sum()
    print(f"  {janela}: R$ {valor:,.2f}")

print(f"\n📅 TOTAL 90 DIAS: R$ {df_forecast_save['cashflow_previsto'].sum():,.2f}")

print("\n⚠️ ALERTAS:")
if len(risco_critico) > 0:
    print(f"  ⚠️ {len(risco_critico)} dias com risco crítico (cashflow negativo)")
else:
    print("  ✅ Nenhum alerta crítico")

print("\n💾 DADOS SALVOS:")
print(f"  ✅ {CATALOG}.gold.forecast_cashflow")

print("\n" + "="*70)

# COMMAND ----------



