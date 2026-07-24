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

# Evita CONFIG_NOT_AVAILABLE (spark.mlflow.modelRegistryUri) ao resolver o registry
# padrão em serverless/Spark Connect — mesma causa já vista em 02_modelo_regressao.py.
mlflow.set_registry_uri("databricks-uc")
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

# Flag de eventos irregulares (Black Friday, semanas de fim de ano) — calendário
# ILUSTRATIVO sobre dados sintéticos, não um calendário real de eventos de negócio.
# Sazonalidade automática do Prophet (semanal/anual) captura padrões que se repetem
# todo ano; eventos não-periódicos (ex: um evento que só acontece a cada N anos) não
# ficam bem representados só pela sazonalidade — por isso viram um regressor externo
# (ver célula de treino). Detecta automaticamente semanas de fim de novembro (Black
# Friday) e as duas últimas semanas de dezembro (Natal/Ano Novo) dentro do período
# observado nos dados.
def _flag_evento_irregular(data):
    return int((data.month == 11 and data.day >= 20) or (data.month == 12 and data.day >= 15))

df_cashflow['evento_irregular'] = df_cashflow['data'].apply(_flag_evento_irregular)

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
# Vamos prever o cash flow líquido (mantém evento_irregular para uso como regressor externo)
df_prophet = df_cashflow[['data', 'cashflow_liquido', 'evento_irregular']].copy()
df_prophet.columns = ['ds', 'y', 'evento_irregular']

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
with mlflow.start_run(run_name=f"prophet_cashflow_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:

    # Configurar Prophet — série agora é semanal, não diária (ver célula anterior)
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,  # não faz sentido com pontos semanais
        yearly_seasonality=False,
        changepoint_prior_scale=0.05  # Sensibilidade a mudanças de tendência
    )

    # Regressor externo para eventos não-periódicos (Black Friday, fim de ano — ver
    # célula "Criar Série Temporal"). A sazonalidade automática do Prophet só captura
    # padrões que se repetem todo ano; um evento fora desse ciclo (ex: uma campanha ou
    # evento que só acontece a cada alguns anos) distorce a curva se não for isolado
    # como sinal próprio.
    model.add_regressor('evento_irregular')

    # Treinar
    print("🔄 Treinando Prophet...")
    model.fit(df_prophet_clean)

    # Fazer forecast para as próximas ~13 semanas (~90 dias), respeitando a frequência
    # semanal da série de treino
    future = model.make_future_dataframe(periods=13, freq='W')

    # O regressor precisa de valor também nas datas futuras. Reaplica a mesma regra
    # usada no histórico (ver _flag_evento_irregular) — datas futuras que caem em
    # semana de Black Friday/fim de ano também entram marcadas.
    future['evento_irregular'] = future['ds'].apply(_flag_evento_irregular)

    forecast = model.predict(future)

    # Separar histórico de forecast
    forecast_future = forecast[forecast['ds'] > df_prophet_clean['ds'].max()].copy()

    # Ajuste IN-SAMPLE: o modelo `model` acima foi treinado com TODO o histórico
    # (df_prophet_clean) e está sendo comparado contra esses mesmos pontos — mede
    # qualidade de ajuste, não capacidade de generalizar para dados nunca vistos. Não
    # confundir com backtest de verdade (ver bloco walk-forward logo abaixo).
    historical_forecast = forecast[forecast['ds'] <= df_prophet_clean['ds'].max()].copy()
    historical_actual = df_prophet_clean.merge(historical_forecast[['ds', 'yhat']], on='ds')

    mae_in_sample = np.mean(np.abs(historical_actual['y'] - historical_actual['yhat']))
    mape_in_sample = np.mean(np.abs((historical_actual['y'] - historical_actual['yhat']) / historical_actual['y'])) * 100

    mlflow.log_metric("mae_in_sample", mae_in_sample)
    mlflow.log_metric("mape_in_sample", mape_in_sample)
    mlflow.log_param("forecast_horizon_days", 90)

    # Salvar modelo como artefato
    mlflow.prophet.log_model(model, "prophet_model")

    # Backtest WALK-FORWARD de verdade: treina um modelo separado só nas primeiras ~80% das
    # semanas, prevê as últimas ~20% (nunca vistas por esse modelo) e compara contra o valor
    # real observado nelas — isso sim mede capacidade de generalização, ao contrário do
    # ajuste in-sample acima. Não usa/substitui o `model` principal (que continua treinado
    # com 100% do histórico para gerar o forecast futuro na próxima célula).
    n_total = len(df_prophet_clean)
    n_train = max(int(n_total * 0.8), n_total - 13)  # nunca deixa menos de ~13 semanas de teste
    n_train = min(n_train, n_total - 4)  # garante pelo menos 4 semanas de holdout
    df_wf_train = df_prophet_clean.iloc[:n_train].copy()
    df_wf_holdout = df_prophet_clean.iloc[n_train:].copy()

    if len(df_wf_train) >= 10 and len(df_wf_holdout) >= 3:
        model_wf = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        model_wf.add_regressor('evento_irregular')
        model_wf.fit(df_wf_train)

        future_wf = model_wf.make_future_dataframe(periods=len(df_wf_holdout), freq='W')
        future_wf['evento_irregular'] = future_wf['ds'].apply(_flag_evento_irregular)
        forecast_wf = model_wf.predict(future_wf)

        holdout_pred = df_wf_holdout.merge(forecast_wf[['ds', 'yhat']], on='ds', how='inner')
        mae_walkforward = np.mean(np.abs(holdout_pred['y'] - holdout_pred['yhat']))
        mape_walkforward = np.mean(np.abs((holdout_pred['y'] - holdout_pred['yhat']) / holdout_pred['y'])) * 100

        mlflow.log_metric("mae_walkforward_out_of_sample", mae_walkforward)
        mlflow.log_metric("mape_walkforward_out_of_sample", mape_walkforward)
        mlflow.log_param("walkforward_train_semanas", len(df_wf_train))
        mlflow.log_param("walkforward_holdout_semanas", len(df_wf_holdout))
        walkforward_ok = True
    else:
        # Histórico curto demais pra separar train/holdout com um mínimo de significância —
        # documenta a limitação em vez de forçar um número pouco confiável.
        mae_walkforward = mape_walkforward = None
        walkforward_ok = False
        print("⚠️ Histórico insuficiente para walk-forward backtest confiável — pulando essa etapa.")

    # Model Card: metodologia do forecast, logada como artefato JSON na mesma run.
    # Resposta direta a um problema real relatado por um head de dados em entrevista:
    # uma projeção de anos anteriores sem metodologia rastreável, que ninguém sabia
    # reconstruir depois que a pessoa que a fez saiu da empresa. Qualquer pessoa (mesmo
    # sem ter treinado o modelo) consegue auditar aqui as premissas, o período de
    # backtest e a acurácia usados para gerar o número.
    avg_intervalo_confianca = (forecast_future['yhat_upper'] - forecast_future['yhat_lower']).mean()
    model_card = {
        "modelo": "Prophet (Facebook/Meta)",
        "gerado_em": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "run_id": run.info.run_id,
        "granularidade_serie": "semanal",
        "horizonte_forecast_dias": 90,
        "horizonte_forecast_semanas": 13,
        "sazonalidade": {
            "diaria": False,
            "semanal": False,
            "anual": False,
            "motivo": "desabilitadas — pontos semanais não sustentam sazonalidade diária/semanal, "
                      "e o histórico disponível (~1 ano de dados sintéticos) não cobre ciclos "
                      "anuais suficientes para uma sazonalidade anual confiável",
        },
        "regressores_externos": [
            {
                "nome": "evento_irregular",
                "descricao": "flag binária para semanas de Black Friday (fim de novembro) e "
                              "fim de ano (Natal/Ano Novo) — calendário ILUSTRATIVO sobre dados "
                              "sintéticos, não um calendário real de eventos de negócio da empresa",
            }
        ],
        "periodo_treino": {
            "inicio": df_prophet_clean['ds'].min().strftime('%Y-%m-%d'),
            "fim": df_prophet_clean['ds'].max().strftime('%Y-%m-%d'),
            "num_pontos": len(df_prophet_clean),
        },
        "metricas_ajuste_in_sample": {
            "descricao": "modelo final avaliado contra os MESMOS dados usados pra treiná-lo — "
                         "mede qualidade de ajuste, NÃO capacidade de generalização",
            "mae": round(float(mae_in_sample), 2),
            "mape_pct": round(float(mape_in_sample), 2),
        },
        "metricas_walkforward_out_of_sample": (
            {
                "descricao": "modelo separado treinado só no início da série, avaliado contra "
                             "semanas nunca vistas — este sim é um backtest de verdade",
                "semanas_treino": len(df_wf_train),
                "semanas_holdout": len(df_wf_holdout),
                "mae": round(float(mae_walkforward), 2),
                "mape_pct": round(float(mape_walkforward), 2),
            }
            if walkforward_ok
            else {"descricao": "não calculado — histórico insuficiente para separar train/holdout"}
        ),
        "intervalo_confianca_medio": round(float(avg_intervalo_confianca), 2),
        "premissas_e_limitacoes": [
            "Dados sintéticos — MAE/MAPE não representam performance em dados reais de produção",
            "changepoint_prior_scale=0.05 (sensibilidade padrão a mudanças de tendência, não "
            "recalibrado por busca de hiperparâmetros)",
            "Outliers extremos removidos por IQR (3x) antes do treino — ver célula de preparação",
        ],
    }
    mlflow.log_dict(model_card, "model_card.json")

    run_id = run.info.run_id

    print("\n" + "="*60)
    print("📊 RESULTADOS DO FORECAST")
    print("="*60)
    print(f"\n🎯 Ajuste in-sample (modelo final vs. dados de treino — qualidade de ajuste):")
    print(f"  MAE:  R$ {mae_in_sample:,.2f}")
    print(f"  MAPE: {mape_in_sample:.2f}%")
    if walkforward_ok:
        print(f"\n🎯 Backtest walk-forward out-of-sample (modelo separado vs. semanas nunca vistas):")
        print(f"  Treino: {len(df_wf_train)} semanas | Holdout: {len(df_wf_holdout)} semanas")
        print(f"  MAE:  R$ {mae_walkforward:,.2f}")
        print(f"  MAPE: {mape_walkforward:.2f}%")
    print(f"\n📅 Período de Forecast:")
    print(f"  Início: {forecast_future['ds'].min().strftime('%Y-%m-%d')}")
    print(f"  Fim:    {forecast_future['ds'].max().strftime('%Y-%m-%d')}")
    print(f"\n💰 Previsão de Cash Flow (próximas ~13 semanas / ~90 dias):")
    print(f"  Total Esperado:  R$ {forecast_future['yhat'].sum():,.2f}")
    print(f"  Média Semanal:   R$ {forecast_future['yhat'].mean():,.2f}")
    print(f"  Limite Inferior: R$ {forecast_future['yhat_lower'].sum():,.2f}")
    print(f"  Limite Superior: R$ {forecast_future['yhat_upper'].sum():,.2f}")
    print(f"\n📄 Model Card salvo como artefato MLflow: model_card.json")
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
print(f"  MAE (ajuste in-sample):  R$ {mae_in_sample:,.2f}")
print(f"  MAPE (ajuste in-sample): {mape_in_sample:.2f}%")
if walkforward_ok:
    print(f"  MAE (backtest walk-forward out-of-sample):  R$ {mae_walkforward:,.2f}")
    print(f"  MAPE (backtest walk-forward out-of-sample): {mape_walkforward:.2f}%")

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



