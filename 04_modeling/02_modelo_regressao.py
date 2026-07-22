# Databricks notebook source
# DBTITLE 1,Modelo de Regressão - Previsão de Valor em Risco
# MAGIC %md
# MAGIC # 🔢 Modelo de Regressão - Previsão de Valor em Risco
# MAGIC
# MAGIC **Objetivo**: Prever o VALOR MONETÁRIO que cada cliente tem em risco de inadimplência (não apenas SE vai inadimplir).
# MAGIC
# MAGIC ## Use Cases
# MAGIC * Provisão de perda (quanto reservar no balanço)
# MAGIC * Priorização de cobrança (focar em clientes com maior valor em risco)
# MAGIC * Limites de crédito dinâmicos
# MAGIC
# MAGIC ## Dataset
# MAGIC `credit_risk.gold.features_ml` (mesma feature store usada pelo classificador em `01_modelo_classificacao_risco`)
# MAGIC
# MAGIC ## Target
# MAGIC `valor_em_risco = total_faturado_90d * (taxa_inadimplencia / 100)` — estimativa do valor faturado nos últimos 90 dias exposto à taxa histórica de inadimplência do cliente.

# COMMAND ----------

# DBTITLE 1,Instalação de Bibliotecas
# xgboost não vem pré-instalado em compute serverless (diferente de clusters com ML Runtime)
%pip install xgboost==2.0.3 mlflow==2.9.2 scikit-learn==1.3.2 --quiet

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,1️⃣ Setup e Imports
dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.xgboost

# Sem isso, mlflow.start_run() tenta resolver o registry URI padrão via config Spark
# (spark.mlflow.modelRegistryUri), que não existe em serverless/Spark Connect
# (CONFIG_NOT_AVAILABLE) -- mesmo problema já visto em 04_modeling/01_ e 05_mlops/01_.
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Shared/{CATALOG}_regressao_valor_risco")

print("✅ Bibliotecas carregadas")
print(f"📦 XGBoost version: {xgb.__version__}")

# COMMAND ----------

# DBTITLE 1,2️⃣ Carregar Feature Store e Criar Target
# Toda a leitura/agregação acontece em Spark; só o dataset final (1 linha por cliente) vai para pandas.
df_features = spark.table(f"{CATALOG}.gold.features_ml")

df_target = df_features.selectExpr(
    "id_cliente",
    "total_faturado_90d * (taxa_inadimplencia / 100) AS valor_em_risco"
)

df_pd = df_features.join(df_target, "id_cliente").toPandas()

print(f"📊 Shape: {df_pd.shape}")
print(f"💰 Valor em risco total: R$ {df_pd['valor_em_risco'].sum():,.2f}")
print(f"💰 Valor médio em risco: R$ {df_pd['valor_em_risco'].mean():,.2f}")
print("\n🎯 Target distribution:")
print(df_pd['valor_em_risco'].describe())

# COMMAND ----------

# DBTITLE 1,3️⃣ Preparar Features para Regressão
# Mesma lógica de exclusão/encoding do classificador (01_modelo_classificacao_risco), para consistência
cols_to_drop = [
    'id_cliente', 'cnpj', 'nome',
    'categoria_rfm', 'perfil_comportamental',
    'taxa_inadimplencia',   # usada para construir o target -> leakage
    'total_faturado_90d',   # usada para construir o target -> leakage
    'valor_em_risco',       # target
]
categorical_features = ['porte', 'setor']

feature_cols = [c for c in df_pd.columns if c not in cols_to_drop]
df_encoded = pd.get_dummies(df_pd[feature_cols], columns=categorical_features, drop_first=False)

X = df_encoded.copy()
y = df_pd['valor_em_risco'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Train set: {X_train.shape} | Test set: {X_test.shape}")

# COMMAND ----------

# DBTITLE 1,4️⃣ Treinar XGBoost Regressor
with mlflow.start_run(run_name="xgboost_regressao_valor_risco") as run:

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "mae_train": mean_absolute_error(y_train, y_pred_train),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
    }
    mlflow.log_metrics(metrics)
    mlflow.xgboost.log_model(model, "model")

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")

    run_id = run.info.run_id

    print("\n" + "="*60)
    print("📊 RESULTADOS DO MODELO DE REGRESSÃO")
    print("="*60)
    for k, v in metrics.items():
        print(f"  {k}: {v:,.2f}" if "r2" not in k else f"  {k}: {v:.4f}")
    print(f"\n📦 Run ID: {run_id}")
    print("\n🏆 Top 5 Features:")
    print(feature_importance.head())
    print("="*60)

# COMMAND ----------

# DBTITLE 1,5️⃣ Batch Inference - Prever Valores em Risco
df_pred = df_pd[['id_cliente', 'valor_em_risco']].copy()
df_pred['valor_previsto'] = model.predict(X)
df_pred['erro_previsao'] = np.abs(df_pred['valor_em_risco'] - df_pred['valor_previsto'])
df_pred['categoria_risco_monetario'] = pd.cut(
    df_pred['valor_previsto'],
    bins=[-np.inf, 0, 5000, 20000, np.inf],
    labels=['Nenhum', 'Baixo', 'Médio', 'Alto']
).astype(str)

# Volta para Spark só na escrita final (tabela pequena, 1 linha por cliente)
spark_df = spark.createDataFrame(df_pred)
spark_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.gold.previsao_valor_inadimplente")

print(f"✅ Tabela salva: {CATALOG}.gold.previsao_valor_inadimplente")
print(f"\n💰 Valor total previsto em risco: R$ {df_pred['valor_previsto'].sum():,.2f}")
print("\n🎯 Top 10 Clientes com Maior Risco Monetário:")
print(df_pred.nlargest(10, 'valor_previsto')[['id_cliente', 'valor_em_risco', 'valor_previsto', 'categoria_risco_monetario']])
