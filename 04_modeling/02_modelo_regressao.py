# Databricks notebook source
# DBTITLE 1,Modelo de Regressão - Previsão de Valor Inadimplente
# MAGIC %md
# MAGIC # 🔢 Modelo de Regressão - Previsão de Valor Inadimplente
# MAGIC
# MAGIC **Objetivo**: Prever o VALOR MONETÁRIO que cada cliente deixará de pagar (não apenas SE vai inadimplir).
# MAGIC
# MAGIC ## Use Cases
# MAGIC * Provisão de perda (quanto reservar no balanço)
# MAGIC * Priorização de cobrança (focar em clientes com maior valor em risco)
# MAGIC * Limites de crédito dinâmicos
# MAGIC * ROI de estratégias de recuperação
# MAGIC
# MAGIC ## Arquitetura
# MAGIC ```
# MAGIC Feature Store → XGBoost Regressor → MLflow → Gold Layer
# MAGIC ```
# MAGIC
# MAGIC ## Target
# MAGIC `valor_em_risco`: Soma de todos os valores em aberto que o cliente tem historicamente não pago

# COMMAND ----------

# DBTITLE 1,1️⃣ Setup e Imports
# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.xgboost
from pyspark.sql import functions as F

# Config MLflow
mlflow.set_experiment("/Users/valdomirovega@hotmail.com/risco_financeiro_experiments")

print("✅ Bibliotecas carregadas")
print(f"📦 XGBoost version: {xgb.__version__}")
print(f"📦 MLflow version: {mlflow.__version__}")

# COMMAND ----------

# DBTITLE 1,2️⃣ Carregar Features e Criar Target
# Carregar feature store
df_features = spark.table("workspace.risco_ml_features.features_clientes").toPandas()

# Criar target: valor_em_risco (soma de valores em aberto de faturas inadimplentes)
df_faturas = spark.sql("""
  SELECT 
    cliente_id,
    SUM(CASE WHEN dias_atraso > 0 THEN valor_em_aberto ELSE 0 END) as valor_em_risco
  FROM workspace.risco_silver.faturas_enriquecidas
  GROUP BY cliente_id
""")

df_target = df_faturas.toPandas()

# Merge
df = df_features.merge(df_target, on='cliente_id', how='left')
df['valor_em_risco'] = df['valor_em_risco'].fillna(0)

print(f"📊 Shape: {df.shape}")
print(f"💰 Valor em risco total: R$ {df['valor_em_risco'].sum():,.2f}")
print(f"💰 Valor médio em risco: R$ {df['valor_em_risco'].mean():,.2f}")
print(f"\n🎯 Target distribution:")
print(df['valor_em_risco'].describe())

# COMMAND ----------

# DBTITLE 1,3️⃣ Preparar Features para Regressão
# Remover target e ID das features
feature_cols = [col for col in df.columns if col not in ['cliente_id', 'inadimplente', 'valor_em_risco']]

X = df[feature_cols].copy()
y = df['valor_em_risco'].copy()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Train set: {X_train.shape}")
print(f"✅ Test set: {X_test.shape}")
print(f"\n📊 Train target stats:")
print(f"  Min: R$ {y_train.min():,.2f}")
print(f"  Max: R$ {y_train.max():,.2f}")
print(f"  Mean: R$ {y_train.mean():,.2f}")
print(f"  Median: R$ {y_train.median():,.2f}")

# COMMAND ----------

# DBTITLE 1,4️⃣ Treinar XGBoost Regressor
# Iniciar MLflow run
with mlflow.start_run(run_name="xgboost_regressao_valor") as run:
    
    # Configurar modelo
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    )
    
    # Treinar
    print("🔄 Treinando XGBoost Regressor...")
    model.fit(X_train, y_train)
    
    # Predições
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métricas
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Log métricas
    mlflow.log_metric("mae_train", mae_train)
    mlflow.log_metric("mae_test", mae_test)
    mlflow.log_metric("rmse_train", rmse_train)
    mlflow.log_metric("rmse_test", rmse_test)
    mlflow.log_metric("r2_train", r2_train)
    mlflow.log_metric("r2_test", r2_test)
    
    # Log modelo
    mlflow.xgboost.log_model(model, "model")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")
    
    run_id = run.info.run_id
    
    print("\n" + "="*60)
    print("📊 RESULTADOS DO MODELO DE REGRESSÃO")
    print("="*60)
    print(f"\n🎯 Métricas de Performance:")
    print(f"  MAE Train:  R$ {mae_train:,.2f}")
    print(f"  MAE Test:   R$ {mae_test:,.2f}")
    print(f"  RMSE Train: R$ {rmse_train:,.2f}")
    print(f"  RMSE Test:  R$ {rmse_test:,.2f}")
    print(f"  R² Train:   {r2_train:.4f}")
    print(f"  R² Test:    {r2_test:.4f}")
    print(f"\n📦 Run ID: {run_id}")
    print("\n🏆 Top 5 Features:")
    print(feature_importance.head())
    print("="*60)

# COMMAND ----------

# DBTITLE 1,5️⃣ Batch Inference - Prever Valores em Risco
# Prever para todos os clientes
df_pred = df.copy()
df_pred['valor_previsto'] = model.predict(X)
df_pred['erro_previsao'] = np.abs(df_pred['valor_em_risco'] - df_pred['valor_previsto'])

# Criar tabela no Gold
df_resultado = df_pred[['cliente_id', 'valor_em_risco', 'valor_previsto', 'erro_previsao']].copy()

# Adicionar categoria de risco monetário
df_resultado['categoria_risco_monetario'] = pd.cut(
    df_resultado['valor_previsto'],
    bins=[-np.inf, 0, 5000, 20000, np.inf],
    labels=['Nenhum', 'Baixo', 'Médio', 'Alto']
)

# Converter para Spark e salvar
spark_df = spark.createDataFrame(df_resultado)
spark_df.write.mode("overwrite").saveAsTable("workspace.risco_gold.previsao_valor_inadimplente")

print("✅ Tabela salva: workspace.risco_gold.previsao_valor_inadimplente")
print(f"\n📊 Distribuição de Risco Monetário:")
print(df_resultado['categoria_risco_monetario'].value_counts().sort_index())
print(f"\n💰 Valor total previsto em risco: R$ {df_resultado['valor_previsto'].sum():,.2f}")
print(f"\n🎯 Top 10 Clientes com Maior Risco Monetário:")
print(df_resultado.nlargest(10, 'valor_previsto')[['cliente_id', 'valor_em_risco', 'valor_previsto', 'categoria_risco_monetario']])

# COMMAND ----------



