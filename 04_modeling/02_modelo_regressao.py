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
%pip install xgboost==2.0.3 mlflow==2.9.2 scikit-learn==1.3.2 optuna==3.6.1 --quiet

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,1️⃣ Setup e Imports
dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

# Sem isso, mlflow.start_run() tenta resolver o registry URI padrão via config Spark
# (spark.mlflow.modelRegistryUri), que não existe em serverless/Spark Connect
# (CONFIG_NOT_AVAILABLE) -- mesmo problema já visto em 04_modeling/01_ e 05_mlops/01_.
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Shared/{CATALOG}_regressao_valor_risco")

# Mesmo padrão de nome/registro usado por 04_modeling/01_modelo_classificacao_risco.py —
# entrada própria no UC Model Registry (não compartilha nome com o classificador).
MODEL_NAME = "credit_risk_regressor"
MODEL_REGISTRY_NAME = f"{CATALOG}.gold.{MODEL_NAME}"
FALLBACK_MODEL_PATH = f"/Volumes/{CATALOG}/gold/model_fallback/{MODEL_NAME}"

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
    'categoria_risco',      # rótulo usado para enviesar a geração sintética -> leakage
    'data_cadastro',        # string crua, não numérica
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

# DBTITLE 1,3️⃣.1 Otimização de Hiperparâmetros (Optuna + Cross-Validation)
# Mesmo padrão do classificador (04_modeling/01_modelo_classificacao_risco.py): busca
# bayesiana com K-Fold (aqui sem stratify — é regressão), minimizando o MAE médio de
# validação em vez de um único split. O modelo final (célula seguinte) é retreinado no
# X_train completo com os melhores parâmetros encontrados aqui.
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 30

def objective(trial):
    trial_params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'objective': 'reg:squarederror',
        'random_state': 42,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_mae_scores = []

    for fold_train_idx, fold_val_idx in kf.split(X_train):
        X_fold_train = X_train.iloc[fold_train_idx]
        X_fold_val = X_train.iloc[fold_val_idx]
        y_fold_train = y_train.iloc[fold_train_idx]
        y_fold_val = y_train.iloc[fold_val_idx]

        fold_model = xgb.XGBRegressor(**trial_params)
        fold_model.fit(X_fold_train, y_fold_train)
        fold_pred = fold_model.predict(X_fold_val)
        fold_mae_scores.append(mean_absolute_error(y_fold_val, fold_pred))

    return np.mean(fold_mae_scores)

print("🔍 Buscando hiperparâmetros (Optuna + 5-fold CV)...")
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

print(f"\n✅ Busca concluída ({N_TRIALS} trials)")
print(f"🎯 Melhor MAE médio (5-fold CV): R$ {study.best_value:,.2f}")
print(f"\n📋 Melhores hiperparâmetros encontrados:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# DBTITLE 1,4️⃣ Treinar XGBoost Regressor
with mlflow.start_run(run_name=f"xgboost_regressao_valor_risco_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:

    params = {
        **study.best_params,
        'objective': 'reg:squarederror',
        'random_state': 42,
    }
    mlflow.log_params(params)
    mlflow.log_metric('cv_mae_mean', study.best_value)
    mlflow.log_param('optuna_n_trials', N_TRIALS)

    model = xgb.XGBRegressor(**params)
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

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")

    # Registro no Unity Catalog Model Registry (mlflow.xgboost.log_model com
    # registered_model_name, não um pickle solto em /tmp) — mesmo padrão de
    # 04_modeling/01_modelo_classificacao_risco.py, reaplicado aqui para o regressor.
    #
    # Fallback: em alguns workspaces o storage interno do Unity Catalog Model Registry pode
    # falhar por motivo de infraestrutura alheio ao código (ex: permissão AWS S3 quebrada na
    # conta). Se o registro falhar por QUALQUER motivo, salva o modelo num Volume UC — sem
    # versionamento nem alias Champion, mas garante que o notebook não trave.
    signature = infer_signature(X_train, model.predict(X_train))

    from mlflow.tracking import MlflowClient
    _client = MlflowClient()
    _registry_ok = False

    try:
        model_info = mlflow.xgboost.log_model(
            model, "model",
            signature=signature,
            registered_model_name=MODEL_REGISTRY_NAME,
        )
        print(f"✅ Modelo registrado no UC Model Registry: {MODEL_REGISTRY_NAME} v{model_info.registered_model_version}")

        # Bootstrap: se este é o primeiro modelo registrado (ainda não existe alias Champion),
        # promove-o como Champion inicial. Diferente do classificador, hoje não existe um
        # 05_mlops/ dedicado ao regressor que faça promoções seguintes com comparação de
        # métricas — esse bootstrap é a única promoção automática até que esse pipeline exista.
        try:
            _client.get_model_version_by_alias(MODEL_REGISTRY_NAME, "Champion")
            print("ℹ️ Já existe um Champion registrado — mantendo-o.")
        except Exception:
            _client.set_registered_model_alias(MODEL_REGISTRY_NAME, "Champion", model_info.registered_model_version)
            print(f"🏆 Nenhum Champion prévio — v{model_info.registered_model_version} promovido a Champion inicial")

        _registry_ok = True

    except Exception as e:
        print(f"⚠️ Falha ao registrar no UC Model Registry ({type(e).__name__}: {e})")
        print(f"↳ Fallback: salvando o modelo direto no Volume {FALLBACK_MODEL_PATH}")
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.gold.model_fallback")
        try:
            dbutils.fs.rm(FALLBACK_MODEL_PATH, recurse=True)  # save_model exige diretório inexistente
        except Exception:
            pass  # não existia ainda, tudo bem
        mlflow.xgboost.save_model(model, FALLBACK_MODEL_PATH)
        print(f"✅ Modelo salvo em {FALLBACK_MODEL_PATH} (sem Model Registry — sem alias Champion)")

    run_id = run.info.run_id

    print("\n" + "="*60)
    print("📊 RESULTADOS DO MODELO DE REGRESSÃO")
    print("="*60)
    for k, v in metrics.items():
        print(f"  {k}: {v:,.2f}" if "r2" not in k else f"  {k}: {v:.4f}")
    print(f"\n📦 Run ID: {run_id}")
    if _registry_ok:
        print(f"📦 Modelo registrado: {MODEL_REGISTRY_NAME} v{model_info.registered_model_version}")
    else:
        print(f"📦 Modelo salvo em fallback: {FALLBACK_MODEL_PATH}")
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
