# Databricks notebook source
# DBTITLE 1,AutoML e Comparação de Algoritmos
# MAGIC %md
# MAGIC # 🤖 AutoML e Comparação de Algoritmos — Classificação de Risco
# MAGIC
# MAGIC ## Objetivo
# MAGIC Complementar o classificador manual (`01_modelo_classificacao_risco.py`, XGBoost com feature
# MAGIC engineering e tuning manual) com dois pontos de comparação:
# MAGIC
# MAGIC 1. **Databricks AutoML** — baseline rápido, gerado automaticamente, útil para validar se o
# MAGIC    esforço de tuning manual está de fato trazendo ganho real sobre o que a plataforma resolve sozinha.
# MAGIC 2. **LightGBM** — mesma feature store, hiperparâmetros equivalentes ao XGBoost, para comparar
# MAGIC    tempo de treino e métricas entre as duas bibliotecas de gradient boosting mais usadas no mercado.
# MAGIC
# MAGIC ## Por que isso importa (não é só "rodar por rodar")
# MAGIC Em produção, decidir entre "deixar o AutoML resolver" vs. "investir em feature engineering e
# MAGIC tuning manual" é uma decisão real de custo/benefício de tempo de time. Este notebook documenta
# MAGIC esse trade-off com números, não opinião.

# COMMAND ----------

# DBTITLE 1,Instalação de Bibliotecas
# LightGBM não vem pré-instalado em compute serverless
%pip install lightgbm==4.3.0 --quiet

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup e Imports
dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import mlflow

# Mesma configuração de registry/experimento dos outros notebooks de modelagem — evita
# CONFIG_NOT_AVAILABLE ao resolver o registry padrão em serverless/Spark Connect.
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/credit_risk_mlops")

print("✅ Bibliotecas carregadas")

# COMMAND ----------

# DBTITLE 1,Carregar e Preparar Dados
# MAGIC %md
# MAGIC ## 1. Carregar Feature Store e Preparar Dados
# MAGIC Mesma lógica de exclusão/encoding de `01_modelo_classificacao_risco.py`, para que a
# MAGIC comparação entre algoritmos seja justa (mesmas features, mesmo split).

# COMMAND ----------

df_spark = spark.table(f"{CATALOG}.gold.features_ml")
df_pd = df_spark.toPandas()

df_pd['inadimplente'] = (df_pd['taxa_inadimplencia'] > 40).astype(int)

cols_to_drop = [
    'id_cliente', 'cnpj', 'nome',
    'categoria_rfm', 'perfil_comportamental',
    'categoria_risco',      # leakage — usado para enviesar a geração sintética
    'data_cadastro',        # string crua
    'taxa_inadimplencia',   # usada para construir o target -> leakage
    'inadimplente',         # target
]
categorical_features = ['porte', 'setor']

feature_cols = [c for c in df_pd.columns if c not in cols_to_drop]
df_encoded = pd.get_dummies(df_pd[feature_cols], columns=categorical_features, drop_first=False)

X = df_encoded.copy()
y = df_pd['inadimplente'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Dados preparados: {X_train.shape[0]:,} treino / {X_test.shape[0]:,} teste, {X.shape[1]} features")

# COMMAND ----------

# DBTITLE 1,Databricks AutoML
# MAGIC %md
# MAGIC ## 2. Databricks AutoML
# MAGIC O AutoML do Databricks recebe o dataset (Spark ou pandas), testa vários algoritmos e
# MAGIC hiperparâmetros automaticamente, e retorna o melhor modelo com o notebook de treino gerado
# MAGIC (útil para auditoria — não é uma caixa preta). Requer cluster com ML Runtime — em serverless
# MAGIC ou workspaces sem a feature habilitada, cai no bloco de exceção abaixo em vez de quebrar o
# MAGIC notebook inteiro.

# COMMAND ----------

automl_summary = None
try:
    from databricks import automl

    df_automl = df_pd[feature_cols + ['inadimplente']].copy()
    df_automl_spark = spark.createDataFrame(df_automl)

    automl_summary = automl.classify(
        dataset=df_automl_spark,
        target_col="inadimplente",
        primary_metric="roc_auc",
        timeout_minutes=15,
    )

    print(f"✅ AutoML concluído")
    print(f"📦 Melhor trial: {automl_summary.best_trial.model_description}")
    print(f"🔗 Notebook de treino gerado: {automl_summary.best_trial.notebook_path}")
    print(f"📊 Métricas do melhor modelo: {automl_summary.best_trial.metrics}")

except Exception as e:
    print(f"⚠️ Databricks AutoML não disponível neste ambiente ({type(e).__name__}: {e})")
    print("↳ Requer cluster com Databricks ML Runtime — não roda em todo tipo de compute.")
    print("↳ Seguindo só com a comparação manual XGBoost vs. LightGBM abaixo.")

# COMMAND ----------

# DBTITLE 1,Treinar LightGBM
# MAGIC %md
# MAGIC ## 3. LightGBM (comparação manual com XGBoost)
# MAGIC Mesmos dados, hiperparâmetros equivalentes ao XGBoost de `01_modelo_classificacao_risco.py`
# MAGIC (profundidade, learning rate, número de árvores, compensação de desbalanceamento).

# COMMAND ----------

class_counts = np.bincount(y_train)
scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0

lgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'objective': 'binary',
    'verbosity': -1,
}

with mlflow.start_run(run_name="lightgbm_classificacao_risco") as run:
    mlflow.log_params(lgb_params)

    import time
    start = time.time()
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train)
    train_duration = time.time() - start

    y_pred = lgb_model.predict(X_test)
    y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]

    lgb_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'train_duration_seconds': train_duration,
    }
    mlflow.log_metrics(lgb_metrics)
    mlflow.lightgbm.log_model(lgb_model, "model")

print(f"✅ LightGBM treinado em {train_duration:.2f}s")
for k, v in lgb_metrics.items():
    print(f"   - {k}: {v:.4f}" if isinstance(v, float) else f"   - {k}: {v}")

# COMMAND ----------

# DBTITLE 1,Comparação Final
# MAGIC %md
# MAGIC ## 4. Comparação: XGBoost (manual) vs. LightGBM vs. AutoML
# MAGIC
# MAGIC As métricas do XGBoost vêm do MLflow (run logada em `01_modelo_classificacao_risco.py`,
# MAGIC mesmo experimento) — não retreina aqui, só compara o que já foi registrado.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name(
    f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/credit_risk_mlops"
)

comparacao = [{
    'modelo': 'LightGBM (manual)',
    'auc_roc': lgb_metrics['auc_roc'],
    'f1_score': lgb_metrics['f1_score'],
    'tempo_treino_s': round(lgb_metrics['train_duration_seconds'], 2),
}]

if automl_summary is not None:
    comparacao.append({
        'modelo': f"AutoML ({automl_summary.best_trial.model_description})",
        'auc_roc': automl_summary.best_trial.metrics.get('val_roc_auc_score', None),
        'f1_score': automl_summary.best_trial.metrics.get('val_f1_score', None),
        'tempo_treino_s': None,
    })

if experiment is not None:
    xgb_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName LIKE 'classificacao_risco_%'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if xgb_runs:
        r = xgb_runs[0]
        comparacao.insert(0, {
            'modelo': 'XGBoost (manual, 01_modelo_classificacao_risco.py)',
            'auc_roc': r.data.metrics.get('auc_roc'),
            'f1_score': r.data.metrics.get('f1_score'),
            'tempo_treino_s': None,
        })

df_comparacao = pd.DataFrame(comparacao)
print("\n" + "="*70)
print("📊 COMPARAÇÃO DE ALGORITMOS")
print("="*70)
print(df_comparacao.to_string(index=False))
print("="*70)

# COMMAND ----------

# DBTITLE 1,Conclusão
# MAGIC %md
# MAGIC ## 5. Conclusão
# MAGIC
# MAGIC - **AutoML** é o caminho certo para um baseline rápido e para validar se vale a pena investir
# MAGIC   tempo de time em tuning manual — se o ganho do modelo manual sobre o AutoML for pequeno,
# MAGIC   pode não justificar o esforço adicional dependendo do prazo do projeto.
# MAGIC - **LightGBM vs. XGBoost**: em datasets maiores que o sintético usado aqui, a diferença de
# MAGIC   tempo de treino do LightGBM tende a ficar mais relevante (é otimizado para volume); com
# MAGIC   métricas de qualidade geralmente próximas para dados tabulares como este.
# MAGIC - O modelo que vai pro Model Registry como Champion continua sendo decidido por
# MAGIC   `05_mlops/01_mlops_pipeline.py` — este notebook é exploratório/comparativo, não substitui
# MAGIC   o pipeline de promoção.
