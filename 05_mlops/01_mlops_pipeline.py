# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,📋 Pipeline MLOps - Classificação de Risco de Crédito
# MAGIC %md
# MAGIC # Pipeline MLOps - Classificação de Risco de Crédito
# MAGIC
# MAGIC ## Objetivo
# MAGIC Este notebook implementa um pipeline completo de MLOps para o modelo de classificação de risco de crédito, incluindo:
# MAGIC
# MAGIC - **Re-treinamento automatizado**: Pipeline parametrizado para atualização do modelo
# MAGIC - **Monitoramento de Drift**: Detecção de mudanças em dados, conceito e target
# MAGIC - **Métricas de Produção**: Tracking contínuo de performance e KPIs
# MAGIC - **Sistema de Alertas**: Notificações automáticas para degradação de performance
# MAGIC - **Dashboard de Monitoramento**: Queries para visualização de métricas
# MAGIC - **Governança**: Documentação, versionamento e estratégias de rollback
# MAGIC
# MAGIC ## Contexto do Modelo
# MAGIC - **Modelo Base**: XGBoost Classifier
# MAGIC - **Performance Atual**: AUC-ROC ≥ 0.85, F1-Score ≥ 0.75
# MAGIC - **Dados**: credit_risk.gold.features_ml (1,000 clientes)
# MAGIC - **Predições**: credit_risk.gold.model_predictions
# MAGIC - **Artefatos**: /tmp/xgboost_model.pkl
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,📦 Instalação de Dependências
# Instalar bibliotecas necessárias para MLOps
%pip install mlflow xgboost scikit-learn scipy --quiet
print("✅ Dependências instaladas (usando drift detection simplificado)")

# COMMAND ----------

# DBTITLE 1,📚 Imports e Bibliotecas
# Imports principais
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import json

# Machine Learning
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# MLflow para tracking
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# Drift detection simplificado (sem evidently)
from scipy import stats

# Databricks
from pyspark.sql import functions as F
from pyspark.sql.types import *

print("✅ Bibliotecas importadas com sucesso")

# COMMAND ----------

# DBTITLE 1,⚙️ Configuração do Ambiente
# MAGIC %md
# MAGIC ## 2. Configuração do Ambiente
# MAGIC
# MAGIC Definição de parâmetros globais, conexões e caminhos para o pipeline MLOps.

# COMMAND ----------

# DBTITLE 1,🔧 Parâmetros de Configuração
# Configurações do Unity Catalog
dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA_GOLD = "gold"

# Tabelas
TABLE_FEATURES = f"{CATALOG}.{SCHEMA_GOLD}.features_ml"
TABLE_PREDICTIONS = f"{CATALOG}.{SCHEMA_GOLD}.model_predictions"
TABLE_METRICS = f"{CATALOG}.{SCHEMA_GOLD}.model_metrics"
TABLE_ALERTS = f"{CATALOG}.{SCHEMA_GOLD}.model_alerts"
TABLE_VERSIONS = f"{CATALOG}.{SCHEMA_GOLD}.model_versions"

# Modelo
MODEL_NAME = "credit_risk_classifier"
MODEL_PATH = "/tmp/xgboost_model.pkl"
MODEL_REGISTRY_NAME = f"{CATALOG}.{SCHEMA_GOLD}.{MODEL_NAME}"

# Thresholds para alertas
ALERT_THRESHOLDS = {
    "auc_roc_min": 0.75,
    "f1_score_min": 0.65,
    "drift_max": 0.30,
    "volume_min": 50,
    "volume_max": 2000
}

# MLflow
mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/credit_risk_mlops")

print(f"✅ Configuração definida:")
print(f"   - Catalog: {CATALOG}")
print(f"   - Schema: {SCHEMA_GOLD}")
print(f"   - Model: {MODEL_NAME}")
print(f"   - Thresholds: {ALERT_THRESHOLDS}")

# COMMAND ----------

# DBTITLE 1,📥 Carregar Modelo Atual
# Carregar modelo XGBoost salvo
try:
    with open(MODEL_PATH, 'rb') as f:
        current_model = pickle.load(f)
    print(f"✅ Modelo carregado de: {MODEL_PATH}")
    print(f"   Tipo: {type(current_model)}")
except FileNotFoundError:
    print("⚠️ Modelo não encontrado. Será necessário treinar um novo modelo.")
    current_model = None

# COMMAND ----------

# DBTITLE 1,🔄 Pipeline de Re-treinamento
# MAGIC %md
# MAGIC ## 3. Pipeline de Re-treinamento
# MAGIC
# MAGIC Funções parametrizadas para re-treinamento automatizado do modelo, incluindo:
# MAGIC - Carregamento de dados incrementais
# MAGIC - Feature engineering
# MAGIC - Treinamento e validação
# MAGIC - Decisão de deploy baseada em métricas

# COMMAND ----------

# DBTITLE 1,📊 Função: Carregar Dados
def load_data(table_name, date_filter=None, sample_fraction=1.0):
    """
    Carrega dados de features para treinamento/validação.
    
    Args:
        table_name: Nome completo da tabela Unity Catalog
        date_filter: Data mínima para filtrar dados novos (opcional)
        sample_fraction: Fração dos dados a carregar (0.0-1.0)
    
    Returns:
        DataFrame Spark com features
    """
    print(f"📥 Carregando dados de: {table_name}")
    
    df = spark.table(table_name)
    
    # Filtrar por data se especificado
    if date_filter:
        df = df.filter(F.col("data_referencia") >= date_filter)
        print(f"   Filtrado: data >= {date_filter}")
    
    # Amostragem se necessário
    if sample_fraction < 1.0:
        df = df.sample(fraction=sample_fraction, seed=42)
        print(f"   Amostragem: {sample_fraction*100}%")
    
    count = df.count()
    print(f"✅ Dados carregados: {count:,} registros")
    
    return df

# Teste
df_features = load_data(TABLE_FEATURES)
print(f"\n📊 Schema: {len(df_features.columns)} colunas")
display(df_features.limit(3))

# COMMAND ----------

# DBTITLE 1,🏗️ Função: Feature Engineering
def prepare_features(df, target_col="perfil_comportamental", exclude_cols=None):
    """
    Prepara features para treinamento (reutilizando lógica da FASE 2).
    
    Args:
        df: DataFrame Spark com dados brutos
        target_col: Nome da coluna target
        exclude_cols: Lista de colunas a excluir
    
    Returns:
        X (features), y (target), feature_names
    """
    print("🏗️ Preparando features...")
    
    # Converter para Pandas
    df_pandas = df.toPandas()
    
    # Definir colunas a excluir
    if exclude_cols is None:
        exclude_cols = [
            "id_cliente", "cnpj", "nome", "perfil_comportamental",
            "categoria_rfm"  # Coluna categórica redundante
        ]
    
    # Separar features e target
    feature_cols = [col for col in df_pandas.columns if col not in exclude_cols]
    
    X = df_pandas[feature_cols].copy()
    y = df_pandas[target_col] if target_col in df_pandas.columns else None
    
    # Encoding de features categóricas
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        from sklearn.preprocessing import LabelEncoder
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        print(f"   - Features categóricas encodadas: {list(categorical_cols)}")
    
    # Converter target para numérico se necessário (Label Encoding)
    if y is not None and y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=y.name)
        print(f"   - Classes mapeadas: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    print(f"✅ Features preparadas:")
    print(f"   - Features: {X.shape[1]} colunas")
    print(f"   - Amostras: {X.shape[0]:,} registros")
    if y is not None:
        print(f"   - Target: {y.name} (classes: {y.nunique()})")
    
    return X, y, feature_cols

# Teste
X, y, feature_names = prepare_features(df_features)
print(f"\n📋 Features: {feature_names[:10]}...") if len(feature_names) > 10 else print(f"\n📋 Features: {feature_names}")

# COMMAND ----------

# DBTITLE 1,🎯 Função: Treinar Modelo
def train_model(X_train, y_train, X_test, y_test, params=None, log_mlflow=True):
    """
    Treina modelo XGBoost com parâmetros customizáveis.
    
    Args:
        X_train, y_train: Dados de treinamento
        X_test, y_test: Dados de teste
        params: Hiperparâmetros do XGBoost (opcional)
        log_mlflow: Se True, registra no MLflow
    
    Returns:
        modelo treinado, métricas de avaliação
    """
    print("🎯 Treinando modelo XGBoost...")
    
    # Parâmetros padrão
    if params is None:
        # Calcular class weights
        class_counts = np.bincount(y_train)
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42
        }
    
    # Treinar
    if log_mlflow:
        mlflow.start_run(run_name=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params(params)
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Avaliar
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) == 2 else 0.0
    }
    
    if log_mlflow:
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")
        mlflow.end_run()
    
    print(f"✅ Modelo treinado:")
    for metric, value in metrics.items():
        print(f"   - {metric}: {value:.4f}")
    
    return model, metrics

# COMMAND ----------

# DBTITLE 1,✅ Função: Validar e Decidir Deploy
def should_deploy(new_metrics, baseline_metrics, threshold=0.02):
    """
    Decide se o novo modelo deve ser deployado baseado em métricas.
    
    Args:
        new_metrics: Métricas do novo modelo
        baseline_metrics: Métricas do modelo atual em produção
        threshold: Melhoria mínima necessária para deploy
    
    Returns:
        (bool, str): (deve_deployar, justificativa)
    """
    print("✅ Validando modelo para deploy...")
    
    # Métricas principais para comparação
    key_metrics = ['auc_roc', 'f1_score']
    
    improvements = {}
    for metric in key_metrics:
        if metric in new_metrics and metric in baseline_metrics:
            improvement = new_metrics[metric] - baseline_metrics[metric]
            improvements[metric] = improvement
    
    # Decisão: pelo menos uma métrica deve melhorar acima do threshold
    significant_improvements = {k: v for k, v in improvements.items() if v > threshold}
    
    if significant_improvements:
        best_improvement = max(significant_improvements.items(), key=lambda x: x[1])
        justification = f"✅ DEPLOY APROVADO: {best_improvement[0]} melhorou +{best_improvement[1]:.4f}"
        should_deploy_flag = True
    else:
        justification = f"❌ DEPLOY NEGADO: Nenhuma melhoria significativa (threshold: {threshold})"
        should_deploy_flag = False
    
    print(f"\n{justification}")
    print(f"\n📊 Comparação:")
    for metric in key_metrics:
        if metric in improvements:
            delta = improvements[metric]
            symbol = "📈" if delta > 0 else "📉"
            print(f"   {symbol} {metric}: {baseline_metrics.get(metric, 0):.4f} → {new_metrics.get(metric, 0):.4f} ({delta:+.4f})")
    
    return should_deploy_flag, justification

# COMMAND ----------

# DBTITLE 1,🔍 Monitoramento de Drift
# MAGIC %md
# MAGIC ## 4. Monitoramento de Drift
# MAGIC
# MAGIC Detecção de mudanças nas distribuições usando Evidently AI:
# MAGIC - **Data Drift**: Mudanças nas features de entrada
# MAGIC - **Target Drift**: Mudanças na variável alvo
# MAGIC - **Concept Drift**: Mudanças na relação features → target

# COMMAND ----------

# DBTITLE 1,📊 Função: Detectar Data Drift
def detect_data_drift(reference_data, current_data, feature_cols, p_value_threshold=0.05):
    """
    Detecta drift nas features usando testes estatísticos (Kolmogorov-Smirnov).
    
    Args:
        reference_data: Dados de referência (baseline)
        current_data: Dados atuais (produção)
        feature_cols: Lista de features a monitorar
        p_value_threshold: Threshold para detecção de drift (padrão: 0.05)
    
    Returns:
        dict com resultados de drift
    """
    print("📊 Detectando Data Drift (método simplificado - KS test)...")
    
    drift_results = {
        'timestamp': datetime.now().isoformat(),
        'n_features': len(feature_cols),
        'n_drifted_features': 0,
        'drift_share': 0.0,
        'dataset_drift': False,
        'drifted_features': []
    }
    
    # Testar cada feature individualmente
    for col in feature_cols:
        try:
            # Kolmogorov-Smirnov test para comparar distribuições
            statistic, p_value = stats.ks_2samp(reference_data[col], current_data[col])
            
            # Drift detectado se p-value < threshold
            if p_value < p_value_threshold:
                drift_results['drifted_features'].append({
                    'feature': col,
                    'drift_score': float(statistic),
                    'p_value': float(p_value)
                })
        except Exception as e:
            print(f"⚠️ Erro ao testar {col}: {e}")
    
    # Calcular estatísticas gerais
    drift_results['n_drifted_features'] = len(drift_results['drifted_features'])
    drift_results['drift_share'] = drift_results['n_drifted_features'] / drift_results['n_features']
    drift_results['dataset_drift'] = drift_results['drift_share'] > 0.3  # >30% features com drift
    
    print(f"✅ Drift detectado:")
    print(f"   - Features analisadas: {drift_results['n_features']}")
    print(f"   - Features com drift: {drift_results['n_drifted_features']}")
    print(f"   - % drift: {drift_results['drift_share']*100:.1f}%")
    print(f"   - Dataset drift: {drift_results['dataset_drift']}")
    
    return drift_results, None

# COMMAND ----------

# DBTITLE 1,🎯 Função: Detectar Target Drift
def detect_target_drift(reference_data, current_data, target_col, p_value_threshold=0.05):
    """
    Detecta drift na variável target usando teste qui-quadrado.
    
    Args:
        reference_data: Dados de referência com target
        current_data: Dados atuais com target
        target_col: Nome da coluna target
        p_value_threshold: Threshold para detecção de drift
    
    Returns:
        dict com resultados de target drift
    """
    print("🎯 Detectando Target Drift (método simplificado - Chi-square test)...")
    
    drift_results = {
        'timestamp': datetime.now().isoformat(),
        'target_name': target_col,
        'drift_detected': False,
        'drift_score': 0.0,
        'p_value': 1.0
    }
    
    try:
        # Calcular distribuições
        ref_counts = reference_data[target_col].value_counts(normalize=True).sort_index()
        curr_counts = current_data[target_col].value_counts(normalize=True).sort_index()
        
        # Alinhar índices
        all_classes = sorted(set(ref_counts.index) | set(curr_counts.index))
        ref_dist = [ref_counts.get(c, 0) for c in all_classes]
        curr_dist = [curr_counts.get(c, 0) for c in all_classes]
        
        # Converter para contagens para chi-square
        ref_n = len(reference_data)
        curr_n = len(current_data)
        observed = [int(p * curr_n) for p in curr_dist]
        expected = [int(p * curr_n) for p in ref_dist]
        
        # Qui-quadrado test
        if sum(expected) > 0:
            statistic, p_value = stats.chisquare(observed, expected)
            
            drift_results['drift_score'] = float(statistic)
            drift_results['p_value'] = float(p_value)
            drift_results['drift_detected'] = p_value < p_value_threshold
    except Exception as e:
        print(f"⚠️ Erro ao calcular target drift: {e}")
    
    print(f"✅ Target Drift:")
    print(f"   - Drift detectado: {drift_results['drift_detected']}")
    print(f"   - Drift score: {drift_results['drift_score']:.4f}")
    print(f"   - P-value: {drift_results['p_value']:.4f}")
    
    return drift_results, None

# COMMAND ----------

# DBTITLE 1,📈 Visualizar Drift por Feature
def visualize_drift(drift_results):
    """
    Visualiza features com drift detectado.
    
    Args:
        drift_results: Resultados do detect_data_drift
    """
    if not drift_results['drifted_features']:
        print("✅ Nenhuma feature com drift significativo")
        return
    
    # Criar DataFrame para visualização
    drift_df = pd.DataFrame(drift_results['drifted_features'])
    drift_df = drift_df.sort_values('drift_score', ascending=False)
    
    print(f"\n🚨 Top Features com Drift:")
    print(drift_df.head(10).to_string(index=False))
    
    # Visualização gráfica (se necessário)
    if len(drift_df) > 0:
        display(drift_df.head(20))

# COMMAND ----------

# DBTITLE 1,📊 Métricas de Produção
# MAGIC %md
# MAGIC ## 5. Métricas de Produção
# MAGIC
# MAGIC Tracking contínuo de performance do modelo em produção:
# MAGIC - KPIs principais (precision, recall, F1, AUC-ROC)
# MAGIC - Volume de predições
# MAGIC - Distribuição de scores
# MAGIC - Taxa de inadimplência real vs predita

# COMMAND ----------

# DBTITLE 1,🗄️ Criar Tabela de Métricas
# Criar tabela para armazenar métricas de produção
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TABLE_METRICS} (
    metric_id STRING,
    timestamp TIMESTAMP,
    model_version STRING,
    period_type STRING COMMENT 'daily, weekly, monthly',
    period_start DATE,
    period_end DATE,
    
    -- Métricas de classificação
    accuracy DOUBLE,
    precision_score DOUBLE,
    recall_score DOUBLE,
    f1_score DOUBLE,
    auc_roc DOUBLE,
    
    -- Volume e distribuição
    n_predictions INT,
    n_high_risk INT,
    n_medium_risk INT,
    n_low_risk INT,
    
    -- Scores
    avg_score DOUBLE,
    median_score DOUBLE,
    
    -- Taxa de inadimplência
    actual_default_rate DOUBLE,
    predicted_default_rate DOUBLE,
    
    -- Metadata
    created_at TIMESTAMP,
    created_by STRING
)
USING DELTA
COMMENT 'Métricas de produção do modelo de risco de crédito'
""")

print(f"✅ Tabela criada: {TABLE_METRICS}")

# COMMAND ----------

# DBTITLE 1,📝 Função: Registrar Métricas
def log_production_metrics(predictions_df, actual_df, model_version, period_type='daily'):
    """
    Registra métricas de produção na tabela.
    
    Args:
        predictions_df: DataFrame com predições
        actual_df: DataFrame com resultados reais (se disponível)
        model_version: Versão do modelo
        period_type: Tipo de período (daily, weekly, monthly)
    
    Returns:
        dict com métricas calculadas
    """
    print(f"📝 Registrando métricas de produção ({period_type})...")
    
    # Converter para Pandas
    pred_pandas = predictions_df.toPandas()
    
    # Calcular métricas básicas
    metrics = {
        'metric_id': f"{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'timestamp': datetime.now(),
        'model_version': model_version,
        'period_type': period_type,
        'period_start': pred_pandas['data_referencia'].min() if 'data_referencia' in pred_pandas.columns else datetime.now().date(),
        'period_end': pred_pandas['data_referencia'].max() if 'data_referencia' in pred_pandas.columns else datetime.now().date(),
        
        # Volume (converter para int para compatibilidade com Spark INT)
        'n_predictions': int(len(pred_pandas)),
        'n_high_risk': int(len(pred_pandas[pred_pandas['categoria_predita'] == 'Alto'])) if 'categoria_predita' in pred_pandas.columns else 0,
        'n_medium_risk': int(len(pred_pandas[pred_pandas['categoria_predita'] == 'Médio'])) if 'categoria_predita' in pred_pandas.columns else 0,
        'n_low_risk': int(len(pred_pandas[pred_pandas['categoria_predita'] == 'Baixo'])) if 'categoria_predita' in pred_pandas.columns else 0,
        
        # Scores
        'avg_score': float(pred_pandas['score_risco'].mean()) if 'score_risco' in pred_pandas.columns else 0.0,
        'median_score': float(pred_pandas['score_risco'].median()) if 'score_risco' in pred_pandas.columns else 0.0,
        
        # Placeholder para métricas que requerem ground truth (usar 0.0 ao invés de None)
        'accuracy': 0.0,
        'precision_score': 0.0,
        'recall_score': 0.0,
        'f1_score': 0.0,
        'auc_roc': 0.0,
        'actual_default_rate': 0.0,
        'predicted_default_rate': float(pred_pandas['probabilidade_inadimplencia'].mean()) if 'probabilidade_inadimplencia' in pred_pandas.columns else 0.0,
        
        'created_at': datetime.now(),
        'created_by': spark.sql('SELECT current_user()').collect()[0][0]
    }
    
    # Se temos ground truth, calcular métricas completas
    if actual_df is not None:
        # TODO: Join predictions com actual e calcular métricas
        pass
    
    # Inserir na tabela usando SQL para evitar problemas de type inference
    spark.sql(f"""
        INSERT INTO {TABLE_METRICS}
        VALUES (
            '{metrics['metric_id']}',
            timestamp'{metrics['timestamp']}',
            '{metrics['model_version']}',
            '{metrics['period_type']}',
            date'{metrics['period_start']}',
            date'{metrics['period_end']}',
            {metrics['accuracy']},
            {metrics['precision_score']},
            {metrics['recall_score']},
            {metrics['f1_score']},
            {metrics['auc_roc']},
            {metrics['n_predictions']},
            {metrics['n_high_risk']},
            {metrics['n_medium_risk']},
            {metrics['n_low_risk']},
            {metrics['avg_score']},
            {metrics['median_score']},
            {metrics['actual_default_rate']},
            {metrics['predicted_default_rate']},
            timestamp'{metrics['created_at']}',
            '{metrics['created_by']}'
        )
    """)
    
    print(f"✅ Métricas registradas: {metrics['metric_id']}")
    print(f"   - Período: {metrics['period_start']} a {metrics['period_end']}")
    print(f"   - Predições: {metrics['n_predictions']:,}")
    print(f"   - Score médio: {metrics['avg_score']:.4f}")
    
    return metrics

# COMMAND ----------

# DBTITLE 1,📊 Query: Métricas por Período
# MAGIC %sql
# MAGIC -- Visualizar métricas agregadas por período
# MAGIC SELECT 
# MAGIC     period_type,
# MAGIC     period_start,
# MAGIC     period_end,
# MAGIC     model_version,
# MAGIC     n_predictions,
# MAGIC     avg_score,
# MAGIC     predicted_default_rate,
# MAGIC     ROUND(n_high_risk * 100.0 / n_predictions, 2) as pct_high_risk,
# MAGIC     timestamp
# MAGIC FROM credit_risk.gold.model_metrics
# MAGIC ORDER BY timestamp DESC
# MAGIC LIMIT 20

# COMMAND ----------

# DBTITLE 1,🚨 Sistema de Alertas
# MAGIC %md
# MAGIC ## 6. Sistema de Alertas
# MAGIC
# MAGIC Monitoramento automático com notificações para:
# MAGIC - Queda de performance abaixo de thresholds
# MAGIC - Drift severo detectado
# MAGIC - Volume anormal de predições
# MAGIC - Mudanças significativas na distribuição de scores

# COMMAND ----------

# DBTITLE 1,🗄️ Criar Tabela de Alertas
# Criar tabela para registrar alertas
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TABLE_ALERTS} (
    alert_id STRING,
    timestamp TIMESTAMP,
    alert_type STRING COMMENT 'performance, drift, volume, distribution',
    severity STRING COMMENT 'low, medium, high, critical',
    
    -- Detalhes do alerta
    metric_name STRING,
    current_value DOUBLE,
    threshold_value DOUBLE,
    deviation DOUBLE,
    
    -- Contexto
    model_version STRING,
    period_start DATE,
    period_end DATE,
    
    -- Mensagem
    message STRING,
    
    -- Status
    status STRING COMMENT 'open, acknowledged, resolved',
    resolved_at TIMESTAMP,
    resolved_by STRING,
    
    -- Metadata
    created_at TIMESTAMP,
    created_by STRING
)
USING DELTA
COMMENT 'Alertas do sistema de monitoramento MLOps'
""")

print(f"✅ Tabela criada: {TABLE_ALERTS}")

# COMMAND ----------

# DBTITLE 1,🔔 Função: Verificar Alertas
def check_alerts(metrics, drift_results, thresholds=ALERT_THRESHOLDS):
    """
    Verifica condições de alerta baseado em métricas e drift.
    
    Args:
        metrics: Dict com métricas de performance
        drift_results: Dict com resultados de drift
        thresholds: Dict com thresholds de alerta
    
    Returns:
        list de alertas detectados
    """
    print("🔔 Verificando alertas...")
    
    alerts = []
    timestamp = datetime.now()
    
    # 1. Alerta de Performance
    if 'auc_roc' in metrics and metrics['auc_roc'] < thresholds['auc_roc_min']:
        alerts.append({
            'alert_id': f"PERF_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            'timestamp': timestamp,
            'alert_type': 'performance',
            'severity': 'high',
            'metric_name': 'auc_roc',
            'current_value': metrics['auc_roc'],
            'threshold_value': thresholds['auc_roc_min'],
            'deviation': metrics['auc_roc'] - thresholds['auc_roc_min'],
            'message': f"AUC-ROC ({metrics['auc_roc']:.4f}) abaixo do threshold ({thresholds['auc_roc_min']:.4f})",
            'status': 'open',
            'created_at': timestamp
        })
    
    if 'f1_score' in metrics and metrics['f1_score'] < thresholds['f1_score_min']:
        alerts.append({
            'alert_id': f"PERF_{timestamp.strftime('%Y%m%d_%H%M%S')}_F1",
            'timestamp': timestamp,
            'alert_type': 'performance',
            'severity': 'high',
            'metric_name': 'f1_score',
            'current_value': metrics['f1_score'],
            'threshold_value': thresholds['f1_score_min'],
            'deviation': metrics['f1_score'] - thresholds['f1_score_min'],
            'message': f"F1-Score ({metrics['f1_score']:.4f}) abaixo do threshold ({thresholds['f1_score_min']:.4f})",
            'status': 'open',
            'created_at': timestamp
        })
    
    # 2. Alerta de Drift
    if drift_results and drift_results.get('drift_share', 0) > thresholds['drift_max']:
        alerts.append({
            'alert_id': f"DRIFT_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            'timestamp': timestamp,
            'alert_type': 'drift',
            'severity': 'critical' if drift_results['drift_share'] > 0.5 else 'high',
            'metric_name': 'drift_share',
            'current_value': drift_results['drift_share'],
            'threshold_value': thresholds['drift_max'],
            'deviation': drift_results['drift_share'] - thresholds['drift_max'],
            'message': f"Data drift severo detectado: {drift_results['drift_share']*100:.1f}% das features ({drift_results['n_drifted_features']}/{drift_results['n_features']})",
            'status': 'open',
            'created_at': timestamp
        })
    
    # 3. Alerta de Volume
    if 'n_predictions' in metrics:
        if metrics['n_predictions'] < thresholds['volume_min']:
            alerts.append({
                'alert_id': f"VOL_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                'timestamp': timestamp,
                'alert_type': 'volume',
                'severity': 'medium',
                'metric_name': 'n_predictions',
                'current_value': float(metrics['n_predictions']),
                'threshold_value': float(thresholds['volume_min']),
                'deviation': float(metrics['n_predictions'] - thresholds['volume_min']),
                'message': f"Volume de predições anormalmente baixo: {metrics['n_predictions']} (min: {thresholds['volume_min']})",
                'status': 'open',
                'created_at': timestamp
            })
        elif metrics['n_predictions'] > thresholds['volume_max']:
            alerts.append({
                'alert_id': f"VOL_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                'timestamp': timestamp,
                'alert_type': 'volume',
                'severity': 'medium',
                'metric_name': 'n_predictions',
                'current_value': float(metrics['n_predictions']),
                'threshold_value': float(thresholds['volume_max']),
                'deviation': float(metrics['n_predictions'] - thresholds['volume_max']),
                'message': f"Volume de predições anormalmente alto: {metrics['n_predictions']} (max: {thresholds['volume_max']})",
                'status': 'open',
                'created_at': timestamp
            })
    
    # Registrar alertas na tabela
    if alerts:
        # Registrar alertas usando SQL INSERT
        current_user = spark.sql('SELECT current_user()').collect()[0][0]
        model_version = metrics.get('model_version', 'unknown')
        
        for alert in alerts:
            spark.sql(f"""
                INSERT INTO {TABLE_ALERTS}
                VALUES (
                    '{alert['alert_id']}',
                    timestamp'{alert['timestamp']}',
                    '{alert['alert_type']}',
                    '{alert['severity']}',
                    '{alert['metric_name']}',
                    {alert['current_value']},
                    {alert['threshold_value']},
                    {alert['deviation']},
                    '{model_version}',
                    NULL,
                    NULL,
                    '{alert['message'].replace("'", "''")}',
                    '{alert['status']}',
                    NULL,
                    NULL,
                    timestamp'{alert['created_at']}',
                    '{current_user}'
                )
            """)
        
        print(f"\n🚨 {len(alerts)} ALERTA(S) DETECTADO(S):")
        for alert in alerts:
            severity_icon = {"low": "ℹ️", "medium": "⚠️", "high": "🔴", "critical": "🚨"}[alert['severity']]
            print(f"   {severity_icon} [{alert['alert_type'].upper()}] {alert['message']}")
    else:
        print("✅ Nenhum alerta detectado")
    
    return alerts

# COMMAND ----------

# DBTITLE 1,📊 Query: Alertas Ativos
# MAGIC %sql
# MAGIC -- Visualizar alertas ativos
# MAGIC SELECT 
# MAGIC     alert_type,
# MAGIC     severity,
# MAGIC     metric_name,
# MAGIC     ROUND(current_value, 4) as current_value,
# MAGIC     ROUND(threshold_value, 4) as threshold_value,
# MAGIC     message,
# MAGIC     status,
# MAGIC     timestamp
# MAGIC FROM credit_risk.gold.model_alerts
# MAGIC WHERE status = 'open'
# MAGIC ORDER BY 
# MAGIC     CASE severity 
# MAGIC         WHEN 'critical' THEN 1
# MAGIC         WHEN 'high' THEN 2
# MAGIC         WHEN 'medium' THEN 3
# MAGIC         ELSE 4
# MAGIC     END,
# MAGIC     timestamp DESC

# COMMAND ----------

# DBTITLE 1,📊 Dashboard de Monitoramento
# MAGIC %md
# MAGIC ## 7. Dashboard de Monitoramento
# MAGIC
# MAGIC Queries SQL para criação de dashboards:
# MAGIC - Tendências de performance ao longo do tempo
# MAGIC - Status de drift por feature
# MAGIC - Volume e distribuição de predições
# MAGIC - Alertas por tipo e severidade

# COMMAND ----------

# DBTITLE 1,📈 Query: Tendências de Performance
# MAGIC %sql
# MAGIC -- Tendências de métricas ao longo do tempo
# MAGIC SELECT 
# MAGIC     DATE(timestamp) as date,
# MAGIC     model_version,
# MAGIC     AVG(auc_roc) as avg_auc_roc,
# MAGIC     AVG(f1_score) as avg_f1_score,
# MAGIC     AVG(precision_score) as avg_precision,
# MAGIC     AVG(recall_score) as avg_recall,
# MAGIC     SUM(n_predictions) as total_predictions
# MAGIC FROM credit_risk.gold.model_metrics
# MAGIC WHERE timestamp >= DATE_SUB(CURRENT_DATE(), 30)  -- Últimos 30 dias
# MAGIC GROUP BY DATE(timestamp), model_version
# MAGIC ORDER BY date DESC

# COMMAND ----------

# DBTITLE 1,📊 Query: Distribuição de Scores
# MAGIC %sql
# MAGIC -- Distribuição de scores de risco por período
# MAGIC SELECT 
# MAGIC     period_type,
# MAGIC     period_start,
# MAGIC     ROUND(n_low_risk * 100.0 / n_predictions, 1) as pct_low_risk,
# MAGIC     ROUND(n_medium_risk * 100.0 / n_predictions, 1) as pct_medium_risk,
# MAGIC     ROUND(n_high_risk * 100.0 / n_predictions, 1) as pct_high_risk,
# MAGIC     ROUND(avg_score, 4) as avg_score,
# MAGIC     ROUND(predicted_default_rate * 100, 2) as predicted_default_pct,
# MAGIC     n_predictions
# MAGIC FROM credit_risk.gold.model_metrics
# MAGIC WHERE timestamp >= DATE_SUB(CURRENT_DATE(), 7)  -- Última semana
# MAGIC ORDER BY timestamp DESC

# COMMAND ----------

# DBTITLE 1,🚨 Query: Resumo de Alertas
# MAGIC %sql
# MAGIC -- Resumo de alertas por tipo e severidade
# MAGIC SELECT 
# MAGIC     alert_type,
# MAGIC     severity,
# MAGIC     COUNT(*) as count_alerts,
# MAGIC     COUNT(CASE WHEN status = 'open' THEN 1 END) as open_alerts,
# MAGIC     COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved_alerts,
# MAGIC     MAX(timestamp) as last_alert
# MAGIC FROM credit_risk.gold.model_alerts
# MAGIC WHERE timestamp >= DATE_SUB(CURRENT_DATE(), 30)  -- Últimos 30 dias
# MAGIC GROUP BY alert_type, severity
# MAGIC ORDER BY 
# MAGIC     CASE severity 
# MAGIC         WHEN 'critical' THEN 1
# MAGIC         WHEN 'high' THEN 2
# MAGIC         WHEN 'medium' THEN 3
# MAGIC         ELSE 4
# MAGIC     END,
# MAGIC     count_alerts DESC

# COMMAND ----------

# DBTITLE 1,📋 Documentação e Governança
# MAGIC %md
# MAGIC ## 8. Documentação e Governança
# MAGIC
# MAGIC Processos, políticas e responsabilidades para gestão do modelo em produção.

# COMMAND ----------

# DBTITLE 1,✅ Processo de Aprovação de Modelo
# MAGIC %md
# MAGIC ### Processo de Aprovação de Modelo
# MAGIC
# MAGIC #### Critérios de Aprovação
# MAGIC 1. **Performance Mínima**:
# MAGIC    - AUC-ROC ≥ 0.75
# MAGIC    - F1-Score ≥ 0.65
# MAGIC    - Precision ≥ 0.70
# MAGIC
# MAGIC 2. **Validação de Drift**:
# MAGIC    - Data drift < 30% das features
# MAGIC    - Target drift score < 0.25
# MAGIC
# MAGIC 3. **Estabilidade**:
# MAGIC    - Testado em pelo menos 3 períodos diferentes
# MAGIC    - Variância de métricas < 5%
# MAGIC
# MAGIC 4. **Documentação**:
# MAGIC    - Relatório de treinamento completo
# MAGIC    - Análise de features importantes
# MAGIC    - Casos de teste documentados
# MAGIC
# MAGIC #### Fluxo de Aprovação
# MAGIC 1. **Data Scientist**: Treina e valida modelo
# MAGIC 2. **ML Engineer**: Revisa código e pipeline
# MAGIC 3. **Risk Manager**: Valida critérios de negócio
# MAGIC 4. **Compliance**: Aprova governança e auditoria
# MAGIC 5. **Deploy**: Automatizado após todas as aprovações
# MAGIC
# MAGIC #### SLAs
# MAGIC - Revisão de código: 2 dias úteis
# MAGIC - Validação de negócio: 3 dias úteis
# MAGIC - Compliance: 5 dias úteis
# MAGIC - Deploy em produção: 1 dia útil após aprovação final

# COMMAND ----------

# DBTITLE 1,🔄 Estratégia de Rollback
# MAGIC %md
# MAGIC ### Estratégia de Rollback
# MAGIC
# MAGIC #### Condições para Rollback Automático
# MAGIC 1. **Performance**: AUC-ROC cai abaixo de 0.70
# MAGIC 2. **Volume**: Redução de >50% nas predições em 1 hora
# MAGIC 3. **Erros**: Taxa de erro >5% das predições
# MAGIC 4. **Drift Crítico**: Drift share >50%
# MAGIC
# MAGIC #### Processo de Rollback
# MAGIC 1. **Detecção**: Sistema de alertas identifica condição crítica
# MAGIC 2. **Notificação**: Alerta enviado para equipe via Slack/Email
# MAGIC 3. **Rollback Automático**: 
# MAGIC    - Reverter para versão anterior do modelo
# MAGIC    - Redirecionar tráfego para modelo stable
# MAGIC    - Período de cooldown: 1 hora
# MAGIC 4. **Investigação**: Análise de logs e métricas
# MAGIC 5. **Correção**: Fix e re-deploy
# MAGIC
# MAGIC #### Versionamento
# MAGIC - Sempre manter últimas 3 versões do modelo
# MAGIC - Modelo "canary": 10% do tráfego em novas versões
# MAGIC - Período de observação: 48 horas
# MAGIC - Rollout gradual: 10% → 50% → 100%
# MAGIC
# MAGIC #### Responsabilidades
# MAGIC - **On-call Engineer**: Responde a alertas críticos (SLA: 15 min)
# MAGIC - **ML Team**: Investiga e corrige (SLA: 2 horas)
# MAGIC - **Risk Team**: Valida impacto de negócio (SLA: 4 horas)

# COMMAND ----------

# DBTITLE 1,🗄️ Criar Tabela de Versionamento
# Criar tabela para controle de versões do modelo
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TABLE_VERSIONS} (
    version_id STRING,
    version_number INT,
    model_name STRING,
    
    -- Treinamento
    training_date TIMESTAMP,
    training_data_start DATE,
    training_data_end DATE,
    n_training_samples INT,
    
    -- Performance
    auc_roc DOUBLE,
    f1_score DOUBLE,
    precision_score DOUBLE,
    recall_score DOUBLE,
    
    -- Artefatos
    model_path STRING,
    mlflow_run_id STRING,
    
    -- Status
    status STRING COMMENT 'development, staging, production, retired',
    deployed_at TIMESTAMP,
    retired_at TIMESTAMP,
    
    -- Metadata
    created_by STRING,
    created_at TIMESTAMP,
    notes STRING
)
USING DELTA
COMMENT 'Histórico de versões do modelo'
""")

print(f"✅ Tabela criada: {TABLE_VERSIONS}")

# COMMAND ----------

# DBTITLE 1,📝 Função: Registrar Versão
def register_model_version(model, metrics, mlflow_run_id=None, status='development'):
    """
    Registra nova versão do modelo na tabela de versionamento.
    
    Args:
        model: Modelo treinado
        metrics: Dict com métricas de performance
        mlflow_run_id: ID da run no MLflow
        status: Status da versão (development, staging, production)
    
    Returns:
        version_id
    """
    print(f"📝 Registrando versão do modelo...")
    
    # Obter próximo número de versão
    existing_versions = spark.table(TABLE_VERSIONS).filter(f"model_name = '{MODEL_NAME}'")
    if existing_versions.count() > 0:
        max_version = existing_versions.agg({"version_number": "max"}).collect()[0][0]
        version_number = max_version + 1
    else:
        version_number = 1
    
    version_id = f"{MODEL_NAME}_v{version_number}"
    
    # Salvar modelo
    model_path = f"/tmp/{version_id}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Registrar versão usando SQL INSERT
    current_user = spark.sql('SELECT current_user()').collect()[0][0]
    training_date = datetime.now()
    notes = f"Treinado com AUC-ROC={metrics.get('auc_roc', 0):.4f}"
    
    spark.sql(f"""
        INSERT INTO {TABLE_VERSIONS}
        VALUES (
            '{version_id}',
            {version_number},
            '{MODEL_NAME}',
            timestamp'{training_date}',
            NULL,
            NULL,
            NULL,
            {metrics.get('auc_roc', 0.0)},
            {metrics.get('f1_score', 0.0)},
            {metrics.get('precision', 0.0)},
            {metrics.get('recall', 0.0)},
            '{model_path}',
            {f"'{mlflow_run_id}'" if mlflow_run_id else "NULL"},
            '{status}',
            NULL,
            NULL,
            '{current_user}',
            timestamp'{training_date}',
            '{notes}'
        )
    """)
    
    print(f"✅ Versão registrada: {version_id}")
    print(f"   - Status: {status}")
    print(f"   - AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
    print(f"   - Path: {model_path}")
    
    return version_id

# COMMAND ----------

# DBTITLE 1,📊 Query: Histórico de Versões
# MAGIC %sql
# MAGIC -- Histórico de versões do modelo
# MAGIC SELECT 
# MAGIC     version_id,
# MAGIC     version_number,
# MAGIC     status,
# MAGIC     ROUND(auc_roc, 4) as auc_roc,
# MAGIC     ROUND(f1_score, 4) as f1_score,
# MAGIC     training_date,
# MAGIC     deployed_at,
# MAGIC     created_by,
# MAGIC     notes
# MAGIC FROM credit_risk.gold.model_versions
# MAGIC ORDER BY version_number DESC

# COMMAND ----------

# DBTITLE 1,🔄 Execução Completa do Pipeline
# MAGIC %md
# MAGIC ## 9. Execução Completa do Pipeline
# MAGIC
# MAGIC Simulação completa do pipeline MLOps:
# MAGIC 1. Carregar dados
# MAGIC 2. Simular drift (alterando dados de teste)
# MAGIC 3. Detectar drift
# MAGIC 4. Re-treinar modelo
# MAGIC 5. Validar e decidir deploy
# MAGIC 6. Registrar métricas e alertas

# COMMAND ----------

# DBTITLE 1,🚀 Executar Pipeline Completo
# Pipeline MLOps End-to-End
print("="*80)
print("🚀 EXECUTANDO PIPELINE MLOPS COMPLETO")
print("="*80)

# 1. Carregar dados
print("\n[1/7] Carregando dados...")
df_current = load_data(TABLE_FEATURES, sample_fraction=1.0)
X_current, y_current, feature_names = prepare_features(df_current)

# Split para referência e teste
X_ref, X_test, y_ref, y_test = train_test_split(
    X_current, y_current, test_size=0.3, random_state=42, stratify=y_current
)

print(f"✅ Dados preparados:")
print(f"   - Referência: {len(X_ref):,} amostras")
print(f"   - Teste: {len(X_test):,} amostras")

# COMMAND ----------

# DBTITLE 1,🎭 Simular Drift
# 2. Simular drift nos dados de teste
print("\n[2/7] Simulando drift nos dados...")

# Drift artificial: adicionar ruído em algumas features
X_test_drift = X_test.copy()
np.random.seed(123)

# Selecionar 30% das features para adicionar drift
n_drift_features = int(len(feature_names) * 0.3)
drift_features = np.random.choice(feature_names, size=n_drift_features, replace=False)

for col in drift_features:
    if col in X_test_drift.columns:
        # Adicionar ruído gaussiano
        noise = np.random.normal(0, X_test_drift[col].std() * 0.5, size=len(X_test_drift))
        X_test_drift[col] = X_test_drift[col] + noise

print(f"✅ Drift simulado em {n_drift_features} features:")
print(f"   {list(drift_features[:5])}...")

# COMMAND ----------

# DBTITLE 1,🔍 Detectar Drift
# 3. Detectar drift
print("\n[3/7] Detectando drift...")

# Data drift
drift_results, drift_report = detect_data_drift(X_ref, X_test_drift, feature_names)

# Visualizar drift
visualize_drift(drift_results)

# Target drift
y_ref_df = pd.DataFrame({'target': y_ref})
y_test_df = pd.DataFrame({'target': y_test})
target_drift_results, target_drift_report = detect_target_drift(y_ref_df, y_test_df, 'target')

print(f"\n✅ Drift detectado:")
print(f"   - Data drift: {drift_results['dataset_drift']}")
print(f"   - Target drift: {target_drift_results['drift_detected']}")

# COMMAND ----------

# DBTITLE 1,🎯 Re-treinar Modelo
# 4. Re-treinar modelo com dados atualizados
print("\n[4/7] Re-treinando modelo...")

# Combinar referência + teste (simulando dados novos)
X_all = pd.concat([X_ref, X_test_drift], axis=0).reset_index(drop=True)
y_all = pd.concat([pd.Series(y_ref), pd.Series(y_test)], axis=0).reset_index(drop=True)

# Novo split
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# Treinar novo modelo
new_model, new_metrics = train_model(
    X_train_new, y_train_new, X_test_new, y_test_new,
    log_mlflow=True
)

print(f"\n✅ Novo modelo treinado")

# COMMAND ----------

# DBTITLE 1,✅ Validar e Decidir Deploy
# 5. Validar modelo e decidir deploy
print("\n[5/7] Validando modelo para deploy...")

# Métricas baseline (modelo atual)
baseline_metrics = {
    'auc_roc': 0.85,
    'f1_score': 0.75,
    'precision': 0.78,
    'recall': 0.73
}

# Decisão de deploy
should_deploy_flag, justification = should_deploy(
    new_metrics, baseline_metrics, threshold=0.02
)

print(f"\n{justification}")

# COMMAND ----------

# DBTITLE 1,📝 Registrar Métricas e Versão
# 6. Registrar métricas de produção
print("\n[6/7] Registrando métricas...")

# Simular predições para registrar métricas
predictions_data = {
    'cliente_id': range(1000),
    'data_referencia': [datetime.now().date()] * 1000,
    'categoria_predita': np.random.choice(['Baixo', 'Médio', 'Alto'], size=1000),
    'score_risco': np.random.uniform(0.1, 0.9, size=1000),
    'probabilidade_inadimplencia': np.random.uniform(0.05, 0.5, size=1000)
}
pred_df = spark.createDataFrame(pd.DataFrame(predictions_data))

# Registrar métricas
production_metrics = log_production_metrics(
    pred_df, None, model_version='v1.0', period_type='daily'
)

# Registrar versão
version_id = register_model_version(
    new_model, new_metrics, status='staging'
)

print(f"\n✅ Métricas e versão registradas")

# COMMAND ----------

# DBTITLE 1,🚨 Verificar Alertas
# 7. Verificar alertas
print("\n[7/7] Verificando alertas...")

# Preparar métricas combinadas para verificação de alertas
combined_metrics = {
    **new_metrics,
    **production_metrics,
    'model_version': version_id
}

# Verificar alertas
alerts = check_alerts(combined_metrics, drift_results)

print("\n" + "="*80)
print("✅ PIPELINE MLOPS EXECUTADO COM SUCESSO")
print("="*80)
print(f"\n📊 Resumo:")
print(f"   - Modelo re-treinado: {version_id}")
print(f"   - Performance: AUC-ROC={new_metrics['auc_roc']:.4f}, F1={new_metrics['f1_score']:.4f}")
print(f"   - Drift detectado: {drift_results['n_drifted_features']}/{drift_results['n_features']} features")
print(f"   - Alertas gerados: {len(alerts)}")
print(f"   - Deploy recomendado: {'SIM ✅' if should_deploy_flag else 'NÃO ❌'}")

# COMMAND ----------

# DBTITLE 1,🎯 Próximos Passos
# MAGIC %md
# MAGIC ## 10. Próximos Passos
# MAGIC
# MAGIC Expansões e melhorias do pipeline MLOps.

# COMMAND ----------

# DBTITLE 1,🔄 Integração com Databricks Jobs
# MAGIC %md
# MAGIC ### Integração com Databricks Jobs
# MAGIC
# MAGIC #### Pipeline Automatizado
# MAGIC Criar um Job no Databricks para executar o pipeline periodicamente:
# MAGIC
# MAGIC **Estrutura do Job**:
# MAGIC 1. **Task 1: Data Validation**
# MAGIC    - Verificar qualidade dos dados
# MAGIC    - Validar schema
# MAGIC    - Detectar anomalias
# MAGIC
# MAGIC 2. **Task 2: Drift Detection**
# MAGIC    - Executar `detect_data_drift()`
# MAGIC    - Executar `detect_target_drift()`
# MAGIC    - Se drift > threshold: alertar e continuar para re-treino
# MAGIC
# MAGIC 3. **Task 3: Model Retraining** (condicional)
# MAGIC    - Carregar dados novos
# MAGIC    - Treinar novo modelo
# MAGIC    - Validar performance
# MAGIC
# MAGIC 4. **Task 4: Model Validation**
# MAGIC    - Executar `should_deploy()`
# MAGIC    - Se aprovado: promover para staging
# MAGIC
# MAGIC 5. **Task 5: Monitoring**
# MAGIC    - Registrar métricas
# MAGIC    - Verificar alertas
# MAGIC    - Enviar relatório
# MAGIC
# MAGIC **Schedules Recomendados**:
# MAGIC - Drift Detection: Diário às 2:00 AM
# MAGIC - Model Retraining: Semanal (Domingo 3:00 AM)
# MAGIC - Monitoring: A cada 6 horas
# MAGIC
# MAGIC **Alertas**:
# MAGIC - Slack: Alertas críticos
# MAGIC - Email: Relatórios diários/semanais
# MAGIC - PagerDuty: Alertas de produção

# COMMAND ----------

# DBTITLE 1,🧪 Framework de A/B Testing
# MAGIC %md
# MAGIC ### Framework de A/B Testing
# MAGIC
# MAGIC #### Estratégia de Teste
# MAGIC **Objetivo**: Validar novo modelo em produção antes de rollout completo.
# MAGIC
# MAGIC **Configuração**:
# MAGIC 1. **Grupos**:
# MAGIC    - Grupo A (controle): 90% do tráfego → modelo atual
# MAGIC    - Grupo B (tratamento): 10% do tráfego → novo modelo
# MAGIC
# MAGIC 2. **Métricas de Sucesso**:
# MAGIC    - **Primária**: AUC-ROC, F1-Score
# MAGIC    - **Secundária**: Precision, Recall, latência
# MAGIC    - **Negócio**: Taxa de inadimplência real, volume de crédito aprovado
# MAGIC
# MAGIC 3. **Critérios de Decisão**:
# MAGIC    - Mínimo 7 dias de teste
# MAGIC    - Mínimo 1,000 predições no grupo B
# MAGIC    - Significância estatística: p-value < 0.05
# MAGIC    - Melhoria mínima: +2% em métrica primária
# MAGIC
# MAGIC 4. **Rollout Gradual**:
# MAGIC    - Semana 1: 10% tráfego
# MAGIC    - Semana 2: 25% tráfego (se aprovado)
# MAGIC    - Semana 3: 50% tráfego
# MAGIC    - Semana 4: 100% tráfego
# MAGIC
# MAGIC **Implementação**:
# MAGIC ```python
# MAGIC # Exemplo de roteamento A/B
# MAGIC def route_to_model(customer_id, model_a, model_b, split=0.9):
# MAGIC     # Hash consistente para garantir mesmo cliente sempre vai para mesmo grupo
# MAGIC     hash_value = hash(customer_id) % 100
# MAGIC     
# MAGIC     if hash_value < split * 100:
# MAGIC         return model_a.predict(features)  # Grupo A
# MAGIC     else:
# MAGIC         return model_b.predict(features)  # Grupo B
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,🚀 Deploy em Model Serving
# MAGIC %md
# MAGIC ### Deploy em Databricks Model Serving
# MAGIC
# MAGIC #### Configuração do Endpoint
# MAGIC **Passos para deploy**:
# MAGIC
# MAGIC 1. **Registrar Modelo no Unity Catalog**:
# MAGIC ```python
# MAGIC import mlflow
# MAGIC
# MAGIC # Registrar modelo
# MAGIC model_uri = f"runs:/{run_id}/model"
# MAGIC mlflow.register_model(
# MAGIC     model_uri=model_uri,
# MAGIC     name=f"{CATALOG}.{SCHEMA_GOLD}.{MODEL_NAME}"
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC 2. **Criar Endpoint de Serving**:
# MAGIC - Workspace → Machine Learning → Serving
# MAGIC - New Serving Endpoint
# MAGIC - Model: credit_risk.gold.credit_risk_classifier
# MAGIC - Workload size: Small (para começar)
# MAGIC - Scale to zero: Enabled (custo-efetivo)
# MAGIC
# MAGIC 3. **Configurar Autoscaling**:
# MAGIC - Min instances: 0
# MAGIC - Max instances: 5
# MAGIC - Target requests per second: 100
# MAGIC
# MAGIC 4. **Habilitar Inference Tables**:
# MAGIC - Tabela de requests: `credit_risk.gold.serving_requests`
# MAGIC - Incluir: timestamp, input, output, latency
# MAGIC
# MAGIC #### Exemplo de Chamada
# MAGIC ```python
# MAGIC import requests
# MAGIC import os
# MAGIC
# MAGIC endpoint_url = f"{os.environ['DATABRICKS_HOST']}/serving-endpoints/credit_risk_classifier/invocations"
# MAGIC token = os.environ['DATABRICKS_TOKEN']
# MAGIC
# MAGIC headers = {
# MAGIC     'Authorization': f'Bearer {token}',
# MAGIC     'Content-Type': 'application/json'
# MAGIC }
# MAGIC
# MAGIC data = {
# MAGIC     'dataframe_records': [{
# MAGIC         'receita_total': 50000,
# MAGIC         'tx_inadimplencia': 0.05,
# MAGIC         'tempo_relacionamento_dias': 730,
# MAGIC         # ... outras features
# MAGIC     }]
# MAGIC }
# MAGIC
# MAGIC response = requests.post(endpoint_url, headers=headers, json=data)
# MAGIC prediction = response.json()
# MAGIC ```
# MAGIC
# MAGIC #### Monitoramento
# MAGIC - **Latência**: P50, P95, P99
# MAGIC - **Throughput**: Requests/segundo
# MAGIC - **Errors**: Taxa de erro 4xx/5xx
# MAGIC - **Cost**: Custo por 1000 predições
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **✅ Pipeline MLOps Completo Configurado**
# MAGIC
# MAGIC Este notebook fornece uma base sólida para:
# MAGIC - ✅ Re-treinamento automatizado
# MAGIC - ✅ Monitoramento de drift
# MAGIC - ✅ Sistema de alertas
# MAGIC - ✅ Governança e versionamento
# MAGIC - ✅ Path para produção via Model Serving
