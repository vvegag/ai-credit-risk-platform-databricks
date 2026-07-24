# Databricks notebook source
# DBTITLE 1,Monitoring & Drift Detection
# MAGIC %md
# MAGIC # 🔍 Monitoring & Drift Detection
# MAGIC
# MAGIC **Objetivo**: Monitorar saúde do modelo em produção e detectar drift (degradação de performance).
# MAGIC
# MAGIC ## Tipos de Drift
# MAGIC * **Data Drift**: Mudança na distribuição das features (input)
# MAGIC * **Concept Drift**: Mudança na relação entre features e target
# MAGIC * **Prediction Drift**: Mudança na distribuição das predições (output)
# MAGIC
# MAGIC ## Métricas Monitoradas
# MAGIC * Distribuição de features (média, desvio padrão)
# MAGIC * Performance do modelo (accuracy, precision, recall)
# MAGIC * Volume de predições por categoria
# MAGIC * Tempo de resposta
# MAGIC
# MAGIC ## Alertas
# MAGIC * ⚠️ Feature drift > 10%
# MAGIC * ⚠️ Performance drop > 5%
# MAGIC * ⚠️ Volume de risco crítico > 30%

# COMMAND ----------

# DBTITLE 1,1️⃣ Setup e Imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

print("✅ Bibliotecas carregadas")
print(f"🕒 Data/Hora da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# COMMAND ----------

# DBTITLE 1,2️⃣ Carregar Dados de Baseline (Treino)
# Carregar a feature store usada no TREINO (baseline). Sem .cache() — vira um
# PERSIST TABLE internamente, não suportado em compute serverless; como o
# Spark DataFrame só é lido uma vez aqui (toPandas), cache não traria ganho.
df_baseline_spark = spark.table(f"{CATALOG}.gold.features_ml")
df_baseline = df_baseline_spark.toPandas()

# Apenas colunas numéricas, removendo identificadores/categóricas
cols_to_exclude = ['id_cliente', 'cnpj', 'nome', 'categoria_rfm', 'perfil_comportamental']
feature_cols = [
    c for c in df_baseline.select_dtypes(include=[np.number]).columns if c not in cols_to_exclude
]
baseline_features = df_baseline[feature_cols].copy()

print(f"✅ Baseline carregado: {baseline_features.shape}")
print(f"📅 Features monitoradas: {len(feature_cols)}")

# Calcular estatísticas do baseline
baseline_stats = baseline_features.describe().T
baseline_stats['feature'] = baseline_stats.index
baseline_stats = baseline_stats[['feature', 'mean', 'std', 'min', 'max']]

print(f"\n📊 Estatísticas do Baseline:")
print(baseline_stats.head())

# COMMAND ----------

# DBTITLE 1,3️⃣ Simular Dados de Produção (Corrente)
# Em produção, esses dados viriam do streaming de inferência
# Aqui vamos simular um pequeno drift adicionando ruído

current_features = baseline_features.copy()

# Simular drift em algumas features (10% das amostras). Seed fixa (padrão do projeto,
# random_state=42 nos demais notebooks) — sem isso, a injeção de drift muda a cada
# execução e o resultado do teste KS deixa de ser reprodutível de uma run pra outra.
np.random.seed(42)
drift_mask = np.random.random(len(current_features)) < 0.1

# Adicionar drift em features específicas
if 'recency_dias' in current_features.columns:
    current_features.loc[drift_mask, 'recency_dias'] *= 1.5  # Aumento de 50%

if 'taxa_inadimplencia' in current_features.columns:
    current_features.loc[drift_mask, 'taxa_inadimplencia'] *= 1.3  # Aumento de 30%

if 'valor_medio_fatura' in current_features.columns:
    current_features.loc[drift_mask, 'valor_medio_fatura'] *= 0.8  # Redução de 20%

print(f"✅ Dados de produção simulados: {current_features.shape}")
print(f"⚠️ {drift_mask.sum()} amostras ({drift_mask.sum()/len(current_features)*100:.1f}%) com drift artificial injetado")

# COMMAND ----------

# DBTITLE 1,4️⃣ Detectar Data Drift (KS Test)
# Usar Kolmogorov-Smirnov test para detectar mudança de distribuição
def detect_drift_ks(baseline, current, threshold=0.05):
    """
    Detecta drift usando KS test
    Returns: p-value (quanto menor, maior o drift)
    """
    statistic, p_value = stats.ks_2samp(baseline, current)
    has_drift = p_value < threshold
    return {'statistic': statistic, 'p_value': p_value, 'has_drift': has_drift}

# Analisar cada feature
drift_results = []

for col in feature_cols:
    baseline_values = baseline_features[col].dropna()
    current_values = current_features[col].dropna()
    
    result = detect_drift_ks(baseline_values, current_values)
    
    # Calcular diferença percentual na média
    mean_diff_pct = ((current_values.mean() - baseline_values.mean()) / baseline_values.mean()) * 100
    
    drift_results.append({
        'feature': col,
        'ks_statistic': result['statistic'],
        'p_value': result['p_value'],
        'has_drift': result['has_drift'],
        'baseline_mean': baseline_values.mean(),
        'current_mean': current_values.mean(),
        'mean_diff_pct': mean_diff_pct
    })

df_drift = pd.DataFrame(drift_results).sort_values('ks_statistic', ascending=False)

print("\n" + "="*80)
print("🔍 DETECÇÃO DE DATA DRIFT")
print("="*80)
print(f"\n📊 Features com drift detectado (p < 0.05):")
features_com_drift = df_drift[df_drift['has_drift']]
if len(features_com_drift) > 0:
    print(features_com_drift[['feature', 'ks_statistic', 'p_value', 'mean_diff_pct']].to_string(index=False))
else:
    print("  Nenhum drift detectado! ✅")

print(f"\n⚠️ Total de features com drift: {len(features_com_drift)} / {len(feature_cols)}")
print("="*80)

# COMMAND ----------

# DBTITLE 1,5️⃣ Monitorar Performance do Modelo
# Carregar predições do modelo (tabela pequena, 1 linha/cliente — gerada por 04_modeling/01_modelo_classificacao_risco)
df_predictions = spark.table(f"{CATALOG}.gold.model_predictions").toPandas()

print("\n" + "="*80)
print("🎯 MONITORING DE PREDIÇÕES")
print("="*80)

print("\n📊 Distribuição de Probabilidades:")
print(f"  Média: {df_predictions['probabilidade_inadimplencia'].mean():.4f}")
print(f"  Mediana: {df_predictions['probabilidade_inadimplencia'].median():.4f}")
print(f"  Desvio Padrão: {df_predictions['probabilidade_inadimplencia'].std():.4f}")

print("\n📊 Distribuição de Perfis:")
print(df_predictions['perfil_comportamental'].value_counts())

print("\n📊 Distribuição de Predições (0=Adimplente, 1=Inadimplente):")
print(df_predictions['predicao_inadimplente'].value_counts())

# Alertas
risco_critico_pct = (df_predictions['perfil_comportamental'] == 'Alto Risco').sum() / len(df_predictions) * 100

print(f"\n⚠️ ALERTAS:")
if risco_critico_pct > 30:
    print(f"  ⚠️ ALERTA: {risco_critico_pct:.1f}% dos clientes estão em risco CRÍTICO (> 30%)")
else:
    print(f"  ✅ Volume de risco crítico sob controle: {risco_critico_pct:.1f}%")

print("="*80)

# COMMAND ----------

# DBTITLE 1,6️⃣ Health Check do Sistema
# Verificar integridade das tabelas
print("\n" + "="*80)
print("🏭 HEALTH CHECK DO SISTEMA")
print("="*80)

tabelas_monitoradas = [
    f'{CATALOG}.bronze.clientes',
    f'{CATALOG}.bronze.faturas',
    f'{CATALOG}.bronze.pagamentos',
    f'{CATALOG}.silver.faturas_enriquecidas',
    f'{CATALOG}.gold.features_ml',
    f'{CATALOG}.gold.model_predictions',
]

health_status = []

for tabela in tabelas_monitoradas:
    try:
        count = spark.table(tabela).count()
        last_update = spark.sql(f"DESCRIBE HISTORY {tabela} LIMIT 1").collect()[0]['timestamp']
        status = '✅ OK'
    except Exception as e:
        count = 0
        last_update = None
        status = f'❌ ERRO: {str(e)[:50]}'
    
    health_status.append({
        'tabela': tabela.split('.')[-1],
        'num_registros': count,
        'ultima_atualizacao': last_update,
        'status': status
    })

df_health = pd.DataFrame(health_status)

print("\n📊 Status das Tabelas:")
print(df_health.to_string(index=False))

print("\n✅ SISTEMA SAUDÁVEL" if all('✅' in s for s in df_health['status']) else "\n⚠️ ATENÇÃO: Algumas tabelas apresentam issues")
print("="*80)

# COMMAND ----------

# DBTITLE 1,7️⃣ Salvar Métricas de Monitoring
# Criar tabela de monitoring
monitoring_record = {
    'timestamp': datetime.now(),
    'num_features_com_drift': len(features_com_drift),
    'total_features': len(feature_cols),
    'percentual_drift': (len(features_com_drift) / len(feature_cols)) * 100,
    'risco_critico_pct': risco_critico_pct,
    'total_predicoes': len(df_predictions),
    'media_probabilidade': df_predictions['probabilidade_inadimplencia'].mean(),
    'alerta_drift': 'SIM' if len(features_com_drift) > 0 else 'NÃO',
    'alerta_risco': 'SIM' if risco_critico_pct > 30 else 'NÃO',
    'status_geral': 'CRÍTICO' if (len(features_com_drift) > 3 or risco_critico_pct > 30) else 'OK'
}

df_monitoring = pd.DataFrame([monitoring_record])

# Salvar (append incremental; overwrite só na primeira execução, quando a tabela ainda não existe)
spark_df_monitoring = spark.createDataFrame(df_monitoring)
table_exists = spark.catalog.tableExists(f"{CATALOG}.gold.monitoring_logs")
spark_df_monitoring.write.mode("append" if table_exists else "overwrite") \
    .saveAsTable(f"{CATALOG}.gold.monitoring_logs")
print(f"✅ Métricas salvas: {CATALOG}.gold.monitoring_logs")

print("\n📊 Último registro de monitoring:")
print(df_monitoring.T)

# COMMAND ----------

# DBTITLE 1,8️⃣ Dashboard Executivo
print("\n" + "="*80)
print("📊 DASHBOARD EXECUTIVO - MONITORING")
print("="*80)

print(f"\n🕒 Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n🔍 DATA DRIFT:")
print(f"  Features com drift: {len(features_com_drift)} / {len(feature_cols)} ({(len(features_com_drift)/len(feature_cols))*100:.1f}%)")
if len(features_com_drift) > 0:
    print(f"  Top 3 features com maior drift:")
    for _, row in features_com_drift.head(3).iterrows():
        print(f"    - {row['feature']}: {row['mean_diff_pct']:+.1f}% (p={row['p_value']:.4f})")

print("\n🎯 PREDIÇÕES:")
print(f"  Total: {len(df_predictions)}")
print(f"  Risco Crítico: {(df_predictions['categoria_risco'] == 'Crítico').sum()} ({risco_critico_pct:.1f}%)")
print(f"  Probabilidade média: {df_predictions['probabilidade_inadimplencia'].mean():.2%}")

print("\n⚠️ ALERTAS ATIVOS:")
alertas = []
if len(features_com_drift) > 3:
    alertas.append(f"  ⚠️ Drift detectado em {len(features_com_drift)} features (> 3)")
if risco_critico_pct > 30:
    alertas.append(f"  ⚠️ Risco crítico elevado: {risco_critico_pct:.1f}% (> 30%)")

if alertas:
    for alerta in alertas:
        print(alerta)
    print("\n🔴 STATUS: CRÍTICO - AÇÃO NECESSÁRIA")
    print("  Recomendações:")
    print("    1. Retreinar modelo com dados recentes")
    print("    2. Revisar estratégia de cobrança")
    print("    3. Investigar mudanças no perfil de clientes")
else:
    print("  ✅ Nenhum alerta crítico")
    print("\n🟢 STATUS: SISTEMA OPERANDO NORMALMENTE")

print("\n" + "="*80)

# COMMAND ----------



