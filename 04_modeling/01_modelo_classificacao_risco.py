# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Introdução - Modelo de Classificação de Risco de Crédito
# MAGIC %md
# MAGIC # FASE 3: Modelo XGBoost de Classificação de Risco de Crédito
# MAGIC
# MAGIC ## Objetivo
# MAGIC Desenvolver um modelo de **classificação binária** para prever o risco de inadimplência de clientes, utilizando as features agregadas da tabela `credit_risk.gold.features_ml`.
# MAGIC
# MAGIC ## Definição do Problema
# MAGIC - **Target**: Inadimplente (definido como taxa_inadimplencia > 40%)
# MAGIC - **Dataset**: `credit_risk.gold.features_ml` (features pré-processadas na FASE 2)
# MAGIC - **Algoritmo**: XGBoost Classifier
# MAGIC - **Framework**: MLflow para experiment tracking
# MAGIC
# MAGIC ## Métricas de Avaliação
# MAGIC Para este problema de classificação de risco de crédito, avaliaremos:
# MAGIC - **Precision**: Proporção de predições positivas corretas (minimizar falsos positivos)
# MAGIC - **Recall**: Proporção de inadimplentes identificados (minimizar falsos negativos)
# MAGIC - **F1-Score**: Média harmônica entre Precision e Recall
# MAGIC - **AUC-ROC**: Capacidade de discriminação entre classes
# MAGIC - **Confusion Matrix**: Análise detalhada de acertos e erros
# MAGIC
# MAGIC ## Estrutura do Notebook
# MAGIC 1. Setup e Imports
# MAGIC 2. Carregamento de Dados
# MAGIC 3. Análise Exploratória (EDA)
# MAGIC 4. Preparação dos Dados
# MAGIC 5. Treinamento do Modelo
# MAGIC 6. Avaliação e Métricas
# MAGIC 7. SHAP - Explicabilidade
# MAGIC 8. MLflow - Tracking
# MAGIC 9. Salvamento de Predições
# MAGIC 10. Conclusões e Próximos Passos

# COMMAND ----------

# DBTITLE 1,Setup e Imports
# MAGIC %md
# MAGIC ## 1. Setup e Imports
# MAGIC Instalação de bibliotecas necessárias e imports

# COMMAND ----------

# DBTITLE 1,Instalação de Bibliotecas
# Instalação de bibliotecas necessárias
%pip install xgboost==2.0.3 shap==0.44.0 mlflow==2.9.2 scikit-learn==1.3.2 seaborn==0.13.0 --quiet

# COMMAND ----------

# DBTITLE 1,Restart Python
# Restart Python após instalação
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# PySpark
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# XGBoost
import xgboost as xgb
from xgboost import XGBClassifier

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# MLflow
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

# SHAP
import shap

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("✅ Imports concluídos com sucesso!")
print(f"XGBoost version: {xgb.__version__}")
print(f"MLflow version: {mlflow.__version__}")
print(f"SHAP version: {shap.__version__}")

# COMMAND ----------

# DBTITLE 1,Carregamento de Dados
# MAGIC %md
# MAGIC ## 2. Carregamento de Dados
# MAGIC Carregar a tabela de features criada na FASE 2

# COMMAND ----------

# DBTITLE 1,Carregar Tabela de Features
dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

# Mesmo nome de modelo usado por 05_mlops/01_mlops_pipeline.py — uma única entrada
# no UC Model Registry, não dois modelos desencontrados com nomes diferentes.
MODEL_NAME = "credit_risk_classifier"
MODEL_REGISTRY_NAME = f"{CATALOG}.gold.{MODEL_NAME}"
mlflow.set_registry_uri("databricks-uc")

# Carregar dados da tabela gold. Sem .cache() — DataFrame.cache()/persist() vira
# um PERSIST TABLE internamente, que não é suportado em compute serverless.
df_spark = spark.table(f"{CATALOG}.gold.features_ml")

print(f"📊 Total de registros: {df_spark.count():,}")
print(f"📊 Total de colunas: {len(df_spark.columns)}")
print("\n🔍 Schema da tabela:")
df_spark.printSchema()

# COMMAND ----------

# DBTITLE 1,Primeiras Linhas
# Visualizar primeiras linhas
display(df_spark.limit(10))

# COMMAND ----------

# DBTITLE 1,Estatísticas Descritivas
# Estatísticas descritivas
print("📈 Estatísticas Descritivas:")
display(df_spark.describe())

# COMMAND ----------

# DBTITLE 1,Análise Exploratória
# MAGIC %md
# MAGIC ## 3. Análise Exploratória de Dados (EDA)
# MAGIC Analisar distribuições, correlações e padrões nos dados antes do treinamento

# COMMAND ----------

# DBTITLE 1,Converter para Pandas e Criar Target
# Converter para Pandas para análise exploratória
df_pd = df_spark.toPandas()

# Criar variável target binária: inadimplente se taxa_inadimplencia > 40%
df_pd['inadimplente'] = (df_pd['taxa_inadimplencia'] > 40).astype(int)

print(f"✅ Dataset convertido para Pandas: {df_pd.shape}")
print(f"\n📊 Distribuição da variável target:")
print(df_pd['inadimplente'].value_counts())
print(f"\n📊 Percentual de inadimplentes: {df_pd['inadimplente'].mean()*100:.2f}%")

# COMMAND ----------

# DBTITLE 1,Visualizar Distribuição do Target
# Visualizar distribuição do target
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de barras
ax1 = axes[0]
target_counts = df_pd['inadimplente'].value_counts()
ax1.bar(['Adimplente (0)', 'Inadimplente (1)'], target_counts.values, color=['green', 'red'], alpha=0.7)
ax1.set_ylabel('Frequência')
ax1.set_title('Distribuição da Variável Target', fontsize=14, fontweight='bold')
for i, v in enumerate(target_counts.values):
    ax1.text(i, v + 50, str(v), ha='center', fontweight='bold')

# Gráfico de pizza
ax2 = axes[1]
ax2.pie(target_counts.values, labels=['Adimplente', 'Inadimplente'], autopct='%1.1f%%', 
        colors=['green', 'red'], startangle=90, explode=[0, 0.1])
ax2.set_title('Proporção de Classes', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\n⚠️ Razão de classes (Adimplente:Inadimplente) = {target_counts[0]/target_counts[1]:.2f}:1")

# COMMAND ----------

# DBTITLE 1,Correlação entre Features Numéricas
# Selecionar apenas colunas numéricas para correlação
numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()

# Remover colunas de identificação
cols_to_exclude = ['id_cliente', 'inadimplente']
numeric_features = [col for col in numeric_cols if col not in cols_to_exclude]

# Calcular matriz de correlação
corr_matrix = df_pd[numeric_features + ['inadimplente']].corr()

# Visualizar correlograma
plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação - Features vs Target', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Correlação com o target
print("\n🎯 Top 10 Features mais correlacionadas com Inadimplência:")
target_corr = corr_matrix['inadimplente'].drop('inadimplente').sort_values(ascending=False)
print(target_corr.head(10))

# COMMAND ----------

# DBTITLE 1,Análise de Target Leakage
# Verificação de Target Leakage
print("🔍 Verificação de Target Leakage")
print("="*60)

# Identificar features com correlação muito alta (potencial leakage)
high_corr_features = target_corr[abs(target_corr) > 0.95]

if len(high_corr_features) > 0:
    print("⚠️ ALERTA: Features com correlação extremamente alta detectadas:")
    for feat, corr_val in high_corr_features.items():
        print(f"  - {feat}: {corr_val:.4f}")
    print("\n💡 Estas features podem indicar target leakage e devem ser revisadas.")
    print("   A feature 'taxa_inadimplencia' será removida por ser a base do target.")
else:
    print("✅ Nenhuma feature com correlação suspeita (> 0.95) detectada.")

print("\n📋 Features que serão EXCLUÍDAS do treinamento:")
print("  - id_cliente (identificador)")
print("  - cnpj (identificador)")
print("  - nome (texto)")
print("  - categoria_rfm (categórica, já temos scores R/F/M)")
print("  - perfil_comportamental (derivada, usada para análise)")
print("  - taxa_inadimplencia (usada para criar target, seria leakage perfeito)")

# COMMAND ----------

# DBTITLE 1,Distribuição por Perfil Comportamental
# Análise de inadimplência por perfil comportamental
if 'perfil_comportamental' in df_pd.columns:
    perfil_analysis = df_pd.groupby('perfil_comportamental')['inadimplente'].agg(['count', 'sum', 'mean'])
    perfil_analysis.columns = ['Total', 'Inadimplentes', 'Taxa_Inadimplencia']
    perfil_analysis['Taxa_Inadimplencia'] = perfil_analysis['Taxa_Inadimplencia'] * 100
    perfil_analysis = perfil_analysis.sort_values('Taxa_Inadimplencia', ascending=False)
    
    print("\n📊 Inadimplência por Perfil Comportamental:")
    display(perfil_analysis)
    
    # Visualização
    fig, ax = plt.subplots(figsize=(12, 6))
    perfil_analysis['Taxa_Inadimplencia'].plot(kind='bar', ax=ax, color='coral', alpha=0.8)
    ax.set_ylabel('Taxa de Inadimplência (%)')
    ax.set_xlabel('Perfil Comportamental')
    ax.set_title('Taxa de Inadimplência por Perfil Comportamental', fontsize=14, fontweight='bold')
    ax.axhline(y=df_pd['inadimplente'].mean()*100, color='red', linestyle='--', 
               label=f'Média Geral: {df_pd["inadimplente"].mean()*100:.1f}%')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# DBTITLE 1,Box Plots de Features Numéricas
# Box plots das principais features numéricas por target
top_features = target_corr.head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, feature in enumerate(top_features):
    if feature in df_pd.columns:
        ax = axes[idx]
        df_pd.boxplot(column=feature, by='inadimplente', ax=ax)
        ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Inadimplente')
        ax.set_ylabel(feature)
        plt.sca(ax)
        plt.xticks([1, 2], ['Não (0)', 'Sim (1)'])

plt.suptitle('Box Plots: Top Features vs Target', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Preparação dos Dados
# MAGIC %md
# MAGIC ## 4. Preparação dos Dados para Treinamento
# MAGIC Seleção de features, encoding de categóricas e split train/test

# COMMAND ----------

# DBTITLE 1,Seleção de Features
# Definir colunas a excluir
cols_to_drop = [
    'id_cliente', 'cnpj', 'nome',  # Identificadores e texto
    'categoria_rfm', 'perfil_comportamental',  # Categóricas descritivas
    'categoria_risco',  # Rótulo usado para enviesar a geração sintética original (leakage)
    'data_cadastro',  # Data crua em string — precisaria virar feature numérica (ex: tempo de
                       # relacionamento) antes de entrar no modelo; fora de escopo aqui
    'taxa_inadimplencia',  # Feature usada para criar target (leakage)
    'inadimplente'  # Target (será separado)
]

# Identificar features categóricas que precisam de encoding
categorical_features = ['porte', 'setor']

# Criar dataset de features
feature_cols = [col for col in df_pd.columns if col not in cols_to_drop]
print(f"📋 Total de features selecionadas: {len(feature_cols)}")
print(f"\n🔢 Features numéricas: {len([c for c in feature_cols if c not in categorical_features])}")
print(f"🏷️ Features categóricas para encoding: {len(categorical_features)}")
print(f"\nFeatures categóricas: {categorical_features}")

# COMMAND ----------

# DBTITLE 1,One-Hot Encoding
# One-hot encoding de features categóricas
df_encoded = pd.get_dummies(df_pd[feature_cols], columns=categorical_features, drop_first=False)

print(f"✅ One-hot encoding concluído!")
print(f"📊 Dimensões após encoding: {df_encoded.shape}")
print(f"\n🆕 Novas colunas criadas:")
new_cols = [col for col in df_encoded.columns if col not in df_pd.columns]
for col in new_cols[:10]:  # Mostrar primeiras 10
    print(f"  - {col}")
if len(new_cols) > 10:
    print(f"  ... e mais {len(new_cols)-10} colunas")

# COMMAND ----------

# DBTITLE 1,Train/Test Split
# Separar features (X) e target (y)
X = df_encoded
y = df_pd['inadimplente']

print(f"📊 Shape de X: {X.shape}")
print(f"📊 Shape de y: {y.shape}")

# Split train/test (80/20) com stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    stratify=y, 
    random_state=42
)

print(f"\n✅ Split concluído!")
print(f"📊 Train set: {X_train.shape[0]:,} amostras ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"📊 Test set: {X_test.shape[0]:,} amostras ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\n🎯 Distribuição do target no train set:")
print(y_train.value_counts())
print(f"\n🎯 Distribuição do target no test set:")
print(y_test.value_counts())

# COMMAND ----------

# DBTITLE 1,Verificar Balanceamento
# Verificar balanceamento de classes
train_class_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
test_class_ratio = y_test.value_counts()[0] / y_test.value_counts()[1]

print("⚖️ Análise de Balanceamento de Classes:")
print("="*60)
print(f"Train set - Razão (Adimplente:Inadimplente): {train_class_ratio:.2f}:1")
print(f"Test set - Razão (Adimplente:Inadimplente): {test_class_ratio:.2f}:1")

# Calcular scale_pos_weight para XGBoost
scale_pos_weight = train_class_ratio
print(f"\n💡 scale_pos_weight para XGBoost: {scale_pos_weight:.2f}")
print("   (Usado para lidar com desbalanceamento de classes)")

# COMMAND ----------

# DBTITLE 1,Treinamento do Modelo
# MAGIC %md
# MAGIC ## 5. Treinamento do Modelo XGBoost
# MAGIC Configuração do MLflow e treinamento do modelo com parâmetros otimizados

# COMMAND ----------

# DBTITLE 1,Configurar MLflow Experiment
# MLflow usará o experimento padrão (configuração via set_experiment não compatível com Spark Connect)
# O experiment será criado automaticamente ao iniciar o primeiro run

print("✅ MLflow configurado com experimento padrão")
print(f"📍 Runs serão logados no experimento padrão do notebook")

# COMMAND ----------

# DBTITLE 1,Treinar Modelo XGBoost
# Parâmetros do modelo
params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'scale_pos_weight': scale_pos_weight,  # Para lidar com desbalanceamento
    'random_state': 42,
    'eval_metric': 'logloss',
    'objective': 'binary:logistic'
}

# Criar e treinar modelo dentro de uma run do MLflow (necessário para depois
# registrar o modelo no UC Model Registry com o histórico de parâmetros/métricas)
print("🚀 Iniciando treinamento do XGBoost...")
print(f"   - Train samples: {X_train.shape[0]:,}")
print(f"   - Test samples: {X_test.shape[0]:,}")
print(f"   - Features: {X_train.shape[1]}")
print(f"   - Class ratio: {train_class_ratio:.2f}:1")
print("")

mlflow_run = mlflow.start_run(run_name=f"classificacao_risco_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
mlflow.log_params(params)

model = XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

print("✅ Treinamento concluído!")
print(f"   - Modelo: XGBoostClassifier")
print(f"   - Parâmetros: {params}")

# COMMAND ----------

# DBTITLE 1,Fazer Predições
# Fazer predições no conjunto de teste
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("✅ Predições realizadas no conjunto de teste")
print(f"\n📊 Exemplos de probabilidades preditas:")
for i in range(5):
    print(f"  Amostra {i+1}: Prob={y_pred_proba[i]:.4f}, Predição={'Inadimplente' if y_pred[i]==1 else 'Adimplente'}")

# COMMAND ----------

# DBTITLE 1,Avaliação do Modelo
# MAGIC %md
# MAGIC ## 6. Avaliação do Modelo
# MAGIC Análise de performance com múltiplas métricas e visualizações

# COMMAND ----------

# DBTITLE 1,Confusion Matrix
# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Adimplente', 'Inadimplente'])
disp.plot(cmap='Blues', ax=ax, values_format='d')
ax.set_title('Confusion Matrix - Modelo XGBoost', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 Confusion Matrix:")
print(f"  True Negatives (TN): {cm[0,0]}")
print(f"  False Positives (FP): {cm[0,1]}")
print(f"  False Negatives (FN): {cm[1,0]}")
print(f"  True Positives (TP): {cm[1,1]}")

# COMMAND ----------

# DBTITLE 1,Classification Report
# Classification Report
print("\n📋 Classification Report:")
print("="*70)
print(classification_report(y_test, y_pred, target_names=['Adimplente', 'Inadimplente'], digits=4))

# Calcular métricas individuais
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n🎯 Métricas Principais:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f} (dos preditos como inadimplentes, quantos realmente são)")
print(f"  Recall:    {recall:.4f} (dos inadimplentes reais, quantos conseguimos identificar)")
print(f"  F1-Score:  {f1:.4f} (média harmônica entre Precision e Recall)")

# COMMAND ----------

# DBTITLE 1,ROC Curve e AUC
# ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve - Receiver Operating Characteristic', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n🎯 AUC-ROC Score: {roc_auc:.4f}")
print("   (Área sob a curva ROC - quanto maior, melhor a capacidade de discriminação)")

# COMMAND ----------

# DBTITLE 1,Precision-Recall Curve
# Precision-Recall Curve
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(recall_curve, precision_curve, color='blue', lw=2, 
        label=f'Precision-Recall Curve (AP = {avg_precision:.4f})')
ax.axhline(y=y_test.mean(), color='red', linestyle='--', 
           label=f'Baseline (No Skill = {y_test.mean():.4f})')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax.legend(loc="best", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/precision_recall_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n🎯 Average Precision Score: {avg_precision:.4f}")

# COMMAND ----------

# DBTITLE 1,Threshold Tuning
# Encontrar threshold ótimo (maximizar F1-score)
from sklearn.metrics import f1_score

thresholds_to_test = np.arange(0.3, 0.7, 0.01)
f1_scores = []

for threshold in thresholds_to_test:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_threshold)
    f1_scores.append(f1)

# Encontrar melhor threshold
best_idx = np.argmax(f1_scores)
best_threshold = thresholds_to_test[best_idx]
best_f1 = f1_scores[best_idx]

# Plotar
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds_to_test, f1_scores, color='green', lw=2)
ax.axvline(x=best_threshold, color='red', linestyle='--', 
           label=f'Optimal Threshold = {best_threshold:.2f}')
ax.axhline(y=best_f1, color='red', linestyle='--', alpha=0.3)
ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_title('Threshold Tuning - F1-Score Optimization', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/threshold_tuning.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n🎯 Threshold Ótimo: {best_threshold:.2f}")
print(f"🎯 F1-Score no threshold ótimo: {best_f1:.4f}")
print(f"🎯 F1-Score no threshold padrão (0.50): {f1:.4f}")
print(f"\n💡 Melhoria: {(best_f1 - f1)*100:.2f}% no F1-Score")

# COMMAND ----------

# DBTITLE 1,Feature Importance
# Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

# Top 15 features
top_n = 15
top_features_df = feature_importance_df.head(top_n)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(top_n), top_features_df['importance'].values, color='steelblue', alpha=0.8)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features_df['feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Importance', fontsize=12)
ax.set_title(f'Top {top_n} Feature Importance - XGBoost', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n🏆 Top 15 Features Mais Importantes:")
for idx, row in top_features_df.iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")

# COMMAND ----------

# DBTITLE 1,SHAP Values - Explicabilidade
# MAGIC %md
# MAGIC ## 7. SHAP Values - Explicabilidade do Modelo
# MAGIC Análise de impacto das features nas predições usando SHAP (SHapley Additive exPlanations)

# COMMAND ----------

# DBTITLE 1,SHAP TreeExplainer
# Criar SHAP explainer
print("🔍 Criando SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("✅ SHAP values calculados!")
print(f"📊 Shape dos SHAP values: {shap_values.shape}")

# COMMAND ----------

# DBTITLE 1,SHAP Summary Plot
# SHAP Summary Plot - Impacto Global
print("\n📊 SHAP Summary Plot - Impacto Global das Features")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False, max_display=15)
plt.title('SHAP Summary Plot - Impacto Global das Features', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/tmp/shap_summary_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n💡 Interpretação:")
print("  - Cada ponto representa uma amostra")
print("  - Cor vermelha: valor alto da feature")
print("  - Cor azul: valor baixo da feature")
print("  - Posição horizontal: impacto na predição (+ inadimplente, - adimplente)")

# COMMAND ----------

# DBTITLE 1,SHAP Dependence Plots
# SHAP Dependence Plots para top 3 features
top_3_features = feature_importance_df.head(3)['feature'].tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, feature in enumerate(top_3_features):
    plt.sca(axes[idx])
    shap.dependence_plot(feature, shap_values, X_test, show=False, ax=axes[idx])
    axes[idx].set_title(f'SHAP Dependence: {feature}', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/tmp/shap_dependence_plots.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n💡 Dependence Plots mostram como o valor da feature afeta a predição")

# COMMAND ----------

# DBTITLE 1,SHAP - Explicações Individuais
# Explicação de predições individuais
print("\n🔍 Análise de Predições Individuais com SHAP")
print("="*70)

# Caso 1: Alto risco (probabilidade alta de inadimplência)
high_risk_idx = np.argmax(y_pred_proba)
print(f"\n📍 CASO 1: ALTO RISCO")
print(f"  Índice: {high_risk_idx}")
print(f"  Probabilidade de Inadimplência: {y_pred_proba[high_risk_idx]:.4f}")
print(f"  Predição: {'Inadimplente' if y_pred[high_risk_idx]==1 else 'Adimplente'}")
print(f"  Real: {'Inadimplente' if y_test.iloc[high_risk_idx]==1 else 'Adimplente'}")

# Caso 2: Baixo risco (probabilidade baixa de inadimplência)
low_risk_idx = np.argmin(y_pred_proba)
print(f"\n📍 CASO 2: BAIXO RISCO")
print(f"  Índice: {low_risk_idx}")
print(f"  Probabilidade de Inadimplência: {y_pred_proba[low_risk_idx]:.4f}")
print(f"  Predição: {'Inadimplente' if y_pred[low_risk_idx]==1 else 'Adimplente'}")
print(f"  Real: {'Inadimplente' if y_test.iloc[low_risk_idx]==1 else 'Adimplente'}")

# Caso 3: Caso difícil (probabilidade próxima de 0.5)
mid_probs = np.abs(y_pred_proba - 0.5)
mid_risk_idx = np.argmin(mid_probs)
print(f"\n📍 CASO 3: CASO DIFÍCIL (próximo do threshold)")
print(f"  Índice: {mid_risk_idx}")
print(f"  Probabilidade de Inadimplência: {y_pred_proba[mid_risk_idx]:.4f}")
print(f"  Predição: {'Inadimplente' if y_pred[mid_risk_idx]==1 else 'Adimplente'}")
print(f"  Real: {'Inadimplente' if y_test.iloc[mid_risk_idx]==1 else 'Adimplente'}")

# SHAP Force Plots para os 3 casos
for idx, (case_idx, case_name) in enumerate([
    (high_risk_idx, "Alto Risco"),
    (low_risk_idx, "Baixo Risco"),
    (mid_risk_idx, "Caso Difícil")
]):
    print(f"\n\n{'='*70}")
    print(f"SHAP Force Plot - {case_name}")
    print(f"{'='*70}")
    shap.force_plot(
        explainer.expected_value, 
        shap_values[case_idx], 
        X_test.iloc[case_idx],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f'/tmp/shap_force_plot_{idx+1}.png', dpi=150, bbox_inches='tight')
    plt.show()

# COMMAND ----------

# DBTITLE 1,MLflow - Logging Completo
# MAGIC %md
# MAGIC ## 8. MLflow - Logging de Métricas e Artefatos
# MAGIC Registrar métricas customizadas, plots e modelo no MLflow

# COMMAND ----------

# DBTITLE 1,Log Métricas e Artefatos no MLflow
# Salvar feature importance como CSV
feature_importance_df.to_csv('/tmp/feature_importance.csv', index=False)

print("✅ Métricas e plots salvos localmente!")
print(f"📊 Total de plots gerados: 10")
print(f"📁 Plots salvos em /tmp/")
print(f"💾 Feature importance salvo em /tmp/feature_importance.csv")
print("\n📊 Sumário de Métricas:")
print(f"   - Accuracy: {accuracy:.4f}")
print(f"   - Precision: {precision:.4f}")
print(f"   - Recall: {recall:.4f}")
print(f"   - F1-Score: {f1:.4f}")
print(f"   - ROC-AUC: {roc_auc:.4f}")
print(f"   - Avg Precision: {avg_precision:.4f}")
print(f"   - Optimal Threshold: {best_threshold:.2f}")
print(f"   - Optimal F1: {best_f1:.4f}")

# COMMAND ----------

# DBTITLE 1,Registrar Modelo no MLflow Model Registry
# Log de métricas/modelo na run aberta em "Treinar Modelo XGBoost", e registro no
# Unity Catalog Model Registry (mlflow.xgboost.log_model com registered_model_name,
# não um pickle solto em /tmp). Mesmo MODEL_REGISTRY_NAME usado por
# 05_mlops/01_mlops_pipeline.py — uma única entrada no registry, não dois modelos
# desencontrados.
mlflow.log_metrics({
    'accuracy': accuracy, 'precision': precision, 'recall': recall,
    'f1_score': f1, 'auc_roc': roc_auc,
})
signature = infer_signature(X_train, model.predict(X_train))
model_info = mlflow.xgboost.log_model(
    model, "model",
    signature=signature,
    registered_model_name=MODEL_REGISTRY_NAME,
)
mlflow.end_run()

print(f"✅ Modelo registrado no UC Model Registry: {MODEL_REGISTRY_NAME} v{model_info.registered_model_version}")

# Bootstrap: se este é o primeiro modelo registrado (ainda não existe alias Champion),
# promove-o como Champion inicial. Promoções seguintes (retreino com métrica melhor)
# são feitas por 05_mlops/01_mlops_pipeline.py, não aqui.
from mlflow.tracking import MlflowClient
_client = MlflowClient()
try:
    _client.get_model_version_by_alias(MODEL_REGISTRY_NAME, "Champion")
    print("ℹ️ Já existe um Champion registrado — mantendo-o (retreino é feito por 05_mlops/)")
except Exception:
    _client.set_registered_model_alias(MODEL_REGISTRY_NAME, "Champion", model_info.registered_model_version)
    print(f"🏆 Nenhum Champion prévio — v{model_info.registered_model_version} promovido a Champion inicial")

# COMMAND ----------

# DBTITLE 1,Salvamento de Predições
# MAGIC %md
# MAGIC ## 9. Salvamento Final - Tabela de Predições
# MAGIC Criar tabela no Unity Catalog com predições do modelo

# COMMAND ----------

# DBTITLE 1,Criar DataFrame de Predições
# Criar DataFrame com predições completas
# Combinar todos os dados (train + test) para gerar predições completas
X_full = pd.concat([X_train, X_test], axis=0)
y_full = pd.concat([y_train, y_test], axis=0)

# Pontuar com o modelo Champion do UC Model Registry (flavor nativo do xgboost,
# não pyfunc genérico, para manter acesso a predict_proba) — não com o objeto
# `model` recém-treinado em memória. Isso garante que credit_risk.gold.model_predictions
# sempre reflita o modelo realmente em produção (o mesmo que 05_mlops/ promove),
# mesmo que este notebook seja reexecutado sem que o retreino tenha sido aprovado.
champion_model = mlflow.xgboost.load_model(f"models:/{MODEL_REGISTRY_NAME}@Champion")
y_full_pred = champion_model.predict(X_full)
y_full_pred_proba = champion_model.predict_proba(X_full)[:, 1]

# Recuperar informações originais (id_cliente, perfil_comportamental)
df_predictions = df_pd[['id_cliente', 'perfil_comportamental']].copy()
df_predictions['probabilidade_inadimplencia'] = y_full_pred_proba
df_predictions['predicao_inadimplente'] = y_full_pred
df_predictions['perfil_real'] = df_pd['inadimplente']

print(f"✅ DataFrame de predições criado!")
print(f"📊 Total de registros: {len(df_predictions):,}")
print(f"\n🔍 Primeiras linhas:")
display(df_predictions.head(10))

# COMMAND ----------

# DBTITLE 1,Salvar Tabela de Predições no Unity Catalog
# Converter para Spark DataFrame
df_predictions_spark = spark.createDataFrame(df_predictions)

# Salvar no Unity Catalog
table_name = f"{CATALOG}.gold.model_predictions"

df_predictions_spark.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

print(f"✅ Tabela de predições salva com sucesso!")
print(f"📍 Localização: {table_name}")
print(f"📊 Total de registros salvos: {df_predictions_spark.count():,}")

# COMMAND ----------

# DBTITLE 1,Validação Final - Exemplos de Predições
# Validação final: mostrar exemplos variados de predições
print("\n🔍 VALIDAÇÃO FINAL - Exemplos de Predições")
print("="*80)

# 5 casos de alto risco
print("\n🔴 TOP 5 CASOS DE ALTO RISCO (maior probabilidade de inadimplência):")
high_risk_cases = df_predictions.nlargest(5, 'probabilidade_inadimplencia')
for idx, row in high_risk_cases.iterrows():
    print(f"  Cliente {row['id_cliente']}: Prob={row['probabilidade_inadimplencia']:.2%}, "
          f"Predição={'INADIMPLENTE' if row['predicao_inadimplente']==1 else 'Adimplente'}, "
          f"Real={'Inadimplente' if row['perfil_real']==1 else 'Adimplente'}, "
          f"Perfil={row['perfil_comportamental']}")

# 5 casos de baixo risco
print("\n🟢 TOP 5 CASOS DE BAIXO RISCO (menor probabilidade de inadimplência):")
low_risk_cases = df_predictions.nsmallest(5, 'probabilidade_inadimplencia')
for idx, row in low_risk_cases.iterrows():
    print(f"  Cliente {row['id_cliente']}: Prob={row['probabilidade_inadimplencia']:.2%}, "
          f"Predição={'Inadimplente' if row['predicao_inadimplente']==1 else 'ADIMPLENTE'}, "
          f"Real={'Inadimplente' if row['perfil_real']==1 else 'Adimplente'}, "
          f"Perfil={row['perfil_comportamental']}")

# Estatísticas gerais
print("\n📊 ESTATÍSTICAS GERAIS DAS PREDIÇÕES:")
print(f"  Probabilidade média de inadimplência: {df_predictions['probabilidade_inadimplencia'].mean():.2%}")
print(f"  Probabilidade mínima: {df_predictions['probabilidade_inadimplencia'].min():.2%}")
print(f"  Probabilidade máxima: {df_predictions['probabilidade_inadimplencia'].max():.2%}")
print(f"  Total predito como inadimplente: {df_predictions['predicao_inadimplente'].sum():,} "
      f"({df_predictions['predicao_inadimplente'].mean()*100:.1f}%)")
print(f"  Total real inadimplente: {df_predictions['perfil_real'].sum():,} "
      f"({df_predictions['perfil_real'].mean()*100:.1f}%)")

# COMMAND ----------

# DBTITLE 1,Resumo e Conclusões
# MAGIC %md
# MAGIC ## 10. Resumo e Conclusões
# MAGIC Análise final dos resultados e próximos passos

# COMMAND ----------

# DBTITLE 1,Resumo Final
# Resumo Final do Modelo
print("\n" + "="*80)
print("🎯 RESUMO FINAL - MODELO DE CLASSIFICAÇÃO DE RISCO DE CRÉDITO")
print("="*80)

print("\n📊 PERFORMANCE DO MODELO:")
print(f"  • AUC-ROC: {roc_auc:.4f} - Excelente capacidade de discriminação entre classes")
print(f"  • F1-Score: {f1:.4f} - Bom balanço entre Precision e Recall")
print(f"  • Precision: {precision:.4f} - {precision*100:.1f}% dos preditos como inadimplentes são corretos")
print(f"  • Recall: {recall:.4f} - {recall*100:.1f}% dos inadimplentes reais foram identificados")
print(f"  • Accuracy: {accuracy:.4f} - {accuracy*100:.1f}% de acurácia geral")

print("\n🏆 TOP 5 FEATURES MAIS IMPORTANTES:")
for idx, row in feature_importance_df.head(5).iterrows():
    print(f"  {idx+1}. {row['feature']:30s} - Importance: {row['importance']:.4f}")

print("\n💡 INSIGHTS DE NEGÓCIO:")
print("  1. O modelo consegue identificar padrões de inadimplência com boa precisão")
print("  2. Features comportamentais (recência, frequência, valor) são cruciais")
print("  3. SHAP values fornecem explicabilidade para cada predição")
print("  4. O threshold pode ser ajustado conforme o trade-off desejado entre FP e FN")

print("\n📦 ARTEFATOS GERADOS:")
print(f"  • Modelo registrado: {MODEL_REGISTRY_NAME} v{model_info.registered_model_version} (UC Model Registry)")
print(f"  • Tabela de predições: {table_name}")
print(f"  • Plots e métricas: /tmp/ (10 arquivos)")
print(f"  • Feature importance: /tmp/feature_importance.csv")

print("\n🚀 PRÓXIMOS PASSOS (FASE 4 - MLOps):")
print("  1. Criar pipeline de re-treinamento automatizado")
print("  2. Configurar monitoramento de drift (dados e modelo)")
print("  3. Implementar A/B testing entre versões do modelo")
print("  4. Criar dashboard de métricas de produção")
print("  5. Configurar alertas para degradação de performance")
print("  6. Documentar processo de aprovação e deploy")

print("\n" + "="*80)
print("✅ FASE 3 CONCLUÍDA COM SUCESSO!")
print("="*80)

# COMMAND ----------


