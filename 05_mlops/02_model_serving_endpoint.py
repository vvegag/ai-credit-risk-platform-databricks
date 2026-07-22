# Databricks notebook source
# DBTITLE 1,Model Serving — Endpoint em Tempo Real
# MAGIC %md
# MAGIC # 🚀 Model Serving — Endpoint em Tempo Real
# MAGIC
# MAGIC ## Objetivo
# MAGIC Expor o modelo Champion (`credit_risk.gold.credit_risk_classifier@Champion`, registrado por
# MAGIC `04_modeling/01_modelo_classificacao_risco.py` e promovido por `05_mlops/01_mlops_pipeline.py`)
# MAGIC como um endpoint REST de baixa latência via Databricks Model Serving — complementa o batch
# MAGIC scoring já existente (`gold.model_predictions`) com um caminho de inferência em tempo real.
# MAGIC
# MAGIC ## Por que apontar pro alias, não pra versão fixa
# MAGIC O endpoint é criado apontando para `@Champion` (alias), não para um número de versão. Quando
# MAGIC `05_mlops/01_mlops_pipeline.py` promove um novo Champion, o endpoint passa a servir o novo
# MAGIC modelo automaticamente na próxima sincronização — sem precisar recriar o endpoint.
# MAGIC
# MAGIC ## Custo
# MAGIC Configurado com **scale-to-zero** (`workload_size="Small"`, `scale_to_zero_enabled=True`) —
# MAGIC o endpoint desliga o compute quando não recebe tráfego, evitando custo ocioso.

# COMMAND ----------

# DBTITLE 1,Setup e Imports
dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
import time

MODEL_NAME = "credit_risk_classifier"
MODEL_REGISTRY_NAME = f"{CATALOG}.gold.{MODEL_NAME}"
ENDPOINT_NAME = f"{CATALOG}_classifier_endpoint"
ALIAS = "Champion"

w = WorkspaceClient()

print(f"✅ Configuração:")
print(f"   - Modelo: {MODEL_REGISTRY_NAME}@{ALIAS}")
print(f"   - Endpoint: {ENDPOINT_NAME}")

# COMMAND ----------

# DBTITLE 1,Descobrir Versão Atual do Champion
# Model Serving precisa de um número de versão explícito na config (não aceita alias
# diretamente) — resolvemos o alias pra versão atual aqui, e reaplicamos esse passo sempre
# que quiser sincronizar o endpoint com um novo Champion promovido.
from mlflow.tracking import MlflowClient

mlflow_client = MlflowClient(registry_uri="databricks-uc")
try:
    champion_version = mlflow_client.get_model_version_by_alias(MODEL_REGISTRY_NAME, ALIAS)
    CHAMPION_VERSION = champion_version.version
    print(f"✅ Champion atual: {MODEL_REGISTRY_NAME} v{CHAMPION_VERSION}")
except Exception as e:
    CHAMPION_VERSION = None
    print(f"⚠️ Nenhum Champion encontrado no registry ({type(e).__name__}: {e})")
    print("↳ Rode 04_modeling/01_modelo_classificacao_risco.py primeiro para registrar/promover um Champion.")

# COMMAND ----------

# DBTITLE 1,Criar ou Atualizar o Endpoint
# Idempotente: cria se não existir, atualiza a versão servida se já existir (ex: quando um
# novo Champion foi promovido desde a última sincronização).
if CHAMPION_VERSION is not None:
    served_entity = ServedEntityInput(
        entity_name=MODEL_REGISTRY_NAME,
        entity_version=CHAMPION_VERSION,
        workload_size="Small",
        scale_to_zero_enabled=True,
    )

    try:
        existing = w.serving_endpoints.get(ENDPOINT_NAME)
        print(f"ℹ️ Endpoint '{ENDPOINT_NAME}' já existe — atualizando para v{CHAMPION_VERSION}...")
        w.serving_endpoints.update_config_and_wait(
            name=ENDPOINT_NAME,
            served_entities=[served_entity],
        )
        print(f"✅ Endpoint atualizado: {ENDPOINT_NAME} agora serve v{CHAMPION_VERSION}")

    except Exception:
        print(f"ℹ️ Endpoint '{ENDPOINT_NAME}' não existe ainda — criando...")
        try:
            w.serving_endpoints.create_and_wait(
                name=ENDPOINT_NAME,
                config=EndpointCoreConfigInput(served_entities=[served_entity]),
            )
            print(f"✅ Endpoint criado: {ENDPOINT_NAME} servindo v{CHAMPION_VERSION}")
        except Exception as e:
            print(f"⚠️ Não foi possível criar o endpoint ({type(e).__name__}: {e})")
            print("↳ Costuma ser falta de permissão de admin/serving no workspace, ou quota de")
            print("  compute — não é bug de código. Confirme com o admin do workspace.")
else:
    print("⏭️ Pulando criação do endpoint (sem Champion registrado)")

# COMMAND ----------

# DBTITLE 1,Testar o Endpoint
# MAGIC %md
# MAGIC ## Testar com uma requisição real
# MAGIC Pega uma linha real de `gold.features_ml` como exemplo de payload, no mesmo formato de
# MAGIC features usado no treino (mesmo one-hot encoding de `porte`/`setor`).

# COMMAND ----------

if CHAMPION_VERSION is not None:
    import pandas as pd

    df_sample = spark.table(f"{CATALOG}.gold.features_ml").limit(1).toPandas()

    cols_to_drop = [
        'id_cliente', 'cnpj', 'nome', 'categoria_rfm', 'perfil_comportamental',
        'categoria_risco', 'data_cadastro', 'taxa_inadimplencia',
    ]
    feature_cols = [c for c in df_sample.columns if c not in cols_to_drop]
    df_sample_encoded = pd.get_dummies(df_sample[feature_cols], columns=['porte', 'setor'])

    try:
        response = w.serving_endpoints.query(
            name=ENDPOINT_NAME,
            dataframe_records=df_sample_encoded.to_dict(orient="records"),
        )
        print("✅ Resposta do endpoint:")
        print(response.predictions)
    except Exception as e:
        print(f"⚠️ Endpoint ainda não está pronto para receber tráfego ({type(e).__name__})")
        print("↳ Endpoints novos levam alguns minutos para provisionar o compute na primeira vez.")
        print(f"↳ Verifique o status em: Workspace → Machine Learning → Serving → {ENDPOINT_NAME}")

# COMMAND ----------

# DBTITLE 1,Exemplo de Chamada via REST (fora do notebook)
# MAGIC %md
# MAGIC ## Exemplo de uso externo (aplicação, dashboard, outro serviço)
# MAGIC ```python
# MAGIC import requests
# MAGIC
# MAGIC endpoint_url = f"{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations"
# MAGIC headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
# MAGIC payload = {"dataframe_records": [{"total_faturado_90d": 50000, "recency_dias": 12, "...": "..."}]}
# MAGIC
# MAGIC response = requests.post(endpoint_url, headers=headers, json=payload)
# MAGIC print(response.json())
# MAGIC ```
# MAGIC
# MAGIC ## Próximos passos (fora de escopo deste notebook)
# MAGIC - Inference Tables (log automático de request/response) para monitorar drift em produção real
# MAGIC - Alertas automáticos se a latência p95 ou a taxa de erro do endpoint ultrapassar um threshold
# MAGIC - Autenticação via Service Principal em vez de token pessoal, para chamadas de sistemas externos
