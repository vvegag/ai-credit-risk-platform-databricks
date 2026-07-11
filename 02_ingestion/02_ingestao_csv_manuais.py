# Databricks notebook source
# MAGIC %md
# MAGIC # Auto Loader para CSVs Manuais
# MAGIC 
# MAGIC **Objetivo**: Ingerir CSVs de forma incremental usando Auto Loader
# MAGIC **Fonte**: `/02_ingestion/sample_data/csvs/`
# MAGIC **Destino**: `{CATALOG}.bronze.*_csv`

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

# Caminho dos CSVs derivado do próprio notebook (funciona em qualquer workspace/Repo/Git folder)
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = "/Workspace" + "/".join(notebook_path.split("/")[:-2])
csv_path = f"{repo_root}/02_ingestion/sample_data/csvs/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configurar Auto Loader para Clientes

print("📥 Configurando Auto Loader para clientes_manuais.csv...\n")

# Checkpoint location
checkpoint_clientes = "/tmp/autoloader_checkpoints/clientes_csv"

# Ler com Auto Loader
df_clientes_csv = (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.schemaLocation", checkpoint_clientes)
    .option("header", "true")
    .option("inferSchema", "true")
    .load(f"{csv_path}clientes_manuais.csv")
)

print("✅ Auto Loader configurado para clientes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Escrever Stream para Bronze

# Escrever streaming para tabela Bronze
query_clientes = (df_clientes_csv.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", checkpoint_clientes)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)  # Processar todos os arquivos disponíveis agora
    .toTable(f"{CATALOG}.bronze.clientes_csv")
)

# Aguardar conclusão
query_clientes.awaitTermination()

print("✅ Dados de clientes_manuais.csv ingeridos!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configurar Auto Loader para Faturas

print("📥 Configurando Auto Loader para faturas_manuais.csv...\n")

checkpoint_faturas = "/tmp/autoloader_checkpoints/faturas_csv"

df_faturas_csv = (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.schemaLocation", checkpoint_faturas)
    .option("header", "true")
    .option("inferSchema", "true")
    .load(f"{csv_path}faturas_manuais.csv")
)

# Escrever stream
query_faturas = (df_faturas_csv.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", checkpoint_faturas)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(f"{CATALOG}.bronze.faturas_csv")
)

query_faturas.awaitTermination()

print("✅ Dados de faturas_manuais.csv ingeridos!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Validar Dados Ingeridos

print("\n🔍 Validando dados ingeridos...\n")

print("📊 Clientes CSV:")
spark.sql(f"SELECT COUNT(*) as total FROM {CATALOG}.bronze.clientes_csv").show()
spark.sql(f"SELECT * FROM {CATALOG}.bronze.clientes_csv LIMIT 5").show(truncate=False)

print("\n📊 Faturas CSV:")
spark.sql(f"SELECT COUNT(*) as total FROM {CATALOG}.bronze.faturas_csv").show()
spark.sql(f"SELECT * FROM {CATALOG}.bronze.faturas_csv LIMIT 5").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Auto Loader Configurado!
# MAGIC 
# MAGIC **Tabelas Criadas**:
# MAGIC - ✅ `{CATALOG}.bronze.clientes_csv`
# MAGIC - ✅ `{CATALOG}.bronze.faturas_csv`
# MAGIC 
# MAGIC **Vantagens do Auto Loader**:
# MAGIC - 📥 Ingestão incremental automática
# MAGIC - 🔄 Schema evolution com mergeSchema
# MAGIC - ⚡ Processamento eficiente de novos arquivos
# MAGIC - ✅ Exactly-once semantics
# MAGIC 
# MAGIC **Como adicionar novos dados**:
# MAGIC Basta adicionar novos CSVs no diretório `/sample_data/csvs/` e re-rodar!

print("\n" + "="*60)
print("✅ AUTO LOADER CONFIGURADO E DADOS INGERIDOS!")
print("="*60)


