# Databricks notebook source
# MAGIC %md
# MAGIC # Auto Loader para CSVs Manuais
# MAGIC 
# MAGIC **Objetivo**: Ingerir CSVs de forma incremental usando Auto Loader
# MAGIC **Fonte**: `/02_ingestion/sample_data/csvs/`
# MAGIC **Destino**: `workspace.risco_bronze.*_csv`

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configurar Auto Loader para Clientes

print("📥 Configurando Auto Loader para clientes_manuais.csv...\n")

# Caminho dos CSVs
csv_path = "/Workspace/Users/valdomirovega@hotmail.com/ai-credit-risk-platform-databricks/02_ingestion/sample_data/csvs/"

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
    .toTable("workspace.risco_bronze.clientes_csv")
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
    .toTable("workspace.risco_bronze.faturas_csv")
)

query_faturas.awaitTermination()

print("✅ Dados de faturas_manuais.csv ingeridos!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Validar Dados Ingeridos

print("\n🔍 Validando dados ingeridos...\n")

print("📊 Clientes CSV:")
spark.sql("SELECT COUNT(*) as total FROM workspace.risco_bronze.clientes_csv").show()
spark.sql("SELECT * FROM workspace.risco_bronze.clientes_csv LIMIT 5").show(truncate=False)

print("\n📊 Faturas CSV:")
spark.sql("SELECT COUNT(*) as total FROM workspace.risco_bronze.faturas_csv").show()
spark.sql("SELECT * FROM workspace.risco_bronze.faturas_csv LIMIT 5").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Auto Loader Configurado!
# MAGIC 
# MAGIC **Tabelas Criadas**:
# MAGIC - ✅ `workspace.risco_bronze.clientes_csv`
# MAGIC - ✅ `workspace.risco_bronze.faturas_csv`
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


