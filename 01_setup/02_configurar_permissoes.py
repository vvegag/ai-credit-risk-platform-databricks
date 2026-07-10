# Databricks notebook source
# MAGIC %md
# MAGIC # Configuração de Permissões - Unity Catalog
# MAGIC 
# MAGIC **Objetivo**: Configurar permissões adequadas para schemas do projeto
# MAGIC 
# MAGIC **Schemas**:
# MAGIC - `workspace.risco_bronze` - Dados brutos
# MAGIC - `workspace.risco_silver` - Dados transformados
# MAGIC - `workspace.risco_gold` - Features para ML
# MAGIC 
# MAGIC **Permissões**:
# MAGIC - Data Engineers: CREATE, SELECT, MODIFY
# MAGIC - Data Scientists: SELECT (todos), CREATE/MODIFY (gold apenas)
# MAGIC - Business Users: SELECT (gold apenas)

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Validar Schemas Existentes

print("🔍 Validando schemas...\n")

schemas = ['risco_bronze', 'risco_silver', 'risco_gold']

for schema in schemas:
    try:
        spark.sql(f"DESCRIBE SCHEMA workspace.{schema}")
        print(f"✅ workspace.{schema} existe")
    except Exception as e:
        print(f"❌ workspace.{schema} NÃO existe - Rode 01_criar_catalogo_schemas.py primeiro")
        print(f"   Erro: {str(e)[:100]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configurar Permissões para Data Engineers
# MAGIC 
# MAGIC Data Engineers precisam de acesso completo para:
# MAGIC - Criar tabelas
# MAGIC - Modificar dados
# MAGIC - Executar pipelines ETL

print("👷 Configurando permissões para Data Engineers...\n")

# Grupo de exemplo - ajuste conforme sua organização
data_engineer_group = "data_engineers"

try:
    for schema in schemas:
        # GRANT CREATE TABLE
        spark.sql(f"""
            GRANT CREATE TABLE ON SCHEMA workspace.{schema} 
            TO `{data_engineer_group}`
        """)
        
        # GRANT SELECT
        spark.sql(f"""
            GRANT SELECT ON SCHEMA workspace.{schema} 
            TO `{data_engineer_group}`
        """)
        
        # GRANT MODIFY
        spark.sql(f"""
            GRANT MODIFY ON SCHEMA workspace.{schema} 
            TO `{data_engineer_group}`
        """)
        
        print(f"✅ Permissões configuradas para workspace.{schema}")
        
except Exception as e:
    print(f"⚠️  Aviso: {str(e)[:200]}")
    print("   Isso é normal se o grupo não existir ainda.")
    print("   Crie o grupo no Unity Catalog primeiro.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configurar Permissões para Data Scientists
# MAGIC 
# MAGIC Data Scientists precisam de:
# MAGIC - SELECT em Bronze/Silver (ler dados)
# MAGIC - SELECT, CREATE, MODIFY em Gold (criar features)

print("🔬 Configurando permissões para Data Scientists...\n")

data_scientist_group = "data_scientists"

try:
    # SELECT em Bronze e Silver
    for schema in ['risco_bronze', 'risco_silver']:
        spark.sql(f"""
            GRANT SELECT ON SCHEMA workspace.{schema} 
            TO `{data_scientist_group}`
        """)
        print(f"✅ SELECT granted em workspace.{schema}")
    
    # Permissões completas em Gold
    spark.sql(f"""
        GRANT CREATE TABLE, SELECT, MODIFY ON SCHEMA workspace.risco_gold 
        TO `{data_scientist_group}`
    """)
    print(f"✅ Permissões completas em workspace.risco_gold")
    
except Exception as e:
    print(f"⚠️  Aviso: {str(e)[:200]}")
    print("   Isso é normal se o grupo não existir ainda.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Configurar Permissões para Business Users
# MAGIC 
# MAGIC Business Users precisam apenas de:
# MAGIC - SELECT em Gold (ler features e métricas)

print("📊 Configurando permissões para Business Users...\n")

business_user_group = "business_users"

try:
    spark.sql(f"""
        GRANT SELECT ON SCHEMA workspace.risco_gold 
        TO `{business_user_group}`
    """)
    print(f"✅ SELECT granted em workspace.risco_gold")
    
except Exception as e:
    print(f"⚠️  Aviso: {str(e)[:200]}")
    print("   Isso é normal se o grupo não existir ainda.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verificar Permissões Configuradas

print("🔍 Verificando permissões configuradas...\n")

for schema in schemas:
    print(f"\n📂 workspace.{schema}:")
    try:
        grants = spark.sql(f"SHOW GRANTS ON SCHEMA workspace.{schema}")
        grants.show(truncate=False)
    except Exception as e:
        print(f"   ⚠️  Não foi possível listar grants: {str(e)[:100]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Criar Grupos (Referência)
# MAGIC 
# MAGIC **Como criar grupos no Unity Catalog**:
# MAGIC 
# MAGIC 1. Acesse: Databricks Workspace → Settings → Identity and Access
# MAGIC 2. Clique em "Groups"
# MAGIC 3. Create Group:
# MAGIC    - `data_engineers`
# MAGIC    - `data_scientists`
# MAGIC    - `business_users`
# MAGIC 4. Adicione usuários aos grupos
# MAGIC 5. Re-rode este notebook
# MAGIC 
# MAGIC **Ou via SQL (se tiver permissões de admin)**:

print("📋 Comandos para criar grupos (copie se necessário):\n")

comandos_criar_grupos = """
-- Criar grupos
CREATE GROUP IF NOT EXISTS data_engineers;
CREATE GROUP IF NOT EXISTS data_scientists;
CREATE GROUP IF NOT EXISTS business_users;

-- Adicionar usuários (exemplo)
-- ALTER GROUP data_engineers ADD USER 'user@example.com';
-- ALTER GROUP data_scientists ADD USER 'scientist@example.com';
-- ALTER GROUP business_users ADD USER 'analyst@example.com';
"""

print(comandos_criar_grupos)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Configuração Completa!
# MAGIC 
# MAGIC Permissões configuradas para:
# MAGIC - ✅ Data Engineers: Acesso completo (CREATE, SELECT, MODIFY)
# MAGIC - ✅ Data Scientists: SELECT em Bronze/Silver, completo em Gold
# MAGIC - ✅ Business Users: SELECT em Gold apenas
# MAGIC 
# MAGIC **Próximos passos**:
# MAGIC 1. Criar grupos no Unity Catalog (se ainda não existem)
# MAGIC 2. Adicionar usuários aos grupos
# MAGIC 3. Popular dados (02_ingestion/)

print("\n" + "="*60)
print("✅ CONFIGURAÇÃO DE PERMISSÕES COMPLETA!")
print("="*60)

