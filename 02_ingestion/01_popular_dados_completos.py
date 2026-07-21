# Databricks notebook source
# MAGIC %md
# MAGIC # Geração de Dados Sintéticos Completos
# MAGIC 
# MAGIC **Objetivo**: Gerar dataset realista com 1000+ clientes, 5000 faturas, 3000 pagamentos
# MAGIC 
# MAGIC **Distribuição**:
# MAGIC - 70% clientes adimplentes (baixo risco)
# MAGIC - 20% clientes risco moderado
# MAGIC - 10% clientes alto risco (inadimplentes)

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
import random
from datetime import datetime, timedelta
import numpy as np

# `spark` já vem pré-injetado e conectado ao Unity Catalog pelo runtime do notebook —
# NUNCA chamar SparkSession.builder.getOrCreate() aqui. Em serverless/Spark Connect isso cria
# uma sessão paralela desconectada do catálogo real: o código roda sem erro, mas as escritas
# não vão para as tabelas reais (bug encontrado rodando de verdade no Job).

dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
CATALOG = dbutils.widgets.get("catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Gerar Clientes (1000 registros)

# COMMAND ----------

print("👥 Gerando 1000 clientes sintéticos...\n")

# Seed para reprodutibilidade
random.seed(42)
np.random.seed(42)

# Listas para geração de nomes realistas
nomes = ["Alpha", "Beta", "Gamma", "Delta", "Omega", "Sigma", "Theta", "Zeta", 
         "Nova", "Stellar", "Prime", "Elite", "Global", "United", "Premium", 
         "Tech", "Digital", "Smart", "Advanced", "Strategic"]

sufixos = ["Corp", "Ltd", "SA", "Industries", "Group", "Solutions", "Systems", 
           "Technologies", "Ventures", "Partners", "Holdings", "Enterprises"]

setores = ["Tecnologia", "Varejo", "Indústria", "Serviços", "Construção", 
           "Agronegócio", "Saúde", "Educação", "Logística", "Financeiro"]

portes = ["Pequeno", "Médio", "Grande"]

# Gerar dados de clientes
clientes_data = []

for i in range(1, 1001):
    nome = f"{random.choice(nomes)} {random.choice(sufixos)}"
    cnpj = f"{random.randint(10, 99)}.{random.randint(100, 999)}.{random.randint(100, 999)}/0001-{random.randint(10, 99)}"
    setor = random.choice(setores)
    porte = random.choice(portes)
    
    # Receita anual varia por porte
    if porte == "Pequeno":
        receita = random.randint(500000, 5000000)
    elif porte == "Médio":
        receita = random.randint(5000000, 50000000)
    else:  # Grande
        receita = random.randint(50000000, 500000000)
    
    # Score de risco inicial (será refinado depois)
    # 70% baixo risco, 20% médio, 10% alto
    rand = random.random()
    if rand < 0.7:
        risco_inicial = "Baixo"
        score_base = random.randint(700, 1000)
    elif rand < 0.9:
        risco_inicial = "Médio"
        score_base = random.randint(400, 699)
    else:
        risco_inicial = "Alto"
        score_base = random.randint(100, 399)
    
    clientes_data.append({
        "id_cliente": i,
        "nome": nome,
        "cnpj": cnpj,
        "setor": setor,
        "porte": porte,
        "receita_anual": receita,
        "score_risco": score_base,
        "categoria_risco": risco_inicial,
        "data_cadastro": (datetime.now() - timedelta(days=random.randint(30, 730))).strftime("%Y-%m-%d")
    })

# Criar DataFrame
schema_clientes = StructType([
    StructField("id_cliente", IntegerType(), False),
    StructField("nome", StringType(), False),
    StructField("cnpj", StringType(), False),
    StructField("setor", StringType(), False),
    StructField("porte", StringType(), False),
    StructField("receita_anual", LongType(), False),
    StructField("score_risco", IntegerType(), False),
    StructField("categoria_risco", StringType(), False),
    StructField("data_cadastro", StringType(), False)
])

df_clientes = spark.createDataFrame(clientes_data, schema_clientes)

print(f"✅ {df_clientes.count()} clientes gerados")
df_clientes.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Gerar Faturas (5000 registros)

# COMMAND ----------

print("📄 Gerando 5000 faturas sintéticas...\n")

faturas_data = []
id_fatura = 1

for id_cliente in range(1, 1001):
    # Cada cliente tem entre 3 a 8 faturas
    num_faturas = random.randint(3, 8)
    
    # Pegar score do cliente
    cliente_row = [c for c in clientes_data if c["id_cliente"] == id_cliente][0]
    categoria_risco = cliente_row["categoria_risco"]
    
    for _ in range(num_faturas):
        # Valores variam por porte
        porte = cliente_row["porte"]
        if porte == "Pequeno":
            valor = random.randint(1000, 50000)
        elif porte == "Médio":
            valor = random.randint(50000, 300000)
        else:
            valor = random.randint(300000, 2000000)
        
        # Data de emissão (últimos 12 meses)
        data_emissao = datetime.now() - timedelta(days=random.randint(1, 365))
        data_vencimento = data_emissao + timedelta(days=random.choice([30, 60, 90]))
        
        # Status depende do risco do cliente
        if categoria_risco == "Baixo":
            status_prob = {"Paga": 0.85, "Pendente": 0.10, "Atrasada": 0.05}
        elif categoria_risco == "Médio":
            status_prob = {"Paga": 0.60, "Pendente": 0.20, "Atrasada": 0.20}
        else:  # Alto
            status_prob = {"Paga": 0.30, "Pendente": 0.30, "Atrasada": 0.40}
        
        rand = random.random()
        if rand < status_prob["Paga"]:
            status = "Paga"
            data_pagamento = data_vencimento - timedelta(days=random.randint(0, 5))
            dias_atraso = 0
        elif rand < status_prob["Paga"] + status_prob["Pendente"]:
            status = "Pendente"
            data_pagamento = None
            dias_atraso = 0
        else:
            status = "Atrasada"
            dias_atraso = random.randint(1, 120)
            data_pagamento = None
        
        faturas_data.append({
            "id_fatura": id_fatura,
            "id_cliente": id_cliente,
            "valor": float(valor),
            "data_emissao": data_emissao.strftime("%Y-%m-%d"),
            "data_vencimento": data_vencimento.strftime("%Y-%m-%d"),
            "data_pagamento": data_pagamento.strftime("%Y-%m-%d") if data_pagamento else None,
            "status": status,
            "dias_atraso": dias_atraso
        })
        
        id_fatura += 1
        
        if id_fatura > 5000:
            break
    
    if id_fatura > 5000:
        break

schema_faturas = StructType([
    StructField("id_fatura", IntegerType(), False),
    StructField("id_cliente", IntegerType(), False),
    StructField("valor", DoubleType(), False),
    StructField("data_emissao", StringType(), False),
    StructField("data_vencimento", StringType(), False),
    StructField("data_pagamento", StringType(), True),
    StructField("status", StringType(), False),
    StructField("dias_atraso", IntegerType(), False)
])

df_faturas = spark.createDataFrame(faturas_data, schema_faturas)

# Coluna de partição (ano-mês de emissão): consultas de monitoring/feature engineering
# filtram por janela temporal (90/180/365d), então particionar por data reduz o volume
# escaneado em produção (data skipping), mesmo com o dataset sintético sendo pequeno hoje.
df_faturas = df_faturas.withColumn("ano_mes_emissao", substring(col("data_emissao"), 1, 7))

print(f"✅ {df_faturas.count()} faturas geradas")
df_faturas.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Gerar Pagamentos (3000 registros)

# COMMAND ----------

print("💰 Gerando 3000 pagamentos sintéticos...\n")

# Pegar apenas faturas pagas
faturas_pagas = [f for f in faturas_data if f["status"] == "Paga"]

pagamentos_data = []

for i, fatura in enumerate(faturas_pagas[:3000]):
    forma_pagamento = random.choice(["Boleto", "Transferência", "Cartão", "PIX"])
    
    pagamentos_data.append({
        "id_pagamento": i + 1,
        "id_fatura": fatura["id_fatura"],
        "id_cliente": fatura["id_cliente"],
        "valor": fatura["valor"],
        "data_pagamento": fatura["data_pagamento"],
        "forma_pagamento": forma_pagamento
    })

schema_pagamentos = StructType([
    StructField("id_pagamento", IntegerType(), False),
    StructField("id_fatura", IntegerType(), False),
    StructField("id_cliente", IntegerType(), False),
    StructField("valor", DoubleType(), False),
    StructField("data_pagamento", StringType(), False),
    StructField("forma_pagamento", StringType(), False)
])

df_pagamentos = spark.createDataFrame(pagamentos_data, schema_pagamentos)

print(f"✅ {df_pagamentos.count()} pagamentos gerados")
df_pagamentos.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Salvar em Unity Catalog - Bronze Layer

# COMMAND ----------

print(f"💾 Salvando dados em {CATALOG}.bronze...\n")

# Salvar Clientes
df_clientes.write.mode("overwrite").saveAsTable(f"{CATALOG}.bronze.clientes")
print(f"✅ {CATALOG}.bronze.clientes criada")

# Salvar Faturas (particionada por ano_mes_emissao — ver comentário na criação da coluna)
df_faturas.write.mode("overwrite").partitionBy("ano_mes_emissao").saveAsTable(f"{CATALOG}.bronze.faturas")
print(f"✅ {CATALOG}.bronze.faturas criada (particionada por ano_mes_emissao)")

# Salvar Pagamentos
df_pagamentos.write.mode("overwrite").saveAsTable(f"{CATALOG}.bronze.pagamentos")
print(f"✅ {CATALOG}.bronze.pagamentos criada")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validação dos Dados

# COMMAND ----------

print("\n🔍 Validando dados gerados...\n")

print("📊 RESUMO:")
print(f"  • Clientes: {spark.table(f'{CATALOG}.bronze.clientes').count():,}")
print(f"  • Faturas: {spark.table(f'{CATALOG}.bronze.faturas').count():,}")
print(f"  • Pagamentos: {spark.table(f'{CATALOG}.bronze.pagamentos').count():,}")

print("\n📈 DISTRIBUIÇÃO DE RISCO (Clientes):")
spark.sql(f"""
    SELECT categoria_risco, COUNT(*) as total,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentual
    FROM {CATALOG}.bronze.clientes
    GROUP BY categoria_risco
    ORDER BY categoria_risco
""").show()

print("📈 DISTRIBUIÇÃO DE STATUS (Faturas):")
spark.sql(f"""
    SELECT status, COUNT(*) as total,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentual
    FROM {CATALOG}.bronze.faturas
    GROUP BY status
    ORDER BY status
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Dados Sintéticos Gerados com Sucesso!
# MAGIC 
# MAGIC **Criadas**:
# MAGIC - ✅ 1000 clientes (70% baixo risco, 20% médio, 10% alto)
# MAGIC - ✅ 5000 faturas (distribuição realista de status)
# MAGIC - ✅ 3000 pagamentos
# MAGIC 
# MAGIC **Próximos passos**:
# MAGIC 1. Gerar PDFs de notas fiscais (sample_data/)
# MAGIC 2. Criar CSVs de exemplo
# MAGIC 3. Rodar transformação Silver (03_feature_engineering/)

# COMMAND ----------

print("\n" + "="*60)
print("✅ FASE 1 - DADOS BASE COMPLETOS!")
print("="*60)


