# Databricks notebook source
# DBTITLE 1,🎲 Gerador de Dados Sintéticos - Inadimplência
# MAGIC %md
# MAGIC # 🎲 Gerador de Dados Sintéticos - Inadimplência
# MAGIC
# MAGIC ## Objetivo
# MAGIC Gerar dados sintéticos realistas para simular o ambiente financeiro de uma empresa de varejo/serviços com múltiplas marcas e clientes B2B/B2C.
# MAGIC
# MAGIC ## Dados a Gerar
# MAGIC
# MAGIC 1. **Clientes** (1000+): CPF/CNPJ, nome, tier, grupo econômico
# MAGIC 2. **Marcas** (50+): Nomes ficticios, segmentos, unidades de negócio
# MAGIC 3. **Faturas** (20k+): Histórico de 12 meses, valores realistas
# MAGIC 4. **Pagamentos**: Com padrões de pontualidade, atraso crônico, inadimplência
# MAGIC 5. **CSVs Manuais**: Simula planilhas do financeiro com fórmulas
# MAGIC
# MAGIC ## Padrões Comportamentais
# MAGIC
# MAGIC - **Pontual** (40%): Paga na data ou até 5 dias após vencimento
# MAGIC - **Crônico** (30%): Sempre paga com 20-40 dias de atraso (mas paga)
# MAGIC - **Instável** (20%): Ora pontual, ora atrasado (imprevisível)
# MAGIC - **Risco** (10%): Múltiplas faturas >90 dias, tendência a não pagar
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Setup e Imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Configurar seed para reprodução
np.random.seed(42)
random.seed(42)
fake = Faker('pt_BR')
Faker.seed(42)

print("✅ Bibliotecas importadas")
print(f"📅 Data de referência: {datetime.now().strftime('%Y-%m-%d')}")

# COMMAND ----------

# DBTITLE 1,Gerar Marcas (Clientes B2B)
# Gerar marcas ficticias
n_marcas = 60

marcas_nomes = [
    f"{fake.company()} {suffix}" 
    for suffix in ['Varejo', 'Distribuidora', 'Atacado', 'Franquia', 'E-commerce'] * 12
]

segmentos = ['Moda', 'Alimentos', 'Eletrônicos', 'Farmácia', 'Cosméticos', 'Pet', 'Casa', 'Esportes']
unidades_negocio = ['CRMBonus', 'Giftback', 'CRMAds', 'Avenidas']
tiers = ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']

marcas_data = []
for i in range(n_marcas):
    marcas_data.append({
        'codigo_marca': f'MRC{str(i+1).zfill(4)}',
        'nome_marca': marcas_nomes[i],
        'cnpj': fake.cnpj(),
        'segmento': random.choice(segmentos),
        'unidade_negocio': random.choice(unidades_negocio),
        'tier': random.choice(tiers),
        'data_cadastro': fake.date_between(start_date='-3y', end_date='-6m')
    })

df_marcas = pd.DataFrame(marcas_data)

print(f"✅ {len(df_marcas)} marcas geradas")
print(f"\nDistribuição por tier:")
print(df_marcas['tier'].value_counts())
print(f"\nAmostra:")
display(df_marcas.head())

# COMMAND ----------

# DBTITLE 1,Gerar Clientes (Pessoas Físicas + Grupos Econômicos)
# Gerar clientes (mix de PF e PJ)
n_clientes = 1200

clientes_data = []

for i in range(n_clientes):
    tipo = 'PJ' if i < 800 else 'PF'  # 800 PJ, 400 PF
    
    if tipo == 'PJ':
        nome = fake.company()
        doc = fake.cnpj()
    else:
        nome = fake.name()
        doc = fake.cpf()
    
    # Associar a uma marca (cliente pertence a uma marca)
    marca = random.choice(df_marcas['codigo_marca'].values)
    
    clientes_data.append({
        'codigo_cliente': f'CLI{str(i+1).zfill(5)}',
        'nome_cliente': nome,
        'documento': doc,
        'tipo_pessoa': tipo,
        'codigo_marca': marca,
        'email': fake.email(),
        'telefone': fake.phone_number(),
        'data_cadastro': fake.date_between(start_date='-2y', end_date='-1m')
    })

df_clientes = pd.DataFrame(clientes_data)

# Enrichment: associar perfil de pagamento
perfis = ['Pontual', 'Cronico', 'Instavel', 'Risco']
weights = [0.40, 0.30, 0.20, 0.10]
df_clientes['perfil_pagamento'] = np.random.choice(perfis, size=len(df_clientes), p=weights)

print(f"✅ {len(df_clientes)} clientes gerados")
print(f"\nDistribuição:")
print(f"  PJ: {(df_clientes['tipo_pessoa'] == 'PJ').sum()}")
print(f"  PF: {(df_clientes['tipo_pessoa'] == 'PF').sum()}")
print(f"\nPerfil de pagamento:")
print(df_clientes['perfil_pagamento'].value_counts())
print(f"\nAmostra:")
display(df_clientes.head())

# COMMAND ----------

# DBTITLE 1,Gerar Faturas com Histórico Temporal (12 meses)
# Gerar faturas com snapshots temporais (simula 12 meses de histórico)
data_hoje = datetime(2026, 7, 2)
data_inicio = data_hoje - timedelta(days=365)

n_snapshots = 12  # 1 snapshot por mês
snapshot_dates = [data_inicio + timedelta(days=30*i) for i in range(n_snapshots)]

print(f"📅 Gerando faturas de {data_inicio.strftime('%Y-%m-%d')} até {data_hoje.strftime('%Y-%m-%d')}")
print(f"   {n_snapshots} snapshots mensais\n")

# Cada cliente gera ~20 faturas ao longo de 12 meses
faturas_data = []
titulo_counter = 1

for _, cliente in df_clientes.iterrows():
    codigo_cliente = cliente['codigo_cliente']
    perfil = cliente['perfil_pagamento']
    
    # Número de faturas varia por cliente
    n_faturas_cliente = random.randint(15, 25)
    
    for j in range(n_faturas_cliente):
        # Data de emissão aleatória nos últimos 12 meses
        data_emissao = data_inicio + timedelta(days=random.randint(0, 330))
        data_vencimento = data_emissao + timedelta(days=random.choice([15, 30, 45, 60]))
        
        # Valor da fatura (variedade realista)
        if cliente['tipo_pessoa'] == 'PJ':
            valor = round(random.uniform(5000, 50000), 2)
        else:
            valor = round(random.uniform(500, 5000), 2)
        
        # Simular pagamento baseado no perfil
        data_pagamento = None
        valor_pagado = 0
        status = 'Pendente'
        
        if data_vencimento < data_hoje - timedelta(days=7):  # Fatura já venceu
            if perfil == 'Pontual':
                # Paga na data ou até 5 dias após
                dias_atraso = random.randint(-5, 5)
                data_pagamento = data_vencimento + timedelta(days=dias_atraso)
                valor_pagado = valor
                status = 'Pago'
                
            elif perfil == 'Cronico':
                # Sempre paga, mas com 20-40 dias de atraso
                dias_atraso = random.randint(20, 40)
                data_pagamento = data_vencimento + timedelta(days=dias_atraso)
                if data_pagamento < data_hoje:
                    valor_pagado = valor
                    status = 'Pago'
                else:
                    status = 'Pendente'
                    
            elif perfil == 'Instavel':
                # 50% paga pontual, 50% atrasa muito
                if random.random() < 0.5:
                    dias_atraso = random.randint(-5, 10)
                    data_pagamento = data_vencimento + timedelta(days=dias_atraso)
                    valor_pagado = valor
                    status = 'Pago'
                else:
                    dias_atraso = random.randint(30, 90)
                    data_pagamento = data_vencimento + timedelta(days=dias_atraso)
                    if data_pagamento < data_hoje:
                        valor_pagado = valor
                        status = 'Pago'
                    else:
                        status = 'Pendente'
                        
            elif perfil == 'Risco':
                # 30% paga com muito atraso, 70% não paga
                if random.random() < 0.3:
                    dias_atraso = random.randint(60, 120)
                    data_pagamento = data_vencimento + timedelta(days=dias_atraso)
                    if data_pagamento < data_hoje:
                        # Pagamento parcial
                        valor_pagado = round(valor * random.uniform(0.5, 1.0), 2)
                        status = 'Parcial' if valor_pagado < valor else 'Pago'
                    else:
                        status = 'Pendente'
                else:
                    # Não pagou
                    status = 'Pendente'
        
        faturas_data.append({
            'titulo': f'INV-{str(titulo_counter).zfill(8)}',
            'codigo_cliente': codigo_cliente,
            'codigo_marca': cliente['codigo_marca'],
            'data_emissao': data_emissao,
            'data_vencimento': data_vencimento,
            'data_pagamento': data_pagamento,
            'valor_titulo': valor,
            'valor_pagado': valor_pagado,
            'status': status,
            'perfil_cliente': perfil
        })
        titulo_counter += 1

df_faturas = pd.DataFrame(faturas_data)

print(f"✅ {len(df_faturas)} faturas geradas")
print(f"\nEstatísticas:")
print(f"  Total faturado: R$ {df_faturas['valor_titulo'].sum():,.2f}")
print(f"  Total recebido: R$ {df_faturas['valor_pagado'].sum():,.2f}")
print(f"  Inadimplência: R$ {(df_faturas['valor_titulo'] - df_faturas['valor_pagado']).sum():,.2f}")
print(f"\nStatus:")
print(df_faturas['status'].value_counts())
print(f"\nAmostra:")
display(df_faturas.head(10))

# COMMAND ----------

# DBTITLE 1,Gerar CSV Manual do Financeiro (Ajustes e Renegociações)
# Simular CSVs manuais que o financeiro cria com ajustes, renegociações, etc.
# Selecionar ~5% das faturas para ter ajustes manuais

faturas_com_ajuste = df_faturas.sample(frac=0.05, random_state=42)

csv_manual_data = []

for _, fatura in faturas_com_ajuste.iterrows():
    tipo_ajuste = random.choice(['Renegociacao', 'Desconto', 'Juros', 'Correcao_Manual'])
    
    if tipo_ajuste == 'Renegociacao':
        # Nova data de vencimento
        nova_data_vencimento = fatura['data_vencimento'] + timedelta(days=random.randint(30, 90))
        valor_ajustado = fatura['valor_titulo']
        observacao = f"Renegociação - novo vencimento: {nova_data_vencimento.strftime('%Y-%m-%d')}"
        
    elif tipo_ajuste == 'Desconto':
        # Desconto para pagamento imediato
        valor_ajustado = round(fatura['valor_titulo'] * random.uniform(0.85, 0.95), 2)
        nova_data_vencimento = fatura['data_vencimento']
        observacao = f"Desconto de {((1 - valor_ajustado/fatura['valor_titulo'])*100):.1f}% para pagamento antecipado"
        
    elif tipo_ajuste == 'Juros':
        # Juros por atraso
        valor_ajustado = round(fatura['valor_titulo'] * random.uniform(1.05, 1.15), 2)
        nova_data_vencimento = fatura['data_vencimento']
        observacao = f"Juros de {((valor_ajustado/fatura['valor_titulo']-1)*100):.1f}% por atraso"
        
    else:  # Correcao_Manual
        valor_ajustado = round(fatura['valor_titulo'] + random.uniform(-500, 500), 2)
        nova_data_vencimento = fatura['data_vencimento']
        observacao = "Correção manual - ajuste contábil"
    
    csv_manual_data.append({
        'titulo': fatura['titulo'],
        'tipo_ajuste': tipo_ajuste,
        'valor_original': fatura['valor_titulo'],
        'valor_ajustado': valor_ajustado,
        'data_vencimento_original': fatura['data_vencimento'],
        'data_vencimento_nova': nova_data_vencimento,
        'responsavel': random.choice(['Financeiro_Maria', 'Financeiro_Joao', 'Cobrança_Carlos']),
        'data_ajuste': datetime.now() - timedelta(days=random.randint(0, 30)),
        'observacao': observacao
    })

df_csv_manual = pd.DataFrame(csv_manual_data)

print(f"✅ {len(df_csv_manual)} ajustes manuais gerados (5% das faturas)")
print(f"\nTipos de ajuste:")
print(df_csv_manual['tipo_ajuste'].value_counts())
print(f"\nAmostra:")
display(df_csv_manual.head())

# COMMAND ----------

# DBTITLE 1,Salvar em Delta Tables (Unity Catalog)
# Converter para Spark DataFrames e salvar no Unity Catalog

print("📦 Salvando dados no Unity Catalog...\n")

# 1. Marcas
spark_marcas = spark.createDataFrame(df_marcas)
spark_marcas.write.mode("overwrite").saveAsTable("risco_financeiro.bronze_raw.marcas_raw")
print(f"✅ Tabela 'marcas_raw' salva ({spark_marcas.count()} linhas)")

# 2. Clientes
spark_clientes = spark.createDataFrame(df_clientes)
spark_clientes.write.mode("overwrite").saveAsTable("risco_financeiro.bronze_raw.clientes_raw")
print(f"✅ Tabela 'clientes_raw' salva ({spark_clientes.count()} linhas)")

# 3. Faturas
spark_faturas = spark.createDataFrame(df_faturas)
spark_faturas.write.mode("overwrite").saveAsTable("risco_financeiro.bronze_raw.faturas_raw")
print(f"✅ Tabela 'faturas_raw' salva ({spark_faturas.count()} linhas)")

# 4. CSV Manual
spark_csv_manual = spark.createDataFrame(df_csv_manual)
spark_csv_manual.write.mode("overwrite").saveAsTable("risco_financeiro.bronze_raw.csv_financeiro_raw")
print(f"✅ Tabela 'csv_financeiro_raw' salva ({spark_csv_manual.count()} linhas)")

print("\n" + "="*70)
print("✅ DADOS SINTÉTICOS GERADOS E SALVOS COM SUCESSO!")
print("="*70)

# COMMAND ----------

# DBTITLE 1,Validação e Sumário
# Validação final e sumário
print("\n📊 SUMÁRIO DOS DADOS GERADOS\n")
print("="*70)

# Estatísticas gerais
print(f"\n🏪 MARCAS:")
print(f"  Total: {len(df_marcas)}")
print(f"  Segmentos: {df_marcas['segmento'].nunique()}")
print(f"  Unidades de Negócio: {df_marcas['unidade_negocio'].nunique()}")

print(f"\n👥 CLIENTES:")
print(f"  Total: {len(df_clientes)}")
print(f"  PJ: {(df_clientes['tipo_pessoa'] == 'PJ').sum()} ({(df_clientes['tipo_pessoa'] == 'PJ').sum()/len(df_clientes)*100:.1f}%)")
print(f"  PF: {(df_clientes['tipo_pessoa'] == 'PF').sum()} ({(df_clientes['tipo_pessoa'] == 'PF').sum()/len(df_clientes)*100:.1f}%)")

print(f"\n💵 FATURAS:")
print(f"  Total: {len(df_faturas)}")
print(f"  Valor Total Faturado: R$ {df_faturas['valor_titulo'].sum():,.2f}")
print(f"  Valor Total Recebido: R$ {df_faturas['valor_pagado'].sum():,.2f}")
print(f"  Inadimplência: R$ {(df_faturas['valor_titulo'] - df_faturas['valor_pagado']).sum():,.2f}")
print(f"  Taxa de Inadimplência: {((df_faturas['valor_titulo'] - df_faturas['valor_pagado']).sum() / df_faturas['valor_titulo'].sum() * 100):.2f}%")

print(f"\n📄 AJUSTES MANUAIS (CSV):")
print(f"  Total: {len(df_csv_manual)}")
print(f"  Tipos: {df_csv_manual['tipo_ajuste'].nunique()}")

print(f"\n📊 PERFIS DE PAGAMENTO:")
for perfil, count in df_clientes['perfil_pagamento'].value_counts().items():
    pct = count / len(df_clientes) * 100
    print(f"  {perfil}: {count} clientes ({pct:.1f}%)")

print("\n" + "="*70)
print("✅ Dados prontos para uso na camada Bronze!")
print("\n👉 Próximo passo: Executar transformação Silver")
print("   Notebook: 03_feature_engineering/01_transformacao_silver")
print("="*70)

# COMMAND ----------



