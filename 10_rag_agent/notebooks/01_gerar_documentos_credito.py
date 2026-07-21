# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,📋 Gerador de Documentos de Crédito - README
# MAGIC %md
# MAGIC # 📄 Gerador de Documentos de Crédito
# MAGIC
# MAGIC ## 🎯 Objetivo
# MAGIC Gerar documentos PDF realistas simulando um cenário real de análise de crédito:
# MAGIC - **Contratos de crédito** com termos e condições
# MAGIC - **Extratos bancários** com movimentações
# MAGIC - **Comprovantes de renda** com histórico salarial
# MAGIC - **Histórico de pagamentos** com faturas anteriores
# MAGIC - **Análises de crédito** com decisões e scores
# MAGIC
# MAGIC ## 📊 Dados Fonte
# MAGIC Os PDFs serão gerados a partir dos dados reais das tabelas:
# MAGIC - `credit_risk.bronze.clientes`
# MAGIC - `credit_risk.bronze.faturas`
# MAGIC - `credit_risk.bronze.pagamentos`
# MAGIC - `credit_risk.gold.features_ml`
# MAGIC
# MAGIC ## 🗂️ Estrutura de Armazenamento
# MAGIC ```
# MAGIC /Volumes/credit_risk/documentos/documentos_credito/
# MAGIC ├── contratos/
# MAGIC ├── extratos_bancarios/
# MAGIC ├── comprovantes_renda/
# MAGIC └── historico_pagamentos/
# MAGIC ```
# MAGIC
# MAGIC ## 🚀 Próximos Passos
# MAGIC 1. ✅ Gerar documentos PDF
# MAGIC 2. ⏭️ Extrair texto e metadados
# MAGIC 3. ⏭️ Criar embeddings vetoriais
# MAGIC 4. ⏭️ Setup RAG Assistant

# COMMAND ----------

# DBTITLE 1,📦 Instalar Bibliotecas de Geração de PDF
# MAGIC %pip install reportlab pypdf pillow faker --quiet
# MAGIC dbutils.library.restartPython()
# MAGIC
# MAGIC print("✅ Bibliotecas instaladas:")
# MAGIC print("  • reportlab - Geração de PDFs")
# MAGIC print("  • pypdf - Manipulação de PDFs")
# MAGIC print("  • pillow - Processamento de imagens")
# MAGIC print("  • faker - Dados sintéticos realistas")

# COMMAND ----------

# DBTITLE 1,🗂️ Setup Volume para Documentos
# Criar schema e volume para documentos
catalog = "credit_risk"
schema = "documentos"
volume = "documentos_credito"

print("📁 Configurando armazenamento de documentos...\n")

# Criar schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
print(f"✅ Schema criado: {catalog}.{schema}")

# Criar volume
spark.sql(f"""
    CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}
""")
print(f"✅ Volume criado: {catalog}.{schema}.{volume}")

# Estrutura de pastas
base_path = f"/Volumes/{catalog}/{schema}/{volume}"
subfolders = [
    "contratos",
    "extratos_bancarios",
    "comprovantes_renda",
    "historico_pagamentos",
    "analises_credito"
]

for folder in subfolders:
    folder_path = f"{base_path}/{folder}"
    try:
        dbutils.fs.mkdirs(folder_path)
        print(f"✅ Pasta criada: {folder}/")
    except:
        print(f"ℹ️  Pasta já existe: {folder}/")

print(f"\n📁 Base path: {base_path}")
print("\n✅ Estrutura de documentos pronta!")

# COMMAND ----------

# DBTITLE 1,📊 Carregar Dados das Tabelas Bronze/Gold
# Carregar dados e enriquecer com campos sintéticos
from pyspark.sql.functions import col, expr, when, rand, round as spark_round
import random
from faker import Faker

fake = Faker('pt_BR')

print("📊 Carregando e enriquecendo dados...\n")

# Clientes - enriquecer com dados sintéticos
df_clientes = spark.table("credit_risk.bronze.clientes") \
    .withColumn("email", expr("concat('contato', id_cliente, '@empresa', id_cliente, '.com.br')")) \
    .withColumn("telefone", expr("concat('(11) 9', LPAD(CAST(FLOOR(rand() * 10000000) AS STRING), 8, '0'))")) \
    .withColumn("score_credito", spark_round(400 + rand() * 400, 0).cast("int")) \
    .withColumn("endereco", expr("concat('Av. Empresarial, ', CAST(id_cliente * 100 AS STRING), ' - São Paulo/SP')")) \
    .limit(50)

print(f"✅ {df_clientes.count()} clientes carregados e enriquecidos")

# Faturas
df_faturas = spark.table("credit_risk.bronze.faturas").limit(200)
print(f"✅ {df_faturas.count()} faturas carregadas")

# Pagamentos
df_pagamentos = spark.table("credit_risk.bronze.pagamentos").limit(150)
print(f"✅ {df_pagamentos.count()} pagamentos carregados")

print("\n✅ Dados carregados com sucesso!")

# Converter para Pandas
clientes_pd = df_clientes.toPandas()
faturas_pd = df_faturas.toPandas()
pagamentos_pd = df_pagamentos.toPandas()

print(f"\n📋 Amostra de dados:")
print(f"  • Cliente ID: {clientes_pd['id_cliente'].iloc[0]}")
print(f"  • Nome: {clientes_pd['nome'].iloc[0]}")
print(f"  • CNPJ: {clientes_pd['cnpj'].iloc[0]}")
print(f"  • Score: {clientes_pd['score_credito'].iloc[0]}")
print(f"  • Email: {clientes_pd['email'].iloc[0]}")
print(f"  • Telefone: {clientes_pd['telefone'].iloc[0]}")

# COMMAND ----------

# DBTITLE 1,📝 Função: Gerar Contrato de Crédito PDF
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from datetime import datetime, timedelta
import random

def gerar_contrato_credito(cliente_row, output_path):
    """
    Gera um contrato de crédito PDF realista
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Estilo customizado
    titulo_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    corpo_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    # Cabeçalho
    story.append(Paragraph("CONTRATO DE CRÉDITO PESSOAL", titulo_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Número do contrato
    num_contrato = f"CT-{cliente_row['id_cliente']}-{random.randint(1000, 9999)}"
    data_contrato = datetime.now().strftime("%d/%m/%Y")
    
    story.append(Paragraph(f"<b>Contrato Nº:</b> {num_contrato}", corpo_style))
    story.append(Paragraph(f"<b>Data:</b> {data_contrato}", corpo_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Identificação das partes
    story.append(Paragraph("<b>1. IDENTIFICAÇÃO DAS PARTES</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    credora = Paragraph("""
        <b>CREDORA:</b> FinCredit S.A. - Instituição Financeira<br/>
        CNPJ: 12.345.678/0001-99<br/>
        Endereço: Av. Paulista, 1000 - São Paulo/SP
    """, corpo_style)
    story.append(credora)
    story.append(Spacer(1, 0.1*inch))
    
    devedor = Paragraph(f"""
        <b>CONTRATANTE:</b> {cliente_row['nome']}<br/>
        CNPJ: {cliente_row['cnpj']}<br/>
        Porte: {cliente_row['porte']} | Setor: {cliente_row['setor']}<br/>
        Receita Anual: R$ {cliente_row['receita_anual']:,.2f}<br/>
        Endereço: {cliente_row['endereco']}<br/>
        E-mail: {cliente_row['email']}<br/>
        Telefone: {cliente_row['telefone']}
    """, corpo_style)
    story.append(devedor)
    story.append(Spacer(1, 0.3*inch))
    
    # Condições do crédito
    story.append(Paragraph("<b>2. CONDIÇÕES DO CRÉDITO</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    valor_credito = random.uniform(5000, 50000)
    taxa_juros = random.uniform(1.5, 3.5)
    prazo_meses = random.choice([12, 18, 24, 36])
    
    condicoes_data = [
        ['Item', 'Valor'],
        ['Valor do Crédito', f'R$ {valor_credito:,.2f}'],
        ['Taxa de Juros (a.m.)', f'{taxa_juros:.2f}%'],
        ['Prazo', f'{prazo_meses} meses'],
        ['Primeira Parcela', (datetime.now() + timedelta(days=30)).strftime('%d/%m/%Y')],
        ['Score de Crédito', str(cliente_row['score_credito'])],
        ['Status Aprovação', 'APROVADO' if cliente_row['score_credito'] > 600 else 'ANÁLISE']
    ]
    
    table = Table(condicoes_data, colWidths=[3*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3*inch))
    
    # Cláusulas
    story.append(Paragraph("<b>3. CLÁUSULAS CONTRATUAIS</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    clausulas = [
        "O DEVEDOR se compromete a pagar as parcelas nas datas estipuladas.",
        "Em caso de atraso, serão aplicados juros de mora de 1% ao mês.",
        "O crédito foi concedido com base na análise de risco e score de crédito.",
        "O DEVEDOR declara que as informações prestadas são verdadeiras.",
        "O presente contrato é regido pelas leis brasileiras."
    ]
    
    for i, clausula in enumerate(clausulas, 1):
        story.append(Paragraph(f"<b>3.{i}.</b> {clausula}", corpo_style))
    
    story.append(Spacer(1, 0.5*inch))
    
    # Assinaturas
    story.append(Paragraph("<b>4. ASSINATURAS</b>", styles['Heading2']))
    story.append(Spacer(1, 0.3*inch))
    
    assinaturas = [
        ['_' * 40, '_' * 40],
        ['FinCredit S.A.', cliente_row['nome']],
        ['CREDORA', 'CONTRATANTE']
    ]
    
    table_assinatura = Table(assinaturas, colWidths=[2.5*inch, 2.5*inch])
    table_assinatura.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(table_assinatura)
    
    # Rodapé
    story.append(Spacer(1, 0.5*inch))
    rodape = Paragraph(
        f"<i>Documento gerado eletronicamente em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>",
        ParagraphStyle('Footer', parent=corpo_style, fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    )
    story.append(rodape)
    
    # Construir PDF
    doc.build(story)
    return num_contrato, valor_credito, taxa_juros, prazo_meses

print("✅ Função gerar_contrato_credito() criada")

# COMMAND ----------

# DBTITLE 1,📊 Função: Gerar Extrato Bancário PDF
def gerar_extrato_bancario(cliente_row, faturas_cliente, pagamentos_cliente, output_path):
    """
    Gera um extrato bancário PDF com movimentações
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Cabeçalho
    titulo_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#0066cc'),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph("EXTRATO BANCÁRIO", titulo_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Informações da conta
    info_conta = f"""
        <b>Banco:</b> Banco Empresarial S.A. - Agência: 0001 - Conta: {random.randint(10000, 99999)}-{random.randint(0,9)}<br/>
        <b>Titular:</b> {cliente_row['nome']}<br/>
        <b>CNPJ:</b> {cliente_row['cnpj']}<br/>
        <b>Período:</b> {(datetime.now() - timedelta(days=90)).strftime('%d/%m/%Y')} a {datetime.now().strftime('%d/%m/%Y')}
    """
    story.append(Paragraph(info_conta, styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))
    
    # Movimentações
    story.append(Paragraph("<b>MOVIMENTAÇÕES</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    # Construir dados da tabela
    movimentacoes = [['Data', 'Descrição', 'Valor', 'Saldo']]
    
    saldo = random.uniform(1000, 10000)
    
    # Adicionar receita mensal (aproximada)
    receita_mensal = cliente_row['receita_anual'] / 12
    movimentacoes.append([
        (datetime.now() - timedelta(days=60)).strftime('%d/%m/%Y'),
        'Recebimento de Clientes',
        f'R$ +{receita_mensal:,.2f}',
        f'R$ {saldo + receita_mensal:,.2f}'
    ])
    saldo += receita_mensal
    
    # Adicionar pagamentos
    for _, pag in pagamentos_cliente.iterrows():
        valor_pag = pag.get('valor_pago', random.uniform(100, 1000))
        saldo -= valor_pag
        movimentacoes.append([
            pag.get('data_pagamento', datetime.now() - timedelta(days=random.randint(1, 90))).strftime('%d/%m/%Y'),
            'Pagamento Fatura',
            f'R$ -{valor_pag:,.2f}',
            f'R$ {saldo:,.2f}'
        ])
    
    # Adicionar outras despesas
    for _ in range(5):
        valor_despesa = random.uniform(50, 500)
        saldo -= valor_despesa
        movimentacoes.append([
            (datetime.now() - timedelta(days=random.randint(1, 90))).strftime('%d/%m/%Y'),
            random.choice(['Supermercado', 'Farmácia', 'Combustível', 'Restaurante', 'Compras Online']),
            f'R$ -{valor_despesa:,.2f}',
            f'R$ {saldo:,.2f}'
        ])
    
    # Ordenar por data (simplificado)
    movimentacoes_sorted = [movimentacoes[0]] + sorted(movimentacoes[1:], key=lambda x: x[0], reverse=True)
    
    # Criar tabela
    table = Table(movimentacoes_sorted, colWidths=[1.2*inch, 2.5*inch, 1.3*inch, 1.3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066cc')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(table)
    
    # Resumo
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>RESUMO DO PERÍODO</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    total_creditos = receita_mensal
    total_debitos = sum([float(m[2].replace('R$ -', '').replace(',', '')) for m in movimentacoes[1:] if '-' in m[2]])
    
    resumo_data = [
        ['Total de Créditos', f'R$ {total_creditos:,.2f}'],
        ['Total de Débitos', f'R$ {total_debitos:,.2f}'],
        ['Saldo Final', f'R$ {saldo:,.2f}']
    ]
    
    table_resumo = Table(resumo_data, colWidths=[3*inch, 2*inch])
    table_resumo.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightblue)
    ]))
    story.append(table_resumo)
    
    # Rodapé
    story.append(Spacer(1, 0.5*inch))
    rodape = Paragraph(
        f"<i>Extrato gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Documento eletrônico</i>",
        ParagraphStyle('Footer', fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    )
    story.append(rodape)
    
    doc.build(story)
    return saldo

print("✅ Função gerar_extrato_bancario() criada")

# COMMAND ----------

# DBTITLE 1,💼 Função: Gerar Comprovante de Renda PDF
def gerar_comprovante_renda(cliente_row, output_path):
    """
    Gera um comprovante de renda PDF
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Cabeçalho empresa
    titulo_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#2e7d32'),
        spaceAfter=10,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph("CONTABILIDADE EMPRESARIAL LTDA", titulo_style))
    story.append(Paragraph("CNPJ: 88.999.777/0001-55 - CRC: SP-123456/O", ParagraphStyle('subtitle', fontSize=10, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("COMPROVANTE DE FATURAMENTO ANUAL", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))
    
    # Dados da empresa
    story.append(Paragraph("<b>DADOS DA EMPRESA</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    dados_func = f"""
        <b>Razão Social:</b> {cliente_row['nome']}<br/>
        <b>CNPJ:</b> {cliente_row['cnpj']}<br/>
        <b>Porte:</b> {cliente_row['porte']}<br/>
        <b>Setor:</b> {cliente_row['setor']}<br/>
        <b>Período de Referência:</b> Janeiro a Dezembro de {datetime.now().year - 1}
    """
    story.append(Paragraph(dados_func, styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))
    
    # Faturamento
    story.append(Paragraph("<b>FATURAMENTO ANUAL</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    receita_bruta = cliente_row['receita_anual']
    impostos = receita_bruta * 0.20
    receita_liquida = receita_bruta - impostos
    
    remuneracao_data = [
        ['Descrição', 'Valor'],
        ['Receita Bruta', f'R$ {receita_bruta:,.2f}'],
        ['(-) Impostos e Taxas', f'R$ -{impostos:,.2f}'],
        ['<b>RECEITA LÍQUIDA</b>', f'<b>R$ {receita_liquida:,.2f}</b>'],
        ['Média Mensal', f'R$ {receita_liquida / 12:,.2f}']
    ]
    
    table = Table(remuneracao_data, colWidths=[3*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e7d32')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -2), colors.white),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    
    story.append(Spacer(1, 0.5*inch))
    
    # Declaração
    declaracao = f"""
        Declaramos, para os devidos fins, que a empresa acima identificada apresentou 
        faturamento anual líquido de <b>R$ {receita_liquida:,.2f}</b> no período de 
        referência, devidamente registrado em livros contábeis.
    """
    story.append(Paragraph(declaracao, styles['BodyText']))
    
    story.append(Spacer(1, 0.5*inch))
    
    # Data e assinatura
    story.append(Paragraph(f"São Paulo, {datetime.now().strftime('%d de %B de %Y')}", 
                          ParagraphStyle('date', alignment=TA_RIGHT)))
    story.append(Spacer(1, 0.5*inch))
    
    story.append(Paragraph("_" * 50, ParagraphStyle('signature', alignment=TA_CENTER)))
    story.append(Paragraph("Contador Responsável", 
                          ParagraphStyle('dept', fontSize=10, alignment=TA_CENTER)))
    story.append(Paragraph("Contabilidade Empresarial LTDA - CRC SP-123456/O", 
                          ParagraphStyle('company', fontSize=10, alignment=TA_CENTER)))
    
    # Rodapé
    story.append(Spacer(1, 0.5*inch))
    rodape = Paragraph(
        f"<i>Documento emitido eletronicamente em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>",
        ParagraphStyle('Footer', fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    )
    story.append(rodape)
    
    doc.build(story)
    return receita_liquida

print("✅ Função gerar_comprovante_renda() criada")

# COMMAND ----------

# DBTITLE 1,🚀 Gerar Todos os Documentos
import os
from tqdm import tqdm

print("🚀 GERANDO DOCUMENTOS PDF\n")
print("="*80)

base_path = "/Volumes/credit_risk/documentos/documentos_credito"

# Limitar para primeiros 20 clientes para exemplo
clientes_sample = clientes_pd.head(20)

documentos_gerados = {
    'contratos': [],
    'extratos': [],
    'comprovantes': []
}

for idx, cliente in tqdm(clientes_sample.iterrows(), total=len(clientes_sample), desc="Gerando documentos"):
    cliente_id = cliente['id_cliente']
    
    # 1. Gerar Contrato
    contrato_path = f"{base_path}/contratos/contrato_{cliente_id}.pdf"
    try:
        num_contrato, valor, taxa, prazo = gerar_contrato_credito(cliente, contrato_path)
        documentos_gerados['contratos'].append({
            'id_cliente': cliente_id,
            'path': contrato_path,
            'numero': num_contrato,
            'valor': valor
        })
    except Exception as e:
        print(f"❌ Erro ao gerar contrato para cliente {cliente_id}: {e}")
    
    # 2. Gerar Extrato Bancário
    faturas_cliente = faturas_pd[faturas_pd['id_cliente'] == cliente_id] if 'id_cliente' in faturas_pd.columns else faturas_pd.head(3)
    pagamentos_cliente = pagamentos_pd[pagamentos_pd['id_cliente'] == cliente_id] if 'id_cliente' in pagamentos_pd.columns else pagamentos_pd.head(3)
    
    extrato_path = f"{base_path}/extratos_bancarios/extrato_{cliente_id}.pdf"
    try:
        saldo = gerar_extrato_bancario(cliente, faturas_cliente, pagamentos_cliente, extrato_path)
        documentos_gerados['extratos'].append({
            'id_cliente': cliente_id,
            'path': extrato_path,
            'saldo_final': saldo
        })
    except Exception as e:
        print(f"❌ Erro ao gerar extrato para cliente {cliente_id}: {e}")
    
    # 3. Gerar Comprovante de Renda
    comprovante_path = f"{base_path}/comprovantes_renda/comprovante_{cliente_id}.pdf"
    try:
        renda_total = gerar_comprovante_renda(cliente, comprovante_path)
        documentos_gerados['comprovantes'].append({
            'id_cliente': cliente_id,
            'path': comprovante_path,
            'renda': renda_total
        })
    except Exception as e:
        print(f"❌ Erro ao gerar comprovante para cliente {cliente_id}: {e}")

print("\n" + "="*80)
print("\n✅ DOCUMENTOS GERADOS COM SUCESSO!\n")
print(f"📄 Contratos: {len(documentos_gerados['contratos'])} PDFs")
print(f"📊 Extratos: {len(documentos_gerados['extratos'])} PDFs")
print(f"💼 Comprovantes: {len(documentos_gerados['comprovantes'])} PDFs")
print(f"\n📁 Total: {sum(len(v) for v in documentos_gerados.values())} documentos")
print(f"\n📍 Localização: {base_path}")
print("\n" + "="*80)

# COMMAND ----------

# DBTITLE 1,📊 Criar Tabela de Metadados dos Documentos
# Criar tabela Delta com metadados dos documentos
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from datetime import datetime

print("📊 Criando tabela de metadados dos documentos...\n")

# Preparar dados de metadados
metadados_list = []

for doc_tipo, docs in documentos_gerados.items():
    for doc in docs:
        metadados_list.append({
            'documento_id': f"{doc_tipo}_{doc['id_cliente']}",
            'id_cliente': doc['id_cliente'],
            'tipo_documento': doc_tipo,
            'caminho_arquivo': doc['path'],
            'data_geracao': datetime.now(),
            'formato': 'PDF',
            'tamanho_kb': 0,  # Será preenchido depois
            'status': 'GERADO'
        })

# Converter para DataFrame Spark
metadados_df = spark.createDataFrame(metadados_list)

# Criar tabela Delta
table_name = "credit_risk.documentos.metadata_documentos"

metadados_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

print(f"✅ Tabela criada: {table_name}")
print(f"📝 {metadados_df.count()} registros inseridos\n")

# Mostrar amostra
print("📋 Amostra dos metadados:")
spark.table(table_name).show(10, truncate=False)

# COMMAND ----------

# DBTITLE 1,✅ Verificar Documentos Gerados
# Listar documentos gerados
print("📂 ESTRUTURA DE DOCUMENTOS GERADOS\n")
print("="*80)

for folder in ['contratos', 'extratos_bancarios', 'comprovantes_renda']:
    folder_path = f"{base_path}/{folder}"
    try:
        files = dbutils.fs.ls(folder_path)
        print(f"\n📁 {folder.upper()}/")
        print(f"   Total: {len(files)} arquivos")
        
        # Mostrar primeiros 3 arquivos
        for file in files[:3]:
            size_kb = file.size / 1024
            print(f"   • {file.name} ({size_kb:.1f} KB)")
        
        if len(files) > 3:
            print(f"   ... e mais {len(files) - 3} arquivos")
    except Exception as e:
        print(f"\n❌ Erro ao listar {folder}: {e}")

print("\n" + "="*80)
print("\n✅ DOCUMENTOS PRONTOS PARA VETORIZAÇÃO!")
print("\n📍 Próximo passo: Notebook 05_VECTOR_SEARCH_SETUP")
print("   → Extrair texto dos PDFs")
print("   → Gerar embeddings")
print("   → Criar índice vetorial no Databricks Vector Search")
print("\n" + "="*80)

# COMMAND ----------

