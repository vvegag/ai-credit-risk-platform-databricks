# Databricks notebook source
# MAGIC %md
# MAGIC # Geração de PDFs de Notas Fiscais
# MAGIC 
# MAGIC **Objetivo**: Criar 10 PDFs de exemplo para validação RAG
# MAGIC 
# MAGIC **Bibliotecas**: reportlab (instalada no cluster)

# COMMAND ----------

# Instalar reportlab se necessário
%pip install reportlab --quiet

# COMMAND ----------

from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from datetime import datetime, timedelta
import random
import os

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Função para Gerar Nota Fiscal

def gerar_nota_fiscal_pdf(
    filename,
    numero_nf,
    cliente_nome,
    cliente_cnpj,
    valor_total,
    data_emissao,
    itens
):
    """
    Gera um PDF de nota fiscal com layout simples
    
    Args:
        filename: Nome do arquivo PDF
        numero_nf: Número da nota fiscal
        cliente_nome: Nome do cliente
        cliente_cnpj: CNPJ do cliente
        valor_total: Valor total da NF
        data_emissao: Data de emissão
        itens: Lista de dicionários com {descricao, quantidade, valor_unitario}
    """
    
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    
    # Margens
    margin = 50
    y = height - margin
    
    # Título
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin, y, "NOTA FISCAL DE SERVIÇOS")
    y -= 30
    
    # Número da NF
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"NF Nº: {numero_nf}")
    y -= 20
    
    c.setFont("Helvetica", 12)
    c.drawString(margin, y, f"Data de Emissão: {data_emissao}")
    y -= 40
    
    # Dados do Cliente
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "DADOS DO CLIENTE")
    y -= 20
    
    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Razão Social: {cliente_nome}")
    y -= 15
    c.drawString(margin, y, f"CNPJ: {cliente_cnpj}")
    y -= 30
    
    # Itens/Serviços
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "SERVIÇOS PRESTADOS")
    y -= 20
    
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, "Descrição")
    c.drawString(margin + 250, y, "Qtd")
    c.drawString(margin + 300, y, "Valor Unit.")
    c.drawString(margin + 380, y, "Total")
    y -= 15
    
    # Linha separadora
    c.line(margin, y, width - margin, y)
    y -= 10
    
    # Itens
    c.setFont("Helvetica", 9)
    for item in itens:
        desc = item['descricao']
        qtd = item['quantidade']
        valor_unit = item['valor_unitario']
        total_item = qtd * valor_unit
        
        c.drawString(margin, y, desc[:35])  # Limitar tamanho
        c.drawString(margin + 250, y, str(qtd))
        c.drawString(margin + 300, y, f"R$ {valor_unit:,.2f}")
        c.drawString(margin + 380, y, f"R$ {total_item:,.2f}")
        y -= 12
    
    y -= 10
    c.line(margin, y, width - margin, y)
    y -= 20
    
    # Total
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin + 300, y, "VALOR TOTAL:")
    c.drawString(margin + 380, y, f"R$ {valor_total:,.2f}")
    y -= 40
    
    # Observações
    c.setFont("Helvetica", 9)
    c.drawString(margin, y, "Observações: Pagamento em até 30 dias após emissão.")
    y -= 12
    c.drawString(margin, y, "Favor efetuar pagamento via transferência bancária.")
    
    # Rodapé
    c.setFont("Helvetica", 8)
    c.drawString(margin, 50, f"Documento gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    c.drawString(margin, 35, "Este documento é válido como comprovante de prestação de serviços")
    
    c.save()
    print(f"✅ PDF gerado: {filename}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Gerar 10 PDFs de Exemplo

print("📄 Gerando 10 PDFs de notas fiscais...\n")

# Diretório de saída
output_dir = "/Workspace/Users/valdomirovega@hotmail.com/ai-credit-risk-platform-databricks/02_ingestion/sample_data/notas_fiscais"

# Dados de exemplo
clientes_exemplo = [
    {"nome": "Alpha Tech Corp", "cnpj": "12.345.678/0001-90"},
    {"nome": "Beta Industries SA", "cnpj": "23.456.789/0001-01"},
    {"nome": "Gamma Solutions Ltd", "cnpj": "34.567.890/0001-12"},
    {"nome": "Delta Systems Group", "cnpj": "45.678.901/0001-23"},
    {"nome": "Omega Technologies", "cnpj": "56.789.012/0001-34"},
    {"nome": "Sigma Ventures SA", "cnpj": "67.890.123/0001-45"},
    {"nome": "Theta Digital Corp", "cnpj": "78.901.234/0001-56"},
    {"nome": "Zeta Global Ltd", "cnpj": "89.012.345/0001-67"},
    {"nome": "Nova Strategic SA", "cnpj": "90.123.456/0001-78"},
    {"nome": "Prime Elite Group", "cnpj": "01.234.567/0001-89"}
]

servicos_exemplo = [
    "Consultoria Estratégica",
    "Desenvolvimento de Software",
    "Manutenção de Sistemas",
    "Treinamento Corporativo",
    "Suporte Técnico",
    "Análise de Dados",
    "Auditoria de Processos",
    "Gestão de Projetos"
]

# Gerar PDFs
for i, cliente in enumerate(clientes_exemplo, 1):
    # Dados da NF
    numero_nf = f"2026{i:04d}"
    data_emissao = (datetime.now() - timedelta(days=random.randint(1, 90))).strftime("%d/%m/%Y")
    
    # Gerar 2-4 itens de serviço
    num_itens = random.randint(2, 4)
    itens = []
    valor_total = 0
    
    for _ in range(num_itens):
        servico = random.choice(servicos_exemplo)
        qtd = random.randint(1, 10)
        valor_unit = random.choice([500, 800, 1200, 1500, 2000, 3000, 5000])
        
        itens.append({
            "descricao": servico,
            "quantidade": qtd,
            "valor_unitario": valor_unit
        })
        
        valor_total += qtd * valor_unit
    
    # Gerar PDF
    filename = f"{output_dir}/nota_fiscal_{i:03d}.pdf"
    
    gerar_nota_fiscal_pdf(
        filename=filename,
        numero_nf=numero_nf,
        cliente_nome=cliente["nome"],
        cliente_cnpj=cliente["cnpj"],
        valor_total=valor_total,
        data_emissao=data_emissao,
        itens=itens
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validar PDFs Gerados

print("\n🔍 Validando PDFs gerados...\n")

import os

pdf_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
print(f"✅ {len(pdf_files)} PDFs gerados com sucesso!")
print(f"📁 Localização: {output_dir}")
print("\nArquivos:")
for pdf in sorted(pdf_files):
    size = os.path.getsize(f"{output_dir}/{pdf}")
    print(f"  • {pdf} ({size:,} bytes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ PDFs Gerados com Sucesso!
# MAGIC 
# MAGIC **Criados**: 10 PDFs de notas fiscais
# MAGIC **Localização**: `/02_ingestion/sample_data/notas_fiscais/`
# MAGIC **Uso**: Validação RAG em `06_rag_validation/`
# MAGIC 
# MAGIC **Próximos passos**:
# MAGIC 1. Criar CSVs de exemplo
# MAGIC 2. Configurar Auto Loader
# MAGIC 3. Testar pipeline RAG

print("\n" + "="*60)
print("✅ PDFS DE NOTAS FISCAIS GERADOS!")
print("="*60)


