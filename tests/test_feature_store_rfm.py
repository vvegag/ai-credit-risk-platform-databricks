"""Testes unitários pra lógica de RFM scoring
(`03_feature_engineering/03_feature_store_rfm.py`), Fase C do roadmap técnico.

Usa a fixture `spark` (SparkSession local, `conftest.py`) pra rodar as transformações
PySpark de verdade sobre um dataset sintético pequeno, em vez de só análise estática. As
funções testadas (`calcular_recency`, `aplicar_rfm_score`) são carregadas direto do notebook
via `load_notebook_functions` (AST) — não duplicamos a lógica aqui, então uma mudança real no
notebook é pega automaticamente pelo teste.
"""
from datetime import date, timedelta

from pyspark.sql import Row

from conftest import load_notebook_functions

NOTEBOOK = "03_feature_engineering/03_feature_store_rfm.py"

_funcs = load_notebook_functions(NOTEBOOK, ["calcular_recency", "aplicar_rfm_score"])
calcular_recency = _funcs["calcular_recency"]
aplicar_rfm_score = _funcs["aplicar_rfm_score"]


def test_calcular_recency_usa_a_fatura_mais_recente_por_cliente(spark):
    hoje = date.today()
    df_faturas = spark.createDataFrame([
        Row(id_cliente=1, data_emissao=hoje - timedelta(days=10)),
        Row(id_cliente=1, data_emissao=hoje - timedelta(days=40)),  # cliente 1: mais antiga, ignorada
        Row(id_cliente=2, data_emissao=hoje - timedelta(days=200)),
    ])

    resultado = {row["id_cliente"]: row["recency_dias"] for row in calcular_recency(df_faturas).collect()}

    assert resultado == {1: 10, 2: 200}


def test_aplicar_rfm_score_bucketiza_recency_em_score_1_a_5(spark):
    df_rfm = spark.createDataFrame(
        [Row(recency_dias=v) for v in (0, 29, 30, 59, 60, 89, 90, 179, 180, 400)]
    )

    resultado = {row["recency_dias"]: row["rfm_score"] for row in aplicar_rfm_score(df_rfm).collect()}

    assert resultado == {
        0: 5, 29: 5,
        30: 4, 59: 4,
        60: 3, 89: 3,
        90: 2, 179: 2,
        180: 1, 400: 1,
    }


def test_aplicar_rfm_score_categoriza_a_partir_do_score(spark):
    df_rfm = spark.createDataFrame(
        [Row(recency_dias=v) for v in (10, 45, 75, 150)]
    )

    resultado = {
        row["recency_dias"]: row["categoria_rfm"]
        for row in aplicar_rfm_score(df_rfm).collect()
    }

    assert resultado == {
        10: "Premium",   # score 5
        45: "Premium",   # score 4
        75: "Regular",   # score 3
        150: "Em Risco",  # score 2
    }
