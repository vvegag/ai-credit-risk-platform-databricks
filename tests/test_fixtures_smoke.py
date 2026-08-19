"""Smoke test pras fixtures de `conftest.py` (SparkSession local + mock de dbutils.widgets).

Não testa lógica de nenhum notebook — só garante que as fixtures em si funcionam, já que
`04_clustering_features_ml.py` e `03_feature_store_rfm.py` (Fase C do roadmap) vão depender
delas para testes unitários de verdade.
"""
from pyspark.sql import Row


def test_spark_fixture_cria_sessao_local_funcional(spark):
    df = spark.createDataFrame([Row(id_cliente=1, valor=10.0), Row(id_cliente=2, valor=20.0)])
    assert df.count() == 2
    assert sorted(row["id_cliente"] for row in df.collect()) == [1, 2]


def test_mock_dbutils_reproduz_padrao_text_get(mock_dbutils):
    mock_dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
    assert mock_dbutils.widgets.get("catalog") == "credit_risk"


def test_mock_dbutils_permite_sobrescrever_valor_nos_testes(mock_dbutils):
    mock_dbutils.widgets.text("catalog", "credit_risk", "Nome do catálogo")
    mock_dbutils.widgets.set("catalog", "credit_risk_teste")
    assert mock_dbutils.widgets.get("catalog") == "credit_risk_teste"


def test_mock_dbutils_get_sem_text_previo_levanta_erro(mock_dbutils):
    try:
        mock_dbutils.widgets.get("inexistente")
        assert False, "esperava KeyError"
    except KeyError:
        pass
