"""Fixtures compartilhadas para testes que precisam de Spark/dbutils de verdade (não apenas
análise estática AST como em `test_leakage_consistency.py`).

`dbutils` e a variável global `spark` só existem dentro de um notebook rodando num cluster
Databricks — fora daquele ambiente, os notebooks nem importam. Essas fixtures reproduzem o
mínimo necessário localmente para testar a *lógica* de transformação (funções PySpark puras
extraídas dos notebooks, ou trechos equivalentes), sem precisar de um workspace real:

- `spark`: uma `SparkSession` local (`local[1]`), criada uma vez por sessão de teste.
- `mock_dbutils`: um mock mínimo de `dbutils.widgets`, cobrindo o único padrão usado em todos
  os notebooks deste projeto — `dbutils.widgets.text(nome, default, label)` seguido de
  `dbutils.widgets.get(nome)` (confirmado varrendo todos os usos de `dbutils.widgets.` no
  repositório — nenhum notebook usa `dropdown`/`combobox`/`multiselect`).
"""
import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("credit-risk-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield session
    session.stop()


class MockDbutilsWidgets:
    """Mock mínimo de `dbutils.widgets`, suficiente para o padrão `text()` + `get()` usado em
    todos os notebooks (não implementa `dropdown`/`combobox`/`multiselect` — nenhum notebook
    deste projeto usa esses tipos de widget hoje)."""

    def __init__(self):
        self._values = {}

    def text(self, name, defaultValue="", label=None):
        self._values.setdefault(name, defaultValue)

    def get(self, name):
        if name not in self._values:
            raise KeyError(f"Widget '{name}' não foi registrado (chame .text() antes de .get())")
        return self._values[name]

    def set(self, name, value):
        """Não existe em `dbutils.widgets` real — atalho só para os testes sobrescreverem o
        valor de um widget (ex: apontar `catalog` para um catálogo de teste)."""
        self._values[name] = value


class MockDbutils:
    def __init__(self):
        self.widgets = MockDbutilsWidgets()


@pytest.fixture
def mock_dbutils():
    return MockDbutils()
