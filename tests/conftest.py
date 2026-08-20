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
import ast
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def load_notebook_functions(rel_path, function_names):
    """Extrai definições de função puras de um notebook Databricks (`.py`) sem executar o
    resto do script — que depende de `dbutils.widgets`/`spark.table` reais e não roda fora
    de um workspace. Usa AST pra isolar só os `FunctionDef` pedidos (e os `import`s do
    módulo, dos quais eles dependem) em vez de duplicar a lógica no teste, que divergiria do
    notebook real com o tempo (o mesmo risco que motivou `test_leakage_consistency.py`)."""
    file_path = REPO_ROOT / rel_path
    source = file_path.read_text(encoding="utf-8")
    lines = [
        ("# " + line if line.strip().startswith("%pip") else line)
        for line in source.split("\n")
    ]
    tree = ast.parse("\n".join(lines), filename=str(file_path))

    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    wanted = set(function_names)
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    missing = wanted - {node.name for node in functions}
    assert not missing, f"{rel_path}: funções não encontradas: {sorted(missing)}"

    module = ast.Module(body=imports + functions, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {}
    exec(compile(module, filename=str(file_path), mode="exec"), namespace)
    return {name: namespace[name] for name in function_names}
