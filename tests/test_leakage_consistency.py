"""Teste de regressão para vazamento (leakage) de dados nos notebooks de modelagem.

Não depende de Spark/Databricks — os notebooks (`# Databricks notebook source`) usam
`dbutils`/`spark`, que não existem fora de um workspace, então este teste faz análise
estática (AST) do código-fonte em vez de importar os arquivos. Ele garante que a lista de
colunas de leakage conhecida (usada para enviesar a geração sintética ou para construir o
target) é sempre excluída de X antes do treino, nos 4 notebooks que treinam modelos sobre
`gold.features_ml`.

Este teste existe porque uma dessas listas (em `05_mlops/01_mlops_pipeline.py`) já divergiu
das outras 3 e treinava com leakage real — ver histórico do commit que corrigiu isso.
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Colunas que SEMPRE precisam estar excluídas de X antes do treino, em qualquer notebook que
# treine sobre gold.features_ml — cada uma é leakage por um motivo diferente (ver comentários
# nos próprios notebooks): categoria_risco enviesou a geração sintética original,
# data_cadastro é uma string crua, taxa_inadimplencia é usada para derivar o target.
REQUIRED_EXCLUDED_COLUMNS = {"categoria_risco", "data_cadastro", "taxa_inadimplencia"}

# (caminho relativo, nome da variável que guarda a lista de exclusão)
FILES_TO_CHECK = [
    ("04_modeling/01_modelo_classificacao_risco.py", "cols_to_drop"),
    ("04_modeling/02_modelo_regressao.py", "cols_to_drop"),
    ("04_modeling/04_automl_lightgbm_comparacao.py", "cols_to_drop"),
    ("05_mlops/01_mlops_pipeline.py", "exclude_cols"),
]


def _strip_notebook_magics(source: str) -> str:
    """Comenta linhas `%pip install ...` (comandos mágicos do Databricks), que não são
    Python válido e quebrariam o ast.parse. O resto do arquivo (incluindo `# MAGIC` e
    `# COMMAND ----------`, que já são comentários) passa direto."""
    lines = []
    for line in source.split("\n"):
        if line.strip().startswith("%pip"):
            lines.append("# " + line)
        else:
            lines.append(line)
    return "\n".join(lines)


def _find_string_list_assignments(tree: ast.AST, var_name: str) -> list[list[str]]:
    """Encontra toda atribuição `var_name = [...]` (em qualquer nível de aninhamento —
    módulo, dentro de função, dentro de if) cujo valor seja uma lista literal de strings, e
    retorna o conteúdo de cada uma encontrada."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if var_name not in targets:
            continue
        if not isinstance(node.value, ast.List):
            continue
        strings = [
            elt.value for elt in node.value.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        ]
        if strings:
            found.append(strings)
    return found


def test_leakage_columns_excluded_in_all_modeling_notebooks():
    failures = []

    for rel_path, var_name in FILES_TO_CHECK:
        file_path = REPO_ROOT / rel_path
        assert file_path.exists(), f"Arquivo esperado não encontrado: {rel_path}"

        source = _strip_notebook_magics(file_path.read_text(encoding="utf-8"))
        tree = ast.parse(source, filename=str(file_path))

        assignments = _find_string_list_assignments(tree, var_name)
        assert assignments, (
            f"{rel_path}: não encontrei nenhuma atribuição `{var_name} = [...]` de strings "
            f"literais — o teste precisa ser atualizado se a forma de excluir colunas mudou."
        )

        # Se houver mais de uma atribuição (ex: um default dentro de uma função), todas
        # precisam conter as colunas obrigatórias — não basta uma delas estar certa.
        for cols in assignments:
            missing = REQUIRED_EXCLUDED_COLUMNS - set(cols)
            if missing:
                failures.append(f"{rel_path} ({var_name}={cols}): faltam {sorted(missing)}")

    assert not failures, "Leakage não excluído corretamente:\n" + "\n".join(failures)
