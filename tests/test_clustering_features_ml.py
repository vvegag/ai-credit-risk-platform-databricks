"""Testes unitários pra lógica de clustering
(`03_feature_engineering/04_clustering_features_ml.py`), Fase C do roadmap técnico.

Usa a fixture `spark` (SparkSession local, `conftest.py`) pra rodar o pipeline PySpark ML
de verdade (`VectorAssembler` → `StandardScaler` → `KMeans` → `ClusteringEvaluator`) sobre
datasets sintéticos pequenos, em vez de só análise estática. As funções testadas
(`preparar_features_escaladas`, `treinar_kmeans_e_avaliar_silhouette`) são carregadas direto
do notebook via `load_notebook_functions` (AST) — não duplicamos a lógica aqui, então uma
mudança real no notebook é pega automaticamente pelo teste.
"""
import random

from pyspark.sql import Row

from conftest import load_notebook_functions

NOTEBOOK = "03_feature_engineering/04_clustering_features_ml.py"

_funcs = load_notebook_functions(
    NOTEBOOK, ["preparar_features_escaladas", "treinar_kmeans_e_avaliar_silhouette"]
)
preparar_features_escaladas = _funcs["preparar_features_escaladas"]
treinar_kmeans_e_avaliar_silhouette = _funcs["treinar_kmeans_e_avaliar_silhouette"]

FEATURE_COLS = [
    "total_faturado_90d",
    "count_faturas_total",
    "taxa_pagamento",
    "taxa_inadimplencia",
    "recency_dias",
    "rfm_score"
]


def _cliente(id_cliente, total_faturado_90d, count_faturas_total, taxa_pagamento,
             taxa_inadimplencia, recency_dias, rfm_score):
    return Row(
        id_cliente=id_cliente,
        total_faturado_90d=total_faturado_90d,
        count_faturas_total=count_faturas_total,
        taxa_pagamento=taxa_pagamento,
        taxa_inadimplencia=taxa_inadimplencia,
        recency_dias=recency_dias,
        rfm_score=rfm_score
    )


def test_silhouette_score_e_alto_para_clusters_bem_separados(spark):
    # Dois grupos de clientes bem distantes entre si em todas as features: "bons pagadores"
    # (faturamento alto, recency baixo) e "inadimplentes" (faturamento baixo, recency alto).
    clientes = (
        [_cliente(i, 10000.0, 20, 0.95, 0.02, 5, 5) for i in range(10)]
        + [_cliente(100 + i, 100.0, 1, 0.10, 0.90, 300, 1) for i in range(10)]
    )
    df = spark.createDataFrame(clientes)

    df_scaled = preparar_features_escaladas(df, FEATURE_COLS)
    _, _, silhouette = treinar_kmeans_e_avaliar_silhouette(df_scaled, k=2, seed=42)

    assert -1.0 <= silhouette <= 1.0
    assert silhouette > 0.7


def test_silhouette_score_e_baixo_para_dataset_homogeneo(spark):
    # Clientes praticamente idênticos com pequeno ruído independente em cada feature (sem
    # nenhum eixo dominante de variação): forçar k=2 não deveria achar separação real.
    rng = random.Random(7)
    clientes = [
        _cliente(
            i,
            5000.0 + rng.uniform(-50, 50),
            10 + rng.randint(-1, 1),
            0.5 + rng.uniform(-0.02, 0.02),
            0.5 + rng.uniform(-0.02, 0.02),
            100 + rng.randint(-5, 5),
            3
        )
        for i in range(20)
    ]
    df = spark.createDataFrame(clientes)

    df_scaled = preparar_features_escaladas(df, FEATURE_COLS)
    _, _, silhouette = treinar_kmeans_e_avaliar_silhouette(df_scaled, k=2, seed=42)

    assert -1.0 <= silhouette <= 1.0
    assert silhouette < 0.5


def test_preparar_features_escaladas_preenche_nulos_com_zero(spark):
    df = spark.createDataFrame(
        [_cliente(1, None, None, None, None, None, None), _cliente(2, 100.0, 5, 0.8, 0.1, 30, 4)]
    )

    df_scaled = preparar_features_escaladas(df, FEATURE_COLS)
    linha_cliente_1 = df_scaled.filter("id_cliente = 1").collect()[0]

    assert linha_cliente_1["features_raw"][0] == 0.0


def test_treinar_kmeans_atribui_todos_os_clientes_a_um_cluster_valido(spark):
    clientes = [_cliente(i, 1000.0 * i, i, 0.5, 0.3, 10 * i, (i % 5) + 1) for i in range(1, 21)]
    df = spark.createDataFrame(clientes)

    df_scaled = preparar_features_escaladas(df, FEATURE_COLS)
    _, df_clustered, _ = treinar_kmeans_e_avaliar_silhouette(df_scaled, k=4, seed=42)

    clusters = {row["cluster"] for row in df_clustered.collect()}
    assert clusters.issubset({0, 1, 2, 3})
    assert df_clustered.count() == 20
