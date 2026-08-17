# 🗺️ Roadmap Técnico — Melhorias Contínuas

Este documento é a fonte de verdade do que falta melhorar no projeto, além do que já está
descrito como "future work" no [README](../README.md#-roadmap). É atualizado por um processo
diário automatizado (ver `.github/workflows/` e a rotina agendada) que pega o próximo item não
concluído, implementa, valida, e marca como feito.

## Regras para quem (humano ou agente) for trabalhar num item daqui

1. **Um item por vez.** Pegue o primeiro item não marcado (`[ ]`) na ordem em que aparecem.
2. **Não quebre o que já está validado.** Mudanças devem ser aditivas sempre que possível —
   não reescreva lógica já testada só por elegância.
3. **Valide antes de commitar**:
   - `python -c "import ast; ..."` em todo `.py` de notebook editado, ignorando linhas
     `%pip install` (comentando-as antes do parse — elas não são Python válido fora do
     Databricks).
   - Scanner de células: nenhuma célula que comece com `# MAGIC %md` pode conter código
     executável depois (vira markdown morto no Databricks — bug real já encontrado neste
     projeto).
   - `pytest tests/` deve passar limpo.
4. **Se um item exigir algo que você não tem** (privilégio de admin no workspace, conta
   Databricks estável, decisão de produto que só o dono do projeto pode tomar) — **não
   force**. Marque o item como `[blocked: <motivo>]` em vez de `[x]`, não implemente uma
   versão degradada só pra marcar como feito, e pare por aí nesse dia.
5. **Commit direto em `main`** (fluxo combinado com o dono do projeto) — mensagem descritiva
   em português, no mesmo estilo dos commits já existentes no histórico.
6. Ao concluir um item, marque `[x]` aqui **e** faça o commit dessa mudança **junto** com o
   código (mesmo commit, ou o imediatamente seguinte) — o checklist não pode ficar
   desatualizado em relação ao código real.

---

## ✅ Fase A — Tuning de hiperparâmetros + validação cruzada (concluída)

- [x] Optuna + StratifiedKFold no classificador (`04_modeling/01_modelo_classificacao_risco.py`)
- [x] Optuna + KFold no regressor (`04_modeling/02_modelo_regressao.py`)

## Fase B — Completar governança do Model Registry

Hoje só `04_modeling/01_modelo_classificacao_risco.py` registra no UC Model Registry com
alias Champion/Challenger (ver `MODEL_REGISTRY_NAME`, `MlflowClient.set_registered_model_alias`,
padrão de fallback pra Volume UC se o registro falhar — reusar esse mesmo padrão, não inventar
um novo).

- [x] Registrar o regressor (`04_modeling/02_modelo_regressao.py`) no UC Model Registry como
      `credit_risk_regressor`, com fallback pra Volume UC.
- [x] Registrar o modelo de forecast (`04_modeling/03_modelo_forecast_cashflow.py`) no UC
      Model Registry como `credit_risk_forecast` (usar `mlflow.prophet.log_model` já
      existente, só adicionar `registered_model_name`).
- [ ] Registrar o LightGBM (`04_modeling/04_automl_lightgbm_comparacao.py`) — avaliar se faz
      sentido registrar como modelo próprio ou só documentar que ele é exploratório/
      comparativo e não deveria ir pro registry (decisão a documentar, não só código).

## Fase C — Testes mais profundos (além do estático)

`tests/test_leakage_consistency.py` hoje só faz análise estática (AST), sem depender de
Spark/Databricks. Testar a lógica de transformação de dados de verdade precisa mockar
`dbutils`/`spark`.

- [ ] Configurar uma `SparkSession` local (`SparkSession.builder.master("local[1]")`) +
      mock simples de `dbutils.widgets` utilizável nos testes.
- [ ] Testes unitários pra lógica de `03_feature_engineering/03_feature_store_rfm.py` (scoring
      RFM) — começar por aqui, não pelos 20 notebooks de uma vez.
- [ ] Testes unitários pra lógica de `03_feature_engineering/04_clustering_features_ml.py`
      (validar que o silhouette score é calculado corretamente sobre um dataset sintético
      pequeno).

## Fase D — Maturidade de infraestrutura (Asset Bundle)

- [ ] Adicionar targets `staging`/`prod` em `databricks.yml` (mesmo Job, catálogo
      parametrizado via `var.catalog` já existente — ex: `credit_risk_staging`,
      `credit_risk_prod`). Documentar o fluxo de promoção em `09_docs/GUIA_USO.md`.
- [ ] Inference Tables no Model Serving endpoint
      (`05_mlops/02_model_serving_endpoint.py`) — log automático de request/response.
- [blocked: requer privilégio de admin real no workspace] `run_as` de service principal no
      Job — não implementar até haver uma conta com esse privilégio confirmado.

## Fase E — Governança/compliance mais fundo

- [ ] Coluna de "Sensitivity"/classificação em `09_docs/DICIONARIO_DADOS.md` (marcar `cnpj`
      como PII) — puramente documentação.
- [blocked: requer privilégio de admin real no workspace] Masking de `cnpj` via Unity
      Catalog column mask — o desenho já está documentado em `09_docs/ARQUITETURA.md`, seção
      "Sensitive data"; não implementar até ter workspace com privilégio confirmado.
- [ ] Nota de governança de custo (dimensionamento de cluster/warehouse, orçamento) em
      `09_docs/ARQUITETURA.md`.
- [ ] Lineage real: descrever passo a passo como gerar/ler o UC Lineage Graph pra
      `gold.features_ml` no workspace, substituindo a frase solta que existe hoje em
      `ARQUITETURA.md`.

## Fase F — Data quality como código

- [ ] `StructType` explícito nos pontos de ingestão que hoje inferem schema (auditar
      `02_ingestion/*.py` e `03_feature_engineering/*.py`, listar quais ainda inferem).
- [ ] `ALTER TABLE ... ADD CONSTRAINT` (`NOT NULL`, `CHECK`) nas tabelas Gold mais críticas
      (`features_ml`, `model_predictions`) — hoje zero constraints declaradas.

---

## Sequência recomendada

A (feita) → B → C → F → D/E — as duas últimas dependem de privilégios de workspace que hoje
não são garantidos numa conta trial/acadêmica; os itens `[blocked: ...]` ficam documentados
como desenho, não implementados, até isso mudar.
