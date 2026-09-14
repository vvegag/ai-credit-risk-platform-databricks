# 🤝 Contributing

Thanks for considering a contribution to **AI Credit Risk Platform**! This document formalizes
the flow already summarized in the [README](README.md#-contributing).

## Flow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to your branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request against `main`

CI (`.github/workflows/tests.yml`) runs the test suite automatically on every push/PR — see
[Running tests locally](#-running-tests-locally) to check it before opening the PR.

## 📓 Notebook conventions

Notebooks are plain `.py` files in Databricks' notebook format:

- First line: `# Databricks notebook source`
- Cells separated by `# COMMAND ----------`
- Optional cell title: `# DBTITLE 1,Cell title`
- Markdown cells: `# MAGIC %md` followed by `# MAGIC` lines — never mix executable code into a
  markdown cell (it becomes dead code that silently never runs in Databricks)

Follow the **PySpark-first convention** described in
[`09_docs/ARQUITETURA.md`](09_docs/ARQUITETURA.md#-pyspark-first-convention): all ETL, joins,
aggregations, and filtering over Bronze/Silver/Gold tables use the Spark DataFrame API or Spark
SQL, not pandas. `.toPandas()` is only acceptable in the specific cases already documented there.

## 🌐 Language conventions

- Code and comments inside notebooks/scripts: **Portuguese** (matches the existing codebase)
- Public documentation (README, `09_docs/*.md`, this file): **English**

## ✅ Running tests locally

```bash
pip install pytest pyspark numpy
pytest tests/ 10_rag_agent/tests/test_config.py -v
```

This is the same command CI runs. It doesn't replace validation against a real Databricks
workspace — the notebooks still require Databricks to actually run — but it catches structural
regressions (e.g. target leakage) before merge.

## 🗺️ Looking for something to work on?

[`09_docs/ROADMAP_TECNICO.md`](09_docs/ROADMAP_TECNICO.md) tracks pending improvements, with the
rules for picking up an item at the top of the file.

## 📄 License

By contributing, you agree that your contributions will be licensed under the project's
[MIT License](LICENSE).
