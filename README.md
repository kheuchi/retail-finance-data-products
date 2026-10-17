# Retail Finance Data Products

**Contents:** [TL;DR](#tldr) · [Architecture](#architecture) · [Where the data comes from](#where-the-data-comes-from) · [How it flows](#how-it-flows) · [Layout](#layout) · [Run it](#run-it) · [Status](#status) · [How it was built](#how-it-was-built)

Synthetic accounting data for a fictional large retailer, and the Databricks pipelines
that turn it into governed finance tables. Part of the
[retail finance platform](https://github.com/kheuchi/retail-finance-platform-control-plane).
Inventory: [`cmdb.yml`](cmdb.yml).

## TL;DR

| Question | Answer |
|---|---|
| What is here? | The data layer: a seeded generator, the Bronze and Silver jobs, their tests, and the bundle that deploys them |
| Where does data come from? | Generated in code (40 stores, 21 months, 3 planted frauds); only ECB FX rates are real |
| Where does it run? | Databricks job clusters in the private workspace, no internet egress |
| How does it ship? | PR → lint, tests (incl. local Spark), bundle validate → merge → gated deploy |
| Status | Bronze ✅ · Silver ✅ · Reconciliation ✅ (4/4 fake journals) · Gold ✅ · quality gates and lineage evidence |

## Architecture

> **TL;DR:** files → volume → Bronze → Silver → Gold, all under Unity Catalog.

![Data platform LLD](https://raw.githubusercontent.com/kheuchi/retail-finance-platform-control-plane/main/docs/architecture/lld-data-platform.png)

Explained in [LLD 3 · Data platform](https://github.com/kheuchi/retail-finance-platform-control-plane/blob/main/docs/architecture/lld-data-platform.md);
the whole platform in the [HLD](https://github.com/kheuchi/retail-finance-platform-control-plane/blob/main/docs/architecture/hld.md).

## Where the data comes from

> **TL;DR:** generated, realistic, with known answers. Detail: [`cmdb.yml`](cmdb.yml) → `sources`, `anomalies`

| Kind | Sources |
|---|---|
| Generated | Stores, products, cashiers, POS sales, refunds, general ledger, budget |
| Real | ECB exchange rates (snapshot in the repo), public holidays (computed) |
| Planted anomalies | A1 refund fraud by one cashier · A2 discount creep at one store · A3 fake manual revenue journals |

No real people, customers or card data. The planted anomalies are the answer key for
the reconciliation and the models.

## How it flows

> **TL;DR:** three jobs, deployed as one bundle. Detail: [`cmdb.yml`](cmdb.yml) → `pipeline`

| Job | Steps | Output |
|---|---|---|
| `generate_and_ingest` | Generate CSV files → Auto Loader | `finance.bronze.*` (text, untouched) |
| `transform_silver` | Type, check rules, dedup, convert to EUR | `finance.silver.*`, `silver.quarantine`, `silver.fx_daily` |
| `build_gold` | Quality gate → reconcile GL vs POS → finance tables → quality gate + lineage | `finance.gold.*`, `ops.dq_results`, `ops.lineage_evidence`, `ops.detection_scores` |

## Layout

> **TL;DR:** one folder per pipeline stage; the folder name says what it does; tests mirror the code.

```
retail-finance-data-products/
├── databricks.yml                 bundle: target workspace, run-as, variables
├── databricks_jobs/               one file per Databricks job, numbered in pipeline order
│   ├── 00_finance_pipeline        orchestrator: runs 02 → 03 → 04 → 05 on a schedule
│   ├── 01_generate_source_files   simulated SAP / POS / budget files into the landing volume
│   ├── 02_ingest_bronze           Auto Loader: files as delivered → Bronze
│   ├── 03_build_silver            typed, checked, quarantined → Silver
│   ├── 04_reconcile_and_build_gold quality gate → GL vs POS reconciliation → Gold → quality gate
│   ├── 05_train_ml_models         fraud detectors, forecast, PSI drift (+ the MLflow experiment)
│   └── 06_publish_agent_tools     the agent's Unity Catalog SQL functions
├── src/retail_finance_data/       the wheel installed on every job cluster
│   ├── source_simulator/          synthetic data model, holidays, ECB rates snapshot
│   ├── bronze/                    ingest.py
│   ├── silver/                    build.py (per-source rules)
│   ├── gold/                      reconcile_gl_pos.py, build_tables.py
│   ├── data_quality/              quality_gate.py (checks, certification, lineage)
│   ├── ml/                        train_models.py, psi_drift.py
│   ├── agent_tools/               functions.sql + publish.py
│   └── ops/                       config_drift.py (deployed jobs vs Git)
├── agents/month_end_close/        the month-end agents (Deep Agents)
│   ├── month_end_agent/           supervisor, sub-agents, tools; google_agent_runtime.py adapter
│   ├── deploy_google.py, run_google.py   deploy to / run on Google Agent Runtime
│   ├── Dockerfile                 image for AgentCore Runtime (designed target)
│   └── requirements-agentcore.txt, requirements-google.txt
├── tests/                         mirrors src/ (local Spark, Java 17); small_dataset.py is shared
└── cmdb.yml                       inventory: sources, anomalies, pipeline, tests, toolchain, status
```

Job entry points (`pyproject.toml` → `[project.scripts]`) keep their names, so the job files only
point at a script name, never at a module path.

## Run it

> **TL;DR:** tests run anywhere with Java 17; deploys only from Actions. Detail: [`cmdb.yml`](cmdb.yml) → `toolchain`

```bash
pip install -e ".[dev]" && pytest                                # tests incl. local Spark (needs Java 17)
python -m retail_finance_data.source_simulator.write_landing_files --out /tmp/landing  # full dataset (~2 min, ~480 MB)
```

To deploy: **Actions → Deploy to Databricks** → type `deploy`, pick a job to run
(`generate_and_ingest` or `transform_silver`).

## Status

> **TL;DR:** the whole medallion is live; ML is next. Detail: [`cmdb.yml`](cmdb.yml) → `status`

| Layer | State |
|---|---|
| Bronze | ✅ 2026-09-26: 8 tables, 4.2m rows, every count equal to the generator |
| Silver | ✅ 2026-09-27: 8 tables, 0 quarantined, 0 duplicates, every count equal to Bronze (~11 min on one node) |
| Gold | ✅ 2026-09-27: reconciliation 4/4, 0 false alarms; 4 finance tables; 2025 net sales EUR 38.88m, margin 34.31% |

## How it was built

> **TL;DR:** the why and the hard parts live in the control-plane stories.

| Story | Covers |
|---|---|
| [3.1 Synthetic accounting data](https://github.com/kheuchi/retail-finance-platform-control-plane/blob/main/docs/stories/3.1-synthetic-accounting-data.md) | Why synthetic, how the books balance, the anomalies |
| [3.2 Catalog and bundle deploy](https://github.com/kheuchi/retail-finance-platform-control-plane/blob/main/docs/stories/3.2-catalog-and-bundle-deploy.md) | Bundle, service principal folder, wheel path |
| [3.3 First job in the private VPC](https://github.com/kheuchi/retail-finance-platform-control-plane/blob/main/docs/stories/3.3-first-job-in-the-private-vpc.md) | Four failed runs and the blocked port |
| [4.1 Silver tables](https://github.com/kheuchi/retail-finance-platform-control-plane/blob/main/docs/stories/4.1-silver-tables.md) | How Spark cleans the data, and the first slow run |
| [4.2 GL vs POS reconciliation](https://github.com/kheuchi/retail-finance-platform-control-plane/blob/main/docs/stories/4.2-gl-pos-reconciliation.md) | Catching the fake journals by amount, not label |
| [4.3 Gold finance tables](https://github.com/kheuchi/retail-finance-platform-control-plane/blob/main/docs/stories/4.3-gold-finance-tables.md) | Definitions, access allow-list, frauds found unprompted |
| [4.4 Quality and lineage](https://github.com/kheuchi/retail-finance-platform-control-plane/blob/main/docs/stories/4.4-quality-and-lineage.md) | Quality gates and the trail from a number to its file |
