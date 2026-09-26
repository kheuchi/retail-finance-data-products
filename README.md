# Retail Finance Data Products

**Contents:** [What this is](#what-this-is) · [Where the data comes from](#where-the-data-comes-from) · [How it flows](#how-it-flows) · [Run it](#run-it) · [Status](#status)

Synthetic accounting data for a fictional large retailer, and the Databricks
pipelines that turn it into governed finance tables. Part of the
[retail finance platform](https://github.com/kheuchi/retail-finance-platform-infra).

## What this is

The data layer of the platform: generate raw data, ingest it into **Bronze**, clean it
into **Silver**, and publish certified **Gold** finance tables that the ML models
and the AI agent read.

## Where the data comes from

> Detail: [`cmdb.yml`](cmdb.yml) → `sources`

Almost everything is **generated** by code in this repo: stores, products, POS sales,
refunds, general ledger, budget. The only real inputs are ECB exchange rates and
public-holiday dates. No real people, customers or card data.

Generated data has one big advantage: we plant **known anomalies** (e.g. refund
fraud at one store), so the models can be scored against a real answer.

## How it flows

> Detail: [`cmdb.yml`](cmdb.yml) → `pipeline`

```text
generator job ─► UC volume (raw files) ─► Auto Loader ─► Bronze ─► Silver ─► Gold
```

Everything runs inside the private Databricks workspace (no internet egress),
deployed from GitHub Actions as a Databricks Asset Bundle.

## Run it

> Detail: [`cmdb.yml`](cmdb.yml) → `toolchain`

```bash
pip install -e ".[dev]" && pytest                       # tests incl. local Spark (needs Java 17), no Databricks needed
python -m retail_finance_data.jobs.generate --out /tmp/landing   # full dataset locally (~2 min, ~480 MB)
```

To deploy: **Actions → Deploy to Databricks** → type `deploy`, pick a job to run (`generate_and_ingest` or `transform_silver`).

## Status

> Detail: [`cmdb.yml`](cmdb.yml) → `status`

Bronze loaded on 2026-09-26: 8 tables, 4.2m rows, every count matching the generator. Silver job built and tested (story 4.1); first run pending.
