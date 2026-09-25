# Retail Finance Data Products

**Contents:** [What this is](#what-this-is) · [Where the data comes from](#where-the-data-comes-from) · [How it flows](#how-it-flows) · [Status](#status)

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

## Status

> Detail: [`cmdb.yml`](cmdb.yml) → `status`

Repository created 2026-09-25. Generator and Bronze ingestion are next.
