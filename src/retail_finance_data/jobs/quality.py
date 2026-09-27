"""Job: data quality gates, certification and lineage evidence (story 4.4).

Runs twice in ``build_gold``, with the Databricks run id so both halves belong to one run:

- ``--stage silver`` before anything is built: checks Silver, records the Delta version and
  row count of every Silver table it approved, and warns if a count dropped since last run;
- ``--stage gold`` after Gold is written: checks Gold, fails if Silver changed under the run,
  writes lineage evidence, and records whether this run's Gold is **certified**.

Every result goes to ``ops.dq_results``. A failed critical check fails the job after
recording; a warning is recorded and lets the run continue. Consumers (ML, the agent) read
``ops.gold_certification`` and use Gold only from a certified run.

Row-level rules live in Silver. The checks here are the ones a single row cannot answer.
``silver_keys_unique`` and ``sales_have_masters`` repeat Silver guarantees on purpose: they
are cheap tripwires that catch a future regression in the Silver code.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

CRITICAL, WARNING = "critical", "warning"
MONEY = ("pos_sales", "refunds", "gl_journal", "budget")
SNAPSHOT = ("pos_sales", "refunds", "gl_journal", "budget", "stores", "products", "fx_daily")
GOLD_TABLES = (
    "daily_revenue",
    "margin",
    "refunds",
    "budget_variance",
    "recon_gl_pos",
    "recon_gl_pos_monthly",
    "recon_exceptions",
)


@dataclass(frozen=True)
class Check:
    name: str
    severity: str
    stage: str
    run: Callable[[dict], tuple[int, str]]  # tables -> (violations, detail)


def _silver_keys_unique(t):
    pos = t["pos_sales"]
    dup = pos.count() - pos.select("transaction_id", "line_no").distinct().count()
    return dup, "tripwire: repeated (transaction_id, line_no) in silver.pos_sales"


def _journals_balance(t):
    from pyspark.sql import functions as F

    off = (
        t["gl_journal"]
        .groupBy("journal_id")
        .agg(F.sum("debit_eur").alias("dr"), F.sum("credit_eur").alias("cr"))
        .where(F.col("dr") != F.col("cr"))
    )
    n = off.count()
    sample = ", ".join(r[0] for r in off.limit(3).collect())
    return n, f"journals where debits != credits: {sample}" if n else "every journal balances"


def _amounts_in_range(t):
    from pyspark.sql import functions as F

    pos = t["pos_sales"].where(
        (F.col("net_amount_eur") <= 0) | (F.col("net_amount_eur") > 50000) | (F.col("quantity") > 500)
    )
    ref = t["refunds"].where((F.col("refund_amount_eur") <= 0) | (F.col("refund_amount_eur") > 10000))
    gl = t["gl_journal"].where((F.col("debit_eur") > 1_000_000) | (F.col("credit_eur") > 1_000_000))
    n_pos, n_ref, n_gl = pos.count(), ref.count(), gl.count()
    return n_pos + n_ref + n_gl, f"out of range (EUR): sales {n_pos}, refunds {n_ref}, ledger lines {n_gl}"


def _sales_have_masters(t):
    pos = t["pos_sales"]
    no_store = pos.join(t["stores"], "store_id", "left_anti").count()
    missing = no_store + pos.join(t["products"], "sku", "left_anti").count()
    return missing, "tripwire: sales lines whose store or product is missing"


def _quarantine(money: bool):
    def run(t):
        from pyspark.sql import functions as F

        in_money = F.col("source").isin(*MONEY)
        q = t["quarantine"].where(in_money if money else ~in_money)
        n = q.count()
        by = ", ".join(f"{r[0]} {r[1]}" for r in q.groupBy("source").count().collect())
        what = "money tables (Gold would be understated)" if money else "master tables"
        return n, f"rows in silver.quarantine from {what}: {by or 'none'}"

    return run


def _refund_within_sale(t):
    """A receipted refund cannot give back more than was paid for that product on that receipt.

    Compared in the till's own currency: in EUR, a CHF refund converted at a later day's rate
    can exceed the sale by a few cents without anything being wrong."""
    from pyspark.sql import functions as F

    paid = (
        t["pos_sales"]
        .groupBy(F.col("transaction_id").alias("original_transaction_id"), "sku")
        .agg(F.sum("gross_amount").alias("paid"))
    )
    over = (
        t["refunds"]
        .where("receipt_present")
        .groupBy("original_transaction_id", "sku")
        .agg(F.sum("refund_amount").alias("refunded"))
        .join(paid, ["original_transaction_id", "sku"])
        .where(F.col("refunded") > F.col("paid") + 0.01)
    )
    return over.count(), "receipted refunds larger than the sale they return"


def expected_open_days(stores, first, last):
    """(store_id, business_date) for every day a store should trade: online every day,
    physical stores Monday to Saturday except national public holidays."""
    from datetime import timedelta

    from retail_finance_data.calendar import public_holidays

    years = list(range(first.year, last.year + 1))
    hol = {c: public_holidays(c, years) for c in ("DE", "CH")}
    days = [first + timedelta(d) for d in range((last - first).days + 1)]
    rows = []
    for s in stores.select("store_id", "country", "format").collect():
        for d in days:
            if s["format"] == "online" or (d.isoweekday() != 7 and d not in hol[s["country"]]):
                rows.append((s["store_id"], d))
    return rows


def _missing_store_days(t):
    """An open day with no sales at all: a missing file, not a quiet day."""
    from pyspark.sql import functions as F

    pos = t["pos_sales"]
    first, last = pos.agg(F.min("business_date"), F.max("business_date")).first()
    expected = pos.sparkSession.createDataFrame(
        expected_open_days(t["stores"], first, last), "store_id string, business_date date"
    )
    sold = pos.select("store_id", "business_date").distinct()
    missing = expected.join(sold, ["store_id", "business_date"], "left_anti")
    n = missing.count()
    sample = ", ".join(f"{r[0]} {r[1]}" for r in missing.orderBy("business_date").limit(3).collect())
    return n, f"open store-days with no sales: {sample}" if n else "every open store-day has sales"


def _volume_collapse(t):
    """Store-days selling under a third of the store's usual volume for that weekday."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    daily = t["pos_sales"].groupBy("store_id", "business_date").agg(F.count("*").alias("lines"))
    usual = Window.partitionBy("store_id", F.dayofweek("business_date"))
    low = daily.withColumn("usual", F.percentile_approx("lines", 0.5).over(usual)).where(
        F.col("lines") < F.col("usual") / 3
    )
    n = low.count()
    sample = ", ".join(f"{r[0]} {r[1]}" for r in low.orderBy("business_date").limit(3).collect())
    return n, f"store-days under a third of usual volume: {sample}" if n else "no volume collapse"


def _gold_adds_up(t):
    from pyspark.sql import functions as F

    gold = t["daily_revenue"].agg(F.sum("net_sales_eur")).first()[0] or 0
    silver = t["pos_sales"].agg(F.sum("net_amount_eur")).first()[0] or 0
    return int(abs(gold - silver) > 1), f"gold.daily_revenue net sales {gold} vs silver {round(silver, 2)}"


CHECKS = [
    Check("silver_keys_unique", CRITICAL, "silver", _silver_keys_unique),
    Check("journals_balance", CRITICAL, "silver", _journals_balance),
    Check("amounts_in_range", CRITICAL, "silver", _amounts_in_range),
    Check("sales_have_masters", CRITICAL, "silver", _sales_have_masters),
    Check("quarantine_money", CRITICAL, "silver", _quarantine(money=True)),
    Check("quarantine_masters", WARNING, "silver", _quarantine(money=False)),
    Check("refund_within_sale", WARNING, "silver", _refund_within_sale),
    Check("missing_store_days", WARNING, "silver", _missing_store_days),
    Check("volume_collapse", WARNING, "silver", _volume_collapse),
    Check("gold_adds_up", CRITICAL, "gold", _gold_adds_up),
]


def result(check, severity, stage, violations, detail) -> dict:
    return {
        "check": check,
        "severity": severity,
        "stage": stage,
        "violations": int(violations),
        "passed": violations == 0,
        "detail": detail,
    }


def run_checks(tables: dict, stage: str) -> list[dict]:
    return [result(c.name, c.severity, stage, *c.run(tables)) for c in CHECKS if c.stage == stage]


def compare_snapshots(before: dict, now: dict, stage: str) -> dict:
    """Silver must not change between the silver gate and the gold gate of one run."""
    if not before:
        return result("silver_unchanged", CRITICAL, stage, 1, "no silver gate recorded for this run")
    changed = sorted(t for t in now if before.get(t) != now[t])
    detail = f"Silver changed during the run: {changed}" if changed else "Silver unchanged since the silver gate"
    return result("silver_unchanged", CRITICAL, stage, len(changed), detail)


def compare_counts(previous: dict, now: dict, stage: str, drop: float = 0.05) -> dict:
    """Warn when a Silver table lost more than ``drop`` of its rows since the previous run."""
    lost = [f"{t} {previous[t]:,} -> {n:,}" for t, n in now.items() if previous.get(t) and n < previous[t] * (1 - drop)]
    detail = f"row counts dropped more than {drop:.0%}: {lost}" if lost else "no row count dropped since last run"
    return result("row_count_drop", WARNING, stage, len(lost), detail)


def enforce(results: list[dict]) -> None:
    """Fail on any critical check, with a message a person can act on."""
    failed = [r for r in results if r["severity"] == CRITICAL and not r["passed"]]
    if failed:
        lines = "; ".join(f"{r['check']}: {r['violations']} ({r['detail']})" for r in failed)
        raise RuntimeError(f"Quality gate failed, {len(failed)} critical check(s): {lines}")


# ---------- lineage evidence ----------


def table_upstreams(api_get, table: str) -> list[tuple[str, list[int]]]:
    """Upstream tables of ``table`` and the ids of the jobs that wrote each link (UC lineage REST API)."""
    d = api_get("/api/2.0/lineage-tracking/table-lineage", {"table_name": table, "include_entity_lineage": "true"})
    out = []
    for u in d.get("upstreams", []):
        t = u.get("tableInfo") or {}
        if t.get("name"):
            name = f"{t['catalog_name']}.{t['schema_name']}.{t['name']}"
            out.append((name, [j.get("job_id") for j in u.get("jobInfos", [])]))
    return out


def lineage_chains(api_get, catalog: str, job_id: int | None) -> list[tuple[str, str, str]]:
    """(gold, silver, bronze) chains; the Gold links must have been written by this job."""
    chains = []
    for g in GOLD_TABLES:
        for silver, jobs in table_upstreams(api_get, f"{catalog}.gold.{g}"):
            if ".silver." not in silver or (job_id and job_id not in jobs):
                continue
            bronze = [b for b, _ in table_upstreams(api_get, silver) if ".bronze." in b] or [""]
            chains += [(f"{catalog}.gold.{g}", silver, b) for b in bronze]
    return chains


def row_trace(spark, catalog: str) -> dict:
    """Follow one Gold number to the files behind it: the store-day with the largest
    reconciliation difference -> its Silver sales lines -> their Bronze rows -> source files."""
    from pyspark.sql import functions as F

    top = (
        spark.table(f"{catalog}.gold.recon_gl_pos")
        .orderBy(F.desc(F.abs("difference_eur")))
        .select("store_id", "business_date", "pos_net_eur")
        .first()
    )
    store, day = top["store_id"], top["business_date"]
    silver = spark.table(f"{catalog}.silver.pos_sales").where(
        (F.col("store_id") == store) & (F.col("business_date") == day)
    )
    bronze = spark.table(f"{catalog}.bronze.pos_sales").where(
        (F.col("store_id") == store) & (F.col("business_date") == str(day))  # Bronze is text
    )
    return {
        "gold_key": f"recon_gl_pos {store} {day} pos_net_eur={top['pos_net_eur']}",
        "silver_rows": silver.count(),
        "bronze_rows": bronze.count(),
        "source_files": [r[0] for r in silver.select("_source_file").distinct().collect()],
    }


def lineage_result(spark, catalog: str, job_id: str | None) -> dict:
    """Write ops.lineage_evidence (table chains + one number traced to its files) and report
    gaps. Unity Catalog records lineage with a lag, so a gap warns; it does not fail."""
    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()

        def api_get(path, query):
            return w.api_client.do("GET", path, query=query)

        chains = lineage_chains(api_get, catalog, int(job_id) if job_id else None)
        trace = row_trace(spark, catalog)
        rows = [("table", g, s, b, "") for g, s, b in chains]
        rows += [
            ("row", trace["gold_key"], f"{trace['silver_rows']} silver rows", f"{trace['bronze_rows']} bronze rows", f)
            for f in trace["source_files"]
        ]
        (
            spark.createDataFrame(rows, "kind string, gold string, silver string, bronze string, source_file string")
            .write.mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(f"{catalog}.ops.lineage_evidence")
        )
        for r in rows:
            print("lineage", r, flush=True)
        traced = {g.split(".")[-1] for g, _, _ in chains}
        missing = [t for t in GOLD_TABLES if t not in traced]
        detail = f"not yet traced (lineage lag): {missing}" if missing else "every gold table traced to Bronze"
        return result("lineage_traced", WARNING, "gold", len(missing), detail)
    except Exception as e:
        return result("lineage_traced", WARNING, "gold", 1, f"lineage not readable: {type(e).__name__}: {str(e)[:200]}")


def main() -> None:
    import uuid

    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="finance")
    p.add_argument("--stage", choices=["silver", "gold"], required=True)
    p.add_argument("--run-id", default=None, help="Databricks job run id: ties both gates of a run together")
    p.add_argument("--job-id", default=None, help="Databricks job id: keeps only lineage written by this job")
    a = p.parse_args()
    spark = SparkSession.builder.getOrCreate()
    c, run_id, ops = a.catalog, a.run_id or str(uuid.uuid4()), f"{a.catalog}.ops"
    silver = {t: spark.table(f"{c}.silver.{t}") for t in (*SNAPSHOT, "quarantine")}
    has_snapshots = spark.catalog.tableExists(f"{ops}.silver_snapshot")

    def version(t):
        return spark.sql(f"DESCRIBE HISTORY {c}.silver.{t} LIMIT 1").first()["version"]

    if a.stage == "silver":
        results = run_checks(silver, "silver")
        snap = {t: (version(t), silver[t].count()) for t in SNAPSHOT}
        prev = {}
        if has_snapshots:
            last = spark.table(f"{ops}.silver_snapshot").agg(F.max("taken_at")).first()[0]
            prev = {
                r[0]: r[1]
                for r in spark.table(f"{ops}.silver_snapshot")
                .where(F.col("taken_at") == last)
                .select("table_name", "row_count")
                .collect()
            }
        results.append(compare_counts(prev, {t: n for t, (_, n) in snap.items()}, "silver"))
        (
            spark.createDataFrame(
                [(run_id, t, int(v), int(n)) for t, (v, n) in snap.items()],
                "run_id string, table_name string, delta_version long, row_count long",
            )
            .withColumn("taken_at", F.current_timestamp())
            .write.mode("append")
            .saveAsTable(f"{ops}.silver_snapshot")
        )
    else:
        results = run_checks({**silver, "daily_revenue": spark.table(f"{c}.gold.daily_revenue")}, "gold")
        before = {}
        if has_snapshots:
            before = {
                r[0]: r[1]
                for r in spark.table(f"{ops}.silver_snapshot")
                .where(F.col("run_id") == run_id)
                .select("table_name", "delta_version")
                .collect()
            }
        results.append(compare_snapshots(before, {t: version(t) for t in SNAPSHOT}, "gold"))
        results.append(lineage_result(spark, c, a.job_id))
        failed = [r["check"] for r in results if r["severity"] == CRITICAL and not r["passed"]]
        (
            spark.createDataFrame(
                [(run_id, not failed, ", ".join(failed))], "run_id string, certified boolean, failed string"
            )
            .withColumn("certified_at", F.current_timestamp())
            .write.mode("append")
            .saveAsTable(f"{ops}.gold_certification")
        )
        print(f"Gold {'CERTIFIED' if not failed else 'NOT certified'} for run {run_id}", flush=True)

    (
        spark.createDataFrame(results)
        .select("check", "severity", "stage", "violations", "passed", "detail")
        .withColumn("run_id", F.lit(run_id))
        .withColumn("checked_at", F.current_timestamp())
        .write.mode("append")
        .saveAsTable(f"{ops}.dq_results")
    )
    for r in results:
        mark = "PASS" if r["passed"] else ("FAIL" if r["severity"] == CRITICAL else "WARN")
        print(f"{mark:<4} {r['severity']:<8} {r['check']:<20} {r['violations']:>6}  {r['detail']}", flush=True)
    enforce(results)


if __name__ == "__main__":
    main()
