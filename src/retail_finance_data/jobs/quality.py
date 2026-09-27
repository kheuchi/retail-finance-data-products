"""Job: data quality gate and lineage evidence (story 4.4).

Runs twice in ``build_gold``: ``--stage silver`` before anything is built from Silver, and
``--stage gold`` after the Gold tables are written. Every check result is recorded in
``ops.dq_results``; a failed **critical** check fails the job (after recording), a failed
**warning** is recorded and printed but lets the run continue.

Row-level rules already live in Silver (types, keys, references, quarantine). The checks here
are the ones a single row cannot answer: journals that do not balance, keys that repeat,
volumes that collapse, Gold that no longer adds up, lineage that is missing.

The gold stage also writes ``ops.lineage_evidence``: for each Gold table, the Silver and
Bronze tables it was built from (Unity Catalog lineage) and a source file behind the Bronze
table, so an auditor can walk from a number back to the file it came from.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

CRITICAL, WARNING = "critical", "warning"


@dataclass(frozen=True)
class Check:
    name: str
    severity: str
    stage: str
    run: Callable[[dict], tuple[int, str]]  # tables -> (violations, detail)


def _silver_keys_unique(t):
    pos = t["pos_sales"]
    dup = pos.count() - pos.select("transaction_id", "line_no").distinct().count()
    return dup, "repeated (transaction_id, line_no) in silver.pos_sales"


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

    bad = t["pos_sales"].where((F.col("net_amount") <= 0) | (F.col("net_amount") > 50000) | (F.col("quantity") > 500))
    return bad.count(), "sales lines with net <= 0, net > 50,000 or quantity > 500"


def _sales_have_masters(t):
    pos = t["pos_sales"]
    no_store = pos.join(t["stores"], "store_id", "left_anti").count()
    missing = no_store + pos.join(t["products"], "sku", "left_anti").count()
    return missing, "sales lines whose store or product is missing"


def _quarantine_empty(t):
    return t["quarantine"].count(), "rows waiting in silver.quarantine"


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
    off = abs(gold - silver) > 1
    return int(off), f"gold.daily_revenue net sales {gold} vs silver {round(silver, 2)}"


CHECKS = [
    Check("silver_keys_unique", CRITICAL, "silver", _silver_keys_unique),
    Check("journals_balance", CRITICAL, "silver", _journals_balance),
    Check("amounts_in_range", CRITICAL, "silver", _amounts_in_range),
    Check("sales_have_masters", CRITICAL, "silver", _sales_have_masters),
    Check("quarantine_empty", WARNING, "silver", _quarantine_empty),
    Check("volume_collapse", WARNING, "silver", _volume_collapse),
    Check("gold_adds_up", CRITICAL, "gold", _gold_adds_up),
]


def run_checks(tables: dict, stage: str) -> list[dict]:
    results = []
    for c in (c for c in CHECKS if c.stage == stage):
        violations, detail = c.run(tables)
        results.append(
            {
                "check": c.name,
                "severity": c.severity,
                "stage": stage,
                "violations": int(violations),
                "passed": violations == 0,
                "detail": detail,
            }
        )
    return results


def enforce(results: list[dict]) -> None:
    """Fail on any critical check, with a message a person can act on."""
    failed = [r for r in results if r["severity"] == CRITICAL and not r["passed"]]
    if failed:
        lines = "; ".join(f"{r['check']}: {r['violations']} ({r['detail']})" for r in failed)
        raise RuntimeError(f"Quality gate failed, {len(failed)} critical check(s): {lines}")


LINEAGE_SQL = """
WITH edges AS (
  SELECT DISTINCT source_table_full_name AS src, target_table_full_name AS tgt
  FROM system.access.table_lineage
  WHERE target_table_catalog = '{catalog}' AND source_table_full_name IS NOT NULL
)
SELECT g.tgt AS gold_table, g.src AS silver_table, b.src AS bronze_table
FROM edges g
JOIN edges b ON b.tgt = g.src
WHERE g.tgt LIKE '{catalog}.gold.%' AND g.src LIKE '{catalog}.silver.%' AND b.src LIKE '{catalog}.bronze.%'
"""


def lineage_evidence(spark, catalog: str):
    """Gold -> Silver -> Bronze -> file, from Unity Catalog lineage plus Bronze's _source_file."""
    from pyspark.sql import functions as F

    chains = spark.sql(LINEAGE_SQL.format(catalog=catalog))
    files = {}
    for (bronze,) in chains.select("bronze_table").distinct().collect():
        files[bronze] = spark.table(bronze).agg(F.min("_source_file")).first()[0]
    sample = spark.createDataFrame(list(files.items()) or [("", "")], "bronze_table string, sample_source_file string")
    return chains.join(sample, "bronze_table", "left").select(
        "gold_table", "silver_table", "bronze_table", "sample_source_file"
    )


GOLD_TABLES = ("daily_revenue", "margin", "refunds", "budget_variance", "recon_gl_pos", "recon_exceptions")


def main() -> None:
    import uuid

    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="finance")
    p.add_argument("--stage", choices=["silver", "gold"], required=True)
    a = p.parse_args()
    spark = SparkSession.builder.getOrCreate()
    c = a.catalog
    names = ("pos_sales", "gl_journal", "stores", "products", "quarantine")
    tables = {t: spark.table(f"{c}.silver.{t}") for t in names}
    if a.stage == "gold":
        tables["daily_revenue"] = spark.table(f"{c}.gold.daily_revenue")

    results = run_checks(tables, a.stage)
    if a.stage == "gold":
        try:
            evidence = lineage_evidence(spark, c).cache()
            evidence.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{c}.ops.lineage_evidence")
            traced = {r[0].split(".")[-1] for r in evidence.select("gold_table").distinct().collect()}
            missing = [t for t in GOLD_TABLES if t not in traced]
            evidence.orderBy("gold_table", "silver_table").show(40, truncate=False)
            detail = f"gold tables without traced lineage: {missing}" if missing else "every gold table traced"
            results.append(
                {
                    "check": "lineage_traced",
                    "severity": WARNING,
                    "stage": "gold",
                    "violations": len(missing),
                    "passed": not missing,
                    "detail": detail,
                }
            )
        except Exception as e:  # lineage is evidence; its absence is reported, not fatal
            results.append(
                {
                    "check": "lineage_traced",
                    "severity": WARNING,
                    "stage": "gold",
                    "violations": 1,
                    "passed": False,
                    "detail": f"lineage not readable: {type(e).__name__}: {str(e)[:200]}",
                }
            )

    run_id = str(uuid.uuid4())
    (
        spark.createDataFrame(results)
        .select("check", "severity", "stage", "violations", "passed", "detail")
        .withColumn("run_id", F.lit(run_id))
        .withColumn("checked_at", F.current_timestamp())
        .write.mode("append")
        .saveAsTable(f"{c}.ops.dq_results")
    )
    for r in results:
        mark = "PASS" if r["passed"] else ("FAIL" if r["severity"] == CRITICAL else "WARN")
        print(f"{mark:<4} {r['severity']:<8} {r['check']:<20} {r['violations']:>6}  {r['detail']}", flush=True)
    enforce(results)


if __name__ == "__main__":
    main()
