"""Job: reconcile general-ledger revenue with point-of-sale sales (story 4.2).

POS (the tills) and GL (the accounting books) are two independent records of the same
revenue. Each day's sales are booked to revenue in one journal, so per store and day the
two must agree. A revenue journal with no sales behind it inflates the books: the classic
way to fake results.

Attribution uses amounts, not labels. On each store-day with a difference:

- no POS sales at all: every revenue journal is unsupported;
- exactly one journal equals the POS total: it explains the sales, the others are unsupported;
- several journals equal the POS total: all of them are reported as a duplicate posting;
- none equals it: the day is reported once, not attributable to a journal.

Every difference is reported. Exceptions are business findings, not job failures. The job
fails before writing anything if its inputs are incomplete (no sales, no revenue lines,
missing FX rates, or rows still in Silver quarantine): a reconciliation on partial data
would accuse the wrong journals.

Scope: revenue means ``REVENUE_ACCOUNTS``. Fake revenue booked to another account is out
of reach of this control (recorded as an accepted limit in the story).
"""

from __future__ import annotations

import argparse

TOLERANCE = 0.01  # EUR: the books balance by construction, so only rounding is allowed
REVENUE_ACCOUNTS = ("4000",)
ANOMALY = "A3_unsupported_manual_revenue"

UNSUPPORTED = "no matching POS sales"
DUPLICATE = "duplicate posting: several journals match the POS total"
UNATTRIBUTED = "difference not attributable to a journal"


def pos_lines_eur(pos_sales, fx_daily):
    from pyspark.sql import functions as F

    rate = fx_daily.select(F.col("rate_date").alias("business_date"), "currency", "units_per_eur")
    return pos_sales.join(F.broadcast(rate), ["business_date", "currency"], "left")


def pos_daily(lines):
    """POS net sales in EUR per store and day.

    Lines are converted unrounded and the day is rounded once, the way the daily journal is
    booked; summing the 4-decimal line amounts instead would drift by cents.
    """
    from pyspark.sql import functions as F

    return lines.groupBy("store_id", "business_date").agg(
        F.round(F.sum(F.col("net_amount") / F.col("units_per_eur")), 2).cast("decimal(18,2)").alias("pos_net_eur")
    )


def revenue_journals(gl_journal):
    """Revenue booked per journal: credits minus debits on the revenue accounts."""
    from pyspark.sql import functions as F

    return (
        gl_journal.where(F.col("account_code").isin(*REVENUE_ACCOUNTS))
        .groupBy("journal_id", "store_id", F.col("posting_date").alias("business_date"), "source", "description")
        .agg(F.sum(F.col("credit_eur") - F.col("debit_eur")).cast("decimal(18,2)").alias("amount_eur"))
    )


def check_inputs(pos_sales, gl_journal, fx_daily, quarantined: int = 0) -> None:
    """Refuse to reconcile partial data. Raises before anything is written."""
    from pyspark.sql import functions as F

    problems = []
    if pos_sales.limit(1).count() == 0:
        problems.append("no POS sales")
    if gl_journal.where(F.col("account_code").isin(*REVENUE_ACCOUNTS)).limit(1).count() == 0:
        problems.append("no revenue lines in the GL")
    missing = pos_lines_eur(pos_sales, fx_daily).where(F.col("units_per_eur").isNull()).count()
    if missing:
        problems.append(f"{missing} POS lines without an FX rate")
    if quarantined:
        problems.append(f"{quarantined} pos_sales/gl_journal rows still in silver.quarantine")
    if problems:
        raise RuntimeError("Reconciliation refused, inputs incomplete: " + "; ".join(problems))


def reconcile(pos_sales, gl_journal, fx_daily):
    """Return (daily, monthly, exceptions) DataFrames."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    pos = pos_daily(pos_lines_eur(pos_sales, fx_daily))
    journals = revenue_journals(gl_journal)
    gl = journals.groupBy("store_id", "business_date").agg(F.sum("amount_eur").alias("gl_revenue_eur"))

    daily = (
        gl.join(pos, ["store_id", "business_date"], "full")
        .fillna(0, ["gl_revenue_eur", "pos_net_eur"])
        .withColumn("difference_eur", (F.col("gl_revenue_eur") - F.col("pos_net_eur")).cast("decimal(18,2)"))
        .withColumn("status", F.when(F.abs("difference_eur") <= TOLERANCE, "matched").otherwise("difference"))
    )

    monthly = (
        daily.groupBy("store_id", F.date_format("business_date", "yyyy-MM").alias("month"))
        .agg(
            F.sum("gl_revenue_eur").alias("gl_revenue_eur"),
            F.sum("pos_net_eur").alias("pos_net_eur"),
            F.sum("difference_eur").alias("difference_eur"),
            F.sum(F.when(F.col("status") == "difference", 1).otherwise(0)).alias("days_with_difference"),
        )
        .withColumn("status", F.when(F.col("days_with_difference") == 0, "matched").otherwise("difference"))
    )

    day = Window.partitionBy("store_id", "business_date")
    matches = F.abs(F.col("amount_eur") - F.col("pos_net_eur")) <= TOLERANCE
    tagged = (
        journals.join(daily.where(F.col("status") == "difference"), ["store_id", "business_date"])
        .withColumn("matches", matches)
        .withColumn("n_match", F.sum(F.col("matches").cast("int")).over(day))
        .withColumn(
            "reason",
            F.when(F.col("pos_net_eur") == 0, F.lit(UNSUPPORTED))
            .when((F.col("n_match") == 1) & ~F.col("matches"), F.lit(UNSUPPORTED))
            .when((F.col("n_match") > 1) & F.col("matches"), F.lit(DUPLICATE))
            .when((F.col("n_match") > 1) & ~F.col("matches"), F.lit(UNSUPPORTED)),
        )
    )
    suspects = tagged.where(F.col("reason").isNotNull()).select(
        "store_id", "business_date", "journal_id", "source", "description", "amount_eur", "reason"
    )
    attributed = suspects.groupBy("store_id", "business_date").agg(F.sum("amount_eur").alias("attributed_eur"))

    exceptions = (
        daily.where(F.col("status") == "difference")
        .join(suspects, ["store_id", "business_date"], "left")
        .join(attributed, ["store_id", "business_date"], "left")
        .select(
            "store_id",
            "business_date",
            "journal_id",
            "source",
            "description",
            "amount_eur",
            F.coalesce("reason", F.lit(UNATTRIBUTED)).alias("reason"),
            # Day totals repeat on each row of the day: never sum them over the table.
            F.col("gl_revenue_eur").alias("day_gl_revenue_eur"),
            F.col("pos_net_eur").alias("day_pos_net_eur"),
            F.col("difference_eur").alias("day_difference_eur"),
            (F.col("difference_eur") - F.coalesce("attributed_eur", F.lit(0)))
            .cast("decimal(18,2)")
            .alias("day_unexplained_eur"),
        )
    )
    return daily, monthly, exceptions


def score(found: set[str], planted: set[str]) -> dict:
    """Detection score against the generator's answer key."""
    return {
        "planted": len(planted),
        "found": len(found & planted),
        "missed": len(planted - found),
        "false_positives": len(found - planted),
    }


def read_answer_key(spark, path: str) -> set[str] | None:
    """Planted A3 journal ids, or None when there is no answer key (real data has none)."""
    from pyspark.errors import AnalysisException
    from pyspark.sql import functions as F

    try:
        truth = spark.read.option("header", "true").csv(path)
    except AnalysisException as e:
        if "PATH_NOT_FOUND" in str(e):
            return None
        raise
    planted = {r[0] for r in truth.where(F.col("anomaly") == ANOMALY).select("entity_id").collect()}
    if not planted:
        raise RuntimeError(f"Answer key {path} has no {ANOMALY} rows: cannot score.")
    return planted


def main() -> None:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="finance")
    a = p.parse_args()
    spark = SparkSession.builder.getOrCreate()
    silver, gold = f"{a.catalog}.silver", f"{a.catalog}.gold"
    pos_sales, gl_journal, fx_daily = (spark.table(f"{silver}.{t}") for t in ("pos_sales", "gl_journal", "fx_daily"))

    quarantined = spark.table(f"{silver}.quarantine").where(F.col("source").isin("pos_sales", "gl_journal")).count()
    check_inputs(pos_sales, gl_journal, fx_daily, quarantined)

    # Compute and check everything first, then write, so the three tables come from one run.
    daily, monthly, exceptions = (df.cache() for df in reconcile(pos_sales, gl_journal, fx_daily))
    n_days, n_diff, n_exc = daily.count(), daily.where("status = 'difference'").count(), exceptions.count()
    monthly.count()
    # Write from a fresh, uncached plan so Unity Catalog records lineage to the Silver sources.
    fresh = dict(
        zip(
            ("recon_gl_pos", "recon_gl_pos_monthly", "recon_exceptions"),
            reconcile(pos_sales, gl_journal, fx_daily),
            strict=True,
        )
    )
    for name, df in fresh.items():
        df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{gold}.{name}")

    print(f"store-days {n_days:,} | matched {n_days - n_diff:,} | with difference {n_diff:,} | exceptions {n_exc}")
    exceptions.orderBy("business_date", "journal_id").show(50, truncate=False)

    planted = read_answer_key(spark, f"/Volumes/{a.catalog}/raw/landing/_ground_truth/planted_records.csv")
    if planted is None:
        print("No answer key: detection not scored (expected on real data).", flush=True)
        return
    found = {r[0] for r in exceptions.where("journal_id IS NOT NULL").select("journal_id").collect()}
    s = score(found, planted)
    print(f"A3 detection: {s}", flush=True)
    (
        spark.createDataFrame([("gl_pos_reconciliation", ANOMALY, *s.values())])
        .toDF("control", "anomaly", "planted", "found", "missed", "false_positives")
        .withColumn("scored_at", F.current_timestamp())
        .write.mode("append")  # a history: one row per run, timestamped
        .saveAsTable(f"{a.catalog}.ops.detection_scores")
    )


if __name__ == "__main__":
    main()
