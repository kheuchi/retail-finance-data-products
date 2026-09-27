"""Job: reconcile general-ledger revenue with point-of-sale sales (story 4.2).

POS (the tills) and GL (the accounting books) are two independent records of the same
revenue. Each day's sales are booked to revenue (account 4000) in one journal, so per
store and day the two must agree. A revenue journal with no sales behind it inflates the
books: the classic way to fake results.

The check compares amounts, never labels. On each store-day, the revenue journal whose
amount equals the POS total is the one that explains the sales; any other revenue journal
on a day that does not reconcile is a suspect, whatever its ``source`` says. Every
difference is reported, attributed to a journal or not.

Exceptions are business findings, not job failures: the job fails only if it cannot run.
"""

from __future__ import annotations

import argparse

TOLERANCE = 0.01  # EUR: the books balance by construction, so only rounding is allowed
ANOMALY = "A3_unsupported_manual_revenue"


def pos_daily(pos_sales, fx_daily):
    """POS net sales in EUR per store and day.

    Lines are converted unrounded and the day is rounded once, the way the daily journal
    is booked; summing the 4-decimal line amounts instead would drift by cents.
    """
    from pyspark.sql import functions as F

    rate = fx_daily.select(F.col("rate_date").alias("business_date"), "currency", "units_per_eur")
    return (
        pos_sales.join(F.broadcast(rate), ["business_date", "currency"], "left")
        .groupBy("store_id", "business_date")
        .agg(F.round(F.sum(F.col("net_amount") / F.col("units_per_eur")), 2).cast("decimal(18,2)").alias("pos_net_eur"))
    )


def revenue_journals(gl_journal):
    """Revenue booked per journal (account 4000: credits minus debits)."""
    from pyspark.sql import functions as F

    return (
        gl_journal.where(F.col("account_code") == "4000")
        .groupBy("journal_id", "store_id", F.col("posting_date").alias("business_date"), "source", "description")
        .agg(F.sum(F.col("credit_eur") - F.col("debit_eur")).cast("decimal(18,2)").alias("amount_eur"))
    )


def reconcile(pos_sales, gl_journal, fx_daily):
    """Return (daily, monthly, exceptions) DataFrames."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    pos = pos_daily(pos_sales, fx_daily)
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

    # On each store-day, the journal closest to the POS total explains the sales if it matches it.
    by_day = Window.partitionBy("store_id", "business_date").orderBy(
        F.abs(F.col("amount_eur") - F.col("pos_net_eur")), "journal_id"
    )
    ranked = journals.join(daily, ["store_id", "business_date"]).withColumn("rn", F.row_number().over(by_day))
    explains = (F.col("rn") == 1) & (F.abs(F.col("amount_eur") - F.col("pos_net_eur")) <= TOLERANCE)
    suspects = ranked.where((F.col("status") == "difference") & ~explains).select(
        "store_id", "business_date", "journal_id", "source", "description", "amount_eur"
    )
    attributed = suspects.groupBy("store_id", "business_date").agg(F.sum("amount_eur").alias("attributed_eur"))

    exceptions = (
        daily.where(F.col("status") == "difference")
        .join(suspects, ["store_id", "business_date"], "left")
        .join(attributed, ["store_id", "business_date"], "left")
        .withColumn(
            "unexplained_eur",
            (F.col("difference_eur") - F.coalesce("attributed_eur", F.lit(0))).cast("decimal(18,2)"),
        )
        .withColumn(
            "reason",
            F.when(F.col("journal_id").isNotNull(), "no matching POS sales").otherwise(
                "difference not attributable to a journal"
            ),
        )
        .select(
            "store_id",
            "business_date",
            "journal_id",
            "source",
            "description",
            "amount_eur",
            "gl_revenue_eur",
            "pos_net_eur",
            "difference_eur",
            "unexplained_eur",
            "reason",
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


def main() -> None:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="finance")
    a = p.parse_args()
    spark = SparkSession.builder.getOrCreate()
    silver, gold = f"{a.catalog}.silver", f"{a.catalog}.gold"

    daily, monthly, exceptions = reconcile(
        spark.table(f"{silver}.pos_sales"), spark.table(f"{silver}.gl_journal"), spark.table(f"{silver}.fx_daily")
    )
    for name, df in [("recon_gl_pos", daily), ("recon_gl_pos_monthly", monthly), ("recon_exceptions", exceptions)]:
        df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{gold}.{name}")

    days = spark.table(f"{gold}.recon_gl_pos")
    n_days = days.count()
    if n_days == 0:
        raise RuntimeError("Reconciliation produced no store-days: Silver is empty or unreadable.")
    n_diff = days.where("status = 'difference'").count()
    exc = spark.table(f"{gold}.recon_exceptions")
    print(f"store-days {n_days:,} | matched {n_days - n_diff:,} | with difference {n_diff:,}", flush=True)
    exc.orderBy("business_date").show(50, truncate=False)

    truth = f"/Volumes/{a.catalog}/raw/landing/_ground_truth/planted_records.csv"
    try:
        planted = {
            r[0]
            for r in spark.read.option("header", "true")
            .csv(truth)
            .where(F.col("anomaly") == ANOMALY)
            .select("entity_id")
            .collect()
        }
    except Exception as e:  # the answer key exists only for synthetic data
        print(f"No ground truth scored ({type(e).__name__}).", flush=True)
        return
    found = {r[0] for r in exc.where("journal_id IS NOT NULL").select("journal_id").collect()}
    s = score(found, planted)
    print(f"A3 detection: {s}", flush=True)
    (
        spark.createDataFrame([("gl_pos_reconciliation", ANOMALY, *s.values())])
        .toDF("control", "anomaly", "planted", "found", "missed", "false_positives")
        .withColumn("scored_at", F.current_timestamp())
        .write.mode("append")
        .saveAsTable(f"{a.catalog}.ops.detection_scores")
    )


if __name__ == "__main__":
    main()
