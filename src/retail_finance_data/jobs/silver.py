"""Job: build Silver tables from Bronze.

Silver keeps Bronze's promises. Every column gets its real type, each business key
appears once, amounts gain an EUR value at the ECB rate of the day, and any row that
breaks a rule is set aside in ``silver.quarantine`` with its reasons. Nothing is fixed
silently and nothing disappears: for every source,

    bronze rows = silver rows + quarantined rows + removed duplicates

and the job fails if that does not hold.

Silver is rebuilt in full from Bronze on every run, so re-running it gives the same
result. At a few million rows that is cheaper than the bookkeeping an incremental
merge needs; revisit if Bronze grows by orders of magnitude.

The transformation (``apply_spec``) is pure DataFrame code, tested locally with
PySpark; ``main`` only reads and writes tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

MONEY = "DECIMAL(18,2)"
EUR = "DECIMAL(18,4)"  # converted amounts keep 4 decimals so sums do not drift by cents
RATE = "DECIMAL(18,6)"


@dataclass(frozen=True)
class Spec:
    columns: dict[str, str]  # name -> Spark SQL type, in output order
    key: tuple[str, ...]  # business key: one row per key in Silver
    optional: frozenset[str] = frozenset()  # columns allowed to be empty
    rules: dict[str, str] = field(default_factory=dict)  # name -> SQL predicate that must hold
    refs: dict[str, tuple[str, str]] = field(default_factory=dict)  # column -> (source, column) it must exist in
    eur: dict[str, tuple[str, str, str]] = field(default_factory=dict)  # new column -> (amount, currency, date)


SPECS: dict[str, Spec] = {
    "stores": Spec(
        columns={
            "store_id": "STRING",
            "store_name": "STRING",
            "country": "STRING",
            "currency": "STRING",
            "region": "STRING",
            "format": "STRING",
            "size_factor": "DOUBLE",
            "open_date": "DATE",
        },
        key=("store_id",),
        rules={
            "country_known": "country IN ('DE', 'CH')",
            "currency_matches_country": "(country, currency) IN (('DE', 'EUR'), ('CH', 'CHF'))",
            "format_known": "format IN ('hypermarket', 'supermarket', 'express', 'online')",
            "size_positive": "size_factor > 0",
        },
    ),
    "products": Spec(
        columns={
            "sku": "STRING",
            "category": "STRING",
            "reduced_vat": "BOOLEAN",
            "list_price_eur": MONEY,
            "unit_cost_eur": MONEY,
        },
        key=("sku",),
        rules={
            "price_positive": "list_price_eur > 0",
            "cost_below_price": "unit_cost_eur > 0 AND unit_cost_eur < list_price_eur",
        },
    ),
    "cashiers": Spec(
        columns={"cashier_id": "STRING", "store_id": "STRING"},
        key=("cashier_id",),
        refs={"store_id": ("stores", "store_id")},
    ),
    "fx_rates": Spec(
        columns={"rate_date": "DATE", "currency": "STRING", "units_per_eur": RATE},
        key=("rate_date", "currency"),
        rules={"rate_positive": "units_per_eur > 0"},
    ),
    "pos_sales": Spec(
        columns={
            "transaction_id": "STRING",
            "line_no": "INT",
            "transaction_ts": "TIMESTAMP",
            "business_date": "DATE",
            "store_id": "STRING",
            "cashier_id": "STRING",
            "sku": "STRING",
            "quantity": "INT",
            "unit_price": MONEY,
            "discount_amount": MONEY,
            "gross_amount": MONEY,
            "vat_rate": "DECIMAL(6,4)",
            "net_amount": MONEY,
            "vat_amount": MONEY,
            "currency": "STRING",
            "payment_method": "STRING",
            "promo_flag": "BOOLEAN",
        },
        key=("transaction_id", "line_no"),
        rules={
            "quantity_positive": "quantity > 0",
            "price_positive": "unit_price > 0",
            "discount_not_negative": "discount_amount >= 0",
            "gross_equals_net_plus_vat": "gross_amount = net_amount + vat_amount",
            "date_matches_timestamp": "business_date = to_date(transaction_ts)",
            "payment_known": "payment_method IN ('card', 'cash', 'voucher')",
            "currency_known": "currency IN ('EUR', 'CHF')",
        },
        refs={"store_id": ("stores", "store_id"), "cashier_id": ("cashiers", "cashier_id"), "sku": ("products", "sku")},
        eur={
            "gross_amount_eur": ("gross_amount", "currency", "business_date"),
            "net_amount_eur": ("net_amount", "currency", "business_date"),
        },
    ),
    "refunds": Spec(
        columns={
            "refund_id": "STRING",
            "refund_ts": "TIMESTAMP",
            "business_date": "DATE",
            "store_id": "STRING",
            "cashier_id": "STRING",
            "original_transaction_id": "STRING",
            "sku": "STRING",
            "quantity": "INT",
            "refund_amount": MONEY,
            "vat_rate": "DECIMAL(6,4)",
            "currency": "STRING",
            "receipt_present": "BOOLEAN",
            "refund_method": "STRING",
            "reason": "STRING",
        },
        key=("refund_id",),
        optional=frozenset({"original_transaction_id", "sku"}),  # no-receipt refunds have neither
        rules={
            "quantity_positive": "quantity > 0",
            "amount_positive": "refund_amount > 0",
            "receipt_means_original": "receipt_present = (original_transaction_id IS NOT NULL)",
            "method_known": "refund_method IN ('cash', 'card')",
        },
        refs={
            "store_id": ("stores", "store_id"),
            "cashier_id": ("cashiers", "cashier_id"),
            "sku": ("products", "sku"),
            "original_transaction_id": ("pos_sales", "transaction_id"),
        },
        eur={"refund_amount_eur": ("refund_amount", "currency", "business_date")},
    ),
    "gl_journal": Spec(
        columns={
            "journal_id": "STRING",
            "line_no": "INT",
            "posting_date": "DATE",
            "account_code": "STRING",
            "account_name": "STRING",
            "store_id": "STRING",
            "debit_eur": MONEY,
            "credit_eur": MONEY,
            "source": "STRING",
            "description": "STRING",
        },
        key=("journal_id", "line_no"),
        rules={
            "amounts_not_negative": "debit_eur >= 0 AND credit_eur >= 0",
            "one_side_only": "(debit_eur > 0) <> (credit_eur > 0)",
        },
        refs={"store_id": ("stores", "store_id")},
    ),
    "budget": Spec(
        columns={"store_id": "STRING", "budget_month": "STRING", "revenue_budget_eur": MONEY, "cogs_budget_eur": MONEY},
        key=("store_id", "budget_month"),
        rules={
            "month_format": "budget_month RLIKE '^[0-9]{4}-(0[1-9]|1[0-2])$'",
            "amounts_not_negative": "revenue_budget_eur >= 0 AND cogs_budget_eur >= 0",
        },
        refs={"store_id": ("stores", "store_id")},
    ),
}

# Masters first: later sources check their references against Silver tables built earlier.
ORDER = ["stores", "products", "cashiers", "fx_rates", "pos_sales", "refunds", "gl_journal", "budget"]
AUDIT = ["_source_file", "_ingested_at"]
SMALL = {"stores", "products", "cashiers"}  # reference tables small enough to broadcast


def daily_fx(fx_rates, until):
    """One rate per (currency, day) from the first published rate to ``until``.

    The ECB publishes on working days only, so weekends and holidays carry the last
    published rate forward (the same rule the generator uses). EUR is added at 1.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    days = fx_rates.agg(F.min("rate_date").alias("lo")).select(
        F.explode(F.sequence("lo", F.lit(until).cast("date"))).alias("rate_date")
    )
    grid = days.crossJoin(fx_rates.select("currency").distinct())
    w = Window.partitionBy("currency").orderBy("rate_date").rowsBetween(Window.unboundedPreceding, 0)
    filled = (
        grid.join(fx_rates.select("rate_date", "currency", "units_per_eur"), ["rate_date", "currency"], "left")
        .withColumn("units_per_eur", F.last("units_per_eur", ignorenulls=True).over(w))
        .where(F.col("units_per_eur").isNotNull())
    )
    eur = days.select("rate_date", F.lit("EUR").alias("currency"), F.lit(1).cast(RATE).alias("units_per_eur"))
    return filled.unionByName(eur)


def apply_spec(bronze, spec: Spec, refs: dict | None = None, fx=None):
    """Check a Bronze DataFrame against its spec. Returns (checked, silver, quarantine).

    ``checked`` holds every Bronze row once, typed, with its list of reasons (empty means
    clean). It is cached, so Bronze is read once and both outputs come from memory.
    Rows stay slim on purpose: the original record is serialised only for rows that are
    quarantined, never for the millions that pass.

    ``refs`` maps a source name to its Silver DataFrame, for reference checks.
    ``fx`` is the output of ``daily_fx``, needed when the spec converts amounts to EUR.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    refs = refs or {}
    cols = list(spec.columns)
    raw = {c: F.col(f"`{c}`") for c in cols}

    df = bronze.select(
        *[F.expr(f"try_cast(`{c}` AS {t})").alias(c) for c, t in spec.columns.items()],
        *[F.when(raw[c].isNull() | (F.trim(raw[c]) == ""), None).otherwise(raw[c]).alias(f"__raw_{c}") for c in cols],
        (F.col("_rescued_data") if "_rescued_data" in bronze.columns else F.lit(None)).alias("__rescued"),
        *AUDIT,
    )

    checks = []
    for c in cols:
        if c not in spec.optional:
            checks.append((F.col(f"__raw_{c}").isNull(), f"missing:{c}"))
        checks.append((F.col(f"__raw_{c}").isNotNull() & F.col(c).isNull(), f"bad_type:{c}"))
    checks += [(~F.coalesce(F.expr(expr), F.lit(True)), f"rule:{name}") for name, expr in spec.rules.items()]
    checks.append((F.col("__rescued").isNotNull(), "unexpected_columns"))

    for i, (col, (source, ref_col)) in enumerate(spec.refs.items()):
        keys = refs[source].select(F.col(ref_col).alias(f"__ref{i}")).distinct()
        if source in SMALL:
            keys = F.broadcast(keys)  # a few hundred keys: ship them to every task, no shuffle
        df = df.join(keys, df[col] == keys[f"__ref{i}"], "left")
        checks.append((F.col(col).isNotNull() & F.col(f"__ref{i}").isNull(), f"ref:{col}"))

    for i, (new, (amount, currency, day)) in enumerate(spec.eur.items()):
        rate = F.broadcast(
            fx.select(
                F.col("rate_date").alias(f"__d{i}"),
                F.col("currency").alias(f"__c{i}"),
                F.col("units_per_eur").alias(f"__r{i}"),
            )
        )
        df = df.join(rate, (df[day] == rate[f"__d{i}"]) & (df[currency] == rate[f"__c{i}"]), "left")
        df = df.withColumn(new, F.round(F.col(amount) / F.col(f"__r{i}"), 4).cast(EUR))
        checks.append((F.col(amount).isNotNull() & F.col(f"__r{i}").isNull(), f"no_fx_rate:{new}"))

    df = df.withColumn("__reasons", F.array_compact(F.array(*[F.when(cond, F.lit(r)) for cond, r in checks])))
    bad = F.size("__reasons") > 0
    checked = df.select(
        *cols,
        *spec.eur,
        *AUDIT,
        "__reasons",
        F.when(bad, F.concat_ws("|", *[F.coalesce(F.col(f"__raw_{k}"), F.lit("")) for k in spec.key])).alias("__key"),
        F.when(bad, F.to_json(F.struct(*[F.col(f"__raw_{c}").alias(c) for c in cols]))).alias("__record"),
    ).cache()

    order = [F.col("_ingested_at").desc_nulls_last(), F.col("_source_file").desc_nulls_last()]
    latest = Window.partitionBy(*spec.key).orderBy(*order)
    silver = (
        checked.where(F.size("__reasons") == 0)
        .withColumn("__rn", F.row_number().over(latest))
        .where("__rn = 1")
        .select(*cols, *spec.eur, *AUDIT)
    )
    quarantine = checked.where(F.size("__reasons") > 0).select(
        F.col("__key").alias("record_key"),
        F.col("__reasons").alias("reasons"),
        F.col("__record").alias("record"),
        *AUDIT,
    )
    return checked, silver, quarantine


def count_stats(bronze, checked, silver, quarantine) -> dict:
    """Row counts for the invariant.

    In the job, pass the written tables as ``silver`` and ``quarantine``: counting a Delta
    table is cheap, recomputing the DataFrame is not. Duplicates are the clean rows that did
    not make it into Silver, so the check reduces to bronze = clean + quarantined. It catches
    rows lost in the split and rows multiplied by a bad join (for example a duplicated FX rate).
    """
    from pyspark.sql import functions as F

    clean = checked.where(F.size("__reasons") == 0).count()
    n_silver = silver.count()
    return {
        "bronze": bronze.count(),
        "silver": n_silver,
        "quarantined": quarantine.count(),
        "duplicates": clean - n_silver,
    }


def check_counts(source: str, stats: dict) -> None:
    kept = stats["silver"] + stats["quarantined"] + stats["duplicates"]
    if stats["bronze"] != kept or stats["duplicates"] < 0:
        raise AssertionError(f"{source}: {stats['bronze']} Bronze rows but {kept} accounted for: {stats}")


QUARANTINE_DDL = """
CREATE TABLE IF NOT EXISTS {catalog}.silver.quarantine (
  source STRING, record_key STRING, reasons ARRAY<STRING>, record STRING,
  _source_file STRING, _ingested_at TIMESTAMP, _quarantined_at TIMESTAMP)
COMMENT 'Bronze rows that broke a Silver rule, with the reasons. Rebuilt per source on every run.'
"""


def main() -> None:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="finance")
    a = p.parse_args()
    spark = SparkSession.builder.getOrCreate()
    spark.sql(QUARANTINE_DDL.format(catalog=a.catalog))

    built, fx = {}, None
    for source in ORDER:
        bronze = spark.table(f"{a.catalog}.bronze.{source}")
        checked, silver, quarantine = apply_spec(bronze, SPECS[source], built, fx)

        target, qtable = f"{a.catalog}.silver.{source}", f"{a.catalog}.silver.quarantine"
        silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(target)
        (
            quarantine.select(F.lit(source).alias("source"), "*", F.current_timestamp().alias("_quarantined_at"))
            .write.mode("overwrite")
            .option("replaceWhere", f"source = '{source}'")
            .saveAsTable(qtable)
        )
        built[source] = spark.table(target)
        stats = count_stats(bronze, checked, built[source], spark.table(qtable).where(F.col("source") == source))
        checked.unpersist()
        # Failing here stops the job, so nothing downstream (Gold) reads a Silver that lost rows.
        check_counts(source, stats)
        print(
            f"silver.{source:<12} {stats['silver']:>10,} rows | quarantined {stats['quarantined']:>6,}"
            f" | duplicates {stats['duplicates']:>6,} | bronze {stats['bronze']:>10,}",
            flush=True,
        )

        if source == "fx_rates":
            until = spark.table(f"{a.catalog}.bronze.pos_sales").agg(F.max("business_date")).first()[0]
            daily_fx(built["fx_rates"], until).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
                f"{a.catalog}.silver.fx_daily"
            )
            fx = spark.table(f"{a.catalog}.silver.fx_daily")


if __name__ == "__main__":
    main()
