"""Job: build the Gold finance tables (story 4.3).

Gold is the only layer analysts, models and the agent read. Four tables:

- ``daily_revenue``    store x channel x day: sales, discounts, cost, transactions
- ``margin``           store x category x month: net sales, cost, gross margin, discount rate
- ``refunds``          store x cashier x month: refunds, and how many had no receipt
- ``budget_variance``  store x month: actual vs budget, pro-rated for the month in progress

Definitions (the same ones finance uses): revenue is net sales in EUR excluding VAT; gross
margin is (net sales - cost of goods) / net sales; returns are refunds net of VAT.

Before writing, the job proves Gold adds up to Silver (no sale or refund lost or doubled)
and that the analysts group cannot read below Gold. It then prints the stores and cashiers
that stand out, without being told where the planted anomalies are.
"""

from __future__ import annotations

import argparse

ANALYSTS = "finance-analysts"
BELOW_GOLD = ("raw", "bronze", "silver", "ops")


def sales_lines(pos_sales, products, stores, fx_daily):
    """POS lines with EUR discount, cost of goods and channel."""
    from pyspark.sql import functions as F

    rate = fx_daily.select(F.col("rate_date").alias("business_date"), "currency", "units_per_eur")
    return (
        pos_sales.join(F.broadcast(rate), ["business_date", "currency"], "left")
        .join(F.broadcast(products.select("sku", "category", "unit_cost_eur")), "sku", "left")
        .join(F.broadcast(stores.select("store_id", "format")), "store_id", "left")
        .withColumn("channel", F.when(F.col("format") == "online", "online").otherwise("store"))
        .withColumn("discount_eur", F.col("discount_amount") / F.col("units_per_eur"))
        .withColumn("cogs_eur", F.col("quantity") * F.col("unit_cost_eur"))
        .withColumn("month", F.date_format("business_date", "yyyy-MM"))
    )


def _money(c, name):
    from pyspark.sql import functions as F

    return F.round(F.sum(c), 2).cast("decimal(18,2)").alias(name)


def daily_revenue(lines):
    from pyspark.sql import functions as F

    return lines.groupBy("store_id", "channel", "business_date").agg(
        F.countDistinct("transaction_id").alias("transactions"),
        F.count("*").alias("lines"),
        F.sum("quantity").alias("units"),
        _money(F.col("gross_amount_eur"), "gross_sales_eur"),
        _money(F.col("discount_eur"), "discount_eur"),
        _money(F.col("net_amount_eur"), "net_sales_eur"),
        _money(F.col("cogs_eur"), "cogs_eur"),
    )


def margin(lines):
    from pyspark.sql import functions as F

    return (
        lines.groupBy("store_id", "category", "month")
        .agg(
            _money(F.col("net_amount_eur"), "net_sales_eur"),
            _money(F.col("cogs_eur"), "cogs_eur"),
            _money(F.col("discount_eur"), "discount_eur"),
            _money(F.col("gross_amount_eur"), "gross_sales_eur"),
        )
        .withColumn("gross_margin_eur", F.col("net_sales_eur") - F.col("cogs_eur"))
        .withColumn("margin_pct", F.round(F.col("gross_margin_eur") / F.col("net_sales_eur"), 4))
        # Share of the list price given away: discount / (what was paid + discount).
        .withColumn(
            "discount_rate", F.round(F.col("discount_eur") / (F.col("gross_sales_eur") + F.col("discount_eur")), 4)
        )
    )


def refunds(refunds_silver):
    from pyspark.sql import functions as F

    no_receipt = ~F.col("receipt_present")
    return (
        refunds_silver.withColumn("month", F.date_format("business_date", "yyyy-MM"))
        .groupBy("store_id", "cashier_id", "month")
        .agg(
            F.count("*").alias("refunds"),
            _money(F.col("refund_amount_eur"), "refund_gross_eur"),
            _money(F.col("refund_amount_eur") / (1 + F.col("vat_rate")), "refund_net_eur"),
            F.sum(no_receipt.cast("int")).alias("no_receipt_refunds"),
            _money(F.when(no_receipt, F.col("refund_amount_eur")).otherwise(0), "no_receipt_eur"),
        )
        .withColumn("no_receipt_share", F.round(F.col("no_receipt_eur") / F.col("refund_gross_eur"), 4))
    )


def budget_variance(lines, budget, last_day):
    """Actual vs budget per store and month. The month in progress is compared with a
    pro-rated budget, so a partial month does not look like a collapse."""
    from pyspark.sql import functions as F

    actual = lines.groupBy("store_id", "month").agg(
        _money(F.col("net_amount_eur"), "actual_net_sales_eur"), _money(F.col("cogs_eur"), "actual_cogs_eur")
    )
    month_start = F.to_date(F.concat(F.col("month"), F.lit("-01")))
    month_end = F.last_day(month_start)
    last = F.lit(last_day).cast("date")
    days_in_month = F.dayofmonth(month_end)
    days_elapsed = F.when(month_end <= last, days_in_month).otherwise(F.dayofmonth(last))
    return (
        budget.select(
            "store_id",
            F.col("budget_month").alias("month"),
            F.col("revenue_budget_eur").alias("budget_net_sales_eur"),
            F.col("cogs_budget_eur").alias("budget_cogs_eur"),
        )
        .join(actual, ["store_id", "month"])
        .withColumn("complete_month", month_end <= last)
        .withColumn(
            "budget_to_date_eur",
            F.round(F.col("budget_net_sales_eur") * days_elapsed / days_in_month, 2).cast("decimal(18,2)"),
        )
        .withColumn("variance_eur", (F.col("actual_net_sales_eur") - F.col("budget_to_date_eur")).cast("decimal(18,2)"))
        .withColumn("variance_pct", F.round(F.col("variance_eur") / F.col("budget_to_date_eur"), 4))
    )


def build(pos_sales, refunds_silver, products, stores, fx_daily, budget):
    """Return a dict of Gold DataFrames."""
    from pyspark.sql import functions as F

    lines = sales_lines(pos_sales, products, stores, fx_daily)
    last_day = pos_sales.agg(F.max("business_date")).first()[0]
    return {
        "daily_revenue": daily_revenue(lines),
        "margin": margin(lines),
        "refunds": refunds(refunds_silver),
        "budget_variance": budget_variance(lines, budget, last_day),
    }


def check_totals(gold: dict, pos_sales, refunds_silver) -> list[str]:
    """Gold must add up to Silver: nothing lost, nothing doubled by a join."""
    from pyspark.sql import functions as F

    def total(df, col):
        return df.agg(F.sum(col)).first()[0] or 0

    problems = []
    silver_net = round(total(pos_sales, "net_amount_eur"), 2)
    for name in ("daily_revenue", "margin"):
        got = total(gold[name], "net_sales_eur")
        if abs(got - silver_net) > 1:  # per-group rounding to cents: at most a few euros over millions of rows
            problems.append(f"{name}: net sales {got} vs Silver {silver_net}")
    if total(gold["daily_revenue"], "lines") != pos_sales.count():
        problems.append("daily_revenue: line count differs from Silver")
    if total(gold["refunds"], "refunds") != refunds_silver.count():
        problems.append("refunds: count differs from Silver")
    if pos_sales.join(gold["margin"].select("store_id").distinct(), "store_id", "left_anti").limit(1).count():
        problems.append("margin: a store with sales is missing")
    return problems


def check_access(grants: dict[str, list[tuple[str, str]]]) -> list[str]:
    """``grants`` maps a securable (``catalog`` or a schema name) to (principal, privilege) pairs.
    The analysts group may use the catalog and read Gold, nothing below."""
    problems = []
    for principal, privilege in grants.get("catalog", []):
        if principal == ANALYSTS and privilege not in ("USE CATALOG", "USE_CATALOG", "BROWSE"):
            problems.append(f"{ANALYSTS} has {privilege} on the whole catalog (inherited by every schema)")
    for schema in BELOW_GOLD:
        for principal, privilege in grants.get(schema, []):
            if principal == ANALYSTS:
                problems.append(f"{ANALYSTS} has {privilege} on {schema}")
    if not any(p == ANALYSTS and "SELECT" in v for p, v in grants.get("gold", [])):
        problems.append(f"{ANALYSTS} cannot SELECT on gold")
    return problems


def _grants(spark, securable: str) -> list[tuple[str, str]]:
    return [(r[0], r[1]) for r in spark.sql(f"SHOW GRANTS ON {securable}").collect()]


def main() -> None:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="finance")
    a = p.parse_args()
    spark = SparkSession.builder.getOrCreate()
    silver = {
        t: spark.table(f"{a.catalog}.silver.{t}")
        for t in ("pos_sales", "refunds", "products", "stores", "fx_daily", "budget")
    }

    grants = {"catalog": _grants(spark, f"CATALOG {a.catalog}")}
    for schema in (*BELOW_GOLD, "gold"):
        grants[schema] = _grants(spark, f"SCHEMA {a.catalog}.{schema}")
    names = ("pos_sales", "refunds", "products", "stores", "fx_daily", "budget")
    tables = build(*(silver[t] for t in names))
    gold = {name: df.cache() for name, df in tables.items()}
    problems = check_access(grants) + check_totals(gold, silver["pos_sales"], silver["refunds"])
    if problems:
        raise RuntimeError("Gold refused: " + "; ".join(problems))

    for name, df in gold.items():
        target = f"{a.catalog}.gold.{name}"
        df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(target)
        spark.sql(f"ALTER TABLE {target} SET TBLPROPERTIES ('quality' = 'gold', 'story' = '4.3')")
        print(f"gold.{name:<16} {spark.table(target).count():>8,} rows", flush=True)

    # KPIs finance would look at first, then what stands out, found from the data alone.
    d = spark.table(f"{a.catalog}.gold.daily_revenue").where("year(business_date) = 2025")
    k = d.agg(F.sum("net_sales_eur").alias("net"), F.sum("cogs_eur").alias("cogs")).first()
    r = spark.table(f"{a.catalog}.gold.refunds").where("month LIKE '2025-%'").agg(F.sum("refund_net_eur")).first()[0]
    gm, ret = (k["net"] - k["cogs"]) / k["net"], r / k["net"]
    print(f"2025: net sales EUR {k['net']:,.0f} | gross margin {gm:.2%} | returns {ret:.2%}")

    m = (
        spark.table(f"{a.catalog}.gold.margin")
        .groupBy("store_id", "month")
        .agg(
            (1 - F.sum("cogs_eur") / F.sum("net_sales_eur")).alias("margin"),
            (F.sum("discount_eur") / (F.sum("gross_sales_eur") + F.sum("discount_eur"))).alias("discount_rate"),
        )
    )
    last_month = m.agg(F.max("month")).first()[0]
    print(f"Largest margin drops, {last_month} vs the store's first 5 months of the year:")
    base = m.where(F.col("month").between(last_month[:4] + "-01", last_month[:4] + "-05"))
    base = base.groupBy("store_id").agg(F.avg("margin").alias("base_margin"))
    m.where(F.col("month") == last_month).join(base, "store_id").withColumn(
        "drop_pts", F.round((F.col("base_margin") - F.col("margin")) * 100, 1)
    ).orderBy(F.desc("drop_pts")).show(3)
    print(f"Highest no-receipt refunds, {last_month}:")
    spark.table(f"{a.catalog}.gold.refunds").where(F.col("month") == last_month).orderBy(
        F.desc("no_receipt_eur")
    ).select("store_id", "cashier_id", "refunds", "no_receipt_refunds", "no_receipt_eur", "no_receipt_share").show(3)


if __name__ == "__main__":
    main()
