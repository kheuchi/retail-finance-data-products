"""Job: build the Gold finance tables (story 4.3).

Gold is the only layer analysts, models and the agent read. Four tables:

- ``daily_revenue``    store x channel x day: sales, discounts, cost, transactions
- ``margin``           store x category x month: net sales, cost, gross margin, discount rate
- ``refunds``          store x cashier x month: refunds, and how many had no receipt
- ``budget_variance``  store x month of the budget year: actual vs budget, pro-rated for the
                       month in progress

Definitions: net sales are POS sales in EUR excluding VAT, before refunds; gross margin is
(net sales - cost of goods) / net sales; returns are refunds net of VAT; the discount rate is
discount / (amount paid incl. VAT + discount).

Before writing, the job proves Gold adds up to Silver and that analysts cannot read below
Gold. It then prints the stores and cashiers that stand out, found from the data alone.
"""

from __future__ import annotations

import argparse

ANALYSTS = "finance-analysts"
BROAD = ("account users", "users")  # groups everyone belongs to: must not reach below Gold either
BELOW_GOLD = ("raw", "bronze", "silver", "ops")
CATALOG_OK = {"USE CATALOG", "USE_CATALOG", "BROWSE"}
GOLD_OK = {"USE SCHEMA", "USE_SCHEMA", "SELECT"}

COMMENTS = {
    "net_sales_eur": "POS sales in EUR, excluding VAT, before refunds",
    "paid_incl_vat_eur": "What customers paid, including VAT, after discounts",
    "discount_eur": "Discounts given, in EUR including VAT",
    "cogs_eur": "Cost of goods sold: quantity x unit cost (EUR)",
    "margin_pct": "(net sales - cost of goods) / net sales",
    "discount_rate": "discount / (paid incl. VAT + discount)",
    "refund_net_eur": "Refunds in EUR excluding VAT",
    "no_receipt_share": "Share of refund value given without a receipt",
    "budget_to_date_eur": "Revenue budget pro-rated to the days with actuals in the month",
    "variance_eur": "Actual net sales - budget to date",
}


def sales_lines(pos_sales, products, stores, fx_daily):
    """POS lines with EUR discount, cost of goods, channel and month."""
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


def _ratio(num, den, name):
    from pyspark.sql import functions as F

    return F.round(F.try_divide(num, den), 4).alias(name)  # null instead of failing on a zero


def daily_revenue(lines):
    from pyspark.sql import functions as F

    return lines.groupBy("store_id", "channel", "business_date").agg(
        F.countDistinct("transaction_id").alias("transactions"),
        F.count("*").alias("lines"),
        F.sum("quantity").alias("units"),
        _money(F.col("gross_amount_eur"), "paid_incl_vat_eur"),
        _money(F.col("discount_eur"), "discount_eur"),
        _money(F.col("net_amount_eur"), "net_sales_eur"),
        _money(F.col("cogs_eur"), "cogs_eur"),
    )


def margin(lines):
    from pyspark.sql import functions as F

    m = lines.groupBy("store_id", "category", "month").agg(
        _money(F.col("net_amount_eur"), "net_sales_eur"),
        _money(F.col("cogs_eur"), "cogs_eur"),
        _money(F.col("discount_eur"), "discount_eur"),
        _money(F.col("gross_amount_eur"), "paid_incl_vat_eur"),
    )
    return m.withColumn("gross_margin_eur", F.col("net_sales_eur") - F.col("cogs_eur")).select(
        "*",
        _ratio(F.col("gross_margin_eur"), F.col("net_sales_eur"), "margin_pct"),
        _ratio(F.col("discount_eur"), F.col("paid_incl_vat_eur") + F.col("discount_eur"), "discount_rate"),
    )


def refunds(refunds_silver):
    from pyspark.sql import functions as F

    no_receipt = ~F.col("receipt_present")
    r = (
        refunds_silver.withColumn("month", F.date_format("business_date", "yyyy-MM"))
        .groupBy("store_id", "cashier_id", "month")
        .agg(
            F.count("*").alias("refunds"),
            _money(F.col("refund_amount_eur"), "refund_gross_eur"),
            _money(F.col("refund_amount_eur") / (1 + F.col("vat_rate")), "refund_net_eur"),
            F.sum(no_receipt.cast("int")).alias("no_receipt_refunds"),
            _money(F.when(no_receipt, F.col("refund_amount_eur")).otherwise(0), "no_receipt_eur"),
        )
    )
    return r.select("*", _ratio(F.col("no_receipt_eur"), F.col("refund_gross_eur"), "no_receipt_share"))


def budget_variance(lines, budget, last_day):
    """Actual vs budget for every store-month with sales in a budgeted year.

    Store-months with sales but no budget stay in, with ``has_budget = false``. The month in
    progress is compared with a budget pro-rated to the days with actuals (revenue and cost).
    """
    from pyspark.sql import functions as F

    years = [r[0] for r in budget.select(F.substring("budget_month", 1, 4)).distinct().collect()]
    actual = (
        lines.where(F.substring("month", 1, 4).isin(years))
        .groupBy("store_id", "month")
        .agg(_money(F.col("net_amount_eur"), "actual_net_sales_eur"), _money(F.col("cogs_eur"), "actual_cogs_eur"))
    )
    month_end = F.last_day(F.to_date(F.concat(F.col("month"), F.lit("-01"))))
    last = F.lit(last_day).cast("date")
    share = F.when(month_end <= last, F.lit(1.0)).otherwise(F.dayofmonth(last) / F.dayofmonth(month_end))
    b = budget.select(
        "store_id",
        F.col("budget_month").alias("month"),
        F.col("revenue_budget_eur").alias("budget_net_sales_eur"),
        F.col("cogs_budget_eur").alias("budget_cogs_eur"),
    )
    v = (
        actual.join(b, ["store_id", "month"], "left")
        .withColumn("has_budget", F.col("budget_net_sales_eur").isNotNull())
        .withColumn("complete_month", month_end <= last)
        .withColumn("budget_to_date_eur", F.round(F.col("budget_net_sales_eur") * share, 2).cast("decimal(18,2)"))
        .withColumn("budget_cogs_to_date_eur", F.round(F.col("budget_cogs_eur") * share, 2).cast("decimal(18,2)"))
        .withColumn("variance_eur", (F.col("actual_net_sales_eur") - F.col("budget_to_date_eur")).cast("decimal(18,2)"))
    )
    return v.select("*", _ratio(F.col("variance_eur"), F.col("budget_to_date_eur"), "variance_pct"))


def build(pos_sales, refunds_silver, products, stores, fx_daily, budget, cache: bool = True):
    """Return a dict of Gold DataFrames, all computed from one pass over the sales lines
    (cached for the checks; uncached for the writes, see ``main``)."""
    from pyspark.sql import functions as F

    lines = sales_lines(pos_sales, products, stores, fx_daily)
    lines = lines.cache() if cache else lines
    last_day = pos_sales.agg(F.max("business_date")).first()[0]
    return {
        "daily_revenue": daily_revenue(lines),
        "margin": margin(lines),
        "refunds": refunds(refunds_silver),
        "budget_variance": budget_variance(lines, budget, last_day),
    }


def check_totals(gold: dict, pos_sales, refunds_silver, budget) -> list[str]:
    """Gold must add up to Silver: nothing lost, nothing doubled by a join."""
    from pyspark.sql import functions as F

    def total(df, col):
        return df.agg(F.sum(col)).first()[0] or 0

    problems = []
    silver_net = total(pos_sales, "net_amount_eur")
    for name in ("daily_revenue", "margin"):
        got = total(gold[name], "net_sales_eur")
        if abs(got - silver_net) > 1:  # rounding each group to cents: well under EUR 1 in total
            problems.append(f"{name}: net sales {got} vs Silver {silver_net}")
    if total(gold["daily_revenue"], "lines") != pos_sales.count():
        problems.append("daily_revenue: line count differs from Silver")
    if abs(total(gold["margin"], "cogs_eur") - total(gold["daily_revenue"], "cogs_eur")) > 1:
        problems.append("margin: cost of goods differs from daily_revenue")
    if total(gold["refunds"], "refunds") != refunds_silver.count():
        problems.append("refunds: count differs from Silver")
    if abs(total(gold["refunds"], "refund_gross_eur") - total(refunds_silver, "refund_amount_eur")) > 1:
        problems.append("refunds: amount differs from Silver")

    years = [r[0] for r in budget.select(F.substring("budget_month", 1, 4)).distinct().collect()]
    in_years = pos_sales.where(F.date_format("business_date", "yyyy").isin(years))
    store_months = in_years.select("store_id", F.date_format("business_date", "yyyy-MM")).distinct().count()
    if gold["budget_variance"].count() != store_months:
        problems.append("budget_variance: store-months differ from Silver")
    if abs(total(gold["budget_variance"], "actual_net_sales_eur") - total(in_years, "net_amount_eur")) > 1:
        problems.append("budget_variance: actual net sales differ from Silver")
    return problems


def check_access(grants: list[tuple[str, str, str, str]]) -> list[str]:
    """Allow-list for who can read what.

    ``grants`` rows are (principal, privilege, object_type, object_name) for grants made
    directly on an object: the catalog, a schema, or a table inside a schema.
    Analysts may hold USE CATALOG/BROWSE on the catalog and USE SCHEMA + SELECT on gold;
    groups everyone belongs to may only browse. Anything else is a problem.
    """
    watched = (ANALYSTS, *BROAD)
    problems = []
    for principal, privilege, otype, name in grants:
        if principal not in watched:
            continue
        schema = name.split(".")[1] if "." in name else None
        if otype == "CATALOG" and privilege in CATALOG_OK:
            continue
        if principal == ANALYSTS and otype == "SCHEMA" and schema == "gold" and privilege in GOLD_OK:
            continue
        problems.append(f"{principal} has {privilege} on {otype.lower()} {name}")
    gold = {v for p, v, t, n in grants if p == ANALYSTS and t == "SCHEMA" and n.endswith(".gold")}
    for needed in ("USE SCHEMA", "SELECT"):
        if not gold & {needed, needed.replace(" ", "_")}:
            problems.append(f"{ANALYSTS} lacks {needed} on gold")
    return problems


def grants_visible(grants: list[tuple[str, str, str, str]], me: str) -> bool:
    """True when the job identity can see other principals' grants on the catalog or a schema.

    Table rows do not count: the runner owns the tables it creates (e.g. ops.model_drift) and
    sees their grants, which says nothing about who may use the schemas. Counting them made
    Gold refuse to publish from the second scheduled run on (story 6.2 regression)."""
    return any(principal != me and otype in ("CATALOG", "SCHEMA") for principal, _, otype, _ in grants)


def read_grants(spark, catalog: str) -> list[tuple[str, str, str, str]]:
    """Direct grants on the catalog, every schema, and every table below Gold."""
    rows = []

    def direct(securable, otype):
        for r in spark.sql(f"SHOW GRANTS ON {securable}").collect():
            d = r.asDict()
            principal = d.get("Principal") or d.get("principal")
            action = d.get("ActionType") or d.get("action_type")
            granted_on = (d.get("ObjectType") or d.get("object_type") or otype).upper()
            key = d.get("ObjectKey") or d.get("object_key") or securable.split()[-1]
            if granted_on == otype:  # inherited rows are judged where they were granted
                rows.append((principal, action, otype, key))

    direct(f"CATALOG {catalog}", "CATALOG")
    for schema in (*BELOW_GOLD, "gold"):
        direct(f"SCHEMA {catalog}.{schema}", "SCHEMA")
    schemas = ", ".join(f"'{s}'" for s in BELOW_GOLD)
    for r in spark.sql(
        f"SELECT grantee, privilege_type, table_schema, table_name FROM {catalog}.information_schema.table_privileges "
        f"WHERE table_schema IN ({schemas})"
    ).collect():
        rows.append((r[0], r[1], "TABLE", f"{catalog}.{r[2]}.{r[3]}"))
    return rows


def print_highlights(spark, catalog: str) -> None:
    """KPIs finance looks at first, then what stands out, found from the data alone."""
    from pyspark.sql import functions as F

    g = f"{catalog}.gold"
    year = spark.table(f"{g}.daily_revenue").agg(F.max(F.year("business_date"))).first()[0] - 1
    k = (
        spark.table(f"{g}.daily_revenue")
        .where(F.year("business_date") == year)
        .agg(F.sum("net_sales_eur").alias("net"), F.sum("cogs_eur").alias("cogs"))
        .first()
    )
    r = spark.table(f"{g}.refunds").where(F.col("month").startswith(str(year))).agg(F.sum("refund_net_eur")).first()[0]
    if k["net"]:
        gm, ret = (k["net"] - k["cogs"]) / k["net"], (r or 0) / k["net"]
        print(f"{year}: net sales EUR {k['net']:,.0f} | gross margin {gm:.2%} | returns {ret:.2%}", flush=True)

    m = (
        spark.table(f"{g}.margin")
        .groupBy("store_id", "month")
        .agg(
            (1 - F.sum("cogs_eur") / F.sum("net_sales_eur")).alias("margin"),
            (F.sum("discount_eur") / (F.sum("paid_incl_vat_eur") + F.sum("discount_eur"))).alias("discount_rate"),
        )
    )
    last = m.agg(F.max("month")).first()[0]
    base = m.where(F.col("month").between(last[:4] + "-01", last[:4] + "-05")).groupBy("store_id")
    base = base.agg(F.avg("margin").alias("base_margin"), F.avg("discount_rate").alias("base_discount"))
    print(f"Largest margin drops, {last} vs January-May:", flush=True)
    m.where(F.col("month") == last).join(base, "store_id").withColumn(
        "drop_pts", F.round((F.col("base_margin") - F.col("margin")) * 100, 1)
    ).orderBy(F.desc("drop_pts")).show(3)
    print(f"Highest no-receipt refunds, {last}:", flush=True)
    spark.table(f"{g}.refunds").where(F.col("month") == last).orderBy(F.desc("no_receipt_eur")).select(
        "store_id", "cashier_id", "refunds", "no_receipt_refunds", "no_receipt_eur", "no_receipt_share"
    ).show(3)


def main() -> None:
    from pyspark.sql import SparkSession

    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="finance")
    a = p.parse_args()
    spark = SparkSession.builder.getOrCreate()
    names = ("pos_sales", "refunds", "products", "stores", "fx_daily", "budget")
    silver = {t: spark.table(f"{a.catalog}.silver.{t}") for t in names}

    grants = read_grants(spark, a.catalog)
    for g in grants:
        print("grant", g, flush=True)
    gold = {name: df.cache() for name, df in build(*(silver[t] for t in names)).items()}
    # Only owners and admins see other principals' grants. The job runs as finance-pipeline-runner,
    # which cannot (by design, story 4.5), so the allow-list is enforced by the platform's access
    # audit in the infra repo's CI. When the job can see them (an admin run), it enforces it too.
    me = spark.sql("SELECT current_user()").first()[0]
    if grants_visible(grants, me):
        problems = check_access(grants)
    else:
        problems = []
        print(f"access allow-list: grants of other principals not visible to {me}; enforced by the platform audit")
    problems += check_totals(gold, silver["pos_sales"], silver["refunds"], silver["budget"])
    if problems:
        raise RuntimeError("Gold refused: " + "; ".join(problems))

    # Write from a fresh, uncached plan: a cached DataFrame hides its source tables, and
    # Unity Catalog then records no lineage for the table written from it.
    fresh = build(*(silver[t] for t in names), cache=False)
    for df in gold.values():
        df.unpersist()
    for name, df in fresh.items():
        target = f"{a.catalog}.gold.{name}"
        df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(target)
        spark.sql(f"ALTER TABLE {target} SET TBLPROPERTIES ('quality' = 'gold', 'story' = '4.3')")
        for col, text in COMMENTS.items():
            if col in df.columns:
                spark.sql(f"ALTER TABLE {target} ALTER COLUMN {col} COMMENT '{text}'")
        print(f"gold.{name:<16} {spark.table(target).count():>8,} rows", flush=True)

    try:  # evidence only: a failure here must not fail a job whose tables are written
        print_highlights(spark, a.catalog)
    except Exception as e:
        print(f"Highlights skipped: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
