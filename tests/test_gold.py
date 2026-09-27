"""Gold finance tables (story 4.3): they add up to Silver and make the planted anomalies visible."""

from datetime import date
from decimal import Decimal

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402
from test_generator import SMALL  # noqa: E402

from retail_finance_data.jobs.gold import budget_variance, build, check_access, check_totals, sales_lines  # noqa: E402


@pytest.fixture(scope="module")
def gold(clean):
    silver, _, _, fx = clean
    return build(silver["pos_sales"], silver["refunds"], silver["products"], silver["stores"], fx, silver["budget"])


def test_gold_adds_up_to_silver(clean, gold):
    silver, _, _, _ = clean
    assert check_totals(gold, silver["pos_sales"], silver["refunds"]) == []
    d = gold["daily_revenue"]
    assert d.agg(F.sum("transactions")).first()[0] == silver["pos_sales"].select("transaction_id").distinct().count()
    assert {r[0] for r in d.select("channel").distinct().collect()} == {"store", "online"}


def test_totals_check_catches_a_doubling_join(clean, gold):
    silver, _, _, _ = clean
    doubled = {**gold, "margin": gold["margin"].unionByName(gold["margin"])}
    assert any("margin" in p for p in check_totals(doubled, silver["pos_sales"], silver["refunds"]))


def test_margin_is_plausible(gold):
    m = gold["margin"].agg(F.sum("net_sales_eur").alias("n"), F.sum("cogs_eur").alias("c")).first()
    assert 0.25 < float((m["n"] - m["c"]) / m["n"]) < 0.45


def test_discount_creep_is_visible(gold):
    """A2: the planted store gives away far more discount after its start date."""
    rate = {
        r["month"]: r["rate"]
        for r in gold["margin"]
        .where(F.col("store_id") == SMALL.a2_store)
        .groupBy("month")
        .agg((F.sum("discount_eur") / (F.sum("gross_sales_eur") + F.sum("discount_eur"))).alias("rate"))
        .collect()
    }
    assert rate["2026-08"] > rate["2026-07"]


def test_refund_fraud_points_at_one_cashier(gold):
    """A1: in the planted store, one cashier holds most no-receipt refunds after the start date."""
    top = (
        gold["refunds"]
        .where((F.col("store_id") == SMALL.a1_store) & (F.col("month") == "2026-08"))
        .orderBy(F.desc("no_receipt_eur"))
        .first()
    )
    store_total = (
        gold["refunds"]
        .where((F.col("store_id") == SMALL.a1_store) & (F.col("month") == "2026-08"))
        .agg(F.sum("no_receipt_eur"))
        .first()[0]
    )
    assert top["cashier_id"] == SMALL.a1_cashier
    assert top["no_receipt_eur"] / store_total > 0.5


def test_budget_variance_prorates_the_month_in_progress(spark, clean):
    silver, _, _, fx = clean
    lines = sales_lines(silver["pos_sales"], silver["products"], silver["stores"], fx)
    store = "S001"
    actual = {
        r["month"]: r["n"]
        for r in lines.where(F.col("store_id") == store)
        .groupBy("month")
        .agg(F.round(F.sum("net_amount_eur"), 2).alias("n"))
        .collect()
    }
    budget = spark.createDataFrame(
        [(store, m, Decimal("31000.00"), Decimal("20000.00")) for m in ("2026-07", "2026-08")],
        "store_id string, budget_month string, revenue_budget_eur decimal(18,2), cogs_budget_eur decimal(18,2)",
    )
    got = {r["month"]: r for r in budget_variance(lines, budget, date(2026, 8, 20)).collect()}
    assert got["2026-07"]["complete_month"] and got["2026-07"]["budget_to_date_eur"] == Decimal("31000.00")
    assert not got["2026-08"]["complete_month"]
    assert got["2026-08"]["budget_to_date_eur"] == Decimal("20000.00")  # 20 of 31 days
    assert got["2026-08"]["variance_eur"] == actual["2026-08"] - Decimal("20000.00")


def test_access_rules():
    good = {
        "catalog": [("finance-analysts", "USE CATALOG"), ("finance-data-engineers", "ALL PRIVILEGES")],
        "gold": [("finance-analysts", "USE SCHEMA"), ("finance-analysts", "SELECT")],
        "silver": [("finance-data-engineers", "ALL PRIVILEGES")],
    }
    assert check_access(good) == []
    leaky = {**good, "silver": [("finance-analysts", "SELECT")]}
    assert check_access(leaky) == ["finance-analysts has SELECT on silver"]
    inherited = {**good, "catalog": [("finance-analysts", "SELECT")]}
    assert "whole catalog" in check_access(inherited)[0]
    assert check_access({"catalog": [], "gold": []}) == ["finance-analysts cannot SELECT on gold"]
