"""Gold finance tables (story 4.3): they add up to Silver and make the planted anomalies visible."""

from datetime import date
from decimal import Decimal

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402
from test_generator import SMALL  # noqa: E402

from retail_finance_data.jobs.gold import (  # noqa: E402
    budget_variance,
    build,
    check_access,
    check_totals,
    grants_visible,
    sales_lines,
)

BUDGET_SCHEMA = "store_id string, budget_month string, revenue_budget_eur decimal(18,2), cogs_budget_eur decimal(18,2)"


@pytest.fixture(scope="module")
def budget(spark, clean):
    """SMALL has no prior year to budget from: give two stores a budget for July-August 2026."""
    rows = [(s, m, Decimal("31000.00"), Decimal("20000.00")) for s in ("S001", "S002") for m in ("2026-07", "2026-08")]
    return spark.createDataFrame(rows, BUDGET_SCHEMA)


@pytest.fixture(scope="module")
def gold(clean, budget):
    silver, _, _, fx = clean
    return build(silver["pos_sales"], silver["refunds"], silver["products"], silver["stores"], fx, budget)


def test_gold_adds_up_to_silver(clean, gold, budget):
    silver, _, _, _ = clean
    assert check_totals(gold, silver["pos_sales"], silver["refunds"], budget) == []
    d = gold["daily_revenue"]
    assert d.agg(F.sum("transactions")).first()[0] == silver["pos_sales"].select("transaction_id").distinct().count()
    assert {r[0] for r in d.select("channel").distinct().collect()} == {"store", "online"}


def test_totals_check_catches_a_doubled_category(clean, gold, budget):
    silver, _, _, _ = clean
    one = gold["margin"].where(F.col("category") == "Grocery")
    doubled = {**gold, "margin": gold["margin"].unionByName(one)}
    assert any(p.startswith("margin") for p in check_totals(doubled, silver["pos_sales"], silver["refunds"], budget))


def test_margin_is_plausible(gold):
    m = gold["margin"].agg(F.sum("net_sales_eur").alias("n"), F.sum("cogs_eur").alias("c")).first()
    assert 0.25 < float((m["n"] - m["c"]) / m["n"]) < 0.45


def _store_month(gold):
    return {
        (r["store_id"], r["month"]): r
        for r in gold["margin"]
        .groupBy("store_id", "month")
        .agg(
            (F.sum("discount_eur") / (F.sum("paid_incl_vat_eur") + F.sum("discount_eur"))).alias("rate"),
            (1 - F.sum("cogs_eur") / F.sum("net_sales_eur")).alias("margin"),
        )
        .collect()
    }


def test_discount_creep_stands_out_against_other_stores(gold):
    """A2: after its start date the planted store's discount rises and its margin falls,
    far more than any other store (promotions are the same for all stores)."""
    sm = _store_month(gold)
    stores = {s for s, _ in sm}
    change = {s: float(sm[(s, "2026-08")]["rate"] - sm[(s, "2026-07")]["rate"]) for s in stores}
    drop = {s: float(sm[(s, "2026-07")]["margin"] - sm[(s, "2026-08")]["margin"]) for s in stores}
    others = [s for s in stores if s != SMALL.a2_store]
    assert change[SMALL.a2_store] > max(change[s] for s in others) + 0.02
    assert drop[SMALL.a2_store] > max(drop[s] for s in others)


def test_refund_fraud_points_at_one_cashier(gold):
    """A1: in the planted store, one cashier holds most no-receipt refunds after the start date."""
    r = gold["refunds"].where((F.col("store_id") == SMALL.a1_store) & (F.col("month") == "2026-08"))
    top = r.orderBy(F.desc("no_receipt_eur")).first()
    assert top["cashier_id"] == SMALL.a1_cashier
    assert top["no_receipt_eur"] / r.agg(F.sum("no_receipt_eur")).first()[0] > 0.5


def test_budget_variance(spark, clean):
    silver, _, _, fx = clean
    lines = sales_lines(silver["pos_sales"], silver["products"], silver["stores"], fx)
    actual = {
        (r["store_id"], r["month"]): r["n"]
        for r in lines.groupBy("store_id", "month").agg(F.round(F.sum("net_amount_eur"), 2).alias("n")).collect()
    }
    budget = spark.createDataFrame(
        [
            ("S001", "2026-07", Decimal("31000.00"), Decimal("20000.00")),
            ("S001", "2026-08", Decimal("31000.00"), Decimal("20000.00")),
            ("S002", "2026-08", Decimal("0.00"), Decimal("0.00")),
        ],
        BUDGET_SCHEMA,
    )
    got = {(r["store_id"], r["month"]): r for r in budget_variance(lines, budget, date(2026, 8, 20)).collect()}
    july, aug = got[("S001", "2026-07")], got[("S001", "2026-08")]
    assert july["complete_month"] and july["budget_to_date_eur"] == Decimal("31000.00")
    assert not aug["complete_month"]
    assert aug["budget_to_date_eur"] == Decimal("20000.00")  # 20 of 31 days
    assert aug["budget_cogs_to_date_eur"] == Decimal("12903.23")  # cost budget pro-rated too
    assert aug["variance_eur"] == actual[("S001", "2026-08")] - Decimal("20000.00")
    assert got[("S002", "2026-08")]["variance_pct"] is None  # zero budget: no division error
    no_budget = got[("S003", "2026-08")]
    assert no_budget["has_budget"] is False and no_budget["actual_net_sales_eur"] == actual[("S003", "2026-08")]


C = "finance"


def test_access_allows_the_intended_grants_and_ignores_inherited_rows():
    grants = [
        ("finance-analysts", "USE CATALOG", "CATALOG", C),
        ("finance-data-engineers", "ALL PRIVILEGES", "CATALOG", C),
        ("finance-analysts", "USE SCHEMA", "SCHEMA", f"{C}.gold"),
        ("finance-analysts", "SELECT", "SCHEMA", f"{C}.gold"),
        ("finance-data-engineers", "SELECT", "TABLE", f"{C}.silver.pos_sales"),  # inherited, not watched
    ]
    assert check_access(grants) == []


@pytest.mark.parametrize(
    "extra, expected",
    [
        (("finance-analysts", "SELECT", "SCHEMA", f"{C}.silver"), "has SELECT on schema finance.silver"),
        (("finance-analysts", "SELECT", "TABLE", f"{C}.bronze.refunds"), "on table finance.bronze.refunds"),
        (("finance-analysts", "SELECT", "CATALOG", C), "SELECT on catalog finance"),
        (("finance-analysts", "MODIFY", "SCHEMA", f"{C}.gold"), "MODIFY on schema finance.gold"),
        (("account users", "SELECT", "SCHEMA", f"{C}.silver"), "account users has SELECT"),
    ],
)
def test_access_rejects_leaks(extra, expected):
    base = [
        ("finance-analysts", "USE CATALOG", "CATALOG", C),
        ("finance-analysts", "USE SCHEMA", "SCHEMA", f"{C}.gold"),
        ("finance-analysts", "SELECT", "SCHEMA", f"{C}.gold"),
    ]
    problems = check_access([*base, extra])
    assert len(problems) == 1 and expected in problems[0]


def test_access_requires_use_schema_and_select_on_gold():
    assert check_access([("finance-analysts", "SELECT", "SCHEMA", f"{C}.gold")]) == [
        "finance-analysts lacks USE SCHEMA on gold"
    ]


def test_grants_visibility():
    """The runner sees only its own grants: the allow-list is then the platform audit's job."""
    runner_view = [("runner-id", "USE CATALOG", "CATALOG", C), ("runner-id", "SELECT", "SCHEMA", f"{C}.gold")]
    admin_view = [*runner_view, ("finance-analysts", "USE CATALOG", "CATALOG", C)]
    assert not grants_visible(runner_view, "runner-id")
    assert grants_visible(admin_view, "runner-id")
