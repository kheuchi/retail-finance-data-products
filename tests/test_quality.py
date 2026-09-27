"""Quality gate (story 4.4): clean data passes, a broken file stops the pipeline with a readable reason."""

from decimal import Decimal

import pytest

pytest.importorskip("pyspark")

from conftest import build_all  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from retail_finance_data.jobs.gold import build  # noqa: E402
from retail_finance_data.jobs.quality import CRITICAL, WARNING, enforce, run_checks  # noqa: E402


def tables(built, quarantine, gold=None):
    t = {k: built[k] for k in ("pos_sales", "gl_journal", "stores", "products")}
    t["quarantine"] = quarantine
    if gold is not None:
        t["daily_revenue"] = gold["daily_revenue"]
    return t


def by_name(results):
    return {r["check"]: r for r in results}


def test_clean_data_passes_both_stages(clean, spark):
    built, _, q, fx = clean
    quarantine = q["pos_sales"].limit(0)
    gold = build(built["pos_sales"], built["refunds"], built["products"], built["stores"], fx, built["budget"])
    silver_results = run_checks(tables(built, quarantine), "silver")
    gold_results = run_checks(tables(built, quarantine, gold), "gold")
    assert all(r["passed"] for r in silver_results + gold_results), silver_results + gold_results
    assert {r["severity"] for r in silver_results} == {CRITICAL, WARNING}
    enforce(silver_results + gold_results)  # does not raise


def test_an_unbalanced_journal_in_a_file_stops_the_pipeline(bronze):
    """One ledger line edited in the landing file: every row is still valid on its own, so
    Silver accepts it, but the journal no longer balances and the gate stops the run."""
    gl = bronze["gl_journal"]
    victim = gl.where("account_code = '4000'").orderBy("journal_id").first()
    broken = gl.withColumn(
        "credit_eur",
        F.when(
            (F.col("journal_id") == victim["journal_id"]) & (F.col("line_no") == victim["line_no"]),
            (F.col("credit_eur").cast("decimal(18,2)") + F.lit(Decimal("10.00"))).cast("string"),
        ).otherwise(F.col("credit_eur")),
    )
    built, stats, q, _ = build_all(bronze, {"gl_journal": broken})
    assert stats["gl_journal"]["quarantined"] == 0  # no single row is wrong
    results = run_checks(tables(built, q["gl_journal"]), "silver")
    assert by_name(results)["journals_balance"]["violations"] == 1
    with pytest.raises(RuntimeError, match=f"journals_balance: 1 .*{victim['journal_id']}"):
        enforce(results)


def test_duplicate_keys_and_absurd_amounts_are_critical(clean):
    built, _, q, _ = clean
    pos = built["pos_sales"]
    bad = pos.unionByName(pos.limit(1)).unionByName(pos.limit(1).withColumn("net_amount", F.lit(Decimal("999999.00"))))
    results = by_name(run_checks({**tables(built, q["pos_sales"]), "pos_sales": bad}, "silver"))
    assert not results["silver_keys_unique"]["passed"]
    assert not results["amounts_in_range"]["passed"]


def test_warnings_do_not_stop_the_run(clean):
    built, _, q, _ = clean
    t = tables(built, q["pos_sales"])
    pos = built["pos_sales"]
    day = pos.where("store_id = 'S001'").agg(F.min("business_date")).first()[0]
    one_txn = pos.where((F.col("store_id") == "S001") & (F.col("business_date") == day)).agg(F.min("transaction_id"))
    keep = one_txn.first()[0]
    collapsed = (F.col("store_id") == "S001") & (F.col("business_date") == day) & (F.col("transaction_id") != keep)
    t["pos_sales"] = pos.where(~collapsed)  # one S001 day keeps a single receipt
    results = run_checks(t, "silver")
    assert not by_name(results)["volume_collapse"]["passed"]
    assert f"S001 {day}" in by_name(results)["volume_collapse"]["detail"]
    enforce(results)  # warnings only: no exception


def test_gold_that_no_longer_adds_up_is_critical(clean):
    built, _, q, fx = clean
    gold = build(built["pos_sales"], built["refunds"], built["products"], built["stores"], fx, built["budget"])
    short = {**gold, "daily_revenue": gold["daily_revenue"].where("store_id <> 'S001'")}
    results = by_name(run_checks(tables(built, q["pos_sales"], short), "gold"))
    assert not results["gold_adds_up"]["passed"]
