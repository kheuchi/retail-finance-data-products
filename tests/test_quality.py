"""Quality gates (story 4.4): clean data passes, a broken file stops the pipeline with a readable reason."""

from decimal import Decimal

import pytest

pytest.importorskip("pyspark")

from conftest import build_all  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from retail_finance_data.jobs.gold import build  # noqa: E402
from retail_finance_data.jobs.quality import (  # noqa: E402
    CRITICAL,
    WARNING,
    compare_counts,
    compare_snapshots,
    enforce,
    identity_checks,
    lineage_chains,
    run_checks,
)


def tables(built, quarantine=None, gold=None):
    t = {k: built[k] for k in ("pos_sales", "refunds", "gl_journal", "stores", "products")}
    t["quarantine"] = quarantine if quarantine is not None else empty_quarantine(built)
    if gold is not None:
        t["daily_revenue"] = gold["daily_revenue"]
    return t


def empty_quarantine(built):
    return built["stores"].select(F.col("store_id").alias("source")).limit(0)


def by_name(results):
    return {r["check"]: r for r in results}


def test_clean_data_passes_both_stages(clean):
    built, _, _, fx = clean
    gold = build(built["pos_sales"], built["refunds"], built["products"], built["stores"], fx, built["budget"])
    silver_results = run_checks(tables(built), "silver")
    gold_results = run_checks(tables(built, gold=gold), "gold")
    assert all(r["passed"] for r in silver_results + gold_results), silver_results + gold_results
    assert {r["severity"] for r in silver_results} == {CRITICAL, WARNING}
    enforce(silver_results + gold_results)  # does not raise


def test_an_unbalanced_journal_in_a_file_stops_the_pipeline(bronze):
    """One ledger line edited in the landing file: every row is still valid on its own, so
    Silver accepts it, but the journal no longer balances and the gate stops the run."""
    gl = bronze["gl_journal"]
    victim = gl.where("account_code = '4000'").orderBy("journal_id").first()
    hit = (F.col("journal_id") == victim["journal_id"]) & (F.col("line_no") == victim["line_no"])
    raised = (F.col("credit_eur").cast("decimal(18,2)") + F.lit(Decimal("10.00"))).cast("string")
    built, stats, _, _ = build_all(
        bronze, {"gl_journal": gl.withColumn("credit_eur", F.when(hit, raised).otherwise(F.col("credit_eur")))}
    )
    assert stats["gl_journal"]["quarantined"] == 0  # no single row is wrong
    results = run_checks(tables(built), "silver")
    assert by_name(results)["journals_balance"]["violations"] == 1
    with pytest.raises(RuntimeError, match=f"journals_balance: 1 .*{victim['journal_id']}"):
        enforce(results)


def test_duplicate_keys_and_absurd_amounts_are_critical(clean):
    built, _, _, _ = clean
    pos = built["pos_sales"]
    huge = pos.limit(1).withColumn("net_amount_eur", F.lit(Decimal("999999.0000")))
    results = by_name(
        run_checks({**tables(built), "pos_sales": pos.unionByName(pos.limit(1)).unionByName(huge)}, "silver")
    )
    assert not results["silver_keys_unique"]["passed"]
    assert not results["amounts_in_range"]["passed"]


def test_quarantined_money_rows_are_critical_masters_only_warn(spark, clean):
    built, _, _, _ = clean
    q = spark.createDataFrame([("refunds",), ("stores",)], "source string")
    results = by_name(run_checks(tables(built, quarantine=q), "silver"))
    assert not results["quarantine_money"]["passed"] and results["quarantine_money"]["severity"] == CRITICAL
    assert not results["quarantine_masters"]["passed"] and results["quarantine_masters"]["severity"] == WARNING


def test_a_missing_file_leaves_an_open_day_empty(clean):
    """A whole store-day gone (a file never arrived) is not a quiet day: it is flagged."""
    built, _, _, _ = clean
    pos = built["pos_sales"]
    day = pos.where("store_id = 'S001'").agg(F.max("business_date")).first()[0]
    gone = pos.where(~((F.col("store_id") == "S001") & (F.col("business_date") == day)))
    r = by_name(run_checks({**tables(built), "pos_sales": gone}, "silver"))["missing_store_days"]
    assert r["violations"] == 1 and f"S001 {day}" in r["detail"]


def test_one_day_volume_collapse_warns_without_stopping(clean):
    built, _, _, _ = clean
    pos = built["pos_sales"]
    day = pos.where("store_id = 'S001'").agg(F.min("business_date")).first()[0]
    keep = pos.where((F.col("store_id") == "S001") & (F.col("business_date") == day)).agg(F.min("transaction_id"))
    keep = keep.first()[0]
    collapsed = (F.col("store_id") == "S001") & (F.col("business_date") == day) & (F.col("transaction_id") != keep)
    results = run_checks({**tables(built), "pos_sales": pos.where(~collapsed)}, "silver")
    r = by_name(results)["volume_collapse"]
    assert not r["passed"] and f"S001 {day}" in r["detail"]
    enforce(results)  # warnings only: no exception


def test_refund_larger_than_its_sale_is_flagged(clean):
    built, _, _, _ = clean
    ref = built["refunds"]
    victim = ref.where("receipt_present").orderBy("refund_id").first()["refund_id"]
    inflated = ref.withColumn(
        "refund_amount",
        F.when(F.col("refund_id") == victim, F.col("refund_amount") + 5000).otherwise(F.col("refund_amount")),
    )
    assert (
        by_name(run_checks({**tables(built), "refunds": inflated}, "silver"))["refund_within_sale"]["violations"] == 1
    )


def test_gold_that_no_longer_adds_up_is_critical(clean):
    built, _, _, fx = clean
    gold = build(built["pos_sales"], built["refunds"], built["products"], built["stores"], fx, built["budget"])
    short = {**gold, "daily_revenue": gold["daily_revenue"].where("store_id <> 'S001'")}
    assert not by_name(run_checks(tables(built, gold=short), "gold"))["gold_adds_up"]["passed"]


def test_silver_must_not_change_during_a_run():
    assert compare_snapshots({"pos_sales": 7}, {"pos_sales": 7}, "gold")["passed"]
    changed = compare_snapshots({"pos_sales": 7}, {"pos_sales": 8}, "gold")
    assert not changed["passed"] and changed["severity"] == CRITICAL and "pos_sales" in changed["detail"]
    assert not compare_snapshots({}, {"pos_sales": 8}, "gold")["passed"]  # no silver gate in this run


def test_row_count_drop_warns():
    assert compare_counts({}, {"pos_sales": 10}, "silver")["passed"]  # first run: nothing to compare
    assert compare_counts({"pos_sales": 100}, {"pos_sales": 96}, "silver")["passed"]
    r = compare_counts({"pos_sales": 100}, {"pos_sales": 90}, "silver")
    assert not r["passed"] and r["severity"] == WARNING


def test_lineage_chains_keep_only_this_jobs_links_and_reach_bronze():
    graph = {
        "finance.gold.margin": [("finance.silver.pos_sales", [42]), ("finance.silver.products", [7])],
        "finance.silver.pos_sales": [("finance.bronze.pos_sales", [9]), ("finance.silver.fx_daily", [9])],
    }

    def api_get(path, query):
        ups = graph.get(query["table_name"], [])
        return {
            "upstreams": [
                {
                    "tableInfo": dict(zip(("catalog_name", "schema_name", "name"), t.split("."), strict=True)),
                    "jobInfos": [{"job_id": j} for j in js],
                }
                for t, js in ups
            ]
        }

    chains = lineage_chains(api_get, "finance", job_id=42)
    assert chains == [("finance.gold.margin", "finance.silver.pos_sales", "finance.bronze.pos_sales")]


def test_identity_checks_pass_when_the_runner_is_powerless():
    executed = []

    def sql(q):
        executed.append(q)
        if q.startswith("GRANT"):
            raise PermissionError("PERMISSION_DENIED")
        return [{"u": "runner-app-id", "admin": False}]

    results = by_name(identity_checks(sql, "finance"))
    assert results["runner_not_admin"]["passed"] and results["runner_cannot_grant"]["passed"]
    assert not any(q.startswith("REVOKE") for q in executed)


def test_identity_checks_fail_and_revoke_when_the_runner_can_grant():
    executed = []

    def sql(q):
        executed.append(q)
        return [{"u": "terraform-platform", "admin": True}]

    results = by_name(identity_checks(sql, "finance"))
    assert not results["runner_not_admin"]["passed"] and not results["runner_cannot_grant"]["passed"]
    assert executed[-1] == "REVOKE SELECT ON SCHEMA finance.silver TO `finance-analysts`"
