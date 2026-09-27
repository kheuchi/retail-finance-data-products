"""GL vs POS reconciliation (story 4.2): exactly the planted A3 journals, and no false accusations."""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

pytest.importorskip("pyspark")

from conftest import with_rows  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from test_generator import SMALL  # noqa: E402

from retail_finance_data.generator import build  # noqa: E402
from retail_finance_data.jobs.reconcile import (  # noqa: E402
    ANOMALY,
    DUPLICATE,
    UNATTRIBUTED,
    UNSUPPORTED,
    check_inputs,
    read_answer_key,
    reconcile,
    score,
)


@pytest.fixture(scope="module")
def planted():
    t = pd.concat(df for name, _, df in build(SMALL) if name == "_ground_truth/planted_records")
    return set(t.loc[t["anomaly"] == ANOMALY, "entity_id"])


def run(silver, fx, gl=None):
    daily, monthly, exceptions = reconcile(silver["pos_sales"], gl if gl is not None else silver["gl_journal"], fx)
    return daily, monthly, exceptions.collect()


def journal_like(spark, gl, **changes):
    """One revenue line copied from a real POS journal, with ``changes`` applied."""
    return with_rows(spark, gl.where("account_code = '4000' AND source = 'POS'"), [changes]).where(
        F.col("journal_id") == changes["journal_id"]
    )


def clean_pos_journal(silver, planted):
    """A POS revenue line on a store-day with no planted journal, chosen deterministically."""
    gl = silver["gl_journal"]
    busy = gl.where(F.col("journal_id").isin(*planted)).select("store_id", "posting_date").distinct()
    return (
        gl.where("source = 'POS' AND account_code = '4000'")
        .join(busy, ["store_id", "posting_date"], "left_anti")
        .orderBy("posting_date", "store_id")
        .first()
    )


def on_day(exc, victim):
    return [r for r in exc if (r["store_id"], r["business_date"]) == (victim["store_id"], victim["posting_date"])]


def test_flags_exactly_the_planted_journals(clean, planted):
    silver, _, _, fx = clean
    _, _, exc = run(silver, fx)
    found = {r["journal_id"] for r in exc if r["journal_id"]}
    assert len(planted) == SMALL.a3_count
    assert score(found, planted) == {"planted": 4, "found": 4, "missed": 0, "false_positives": 0}
    assert {r["reason"] for r in exc} == {UNSUPPORTED}
    assert all(r["day_unexplained_eur"] == 0 for r in exc)  # each difference fully explained


def test_every_other_store_day_matches_and_months_add_up(clean, planted):
    silver, _, _, fx = clean
    daily, monthly, _ = run(silver, fx)
    diff_days = daily.where("status = 'difference'").count()
    assert 1 <= diff_days <= len(planted)  # two planted journals can share a store-day
    assert daily.where("status = 'matched'").where(F.abs("difference_eur") > 0).count() == 0
    # Monthly differences are exactly the planted amounts, per store and month.
    fakes = silver["gl_journal"].where(F.col("journal_id").isin(*planted)).where("account_code = '4000'")
    expected = {
        (r["store_id"], r["m"]): r["amt"]
        for r in fakes.groupBy("store_id", F.date_format("posting_date", "yyyy-MM").alias("m"))
        .agg(F.sum("credit_eur").alias("amt"))
        .collect()
    }
    got = {(r["store_id"], r["month"]): r["difference_eur"] for r in monthly.where("status = 'difference'").collect()}
    assert got == expected


def test_fake_equal_to_the_pos_total_is_reported_as_duplicate(spark, clean, planted):
    """Labels are not trusted and ties are not broken alphabetically: both journals are reported."""
    silver, _, _, fx = clean
    gl = silver["gl_journal"]
    victim = clean_pos_journal(silver, planted)
    fake = journal_like(
        spark,
        gl.where(F.col("journal_id") == victim["journal_id"]),
        journal_id="J-AAA-FAKE",
        credit_eur=victim["credit_eur"],
        debit_eur=victim["debit_eur"],
    )
    _, _, exc = run(silver, fx, gl.unionByName(fake))
    day = on_day(exc, victim)
    assert {r["journal_id"] for r in day} == {"J-AAA-FAKE", victim["journal_id"]}
    assert {r["reason"] for r in day} == {DUPLICATE}


def test_fake_on_a_day_without_sales_is_unsupported(spark, clean):
    silver, _, _, fx = clean
    gl = silver["gl_journal"]
    store = silver["stores"].where("format <> 'online' AND country = 'DE'").orderBy("store_id").first()["store_id"]
    sunday = date(2026, 8, 2)  # German stores are closed on Sundays
    assert sunday.isoweekday() == 7
    fake = journal_like(
        spark,
        gl,
        journal_id="J-SUNDAY-FAKE",
        store_id=store,
        posting_date=sunday,
        credit_eur=Decimal("999.00"),
        debit_eur=Decimal("0.00"),
    )
    _, _, exc = run(silver, fx, gl.unionByName(fake))
    row = [r for r in exc if r["journal_id"] == "J-SUNDAY-FAKE"]
    assert len(row) == 1 and row[0]["reason"] == UNSUPPORTED and row[0]["day_pos_net_eur"] == 0


def test_sales_missing_from_the_books_are_reported(clean, planted):
    silver, _, _, fx = clean
    victim = clean_pos_journal(silver, planted)
    _, _, exc = run(silver, fx, silver["gl_journal"].where(F.col("journal_id") != victim["journal_id"]))
    day = on_day(exc, victim)
    assert len(day) == 1
    assert day[0]["journal_id"] is None and day[0]["reason"] == UNATTRIBUTED
    assert day[0]["day_difference_eur"] < 0


def test_no_innocent_journal_is_blamed_when_nothing_matches(clean, planted):
    """A POS journal booked EUR 5 too high: report the day, do not accuse the journal."""
    silver, _, _, fx = clean
    victim = clean_pos_journal(silver, planted)
    hit = (F.col("journal_id") == victim["journal_id"]) & (F.col("line_no") == victim["line_no"])
    gl = silver["gl_journal"].withColumn(
        "credit_eur", F.when(hit, F.col("credit_eur") + F.lit(Decimal("5.00"))).otherwise(F.col("credit_eur"))
    )
    _, _, exc = run(silver, fx, gl)
    day = on_day(exc, victim)
    assert len(day) == 1 and day[0]["journal_id"] is None and day[0]["day_difference_eur"] == Decimal("5.00")


def test_refuses_incomplete_inputs(clean):
    silver, _, _, fx = clean
    pos, gl = silver["pos_sales"], silver["gl_journal"]
    with pytest.raises(RuntimeError, match="no POS sales"):
        check_inputs(pos.limit(0), gl, fx)
    with pytest.raises(RuntimeError, match="without an FX rate"):
        check_inputs(pos, gl, fx.where("currency <> 'CHF'"))
    with pytest.raises(RuntimeError, match="quarantine"):
        check_inputs(pos, gl, fx, quarantined=3)
    check_inputs(pos, gl, fx)  # complete inputs pass


def test_answer_key(spark, tmp_path, landing, planted):
    assert read_answer_key(spark, str(tmp_path / "missing.csv")) is None
    assert read_answer_key(spark, str(landing / "_ground_truth" / "planted_records.csv")) == planted
    other = tmp_path / "key.csv"
    other.write_text("anomaly,entity_type,entity_id,detail\nA1_no_receipt_refunds,refund,R1,x\n")
    with pytest.raises(RuntimeError, match="no A3"):
        read_answer_key(spark, str(other))


def test_score():
    assert score({"a", "x"}, {"a", "b"}) == {"planted": 2, "found": 1, "missed": 1, "false_positives": 1}
