"""GL vs POS reconciliation (story 4.2): it must flag exactly the planted A3 journals."""

from decimal import Decimal

import pandas as pd
import pytest

pytest.importorskip("pyspark")

from conftest import with_rows  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from test_generator import SMALL  # noqa: E402

from retail_finance_data.generator import build  # noqa: E402
from retail_finance_data.jobs.reconcile import ANOMALY, reconcile, score  # noqa: E402


@pytest.fixture(scope="module")
def planted():
    truth = pd.concat(df for name, _, df in build(SMALL) if name == "_ground_truth/planted_records")
    return set(truth.loc[truth["anomaly"] == ANOMALY, "entity_id"])


def run(silver, fx, gl=None):
    daily, monthly, exceptions = reconcile(silver["pos_sales"], gl if gl is not None else silver["gl_journal"], fx)
    return daily, monthly, exceptions.collect()


def test_flags_exactly_the_planted_journals(clean, planted):
    silver, _, _, fx = clean
    daily, monthly, exc = run(silver, fx)
    found = {r["journal_id"] for r in exc if r["journal_id"]}
    assert len(planted) == SMALL.a3_count
    assert score(found, planted) == {"planted": 4, "found": 4, "missed": 0, "false_positives": 0}
    assert all(r["reason"] == "no matching POS sales" for r in exc)
    assert all(r["unexplained_eur"] == 0 for r in exc)  # each difference fully explained by its journal


def test_every_other_store_day_matches_to_the_cent(clean, planted):
    silver, _, _, fx = clean
    daily, monthly, _ = run(silver, fx)
    diff_days = daily.where("status = 'difference'").count()
    assert daily.count() > 100
    assert diff_days <= len(planted)  # two planted journals can share a store-day
    assert daily.where("status = 'matched'").where(F.abs("difference_eur") > 0).count() == 0
    assert monthly.where("status = 'difference'").count() >= 1


def test_a_disguised_journal_is_still_caught(spark, clean, planted):
    """Labels are not trusted: a fake journal calling itself POS must still be flagged."""
    silver, _, _, fx = clean
    gl = silver["gl_journal"]
    fake = with_rows(
        spark,
        gl.where("account_code = '4000' AND source = 'POS'"),
        [{"journal_id": "J-FAKE-POS", "line_no": 1, "credit_eur": Decimal("500.00"), "debit_eur": Decimal("0.00")}],
    ).where("journal_id = 'J-FAKE-POS'")
    _, _, exc = run(silver, fx, gl.unionByName(fake))
    found = {r["journal_id"] for r in exc if r["journal_id"]}
    assert "J-FAKE-POS" in found
    assert found - planted == {"J-FAKE-POS"}


def test_sales_missing_from_the_books_are_reported(clean):
    silver, _, _, fx = clean
    gl = silver["gl_journal"]
    victim = gl.where("source = 'POS'").select("journal_id", "store_id", "posting_date").first()
    _, _, exc = run(silver, fx, gl.where(F.col("journal_id") != victim["journal_id"]))
    row = [r for r in exc if r["store_id"] == victim["store_id"] and r["business_date"] == victim["posting_date"]]
    assert len(row) == 1
    assert row[0]["journal_id"] is None
    assert row[0]["reason"] == "difference not attributable to a journal"
    assert row[0]["difference_eur"] < 0


def test_score():
    assert score({"a", "x"}, {"a", "b"}) == {"planted": 2, "found": 1, "missed": 1, "false_positives": 1}
