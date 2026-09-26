"""Silver rules, run with local Spark on files written exactly as the landing volume gets them."""

from datetime import datetime

import pandas as pd
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from test_generator import SMALL  # noqa: E402

from retail_finance_data.jobs.generate import write_all  # noqa: E402
from retail_finance_data.jobs.silver import (  # noqa: E402
    ORDER,
    SPECS,
    apply_spec,
    check_counts,
    count_stats,
    daily_fx,
)


@pytest.fixture(scope="module")
def spark():
    s = (
        SparkSession.builder.master("local[2]")
        .appName("silver-tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield s
    s.stop()


@pytest.fixture(scope="module")
def bronze(spark, tmp_path_factory):
    """Bronze as Auto Loader leaves it: every column a string, plus the audit columns."""
    out = tmp_path_factory.mktemp("landing")
    write_all(SMALL, str(out))
    return {
        s: spark.read.option("header", "true")
        .csv(str(out / s))
        .select(
            "*",
            F.lit(None).cast("string").alias("_rescued_data"),
            F.col("_metadata.file_path").alias("_source_file"),
            F.lit(datetime(2026, 9, 26, 12)).alias("_ingested_at"),
        )
        for s in ORDER
    }


def build_all(bronze, override=None):
    """Run every source in job order; ``override`` swaps in a modified Bronze DataFrame."""
    override = override or {}
    built, fx, stats, quarantine = {}, None, {}, {}
    for s in ORDER:
        source = override.get(s, bronze[s])
        checked, silver, q = apply_spec(source, SPECS[s], built, fx)
        built[s], stats[s], quarantine[s] = silver, count_stats(source, checked, silver, q), q
        if s == "fx_rates":
            fx = daily_fx(silver, "2026-09-30")
    return built, stats, quarantine, fx


@pytest.fixture(scope="module")
def clean(bronze):
    return build_all(bronze)


def with_rows(spark, df, changes):
    """``df`` plus copies of its first row, each updated with one dict of ``changes``."""
    base = df.limit(1).collect()[0].asDict()
    rows = [{**base, **c} for c in changes]
    return df.unionByName(spark.createDataFrame(rows, schema=df.schema))


def reasons(q):
    return {r["record_key"]: set(r["reasons"]) for r in q.collect()}


def test_clean_data_passes_untouched(clean):
    _, stats, _, _ = clean
    for s, st in stats.items():
        assert st["quarantined"] == 0, (s, st)
        assert st["duplicates"] == 0, (s, st)
        assert st["silver"] == st["bronze"], (s, st)
        assert st["bronze"] > 0 or s == "budget", (s, st)  # the small config has no prior year to budget from
        check_counts(s, st)


def test_columns_get_real_types(clean):
    built, _, _, _ = clean
    types = dict(built["pos_sales"].dtypes)
    assert types["business_date"] == "date"
    assert types["transaction_ts"] == "timestamp"
    assert types["quantity"] == "int"
    assert types["net_amount"] == "decimal(18,2)"
    assert types["promo_flag"] == "boolean"
    assert types["net_amount_eur"] == "decimal(18,4)"


def test_chf_converted_at_last_published_rate(clean):
    built, _, _, _ = clean
    from importlib import resources

    with resources.files("retail_finance_data.reference").joinpath("ecb_fx_rates.csv").open("r") as f:
        ecb = pd.read_csv(f, parse_dates=["rate_date"])
    chf = ecb.loc[ecb["currency"] == "CHF"].set_index("rate_date")["units_per_eur"].sort_index()

    rows = (
        built["pos_sales"].where("currency = 'CHF'").select("business_date", "net_amount", "net_amount_eur").collect()
    )
    assert rows
    saturdays = 0
    for r in rows:
        day = pd.Timestamp(r["business_date"])
        rate = chf.loc[:day].iloc[-1]  # last rate published on or before the day
        saturdays += day.dayofweek == 5
        assert float(r["net_amount_eur"]) == pytest.approx(round(float(r["net_amount"]) / rate, 4), abs=1e-4)
    assert saturdays, "expected Swiss Saturday sales, where the ECB publishes no rate"

    eur = built["pos_sales"].where("currency = 'EUR'").where("net_amount_eur <> net_amount").count()
    assert eur == 0


def test_bad_sales_rows_are_quarantined_with_reasons(spark, bronze):
    first = bronze["pos_sales"].limit(1).collect()[0]
    later = datetime(2026, 9, 27)
    bad = with_rows(
        spark,
        bronze["pos_sales"],
        [
            {"_ingested_at": later},  # same key again: a duplicate, the later copy wins
            {"transaction_id": "BAD-QTY", "quantity": "-1"},
            {"transaction_id": "BAD-DATE", "business_date": "2026-13-45"},
            {"transaction_id": "BAD-STORE", "store_id": "S999"},
            {"transaction_id": "BAD-SUM", "gross_amount": "999.99"},
            {"transaction_id": "BAD-SKU", "sku": None},
        ],
    )
    built, stats, quarantine, _ = build_all(bronze, {"pos_sales": bad})
    st = stats["pos_sales"]
    check_counts("pos_sales", st)
    assert st["duplicates"] == 1
    assert st["quarantined"] == 5
    assert st["silver"] == bronze["pos_sales"].count()

    by_id = {k.split("|")[0]: v for k, v in reasons(quarantine["pos_sales"]).items()}
    assert "rule:quantity_positive" in by_id["BAD-QTY"]
    assert "bad_type:business_date" in by_id["BAD-DATE"]
    assert "ref:store_id" in by_id["BAD-STORE"]
    assert "rule:gross_equals_net_plus_vat" in by_id["BAD-SUM"]
    assert "missing:sku" in by_id["BAD-SKU"]

    kept = built["pos_sales"].where(
        (F.col("transaction_id") == first["transaction_id"]) & (F.col("line_no") == int(first["line_no"]))
    )
    assert kept.count() == 1
    assert kept.first()["_ingested_at"] == later


def test_refund_receipts_must_point_at_real_sales(spark, bronze):
    receipted = bronze["refunds"].where("receipt_present = 'True'")
    bad = with_rows(
        spark,
        receipted,
        [
            {"refund_id": "R-NO-ORIGINAL", "original_transaction_id": None},
            {"refund_id": "R-GHOST-SALE", "original_transaction_id": "T20990101-S001-00001"},
        ],
    )
    _, stats, quarantine, _ = build_all(bronze, {"refunds": bad})
    check_counts("refunds", stats["refunds"])
    got = reasons(quarantine["refunds"])
    assert "rule:receipt_means_original" in got["R-NO-ORIGINAL"]
    assert "ref:original_transaction_id" in got["R-GHOST-SALE"]


def test_count_check_catches_lost_rows():
    with pytest.raises(AssertionError):
        check_counts("x", {"bronze": 10, "silver": 8, "quarantined": 1, "duplicates": 0})
