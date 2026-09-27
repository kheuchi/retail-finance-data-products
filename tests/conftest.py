"""Shared Spark fixtures: one local session and one generated dataset for all Spark tests."""

from datetime import datetime

import pytest

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
except ImportError:  # the generator tests still run without pyspark
    SparkSession = None

from test_generator import SMALL

from retail_finance_data.jobs.generate import write_all


def build_all(bronze, override=None):
    """Run every Silver source in job order; ``override`` swaps in a modified Bronze DataFrame."""
    from retail_finance_data.jobs.silver import ORDER, SPECS, apply_spec, count_stats, daily_fx

    override = override or {}
    built, fx, stats, quarantine = {}, None, {}, {}
    for s in ORDER:
        source = override.get(s, bronze[s])
        checked, silver, q = apply_spec(source, SPECS[s], built, fx)
        built[s], stats[s], quarantine[s] = silver, count_stats(source, checked, silver, q), q
        if s == "fx_rates":
            fx = daily_fx(silver, "2026-07-01", "2026-09-30")
    return built, stats, quarantine, fx


def with_rows(spark, df, changes):
    """``df`` plus copies of its first row, each updated with one dict of ``changes``."""
    base = df.limit(1).collect()[0].asDict()
    rows = [{**base, **c} for c in changes]
    return df.unionByName(spark.createDataFrame(rows, schema=df.schema))


@pytest.fixture(scope="session")
def spark():
    if SparkSession is None:
        pytest.skip("pyspark not installed")
    s = (
        SparkSession.builder.master("local[2]")
        .appName("tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield s
    s.stop()


@pytest.fixture(scope="session")
def landing(tmp_path_factory):
    out = tmp_path_factory.mktemp("landing")
    write_all(SMALL, str(out))
    return out


@pytest.fixture(scope="session")
def bronze(spark, landing):
    """Bronze as Auto Loader leaves it: every column a string, plus the audit columns."""
    from retail_finance_data.jobs.silver import ORDER

    return {
        s: spark.read.option("header", "true")
        .csv(str(landing / s))
        .select(
            "*",
            F.lit(None).cast("string").alias("_rescued_data"),
            F.col("_metadata.file_path").alias("_source_file"),
            F.lit(datetime(2026, 9, 26, 12)).alias("_ingested_at"),
        )
        for s in ORDER
    }


@pytest.fixture(scope="session")
def clean(bronze):
    return build_all(bronze)
