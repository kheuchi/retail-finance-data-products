"""Job: ingest landing files into Bronze tables with Auto Loader.

Bronze keeps the data exactly as delivered: every column is a string, nothing is
cleaned, and two audit columns record which file each row came from and when it
landed. Typing and validation happen in Silver.

Auto Loader tracks which files it has already processed, so re-running this job
only picks up new files. It runs with availableNow: process what is there, then stop.
"""

from __future__ import annotations

import argparse

SOURCES = ["stores", "products", "cashiers", "fx_rates", "pos_sales", "refunds", "gl_journal", "budget"]


def ingest(spark, catalog: str, source: str) -> None:
    from pyspark.sql import functions as F

    landing = f"/Volumes/{catalog}/raw/landing"
    system = f"{landing}/_system/{source}"
    stream = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("cloudFiles.inferColumnTypes", "false")  # Bronze stays untyped
        .option("cloudFiles.schemaLocation", f"{system}/schema")
        .option("rescuedDataColumn", "_rescued_data")
        .load(f"{landing}/{source}/")
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingested_at", F.current_timestamp())
    )
    (
        stream.writeStream.option("checkpointLocation", f"{system}/checkpoint")
        .trigger(availableNow=True)
        .toTable(f"{catalog}.bronze.{source}")
        .awaitTermination()
    )
    n = spark.table(f"{catalog}.bronze.{source}").count()
    print(f"bronze.{source:<12} {n:>10,} rows", flush=True)


def main() -> None:
    from pyspark.sql import SparkSession

    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="finance")
    p.add_argument("--sources", default=",".join(SOURCES))
    a = p.parse_args()
    spark = SparkSession.builder.getOrCreate()
    for source in a.sources.split(","):
        ingest(spark, a.catalog, source)


if __name__ == "__main__":
    main()
