"""The small synthetic dataset every test builds: six stores, four weeks, all three planted anomalies."""

from datetime import date

from retail_finance_data.source_simulator.generator import Config

SMALL = Config(
    n_stores=6,
    n_swiss_stores=1,
    n_products=80,
    base_txn_per_day=20,
    start=date(2026, 7, 25),
    end=date(2026, 8, 20),
    a1_store="S002",
    a1_cashier="S002-C03",
    a1_start=date(2026, 8, 1),
    a2_store="S003",
    a2_start=date(2026, 8, 1),
    a3_month=(2026, 8),
    a3_count=4,
)
