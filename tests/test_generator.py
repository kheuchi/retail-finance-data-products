"""Properties the synthetic data must have for the finance use cases to mean anything."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from retail_finance_data.calendar import easter_sunday, public_holidays
from retail_finance_data.generator import Config, build, make_budget

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


@pytest.fixture(scope="module")
def data():
    out = {}
    for name, _period, df in build(SMALL):
        out.setdefault(name, []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def test_same_seed_same_data():
    a = [df for _, _, df in build(SMALL)]
    b = [df for _, _, df in build(SMALL)]
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(x, y)


def test_every_journal_balances(data):
    gl = data["gl_journal"]
    diff = gl.groupby("journal_id")[["debit_eur", "credit_eur"]].sum().round(2)
    assert (diff["debit_eur"] == diff["credit_eur"]).all()


def test_gl_revenue_reconciles_to_pos(data):
    """Account 4000 posted from POS equals POS net sales in EUR, per store and day."""
    gl = data["gl_journal"]
    pos_gl = (
        gl[(gl.account_code == "4000") & (gl.source == "POS")].groupby(["posting_date", "store_id"])["credit_eur"].sum()
    )
    fx = data["fx_rates"]
    chf = fx[fx.currency == "CHF"].set_index("rate_date")["units_per_eur"]
    s = data["pos_sales"]
    days = pd.Index(sorted(s.business_date.unique()))
    rate = chf.reindex(chf.index.union(days)).ffill().reindex(days)
    net_eur = np.where(s.currency == "EUR", s.net_amount, s.net_amount / rate.reindex(s.business_date).to_numpy())
    pos = s.assign(net_eur=net_eur).groupby(["business_date", "store_id"])["net_eur"].sum().round(2)
    pos.index.names = pos_gl.index.names
    pd.testing.assert_series_equal(pos_gl.sort_index(), pos.sort_index(), check_names=False, atol=0.01)


def test_physical_stores_closed_on_sundays_and_holidays(data):
    s, stores = data["pos_sales"], data["stores"]
    phys = s.merge(stores[["store_id", "format", "country"]], on="store_id")
    phys = phys[phys.format != "online"]
    d = pd.to_datetime(phys.business_date)
    assert not (d.dt.dayofweek == 6).any()
    for c in ("DE", "CH"):
        hol = {h.isoformat() for h in public_holidays(c, [2026])}
        assert not phys[(phys.country == c) & phys.business_date.isin(hol)].shape[0]


def test_receipted_refunds_point_at_real_sales(data):
    r, s = data["refunds"], data["pos_sales"]
    receipted = r[r.receipt_present]
    assert len(receipted) > 0
    assert receipted.original_transaction_id.isin(s.transaction_id).all()
    assert r[~r.receipt_present].original_transaction_id.isna().all()


def test_planted_anomalies_are_present_and_labelled(data):
    truth = data["_ground_truth/planted_records"]
    a1_ids = set(truth.loc[truth.anomaly == "A1_no_receipt_refunds", "entity_id"])
    r = data["refunds"]
    planted = r[r.refund_id.isin(a1_ids)]
    assert len(planted) >= 10
    assert (planted.cashier_id == SMALL.a1_cashier).all()
    assert (~planted.receipt_present).all()
    assert (planted.business_date >= SMALL.a1_start.isoformat()).all()

    manual = data["gl_journal"].query("source == 'MANUAL'").journal_id.unique()
    assert len(manual) == SMALL.a3_count
    assert set(manual) == set(truth.loc[truth.anomaly == "A3_unsupported_manual_revenue", "entity_id"])

    s = data["pos_sales"]
    late = s.business_date >= "2026-08-10"
    disc = (s.discount_amount / (s.quantity * s.unit_price))[late]
    assert disc[s.store_id[late] == SMALL.a2_store].mean() > disc[s.store_id[late] != SMALL.a2_store].mean()


def test_budget_is_prior_year_plus_growth():
    sm = pd.DataFrame(
        {
            "store_id": ["S001", "S001"],
            "year": [2025, 2025],
            "month": [1, 2],
            "net_eur": [100_000.0, 50_000.0],
            "cogs_eur": [60_000.0, 30_000.0],
        }
    )
    b = make_budget(Config(end=date(2026, 9, 25)), sm)
    assert list(b.budget_month) == ["2026-01", "2026-02"]
    assert list(b.revenue_budget_eur) == [105_000.0, 52_500.0]


def test_easter():
    assert easter_sunday(2025) == date(2025, 4, 20)
    assert easter_sunday(2026) == date(2026, 4, 5)
