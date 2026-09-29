"""Stage 5 models (stories 5.1, 5.2): planted patterns found without labels; forecast chosen honestly."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from retail_finance_data.jobs import ml  # noqa: E402

RNG = np.random.default_rng(7)
MONTHS = [f"2026-{m:02d}" for m in range(1, 10)]


def refunds_table(fraud_cashier="S005-C03", fraud_months=("2026-08", "2026-09")):
    rows = []
    for s in range(1, 21):
        for c in range(1, 7):
            for m in MONTHS:
                n = int(RNG.poisson(8)) + 1
                nr = int(RNG.binomial(n, 0.1))
                gross = float(n * RNG.uniform(20, 60))
                nr_eur = gross * nr / n
                cid = f"S{s:03d}-C{c:02d}"
                if cid == fraud_cashier and m in fraud_months:
                    nr, n = nr + 60, n + 60
                    nr_eur, gross = nr_eur + 6000.0, gross + 6000.0
                rows.append((f"S{s:03d}", cid, m, n, gross, nr, nr_eur))
    return pd.DataFrame(
        rows,
        columns=[
            "store_id",
            "cashier_id",
            "month",
            "refunds",
            "refund_gross_eur",
            "no_receipt_refunds",
            "no_receipt_eur",
        ],
    )


def margin_table(creep_store="S007", start="2026-06"):
    rows = []
    for s in range(1, 21):
        for m in MONTHS:
            promo = 0.02 if m == "2026-07" else 0.0  # chain-wide promotion: must not flag everyone
            extra = (
                0.02 * (MONTHS.index(m) - MONTHS.index(start) + 1) if f"S{s:03d}" == creep_store and m >= start else 0.0
            )
            for cat in ("Grocery", "Fresh"):
                paid = 10000 * RNG.uniform(0.95, 1.05)
                disc = paid * (0.05 + promo + extra + RNG.normal(0, 0.003))
                net = paid / 1.19
                cogs = net * (0.65 + promo + extra * 0.9 + RNG.normal(0, 0.004))
                rows.append((f"S{s:03d}", cat, m, net, cogs, disc, paid))
    return pd.DataFrame(
        rows,
        columns=["store_id", "category", "month", "net_sales_eur", "cogs_eur", "discount_eur", "paid_incl_vat_eur"],
    )


def test_the_model_sees_only_its_feature_columns_never_the_answer_key():
    """Even if an answer-key column sits in the frame, the detector's input is exactly its features."""
    table = refunds_table()
    table["planted"] = table["cashier_id"].eq("S005-C03")
    model, _, _ = ml.isolation_scores(ml.cashier_features(table), ml.CASHIER_FEATURES, contamination=0.005)
    assert model.n_features_in_ == len(ml.CASHIER_FEATURES)
    assert not {"planted", "anomaly", "cashier_id", "store_id"} & set(ml.CASHIER_FEATURES + ml.MARGIN_FEATURES)


def test_cashier_detector_ranks_the_fraudster_first():
    cf = ml.cashier_features(refunds_table())
    _, score, flag = ml.isolation_scores(cf, ml.CASHIER_FEATURES, contamination=0.005)
    scores = ml.ranked(cf, score, flag, ["store_id", "cashier_id"])
    ev = ml.evaluate_detector(scores, "cashier_id", "S005-C03", ["2026-08", "2026-09"], top=1)
    assert ev["in_top"], ev


def test_margin_detector_finds_the_creeping_store_not_the_chain_promotion():
    mf, dropped = ml.margin_features(margin_table())
    assert dropped > 0  # first months of each store have no history: counted, not hidden
    _, score, flag = ml.isolation_scores(mf, ml.MARGIN_FEATURES, contamination=0.02)
    alerts = ml.ranked(mf, score, flag, ["store_id"])
    ev = ml.evaluate_detector(alerts, "store_id", "S007", ["2026-08", "2026-09"])
    assert ev["in_top"], ev
    july = alerts[alerts["month"] == "2026-07"]
    assert july["flagged"].sum() <= 2  # a chain-wide promotion is netted out
    assert set(ev["false_flags_by_month"]) == {"2026-08", "2026-09"}


def daily_table():
    rows = []
    for s in range(1, 11):
        level = 1000 * (1 + s / 10)
        for d in pd.date_range("2025-01-01", "2026-09-25"):
            season = 1 + 0.3 * (d.month == 12) - 0.1 * (d.month in (1, 2))
            rows.append((f"S{s:03d}", d.date(), level * season * (1.03 ** (d.year - 2025)) * RNG.uniform(0.9, 1.1)))
    return pd.DataFrame(rows, columns=["store_id", "business_date", "net_sales_eur"])


def test_baseline_is_last_year_times_recent_growth():
    panel = ml.monthly_sales(daily_table())
    p = ml.baseline_forecast(panel[panel["month"] <= pd.Period("2026-05", "M")], pd.Period("2026-06", "M"))
    h = panel.pivot(index="store_id", columns="month", values="net_sales_eur")
    rec = [pd.Period(m, "M") for m in ("2026-05", "2026-04", "2026-03")]
    expected = h[pd.Period("2025-06", "M")] * h[rec].sum(axis=1) / h[[m - 12 for m in rec]].sum(axis=1)
    pd.testing.assert_series_equal(p, expected, check_names=False)


def test_backtest_picks_the_better_method_and_forecast_has_ordered_ranges():
    panel = ml.monthly_sales(daily_table())
    bt = ml.backtest(panel, pd.Period("2026-08", "M"))
    assert bt["winner"] in ("model", "baseline")
    assert bt["winner"] == ("model" if bt["model_mape"] < bt["baseline_mape"] else "baseline")
    assert set(bt["train_rows_by_horizon"]) == {1, 2, 3}  # every horizon is trained, not just the easy one
    fc, model = ml.forecast(panel, pd.Period("2026-08", "M"), bt["winner"], bt["errors"][bt["winner"]])
    assert (model is None) == (bt["winner"] == "baseline")
    assert set(fc["month"]) == {"2026-09", "2026-10", "2026-11"}
    assert (fc["low_80_eur"] <= fc["forecast_eur"]).all() and (fc["forecast_eur"] <= fc["high_80_eur"]).all()
    assert len(fc) == 30


def test_ranks_are_stable_across_reruns():
    cf = ml.cashier_features(refunds_table())
    runs = [
        ml.ranked(cf, *ml.isolation_scores(cf, ml.CASHIER_FEATURES)[1:], ["store_id", "cashier_id"]) for _ in range(2)
    ]
    pd.testing.assert_frame_equal(runs[0], runs[1])
