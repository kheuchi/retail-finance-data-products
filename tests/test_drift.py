"""Story 6.2: PSI flags a shifted population, not a stable one, and never crashes on odd inputs."""

import numpy as np
import pandas as pd

from retail_finance_data import drift

RNG = np.random.default_rng(11)


def test_same_distribution_is_stable_and_a_shift_is_drift():
    ref = RNG.normal(0, 1, 5000)
    assert drift.psi(ref, RNG.normal(0, 1, 5000)) < 0.1
    assert drift.psi(ref, RNG.normal(1, 1, 5000)) > 0.2


def test_small_populations_are_not_flagged_by_noise():
    # 40 stores a month, same distribution: 10 fixed bins would read about 0.2 ("drift")
    values = [drift.psi(RNG.normal(0, 1, 360), RNG.normal(0, 1, 40)) for _ in range(200)]
    assert np.median(values) < 0.1
    assert drift.psi(RNG.normal(0, 1, 360), RNG.normal(1.5, 1, 40)) > 0.2


def test_mostly_zero_feature_still_sees_a_shift():
    ref = np.where(RNG.random(1000) < 0.92, 0.0, RNG.uniform(0.1, 1, 1000))  # 92% zeros
    assert drift.psi(ref, np.where(RNG.random(300) < 0.92, 0.0, RNG.uniform(0.1, 1, 300))) < 0.1
    assert drift.psi(ref, RNG.uniform(0.1, 1, 300)) > 0.2  # every current value non-zero


def test_constant_or_empty_inputs_do_not_crash():
    zeros = np.zeros(100)
    assert drift.psi(zeros, zeros) == 0
    assert drift.psi(zeros, np.ones(100)) > 0.2
    assert np.isnan(drift.psi([], [1.0]))


def test_each_month_is_compared_with_its_months_4_to_12_back():
    months = [str(p) for p in pd.period_range("2025-01", "2026-09", freq="M")]
    df = pd.DataFrame([{"month": m, "x": RNG.normal(0, 1)} for m in months for _ in range(200)])
    df.loc[df["month"] == "2026-09", "x"] += 2  # the whole population moves in September
    out = drift.feature_drift(df, ["x"], ["2025-03", "2026-08", "2026-09"]).set_index("month")
    assert out.loc["2025-03", "status"] == "no reference"
    assert out.loc["2026-08", "reference"] == "2025-08..2026-04 (9 months)"
    assert out.loc["2026-08", "status"] == "stable"
    assert out.loc["2026-09", "status"] == "drift"


def test_sales_growth_removes_seasonality():
    rows = []
    for year, level in ((2025, 100.0), (2026, 110.0)):
        for month in range(1, 13):
            season = 2.0 if month == 12 else 1.0
            rows.append({"store_id": "S001", "month": f"{year}-{month:02d}", "net_sales_eur": level * season})
    g = drift.sales_growth(pd.DataFrame(rows))
    assert len(g) == 12 and np.allclose(g["yoy_growth"], 0.10)


def test_reference_window_is_calendar_months_even_with_gaps():
    gaps = ("2025-10", "2025-11")
    months = [str(p) for p in pd.period_range("2025-01", "2026-09", freq="M") if str(p) not in gaps]
    df = pd.DataFrame([{"month": m, "x": RNG.normal(0, 1)} for m in months for _ in range(100)])
    out = drift.feature_drift(df, ["x"], ["2026-08"]).iloc[0]
    assert out["reference"] == "2025-08..2026-04 (7 months)"  # the gap is not filled from earlier months
