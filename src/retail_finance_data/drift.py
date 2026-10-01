"""Model-input drift (story 6.2): does this month's data still look like what the model knows?

PSI (population stability index) compares two distributions of one feature: split the
reference into quantile bins, then sum (current share - reference share) * ln(current / reference)
over the bins. Rule of thumb: below 0.1 stable, 0.1-0.2 watch, above 0.2 drift.

PSI sees a shift of the whole population (a chain-wide promotion, a pipeline bug that halves
refunds). One cashier or one store barely moves it: finding those is the detectors' job.
Drift is a warning, never a reason to stop the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GAP, WINDOW, MIN_REF_MONTHS = 4, 9, 3  # reference = months 4-12 before the month checked
EPS = 1e-4  # floor for empty bins, so ln() stays finite


def psi(reference, current, bins: int | None = None) -> float:
    """PSI of ``current`` against ``reference``. By default about 10 current rows per bin
    (2 to 10 bins): with few rows, many bins measure sampling noise. 40 stores in 10 bins read
    about 0.2 on identical distributions, i.e. "drift" every month."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    bins = bins or int(np.clip(len(cur) // 10, 2, 10))
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant reference: "same value" vs "anything else"
        same = float(np.isclose(cur, ref[0]).mean())
        r, c = np.array([1.0, 0.0]), np.array([same, 1.0 - same])
    else:
        inner = edges[1:-1]
        r = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=len(inner) + 1) / len(ref)
        c = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=len(inner) + 1) / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def status(value: float) -> str:
    if np.isnan(value):
        return "no reference"
    return "stable" if value < 0.1 else "watch" if value < 0.2 else "drift"


def feature_drift(features: pd.DataFrame, cols: list[str], months: list[str]) -> pd.DataFrame:
    """PSI of each feature for each month in ``months``, against that month's own reference
    window (months ``GAP``..``GAP + WINDOW - 1`` earlier, the same gap as the margin baseline)."""
    f = features.assign(month=features["month"].astype(str))
    known = sorted(f["month"].unique())
    rows = []
    for m in months:
        i = known.index(m)
        ref_months = known[max(0, i - GAP - WINDOW + 1) : max(0, i - GAP + 1)]
        ref = f[f["month"].isin(ref_months)]
        cur = f[f["month"] == m]
        for col in cols:
            value = psi(ref[col], cur[col]) if len(ref_months) >= MIN_REF_MONTHS else float("nan")
            rows.append(
                {
                    "month": m,
                    "feature": col,
                    "psi": round(value, 4),  # NaN when there is no reference window
                    "status": status(value),
                    "reference": f"{ref_months[0]}..{ref_months[-1]}" if ref_months else "",
                }
            )
    return pd.DataFrame(rows)


def sales_growth(monthly: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year growth per store and month: the forecast's input without its seasonality
    (raw sales would 'drift' every December)."""
    m = monthly.assign(month=pd.PeriodIndex(monthly["month"].astype(str), freq="M"))
    last_year = m.assign(month=m["month"] + 12).rename(columns={"net_sales_eur": "last_year_eur"})
    g = m.merge(last_year, on=["store_id", "month"], how="inner")
    g["yoy_growth"] = g["net_sales_eur"] / g["last_year_eur"].where(g["last_year_eur"] > 0) - 1
    return g.assign(month=g["month"].astype(str))[["store_id", "month", "yoy_growth"]]
