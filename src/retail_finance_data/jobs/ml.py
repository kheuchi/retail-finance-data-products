"""Job: fraud detectors and revenue forecast on certified Gold (stories 5.1, 5.2).

Everything here reads Gold only, and only when the latest Gold build is certified
(``ops.gold_certification``). The Gold tables are small (thousands of rows), so models are
trained in pandas and scikit-learn on the job cluster; Spark only reads and writes tables.

Honesty rules:
- The detectors are unsupervised and never see the answer key; the planted anomalies are
  used only afterwards, to score them.
- The forecast ships only if it beats a seasonal-naive baseline in a backtest; otherwise the
  baseline ships, and the scores say so.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

SEED = 20260929

# ---------------------------------------------------------------- 5.1 detectors

CASHIER_FEATURES = [
    "log_refunds",
    "log_no_receipt_eur",
    "no_receipt_share",
    "peer_no_receipt_ratio",
    "peer_refunds_ratio",
]
MARGIN_FEATURES = ["margin_drop_vs_own", "discount_rise_vs_own"]


def cashier_features(refunds: pd.DataFrame) -> pd.DataFrame:
    """One row per cashier-month, compared with the other cashiers of the same store that month."""
    r = refunds.copy()
    for c in ("refunds", "no_receipt_refunds"):
        r[c] = r[c].astype(float)
    for c in ("no_receipt_eur", "refund_gross_eur"):
        r[c] = r[c].astype(float)
    r["no_receipt_share"] = (r["no_receipt_eur"] / r["refund_gross_eur"].where(r["refund_gross_eur"] > 0)).fillna(0.0)
    peers = r.groupby(["store_id", "month"])
    r["peer_no_receipt_ratio"] = np.log1p(r["no_receipt_eur"]) - np.log1p(peers["no_receipt_eur"].transform("median"))
    r["peer_refunds_ratio"] = np.log1p(r["refunds"]) - np.log1p(peers["refunds"].transform("median"))
    r["log_refunds"] = np.log1p(r["refunds"])
    r["log_no_receipt_eur"] = np.log1p(r["no_receipt_eur"])
    return r


def margin_features(margin: pd.DataFrame, baseline_months: int = 6) -> pd.DataFrame:
    """One row per store-month: margin and discount rate against the store's own recent past,
    net of what moved for every store that month (promotions are chain-wide)."""
    m = margin.copy()
    for c in ("net_sales_eur", "cogs_eur", "discount_eur", "paid_incl_vat_eur"):
        m[c] = m[c].astype(float)
    sm = m.groupby(["store_id", "month"], as_index=False)[
        ["net_sales_eur", "cogs_eur", "discount_eur", "paid_incl_vat_eur"]
    ].sum()
    sm["margin"] = 1 - sm["cogs_eur"] / sm["net_sales_eur"]
    sm["discount_rate"] = sm["discount_eur"] / (sm["paid_incl_vat_eur"] + sm["discount_eur"])
    sm = sm.sort_values(["store_id", "month"])
    own = sm.groupby("store_id")
    past = lambda s: s.shift(1).rolling(baseline_months, min_periods=3).median()  # noqa: E731
    sm["margin_base"] = own["margin"].transform(past)
    sm["discount_base"] = own["discount_rate"].transform(past)
    sm["margin_drop"] = sm["margin_base"] - sm["margin"]
    sm["discount_rise"] = sm["discount_rate"] - sm["discount_base"]
    fleet = sm.groupby("month")
    sm["margin_drop_vs_own"] = sm["margin_drop"] - fleet["margin_drop"].transform("median")
    sm["discount_rise_vs_own"] = sm["discount_rise"] - fleet["discount_rise"].transform("median")
    return sm.dropna(subset=MARGIN_FEATURES).reset_index(drop=True)


def isolation_scores(features: pd.DataFrame, cols: list[str], contamination: float = 0.01):
    """Fit an Isolation Forest; return (model, anomaly score where higher = more unusual, flag)."""
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=SEED)
    x = features[cols].to_numpy(dtype=float)
    model.fit(x)
    score = -model.score_samples(x)
    flag = model.predict(x) == -1
    return model, score, flag


def ranked(features: pd.DataFrame, score, flag, keys: list[str]) -> pd.DataFrame:
    out = features.assign(anomaly_score=np.round(score, 4), flagged=flag)
    out["rank_in_month"] = out.groupby("month")["anomaly_score"].rank(ascending=False, method="first").astype(int)
    return out.sort_values(["month", "rank_in_month"])[[*keys, "month", "anomaly_score", "rank_in_month", "flagged"]]


def evaluate_detector(scores: pd.DataFrame, entity_col: str, entity: str, months: list[str], top: int = 3) -> dict:
    """Where the planted entity ranks each month, and how many flags are not it."""
    s = scores[scores["month"].isin(months)]
    hit = s[s[entity_col] == entity].set_index("month")["rank_in_month"].to_dict()
    false_flags = int(((s[entity_col] != entity) & s["flagged"]).sum())
    return {
        "ranks": {m: int(hit.get(m, 10**6)) for m in months},
        "in_top": all(hit.get(m, 10**6) <= top for m in months),
        "false_flags_in_those_months": false_flags,
        "flags_total": int(scores["flagged"].sum()),
    }


# ---------------------------------------------------------------- 5.2 forecast


def monthly_sales(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["month"] = pd.to_datetime(d["business_date"]).dt.to_period("M")
    d["net_sales_eur"] = d["net_sales_eur"].astype(float)
    return d.groupby(["store_id", "month"], as_index=False)["net_sales_eur"].sum()


def baseline_forecast(history: pd.DataFrame, target: pd.Period) -> pd.Series:
    """Same month last year x growth over the last 3 months vs the same 3 months a year earlier."""
    h = history.pivot(index="store_id", columns="month", values="net_sales_eur")
    last = history["month"].max()
    recent = [last - i for i in range(3)]
    growth = h[recent].sum(axis=1) / h[[m - 12 for m in recent]].sum(axis=1)
    return h[target - 12] * growth


def _features(panel: pd.DataFrame, target: pd.Period, last: pd.Period) -> pd.DataFrame:
    """Features for predicting ``target`` using only months up to ``last``."""
    h = panel.pivot(index="store_id", columns="month", values="net_sales_eur")
    recent = [last - i for i in range(3)]
    f = pd.DataFrame(index=h.index)
    f["lag_12"] = h[target - 12]
    f["lag_1"] = h[last]
    f["lag_2"] = h[last - 1]
    f["growth"] = h[recent].sum(axis=1) / h[[m - 12 for m in recent]].sum(axis=1)
    f["month_of_year"] = target.month
    f["horizon"] = (target - last).n
    f["store_level"] = h[[m for m in h.columns if m <= last]].mean(axis=1)
    return f


FORECAST_FEATURES = ["lag_12", "lag_1", "lag_2", "growth", "month_of_year", "horizon", "store_level"]


def train_forecaster(panel: pd.DataFrame, last: pd.Period):
    """Gradient boosting on every (target, as-of) pair available up to ``last``: predicts the
    ratio to last year's same month, which keeps stores of different sizes comparable."""
    from sklearn.ensemble import GradientBoostingRegressor

    rows = []
    months = sorted(panel["month"].unique())
    for asof in months:
        for h in (1, 2, 3):
            tgt = asof + h
            if tgt > last or tgt - 12 < months[0] or asof - 14 < months[0]:
                continue
            f = _features(panel[panel["month"] <= tgt], tgt, asof)
            actual = panel[panel["month"] == tgt].set_index("store_id")["net_sales_eur"]
            f["y"] = actual / f["lag_12"]
            rows.append(f.dropna())
    train = pd.concat(rows)
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED
    )
    model.fit(train[FORECAST_FEATURES], train["y"])
    return model, len(train)


def model_forecast(model, panel: pd.DataFrame, target: pd.Period, last: pd.Period) -> pd.Series:
    f = _features(panel[panel["month"] <= last], target, last)
    return pd.Series(model.predict(f[FORECAST_FEATURES]) * f["lag_12"].to_numpy(), index=f.index)


def mape(pred: pd.Series, actual: pd.Series) -> float:
    a, p = actual.align(pred, join="inner")
    return float((abs(p - a) / a).mean())


def backtest(panel: pd.DataFrame, last_complete: pd.Period, horizon: int = 3) -> dict:
    """Train on months before the last ``horizon`` complete months, forecast them, compare."""
    cutoff = last_complete - horizon
    hist = panel[panel["month"] <= cutoff]
    model, n_train = train_forecaster(hist, cutoff)
    out = {"cutoff": str(cutoff), "train_rows": n_train, "months": {}, "errors": {"model": [], "baseline": []}}
    for h in range(1, horizon + 1):
        tgt = cutoff + h
        actual = panel[panel["month"] == tgt].set_index("store_id")["net_sales_eur"]
        pm, pb = model_forecast(model, panel, tgt, cutoff), baseline_forecast(hist, tgt)
        out["months"][str(tgt)] = {"model_mape": mape(pm, actual), "baseline_mape": mape(pb, actual)}
        for name, p in (("model", pm), ("baseline", pb)):
            a, pp = actual.align(p, join="inner")
            out["errors"][name] += list((pp - a) / a)
    out["model_mape"] = float(np.mean([v["model_mape"] for v in out["months"].values()]))
    out["baseline_mape"] = float(np.mean([v["baseline_mape"] for v in out["months"].values()]))
    out["winner"] = "model" if out["model_mape"] < out["baseline_mape"] else "baseline"
    return out


def forecast(
    panel: pd.DataFrame, last_complete: pd.Period, winner: str, errors: list[float], horizon: int = 3
) -> pd.DataFrame:
    """Next ``horizon`` months after the last complete month, with an 80% range from backtest errors."""
    lo, hi = np.quantile(errors, [0.1, 0.9]) if len(errors) else (0.0, 0.0)
    hist = panel[panel["month"] <= last_complete]
    model = train_forecaster(hist, last_complete)[0] if winner == "model" else None
    rows = []
    for h in range(1, horizon + 1):
        tgt = last_complete + h
        p = model_forecast(model, panel, tgt, last_complete) if model is not None else baseline_forecast(hist, tgt)
        for store, v in p.items():
            rows.append(
                (
                    store,
                    str(tgt),
                    round(float(v), 2),
                    round(float(v / (1 + hi)), 2),
                    round(float(v / (1 + lo)), 2),
                    winner,
                )
            )
    return pd.DataFrame(rows, columns=["store_id", "month", "forecast_eur", "low_80_eur", "high_80_eur", "method"])


# ---------------------------------------------------------------- job


def require_certified(spark, catalog: str) -> str:
    from pyspark.sql import functions as F

    last = spark.table(f"{catalog}.ops.gold_certification").orderBy(F.desc("certified_at")).first()
    if last is None or not last["certified"]:
        raise RuntimeError(f"Latest Gold build is not certified ({last}): models run on certified Gold only.")
    return last["run_id"]


def main() -> None:
    import mlflow
    import mlflow.sklearn  # noqa: F401  (registers the sklearn flavour)
    from pyspark.sql import SparkSession

    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="finance")
    p.add_argument("--experiment", required=True)
    a = p.parse_args()
    spark = SparkSession.builder.getOrCreate()
    c = a.catalog
    gold_run = require_certified(spark, c)
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(a.experiment)
    gold = {
        t: spark.table(f"{c}.gold.{t}").toPandas() for t in ("refunds", "margin", "daily_revenue", "budget_variance")
    }
    truth = spark.read.option("header", "true").csv(f"/Volumes/{c}/raw/landing/_ground_truth/anomalies.csv").toPandas()

    # 5.1 cashier refunds
    cf = cashier_features(gold["refunds"])
    with mlflow.start_run(run_name="fraud_cashier_refunds"):
        model, score, flag = isolation_scores(cf, CASHIER_FEATURES, contamination=0.005)
        scores = ranked(cf, score, flag, ["store_id", "cashier_id"])
        a1 = truth[truth["anomaly"] == "A1_no_receipt_refunds"].iloc[0]
        months = sorted(m for m in scores["month"].unique() if m >= a1["start_date"][:7])
        ev = evaluate_detector(scores, "cashier_id", a1["cashier_id"], months)
        mlflow.log_params({"model": "IsolationForest", "features": ",".join(CASHIER_FEATURES), "gold_run": gold_run})
        mlflow.log_metrics({f"a1_rank_{m}": r for m, r in ev["ranks"].items()} | {"flags_total": ev["flags_total"]})
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=f"{c}.ml.fraud_cashier_refunds",
            input_example=cf[CASHIER_FEATURES].head(3),
        )
        print("A1 cashier detector:", ev, flush=True)
    spark.createDataFrame(scores).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{c}.gold.fraud_scores"
    )

    # 5.1 store margin drift
    mf = margin_features(gold["margin"])
    with mlflow.start_run(run_name="fraud_store_margin"):
        model, score, flag = isolation_scores(mf, MARGIN_FEATURES, contamination=0.02)
        alerts = ranked(mf, score, flag, ["store_id"])
        a2 = truth[truth["anomaly"] == "A2_discount_creep"].iloc[0]
        months = [m for m in sorted(alerts["month"].unique()) if m >= "2026-07"]
        ev2 = evaluate_detector(alerts, "store_id", a2["store_id"], months)
        mlflow.log_params({"model": "IsolationForest", "features": ",".join(MARGIN_FEATURES), "gold_run": gold_run})
        mlflow.log_metrics({f"a2_rank_{m}": r for m, r in ev2["ranks"].items()} | {"flags_total": ev2["flags_total"]})
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=f"{c}.ml.fraud_store_margin",
            input_example=mf[MARGIN_FEATURES].head(3),
        )
        print("A2 margin detector:", ev2, flush=True)
    spark.createDataFrame(alerts).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{c}.gold.margin_alerts"
    )

    # 5.2 forecast
    panel = monthly_sales(gold["daily_revenue"])
    bv = gold["budget_variance"]
    complete = sorted(bv.loc[bv["complete_month"], "month"].unique())
    last_complete = pd.Period(complete[-1], "M")
    with mlflow.start_run(run_name="revenue_forecast"):
        bt = backtest(panel, last_complete)
        fc = forecast(panel, last_complete, bt["winner"], bt["errors"][bt["winner"]])
        mlflow.log_params(
            {"winner": bt["winner"], "cutoff": bt["cutoff"], "train_rows": bt["train_rows"], "gold_run": gold_run}
        )
        mlflow.log_metrics({"model_mape": bt["model_mape"], "baseline_mape": bt["baseline_mape"]})
        if bt["winner"] == "model":
            model, _ = train_forecaster(panel[panel["month"] <= last_complete], last_complete)
            mlflow.sklearn.log_model(model, name="model", registered_model_name=f"{c}.ml.revenue_forecast")
        print("Forecast backtest:", {k: v for k, v in bt.items() if k != "errors"}, flush=True)
    spark.createDataFrame(fc).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{c}.gold.revenue_forecast"
    )
    print(fc.groupby("month")[["forecast_eur", "low_80_eur", "high_80_eur"]].sum().round(0).to_string(), flush=True)


if __name__ == "__main__":
    main()
