"""Job: fraud detectors and revenue forecast on certified Gold (stories 5.1, 5.2).

Everything here reads Gold only, and only when the latest Gold build is certified
(``ops.gold_certification``). The Gold tables are small (thousands of rows), so models are
trained in pandas and scikit-learn on the job cluster; Spark only reads and writes tables.

Honesty rules:
- The detectors are unsupervised and never see the answer key; the planted anomalies are
  used only afterwards, to score them, and only when an answer key exists (synthetic data).
- The forecast ships only if it beats a seasonal-naive baseline in a backtest; otherwise the
  baseline ships, and the scores say so.

Order: compute everything, log to MLflow and register, then write the three Gold tables with
the certified Gold run they came from, and append the input-drift checks to ``ops.model_drift``
(story 6.2: warnings, never a failure). A failure before the writes leaves Gold untouched.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from retail_finance_data import drift

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
    for c in ("refunds", "no_receipt_refunds", "no_receipt_eur", "refund_gross_eur"):
        r[c] = r[c].astype(float)
    r["no_receipt_share"] = (r["no_receipt_eur"] / r["refund_gross_eur"].where(r["refund_gross_eur"] > 0)).fillna(0.0)
    peers = r.groupby(["store_id", "month"])
    r["peer_no_receipt_ratio"] = np.log1p(r["no_receipt_eur"]) - np.log1p(peers["no_receipt_eur"].transform("median"))
    r["peer_refunds_ratio"] = np.log1p(r["refunds"]) - np.log1p(peers["refunds"].transform("median"))
    r["log_refunds"] = np.log1p(r["refunds"])
    r["log_no_receipt_eur"] = np.log1p(r["no_receipt_eur"])
    return r


def margin_features(margin: pd.DataFrame, gap: int = 4, window: int = 9) -> tuple[pd.DataFrame, int]:
    """One row per store-month: margin and discount rate against the store's own months
    ``gap``..``gap + window - 1`` earlier, net of what moved for every store that month.

    Skipping the most recent months keeps a slow creep from becoming its own baseline: a
    change must last more than ``gap`` months before it enters the comparison. Returns the
    features and how many store-months were dropped for lack of history.
    """
    m = margin.copy()
    cols = ["net_sales_eur", "cogs_eur", "discount_eur", "paid_incl_vat_eur"]
    for c in cols:
        m[c] = m[c].astype(float)
    sm = m.groupby(["store_id", "month"], as_index=False)[cols].sum()
    sm["margin"] = 1 - sm["cogs_eur"] / sm["net_sales_eur"]
    sm["discount_rate"] = sm["discount_eur"] / (sm["paid_incl_vat_eur"] + sm["discount_eur"])
    sm = sm.sort_values(["store_id", "month"])
    own = sm.groupby("store_id")
    past = lambda s: s.shift(gap).rolling(window, min_periods=3).median()  # noqa: E731
    sm["margin_base"] = own["margin"].transform(past)
    sm["discount_base"] = own["discount_rate"].transform(past)
    sm["margin_drop"] = sm["margin_base"] - sm["margin"]
    sm["discount_rise"] = sm["discount_rate"] - sm["discount_base"]
    fleet = sm.groupby("month")
    sm["margin_drop_vs_own"] = sm["margin_drop"] - fleet["margin_drop"].transform("median")
    sm["discount_rise_vs_own"] = sm["discount_rise"] - fleet["discount_rise"].transform("median")
    kept = sm.dropna(subset=MARGIN_FEATURES).reset_index(drop=True)
    return kept, len(sm) - len(kept)


def isolation_scores(features: pd.DataFrame, cols: list[str], contamination: float = 0.01):
    """Fit an Isolation Forest on ``cols`` only; return (model, score where higher = more unusual, flag).

    ``contamination`` sets how many rows are flagged (the review-queue size), not the ranking."""
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=SEED)
    x = features[cols].to_numpy(dtype=float)
    model.fit(x)
    return model, -model.score_samples(x), model.predict(x) == -1


def ranked(features: pd.DataFrame, score, flag, keys: list[str]) -> pd.DataFrame:
    """Rank within each month on the raw score, ties broken by key, so reruns rank identically."""
    out = features.assign(anomaly_score=score, flagged=flag)
    out = out.sort_values(["month", "anomaly_score", *keys], ascending=[True, False, *[True] * len(keys)])
    out["rank_in_month"] = out.groupby("month").cumcount() + 1
    out["anomaly_score"] = out["anomaly_score"].round(4)
    return out[[*keys, "month", "anomaly_score", "rank_in_month", "flagged"]].reset_index(drop=True)


def evaluate_detector(scores: pd.DataFrame, entity_col: str, entity: str, months: list[str], top: int = 3) -> dict:
    """Where the planted entity ranks each month, and the flags that are not it, per month."""
    s = scores[scores["month"].isin(months)]
    hit = s[s[entity_col] == entity].set_index("month")["rank_in_month"].to_dict()
    false = s[(s[entity_col] != entity) & s["flagged"]].groupby("month").size().to_dict()
    return {
        "ranks": {m: int(hit.get(m, 10**6)) for m in months},
        "in_top": all(hit.get(m, 10**6) <= top for m in months),
        "false_flags_by_month": {m: int(false.get(m, 0)) for m in months},
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


FORECAST_FEATURES = ["lag_12", "lag_1", "lag_2", "month_of_year", "horizon", "store_level"]


def _features(panel: pd.DataFrame, target: pd.Period, last: pd.Period) -> pd.DataFrame:
    """Features for predicting ``target`` using only months up to ``last`` (plus last year's
    value of the target month, which is always before ``last``)."""
    h = panel.pivot(index="store_id", columns="month", values="net_sales_eur")
    f = pd.DataFrame(index=h.index)
    f["lag_12"] = h[target - 12]
    f["lag_1"] = h[last]
    f["lag_2"] = h[last - 1]
    f["month_of_year"] = target.month
    f["horizon"] = (target - last).n
    f["store_level"] = h[[m for m in h.columns if m <= last]].mean(axis=1)
    return f


def training_rows(panel: pd.DataFrame, last: pd.Period) -> pd.DataFrame:
    """Every (as-of, horizon 1-3) pair whose target is known by ``last`` and has a value a year
    earlier. The target is the ratio to last year's month, so stores of any size compare."""
    months = sorted(panel["month"].unique())
    rows = []
    for asof in months:
        for h in (1, 2, 3):
            tgt = asof + h
            if tgt > last or tgt - 12 < months[0] or asof - 1 < months[0]:
                continue
            f = _features(panel[panel["month"] <= tgt], tgt, asof)
            actual = panel[panel["month"] == tgt].set_index("store_id")["net_sales_eur"]
            f["y"] = actual / f["lag_12"]
            rows.append(f.replace([np.inf, -np.inf], np.nan).dropna())
    return pd.concat(rows) if rows else pd.DataFrame(columns=[*FORECAST_FEATURES, "y"])


def train_forecaster(panel: pd.DataFrame, last: pd.Period):
    from sklearn.ensemble import GradientBoostingRegressor

    train = training_rows(panel, last)
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED
    )
    model.fit(train[FORECAST_FEATURES], train["y"])
    return model, train["horizon"].value_counts().sort_index().to_dict()


def model_forecast(model, panel: pd.DataFrame, target: pd.Period, last: pd.Period) -> pd.Series:
    f = _features(panel[panel["month"] <= last], target, last)
    return pd.Series(model.predict(f[FORECAST_FEATURES]) * f["lag_12"].to_numpy(), index=f.index)


def mape(pred: pd.Series, actual: pd.Series) -> float:
    a, p = actual.align(pred, join="inner")
    ok = (a > 0) & p.notna()
    return float((abs(p[ok] - a[ok]) / a[ok]).mean())


def backtest(panel: pd.DataFrame, last_complete: pd.Period, horizon: int = 3) -> dict:
    """Train on months up to ``horizon`` before the last complete month, forecast the rest, compare."""
    cutoff = last_complete - horizon
    hist = panel[panel["month"] <= cutoff]
    model, rows_by_h = train_forecaster(hist, cutoff)
    out = {"cutoff": str(cutoff), "train_rows_by_horizon": rows_by_h, "months": {}, "errors": {}}
    for h in range(1, horizon + 1):
        tgt = cutoff + h
        actual = panel[panel["month"] == tgt].set_index("store_id")["net_sales_eur"]
        preds = {"model": model_forecast(model, panel, tgt, cutoff), "baseline": baseline_forecast(hist, tgt)}
        out["months"][str(tgt)] = {f"{k}_mape": mape(p, actual) for k, p in preds.items()}
        for k, p in preds.items():
            a, pp = actual.align(p, join="inner")
            err = ((pp - a) / a).replace([np.inf, -np.inf], np.nan).dropna()
            out["errors"].setdefault(k, {})[h] = list(err)
    for k in ("model", "baseline"):
        out[f"{k}_mape"] = float(np.mean([v[f"{k}_mape"] for v in out["months"].values()]))
    out["winner"] = "model" if out["model_mape"] < out["baseline_mape"] else "baseline"
    return out


def forecast(panel: pd.DataFrame, last_complete: pd.Period, winner: str, errors: dict, horizon: int = 3):
    """Next ``horizon`` months after the last complete month, with an 80% range per horizon from
    that horizon's backtest errors. Returns (forecast table, model or None)."""
    hist = panel[panel["month"] <= last_complete]
    model = train_forecaster(hist, last_complete)[0] if winner == "model" else None
    rows = []
    for h in range(1, horizon + 1):
        tgt = last_complete + h
        err = errors.get(h, [])
        lo, hi = np.quantile(err, [0.1, 0.9]) if len(err) else (0.0, 0.0)
        p = model_forecast(model, panel, tgt, last_complete) if model is not None else baseline_forecast(hist, tgt)
        for store, v in p.dropna().items():
            rows.append(
                (
                    store,
                    str(tgt),
                    h,
                    round(float(v), 2),
                    round(float(v / (1 + hi)), 2),
                    round(float(v / (1 + lo)), 2),
                    winner,
                )
            )
    cols = ["store_id", "month", "horizon", "forecast_eur", "low_80_eur", "high_80_eur", "method"]
    return pd.DataFrame(rows, columns=cols), model


# ---------------------------------------------------------------- job


GOLD_INPUTS = ("refunds", "margin", "daily_revenue", "budget_variance")


def rewritten_after(last_writes: dict, certified_at) -> list[str]:
    """Gold inputs written after the certification: a later Gold build, not yet certified, has
    started rewriting them, so the data on disk is no longer the data that was certified."""
    return sorted(t for t, ts in last_writes.items() if ts > certified_at)


def require_certified(spark, catalog: str) -> str:
    from pyspark.sql import functions as F

    last = spark.table(f"{catalog}.ops.gold_certification").orderBy(F.desc("certified_at")).first()
    if last is None or not last["certified"]:
        raise RuntimeError(f"Latest Gold build is not certified ({last}): models run on certified Gold only.")
    writes = {t: spark.sql(f"DESCRIBE HISTORY {catalog}.gold.{t} LIMIT 1").first()["timestamp"] for t in GOLD_INPUTS}
    late = rewritten_after(writes, last["certified_at"])
    if late:
        raise RuntimeError(
            f"Gold {late} rewritten after certification of run {last['run_id']}: wait for the next certified build."
        )
    return last["run_id"]


def read_answer_key(spark, catalog: str) -> pd.DataFrame | None:
    """The generator's summary of planted anomalies, or None (real data has no answer key)."""
    from pyspark.errors import AnalysisException

    try:
        return (
            spark.read.option("header", "true")
            .csv(f"/Volumes/{catalog}/raw/landing/_ground_truth/anomalies.csv")
            .toPandas()
        )
    except AnalysisException as e:
        if "PATH_NOT_FOUND" in str(e):
            return None
        raise


def main() -> None:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="finance")
    p.add_argument("--experiment", required=True)
    a = p.parse_args()
    spark = SparkSession.builder.getOrCreate()
    c = a.catalog
    gold_run = require_certified(spark, c)
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(a.experiment)
    gold = {t: spark.table(f"{c}.gold.{t}").toPandas() for t in GOLD_INPUTS}
    truth = read_answer_key(spark, c)
    bv = gold["budget_variance"]
    last_complete = pd.Period(sorted(bv.loc[bv["complete_month"], "month"].unique())[-1], "M")
    checked = [str(last_complete - k) for k in range(5, -1, -1)]  # drift: the last 6 complete months

    def input_drift(model_name, features, cols):
        try:  # a warning, never a failure (story 6.2)
            months = [m for m in checked if m in set(features["month"].astype(str))]
            d = drift.feature_drift(features, cols, months).assign(model=model_name)
            mlflow.log_metrics({f"psi_{x.feature}_{x.month}": x.psi for x in d.itertuples() if pd.notna(x.psi)})
            return d
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: input drift for {model_name} not computed: {e!r}", flush=True)
            return pd.DataFrame()

    print(f"certified Gold run {gold_run}; mlflow {mlflow.__version__}; answer key: {truth is not None}", flush=True)

    def log_model(model, x, name):
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"{c}.ml.{name}",
            signature=infer_signature(x, model.predict(x)),
            input_example=x.head(3),
        )

    # 5.1 cashier refunds
    cf = cashier_features(gold["refunds"])
    model, score, flag = isolation_scores(cf, CASHIER_FEATURES, contamination=0.005)
    scores = ranked(cf, score, flag, ["store_id", "cashier_id"])
    with mlflow.start_run(run_name="fraud_cashier_refunds"):
        mlflow.log_params(
            {
                "model": "IsolationForest",
                "contamination": 0.005,
                "features": ",".join(CASHIER_FEATURES),
                "gold_run": gold_run,
            }
        )
        mlflow.log_metric("flags_total", int(scores["flagged"].sum()))
        if truth is not None:
            a1 = truth[truth["anomaly"] == "A1_no_receipt_refunds"].iloc[0]
            months = sorted(m for m in scores["month"].unique() if m >= a1["start_date"][:7])
            ev = evaluate_detector(scores, "cashier_id", a1["cashier_id"], months)
            mlflow.log_metrics(
                {f"a1_rank_{m}": r for m, r in ev["ranks"].items()}
                | {f"false_flags_{m}": n for m, n in ev["false_flags_by_month"].items()}
                | {"a1_in_top3": int(ev["in_top"])}
            )
            print("A1 cashier detector:", ev, flush=True)
        log_model(model, cf[CASHIER_FEATURES], "fraud_cashier_refunds")
        drifts = [input_drift("fraud_cashier_refunds", cf, CASHIER_FEATURES)]

    # 5.1 store margin drift
    mf, dropped = margin_features(gold["margin"])
    model, score, flag = isolation_scores(mf, MARGIN_FEATURES, contamination=0.02)
    alerts = ranked(mf, score, flag, ["store_id"])
    with mlflow.start_run(run_name="fraud_store_margin"):
        mlflow.log_params(
            {
                "model": "IsolationForest",
                "contamination": 0.02,
                "baseline": "own months 4-12 back",
                "gold_run": gold_run,
            }
        )
        mlflow.log_metrics({"flags_total": int(alerts["flagged"].sum()), "store_months_without_history": dropped})
        if truth is not None:
            a2 = truth[truth["anomaly"] == "A2_discount_creep"].iloc[0]
            months = sorted(m for m in alerts["month"].unique() if m >= a2["start_date"][:7])
            ev2 = evaluate_detector(alerts, "store_id", a2["store_id"], months)
            mlflow.log_metrics(
                {f"a2_rank_{m}": r for m, r in ev2["ranks"].items()}
                | {f"false_flags_{m}": n for m, n in ev2["false_flags_by_month"].items()}
            )
            print("A2 margin detector (from its start month):", ev2, flush=True)
        log_model(model, mf[MARGIN_FEATURES], "fraud_store_margin")
        drifts.append(input_drift("fraud_store_margin", mf, MARGIN_FEATURES))

    # 5.2 forecast
    panel = monthly_sales(gold["daily_revenue"])
    bt = backtest(panel, last_complete)
    fc, fmodel = forecast(panel, last_complete, bt["winner"], bt["errors"][bt["winner"]])
    with mlflow.start_run(run_name="revenue_forecast"):
        mlflow.log_params(
            {
                "winner": bt["winner"],
                "cutoff": bt["cutoff"],
                "gold_run": gold_run,
                "train_rows_by_horizon": str(bt["train_rows_by_horizon"]),
            }
        )
        mlflow.log_metrics(
            {"model_mape": bt["model_mape"], "baseline_mape": bt["baseline_mape"]}
            | {f"{k}_{m}": v for m, d in bt["months"].items() for k, v in d.items()}
        )
        if fmodel is not None:
            x = training_rows(panel[panel["month"] <= last_complete], last_complete)[FORECAST_FEATURES]
            log_model(fmodel, x, "revenue_forecast")
        print("Forecast backtest:", {k: v for k, v in bt.items() if k != "errors"}, flush=True)
        try:
            growth = drift.sales_growth(panel)
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: sales growth for drift not computed: {e!r}", flush=True)
            growth = None
        if growth is not None:
            drifts.append(input_drift("revenue_forecast", growth, ["yoy_growth"]))

    # Drift is a warning (story 6.2): printed and kept as a history, never a reason to fail.
    computed = [d for d in drifts if len(d)]
    model_drift = pd.concat(computed, ignore_index=True) if computed else None
    if model_drift is not None:
        watch = model_drift[model_drift["status"].isin(["watch", "drift"])]
        print(f"Input drift, last 6 complete months: {len(watch)} of {len(model_drift)} checks watch/drift", flush=True)
        if len(watch):
            print(watch.to_string(index=False), flush=True)

    # Writes last, all from the same certified Gold run.
    for name, df in (("fraud_scores", scores), ("margin_alerts", alerts), ("revenue_forecast", fc)):
        out = spark.createDataFrame(df.assign(gold_run=gold_run)).withColumn("scored_at", F.current_timestamp())
        out.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{c}.gold.{name}")
    if model_drift is not None:
        try:  # after the Gold writes: a failure here must not fail a job whose scores are written
            (
                spark.createDataFrame(model_drift.assign(gold_run=gold_run))
                .withColumn("psi", F.when(F.isnan("psi"), None).otherwise(F.col("psi")))
                .withColumn("scored_at", F.current_timestamp())
                .write.mode("append")  # a history; a rerun adds rows: read the latest scored_at per gold_run
                .saveAsTable(f"{c}.ops.model_drift")
            )
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: ops.model_drift not written: {e!r}", flush=True)
    print(fc.groupby("month")[["forecast_eur", "low_80_eur", "high_80_eur"]].sum().round(0).to_string(), flush=True)


if __name__ == "__main__":
    main()
