"""Synthetic accounting data for a fictional retailer.

Everything is seeded, so the same config always produces the same data. The general
ledger is derived from the point-of-sale data, so the two reconcile by construction,
except where anomalies are deliberately planted. Planted anomalies are listed in a
ground-truth table so that detection models can be scored against a known answer.

Only numpy and pandas are used: both ship with the Databricks Runtime, and the
workspace cannot install anything from the internet.
"""

from __future__ import annotations

import calendar as _cal
from dataclasses import dataclass, field
from datetime import date, timedelta
from importlib import resources

import numpy as np
import pandas as pd

from .calendar import public_holidays

CATEGORIES = pd.DataFrame(
    {
        "category": ["Grocery", "Fresh", "Beverages", "Household", "Health & Beauty", "Apparel", "Electronics"],
        "price_lo": [0.8, 1.0, 0.7, 1.5, 2.0, 8.0, 15.0],
        "price_hi": [12.0, 15.0, 25.0, 30.0, 40.0, 90.0, 400.0],
        "margin_lo": [0.18, 0.25, 0.22, 0.28, 0.35, 0.45, 0.12],
        "margin_hi": [0.30, 0.40, 0.35, 0.40, 0.55, 0.60, 0.25],
        "reduced_vat": [True, True, False, False, False, False, False],
        "weight": [0.30, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05],
    }
)
VAT = {"DE": (0.19, 0.07), "CH": (0.081, 0.026)}  # (standard, reduced)
LOCAL_PRICE_FACTOR = {"EUR": 1.0, "CHF": 1.20}  # Swiss shelf prices are higher, in CHF
FORMATS = {"hypermarket": 2.0, "supermarket": 1.0, "express": 0.5, "online": 0.8}
ACCOUNTS = {
    "1000": "Cash",
    "1200": "Card clearing",
    "1300": "Gift vouchers",
    "1400": "Inventory",
    "2200": "VAT payable",
    "4000": "Revenue",
    "4010": "Sales returns",
    "5000": "Cost of goods sold",
}
PAYMENT_ACCOUNT = {"card": "1200", "cash": "1000", "voucher": "1300"}


@dataclass(frozen=True)
class Config:
    seed: int = 20260925
    start: date = date(2025, 1, 1)
    end: date = date(2026, 9, 25)
    n_stores: int = 40
    n_swiss_stores: int = 4
    n_products: int = 1200
    cashiers_per_store: int = 6
    base_txn_per_day: float = 60.0
    yearly_growth: float = 0.03
    budget_growth: float = 0.05
    # Planted anomalies (ground truth)
    a1_store: str = "S017"
    a1_cashier: str = "S017-C03"
    a1_start: date = date(2026, 8, 1)
    a2_store: str = "S031"
    a2_start: date = date(2026, 6, 1)
    a2_max_extra_discount: float = 0.12
    a3_month: tuple[int, int] = (2026, 8)
    a3_count: int = 4
    extra: dict = field(default_factory=dict)


# ─────────────────────────── master data ───────────────────────────


def make_stores(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    n_de = cfg.n_stores - cfg.n_swiss_stores
    ids = [f"S{i:03d}" for i in range(1, cfg.n_stores + 1)]
    country = ["DE"] * n_de + ["CH"] * cfg.n_swiss_stores
    regions_de, regions_ch = ["North", "South", "East", "West"], ["Zurich", "Geneva"]
    region = [regions_de[i % 4] if c == "DE" else regions_ch[i % 2] for i, c in enumerate(country)]
    fmt = rng.choice(["hypermarket", "supermarket", "express"], size=cfg.n_stores, p=[0.2, 0.55, 0.25]).tolist()
    fmt[n_de - 1] = "online"  # one German online store, open every day
    region[n_de - 1] = "Online"
    open_year = rng.integers(2012, 2024, size=cfg.n_stores)
    return pd.DataFrame(
        {
            "store_id": ids,
            "store_name": [f"{r} {f.title()} {i[1:]}" for r, f, i in zip(region, fmt, ids, strict=True)],
            "country": country,
            "currency": ["EUR" if c == "DE" else "CHF" for c in country],
            "region": region,
            "format": fmt,
            "size_factor": [round(FORMATS[f] * float(rng.uniform(0.8, 1.2)), 3) for f in fmt],
            "open_date": [date(int(y), int(rng.integers(1, 13)), 1).isoformat() for y in open_year],
        }
    )


def make_products(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    cat_idx = rng.choice(len(CATEGORIES), size=cfg.n_products, p=CATEGORIES["weight"].to_numpy())
    c = CATEGORIES.iloc[cat_idx].reset_index(drop=True)
    price = np.exp(rng.uniform(np.log(c["price_lo"]), np.log(c["price_hi"])))
    margin = rng.uniform(c["margin_lo"], c["margin_hi"])
    price = np.round(np.floor(price) + 0.99, 2)  # shelf prices end in .99
    cost = np.round(price / 1.19 * (1 - margin), 2)
    return pd.DataFrame(
        {
            "sku": [f"P{i:05d}" for i in range(1, cfg.n_products + 1)],
            "category": c["category"],
            "reduced_vat": c["reduced_vat"],
            "list_price_eur": price,
            "unit_cost_eur": cost,
        }
    )


def make_cashiers(cfg: Config, stores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in stores.itertuples():
        if s.format == "online":
            rows.append((f"{s.store_id}-WEB", s.store_id))
            continue
        rows += [(f"{s.store_id}-C{n:02d}", s.store_id) for n in range(1, cfg.cashiers_per_store + 1)]
    return pd.DataFrame(rows, columns=["cashier_id", "store_id"])


def load_fx() -> pd.DataFrame:
    """ECB euro reference rates, committed as a snapshot. Columns: rate_date, currency, units_per_eur."""
    with resources.files("retail_finance_data.reference").joinpath("ecb_fx_rates.csv").open("r") as f:
        fx = pd.read_csv(f, parse_dates=["rate_date"])
    return fx


def daily_rate(fx: pd.DataFrame, currency: str, days: pd.DatetimeIndex) -> np.ndarray:
    """Units of `currency` per EUR for each day, carrying the last published rate over weekends."""
    if currency == "EUR":
        return np.ones(len(days))
    s = fx.loc[fx["currency"] == currency].set_index("rate_date")["units_per_eur"].sort_index()
    return s.reindex(s.index.union(days)).ffill().bfill().reindex(days).to_numpy()


# ─────────────────────────── helpers ───────────────────────────


def months_in_range(cfg: Config) -> list[tuple[int, int]]:
    out, y, m = [], cfg.start.year, cfg.start.month
    while (y, m) <= (cfg.end.year, cfg.end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _season(month: int) -> float:
    return {1: 0.88, 2: 0.9, 3: 0.97, 4: 1.0, 5: 1.0, 6: 0.98, 7: 0.95, 8: 0.95, 9: 1.0, 10: 1.02, 11: 1.12, 12: 1.38}[
        month
    ]


_DOW = np.array([0.9, 0.92, 0.95, 1.0, 1.15, 1.35, 1.0])  # Mon..Sun


def _promo_table(cfg: Config, rng: np.random.Generator) -> dict[tuple[int, int], np.ndarray]:
    """For each ISO (year, week): discount rate per category. Two categories on promotion each week."""
    table, d = {}, cfg.start
    while d <= cfg.end:
        key = d.isocalendar()[:2]
        if key not in table:
            pct = np.zeros(len(CATEGORIES))
            pct[rng.choice(len(CATEGORIES), size=2, replace=False)] = rng.choice([0.10, 0.15, 0.20, 0.25], size=2)
            table[key] = pct
        d += timedelta(days=7 if d.isoweekday() == 1 else 1)
    return table


def _ids(prefix: str, days: np.ndarray, stores: np.ndarray, seq: np.ndarray, width: int = 5) -> np.ndarray:
    return (
        pd.Series(prefix + pd.to_datetime(days).strftime("%Y%m%d"))
        + "-"
        + pd.Series(stores)
        + "-"
        + pd.Series(seq).astype(str).str.zfill(width)
    ).to_numpy()


# ─────────────────────────── one month ───────────────────────────


@dataclass
class MonthData:
    pos_sales: pd.DataFrame
    refunds: pd.DataFrame
    gl: pd.DataFrame
    store_month: pd.DataFrame  # store_id, net_eur, cogs_eur (for budgeting)
    planted: pd.DataFrame  # ground-truth rows


def generate_month(
    cfg: Config, year: int, month: int, stores, products, cashiers, fx, promos, rng: np.random.Generator
) -> MonthData:
    first = max(date(year, month, 1), cfg.start)
    last = min(date(year, month, _cal.monthrange(year, month)[1]), cfg.end)
    days = pd.date_range(first, last, freq="D")
    n_s, n_d = len(stores), len(days)

    hol = {c: public_holidays(c, [year]) for c in ("DE", "CH")}
    is_online = (stores["format"] == "online").to_numpy()
    country = stores["country"].to_numpy()
    open_mask = np.ones((n_s, n_d), dtype=bool)
    for j, d in enumerate(days.date):
        for i in range(n_s):
            if not is_online[i] and (d.isoweekday() == 7 or d in hol[country[i]]):
                open_mask[i, j] = False

    growth = (1 + cfg.yearly_growth) ** ((days.year - cfg.start.year).to_numpy())
    lam = (
        cfg.base_txn_per_day
        * stores["size_factor"].to_numpy()[:, None]
        * _DOW[days.dayofweek.to_numpy()][None, :]
        * _season(month)
        * growth[None, :]
    )
    n_txn = rng.poisson(lam * open_mask)

    # ── transactions ──
    s_idx = np.repeat(np.repeat(np.arange(n_s), n_d), n_txn.ravel())
    d_idx = np.repeat(np.tile(np.arange(n_d), n_s), n_txn.ravel())
    T = len(s_idx)
    seq = np.concatenate([np.arange(k) for k in n_txn.ravel()]) if T else np.array([], dtype=int)
    store_ids = stores["store_id"].to_numpy()[s_idx]
    txn_day = days.to_numpy()[d_idx]
    txn_id = _ids("T", txn_day, store_ids, seq)
    minutes = np.where(is_online[s_idx], rng.integers(0, 1440, T), rng.integers(8 * 60, 20 * 60, T))
    txn_ts = txn_day + minutes.astype("timedelta64[m]") + rng.integers(0, 60, T).astype("timedelta64[s]")
    payment = rng.choice(["card", "cash", "voucher"], size=T, p=[0.68, 0.27, 0.05])
    payment[is_online[s_idx]] = "card"

    cash_by_store = cashiers.groupby("store_id")["cashier_id"].apply(list).to_dict()
    pick = rng.random(T)
    cashier = np.array(
        [cash_by_store[s][int(p * len(cash_by_store[s]))] for s, p in zip(store_ids, pick, strict=True)], dtype=object
    )

    # ── lines ──
    n_lines = 1 + rng.poisson(1.8, T)
    li = np.repeat(np.arange(T), n_lines)
    N = len(li)
    line_no = np.concatenate([np.arange(1, k + 1) for k in n_lines]) if N else np.array([], dtype=int)
    popularity = 1.0 / np.arange(1, len(products) + 1) ** 0.8
    popularity /= popularity.sum()
    sku_i = rng.choice(len(products), size=N, p=popularity)
    qty = 1 + rng.poisson(0.3, N)
    currency = stores["currency"].to_numpy()[s_idx[li]]
    price = np.round(products["list_price_eur"].to_numpy()[sku_i] * np.vectorize(LOCAL_PRICE_FACTOR.get)(currency), 2)

    cat_code = pd.Categorical(products["category"], categories=CATEGORIES["category"]).codes[sku_i]
    week_keys = [d.isocalendar()[:2] for d in days.date]
    promo_by_day = np.stack([promos[k] for k in week_keys])  # (n_d, n_cat)
    promo_pct = promo_by_day[d_idx[li], cat_code]

    extra = np.zeros(N)
    a2 = (store_ids[li] == cfg.a2_store) & (txn_day[li] >= np.datetime64(cfg.a2_start))
    if a2.any():
        ramp_days = (np.datetime64(cfg.end) - np.datetime64(cfg.a2_start)).astype(int) or 1
        prog = (txn_day[li][a2] - np.datetime64(cfg.a2_start)).astype("timedelta64[D]").astype(int) / ramp_days
        extra[a2] = np.clip(prog, 0, 1) * cfg.a2_max_extra_discount
    disc_pct = np.clip(promo_pct + extra, 0, 0.6)

    gross_before = qty * price
    discount = np.round(gross_before * disc_pct, 2)
    gross = np.round(gross_before - discount, 2)
    reduced = products["reduced_vat"].to_numpy()[sku_i]
    ctry = country[s_idx[li]]
    vat_rate = np.where(
        reduced, np.where(ctry == "DE", VAT["DE"][1], VAT["CH"][1]), np.where(ctry == "DE", VAT["DE"][0], VAT["CH"][0])
    )
    net = np.round(gross / (1 + vat_rate), 2)
    vat_amt = np.round(gross - net, 2)

    sales = pd.DataFrame(
        {
            "transaction_id": txn_id[li],
            "line_no": line_no,
            "transaction_ts": pd.to_datetime(txn_ts[li]).strftime("%Y-%m-%d %H:%M:%S"),
            "business_date": pd.to_datetime(txn_day[li]).strftime("%Y-%m-%d"),
            "store_id": store_ids[li],
            "cashier_id": cashier[li],
            "sku": products["sku"].to_numpy()[sku_i],
            "quantity": qty,
            "unit_price": price,
            "discount_amount": discount,
            "gross_amount": gross,
            "vat_rate": vat_rate,
            "net_amount": net,
            "vat_amount": vat_amt,
            "currency": currency,
            "payment_method": payment[li],
            "promo_flag": promo_pct > 0,
        }
    )

    # ── refunds with receipt (~1.2% of lines, back at the same store within 10 days) ──
    refunds, planted = _refunds(cfg, sales, days, open_mask, stores, cashiers, rng, year, month)

    # ── general ledger, in EUR ──
    gl = _ledger(cfg, sales, refunds, products, stores, fx, days, rng, year, month)
    planted = pd.concat([planted, gl.attrs.pop("planted")], ignore_index=True)

    fxr = {c: pd.Series(daily_rate(fx, c, days), index=days.strftime("%Y-%m-%d")) for c in ("EUR", "CHF")}
    rate = np.where(sales["currency"] == "EUR", 1.0, fxr["CHF"].reindex(sales["business_date"]).to_numpy())
    cogs = qty * products["unit_cost_eur"].to_numpy()[sku_i]
    sm = pd.DataFrame({"store_id": sales["store_id"], "net_eur": sales["net_amount"] / rate, "cogs_eur": cogs})
    store_month = sm.groupby("store_id", as_index=False).sum()
    store_month["year"], store_month["month"] = year, month
    return MonthData(sales, refunds, gl, store_month, planted)


def _refunds(cfg, sales, days, open_mask, stores, cashiers, rng, year, month):
    cols = [
        "refund_id",
        "refund_ts",
        "business_date",
        "store_id",
        "cashier_id",
        "original_transaction_id",
        "sku",
        "quantity",
        "refund_amount",
        "vat_rate",
        "currency",
        "receipt_present",
        "refund_method",
        "reason",
    ]
    parts, planted = [], []
    open_days = {s: days[open_mask[i]] for i, s in enumerate(stores["store_id"])}
    reasons = np.array(["damaged", "wrong size", "changed mind", "defective", "other"])

    pick = rng.random(len(sales)) < 0.012
    r = sales.loc[pick].copy()
    if len(r):
        lag = rng.integers(0, 11, len(r))
        rd = pd.to_datetime(r["business_date"]) + pd.to_timedelta(lag, unit="D")
        out = []
        for s, d in zip(r["store_id"], rd, strict=True):
            od = open_days[s]
            k = od.searchsorted(d)
            out.append(od[min(k, len(od) - 1)] if len(od) else pd.NaT)
        r["rdate"] = out
        r = r.dropna(subset=["rdate"])
        parts.append(
            pd.DataFrame(
                {
                    "business_date": pd.to_datetime(r["rdate"]).dt.strftime("%Y-%m-%d"),
                    "store_id": r["store_id"],
                    "cashier_id": r["cashier_id"],
                    "original_transaction_id": r["transaction_id"],
                    "sku": r["sku"],
                    "quantity": r["quantity"],
                    "refund_amount": r["gross_amount"],
                    "vat_rate": r["vat_rate"],
                    "currency": r["currency"],
                    "receipt_present": True,
                    "refund_method": np.where(r["payment_method"] == "cash", "cash", "card"),
                    "reason": rng.choice(reasons, len(r), p=[0.3, 0.2, 0.3, 0.15, 0.05]),
                    "planted": False,
                }
            )
        )

    # ── background no-receipt refunds: small, rare ──
    phys = stores.loc[stores["format"] != "online"]
    cash_by_store = cashiers.groupby("store_id")["cashier_id"].apply(list).to_dict()
    rows = []
    for s, ctry, cur in zip(phys["store_id"], phys["country"], phys["currency"], strict=True):
        for d in open_days[s]:
            for _ in range(rng.poisson(0.25)):
                rows.append((d, s, rng.choice(cash_by_store[s]), round(float(rng.uniform(5, 60)), 2), ctry, cur, False))
        # ── A1: one cashier, no-receipt refunds ramping up ──
        if s == cfg.a1_store:
            start = np.datetime64(cfg.a1_start)
            span = max((np.datetime64(cfg.end) - start).astype(int), 1)
            for d in open_days[s]:
                if np.datetime64(d.date()) >= start:
                    prog = (np.datetime64(d.date()) - start).astype(int) / span
                    for _ in range(int(round(1 + 7 * prog))):
                        rows.append((d, s, cfg.a1_cashier, round(float(rng.uniform(25, 180)), 2), ctry, cur, True))
    if rows:
        nr = pd.DataFrame(rows, columns=["d", "store_id", "cashier_id", "amount", "country", "currency", "planted"])
        std_vat = np.where(nr["country"] == "DE", VAT["DE"][0], VAT["CH"][0])
        parts.append(
            pd.DataFrame(
                {
                    "business_date": pd.to_datetime(nr["d"]).dt.strftime("%Y-%m-%d"),
                    "store_id": nr["store_id"],
                    "cashier_id": nr["cashier_id"],
                    "original_transaction_id": None,
                    "sku": None,
                    "quantity": 1,
                    "refund_amount": nr["amount"],
                    "vat_rate": std_vat,
                    "currency": nr["currency"],
                    "receipt_present": False,
                    "refund_method": "cash",
                    "reason": np.where(nr["planted"], "changed mind", rng.choice(reasons, len(nr))),
                    "planted": nr["planted"],
                }
            )
        )

    if not parts:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=["anomaly", "entity_type", "entity_id", "detail"])
    ref = pd.concat(parts, ignore_index=True).sort_values(["business_date", "store_id"], kind="stable")
    ref = ref.reset_index(drop=True)
    seq = ref.groupby(["business_date", "store_id"]).cumcount().to_numpy()
    ref["refund_id"] = _ids("R", pd.to_datetime(ref["business_date"]).to_numpy(), ref["store_id"].to_numpy(), seq, 4)
    minutes = rng.integers(9 * 60, 19 * 60, len(ref))
    ref["refund_ts"] = (pd.to_datetime(ref["business_date"]) + pd.to_timedelta(minutes, unit="m")).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    p = ref.loc[ref["planted"]]
    planted.append(
        pd.DataFrame(
            {
                "anomaly": "A1_no_receipt_refunds",
                "entity_type": "refund",
                "entity_id": p["refund_id"],
                "detail": "cashier " + p["cashier_id"],
            }
        )
    )
    return ref[cols], pd.concat(planted, ignore_index=True)


def _ledger(cfg, sales, refunds, products, stores, fx, days, rng, year, month):
    day_str = days.strftime("%Y-%m-%d")
    chf = pd.Series(daily_rate(fx, "CHF", days), index=day_str)

    def to_eur(amount, currency, bdate):
        rate = np.where(currency == "EUR", 1.0, chf.reindex(bdate).to_numpy())
        return amount / rate

    lines = []

    def journal(jid, bdate, store, source, desc, entries):
        """entries: list of (account, debit, credit). Rounded to cents; residual folded into the first debit."""
        ent = [(a, round(dr, 2), round(cr, 2)) for a, dr, cr in entries if round(dr, 2) or round(cr, 2)]
        if not ent:
            return
        diff = round(sum(c for _, _, c in ent) - sum(d for _, d, _ in ent), 2)
        if diff:
            i = next(k for k, e in enumerate(ent) if e[1])
            a, d, c = ent[i]
            ent[i] = (a, round(d + diff, 2), c)
        for n, (a, d, c) in enumerate(ent, 1):
            lines.append((jid, n, bdate, a, ACCOUNTS[a], store, d, c, source, desc))

    s = sales.assign(
        gross_eur=to_eur(sales["gross_amount"].to_numpy(), sales["currency"].to_numpy(), sales["business_date"]),
        net_eur=to_eur(sales["net_amount"].to_numpy(), sales["currency"].to_numpy(), sales["business_date"]),
        cogs_eur=sales["quantity"] * sales["sku"].map(products.set_index("sku")["unit_cost_eur"]),
    )
    s["vat_eur"] = s["gross_eur"] - s["net_eur"]
    by_pay = s.pivot_table(
        index=["business_date", "store_id"], columns="payment_method", values="gross_eur", aggfunc="sum", fill_value=0.0
    )
    tot = s.groupby(["business_date", "store_id"])[["net_eur", "vat_eur", "cogs_eur"]].sum()
    for (bd, st), row in tot.iterrows():
        pays = by_pay.loc[(bd, st)]
        net_r, vat_r = round(row["net_eur"], 2), round(row["vat_eur"], 2)
        journal(
            f"J{bd.replace('-', '')}-{st}-POS",
            bd,
            st,
            "POS",
            "Daily sales",
            [(PAYMENT_ACCOUNT[p], float(pays.get(p, 0.0)), 0.0) for p in ("card", "cash", "voucher")]
            + [("4000", 0.0, net_r), ("2200", 0.0, vat_r)],
        )
        journal(
            f"J{bd.replace('-', '')}-{st}-COGS",
            bd,
            st,
            "COGS",
            "Cost of goods sold",
            [("5000", row["cogs_eur"], 0.0), ("1400", 0.0, row["cogs_eur"])],
        )

    if len(refunds):
        r = refunds.assign(
            gross_eur=to_eur(
                refunds["refund_amount"].to_numpy(), refunds["currency"].to_numpy(), refunds["business_date"]
            )
        )
        r["net_eur"] = r["gross_eur"] / (1 + r["vat_rate"].astype(float))
        for (bd, st), g in r.groupby(["business_date", "store_id"]):
            cash = g.loc[g["refund_method"] == "cash", "gross_eur"].sum()
            card = g.loc[g["refund_method"] == "card", "gross_eur"].sum()
            net_r = round(g["net_eur"].sum(), 2)
            vat_r = round(round(g["gross_eur"].sum(), 2) - net_r, 2)
            journal(
                f"J{bd.replace('-', '')}-{st}-REF",
                bd,
                st,
                "REFUND",
                "Daily refunds",
                [("4010", net_r, 0.0), ("2200", vat_r, 0.0), ("1000", 0.0, cash), ("1200", 0.0, card)],
            )

    planted = pd.DataFrame(columns=["anomaly", "entity_type", "entity_id", "detail"])
    if (year, month) == cfg.a3_month:
        rows = []
        cand = [d for d in day_str if d in set(s["business_date"])]
        for k in range(cfg.a3_count):
            bd = cand[int(rng.integers(len(cand)))]
            st = stores["store_id"].iloc[int(rng.integers(len(stores)))]
            amt = round(float(rng.uniform(3000, 12000)), 2)
            jid = f"J{bd.replace('-', '')}-{st}-MAN{k + 1}"
            journal(jid, bd, st, "MANUAL", "Manual revenue adjustment", [("1200", amt, 0.0), ("4000", 0.0, amt)])
            rows.append(("A3_unsupported_manual_revenue", "journal", jid, f"EUR {amt:,.2f} with no sales behind it"))
        planted = pd.DataFrame(rows, columns=planted.columns)

    gl = pd.DataFrame(
        lines,
        columns=[
            "journal_id",
            "line_no",
            "posting_date",
            "account_code",
            "account_name",
            "store_id",
            "debit_eur",
            "credit_eur",
            "source",
            "description",
        ],
    )
    gl.attrs["planted"] = planted
    return gl


# ─────────────────────────── budget + ground truth ───────────────────────────


def make_budget(cfg: Config, store_months: pd.DataFrame) -> pd.DataFrame:
    """Budget for the last year in range = prior-year actuals x (1 + budget growth), rounded to EUR 100."""
    prior = store_months.loc[store_months["year"] == cfg.end.year - 1]
    b = prior.assign(year=cfg.end.year)
    b["revenue_budget_eur"] = (b["net_eur"] * (1 + cfg.budget_growth) / 100).round() * 100
    b["cogs_budget_eur"] = (b["cogs_eur"] / b["net_eur"] * b["revenue_budget_eur"]).round(-2)
    b["budget_month"] = b["year"].astype(str) + "-" + b["month"].astype(str).str.zfill(2)
    return b[["store_id", "budget_month", "revenue_budget_eur", "cogs_budget_eur"]].reset_index(drop=True)


def anomaly_summary(cfg: Config) -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "A1_no_receipt_refunds",
                cfg.a1_store,
                cfg.a1_cashier,
                cfg.a1_start.isoformat(),
                "One cashier issues no-receipt cash refunds, ramping from 1 to 8 a day",
            ),
            (
                "A2_discount_creep",
                cfg.a2_store,
                "",
                cfg.a2_start.isoformat(),
                f"Extra discount on every line rising to {cfg.a2_max_extra_discount:.0%}: margin erosion",
            ),
            (
                "A3_unsupported_manual_revenue",
                "",
                "",
                f"{cfg.a3_month[0]}-{cfg.a3_month[1]:02d}-01",
                f"{cfg.a3_count} manual revenue journals with no sales behind them: GL vs POS reconciliation breaks",
            ),
        ],
        columns=["anomaly", "store_id", "cashier_id", "start_date", "description"],
    )


# ─────────────────────────── whole dataset ───────────────────────────


def build(cfg: Config):
    """Yield (name, month_key, DataFrame) for every output file. Master data first, then month by month.

    Each month gets its own seeded generator, so any month can be regenerated alone and
    come out identical.
    """
    rng = np.random.default_rng(cfg.seed)
    stores = make_stores(cfg, rng)
    products = make_products(cfg, rng)
    cashiers = make_cashiers(cfg, stores)
    promos = _promo_table(cfg, rng)
    fx = load_fx()
    yield "stores", None, stores
    yield "products", None, products
    yield "cashiers", None, cashiers
    yield "fx_rates", None, fx.assign(rate_date=fx["rate_date"].dt.strftime("%Y-%m-%d"))

    store_months, planted = [], []
    for y, m in months_in_range(cfg):
        md = generate_month(cfg, y, m, stores, products, cashiers, fx, promos, np.random.default_rng([cfg.seed, y, m]))
        key = f"{y}-{m:02d}"
        yield "pos_sales", key, md.pos_sales
        yield "refunds", key, md.refunds
        yield "gl_journal", key, md.gl
        store_months.append(md.store_month)
        planted.append(md.planted)

    yield "budget", str(cfg.end.year), make_budget(cfg, pd.concat(store_months, ignore_index=True))
    yield "_ground_truth/anomalies", None, anomaly_summary(cfg)
    yield "_ground_truth/planted_records", None, pd.concat(planted, ignore_index=True)
