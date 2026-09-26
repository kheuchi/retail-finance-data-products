"""Job: generate the synthetic source files into the Unity Catalog landing volume.

Layout, one folder per source system extract, as an ERP would drop them:

    /Volumes/<catalog>/raw/landing/<source>/<source>_<period>.csv

Writing is idempotent: the same config produces byte-identical files with the same
names, so Auto Loader never ingests a regenerated file twice.
"""

from __future__ import annotations

import argparse
import os
from datetime import date

from retail_finance_data.generator import Config, build


def write_all(cfg: Config, out_dir: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, period, df in build(cfg):
        folder = os.path.join(out_dir, name if not name.startswith("_") else os.path.dirname(name))
        os.makedirs(folder, exist_ok=True)
        base = os.path.basename(name)
        fname = f"{base}_{period}.csv" if period else f"{base}.csv"
        df.to_csv(os.path.join(folder, fname), index=False)
        counts[name] = counts.get(name, 0) + len(df)
        print(f"wrote {name:<32} {period or '':<8} {len(df):>9,} rows", flush=True)
    return counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="finance")
    p.add_argument("--out", default=None, help="Override the output directory (local runs and tests).")
    p.add_argument("--end", default=None, help="Last business date, YYYY-MM-DD. Default: the Config default.")
    a = p.parse_args()
    cfg = Config() if not a.end else Config(end=date.fromisoformat(a.end))
    out = a.out or f"/Volumes/{a.catalog}/raw/landing"
    counts = write_all(cfg, out)
    print("TOTAL", {k: f"{v:,}" for k, v in counts.items()})


if __name__ == "__main__":
    main()
