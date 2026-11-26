"""Convert IEX "DAM Market Snapshot" Excel exports into the canonical CSV.

Usage:
    python scripts/build_dataset.py [--raw data/raw/iex] [--out data/iex_dam.csv]

Reads every .xlsx file in --raw, keeps only data rows (a dd-mm-yyyy date plus a
"HH:MM - HH:MM" time block), drops the summary rows, de-duplicates days that appear
in more than one file and writes `date, block, mcp_rs_mwh, purchase_bid_mw, sell_bid_mw, mcv_mw` (see data/README.md).
Days that do not have exactly 96 blocks are reported and excluded.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DATE_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
BLOCK_RE = re.compile(r"^(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2})$")


def num(x):
    return pd.to_numeric(str(x).replace(",", ""), errors="coerce")


def read_export(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None, dtype=str)
    rows = []
    for _, r in raw.iterrows():
        date, blk, mcp = str(r.iloc[0]).strip(), str(r.iloc[2]).strip(), r.iloc[7]
        m = BLOCK_RE.match(blk)
        if not (DATE_RE.match(date) and m):
            continue
        block = int(m.group(1)) * 4 + int(m.group(2)) // 15 + 1
        rows.append({"date": pd.to_datetime(date, format="%d-%m-%Y"), "block": block,
                     "mcp_rs_mwh": num(mcp),
                     "purchase_bid_mw": num(r.iloc[3]), "sell_bid_mw": num(r.iloc[4]), "mcv_mw": num(r.iloc[5]),
                     "source_file": path.name})
    return pd.DataFrame(rows, columns=["date", "block", "mcp_rs_mwh", "purchase_bid_mw", "sell_bid_mw", "mcv_mw", "source_file"])


def build(raw_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    files = sorted(raw_dir.glob("*.xlsx"))
    if not files:
        raise SystemExit(f"no .xlsx files found in {raw_dir}")
    frames = [read_export(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    notes = [f"files read: {len(files)}"]
    dup = df.duplicated(["date", "block"], keep="last")
    if dup.any():
        notes.append(f"{int(dup.sum())} duplicate (date, block) rows dropped (overlapping files)")
    df = df[~dup]
    bad_price = df["mcp_rs_mwh"].isna()
    if bad_price.any():
        notes.append(f"{int(bad_price.sum())} rows with unreadable MCP dropped")
    df = df[~bad_price]
    counts = df.groupby("date")["block"].agg(["count", "nunique"])
    bad_days = counts[(counts["count"] != 96) | (counts["nunique"] != 96)].index
    if len(bad_days):
        notes.append(f"{len(bad_days)} incomplete days excluded: "
                     + ", ".join(str(d.date()) for d in bad_days[:10]) + (" ..." if len(bad_days) > 10 else ""))
    df = df[~df["date"].isin(bad_days)].sort_values(["date", "block"])
    days = sorted(df["date"].unique())
    if days:
        full = pd.date_range(days[0], days[-1])
        missing = full.difference(pd.DatetimeIndex(days))
        notes.append(f"range {days[0].date()} to {days[-1].date()}, {len(days)} complete days")
        if len(missing):
            notes.append(f"{len(missing)} calendar days missing (the loader rejects gaps): "
                         + ", ".join(str(d.date()) for d in missing[:10]) + (" ..." if len(missing) > 10 else ""))
    return df[["date", "block", "mcp_rs_mwh", "purchase_bid_mw", "sell_bid_mw", "mcv_mw"]], notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/iex")
    ap.add_argument("--out", default="data/iex_dam.csv")
    args = ap.parse_args()
    df, notes = build(Path(args.raw))
    for n in notes:
        print(n)
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} rows, max MCP {out['mcp_rs_mwh'].max():.2f} Rs/MWh")


if __name__ == "__main__":
    main()
