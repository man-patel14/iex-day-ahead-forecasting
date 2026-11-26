"""Add a bid-time temperature forecast column (temp_fc_c) to the price CSV.

Source: Open-Meteo Previous Model Runs API, variable `temperature_2m_previous_day2`
(the forecast issued 48 h before the valid hour). A forecast issued two days before
the delivery hour is always available when bids for that day are placed.
The Historical Forecast API is NOT used: it stitches the first hours of each model
run and behaves like a nowcast, which would leak information.

Usage:
    python scripts/add_weather.py [--in data/iex_dam.csv] [--out data/iex_dam_weather.csv]

The temperature is the simple mean over a few large demand centres (a rough proxy
for national demand-weighted temperature). Hourly values are repeated for the four
15-minute blocks of each hour. Needs internet access to previous-runs-api.open-meteo.com.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
VAR = "temperature_2m_previous_day2"
CITIES = {                      # name: (latitude, longitude)
    "Delhi": (28.61, 77.21),
    "Mumbai": (19.08, 72.88),
    "Kolkata": (22.57, 88.36),
    "Chennai": (13.08, 80.27),
    "Bengaluru": (12.97, 77.59),
}
ARCHIVE_START = pd.Timestamp("2024-01-01")   # earlier days have no archived forecast; left empty
CHUNK_DAYS = 120                # keep each request small


def http_get_json(url: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as exc:                     # network or HTTP error
            if attempt == retries - 1:
                raise SystemExit(f"request failed after {retries} tries: {exc}\nURL: {url}")
            time.sleep(2 ** attempt * 3)


def parse_hourly(payload: dict) -> pd.Series:
    """Return a Series indexed by local hourly timestamp. Fails loudly on a bad response."""
    if "hourly" not in payload or VAR not in payload["hourly"]:
        raise SystemExit(f"unexpected response (no hourly.{VAR}): {str(payload)[:300]}")
    h = payload["hourly"]
    s = pd.Series(h[VAR], index=pd.to_datetime(h["time"]), dtype="float64")
    if s.index.has_duplicates:
        raise SystemExit("duplicate hourly timestamps in response")
    return s


def fetch_city(lat: float, lon: float, start: str, end: str, get=http_get_json) -> pd.Series:
    parts = []
    for a in pd.date_range(start, end, freq=f"{CHUNK_DAYS}D"):
        b = min(a + pd.Timedelta(days=CHUNK_DAYS - 1), pd.Timestamp(end))
        q = urllib.parse.urlencode({
            "latitude": lat, "longitude": lon, "hourly": VAR, "timezone": "Asia/Kolkata",
            "start_date": a.date(), "end_date": b.date()})
        parts.append(parse_hourly(get(f"{API}?{q}")))
    return pd.concat(parts)


def hourly_to_blocks(temp_hourly: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly temperature -> one value per (date, block); each hour covers 4 blocks."""
    rows = []
    for d in dates:
        vals = temp_hourly.reindex(pd.date_range(d, periods=24, freq="h")).to_numpy()
        for b in range(1, 97):
            rows.append((d, b, vals[(b - 1) // 4]))
    return pd.DataFrame(rows, columns=["date", "block", "temp_fc_c"])


def add_weather(df: pd.DataFrame, get=http_get_json) -> tuple[pd.DataFrame, list[str]]:
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    fetch_dates = dates[dates >= ARCHIVE_START]
    if len(fetch_dates):
        start, end = str(fetch_dates[0].date()), str(fetch_dates[-1].date())
        series = [fetch_city(lat, lon, start, end, get) for lat, lon in CITIES.values()]
        mean_temp = pd.concat(series, axis=1).mean(axis=1)
    else:
        mean_temp = pd.Series(dtype="float64")
    blocks = hourly_to_blocks(mean_temp, dates)
    out = df.drop(columns=["temp_fc_c"], errors="ignore").merge(blocks, on=["date", "block"], how="left")
    notes = [f"cities averaged: {', '.join(CITIES)}",
             f"temperature range: {out['temp_fc_c'].min():.1f} to {out['temp_fc_c'].max():.1f} C"]
    missing = int(out["temp_fc_c"].isna().sum())
    if missing:
        notes.append(f"WARNING: {missing} blocks have no temperature forecast (left empty)")
    return out, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="data/iex_dam.csv")
    ap.add_argument("--out", default="data/iex_dam_weather.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.inp, parse_dates=["date"])
    out, notes = add_weather(df)
    for n in notes:
        print(n)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} rows")


if __name__ == "__main__":
    main()
