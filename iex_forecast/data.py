"""Market data container, CSV loader, data-quality checks and a synthetic generator.

The synthetic generator exists so the pipeline, harness and tests run end to end
without any download. Results on synthetic data prove the *machinery* works. They say
nothing about the real IEX market.

Canonical CSV schema (one row per 15-minute block, see data/README.md):
    date, block, mcp_rs_mwh, temp_fc_c, solar_fc_mw, demand_fc_mw [, is_holiday]
`block` runs 1..96. `mcp_rs_mwh` is the market clearing price exactly as IEX publishes it.
The *_fc_* columns must be forecasts that existed at bid time, never realised values.
Only `date`, `block` and `mcp_rs_mwh` are required; missing forecast columns become NaN
and the corresponding feature groups are skipped in the ablation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

import numpy as np
import pandas as pd

from .config import BLOCKS_PER_DAY, PRICE_CAP

B = BLOCKS_PER_DAY
CSV_COLUMNS = ["date", "block", "mcp_rs_mwh", "temp_fc_c", "solar_fc_mw", "demand_fc_mw", "is_holiday",
               "purchase_bid_mw", "sell_bid_mw", "mcv_mw"]


@dataclass
class MarketData:
    dates: pd.DatetimeIndex            # one entry per delivery day
    price: np.ndarray                  # (D, 96) realised DAM price, Rs/kWh
    temp_fc: np.ndarray                # (D, 96) day-ahead temperature FORECAST (NaN if unavailable)
    demand_fc: np.ndarray              # (D, 96) day-ahead demand FORECAST (NaN if unavailable)
    holiday: np.ndarray                # (D,) bool
    source: str = "unknown"
    solar_fc: np.ndarray | None = field(default=None)   # (D, 96) day-ahead solar FORECAST
    # Realised auction volumes (MW). These are OUTCOMES: features may only read them for days
    # that are already known at bid time (see features.day_features).
    buy_bid: np.ndarray | None = field(default=None)    # (D, 96) purchase bid
    sell_bid: np.ndarray | None = field(default=None)   # (D, 96) sell bid
    mcv: np.ndarray | None = field(default=None)        # (D, 96) market clearing volume

    def __post_init__(self):
        for name in ("solar_fc", "buy_bid", "sell_bid", "mcv"):
            if getattr(self, name) is None:
                setattr(self, name, np.full_like(self.price, np.nan, dtype=float))

    @property
    def n_days(self) -> int:
        return len(self.dates)


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
_HOLIDAYS_MD = {(1, 26), (3, 8), (8, 15), (10, 2), (11, 12), (12, 25)}


def make_synthetic(n_days: int = 730, seed: int = 7) -> MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    doy = dates.day_of_year.values
    weekend = (dates.day_of_week.values >= 5).astype(float)
    holiday = np.array([(d.month, d.day) in _HOLIDAYS_MD for d in dates])
    h = (np.arange(B) + 0.5) * 24 / B

    # temperature: seasonal + persistent anomaly + diurnal
    anom = np.zeros(n_days)
    for i in range(1, n_days):
        anom[i] = 0.7 * anom[i - 1] + rng.normal(0, 1.6)
    temp_daily = 26 + 8 * np.sin(2 * np.pi * (doy - 60) / 365) + anom
    temp = temp_daily[:, None] + 5 * np.sin(2 * np.pi * (h - 9) / 24)[None, :]

    # demand (GW): cooling load + evening shape + weekday effect + trend
    shape = 1 + 0.12 * np.sin(2 * np.pi * (h - 14) / 24) + 0.08 * np.exp(-(((h - 20) / 1.8) ** 2))
    cooling = 3.0 * np.maximum(temp - 27, 0)
    day_factor = 1 - 0.07 * weekend - 0.10 * holiday
    trend = 1 + 0.00011 * np.arange(n_days)
    demand = (140 + cooling) * shape[None, :] * (day_factor * trend)[:, None]
    demand *= 1 + rng.normal(0, 0.01, (n_days, 1))

    # solar (GW): midday dip, weaker in monsoon
    monsoon = 1 - 0.45 * np.exp(-(((doy - 215) / 35) ** 2))
    cloud = monsoon * np.clip(rng.normal(0.9, 0.1, n_days), 0.4, 1.0)
    solar = 28 * np.maximum(np.sin(np.pi * (h - 6) / 12), 0)[None, :] * cloud[:, None]

    net = demand - solar
    z = (net - net.mean()) / net.std()

    # slow latent factor: fuel cost / hydrology / outages. Gives price lags real signal.
    lvl = np.zeros(n_days)
    for i in range(1, n_days):
        lvl[i] = 0.96 * lvl[i - 1] + rng.normal(0, 0.25)

    noise = np.zeros((n_days, B))
    for b in range(1, B):
        noise[:, b] = 0.8 * noise[:, b - 1] + rng.normal(0, 0.18, n_days)

    price = 3.4 + 1.0 * z + 1.1 * np.exp(1.0 * (z - 1.2)) + lvl[:, None] + noise
    spike = (z > 1.9) & (rng.random((n_days, B)) < 0.10)
    price = np.where(spike, PRICE_CAP - rng.uniform(0, 0.8, (n_days, B)), price)
    price = np.clip(price, 0.5, PRICE_CAP)

    # day-ahead *forecasts* = truth + forecast error (what is really known at bid time).
    # Units are MW so the arrays match the CSV schema.
    temp_fc = temp + rng.normal(0, 0.8, (n_days, 1)) + rng.normal(0, 0.7, (n_days, B))
    demand_fc = demand * 1000 * (1 + rng.normal(0, 0.012, (n_days, B)))
    solar_fc = solar * 1000 * (1 + rng.normal(0, 0.05, (n_days, B)))

    return MarketData(dates, price, temp_fc, demand_fc, holiday, source="synthetic", solar_fc=solar_fc)


# --------------------------------------------------------------------------- #
# Loading and writing
# --------------------------------------------------------------------------- #
def from_long_frame(df: pd.DataFrame, holidays: set | None = None, source: str = "csv") -> MarketData:
    """Build MarketData from a long 15-minute frame.

    Required columns: timestamp (IST, 15-min), price (Rs/kWh).
    Optional: temp_fc, solar_fc, demand_fc, is_holiday. These must be the forecasts that were
    AVAILABLE at bid time, not realised values: using realised weather is the most common
    hidden leakage in electricity price projects.
    Incomplete days are dropped. `holidays` is a set of datetime.date.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.normalize()
    df["block"] = df["timestamp"].dt.hour * 4 + df["timestamp"].dt.minute // 15
    for col in ("temp_fc", "solar_fc", "demand_fc", "buy_bid", "sell_bid", "mcv"):
        if col not in df.columns:
            df[col] = np.nan
    df = df.drop_duplicates(["date", "block"]).sort_values(["date", "block"])
    counts = df.groupby("date")["block"].count()
    df = df[df["date"].isin(counts[counts == B].index)]

    def pivot(col):
        return df.pivot(index="date", columns="block", values=col).sort_index()

    p = pivot("price")
    dates = pd.DatetimeIndex(p.index)
    if len(dates) > 1 and dates.to_series().diff().dropna().dt.days.max() > 1:
        raise ValueError("gaps in dates: fill or split the series before loading")
    if "is_holiday" in df.columns:
        hol = pivot("is_holiday").max(axis=1).astype(bool).values
    else:
        hol = np.array([d.date() in (holidays or set()) for d in dates])
    return MarketData(dates, p.values, pivot("temp_fc").values, pivot("demand_fc").values, hol,
                      source=source, solar_fc=pivot("solar_fc").values, buy_bid=pivot("buy_bid").values,
                      sell_bid=pivot("sell_bid").values, mcv=pivot("mcv").values)


def load_csv(path, holidays: set | None = None, source: str = "csv") -> MarketData:
    """Load the canonical CSV (see module docstring). MCP is converted from Rs/MWh to Rs/kWh."""
    raw = pd.read_csv(path)
    missing = {"date", "block", "mcp_rs_mwh"} - set(raw.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if not raw["block"].between(1, B).all():
        raise ValueError(f"`block` must be in 1..{B}")
    ts = pd.to_datetime(raw["date"]) + pd.to_timedelta((raw["block"] - 1) * 15, unit="min")
    df = pd.DataFrame({"timestamp": ts, "price": raw["mcp_rs_mwh"] / 1000.0})
    for src, dst in (("temp_fc_c", "temp_fc"), ("solar_fc_mw", "solar_fc"), ("demand_fc_mw", "demand_fc"),
                     ("is_holiday", "is_holiday"), ("purchase_bid_mw", "buy_bid"), ("sell_bid_mw", "sell_bid"),
                     ("mcv_mw", "mcv")):
        if src in raw.columns:
            df[dst] = raw[src].values
    return from_long_frame(df, holidays=holidays, source=source)


def to_long_frame(data: MarketData) -> pd.DataFrame:
    """Inverse of load_csv: MarketData -> canonical CSV frame."""
    n = data.n_days
    out = pd.DataFrame({
        "date": np.repeat(data.dates.strftime("%Y-%m-%d").values, B),
        "block": np.tile(np.arange(1, B + 1), n),
        "mcp_rs_mwh": (data.price * 1000.0).ravel().round(2),
        "temp_fc_c": data.temp_fc.ravel().round(2),
        "solar_fc_mw": data.solar_fc.ravel().round(1),
        "demand_fc_mw": data.demand_fc.ravel().round(1),
        "is_holiday": np.repeat(data.holiday.astype(int), B),
        "purchase_bid_mw": data.buy_bid.ravel().round(1),
        "sell_bid_mw": data.sell_bid.ravel().round(1),
        "mcv_mw": data.mcv.ravel().round(1),
    })
    return out[CSV_COLUMNS]


def truncate(data: MarketData, end_date) -> MarketData:
    """Keep only delivery days up to and including `end_date` (used for earlier evaluation periods)."""
    n = int((data.dates <= pd.Timestamp(end_date)).sum())
    if n == 0:
        raise ValueError(f"no data on or before {end_date}")
    cut = {}
    for f in fields(data):
        v = getattr(data, f.name)
        if isinstance(v, (np.ndarray, pd.DatetimeIndex)) and len(v) == len(data.dates):
            cut[f.name] = v[:n]
    return replace(data, **cut)


def check_data(data: MarketData) -> list[str]:
    """Data-quality warnings. An empty list means nothing suspicious was found."""
    w = []
    if data.n_days < 365:
        w.append(f"only {data.n_days} days: too short for seasonal conclusions (aim for 2+ years)")
    if np.isnan(data.price).any():
        w.append("NaN values in price")
    if (data.price > PRICE_CAP * 1.001).any():
        w.append(f"prices above the configured cap of Rs {PRICE_CAP}/kWh: older cap regime or HP-DAM data mixed in?")
    at_cap = float(np.mean(data.price >= PRICE_CAP * 0.999))
    if at_cap > 0.05:
        w.append(f"{at_cap:.1%} of blocks sit at the price cap: check units (Rs/MWh vs Rs/kWh) and regimes")
    const = data.price.std(axis=1) == 0
    cap_days = const & (data.price.min(axis=1) >= PRICE_CAP * 0.999)
    odd = const & ~cap_days
    if odd.any():
        w.append(f"{int(odd.sum())} day(s) have one constant price below the cap (forward-filled or missing data?): "
                 + ", ".join(str(d.date()) for d in data.dates[odd][:5]))
    if cap_days.any():
        w.append(f"{int(cap_days.sum())} day(s) are entirely at the price cap (plausible in a supply shortage; "
                 "check against the exchange's published data)")
    for name, arr in (("temp_fc", data.temp_fc), ("solar_fc", data.solar_fc), ("demand_fc", data.demand_fc)):
        if np.isnan(arr).all():
            w.append(f"{name} not provided: its feature group is skipped")
    return w
