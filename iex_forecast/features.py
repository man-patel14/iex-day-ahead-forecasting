"""Feature construction with a structural no-leakage guarantee.

`day_features(data, d)` builds features for delivery day `d`. Its ONLY access to
realised prices is `data.price[: d - INFO_LAG_DAYS + 1]`, i.e. days up to the last day
known at bid time. Weather, solar and demand enter only as day-ahead FORECASTS for day d.
tests/test_leakage.py proves this by corrupting every price after the last known day and
checking that the features do not change.
"""
from __future__ import annotations

import warnings

import numpy as np

from .config import BLOCKS_PER_DAY, INFO_LAG_DAYS, MIN_HISTORY_DAYS
from .data import MarketData

PRICE = ["p_last", "p_d7", "last_mean", "last_max", "last_std", "win7_mean", "win7_block_mean"]
CALENDAR = ["block", "dow", "weekend", "month", "doy_sin", "doy_cos", "holiday"]
WEATHER = ["temp_fc", "temp_fc_dmax", "temp_fc_dmean"]
# Temperature anomaly: forecast minus the mean forecast for the same block over the previous
# 30 days (forecasts of past days are known at bid time). Removes the seasonal level so the
# model cannot use raw temperature as a stand-in for the calendar.
WEATHER_ANOM = ["temp_anom", "temp_anom_dmean"]
ANOM_WINDOW = 30
# Auction tightness of the LAST KNOWN day (supply vs demand in the exchange auction).
VOLUME = ["vol_ratio_last", "vol_ratio_last_mean", "mcv_last"]
SOLAR = ["solar_fc", "solar_fc_dmax"]
DEMAND = ["demand_fc", "demand_fc_dmax"]

ALL_NAMES = PRICE + CALENDAR + WEATHER + WEATHER_ANOM + VOLUME + SOLAR + DEMAND

# Ablation ladder. Each rung adds one information source.
FEATURE_SETS = {
    "price_only": PRICE + CALENDAR,
    "price+weather": PRICE + CALENDAR + WEATHER,
    "price+weather_anom": PRICE + CALENDAR + WEATHER_ANOM,      # side branch, not part of the ladder
    "price+volumes": PRICE + CALENDAR + VOLUME,                 # side branch, not part of the ladder
    "price+weather+solar": PRICE + CALENDAR + WEATHER + SOLAR,
    "price+weather+solar+demand": PRICE + CALENDAR + WEATHER + SOLAR + DEMAND,
}


def usable_feature_sets(data: MarketData) -> dict:
    """Feature sets whose exogenous inputs are actually present in `data`."""
    present = {"temp_fc": data.temp_fc, "temp_anom": data.temp_fc,
               "solar_fc": data.solar_fc, "demand_fc": data.demand_fc,
               "vol_ratio_last": data.sell_bid, "vol_ratio_last_mean": data.sell_bid, "mcv_last": data.mcv}
    out = {}
    for name, cols in FEATURE_SETS.items():
        need = [c for c in cols if c in present]
        if all(not np.isnan(present[c]).all() for c in need):
            out[name] = cols
    return out


def day_features(data: MarketData, d: int) -> np.ndarray:
    """Return (96, len(ALL_NAMES)) feature matrix for delivery day d."""
    if d < MIN_HISTORY_DAYS:
        raise ValueError(f"need d >= {MIN_HISTORY_DAYS}")
    last_known = d - INFO_LAG_DAYS
    known = data.price[: last_known + 1]          # <- the ONLY price access (volumes: only index last_known)

    p_last = known[last_known]                         # most recent day known at bid time
    p_d7 = known[last_known - (7 - INFO_LAG_DAYS)]     # same weekday last week (day d-7)
    win = known[last_known - 6: last_known + 1]        # trailing 7 known days

    date = data.dates[d]
    ones = np.ones(BLOCKS_PER_DAY)
    doy = date.day_of_year
    tf, sf, df_ = data.temp_fc[d], data.solar_fc[d], data.demand_fc[d]
    past_tf = data.temp_fc[max(0, d - ANOM_WINDOW): d]     # forecasts for days before d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)    # all-NaN slices stay NaN
        anom = tf - np.nanmean(past_tf, axis=0) if len(past_tf) else np.full(BLOCKS_PER_DAY, np.nan)

    buy, sell, mcv = data.buy_bid[last_known], data.sell_bid[last_known], data.mcv[last_known]
    ratio = sell / np.where(buy > 0, buy, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ratio_mean = np.nanmean(ratio) if np.isfinite(ratio).any() else np.nan

    cols = {
        "vol_ratio_last": ratio,
        "vol_ratio_last_mean": ones * ratio_mean,
        "mcv_last": mcv,
        "p_last": p_last,
        "p_d7": p_d7,
        "last_mean": ones * p_last.mean(),
        "last_max": ones * p_last.max(),
        "last_std": ones * p_last.std(),
        "win7_mean": ones * win.mean(),
        "win7_block_mean": win.mean(axis=0),
        "block": np.arange(BLOCKS_PER_DAY, dtype=float),
        "dow": ones * date.day_of_week,
        "weekend": ones * float(date.day_of_week >= 5),
        "month": ones * date.month,
        "doy_sin": ones * np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": ones * np.cos(2 * np.pi * doy / 365.25),
        "holiday": ones * float(data.holiday[d]),
        "temp_fc": tf,
        "temp_fc_dmax": ones * tf.max(),
        "temp_fc_dmean": ones * tf.mean(),
        "temp_anom": anom,
        "temp_anom_dmean": ones * np.mean(anom),
        "solar_fc": sf,
        "solar_fc_dmax": ones * sf.max(),
        "demand_fc": df_,
        "demand_fc_dmax": ones * df_.max(),
    }
    return np.column_stack([cols[n] for n in ALL_NAMES])


def build_all(data: MarketData) -> np.ndarray:
    """(D, 96, F) array; days before MIN_HISTORY_DAYS are NaN."""
    out = np.full((data.n_days, BLOCKS_PER_DAY, len(ALL_NAMES)), np.nan)
    for d in range(MIN_HISTORY_DAYS, data.n_days):
        out[d] = day_features(data, d)
    return out


def column_index(names: list[str]) -> list[int]:
    return [ALL_NAMES.index(n) for n in names]
