"""Point, probabilistic and directional metrics, stratified reports, Diebold-Mariano."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import PERIODS, PRICE_CAP, QUANTILES


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def pinball(y, pq, quantiles=QUANTILES):
    """Mean pinball loss. pq has shape (..., Q)."""
    q = np.array(quantiles)
    diff = y[..., None] - pq
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y, lo, hi):
    """Share of actuals inside [lo, hi]. For P10-P90 the target is 0.80."""
    return float(np.mean((y >= lo) & (y <= hi)))


def directional_accuracy(y, p, ref):
    """Share of blocks where sign(actual - ref) == sign(forecast - ref).

    `ref` is the last price known at bid time for that block (last known day at bid time).
    """
    a, f = np.sign(y - ref), np.sign(p - ref)
    if not (f != 0).any():      # forecast IS the reference (e.g. naive lag-2): undefined
        return float("nan")
    keep = a != 0
    return float(np.mean(a[keep] == f[keep]))


def summarise(y, p50, ref, pq=None):
    out = {"MAE": mae(y, p50), "RMSE": rmse(y, p50), "DirAcc": directional_accuracy(y, p50, ref)}
    if pq is not None:
        out["Pinball"] = pinball(y, pq)
        out["Cov80"] = coverage(y, pq[..., 0], pq[..., -1])
    return out


def stratified_mae(y, p, dates):
    """MAE by block-of-day period, season, weekday/weekend and spike days."""
    n_days, n_blocks = y.shape
    hours = np.arange(n_blocks) / 4
    rows = []
    for name, (a, b) in PERIODS.items():
        m = (hours >= a) & (hours < b)
        rows.append(("time of day", name, mae(y[:, m], p[:, m])))
    month = dates.month.values
    seasons = {"winter (Dec-Feb)": [12, 1, 2], "summer (Mar-Jun)": [3, 4, 5, 6],
               "monsoon (Jul-Sep)": [7, 8, 9], "post-monsoon (Oct-Nov)": [10, 11]}
    for name, ms in seasons.items():
        m = np.isin(month, ms)
        if m.any():
            rows.append(("season", name, mae(y[m], p[m])))
    wk = dates.day_of_week.values >= 5
    rows.append(("day type", "weekday", mae(y[~wk], p[~wk])))
    if wk.any():
        rows.append(("day type", "weekend", mae(y[wk], p[wk])))
    sp = (y >= 0.95 * PRICE_CAP).sum(axis=1) >= 12      # 3+ hours at the cap
    if sp.any():
        rows.append(("regime", f"cap days: 12+ blocks at cap (n={int(sp.sum())})", mae(y[sp], p[sp])))
        rows.append(("regime", "other days", mae(y[~sp], p[~sp])))
    return pd.DataFrame(rows, columns=["group", "slice", "MAE"])


def diebold_mariano(y, p1, p2, max_lag=7):
    """DM test on daily mean absolute errors (HAC / Newey-West variance).

    Negative statistic => model 1 has lower loss than model 2. Returns (stat, p_value).
    """
    return paired_loss_test(np.abs(y - p1).mean(axis=1), np.abs(y - p2).mean(axis=1), max_lag)


def paired_loss_test(loss_a, loss_b, max_lag=7):
    """Diebold-Mariano-style test on two daily loss series (HAC variance).

    Negative statistic => series A has the lower loss. Returns (stat, p_value).
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n, dm = len(d), d.mean()
    s = np.var(d)
    for k in range(1, max_lag + 1):
        s += 2 * (1 - k / (max_lag + 1)) * np.mean((d[k:] - dm) * (d[:-k] - dm))
    if s <= 0:
        return float("nan"), float("nan")
    stat = dm / np.sqrt(s / n)
    return float(stat), float(2 * (1 - norm.cdf(abs(stat))))


def block_bootstrap_ci(d, block: int = 7, n_boot: int = 2000, level: float = 0.95, seed: int = 0):
    """Moving-block bootstrap confidence interval for the mean of a daily series.

    Blocks of `block` consecutive days keep the day-to-day dependence of errors. Returns (lo, hi).
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    block = min(block, n)
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))
