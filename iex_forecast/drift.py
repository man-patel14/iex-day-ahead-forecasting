"""Cheap production monitoring: input drift (PSI) and error drift (rolling MAE alert)."""
from __future__ import annotations

import numpy as np


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index. Rule of thumb: <0.1 stable, 0.1-0.25 watch, >0.25 drifted."""
    expected, actual = np.asarray(expected).ravel(), np.asarray(actual).ravel()
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def rolling_mae_alert(daily_mae, window: int = 30, factor: float = 1.5):
    """Flag days where the recent MAE exceeds `factor` x the long-run MAE before the window."""
    daily_mae = np.asarray(daily_mae, dtype=float)
    alerts = np.zeros(len(daily_mae), dtype=bool)
    for i in range(2 * window, len(daily_mae)):
        base = daily_mae[:i - window].mean()
        alerts[i] = daily_mae[i - window:i].mean() > factor * base
    return alerts
