"""Plug-in point for foundation models (TimesFM, Kronos) with a structural leak guard.

Any forecaster is a function:
    fn(history, date, temp_fc_day, demand_fc_day) -> array of shape (96,)
where `history` is realised prices up to the last day known at bid time ONLY, shape (n_known_days, 96).
`evaluate_point_forecaster` hands it exactly that slice, so a bad adapter cannot peek.

The two adapters below are intentionally left unimplemented: the libraries were not available
when this project was built, and an untested adapter would look finished but may not work. Phase 3, step 8 of docs/ROADMAP.md describes the work.
"""
from __future__ import annotations

import numpy as np

from .config import INFO_LAG_DAYS


def evaluate_point_forecaster(data, test_days, fn) -> np.ndarray:
    """Run `fn` for each test day with only bid-time information. Returns (n_test, 96)."""
    out = []
    for d in test_days:
        history = data.price[: d - INFO_LAG_DAYS + 1].copy()
        history.setflags(write=False)
        pred = np.asarray(fn(history, data.dates[d], data.temp_fc[d], data.demand_fc[d]))
        if pred.shape != (96,):
            raise ValueError(f"forecaster must return 96 values, got {pred.shape}")
        out.append(pred)
    return np.stack(out)


def seasonal_naive_fn(history, date, temp_fc_day, demand_fc_day):
    """Reference forecaster: same block last week (day d-7 = history[-5])."""
    return history[-(7 - INFO_LAG_DAYS) - 1]


def timesfm_forecaster():
    raise NotImplementedError(
        "Step 8: install the TimesFM package per its official README, flatten `history` "
        "to a 1-D series (days x 96 blocks, in time order), forecast 96 steps ahead "
        "(day d starts right after the last known day when INFO_LAG_DAYS = 1, so forecast 96 steps), "
        "and return a (96,) array. Start zero-shot; add covariates later."
    )


def kronos_forecaster():
    raise NotImplementedError(
        "Step 8: Kronos is reported to be a foundation model for financial candlestick "
        "(K-line) data; verify that in its repo/paper. Feeding it price-only series is an "
        "experiment. Report the result honestly even if it loses to the baselines."
    )
