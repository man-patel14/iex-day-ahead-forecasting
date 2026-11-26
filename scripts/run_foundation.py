"""Zero-shot foundation-model comparison on the SAME test days as the GBM backtest.

Usage (after run_backtest.py has written <results>/predictions.npz):
    python scripts/run_foundation.py --data data/iex_dam_weather.csv --results results --out results/foundation.md
For the earlier fold, run run_backtest.py with --end-date and a separate --out, then pass that folder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from iex_forecast.chronos_adapter import ChronosForecaster
from iex_forecast.config import INFO_LAG_DAYS
from iex_forecast.data import load_csv
from iex_forecast.metrics import block_bootstrap_ci, coverage, mae, paired_loss_test, pinball


def compare(actual, gbm_q, fm_q, seasonal):
    """Metrics table and paired tests. Quantile arrays: (days, 96, 3) = P10, P50, P90."""
    rows = {}
    for name, q in (("GBM (chosen rung, calibrated band)", gbm_q), ("Chronos-2 zero-shot (uncalibrated band)", fm_q)):
        rows[name] = {"MAE": mae(actual, q[..., 1]), "Pinball": pinball(actual, q),
                      "Cov80": coverage(actual, q[..., 0], q[..., 2])}
    rows["Seasonal naive (same block last week)"] = {"MAE": mae(actual, seasonal), "Pinball": np.nan, "Cov80": np.nan}
    loss_g = np.abs(actual - gbm_q[..., 1]).mean(axis=1)
    loss_f = np.abs(actual - fm_q[..., 1]).mean(axis=1)
    stat, p = paired_loss_test(loss_f, loss_g)
    lo, hi = block_bootstrap_ci(loss_f - loss_g)
    return pd.DataFrame(rows).T, (stat, p, lo, hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/foundation.md")
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--context-days", type=int, default=14)
    args = ap.parse_args()

    data = load_csv(args.data)
    npz = np.load(Path(args.results) / "predictions.npz")
    dates = pd.to_datetime(npz["dates"])
    idx = np.array([int(np.where(data.dates == d)[0][0]) for d in dates])
    actual, gbm_q = npz["actual"], npz["quantiles"]

    histories = {int(d): data.price[: d - INFO_LAG_DAYS + 1].copy() for d in idx}  # bid-time slice only
    fm = ChronosForecaster(args.model, args.context_days)
    fm_q = fm.forecast_days(histories)
    seasonal = np.stack([data.price[d - 7] for d in idx])

    table, (stat, p, lo, hi) = compare(actual, gbm_q, fm_q, seasonal)
    text = (f"# Foundation-model comparison ({dates[0].date()} to {dates[-1].date()}, {len(idx)} days)\n\n"
            f"Model: {args.model}, zero-shot, context {args.context_days} days, prices only.\n\n"
            "```\n" + table.round(3).to_string() + "\n```\n\n"
            f"Paired MAE difference (Chronos minus GBM, daily): DM stat {stat:.2f}, p = {p:.3f}; "
            f"95% block-bootstrap CI of the mean difference [{lo:.3f}, {hi:.3f}] Rs/kWh. "
            f"Positive values mean Chronos is worse.\n")
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
