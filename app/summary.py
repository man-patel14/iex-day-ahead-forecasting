"""Pure-numpy/pandas summaries behind the dashboard (no Streamlit import, so they are unit-tested)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def day_view(z, i: int) -> dict:
    """KPIs and series for test day `i` from a loaded predictions.npz-like mapping."""
    q, actual, bench, buy = z["quantiles"][i], z["actual"][i], z["benchmark"][i], z["buy"][i].astype(bool)
    cost_agent = float(np.where(buy, actual, bench).mean())
    cost_bench = float(bench.mean())
    return {
        "blocks_to_buy": int(buy.sum()),
        "day_mae": float(np.abs(actual - q[:, 1]).mean()),
        "coverage": float(((actual >= q[:, 0]) & (actual <= q[:, 2])).mean()),
        "cost_agent": cost_agent,
        "cost_benchmark": cost_bench,
        "saving_pct": 100 * (cost_bench - cost_agent) / cost_bench,
        "mean_confidence": float(z["confidence"][i].mean()),
    }


def monthly_table(z) -> pd.DataFrame:
    """Per-month test-year summary: forecast error, band coverage and the agent's saving."""
    dates = pd.to_datetime(z["dates"])
    q, actual, bench, buy = z["quantiles"], z["actual"], z["benchmark"], z["buy"].astype(bool)
    rows = []
    for period, idx in pd.Series(np.arange(len(dates)), index=dates).groupby(dates.to_period("M")):
        ix = idx.values
        cost_a = np.where(buy[ix], actual[ix], bench[ix]).mean()
        cost_b = bench[ix].mean()
        rows.append({
            "month": str(period),
            "days": len(ix),
            "MAE P50 (Rs/kWh)": float(np.abs(actual[ix] - q[ix][..., 1]).mean()),
            "MAE 7-day block mean (Rs/kWh)": float(np.abs(actual[ix] - bench[ix]).mean()),
            "Cov80": float(((actual[ix] >= q[ix][..., 0]) & (actual[ix] <= q[ix][..., 2])).mean()),
            "agent saving vs benchmark (%)": 100 * (cost_b - cost_a) / cost_b,
        })
    return pd.DataFrame(rows)


def models_table(results: dict) -> pd.DataFrame:
    """Model comparison from results.json (`models` maps model name -> metrics dict)."""
    df = pd.DataFrame(results["models"]).T
    df.index.name = "model"
    keep = [c for c in ("MAE", "RMSE", "Pinball", "Cov80") if c in df.columns]
    return df[keep].reset_index()


def agent_table(results: dict) -> pd.DataFrame:
    df = pd.DataFrame(results["agent"])
    return df.rename(columns={"strategy": "strategy"})


def display_table(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """Round numbers for st.dataframe. A column with missing values becomes text with 'n/a'
    (a mixed float/str column cannot be converted to an Arrow table)."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            if out[c].isna().any():
                out[c] = out[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.{digits}f}")
            else:
                out[c] = out[c].round(digits)
    return out
