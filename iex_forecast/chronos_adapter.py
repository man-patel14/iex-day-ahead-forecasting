"""Zero-shot Chronos-2 adapter (Apache-2.0 foundation model, quantile output).

Status: the pure helpers below are unit-tested with a fake pipeline. The call into the real
`chronos-forecasting` package could NOT be executed in the build sandbox (no model download), so
`ChronosForecaster` is verified only when run on a machine with the package and model weights.

Leak guard: `forecast_days` receives, per test day, only the history slice supplied by the caller
(prices up to day D - INFO_LAG_DAYS). It never sees the data object.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)
BLOCKS = 96


def history_to_frame(histories: dict, context_days: int, freq: str = "15min") -> pd.DataFrame:
    """Long frame (id, timestamp, target) from {id: (n_days, 96) array}; last `context_days` days only."""
    parts = []
    for key, h in histories.items():
        h = np.asarray(h, dtype=float)[-context_days:]
        values = h.reshape(-1)
        ts = pd.date_range("2000-01-01", periods=len(values), freq=freq)
        parts.append(pd.DataFrame({"id": str(key), "timestamp": ts, "target": values}))
    return pd.concat(parts, ignore_index=True)


def frame_to_quantiles(pred_df: pd.DataFrame, keys, quantiles=QUANTILES) -> np.ndarray:
    """Convert a predict_df result to an array (n_keys, 96, n_quantiles). Column names may be 0.1 or '0.1'."""
    cols = {}
    for q in quantiles:
        match = [c for c in pred_df.columns if str(c) == str(q) or (isinstance(c, float) and abs(c - q) < 1e-9)]
        if not match:
            raise ValueError(f"quantile column {q} not found in {list(pred_df.columns)}")
        cols[q] = match[0]
    out = np.empty((len(keys), BLOCKS, len(quantiles)))
    for i, key in enumerate(keys):
        sub = pred_df[pred_df["id"].astype(str) == str(key)].sort_values("timestamp")
        if len(sub) != BLOCKS:
            raise ValueError(f"id {key}: expected {BLOCKS} forecast rows, got {len(sub)}")
        for j, q in enumerate(quantiles):
            out[i, :, j] = sub[cols[q]].to_numpy(dtype=float)
    return np.sort(out, axis=2)   # enforce non-crossing quantiles


class ChronosForecaster:
    """Batch day-ahead forecaster. Requires `pip install chronos-forecasting` and downloads weights."""

    def __init__(self, model_id: str = "amazon/chronos-2", context_days: int = 14, batch: int = 16, device: str = "cpu"):
        from chronos import Chronos2Pipeline  # imported lazily: optional dependency
        self.pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device)
        self.context_days, self.batch = context_days, batch

    def forecast_days(self, histories: dict) -> np.ndarray:
        keys = list(histories)
        chunks = []
        for i in range(0, len(keys), self.batch):
            ks = keys[i:i + self.batch]
            df = history_to_frame({k: histories[k] for k in ks}, self.context_days)
            pred = self.pipeline.predict_df(df, prediction_length=BLOCKS, quantile_levels=list(QUANTILES),
                                            id_column="id", timestamp_column="timestamp", target="target")
            chunks.append(frame_to_quantiles(pred, ks))
        return np.concatenate(chunks)
