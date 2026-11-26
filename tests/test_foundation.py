import numpy as np
import pandas as pd
import pytest

from iex_forecast.chronos_adapter import frame_to_quantiles, history_to_frame


def test_history_to_frame_truncates_and_orders():
    h = np.arange(5 * 96, dtype=float).reshape(5, 96)
    df = history_to_frame({7: h}, context_days=2)
    assert len(df) == 192 and df["target"].iloc[0] == 3 * 96
    assert df["timestamp"].is_monotonic_increasing


def test_frame_to_quantiles_sorts_and_checks():
    ts = pd.date_range("2000-01-01", periods=96, freq="15min")
    df = pd.DataFrame({"id": "1", "timestamp": ts, "0.1": 5.0, "0.5": 4.0, "0.9": 6.0})
    q = frame_to_quantiles(df, ["1"])
    assert q.shape == (1, 96, 3) and (q[0, :, 0] <= q[0, :, 1]).all() and (q[0, :, 1] <= q[0, :, 2]).all()
    with pytest.raises(ValueError):
        frame_to_quantiles(df.iloc[:50], ["1"])
