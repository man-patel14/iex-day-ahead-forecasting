import numpy as np

import pandas as pd

from app.summary import agent_table, day_view, display_table, models_table, monthly_table


def _fake(n=70, seed=0):
    rng = np.random.default_rng(seed)
    actual = rng.uniform(2, 8, (n, 96))
    pq = np.stack([actual - 1, actual + rng.normal(0, 0.3, actual.shape), actual + 1], axis=-1)
    return {
        "quantiles": pq, "actual": actual, "benchmark": np.full_like(actual, 5.0),
        "buy": actual < 5.0, "confidence": np.full_like(actual, 0.7),
        "dates": np.array([str(d.date()) for d in __import__("pandas").date_range("2026-01-01", periods=n)]),
    }


def test_day_view_kpis_are_consistent():
    z = _fake()
    k = day_view(z, 3)
    assert k["blocks_to_buy"] == int((z["actual"][3] < 5.0).sum())
    assert k["coverage"] == 1.0                       # band was built around the actual
    assert k["saving_pct"] > 0                        # buying only when below the benchmark saves money
    assert abs(k["cost_benchmark"] - 5.0) < 1e-9


def test_monthly_table_splits_by_month_and_counts_days():
    t = monthly_table(_fake(n=70))
    assert list(t["month"]) == ["2026-01", "2026-02", "2026-03"]
    assert t["days"].sum() == 70 and (t["agent saving vs benchmark (%)"] > 0).all()


def test_results_tables():
    res = {"models": {"a": {"MAE": 0.5, "RMSE": 1.0, "Pinball": 0.2, "Cov80": 0.8, "DirAcc": 0.6}},
           "agent": [{"strategy": "s", "avg cost (Rs/kWh)": 4.0}]}
    assert list(models_table(res).columns) == ["model", "MAE", "RMSE", "Pinball", "Cov80"]
    assert agent_table(res).shape == (1, 2)


def test_display_table_shows_missing_as_na_and_rounds():
    out = display_table(pd.DataFrame({"m": ["a", "b"], "x": [0.123456, float("nan")]}))
    assert list(out["x"]) == ["0.123", "n/a"] and list(out["m"]) == ["a", "b"]
