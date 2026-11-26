import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "add_weather", Path(__file__).resolve().parents[1] / "scripts" / "add_weather.py")
aw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aw)


def fake_get(url):
    q = dict(p.split("=") for p in url.split("?")[1].split("&"))
    idx = pd.date_range(q["start_date"], q["end_date"] + " 23:00", freq="h")
    lat = float(q["latitude"])
    return {"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in idx],
                       aw.VAR: [lat + t.hour for t in idx]}}


def test_add_weather_repeats_hour_over_four_blocks():
    dates = pd.date_range("2025-01-01", periods=3)
    df = pd.DataFrame([(d, b, 4000.0) for d in dates for b in range(1, 97)],
                      columns=["date", "block", "mcp_rs_mwh"])
    out, _ = aw.add_weather(df, get=fake_get)
    day = out[out.date == "2025-01-02"].temp_fc_c.to_numpy()
    assert len(out) == len(df) and not np.isnan(out.temp_fc_c).any()
    assert (day[0:4] == day[0]).all() and day[4] == pytest.approx(day[0] + 1)


def test_bad_response_fails_loudly():
    with pytest.raises(SystemExit):
        aw.parse_hourly({"error": True})


def test_chunking_covers_long_ranges():
    s = aw.fetch_city(1.0, 2.0, "2025-01-01", "2025-12-31", get=fake_get)
    assert len(s) == 365 * 24 and not s.index.has_duplicates


def test_days_before_the_archive_are_not_requested_and_stay_empty():
    calls = []

    def spy(url):
        calls.append(url)
        return fake_get(url)

    dates = pd.date_range("2023-12-30", periods=5)           # two days before, three after 2024-01-01
    df = pd.DataFrame([(d, b, 4000.0) for d in dates for b in range(1, 97)],
                      columns=["date", "block", "mcp_rs_mwh"])
    out, _ = aw.add_weather(df, get=spy)
    assert all("start_date=2024-01-01" in u for u in calls)
    early = out[out.date < "2024-01-01"].temp_fc_c
    assert early.isna().all() and out[out.date >= "2024-01-01"].temp_fc_c.notna().all()
