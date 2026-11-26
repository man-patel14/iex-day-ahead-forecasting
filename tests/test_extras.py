import numpy as np
import pandas as pd
import pytest

from iex_forecast.data import check_data, from_long_frame, load_csv, make_synthetic, to_long_frame
from iex_forecast.drift import psi, rolling_mae_alert
from iex_forecast.ensemble import blend, inverse_error_weights
from iex_forecast.features import FEATURE_SETS, build_all, column_index, usable_feature_sets
from iex_forecast.foundation import (evaluate_point_forecaster, kronos_forecaster,
                                     seasonal_naive_fn)
from iex_forecast.models import seasonal_naive
from iex_forecast.news import headline_score, news_for_delivery_days
from iex_forecast.spike import walk_forward_spike


def _long(d):
    ts = pd.concat([pd.Series(pd.date_range(x, periods=96, freq="15min")) for x in d.dates], ignore_index=True)
    return pd.DataFrame({"timestamp": ts, "price": d.price.ravel(),
                         "temp_fc": d.temp_fc.ravel(), "demand_fc": d.demand_fc.ravel()})


def test_from_long_frame_roundtrip():
    d = make_synthetic(n_days=20, seed=3)
    df = _long(d)
    df = pd.concat([df, df.iloc[:5]])            # duplicates must be dropped
    back = from_long_frame(df)
    assert np.allclose(back.price, d.price) and back.n_days == 20
    assert np.isnan(back.solar_fc).all()         # absent column becomes NaN, not an error


def test_from_long_frame_drops_incomplete_day():
    d = make_synthetic(n_days=5, seed=3)
    assert from_long_frame(_long(d).iloc[:-10]).n_days == 4


def test_csv_roundtrip_and_unit_conversion(tmp_path):
    d = make_synthetic(n_days=10, seed=3)
    path = tmp_path / "market.csv"
    frame = to_long_frame(d)
    frame.to_csv(path, index=False)
    back = load_csv(path)
    assert np.allclose(back.price, d.price, atol=1e-5)         # Rs/MWh in the file, Rs/kWh in memory
    assert np.allclose(back.solar_fc, d.solar_fc, atol=0.1)
    assert (back.holiday == d.holiday).all()


def test_csv_unit_is_rs_per_mwh(tmp_path):
    rows = [{"date": f"2024-01-0{day}", "block": b, "mcp_rs_mwh": 4000.0} for day in (1, 2) for b in range(1, 97)]
    path = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    d = load_csv(path)
    assert np.allclose(d.price, 4.0)
    assert np.isnan(d.temp_fc).all() and np.isnan(d.demand_fc).all()


def test_csv_rejects_bad_schema(tmp_path):
    pd.DataFrame({"date": ["2024-01-01"], "block": [1]}).to_csv(tmp_path / "a.csv", index=False)
    with pytest.raises(ValueError):
        load_csv(tmp_path / "a.csv")
    pd.DataFrame({"date": ["2024-01-01"], "block": [97], "mcp_rs_mwh": [4000.0]}).to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError):
        load_csv(tmp_path / "b.csv")


def test_check_data_flags_problems():
    d = make_synthetic(n_days=30, seed=3)
    w = check_data(d)
    assert any("too short" in x for x in w)
    d.price[5, 10] = 12.0
    assert any("above the configured cap" in x for x in check_data(d))
    assert not any("NaN" in x for x in check_data(make_synthetic(n_days=400, seed=3)))


def test_usable_feature_sets_skips_missing_groups():
    d = make_synthetic(n_days=30, seed=3)
    assert list(usable_feature_sets(d)) == [k for k in FEATURE_SETS if k != "price+volumes"]   # no volumes in synthetic data
    d.solar_fc[:] = np.nan
    d.demand_fc[:] = np.nan
    assert list(usable_feature_sets(d)) == ["price_only", "price+weather", "price+weather_anom"]


def test_features_survive_missing_forecasts():
    d = make_synthetic(n_days=40, seed=3)
    d.temp_fc[:] = np.nan
    X = build_all(d)
    assert np.isfinite(X[20][:, column_index(FEATURE_SETS["price_only"])]).all()


def test_news_uses_previous_day_only():
    h = pd.DataFrame({"date": ["2024-05-01"], "headline": ["Coal shortage and heatwave hit grid"]})
    f = news_for_delivery_days(h, pd.to_datetime(["2024-05-01", "2024-05-02"]))
    assert f.loc["2024-05-01", "news_count"] == 0      # same-day news NOT usable
    assert f.loc["2024-05-02", "news_count"] == 1 and f.loc["2024-05-02", "news_score"] < 0
    assert headline_score("Rain brings relief, demand falls") > 0


def test_ensemble_weights_favour_better_model():
    y = np.random.default_rng(0).normal(5, 1, (50, 96))
    p = {"good": y + 0.1, "bad": y + 1.0}
    w = inverse_error_weights(y, p)
    assert w["good"] > w["bad"] and abs(sum(w.values()) - 1) < 1e-9
    assert np.abs(blend(p, w) - y).mean() < np.abs(p["bad"] - y).mean()


def test_drift_detectors():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    assert psi(base, rng.normal(0, 1, 5000)) < 0.1
    assert psi(base, rng.normal(1.5, 1, 5000)) > 0.25
    mae = np.r_[np.full(100, 0.5), np.full(40, 2.0)]
    assert rolling_mae_alert(mae)[-1] and not rolling_mae_alert(mae)[70]


def test_foundation_harness_matches_seasonal_naive_and_blocks_peeking():
    d = make_synthetic(n_days=60, seed=4)
    days = np.arange(30, 60)
    out = evaluate_point_forecaster(d, days, seasonal_naive_fn)
    assert np.allclose(out, seasonal_naive(d, days, 7))

    def cheater(history, date, t, dm):
        history[-1] = 0                                # must fail: history is read-only
        return np.zeros(96)

    with pytest.raises(ValueError):
        evaluate_point_forecaster(d, days, cheater)
    with pytest.raises(NotImplementedError):
        kronos_forecaster()


def test_spike_classifier_runs():
    d = make_synthetic(n_days=200, seed=5)
    X = build_all(d)
    p = walk_forward_spike(X, d.price, column_index(FEATURE_SETS["price+weather+solar+demand"]),
                           test_start=120, refit_every=40)
    assert p.shape == (80, 96) and (p >= 0).all() and (p <= 1).all()


def test_temperature_anomaly_uses_only_forecasts_up_to_day_d():
    from iex_forecast.features import day_features
    d = make_synthetic(n_days=80, seed=6)
    ref = day_features(d, 50)
    d2 = make_synthetic(n_days=80, seed=6)
    d2.temp_fc[51:] = 99.0                      # forecasts for later days must not matter
    assert np.allclose(day_features(d2, 50), ref, equal_nan=True)
    i = column_index(["temp_anom"])[0]
    assert abs(np.nanmean(ref[:, i])) < 8       # anomaly, not the raw level


def test_volume_features_use_only_last_known_day():
    from iex_forecast.config import INFO_LAG_DAYS
    from iex_forecast.features import day_features
    rng = np.random.default_rng(1)
    d = make_synthetic(n_days=60, seed=6)
    d.buy_bid = rng.uniform(10000, 15000, d.price.shape)
    d.sell_bid = rng.uniform(15000, 25000, d.price.shape)
    d.mcv = rng.uniform(5000, 9000, d.price.shape)
    assert "price+volumes" in usable_feature_sets(d)
    ref = day_features(d, 40)
    last = 40 - INFO_LAG_DAYS
    for arr in (d.buy_bid, d.sell_bid, d.mcv):
        arr[last + 1:] = 1e9                    # outcomes after the last known day must not matter
    assert np.allclose(day_features(d, 40), ref, equal_nan=True)
    i = column_index(["vol_ratio_last"])[0]
    assert (ref[:, i] > 1).all()


def test_volume_columns_round_trip(tmp_path):
    d = make_synthetic(n_days=10, seed=3)
    d.buy_bid[:] = 12000.0
    d.sell_bid[:] = 18000.0
    d.mcv[:] = 7000.0
    to_long_frame(d).to_csv(tmp_path / "m.csv", index=False)
    back = load_csv(tmp_path / "m.csv")
    assert np.allclose(back.sell_bid, 18000.0) and np.allclose(back.mcv, 7000.0)


def test_truncate_keeps_every_array_aligned():
    from iex_forecast.data import truncate
    d = make_synthetic(n_days=50, seed=2)
    t = truncate(d, d.dates[29])
    assert t.n_days == 30 and t.price.shape == (30, 96) and t.sell_bid.shape == (30, 96)
    assert len(t.holiday) == 30 and np.allclose(t.price, d.price[:30], equal_nan=True)
    with pytest.raises(ValueError):
        truncate(d, "1999-01-01")


def test_constant_price_days_are_split_into_cap_days_and_suspicious_days():
    d = make_synthetic(n_days=400, seed=3)
    d.price[10, :] = 10.0           # a whole day at the cap
    d.price[20, :] = 4.0            # a whole day at one price below the cap
    w = " | ".join(check_data(d))
    assert "1 day(s) are entirely at the price cap" in w
    assert "1 day(s) have one constant price below the cap" in w
