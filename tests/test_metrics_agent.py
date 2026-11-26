import numpy as np

from iex_forecast.agent import decide
from iex_forecast.backtest import walk_forward
from iex_forecast.config import INFO_LAG_DAYS
from iex_forecast.data import make_synthetic
from iex_forecast.features import FEATURE_SETS, build_all, column_index
from iex_forecast.metrics import coverage, diebold_mariano, pinball
from iex_forecast.models import QuantileGBM


def test_pinball_zero_for_perfect_forecast():
    y = np.array([[1.0, 2.0]])
    assert pinball(y, np.repeat(y[..., None], 3, axis=-1)) == 0.0


def test_coverage():
    y = np.array([1.0, 5.0, 9.0])
    assert coverage(y, np.full(3, 2.0), np.full(3, 8.0)) == 1 / 3


def test_dm_detects_clearly_better_model():
    rng = np.random.default_rng(0)
    y = rng.normal(5, 1, (200, 96))
    good, bad = y + rng.normal(0, 0.1, y.shape), y + rng.normal(0, 1.0, y.shape)
    stat, p = diebold_mariano(y, good, bad)
    assert stat < 0 and p < 0.01


def test_wider_band_makes_agent_more_cautious():
    bench = np.full((1, 96), 5.0)
    tight = np.stack([np.full((1, 96), v) for v in (4.0, 4.5, 4.8)], axis=-1)
    wide = np.stack([np.full((1, 96), v) for v in (2.0, 4.5, 8.0)], axis=-1)
    assert decide(tight, bench).action.all()
    assert not decide(wide, bench).action.any()


def test_spike_guard_defers():
    bench = np.full((1, 96), 9.0)
    pq = np.stack([np.full((1, 96), v) for v in (1.0, 2.0, 9.9)], axis=-1)
    sig = decide(pq, bench)
    assert sig.spike_risk.all() and not sig.action.any()


def test_walk_forward_never_trains_on_unknown_prices():
    data = make_synthetic(n_days=110, seed=2)
    X = build_all(data)
    cols = column_index(FEATURE_SETS["price_only"])
    out = walk_forward(
        X, data.price, cols, test_start=70, refit_every=20, cal_days=15,
        make_model=lambda: QuantileGBM(n_iter=5),
    )
    for f in out["folds"]:
        assert f["last_train_day"] <= f["first_test_day"] - INFO_LAG_DAYS
    assert list(out["days"]) == list(range(70, 110))
    # the conformal step may widen or narrow the band, never move the median, and the band keeps the median inside
    for key in ("cal", "cal_fixed"):
        assert np.allclose(out[key][..., 1], out["raw"][..., 1])
        assert (out[key][..., 0] <= out[key][..., 1] + 1e-12).all()
        assert (out[key][..., 2] >= out[key][..., 1] - 1e-12).all()


def test_conformal_restores_nominal_coverage():
    from iex_forecast.backtest import conformal_width

    rng = np.random.default_rng(0)
    n = 4000
    y = rng.normal(0, 1, (n, 1))
    # an overconfident band: +-0.3 around 0 covers far less than 80%
    pq = np.stack([np.full((n, 1), -0.3), np.zeros((n, 1)), np.full((n, 1), 0.3)], axis=-1)
    m = conformal_width(y[:2000], pq[:2000])
    assert coverage(y[2000:], pq[2000:, :, 0] - m, pq[2000:, :, 2] + m) == pytest_approx(0.80, 0.03)


def pytest_approx(v, tol):
    import pytest
    return pytest.approx(v, abs=tol)


def test_tune_agent_returns_cheapest_pair_from_the_grid():
    from iex_forecast.agent import GUARD_GRID, LAMBDA_GRID, tune_agent
    rng = np.random.default_rng(0)
    actual = rng.uniform(2, 8, (30, 96))
    bench = np.full_like(actual, 5.0)
    pq = np.stack([actual - 0.5, actual, actual + 0.5], axis=-1)      # a very good forecast
    lam, guard, table = tune_agent(actual, bench, pq)
    assert lam in LAMBDA_GRID and guard in GUARD_GRID
    costs = [r["avg cost (Rs/kWh)"] for r in table]
    assert costs == sorted(costs) and len(table) == len(LAMBDA_GRID) * len(GUARD_GRID)


def test_paired_loss_test_detects_a_consistent_gap():
    from iex_forecast.metrics import paired_loss_test
    rng = np.random.default_rng(0)
    base = rng.normal(5, 0.3, 200)
    stat, p = paired_loss_test(base - 0.2 + rng.normal(0, 0.05, 200), base)
    assert stat < -5 and p < 0.001
    stat, p = paired_loss_test(base, base + rng.normal(0, 0.05, 200))
    assert p > 0.01


def test_block_bootstrap_ci_brackets_the_mean_and_widens_with_noise():
    from iex_forecast.metrics import block_bootstrap_ci
    rng = np.random.default_rng(0)
    calm = -0.05 + rng.normal(0, 0.02, 365)
    noisy = -0.05 + rng.normal(0, 0.5, 365)
    lo, hi = block_bootstrap_ci(calm)
    assert lo < calm.mean() < hi and hi < 0
    lo2, hi2 = block_bootstrap_ci(noisy)
    assert (hi2 - lo2) > 5 * (hi - lo)


def test_rolling_calibration_uses_only_days_known_at_bid_time():
    d = make_synthetic(n_days=240, seed=11)
    X = build_all(d)
    idx = column_index(FEATURE_SETS["price_only"])
    base = walk_forward(X, d.price, idx, test_start=180, refit_every=60)
    assert base["cal"].shape == base["cal_fixed"].shape == base["raw"].shape
    # destroy every price from test day 200 on: predictions for days before 200 must not change
    y2 = d.price.copy()
    y2[200:] = 99.0
    out2 = walk_forward(X, y2, idx, test_start=180, refit_every=60)
    n_same = 200 - 180                                # test days 180..199 only see prices before day 200
    assert np.allclose(base["cal"][:n_same], out2["cal"][:n_same])
    assert np.allclose(base["raw"][:n_same], out2["raw"][:n_same])
