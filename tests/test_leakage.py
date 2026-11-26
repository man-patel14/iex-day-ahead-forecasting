import copy

import numpy as np
import pytest

from iex_forecast.config import INFO_LAG_DAYS
from iex_forecast.data import make_synthetic
from iex_forecast.features import ALL_NAMES, build_all, day_features
from iex_forecast.models import seasonal_naive


@pytest.fixture(scope="module")
def data():
    return make_synthetic(n_days=120, seed=1)


def corrupt_unknown_prices(data, d, value=None):
    """Return a copy where every price NOT known at bid time for delivery day d is destroyed."""
    dirty = copy.deepcopy(data)
    first_unknown = d - INFO_LAG_DAYS + 1
    shape = dirty.price[first_unknown:].shape
    dirty.price[first_unknown:] = value if value is not None else np.random.default_rng(0).uniform(0, 10, shape)
    return dirty


def test_features_ignore_prices_unknown_at_bid_time(data):
    """Corrupt every price after the last known day; features for day d must not move."""
    d = 60
    clean = day_features(data, d)
    assert np.allclose(clean, day_features(corrupt_unknown_prices(data, d), d), equal_nan=True)


def test_features_do_use_known_history(data):
    """Sanity check the test above is not vacuous: changing the last known day MUST change features."""
    d = 60
    dirty = copy.deepcopy(data)
    dirty.price[d - INFO_LAG_DAYS] += 1.0
    assert not np.allclose(day_features(data, d), day_features(dirty, d), equal_nan=True)


def test_every_day_is_leak_free(data):
    X = build_all(data)
    for d in (20, 45, 90, 119):
        assert np.allclose(X[d], day_features(corrupt_unknown_prices(data, d, -999), d), equal_nan=True)


def test_baseline_that_uses_an_unknown_day_is_rejected(data):
    with pytest.raises(ValueError):
        seasonal_naive(data, [30], lag_days=INFO_LAG_DAYS - 1)


def test_feature_width(data):
    assert day_features(data, 30).shape == (96, len(ALL_NAMES))
