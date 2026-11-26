"""Forecasters: seasonal-naive baselines and a quantile gradient-boosting model."""
from __future__ import annotations

import os

import numpy as np

from .config import QUANTILES

# Backend: LightGBM if installed AND requested (IEX_BACKEND=lightgbm), else scikit-learn.
# NOTE: the LightGBM branch was written but could not be executed in the build
# sandbox (package not installable there). The scikit-learn branch is the tested one.
_BACKEND = os.environ.get("IEX_BACKEND", "sklearn").lower()


class QuantileGBM:
    """One gradient-boosted model per quantile, with monotone (non-crossing) output."""

    def __init__(self, quantiles=QUANTILES, n_iter=150, lr=0.06, seed=0):
        self.quantiles = tuple(quantiles)
        self.n_iter, self.lr, self.seed = n_iter, lr, seed
        self.models = []
        self.backend = _BACKEND

    def _make(self, q):
        if self.backend == "lightgbm":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                objective="quantile", alpha=q, n_estimators=self.n_iter,
                learning_rate=self.lr, num_leaves=31, min_child_samples=100,
                random_state=self.seed, verbose=-1,
            )
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=self.n_iter, learning_rate=self.lr,
            max_leaf_nodes=31, min_samples_leaf=100, early_stopping=False,
            random_state=self.seed,
        )

    def fit(self, X, y):
        self.models = [self._make(q).fit(X, y) for q in self.quantiles]
        return self

    def predict(self, X):
        out = np.column_stack([m.predict(X) for m in self.models])
        return np.sort(out, axis=1)  # fix quantile crossing


def seasonal_naive(data, days, lag_days: int):
    """Forecast day d = realised prices of day d-lag_days. Needs lag_days >= INFO_LAG_DAYS."""
    from .config import INFO_LAG_DAYS

    if lag_days < INFO_LAG_DAYS:
        raise ValueError(
            f"lag {lag_days} would use prices not yet known at bid time (need >= {INFO_LAG_DAYS})"
        )
    return np.stack([data.price[d - lag_days] for d in days])
