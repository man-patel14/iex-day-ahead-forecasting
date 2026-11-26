"""Spike-probability classifier: P(price in this block >= 95% of the cap).

Same walk-forward / information-lag rules as the price model. Useful because quantile
regression is weak at rare extreme events, and a trader cares most about those.
"""
from __future__ import annotations

import numpy as np

from .config import BLOCKS_PER_DAY, INFO_LAG_DAYS, MIN_HISTORY_DAYS, PRICE_CAP

SPIKE_LEVEL = 0.95 * PRICE_CAP


def label_spikes(y):
    return y >= SPIKE_LEVEL


def walk_forward_spike(X_all, y_all, col_idx, test_start, refit_every=45, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    D = X_all.shape[0]
    probs = []
    for T in range(test_start, D, refit_every):
        tr = np.arange(MIN_HISTORY_DAYS, T - INFO_LAG_DAYS + 1)
        Xtr = X_all[tr][:, :, col_idx].reshape(-1, len(col_idx))
        ytr = label_spikes(y_all[tr]).reshape(-1)
        te = np.arange(T, min(T + refit_every, D))
        Xte = X_all[te][:, :, col_idx].reshape(-1, len(col_idx))
        if ytr.sum() < 20:                       # too few spikes to learn from
            p = np.zeros(len(Xte))
        else:
            clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08,
                                                 min_samples_leaf=50, random_state=seed).fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
        probs.append(p.reshape(len(te), BLOCKS_PER_DAY))
    return np.concatenate(probs)
