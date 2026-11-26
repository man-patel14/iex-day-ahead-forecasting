"""Walk-forward (rolling-origin) backtest that respects the bid-time information set."""
from __future__ import annotations

import numpy as np

from .config import BLOCKS_PER_DAY, INFO_LAG_DAYS, MIN_HISTORY_DAYS, PRICE_CAP, QUANTILES
from .models import QuantileGBM


def conformal_width(y, pq, alpha=0.2):
    """Split-conformal margin for the P10-P90 band (CQR, Romano et al. 2019).

    Score = how far the actual falls outside the band (negative if inside).
    Returns the finite-sample-adjusted (1-alpha) quantile of the scores.
    """
    scores = np.maximum(pq[..., 0] - y, y - pq[..., -1]).ravel()
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level))


def walk_forward(X_all, y_all, col_idx, test_start, refit_every=45, cal_days=60, make_model=None,
                 rolling_cal=True):
    """Refit every `refit_every` test days; predict each test day from data known at bid time.

    For the first day T of a fold, bidding happens on day T-1, so the latest realised
    price known is day T - INFO_LAG_DAYS. Training uses target days up to that day, of which
    the most recent `cal_days` are held out to calibrate the P10-P90 band (conformal).

    Two calibrations are returned:
      cal_fixed  one conformal margin per fold, from that fold's `cal_days` calibration days
      cal        (default) a ROLLING margin: for each test day D it uses the model's errors on
                 the latest `cal_days` days that are known at bid time (days <= D - INFO_LAG_DAYS),
                 which adapts to drift between refits. With rolling_cal=False it equals cal_fixed.

    Returns dict with
      raw  (n_test, 96, Q)  quantiles straight from the model
      cal, cal_fixed (n_test, 96, Q)  P10/P90 widened by a conformal margin (P50 unchanged)
      days, folds (log of the last day used for training/calibration per fold)
    """
    make_model = make_model or (lambda: QuantileGBM())
    D = X_all.shape[0]
    raw, cal, cal_fixed, days, folds = [], [], [], [], []
    for T in range(test_start, D, refit_every):
        last_known = T - INFO_LAG_DAYS
        all_days = np.arange(MIN_HISTORY_DAYS, last_known + 1)
        fit_days = all_days[:-cal_days] if cal_days else all_days
        cal_set = all_days[-cal_days:] if cal_days else None

        flat = lambda ds: X_all[ds][:, :, col_idx].reshape(-1, len(col_idx))
        model = make_model().fit(flat(fit_days), y_all[fit_days].reshape(-1))

        test_days = np.arange(T, min(T + refit_every, D))
        p = model.predict(flat(test_days)).reshape(len(test_days), BLOCKS_PER_DAY, len(QUANTILES))
        raw.append(p)

        q = p.copy()
        roll = p.copy()
        margin = 0.0
        if cal_set is not None:
            pc = model.predict(flat(cal_set)).reshape(len(cal_set), BLOCKS_PER_DAY, len(QUANTILES))
            margin = conformal_width(y_all[cal_set], pc)
            q[..., 0] = np.minimum(np.clip(q[..., 0] - margin, 0.0, None), p[..., 1])     # band keeps the median inside
            q[..., -1] = np.maximum(np.clip(q[..., -1] + margin, None, PRICE_CAP), p[..., 1])
            if rolling_cal:
                seq_days = np.concatenate([cal_set, test_days])
                seq_pred = np.concatenate([pc, p])
                for k, day in enumerate(test_days):
                    known = (seq_days <= day - INFO_LAG_DAYS) & (seq_days > day - INFO_LAG_DAYS - cal_days)
                    m = conformal_width(y_all[seq_days[known]], seq_pred[known])
                    roll[k, :, 0] = np.minimum(np.clip(p[k, :, 0] - m, 0.0, None), p[k, :, 1])
                    roll[k, :, -1] = np.maximum(np.clip(p[k, :, -1] + m, None, PRICE_CAP), p[k, :, 1])
        cal_fixed.append(q)
        cal.append(roll if rolling_cal else q)

        days.extend(test_days.tolist())
        folds.append({
            "first_test_day": int(T),
            "last_train_day": int(all_days[-1]),   # last day whose price was used (fit or calibration)
            "conformal_margin": margin,
        })
    return {"raw": np.concatenate(raw), "cal": np.concatenate(cal), "cal_fixed": np.concatenate(cal_fixed),
            "days": np.array(days), "folds": folds}
