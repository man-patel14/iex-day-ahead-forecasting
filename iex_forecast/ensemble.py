"""Blend several point forecasts. Weights must come from a VALIDATION period,
never from the test period you report results on."""
from __future__ import annotations

import numpy as np


def inverse_error_weights(val_actual, val_preds: dict) -> dict:
    """Weight each model by 1/MAE on the validation window (weights sum to 1)."""
    inv = {k: 1.0 / np.mean(np.abs(val_actual - p)) for k, p in val_preds.items()}
    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()}


def blend(preds: dict, weights: dict):
    return sum(weights[k] * preds[k] for k in preds)
