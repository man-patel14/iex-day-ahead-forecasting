"""FastAPI service over saved backtest predictions.

Run:  pip install fastapi uvicorn && uvicorn app.api:app --reload
NOTE: written but NOT executed in the build sandbox (fastapi not installed there).
It serves stored predictions; wiring it to a live daily forecast is Step 10 of the guide.
"""
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

app = FastAPI(title="IEX day-ahead signal API (decision support, not trading advice)")
_PRED = Path(__file__).resolve().parents[1] / "results" / "predictions.npz"


def _load():
    if not _PRED.exists():
        raise HTTPException(503, "run `python run_backtest.py` first")
    return np.load(_PRED)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast/{date}")
def forecast(date: str):
    z = _load()
    dates = list(z["dates"])
    if date not in dates:
        raise HTTPException(404, f"date not in results; available {dates[0]}..{dates[-1]}")
    i = dates.index(date)
    q = z["quantiles"][i]
    return {
        "date": date,
        "p10": q[:, 0].round(3).tolist(),
        "p50": q[:, 1].round(3).tolist(),
        "p90": q[:, 2].round(3).tolist(),
        "buy_on_dam": z["buy"][i].tolist(),
        "confidence": z["confidence"][i].round(3).tolist(),
        "disclaimer": "decision support on stored backtest data; not trading advice",
    }
