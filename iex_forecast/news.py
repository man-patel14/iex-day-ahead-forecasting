"""News -> numeric features (simple lexicon approach, deliberately basic).

Leak guard: news published on day d-1 (before bid time) may be used for delivery day d.
`news_for_delivery_days` shifts by one day so you cannot accidentally use same-day news.
Start without news; add it only if the ablation shows it helps.
"""
from __future__ import annotations

import pandas as pd

POSITIVE = ["surplus", "falls", "fall in", "decline", "eases", "cheaper", "rain", "relief",
            "record solar", "record renewable", "lower demand"]
NEGATIVE = ["shortage", "surge", "spike", "heatwave", "heat wave", "outage", "deficit",
            "record demand", "coal shortage", "breakdown", "tripping", "price cap"]


def headline_score(text: str) -> float:
    """-1 (price-bullish: shortage/heat) ... +1 (price-bearish: surplus/rain)."""
    t = text.lower()
    pos = sum(k in t for k in POSITIVE)
    neg = sum(k in t for k in NEGATIVE)
    return (pos - neg) / max(1, pos + neg)


def news_for_delivery_days(headlines: pd.DataFrame, delivery_dates) -> pd.DataFrame:
    """headlines: columns [date, headline]. Returns one row per delivery date.

    Features come from headlines published on the PREVIOUS calendar day.
    """
    h = headlines.copy()
    h["date"] = pd.to_datetime(h["date"]).dt.normalize()
    h["score"] = h["headline"].map(headline_score)
    h["shortage"] = h["headline"].str.lower().str.contains("shortage|outage|deficit")
    daily = h.groupby("date").agg(
        news_count=("headline", "count"), news_score=("score", "mean"),
        news_shortage=("shortage", "max"),
    )
    daily.index = daily.index + pd.Timedelta(days=1)       # usable from the NEXT day
    out = daily.reindex(pd.DatetimeIndex(delivery_dates).normalize())
    out["news_count"] = out["news_count"].fillna(0)
    out["news_score"] = out["news_score"].fillna(0.0)
    out["news_shortage"] = out["news_shortage"].map(lambda v: 0.0 if pd.isna(v) else float(bool(v)))
    return out
