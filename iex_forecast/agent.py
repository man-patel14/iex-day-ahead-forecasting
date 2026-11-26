"""Rule-based trader agent + stylised procurement simulation.

Decision support, NOT trading advice. Decisions are deterministic and auditable;
the text explanation is a template (an LLM can be plugged into `explain_day`, but
it must never change the decision).

Stylised economics (an assumption, state it whenever you quote a number):
  A distribution company must procure each block. For every block it either
    BUYS on the day-ahead market  -> pays the realised DAM price, or
    DEFERS to its alternative source -> pays the BENCHMARK price.
  Benchmark = trailing 7-day mean of that block, known at bid time (a stand-in for
  a bilateral / PPA price). Real procurement has volumes, limits and penalties
  that this ignores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PRICE_CAP

RISK_LAMBDA = 0.5      # fixed a priori. NOT tuned on the test period.
SPIKE_GUARD = 0.95     # defer when P90 is within 5% of the price cap


@dataclass
class AgentSignals:
    action: np.ndarray       # (n, 96) bool: True = BUY on DAM, False = DEFER
    confidence: np.ndarray   # (n, 96) 0..1, higher = tighter P10-P90 band
    spike_risk: np.ndarray   # (n, 96) bool


def decide(pq, benchmark, lam: float = RISK_LAMBDA, spike_guard: float = SPIKE_GUARD) -> AgentSignals:
    """Buy only if the RISK-ADJUSTED expected price is below the benchmark.

    risk-adjusted price = P50 + lam * (P90 - P50): leans on the upper band, so a
    wide (uncertain) forecast makes the agent more reluctant to commit.
    """
    p10, p50, p90 = pq[..., 0], pq[..., 1], pq[..., 2]
    spike = p90 >= spike_guard * PRICE_CAP
    buy = (p50 + lam * (p90 - p50) < benchmark) & ~spike
    width = (p90 - p10) / np.maximum(p50, 1e-6)
    conf = np.clip(1 - width / 1.5, 0, 1)
    return AgentSignals(buy, conf, spike)


def procurement_cost(actual, benchmark, buy) -> float:
    return float(np.where(buy, actual, benchmark).mean())


LAMBDA_GRID = (-0.5, -0.25, 0.0, 0.25, 0.5)
GUARD_GRID = (0.90, 0.95, 0.99, 1.01)      # 1.01 = spike guard off


def tune_agent(actual, benchmark, pq, lambdas=LAMBDA_GRID, guards=GUARD_GRID):
    """Grid-search the two agent parameters on a VALIDATION window (never the test period).

    Returns (best_lambda, best_guard, table) where table lists every combination with its
    average procurement cost, sorted from cheapest.
    """
    rows = []
    for lam in lambdas:
        for g in guards:
            cost = procurement_cost(actual, benchmark, decide(pq, benchmark, lam, g).action)
            rows.append({"lambda": lam, "spike guard": g, "avg cost (Rs/kWh)": cost})
    rows.sort(key=lambda r: r["avg cost (Rs/kWh)"])
    return rows[0]["lambda"], rows[0]["spike guard"], rows


def simulate(actual, benchmark, last_known, pq, lam: float = RISK_LAMBDA, spike_guard: float = SPIKE_GUARD):
    """Compare strategies. All costs in Rs/kWh averaged over every block."""
    sig = decide(pq, benchmark, lam, spike_guard)
    strategies = {
        "always buy DAM": np.ones_like(actual, dtype=bool),
        "never buy (benchmark only)": np.zeros_like(actual, dtype=bool),
        "naive rule (buy if last known price < benchmark)": last_known < benchmark,
        "agent (quantile, risk-adjusted)": sig.action,
        "oracle (hindsight, upper bound)": actual < benchmark,
    }
    costs = {k: procurement_cost(actual, benchmark, v) for k, v in strategies.items()}
    bench = costs["never buy (benchmark only)"]
    oracle = costs["oracle (hindsight, upper bound)"]
    rows = []
    for k, v in strategies.items():
        c = costs[k]
        rows.append({
            "strategy": k,
            "avg cost (Rs/kWh)": c,
            "saving vs benchmark (%)": 100 * (bench - c) / bench,
            "share of oracle saving captured (%)": 100 * (bench - c) / (bench - oracle)
            if bench != oracle else float("nan"),
            "blocks bought on DAM (%)": 100 * v.mean(),
        })
    return rows, sig


def explain_day(date, pq_day, temp_fc_day, sig_day_action, conf_day, spike_day) -> str:
    """Template explanation for one delivery day. pq_day: (96, 3)."""
    hours = np.arange(96) / 4
    ev = (hours >= 18) & (hours < 22)
    p10, p50, p90 = pq_day[ev, 0].mean(), pq_day[ev, 1].mean(), pq_day[ev, 2].mean()
    conf = float(conf_day[ev].mean())
    level = "HIGH" if conf > 0.66 else "MEDIUM" if conf > 0.33 else "LOW"
    drivers = []
    if temp_fc_day.max() >= 38:
        drivers.append(f"heat (forecast max {temp_fc_day.max():.0f} C)")
    if date.day_of_week >= 5:
        drivers.append("weekend demand dip")
    if not drivers:
        drivers.append("normal seasonal pattern")
    n_buy, n_spike = int(sig_day_action.sum()), int(spike_day.sum())
    msg = (
        f"{date.date()}: evening peak (18-22h) P50 Rs{p50:.2f}/kWh "
        f"(P10 {p10:.2f} - P90 {p90:.2f}), confidence {level}. "
        f"Drivers: {', '.join(drivers)}. Plan: buy on DAM in {n_buy}/96 blocks."
    )
    if n_spike:
        msg += f" Spike risk in {n_spike} blocks: deferred."
    return msg
