"""End-to-end walk-forward backtest: baselines, ablation, probabilistic metrics, agent simulation.

Usage:
    python run_backtest.py                              built-in synthetic data
    python run_backtest.py --data data/iex.csv --real   your own CSV (canonical schema, see data/README.md)

Writes (to --out, default results/): RESULTS.md, results.json, run_meta.json,
predictions.npz and two figures.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iex_forecast.agent import explain_day, simulate, tune_agent
from iex_forecast.backtest import walk_forward
from iex_forecast.config import INFO_LAG_DAYS
from iex_forecast.data import check_data, load_csv, make_synthetic, truncate
from iex_forecast.features import build_all, column_index, usable_feature_sets
from iex_forecast.metrics import (block_bootstrap_ci, diebold_mariano, paired_loss_test, stratified_mae,
                                  summarise)
from iex_forecast.models import seasonal_naive

MIN_TRAIN_DAYS = 180
MIN_TEST_DAYS = 60
VAL_DAYS = 240          # validation window directly before the test period (model + agent selection)


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = ["n/a" if isinstance(v, (float, np.floating)) and np.isnan(v)
                 else floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="canonical CSV; omit to use the built-in synthetic generator")
    ap.add_argument("--real", action="store_true", help="declare that --data is real market data")
    ap.add_argument("--test-days", type=int, default=365, help="length of the walk-forward test period")
    ap.add_argument("--refit-every", type=int, default=45, help="refit the models every N test days")
    ap.add_argument("--end-date", help="ignore all days after this date (YYYY-MM-DD): evaluates an earlier test period")
    ap.add_argument("--out", default="results")
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data:
        data = load_csv(args.data, source="real" if args.real else "unverified csv")
        label = ("REAL (declared by the user; provenance is not verified by this code)" if args.real
                 else "UNVERIFIED CSV (not declared real; treat as synthetic)")
        short = "real data" if args.real else "unverified data"
    else:
        data = make_synthetic()
        label = "SYNTHETIC (built-in generator)"
        short = "synthetic data"
    if args.end_date:
        data = truncate(data, args.end_date)
    warnings = check_data(data)
    for w in warnings:
        print("WARNING:", w)

    test_start = max(data.n_days - args.test_days, MIN_TRAIN_DAYS)
    if data.n_days < MIN_TRAIN_DAYS + MIN_TEST_DAYS:
        raise SystemExit(f"need at least {MIN_TRAIN_DAYS + MIN_TEST_DAYS} days of data, got {data.n_days}")
    if data.n_days - test_start < MIN_TEST_DAYS:
        raise SystemExit(f"--test-days {args.test_days} is too short: the test period must be at least "
                         f"{MIN_TEST_DAYS} days")
    X = build_all(data)
    y = data.price
    print(f"data: {label}, {data.n_days} days, test from day {test_start} ({data.dates[test_start].date()})")

    # ---- baselines ------------------------------------------------------- #
    test_days = np.arange(test_start, data.n_days)
    actual = y[test_days]
    last_known = seasonal_naive(data, test_days, INFO_LAG_DAYS)
    lag7 = seasonal_naive(data, test_days, 7)
    ref = last_known
    naive_name = "naive (last known day)"
    snaive_name = "seasonal naive (same block last week)"
    results = {naive_name: summarise(actual, last_known, ref), snaive_name: summarise(actual, lag7, ref)}
    point = {naive_name: last_known, snaive_name: lag7}

    # ---- GBM ablation ---------------------------------------------------- #
    sets = usable_feature_sets(data)
    quant, labels = {}, []
    for name, feats in sets.items():
        ts = time.time()
        out = walk_forward(X, y, column_index(feats), test_start, args.refit_every)
        assert (out["days"] == test_days).all()
        pq = out["cal"]
        lab = f"GBM quantile: {name}"
        labels.append(lab)
        results[lab] = summarise(actual, pq[..., 1], ref, pq)
        results[lab]["Cov80 raw"] = summarise(actual, pq[..., 1], ref, out["raw"])["Cov80"]
        results[lab]["Cov80 fixed margin"] = summarise(actual, pq[..., 1], ref, out["cal_fixed"])["Cov80"]
        point[lab], quant[lab] = pq[..., 1], pq
        print(f"  {lab:50s} MAE {results[lab]['MAE']:.3f}  ({time.time()-ts:.0f}s)")

    # ---- validation window (before the test period): model choice + agent tuning -------- #
    val_start = test_start - VAL_DAYS
    val_days = np.arange(val_start, test_start)
    val_ok = val_start >= MIN_TRAIN_DAYS
    val_mae, val_quant = {}, {}
    if val_ok:
        yv = y[:test_start]
        for name, feats in sets.items():
            outv = walk_forward(X[:test_start], yv, column_index(feats), val_start, args.refit_every)
            lab = f"GBM quantile: {name}"
            val_quant[lab] = outv["cal"]
            val_mae[lab] = float(np.abs(outv["cal"][..., 1] - yv[val_days]).mean())
            print(f"  validation {lab:42s} MAE {val_mae[lab]:.3f}")
    else:
        print(f"  validation window skipped: needs {MIN_TRAIN_DAYS + VAL_DAYS} days before the test period")

    res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    # Descriptive only: the rung with the lowest test MAE. Choosing on the test set is
    # optimistic, so every rung is reported in every table below.
    if val_ok:
        best_label = min(labels, key=lambda k: val_mae[k])      # chosen on the validation window
        selected_by = "validation MAE (before the test period)"
    else:
        best_label = min(labels, key=lambda k: results[k]["MAE"])
        selected_by = "test MAE (descriptive only: no validation window available)"
    best_q = quant[best_label]

    # ---- significance ---------------------------------------------------- #
    pairs = [(lab, base) for lab in labels for base in (snaive_name, naive_name)]
    branches = [l for l in labels if "anom" in l or "volumes" in l]
    ladder = [l for l in labels if l not in branches]
    pairs += [(ladder[i], ladder[i - 1]) for i in range(1, len(ladder))]   # each ablation rung vs the previous
    for l in branches:                                                   # side branches vs ladder rungs
        pairs += [(l, x) for x in (ladder[:2] if "anom" in l else ladder[:1])]
    dm_rows = []
    for a, b in pairs:
        stat, p = diebold_mariano(actual, point[a], point[b])
        dd = np.abs(actual - point[a]).mean(axis=1) - np.abs(actual - point[b]).mean(axis=1)
        lo, hi = block_bootstrap_ci(dd)
        dm_rows.append({"model A": a, "vs model B": b, "MAE diff (A-B)": float(dd.mean()),
                        "95% CI": f"[{lo:.3f}, {hi:.3f}]", "DM stat (neg = A better)": stat, "p-value": p})
    dm_df = pd.DataFrame(dm_rows)

    # ---- stratified ------------------------------------------------------ #
    dts = data.dates[test_days]
    strat = stratified_mae(actual, point[best_label], dts).rename(columns={"MAE": "best rung MAE"})
    strat["seasonal naive MAE"] = stratified_mae(actual, lag7, dts)["MAE"].values

    # ---- agent ----------------------------------------------------------- #
    lo = INFO_LAG_DAYS + 6           # trailing 7 known days: d-lag-6 .. d-lag
    benchmark = np.stack([y[d - lo: d - INFO_LAG_DAYS + 1].mean(axis=0) for d in test_days])
    lam_t, guard_t, tune_rows = None, None, []
    if val_ok:
        lov = INFO_LAG_DAYS + 6
        bench_v = np.stack([y[d - lov: d - INFO_LAG_DAYS + 1].mean(axis=0) for d in val_days])
        lam_t, guard_t, tune_rows = tune_agent(y[val_days], bench_v, val_quant[best_label])
    rows, sig = simulate(actual, benchmark, last_known, best_q)
    agent_rows = [r for r in rows if not r["strategy"].startswith("agent")]
    for lab in labels:                                    # one agent row per model rung
        r_lab, _ = simulate(actual, benchmark, last_known, quant[lab])
        row = dict(next(r for r in r_lab if r["strategy"].startswith("agent")))
        row["strategy"] = f"agent ({lab.replace('GBM quantile: ', '')})"
        agent_rows.insert(len(agent_rows) - 1, row)        # keep the oracle last
    if val_ok:                                            # tuned agent: parameters chosen on validation only
        r_t, sig = simulate(actual, benchmark, last_known, best_q, lam_t, guard_t)
        row = dict(next(r for r in r_t if r["strategy"].startswith("agent")))
        row["strategy"] = f"agent TUNED on validation ({best_label.replace('GBM quantile: ', '')}, lambda={lam_t}, guard={guard_t})"
        agent_rows.insert(len(agent_rows) - 1, row)
        daily = lambda buy: np.where(buy, actual, benchmark).mean(axis=1)      # daily cost, Rs/kWh
        stat_a, p_a = paired_loss_test(daily(sig.action), daily(last_known < benchmark))
        dc = daily(sig.action) - daily(last_known < benchmark)
        lo_c, hi_c = block_bootstrap_ci(dc)
        agent_test_md = (f"Tuned agent vs naive rule, daily procurement cost: mean difference "
                         f"{dc.mean():+.3f} Rs/kWh (95% block-bootstrap CI [{lo_c:+.3f}, {hi_c:+.3f}]; "
                         f"negative = agent cheaper), DM statistic {stat_a:.2f}, p-value {p_a:.4f}.")
    else:
        agent_test_md = ""
    rows = agent_rows
    agent_df = pd.DataFrame(rows)

    np.savez(out_dir / "predictions.npz", quantiles=best_q, actual=actual,
             dates=np.array([str(d.date()) for d in dts]), benchmark=benchmark,
             buy=sig.action, confidence=sig.confidence)

    # sample explanations (3 days: hottest, a weekend, a spike day)
    tmax = np.nan_to_num(data.temp_fc[test_days].max(axis=1), nan=-1.0)
    hot = int(np.argmax(tmax))
    wk_idx = np.where(dts.day_of_week >= 5)[0]
    wkend = int(wk_idx[min(10, len(wk_idx) - 1)])
    spk = int(np.argmax(actual.max(axis=1) >= 9.5)) if (actual.max(axis=1) >= 9.5).any() else 0
    explanations = [
        explain_day(dts[i], best_q[i], data.temp_fc[test_days[i]], sig.action[i],
                    sig.confidence[i], sig.spike_risk[i])
        for i in (hot, wkend, spk)
    ]

    # ---- figures --------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(9, 4))
    gb = res_df[res_df["model"].str.startswith("GBM")]
    base = res_df[~res_df["model"].str.startswith("GBM")]
    ax.barh(list(base["model"]) + list(gb["model"]), list(base["MAE"]) + list(gb["MAE"]),
            color=["#9aa0a6"] * len(base) + ["#1f77b4"] * len(gb))
    ax.set_xlabel("MAE (Rs/kWh), lower is better")
    ax.set_title(f"Walk-forward MAE: baselines vs ablation ({short})")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_mae.png", dpi=140)
    plt.close(fig)

    wk = slice(hot, hot + 7)
    n_wk = len(best_q[wk])
    x = np.arange(n_wk * 96)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x, best_q[wk][..., 0].ravel(), best_q[wk][..., 2].ravel(), alpha=0.25, label="P10-P90")
    ax.plot(x, best_q[wk][..., 1].ravel(), lw=1.2, label="P50 forecast")
    ax.plot(x, actual[wk].ravel(), lw=1.0, color="k", label="actual")
    ax.set_xlabel(f"15-min block ({n_wk} days)")
    ax.set_ylabel("Rs/kWh")
    ax.set_title(f"One test week from the hottest forecast day ({short})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "forecast_week.png", dpi=140)
    plt.close(fig)

    # ---- write results --------------------------------------------------- #
    if val_ok:
        vt = pd.DataFrame([{"model": k, "validation MAE": v} for k, v in val_mae.items()])
        top = pd.DataFrame(tune_rows[:5])
        val_md = (md_table(vt) + "\n\nBest 5 agent parameter pairs on the validation window:\n\n" + md_table(top))
    else:
        val_md = "Skipped: not enough history before the test period."
    warn_md = "\n".join(f"- {w}" for w in warnings) or "- none"
    caveat = (
        "> These numbers come from a synthetic generator built to exercise the pipeline. They show that\n"
        "> the harness works; they are **not** findings about the real IEX market."
        if "SYNTHETIC" in label or "UNVERIFIED" in label else
        "> Real-data run. Confirm that the forecast columns were genuinely available at bid time."
    )
    md = f"""# Backtest results

> **Data: {label}.**
{caveat}

Test period: {dts[0].date()} to {dts[-1].date()} ({len(test_days)} days x 96 blocks).
Bid-time information: realised prices up to day D-{INFO_LAG_DAYS}; weather, solar and demand are
day-ahead *forecasts*. Models are refit every {args.refit_every} days on data known at that time.
Boosted-tree backend: scikit-learn HistGradientBoosting (quantile loss), unless `IEX_BACKEND=lightgbm`.
News features: not included (no news data source wired in).

Data-quality warnings:
{warn_md}

## Model comparison and ablation
`DirAcc` = share of blocks where the forecast gets the direction of change vs the last
known day right; n/a for the naive baseline, which IS that reference.
`Cov80` = share of actuals inside the conformally calibrated P10-P90 band (target 0.80);
`Cov80 raw` = same band before calibration; `Cov80 fixed margin` = one conformal margin per refit
(the earlier design). The default band (`Cov80`) uses a rolling margin: for every test day it takes
the model's errors on the latest 60 days known at bid time. Pinball uses the default band.

{md_table(res_df)}

## Significance (Diebold-Mariano on daily MAE, with 95% block-bootstrap intervals of the MAE difference)
{md_table(dm_df, "{:.4f}")}

## Where the chosen rung ({best_label}; selected by {selected_by}) is strong / weak (MAE)
{md_table(strat)}

## Stylised procurement simulation (agent)
One agent row per model rung with the fixed default parameters (lambda=0.5, guard=0.95), plus one row
whose model and parameters were chosen on the validation window ({selected_by}). Sample explanations and
`predictions.npz` use the chosen rung and, when available, the tuned parameters.
A buyer either buys each block on the day-ahead market or defers to a benchmark price
(trailing 7-day mean of that block). Agent parameters were fixed before looking at the
test period. This is a toy economic model, not a trading P&L.

{md_table(agent_df)}

{agent_test_md}

## Validation window (days {val_start}-{test_start - 1}, before the test period)
{val_md}

## Sample agent explanations
""" + "\n".join(f"- {e}" for e in explanations) + f"""

![ablation](ablation_mae.png)
![forecast](forecast_week.png)

_Runtime: {time.time()-t0:.0f}s_
"""
    (out_dir / "RESULTS.md").write_text(md)
    (out_dir / "results.json").write_text(json.dumps(
        {"data_source": label, "models": results, "agent": rows, "dm": dm_rows}, indent=2, default=float))
    (out_dir / "run_meta.json").write_text(json.dumps({
        "data_source": label, "data_path": args.data, "n_days": data.n_days,
        "test_start": str(dts[0].date()), "test_end": str(dts[-1].date()),
        "info_lag_days": INFO_LAG_DAYS, "warnings": warnings,
    }, indent=2))
    print(md)


if __name__ == "__main__":
    main()
