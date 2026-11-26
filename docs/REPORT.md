# Day-ahead price forecasting and procurement signals on the Indian Energy Exchange

## Summary

This project forecasts the 96 fifteen-minute clearing prices of the IEX day-ahead market (DAM) using only information
available at bid time, attaches calibrated P10-P90 bands, and feeds the forecast into a rule-based procurement agent.
It is evaluated on real exchange data (1,304 delivery days, May 2023 to November 2026) in two consecutive, walk-forward
test years. Every number below comes from `results/RESULTS.md`, `results_fold1/RESULTS.md` and the foundation-model runs, and is
discussed with its limits in `docs/FINDINGS.md`.

## Problem

Bids for day D must be submitted before day D prices exist, and the latest realised prices are those of day D-1.
The questions are: how accurately can the 96 block prices be forecast, is the stated uncertainty calibrated, and does
acting on the forecast lower procurement cost in a simplified simulation?

## Data

- IEX DAM market snapshots (15-minute resolution) exported from the exchange website: MCP, purchase bid, sell bid and
  cleared volume. Raw files are not redistributed.
- Bid-time temperature forecasts from the Open-Meteo Previous Model Runs API (five-city mean; available from January 2024).
- Price cap regime: the series starts in May 2023, after the April 2023 cap revision (Rs 10/kWh).
- Eight days in 2023 clear at the cap in every block with purchase bids far above sell bids. They were inspected and
  kept as genuine supply-shortage days.

## Method

- **Leakage control.** Features may read only prices up to D-1 and day-ahead forecasts. Unit tests corrupt all later
  data and assert that features and predictions do not change.
- **Model.** Quantile gradient boosting (P10/P50/P90) refit every 45 days on data known at that time.
- **Uncertainty.** Conformalised quantile regression with a rolling 60-day margin using only days known at bid time.
- **Ablation ladder.** Price-only, plus weather, plus weather anomaly, plus auction volumes. The rung is chosen on a
  240-day validation window before each test year and then applied unchanged.
- **Agent.** Buy a block on the DAM if the forecast P50 is below a benchmark (trailing 7-day block mean), otherwise
  defer. Parameters are tuned on the validation window only.
- **Statistics.** Diebold-Mariano tests with HAC variance and moving-block bootstrap intervals on daily errors.

## Results

| | Fold 1 (Nov 2024 - Nov 2025) | Fold 2 (Nov 2025 - Nov 2026) |
|---|---|---|
| Naive (last known day), MAE Rs/kWh | 0.657 | 0.633 |
| Seasonal naive, MAE | 0.898 | 1.237 |
| Chosen GBM, MAE | 0.530 | 0.546 |
| Zero-shot Chronos-2, MAE | 0.605 | 0.639 |
| GBM P10-P90 coverage (target 0.80) | 0.796 | 0.816 |
| Tuned agent saving vs benchmark | 5.7% | 7.2% |
| Naive buy rule saving vs benchmark | 3.7% | 5.2% |

- The GBM beats both naive baselines in both folds with confidence intervals that exclude zero.
- Calibrated coverage is close to 80% in both folds; uncalibrated coverage was 70-79%.
- The validation-tuned agent is cheaper than the "yesterday's price" rule in both folds (CIs exclude zero).
- The zero-shot Chronos-2 foundation model, using a 14-day price-only context, is worse than the trained GBM in both folds
  (Diebold-Mariano p < 0.001). Its band is uncalibrated, so only the MAE comparison is like-for-like.

## What did not hold up

- Auction-volume features looked helpful in one fold and did not replicate in the other.
- Raw temperature did not help consistently; the anomaly form helped marginally in one fold.
- Rolling calibration changed coverage only slightly compared with one margin per refit.
- Differences below about 0.01 Rs/kWh MAE are within run-to-run noise.

## Limits

- Two test years of one market; results may not transfer to other regimes.
- The procurement simulation is a stylised toy (no volumes, limits or penalties), not a trading P&L, and not advice.
- Solar and demand day-ahead forecasts are not in the data, so those ablation rungs were skipped.
- The foundation-model test covers one model, zero-shot; fine-tuning and covariates were not tried.
- The dashboard replays the backtest and is not a live forecast.

## Reproduce

See `README.md`: build the dataset from IEX exports, add weather, run `run_backtest.py` for each fold, then
`scripts/run_foundation.py`. The test suite (44 tests) runs with `python -m pytest -q`.
