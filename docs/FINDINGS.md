# Findings on real IEX day-ahead data

Data: IEX DAM market snapshot exports (15-minute blocks), 2023-05-01 to 2026-11-24, 1304 complete days,
exported manually from the exchange website (May 2023 onward, i.e. after the April 2023 price-cap change).
Two chronological evaluation folds, each 365 days x 96 blocks, models refit every 45 days on data known at that time:

| Fold | Test period | History before the test |
|---|---|---|
| 1 | 2024-11-25 to 2025-11-24 | 574 days (from 2023-05-01) |
| 2 | 2025-11-25 to 2026-11-24 | 939 days |

Temperature forecasts come from the Open-Meteo Previous Model Runs API (`temperature_2m_previous_day2`, mean
of five cities); they exist from about 20 January 2024, earlier days are left empty. Volume features are the
previous day's purchase bid, sell bid and market clearing volume from the same exports.

## Forecast accuracy (MAE in Rs/kWh, lower is better)

| Model | Fold 1 | Fold 2 |
|---|---|---|
| Naive (last known day) | 0.657 | 0.633 |
| Seasonal naive (same block last week) | 0.898 | 1.237 |
| GBM quantile, price only | 0.528 | 0.533 |
| GBM + temperature forecast | 0.527 | 0.557 |
| GBM + temperature anomaly | 0.520 | 0.546 |
| GBM + previous-day auction volumes | 0.530 | 0.527 |

Differences in daily MAE (A minus B, negative = A better) with 95% moving-block bootstrap intervals (7-day blocks):

| Comparison | Fold 1 | Fold 2 |
|---|---|---|
| Price-only vs naive | -0.129 [-0.164, -0.094] | -0.100 [-0.133, -0.062] |
| Price-only vs seasonal naive | -0.369 [-0.455, -0.293] | -0.704 [-0.866, -0.551] |
| Volumes vs price-only | +0.001 [-0.011, 0.014] (p 0.83) | -0.006 [-0.019, 0.007] (p 0.37) |
| Temperature vs price-only | -0.001 [-0.009, 0.006] (p 0.71) | +0.023 [0.007, 0.044] (p 0.014) |
| Temperature anomaly vs price-only | -0.009 [-0.017, -0.001] (p 0.040) | +0.013 [0.000, 0.027] (p 0.061) |

Calibration (share of actuals inside the P10-P90 band, target 0.80), price-only model:

| | Fold 1 | Fold 2 |
|---|---|---|
| Raw model band | 0.711 | 0.769 |
| One conformal margin per refit | 0.783 | 0.810 |
| Rolling margin (default) | 0.798 | 0.805 |

## Procurement simulation (stylised, not a trading P&L)

Benchmark = trailing 7-day mean of each block. "Buy" pays the realised day-ahead price, "defer" pays the
benchmark. Saving is relative to always paying the benchmark.

| Strategy | Fold 1 saving | Fold 2 saving |
|---|---|---|
| Always buy on the exchange | 0.04% | -0.19% |
| Naive rule (buy if last known price < benchmark) | 3.66% | 5.15% |
| Agent, default parameters (lambda 0.5, spike guard 0.95) | 3.1-3.8% | 4.6-4.9% |
| **Agent, parameters tuned on a validation window** | **5.75%** | **7.18%** |
| Oracle (hindsight upper bound) | 9.47% | 9.53% |

Protocol: the 240 days before each test period form a validation window. The model rung and the two agent
parameters (risk weight lambda and spike guard) were chosen there, then applied unchanged to the test year.
In both folds the chosen setting was lambda = 0 and no spike guard, i.e. "buy if the P50 forecast is below the
benchmark". Tuned agent vs the naive rule, paired on daily procurement cost: fold 1 -0.083 Rs/kWh
[-0.101, -0.064]; fold 2 -0.099 Rs/kWh [-0.127, -0.072]. The agent captured 61% and 75% of the oracle's saving.

## Foundation-model comparison (zero-shot Chronos-2)

Amazon's Chronos-2 (120M parameters, Apache-2.0) forecast each delivery day from the previous 14 days of prices only,
with no fine-tuning, on exactly the same test days as the GBM. Its history slice is cut at D-1 by the harness.

| fold (test period) | GBM MAE | Chronos-2 MAE | seasonal naive MAE | MAE difference, Chronos minus GBM (95% CI) | Cov80 GBM / Chronos |
|---|---|---|---|---|---|
| 2024-11-25 to 2025-11-24 | 0.530 | 0.605 | 0.898 | +0.075 [0.046, 0.108] | 0.796 / 0.695 |
| 2025-11-25 to 2026-11-24 | 0.546 | 0.639 | 1.237 | +0.093 [0.045, 0.148] | 0.816 / 0.717 |

- The trained GBM beats the zero-shot foundation model in both folds (Diebold-Mariano p < 0.001); Chronos-2 is
  still far better than seasonal naive.
- Chronos-2 uses a short price-only context and no market-specific training, while the GBM is refit on up to
  about 2.5 years of IEX data. The comparison shows that a general-purpose model does not replace a trained one here;
  it does not show that foundation models are weak in general (fine-tuning and covariates were not tried).
- Pinball and Cov80 are not like-for-like: the GBM band is conformally calibrated, the Chronos band is not.
- No ensemble was built: the foundation model is clearly behind, so there is no validation-based case for blending.

## What the results support (held in both folds)

1. A gradient-boosted quantile model forecasts next-day block prices better than "yesterday's prices"
   (10-20% lower MAE) and far better than "last week's prices", on a leakage-controlled walk-forward test.
2. After conformal calibration the P10-P90 band covers about 80% of outcomes. The raw model band covers only
   71-77%, so calibration is necessary.
3. In the stylised simulation, a forecast-based buy rule tuned on a validation window beats the
   "yesterday's price" rule by about 2 percentage points of saving in both folds.

## What the results do not support

- **Auction-volume features.** With 694 days of history the volume features lowered MAE by 0.020-0.026
  (three runs on one test year). With history extended back to May 2023 that gain disappeared in both
  folds (+0.001 and -0.006, intervals include zero). The earlier result did not replicate and is not claimed.
- **Temperature forecasts.** The five-city mean has no consistent effect: raw temperature is neutral in
  fold 1 and worse in fold 2; the anomaly version is slightly better in fold 1 (p 0.04) and slightly worse in
  fold 2 (p 0.06). The sign changes between folds. This applies to this specific feature and data length; it is
  not a statement that weather does not matter for Indian power prices.
- **Rolling (adaptive) calibration.** It moves coverage closer to 0.80 in fold 1 (0.783 to 0.798) and makes no
  difference in fold 2. The gain is small and is not claimed as established.
- **Untuned agent.** With default parameters the agent was no better than the naive rule. The tuned result
  depends on parameter selection on the validation window.
- **Model selection on the validation window.** The chosen rung differed between folds (temperature anomaly in
  fold 2, volumes in fold 1) and the rungs differ by little, so the choice is close to arbitrary.
- No claim about trading profit. The simulation has no volumes, limits or penalties, and "defer" costs the
  benchmark with no penalty.

## Caveats

- Run-to-run variation: the same price-only model produced MAE 0.549 and 0.553 on identical data in
  different runs, although seeds are fixed. The cause is not established. Differences below about 0.01 MAE are
  treated as noise.
- Several variants were compared per fold (temperature, anomaly, volumes, calibration). p-values are not
  corrected for this; the replication across two folds is the main protection.
- The folds are chronological, not independent: fold 2's history contains fold 1's test year.
- 16-19% of blocks sit at the Rs 10/kWh cap. Cap-hit days are the majority in both test periods.
- The data-quality check flags 8 days in 2023 (1-4 Sep, 10-13 Oct) with a single price. They were inspected:
  every block clears at the Rs 10/kWh cap, purchase bids (about 20-25 GW) far exceed sell bids (about 3-5 GW) and
  the cleared volume equals the sell bid. They are genuine supply-shortage days and are kept in the data.
- Solar and demand day-ahead forecasts are not in the data, so those ablation rungs are skipped.
- Market rules may have changed within the period (market coupling, bidding rules); this has not been examined.
- Data provenance is declared, not verified by code. The raw exports are not redistributed in this repository.
