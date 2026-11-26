# Roadmap

Each step lists its goal, the relevant files, and an acceptance criterion.
Steps 0-2 and 4-6 are done in a first version (see `docs/FINDINGS.md`); step 3 (leakage tests on real data), 7 (regimes), 8-12 and the agent validation tuning are still to do.

## Phase 1: Foundations
| Step | Goal | Files | Done when |
|---|---|---|---|
| 0 | Environment works | `requirements.txt` | `python -m pytest -q` passes; `python run_backtest.py` writes `results/RESULTS.md` |
| 1 | Market rules recorded | `docs/market_notes.md`, `iex_forecast/config.py` | Open questions in the notes are closed with primary sources |
| 2 | Real data loaded | `iex_forecast/data.py`, `docs/DATA_SOURCES.md` | A real CSV runs through `run_backtest.py --real` with no data-quality warnings left unexplained |
| 3 | Leakage tests on real data | `tests/test_leakage.py` | Tests pass; every forecast column is confirmed to be a bid-time forecast |
| 4 | Honest baselines | `iex_forecast/models.py` | Naive and seasonal-naive MAE reported on the real test period |

## Phase 2: Modelling
| Step | Goal | Files | Done when |
|---|---|---|---|
| 5 | Quantile gradient boosting | `iex_forecast/models.py`, `backtest.py` | Walk-forward results beat the baselines; hyperparameters chosen on a validation window, never the test period |
| 6 | Calibration, ablation, significance | `backtest.py`, `metrics.py` | Coverage near 80%; a Diebold-Mariano p-value accompanies every claimed improvement |
| 7 | Regimes and spikes | `iex_forecast/spike.py`, `metrics.py` | Errors reported separately before and after each regime boundary; spike precision/recall reported |

## Phase 3: Advanced
| Step | Goal | Files | Done when |
|---|---|---|---|
| 8 | Foundation models and ensemble | `chronos_adapter.py`, `scripts/run_foundation.py` | Done: zero-shot Chronos-2 compared with the GBM in both folds (GBM better); ensemble not built; TimesFM and Kronos not tried |
| 9 | Decision layer | `iex_forecast/agent.py` | Procurement simulation compared with naive and oracle strategies. Extension: a battery-storage (BESS) agent with state-of-charge limits and round-trip efficiency |
| 10 | Live inputs and serving | `app/` | Live weather forecasts (e.g. Open-Meteo) feed the same feature pipeline the backtest uses; API and dashboard run |
| 11 | Operations | `iex_forecast/drift.py`, `Dockerfile`, `.github/` | CI green; drift checks scheduled; experiment tracking in place |
| 12 | Reporting | `README.md`, `results/` | Every published number is reproducible from this repository |

## Principles
- Features use realised prices only up to the last day known at bid time; weather, solar and demand enter as forecasts.
- Hyperparameters and agent parameters are never tuned on the test period.
- An improvement is claimed only together with a significance test.
- Simulations are decision support, not trading advice.
- Synthetic results are labelled as synthetic everywhere they appear.
