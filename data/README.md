# data/

CSV files are git-ignored. Put your market data here.

## Canonical schema

One row per 15-minute block, 96 rows per delivery day, no missing days.

| column | required | meaning |
|---|---|---|
| date | yes | delivery date, `YYYY-MM-DD` |
| block | yes | 1..96 (block 1 = 00:00-00:15 IST) |
| mcp_rs_mwh | yes | market clearing price in Rs/MWh, exactly as IEX publishes it |
| temp_fc_c | no | temperature **forecast** available at bid time |
| solar_fc_mw | no | solar generation **forecast** available at bid time |
| demand_fc_mw | no | demand **forecast** available at bid time |
| is_holiday | no | 1 on public holidays, else 0 |
| purchase_bid_mw | no | purchase bid volume of that block (realised; only previous days are used) |
| sell_bid_mw | no | sell bid volume of that block (realised; only previous days are used) |
| mcv_mw | no | market clearing volume of that block (realised; only previous days are used) |

Missing optional columns are allowed: their feature groups are skipped in the ablation.
Realised weather, solar or demand must **not** go in the `*_fc_*` columns. That is leakage.

## Create a template

    python scripts/make_synthetic_csv.py      # writes data/synthetic_demo.csv (SYNTHETIC)

## Run on your own file

    python run_backtest.py --data data/your_file.csv --real

See docs/DATA_SOURCES.md and docs/market_notes.md.
