# Data sources

Verify availability and terms of use for every source before use. Nothing here has been downloaded by this project.

## Prices
IEX day-ahead market data, 15-minute blocks, available from April 2022
([market snapshot](https://iexindia.com/market-data/day-ahead-market/market-snapshot)).
Use the **MCP** column (Rs/MWh). Review IEX terms of use before any automated collection. Prefer published downloads
and reports over scraping.

## Weather
The model needs the temperature **forecast that existed at bid time**, not the realised temperature.
Archived forecast products (for example Open-Meteo's historical forecast data) are the natural source. Check what
forecast issue time each archive represents. If only realised weather is available, either skip the weather group
or add noise and say so in the README: do not present it as a forecast.

## Solar and demand
Grid-India publishes daily reports with demand and generation. Use day-ahead forecasts where they are published.
Realised values may only enter the model with the same lag as prices (known up to the last day before bid time),
never for the target day.

## Holidays and news
Official holiday lists (`is_holiday` column). News is optional; `iex_forecast/news.py` only uses headlines from the
previous day.

## Checklist before loading
- [ ] 96 blocks per day, IST timestamps, no missing days
- [ ] Price in Rs/MWh in the file (the loader converts)
- [ ] Only standard DAM prices, no High-Price DAM rows
- [ ] Forecast columns are forecasts available at bid time
- [ ] You know which price-cap regime each part of the series belongs to (docs/market_notes.md)

Then run `python run_backtest.py --data data/your_file.csv --real` and read the warnings it prints.
