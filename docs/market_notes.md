# Market notes

Facts the project depends on, with sources and verification status.
Research date: 22 November 2026. "Verified" means read directly on the cited page.
"Secondary" means a third-party article or blog; confirm against IEX or CERC before relying on it.

| Topic | Finding | Source | Status |
|---|---|---|---|
| Block length | 15-minute blocks, 96 per delivery day. Data in this format exists from delivery day 1 April 2022. | [IEX DAM market snapshot](https://iexindia.com/market-data/day-ahead-market/market-snapshot) | Verified |
| Snapshot columns | Date, Hour, Time Block, Purchase Bid (MW), Sell Bid (MW), MCV (MW), Final Scheduled Volume (MW), MCP (Rs/MWh) | same page | Verified |
| Price unit | IEX publishes MCP in **Rs/MWh**. The loader divides by 1000 to work in Rs/kWh. | same page | Verified |
| Price cap | Rs 10/kWh for DAM (including Green DAM), RTM and other segments, effective 4 April 2023 "until further orders". The High-Price DAM is a separate segment capped at Rs 20/kWh. | [Power Line, 4 Apr 2023](https://powerline.net.in/2023/04/04/cerc-reduces-ceiling-price-for-power-trading/) | Secondary |
| Earlier caps | Higher caps applied before April 2023 (a headline from April 2022 mentions Rs 12/kWh; the cap was later extended through December 2022). Details not checked. | search result headlines only | Unverified |
| Cap in 2026 | Still Rs 10/kWh as of 8 June 2026. A solar industry body petitioned CERC to raise it; the order was reserved. Outcome after June 2026 not checked. | [Solarbytes, 8 Jun 2026](https://solarbytes.info/india-bytes/india-solar-industry-seeks-higher-power-prices-amid-record-demand-2026-12014843) | Secondary |
| Bid window | Bids for next-day delivery are submitted between 10:00 and 12:00; results are published in the early afternoon. | [Earthsync explainer](https://earthsync.io/resources/iex-dam-gdam-rtm-explained) | Secondary, **verify** |
| Auction design | Double-sided uniform-price auction; block bids are all-or-nothing. | same page | Secondary |
| Market coupling | CERC approved coupling in July 2025: bids from all exchanges are aggregated into a single clearing price, phased from January 2026 starting with the DAM. Actual rollout status not confirmed. | [Legal Parivar](https://news.legalparivar.com/cerc-reviews-power-exchange-fees-as-market-coupling-nears-january-2026/) | Secondary, **verify status** |

## What this means for the code

**Information lag (`INFO_LAG_DAYS = 1`).** Bids for delivery day D close on day D-1. The DAM price for delivery day D-1 was
cleared on day D-2, so it is already known at bid time. The last day whose price is known is therefore D-1. If the
gate-closure time or this clearing schedule turns out to differ, change `INFO_LAG_DAYS` in `iex_forecast/config.py`
and re-run the tests; every feature, baseline and walk-forward fold follows from that one constant.

**Regime changes.** The price series is not stationary in its rules:
1. The price cap changed (April 2023). Prices before and after are generated under different ceilings.
2. Market coupling (2026) changes how the clearing price is formed.
3. High-Price DAM data must never be mixed into the standard DAM series.

Practical consequences: prefer training data after April 2023, report errors before and after each regime boundary
separately, and treat any model trained across a boundary with suspicion until it is shown to hold on both sides.

## Open questions to settle with primary sources
- [ ] Official gate-closure time for the DAM (IEX procedure document or CERC regulations)
- [ ] Current price cap and its effective date (latest CERC order)
- [ ] Whether and from when DAM prices are market-coupled
- [ ] Terms of use for downloading IEX market data in bulk
