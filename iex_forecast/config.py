"""Project-wide constants. Change them here, nowhere else.

Facts below are tracked, with sources and verification status, in docs/market_notes.md.
"""
import datetime

BLOCKS_PER_DAY = 96              # IEX DAM clears in 15-minute blocks (verified on IEX market snapshot page)

# Price cap in Rs/kWh for the standard DAM (CERC order effective 4 April 2023, "until further orders").
# Re-check the current cap before running on recent data: a revision was being sought in mid-2026.
PRICE_CAP = 10.0
HP_DAM_PRICE_CAP = 20.0          # High-Price DAM is a separate segment; do not mix it into DAM series

# Bid window for next-day delivery. NOT yet verified against an official IEX/CERC document
# (secondary source: bids 10:00-12:00). Update once confirmed.
GATE_CLOSURE_TIME = datetime.time(12, 0)

# Information lag, in days. Bids for delivery day D are placed on day D-1, and the DAM
# prices for delivery day D-1 were already cleared on day D-2. So at bid time the last
# delivery day whose price is known is D-1.
INFO_LAG_DAYS = 1

MIN_HISTORY_DAYS = 14            # first day features can be built for
QUANTILES = (0.1, 0.5, 0.9)      # P10 / P50 / P90 forecasts

# Evaluation periods (hour of day, inclusive start / exclusive end)
PERIODS = {
    "night (00-06)": (0, 6),
    "morning (06-10)": (6, 10),
    "solar (10-17)": (10, 17),
    "evening peak (17-23)": (17, 23),
    "late (23-24)": (23, 24),
}
