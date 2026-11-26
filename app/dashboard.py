"""Streamlit dashboard: replay of the walk-forward test year (not a live forecast).

Run:  pip install -r requirements-app.txt && streamlit run app/dashboard.py
All numbers come from results/ (run `python run_backtest.py` first). The calculations live in
app/summary.py and are unit-tested; this file only lays them out.
"""
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plots import day_figure  # noqa: E402
from summary import agent_table, day_view, display_table, models_table, monthly_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PRED, META, RES = RESULTS / "predictions.npz", RESULTS / "run_meta.json", RESULTS / "results.json"

st.set_page_config(page_title="IEX day-ahead forecast", layout="wide")
st.title("IEX day-ahead price forecast and procurement signal")
st.caption("Replay of the walk-forward test year with calibrated P10-P90 bands. "
           "Not a live forecast. Decision support, not trading advice.")

if not PRED.exists():
    st.error("results/predictions.npz not found. It is not distributed with the repository because it contains exchange prices. "
             "Build the dataset and run `python run_backtest.py --data data/iex_dam_weather.csv --real` (see README).")
    st.stop()

meta = json.loads(META.read_text()) if META.exists() else {}
results = json.loads(RES.read_text()) if RES.exists() else None
source = meta.get("data_source", "unknown")
if "REAL" in source:
    st.info(f"Data source: {source}. Test period {meta.get('test_start', '?')} to {meta.get('test_end', '?')}.")
else:
    st.warning(f"Data source: {source}. These results are not about the real market.")

z = np.load(PRED)
dates = list(z["dates"])

tab_day, tab_perf, tab_method = st.tabs(["Day view", "Test-year performance", "Method and limits"])

with tab_day:
    day = st.selectbox("Delivery day", dates, index=0)
    i = dates.index(day)
    k = day_view(z, i)
    q, actual, bench, buy = z["quantiles"][i], z["actual"][i], z["benchmark"][i], z["buy"][i].astype(bool)
    left, right = st.columns([3, 2])
    with left:
        st.pyplot(day_figure(q, actual, bench, buy))
    with right:
        m1, m2 = st.columns(2)
        m1.metric("Blocks bought", f"{k['blocks_to_buy']}/96")
        m2.metric("Day MAE (Rs/kWh)", f"{k['day_mae']:.2f}")
        m1.metric("Band coverage", f"{100 * k['coverage']:.0f}%")
        m2.metric("Confidence", f"{k['mean_confidence']:.2f}")
        m1.metric("Saving vs benchmark", f"{k['saving_pct']:+.1f}%")
        m2.metric("Agent cost (Rs/kWh)", f"{k['cost_agent']:.2f}")
    st.caption("Saving is the agent's cost against always paying the benchmark (negative = the agent paid more that day). "
               "Confidence is 1 minus the relative band width: a tighter band means higher confidence. "
               "Band coverage is the share of the 96 blocks whose actual price fell inside P10-P90.")

with tab_perf:
    monthly = monthly_table(z)
    st.subheader("Month by month")
    st.dataframe(display_table(monthly), hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Forecast error by month (Rs/kWh, lower is better)")
        st.line_chart(monthly.set_index("month")[["MAE P50 (Rs/kWh)", "MAE 7-day block mean (Rs/kWh)"]],
                      color=["#2dd4bf", "#fbbf24"])
    with c2:
        st.caption("Agent saving vs benchmark by month (%)")
        st.bar_chart(monthly.set_index("month")[["agent saving vs benchmark (%)"]], color="#2dd4bf")
    if results:
        st.subheader("Models on the whole test year")
        st.dataframe(display_table(models_table(results)), hide_index=True)
        st.subheader("Procurement simulation (stylised)")
        st.dataframe(display_table(agent_table(results)), hide_index=True)
        st.caption("Full tables, significance tests and the validation window are in results/RESULTS.md.")

with tab_method:
    st.markdown("""
**What this is.** A quantile gradient-boosting model forecasts the 96 fifteen-minute prices of the IEX
day-ahead market, with a conformally calibrated P10-P90 band. A rule-based agent decides, block by block,
whether to buy on the exchange or defer to a benchmark price.

**No look-ahead.** Features use realised prices only up to the day before bidding. Models are refit on data
known at that time. A unit test destroys all later prices and checks that the features do not change.

**What the numbers mean.** Every figure here comes from a backtest of one test year on data exported from
the exchange website. See `docs/FINDINGS.md` for what is and is not established.

**Limits.** One test year and one regime. The procurement simulation has no volumes, limits or penalties,
so its savings are not trading profit. Market rules may change (price cap, market coupling).
""")
