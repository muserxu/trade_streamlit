"""Momentum Blend Strategy — phone-friendly dashboard (Streamlit).

Display-only. Shows the current target allocation, the momentum signal that
drives it, 2026 YTD, the yearly summary, and the daily table. Fetches fresh
prices live each time it loads. Contains NO IBKR connection, NO account info,
NO trading — it only computes and displays public market math.

Run locally:   streamlit run app.py
Deploy free:   push app.py + momentum_blend_strategy.py + requirements.txt to a
               private GitHub repo, then connect it on share.streamlit.io
"""
import streamlit as st
import pandas as pd

import momentum_blend_strategy as S

st.set_page_config(page_title="Momentum Blend", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600)   # cache for 1h so reopening is instant; refresh button clears it
def load():
    """Fetch prices and build the strategy outputs (cached)."""
    prices = S.fetch_prices(S.ALL_TICKERS)
    daily = S.build_daily_frame(prices)          # chronological
    summary = S.yearly_summary(daily, prices)
    # current decision (latest month-end)
    me = S.month_end_indices(prices)
    weights, scores = S.decide_holdings(prices, me[-1])
    cash_s = S.blended_momentum(prices, S.CASH, me[-1])
    if cash_s is not None:
        scores[S.CASH] = cash_s
    info = {
        'allocation': S.format_allocation(weights),
        'decided_on': prices.index[me[-1]],
        'through': prices.index[-1],
        'scores': scores,
        'weights': weights,
        'ytd': daily.iloc[-1]['YTD_Return'],
    }
    return daily, summary, info


# ---- header ----
st.title("📈 Momentum Blend Strategy")
col_a, col_b = st.columns([3, 1])
with col_b:
    if st.button("🔄 Refresh prices"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Fetching prices and computing…"):
    daily, summary, info = load()

# ---- top metrics ----
st.subheader("Current standing")
m1, m2, m3 = st.columns(3)
m1.metric("Target allocation", info['allocation'])
m2.metric("2026 YTD", info['ytd'])
m3.metric("Signal as of", info['decided_on'].strftime('%Y-%m-%d'),
          help=f"prices through {info['through']:%Y-%m-%d}")

# ---- momentum signal ----
st.subheader("Momentum signal (blended 3/6/12-month, %)")
scores = info['scores']
mom_rows = []
held = set(info['weights'].keys())
for a in S.UNIVERSE:
    s = scores.get(a)
    if s is not None:
        mom_rows.append({'Asset': a, 'Momentum %': round(s, 1),
                         'Held now': '✅' if a in held else ''})
mom_df = pd.DataFrame(mom_rows).sort_values('Momentum %', ascending=False)
cash_score = scores.get(S.CASH)

cc1, cc2 = st.columns([2, 1])
with cc1:
    st.dataframe(mom_df, hide_index=True, use_container_width=True)
    if cash_score is not None:
        st.caption(f"Cash threshold (BIL): {cash_score:+.1f}%  — picks below this go to cash.")
with cc2:
    st.bar_chart(mom_df.set_index('Asset')['Momentum %'])

# ---- yearly summary ----
st.subheader("Yearly returns vs VOO buy & hold")
ysum = summary.copy()
ysum['Year'] = ysum['Year'].astype(int).astype(str)
ysum = ysum.rename(columns={
    'Strategy_Pct': 'Strategy %', 'VOO_Pct': 'VOO %',
    'Outperf_Pct': 'Diff %', 'Strat_Cumulative_Pct': 'Strat Cum %',
    'VOO_Cumulative_Pct': 'VOO Cum %'})
for c in ['Strategy %', 'VOO %', 'Diff %', 'Strat Cum %', 'VOO Cum %']:
    ysum[c] = ysum[c].round(2)
st.dataframe(ysum, hide_index=True, use_container_width=True)

# ---- daily table (newest first) ----
st.subheader("Daily tracker")
show_all = st.checkbox("Show full history", value=False)
daily_display = daily.iloc[::-1].reset_index(drop=True)   # newest first
if not show_all:
    daily_display = daily_display.head(30)
    st.caption("Showing the 30 most recent rows. Tick the box for full history.")
st.dataframe(daily_display, hide_index=True, use_container_width=True)

st.divider()
st.caption("Display only — no trading, no account data. "
           "Strategy: top-2 of "
           f"{', '.join(S.UNIVERSE)} by blended {'/'.join(map(str, S.LOOKBACK_MONTHS))}-mo "
           f"momentum, dual-momentum vs {S.CASH}, monthly rebalance.")
