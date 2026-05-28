"""
app.py — Nifty Momentum Strategy Backtester (Streamlit UI)

Run locally:
    streamlit run app.py

Or deploy to Streamlit Cloud — point at the repo, set Python version 3.10+.
"""

from __future__ import annotations
import datetime as dt

import pandas as pd
import streamlit as st

from data import (
    UNIVERSE_CONFIG, load_universe, get_prices, get_benchmark,
    get_benchmark_ticker,
)
from strategy import StrategyConfig, run_backtest, compute_daily_indicators
from metrics import (
    compute_metrics,
    yearly_breakdown,
    capital_deployment_series,
    enrich_trades_with_entry_info,
)
from charts import (
    portfolio_value_chart, drawdown_chart,
    capital_deployment_chart, yearly_returns_chart,
)


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Nifty Momentum Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# SIDEBAR — STRATEGY PARAMETERS
# ---------------------------------------------------------------------------

st.sidebar.title("Strategy Parameters")

# Date defaults: 5 years ago to today
default_end = dt.date.today()
default_start = default_end - dt.timedelta(days=5 * 365)

start_date = st.sidebar.date_input(
    "Start Date", value=default_start,
    min_value=dt.date(2010, 1, 1), max_value=default_end,
)
end_date = st.sidebar.date_input(
    "End Date", value=default_end,
    min_value=start_date, max_value=default_end,
)

universe_name = st.sidebar.selectbox(
    "Universe",
    options=list(UNIVERSE_CONFIG.keys()),
    index=1,  # default Nifty 500
)

rebalance_mode = st.sidebar.selectbox(
    "Rebalance schedule",
    options=[
        "Every N days",
        "Monthly on day",
        "Weekly on day",
        "Last trading day of month",
        "Last weekday of month",
    ],
    index=3,
    help="How rebalance dates are picked. 'Every N days' = legacy interval; "
         "'Monthly on day' = fixed day-of-month; 'Weekly on day' = fixed weekday "
         "(snaps forward on market holidays); 'Last trading day of month' = "
         "fires on the final NSE trading session each month; 'Last weekday of "
         "month' = fires on the last occurrence of the chosen weekday each month.",
)
rebalance_days = 25
rebalance_day_of_month: int | None = None
rebalance_day_of_week: int | None = None
rebalance_last_weekday: int | None = None
rebalance_last_trading_day: bool = False
_dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
if rebalance_mode == "Every N days":
    rebalance_days = st.sidebar.number_input(
        "Rebalance days (calendar)",
        min_value=7, max_value=90, value=30, step=1,
        help="30 = monthly (recommended). Tested: 15-day overfits.",
    )
elif rebalance_mode == "Monthly on day":
    rebalance_day_of_month = st.sidebar.number_input(
        "Day of month",
        min_value=1, max_value=31, value=25, step=1,
        help="Rebalance on this day each month. Snaps to next trading day on "
             "holidays/weekends. Falls back to month's last trading day if "
             "the day-of-month is past the end of a short month.",
    )
elif rebalance_mode == "Weekly on day":
    _selected_dow = st.sidebar.selectbox(
        "Day of week",
        options=_dow_names,
        index=1,  # Tuesday default
        help="Rebalance on this weekday each week. Snaps forward on holidays.",
    )
    rebalance_day_of_week = _dow_names.index(_selected_dow)
    st.sidebar.warning(
        "⚠️ Weekly rebalancing was tested historically and produced worse "
        "OOS returns than 30-day (6.83% vs 9.76% CAGR) due to whipsaw. "
        "Use with caution.",
        icon="⚠️",
    )
elif rebalance_mode == "Last trading day of month":
    rebalance_last_trading_day = True
    st.sidebar.caption(
        "📅 Fires on the final NSE trading session of each calendar month. "
        "Most defensible monthly schedule — no DOM cherry-picking risk."
    )
elif rebalance_mode == "Last weekday of month":
    _selected_last_dow = st.sidebar.selectbox(
        "Weekday",
        options=_dow_names,
        index=4,  # default last Friday of month
        help="Picks the LAST occurrence of this weekday in each month. "
             "Falls back to the month's last trading day if no such weekday "
             "traded that month.",
    )
    rebalance_last_weekday = _dow_names.index(_selected_last_dow)

capital = st.sidebar.number_input(
    "Initial Capital (₹)",
    min_value=1_00_000, max_value=10_00_00_000, value=10_00_000, step=1_00_000,
    format="%d",
)

# --- Seed status panel
from data import load_price_seed, load_benchmark_seed
seed_df = load_price_seed()
bench_seed_df = load_benchmark_seed()
if not seed_df.empty:
    seed_min = seed_df["date"].min().date()
    seed_max = seed_df["date"].max().date()
    n_seed_syms = seed_df["symbol"].nunique()
    st.sidebar.caption(
        f"📦 Seed: {n_seed_syms} symbols, "
        f"{seed_min} → {seed_max}"
    )
else:
    st.sidebar.caption("📦 Seed: empty (will use yfinance)")

use_seed_only = st.sidebar.checkbox(
    "Use seed only (skip all yfinance calls)",
    value=True,
    help=(
        "OFF (default): the app fetches ONLY the gap between your seed's last "
        "date and your end date. First run after a long gap may take a few "
        "minutes; subsequent runs are instant because the seed is updated.\n\n"
        "ON: never call yfinance. End date is clamped to the seed's last date. "
        "Use this if yfinance is throttling you."
    ),
)

with st.sidebar.expander("Advanced Parameters"):
    n_holdings = st.number_input("Number of holdings", 5, 50, 20, 1)
    mom_lookback = st.number_input("Momentum lookback (trading days)",
                                    60, 504, 252, 1)
    mom_skip = st.number_input("Momentum skip (trading days)",
                                0, 60, 21, 1,
                                help="Skip recent days to avoid reversal effect.")
    use_regime = st.checkbox("Use regime filter (go to cash in bear markets)",
                              value=False)
    regime_sma = st.number_input("Regime SMA period (days)",
                                  50, 400, 200, 10)
    sector_cap = st.number_input("Max positions per sector", 1, 100, 5, 1)
    disable_trimming = st.checkbox(
        "Disable trimming (let winners run)",
        value=True,
        help="If checked, the strategy never sells shares of a name that is still in the top-20 target. "
             "Reduces trade count and STCG. Allows portfolio to drift top-heavy over time.",
    )
    cost_pct = st.number_input("Round-trip cost (%)",
                                0.0, 1.0, 0.10, 0.05) / 100
    min_price = st.number_input("Min stock price (₹)", 1, 1000, 50, 1)
    min_dollar_vol_cr = st.number_input(
        "Min 20-day median ₹ volume (crores)", 0.1, 100.0, 5.0, 0.5,
    )
    max_daily_move = st.number_input(
        "Max single-day move % (anti-spec filter)",
        1.0, 50.0, 15.0, 1.0,
    ) / 100

run_button = st.sidebar.button("▶️ Run Backtest", type="primary",
                                use_container_width=True)


# ---------------------------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------------------------

st.title("📈 Nifty Momentum Strategy Backtester")
st.markdown(
    "Daily-timeframe 12-1 momentum strategy with regime filter and sector caps. "
    "Holds the top N momentum stocks from your chosen universe, rebalanced periodically."
)

with st.expander("⚠️ Backtest caveats — read before trusting results"):
    st.markdown(
        """
        - **Survivorship bias:** the constituent CSVs reflect the *current*
          index members. Stocks that were added during the test window appear
          as if they were always investable; stocks that were removed are
          missing. This inflates historical returns.
        - **No fundamentals:** the strategy uses only OHLCV data. It cannot
          react to earnings, news, or shareholding changes.
        - **Tax not modelled:** all returns are pre-tax. Annual turnover for
          this strategy is typically 200-700%, mostly STCG (20% in India).
        - **Slippage assumption:** 0.10% round-trip is realistic for liquid
          mid-caps but optimistic for small-caps.
        - **Past performance ≠ future returns.** Especially in momentum.
        """
    )


# ---------------------------------------------------------------------------
# BACKTEST EXECUTION
# ---------------------------------------------------------------------------

if not run_button:
    st.info("Configure parameters in the sidebar and click **Run Backtest**.")
    st.stop()


cfg = StrategyConfig(
    capital=float(capital),
    n_holdings=int(n_holdings),
    rebalance_days=int(rebalance_days),
    rebalance_day_of_month=rebalance_day_of_month,
    rebalance_day_of_week=rebalance_day_of_week,
    rebalance_last_weekday=rebalance_last_weekday,
    rebalance_last_trading_day=rebalance_last_trading_day,
    cost_round_trip=float(cost_pct),
    momentum_lookback_td=int(mom_lookback),
    momentum_skip_td=int(mom_skip),
    use_regime_filter=bool(use_regime),
    regime_sma=int(regime_sma),
    min_price=float(min_price),
    min_dollar_vol_20d=float(min_dollar_vol_cr * 1_00_00_000),
    max_daily_move_pct=float(max_daily_move),
    sector_cap=int(sector_cap),
    disable_trimming=bool(disable_trimming),
)

start_str = start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")

# If user opted into seed-only mode, clamp end_str to what the seed has
if use_seed_only and not seed_df.empty:
    seed_max_ts = seed_df["date"].max()
    if pd.Timestamp(end_str) > seed_max_ts:
        clamped_end = seed_max_ts.strftime("%Y-%m-%d")
        st.info(
            f"📦 Seed-only mode: end date clamped from {end_str} to {clamped_end} "
            f"(latest date available in the seed)."
        )
        end_str = clamped_end

# --- Load universe
with st.spinner(f"Loading {universe_name} universe..."):
    try:
        universe = load_universe(universe_name)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    symbols = universe["yf_symbol"].tolist()
    sector_map = dict(zip(universe["yf_symbol"], universe["sector"]))

# --- Load prices (may trigger yfinance fetch for delta)
progress_placeholder = st.empty()
progress_bar = progress_placeholder.progress(0, text="Loading price data...")

def price_cb(done, total, msg):
    progress_bar.progress(min(done / total, 1.0), text=msg)

try:
    prices = get_prices(symbols, start_str, end_str,
                        progress_callback=price_cb,
                        use_seed_only=use_seed_only)
except Exception as e:
    st.error(f"Price load failed: {e}")
    st.stop()
finally:
    progress_placeholder.empty()

if prices.empty:
    st.error("No price data available for the selected window.")
    st.stop()

st.success(
    f"Loaded {len(prices):,} bars across {prices['symbol'].nunique()} symbols."
)

# --- Load benchmark
benchmark_failed = False
with st.spinner(f"Loading benchmark for {universe_name}..."):
    try:
        bench_close, bench_ticker = get_benchmark(universe_name, start_str, end_str,
                                                   use_seed_only=use_seed_only)
    except Exception as e:
        st.warning(
            f"⚠️ Benchmark load failed: {e}\n\n"
            "**Auto-disabling the regime filter for this run** since it depends "
            "on the benchmark. Strategy will stay fully invested. Drawdowns will "
            "be larger than usual. Try again later when yfinance recovers, or "
            "supply a seed parquet."
        )
        bench_close = pd.Series(dtype=float)
        bench_ticker = "N/A"
        benchmark_failed = True
        if cfg.use_regime_filter:
            cfg.use_regime_filter = False  # auto-disable rather than sit in cash

# --- Compute indicators
with st.spinner("Computing momentum indicators..."):
    prices_with_ind = compute_daily_indicators(prices, cfg)

# --- Run backtest
progress_placeholder = st.empty()
progress_bar = progress_placeholder.progress(0, text="Running backtest...")

def bt_cb(done, total, msg):
    progress_bar.progress(min(done / total, 1.0), text=msg)

try:
    result = run_backtest(
        prices_with_ind, bench_close, sector_map, cfg,
        progress_callback=bt_cb,
    )
    result.benchmark_ticker = bench_ticker
except Exception as e:
    st.error(f"Backtest failed: {e}")
    raise
finally:
    progress_placeholder.empty()

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------

m_strategy = compute_metrics(result.equity, "Strategy")
m_benchmark = compute_metrics(result.benchmark, f"{universe_name} ({bench_ticker})") \
    if not result.benchmark.empty else {}

# --- Performance Comparison: 3 columns
st.header("📊 Performance Comparison")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🎯 Strategy")
    st.metric("Total Return", f"{m_strategy['total_return_pct']:+.2f}%")
    st.metric("CAGR", f"{m_strategy['cagr_pct']:+.2f}%")
    st.metric("Sharpe Ratio", f"{m_strategy['sharpe']:.2f}")
    st.metric("Max Drawdown", f"{m_strategy['max_drawdown_pct']:.2f}%")
    st.metric("Final Value", f"₹{m_strategy['final_value']:,.0f}")

with col2:
    st.subheader(f"📉 {universe_name} Benchmark")
    if m_benchmark:
        st.metric("Total Return", f"{m_benchmark['total_return_pct']:+.2f}%")
        st.metric("CAGR", f"{m_benchmark['cagr_pct']:+.2f}%")
        st.metric("Sharpe Ratio", f"{m_benchmark['sharpe']:.2f}")
        st.metric("Max Drawdown", f"{m_benchmark['max_drawdown_pct']:.2f}%")
        st.metric("Final Value", f"₹{m_benchmark['final_value']:,.0f}")
    else:
        st.info("No benchmark data available.")

with col3:
    st.subheader("🏆 Strategy vs Benchmark")
    if m_benchmark:
        st.metric("Excess Return",
                  f"{m_strategy['total_return_pct'] - m_benchmark['total_return_pct']:+.2f}pp")
        st.metric("Excess CAGR",
                  f"{m_strategy['cagr_pct'] - m_benchmark['cagr_pct']:+.2f}pp")
        st.metric("Sharpe vs Bench",
                  f"{m_strategy['sharpe'] - m_benchmark['sharpe']:+.2f}")
        st.metric("DD vs Bench",
                  f"{m_strategy['max_drawdown_pct'] - m_benchmark['max_drawdown_pct']:+.2f}pp")
        st.metric("Total Trades", f"{len(result.trades):,}")
    else:
        st.info("No benchmark to compare against.")

# --- Portfolio value chart
st.header("📈 Portfolio Value")
st.plotly_chart(
    portfolio_value_chart(result.equity, result.benchmark, universe_name),
    use_container_width=True,
)

# --- Drawdown chart
st.plotly_chart(
    drawdown_chart(result.equity, result.benchmark),
    use_container_width=True,
)

# --- Yearly returns
st.header("📅 Yearly Performance")
yearly = yearly_breakdown(result.equity)
if not yearly.empty:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.plotly_chart(yearly_returns_chart(yearly), use_container_width=True)
    with col_b:
        st.dataframe(yearly, hide_index=True, use_container_width=True)

# --- Capital deployment
st.header("💰 Capital Deployment")
deployed = capital_deployment_series(result.equity, result.holdings_log)
st.plotly_chart(capital_deployment_chart(deployed), use_container_width=True)
st.caption(
    f"Strategy was invested on {(deployed > 50).sum()} of {len(deployed)} days "
    f"({100 * (deployed > 50).sum() / len(deployed):.1f}%) "
    "— rest of the time it sat in cash due to the regime filter."
)

# --- Next signal (what would the strategy buy at the next rebalance)
st.header("🎯 Next Rebalance Signal")
ns = result.next_signal
if ns.get("in_cash"):
    st.warning(
        f"As of {ns['as_of'].date()}: regime filter is **OFF** — strategy "
        "would hold cash. Wait for the regime to turn before deploying capital."
    )
elif ns.get("target_symbols"):
    st.success(
        f"As of {ns['as_of'].date()}: strategy would hold these "
        f"{len(ns['target_symbols'])} stocks at the next rebalance:"
    )
    next_df = pd.DataFrame(ns["target_symbols"])
    st.dataframe(next_df, hide_index=True, use_container_width=True)
else:
    st.info("No signal data available.")

# --- Current Positions
st.header("📦 Current Positions (end of backtest)")
if not result.final_positions.empty:
    pos = result.final_positions.copy()
    pos = pos.sort_values("pnl_pct", ascending=False)
    st.dataframe(pos, hide_index=True, use_container_width=True)
    total_invested = pos["invested"].sum()
    total_current = pos["current_value"].sum()
    total_pnl = pos["unrealized_pnl"].sum()
    st.info(
        f"**Total positions:** {len(pos)} · "
        f"**Invested:** ₹{total_invested:,.0f} · "
        f"**Current value:** ₹{total_current:,.0f} · "
        f"**Unrealized P&L:** ₹{total_pnl:+,.0f}"
    )
else:
    st.info("No open positions at the end of the backtest.")

# --- Recent Trades
st.header("📋 Recent Trades")
enriched_trades = enrich_trades_with_entry_info(result.trades)
if not enriched_trades.empty:
    recent = enriched_trades.sort_values("date", ascending=False).head(50).copy()
    recent["date"] = pd.to_datetime(recent["date"]).dt.date
    if "entry_date" in recent.columns:
        recent["entry_date"] = pd.to_datetime(recent["entry_date"]).dt.date
    display_cols = [
        "date", "action", "symbol", "shares", "price", "value",
        "entry_date", "entry_price", "holding_days", "pnl_pct",
    ]
    display_cols = [c for c in display_cols if c in recent.columns]
    st.dataframe(
        recent[display_cols],
        hide_index=True,
        use_container_width=True,
        column_config={
            "pnl_pct": st.column_config.NumberColumn("pnl_pct", format="%.2f%%"),
            "entry_price": st.column_config.NumberColumn("entry_price", format="%.2f"),
            "price": st.column_config.NumberColumn("price", format="%.2f"),
            "value": st.column_config.NumberColumn("value", format="%.0f"),
            "holding_days": st.column_config.NumberColumn("holding_days", format="%d"),
        },
    )
    st.caption(
        f"Showing 50 most recent of {len(enriched_trades):,} total trades. "
        "Entry columns populated for SELL/TRIM rows."
    )
else:
    st.info("No trades executed.")

# --- Downloads
st.header("⬇️ Download Backtest Data")
col_d1, col_d2, col_d3 = st.columns(3)
with col_d1:
    st.download_button(
        "Download trades (CSV)",
        enriched_trades.to_csv(index=False).encode("utf-8"),
        file_name=f"trades_{universe_name.lower().replace(' ', '')}_{end_str}.csv",
        mime="text/csv",
    )
with col_d2:
    st.download_button(
        "Download equity curve (CSV)",
        result.equity.to_csv().encode("utf-8"),
        file_name=f"equity_{universe_name.lower().replace(' ', '')}_{end_str}.csv",
        mime="text/csv",
    )
with col_d3:
    st.download_button(
        "Download holdings log (CSV)",
        result.holdings_log.to_csv(index=False).encode("utf-8"),
        file_name=f"holdings_{universe_name.lower().replace(' ', '')}_{end_str}.csv",
        mime="text/csv",
    )
