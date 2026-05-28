"""
strategy.py — v5 Momentum Strategy logic, extracted from the backtest script.

Pure functions, no Streamlit dependencies. The Streamlit app calls run_backtest()
with parameters; this module returns structured results (equity curve, trades,
holdings log) that the UI layer formats.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CONFIG DATACLASS — all tunable parameters in one place
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """All v5 strategy parameters. Defaults match the validated v5 settings."""
    capital: float = 10_00_0000
    n_holdings: int = 20
    rebalance_days: int = 30
    # Optional explicit schedules. When set, they OVERRIDE rebalance_days.
    # Priority: day_of_week > day_of_month > rebalance_days interval.
    rebalance_day_of_month: int | None = None  # 1-28 (or higher; snaps to last trading day of month if past)
    rebalance_day_of_week: int | None = None   # 0=Mon, 1=Tue, ..., 4=Fri (NSE trades Mon-Fri)
    rebalance_last_weekday: int | None = None  # 0=Mon..4=Fri; last occurrence in each month
    rebalance_last_trading_day: bool = False   # last trading day of each month
    cost_round_trip: float = 0.001

    # Momentum signal in trading days
    momentum_lookback_td: int = 252
    momentum_skip_td: int = 21

    # Market-wide regime filter (benchmark > N-day SMA -> invested)
    use_regime_filter: bool = True
    regime_sma: int = 200

    # Per-stock SMA filter (close > own N-day SMA required for selection)
    use_per_stock_sma_filter: bool = False
    per_stock_sma_period: int = 200

    # Quality filters
    min_price: float = 50
    min_dollar_vol_20d: float = 5_00_00_000
    max_daily_move_pct: float = 0.15

    # Sector concentration
    sector_cap: int = 5

    # Position-sizing behavior
    disable_trimming: bool = False  # If True, never sell shares of a name that's still in target_symbols.
                                    # Lets winners run; saves trading cost + STCG. Portfolio drifts top-heavy over time.

    # Daily EMA exit (per-position): when set, every day between rebalances
    # we check whether each held position closed below its N-day EMA. If yes,
    # we exit that single position immediately. Sold names can re-enter at
    # the next scheduled rebalance (default re-entry policy). Common values:
    # 21 (fast / whippy), 49 (slower trend signal), 100 (very slow).
    daily_ema_exit_period: int | None = None


@dataclass
class BacktestResult:
    """Structured output from run_backtest()."""
    equity: pd.Series                      # daily portfolio value
    trades: pd.DataFrame                   # action / date / symbol / shares / price / value
    holdings_log: pd.DataFrame             # one row per rebalance
    final_positions: pd.DataFrame          # current portfolio at end of backtest
    next_signal: dict                      # what would be bought at next rebalance
    benchmark: pd.Series                   # daily benchmark value, rebased to capital
    config: StrategyConfig
    benchmark_ticker: str
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------------------------

def compute_daily_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Add per-stock rolling indicators:
      - mom_signal: momentum (close[-skip] / close[-lookback] - 1)
      - max_abs_ret_20d: max abs daily return in last 20 bars (anti-spec filter)
      - median_dollar_vol_20d: median (close*volume) over last 20 bars (liquidity)
    """
    def per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").reset_index(drop=True).copy()
        c = g["close"]
        v = g["volume"]

        c_skip = c.shift(cfg.momentum_skip_td)
        c_back = c.shift(cfg.momentum_lookback_td)
        g["mom_signal"] = c_skip / c_back - 1

        ret_1d = c.pct_change().abs()
        g["max_abs_ret_20d"] = ret_1d.rolling(20).max()

        g["median_dollar_vol_20d"] = (c * v).rolling(20).median()

        sma = c.rolling(cfg.per_stock_sma_period).mean()
        g["close_above_sma"] = c > sma
        return g

    parts = []
    for sym, gr in df.groupby("symbol"):
        sub = per_symbol(gr.copy())
        sub["symbol"] = sym
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def build_regime_signal(bench: pd.Series, sma_period: int) -> pd.Series:
    """True when benchmark > sma_period SMA. NaN before SMA can be computed."""
    sma = bench.rolling(sma_period).mean()
    return bench > sma


def generate_rebalance_dates(
    daily_dates: pd.DatetimeIndex,
    rebalance_days: int,
    day_of_month: int | None = None,
    day_of_week: int | None = None,
    last_weekday: int | None = None,
    last_trading_day: bool = False,
) -> list[pd.Timestamp]:
    """Generate rebalance dates, snapped to trading days.

    Schedule precedence (first that's set wins):
      1. last_trading_day: last trading day of each month
      2. last_weekday: last occurrence of that weekday in each month
         (e.g. last_weekday=4 -> last Friday). Falls back to last trading
         day of the month if no such weekday traded.
      3. day_of_week: weekly on that weekday.
      4. day_of_month: monthly on that day-of-month.
      5. Otherwise: every `rebalance_days` calendar days (legacy interval).
    """
    if len(daily_dates) == 0:
        return []
    if last_trading_day:
        return _generate_last_trading_day_of_month_dates(daily_dates)
    if last_weekday is not None:
        return _generate_last_weekday_of_month_dates(daily_dates, int(last_weekday))
    if day_of_week is not None:
        return _generate_weekly_rebalance_dates(daily_dates, int(day_of_week))
    if day_of_month is not None:
        return _generate_monthly_rebalance_dates(daily_dates, int(day_of_month))

    start = daily_dates.min()
    end = daily_dates.max()
    target_dates = pd.date_range(start, end, freq=f"{rebalance_days}D")
    rebal_dates = []
    seen = set()
    for td in target_dates:
        cands = daily_dates[daily_dates >= td]
        if len(cands) > 0 and cands[0] not in seen:
            seen.add(cands[0])
            rebal_dates.append(cands[0])
    return rebal_dates


def _generate_monthly_rebalance_dates(
    daily_dates: pd.DatetimeIndex, day_of_month: int
) -> list[pd.Timestamp]:
    """For each year-month in `daily_dates`, return the first trading day
    whose day-of-month >= `day_of_month`. If no such trading day exists in
    that month (e.g. user asked for DOM=31 but the month has 30 days), fall
    back to the last trading day of that month."""
    if not 1 <= day_of_month <= 31:
        raise ValueError(f"day_of_month must be in 1..31, got {day_of_month}")
    idx = pd.DatetimeIndex(daily_dates).sort_values()
    rebal: list[pd.Timestamp] = []
    grouped: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for d in idx:
        grouped.setdefault((d.year, d.month), []).append(d)
    for key in sorted(grouped):
        days = grouped[key]
        cands = [d for d in days if d.day >= day_of_month]
        if cands:
            rebal.append(cands[0])
        else:
            rebal.append(days[-1])
    return rebal


def _generate_last_weekday_of_month_dates(
    daily_dates: pd.DatetimeIndex, weekday: int
) -> list[pd.Timestamp]:
    """For each year-month, return the LAST trading day whose weekday matches
    `weekday` (0=Mon..6=Sun). Falls back to the last trading day of the month
    if no day with that weekday traded."""
    if not 0 <= weekday <= 6:
        raise ValueError(f"weekday must be in 0..6, got {weekday}")
    idx = pd.DatetimeIndex(daily_dates).sort_values()
    grouped: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for d in idx:
        grouped.setdefault((d.year, d.month), []).append(d)
    rebal: list[pd.Timestamp] = []
    for key in sorted(grouped):
        days = grouped[key]
        matches = [d for d in days if d.weekday() == weekday]
        rebal.append(matches[-1] if matches else days[-1])
    return rebal


def _generate_last_trading_day_of_month_dates(
    daily_dates: pd.DatetimeIndex,
) -> list[pd.Timestamp]:
    """Last trading day of each year-month."""
    idx = pd.DatetimeIndex(daily_dates).sort_values()
    grouped: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for d in idx:
        grouped.setdefault((d.year, d.month), []).append(d)
    return [grouped[k][-1] for k in sorted(grouped)]


def _generate_weekly_rebalance_dates(
    daily_dates: pd.DatetimeIndex, day_of_week: int
) -> list[pd.Timestamp]:
    """For each ISO week in `daily_dates`, return the first trading day
    whose weekday >= `day_of_week`. Falls back to the last trading day of
    the week if the target DOW is a holiday and no later day exists that
    week (e.g. Friday holiday and target was Friday)."""
    if not 0 <= day_of_week <= 6:
        raise ValueError(f"day_of_week must be in 0..6, got {day_of_week}")
    idx = pd.DatetimeIndex(daily_dates).sort_values()
    iso = idx.isocalendar()
    grouped: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for d, y, w in zip(idx, iso.year, iso.week):
        grouped.setdefault((int(y), int(w)), []).append(d)
    rebal: list[pd.Timestamp] = []
    for key in sorted(grouped):
        days = grouped[key]
        cands = [d for d in days if d.weekday() >= day_of_week]
        if cands:
            rebal.append(cands[0])
        else:
            rebal.append(days[-1])
    return rebal


# ---------------------------------------------------------------------------
# BACKTEST ENGINE
# ---------------------------------------------------------------------------

def run_backtest(
    daily_df: pd.DataFrame,
    benchmark: pd.Series,
    sector_map: dict,
    cfg: StrategyConfig,
    progress_callback: Optional[Callable] = None,
) -> BacktestResult:
    """
    Run v5 momentum backtest.

    daily_df: long-format OHLCV with mom_signal/max_abs_ret_20d/median_dollar_vol_20d
              already computed (from compute_daily_indicators).
    benchmark: daily close series for the benchmark.
    sector_map: dict {yf_symbol: sector_name} for sector-cap enforcement.
    cfg: StrategyConfig
    progress_callback(step, total, message): optional UI progress hook.
    """
    if progress_callback:
        progress_callback(1, 5, "building price pivot...")

    # Build wide pivot of daily close (one column per symbol)
    daily_close = daily_df.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index().ffill()

    # Regime signal (True = invested, False = cash)
    if cfg.use_regime_filter:
        regime = build_regime_signal(benchmark, cfg.regime_sma)
        regime_daily = regime.reindex(daily_close.index).ffill().fillna(False)
    else:
        regime_daily = pd.Series(True, index=daily_close.index)

    # Rebalance dates
    rebal_dates = generate_rebalance_dates(
        daily_close.index,
        cfg.rebalance_days,
        day_of_month=cfg.rebalance_day_of_month,
        day_of_week=cfg.rebalance_day_of_week,
        last_weekday=cfg.rebalance_last_weekday,
        last_trading_day=cfg.rebalance_last_trading_day,
    )

    if progress_callback:
        progress_callback(2, 5, f"prepping pivots for {len(rebal_dates)} rebalances...")

    # Pre-pivot indicators for fast date lookup
    mom_pivot = daily_df.pivot_table(
        index="date", columns="symbol", values="mom_signal", aggfunc="last"
    )
    max_ret_pivot = daily_df.pivot_table(
        index="date", columns="symbol", values="max_abs_ret_20d", aggfunc="last"
    )
    liq_pivot = daily_df.pivot_table(
        index="date", columns="symbol", values="median_dollar_vol_20d", aggfunc="last"
    )
    sma_pivot = daily_df.pivot_table(
        index="date", columns="symbol", values="close_above_sma", aggfunc="last"
    ) if "close_above_sma" in daily_df.columns else None

    # Daily EMA exit pivot (per-stock per-day). When close < EMA on any day
    # between rebalances, exit that position immediately. Re-entry policy:
    # the position can re-enter only at the next scheduled rebalance.
    ema_pivot = None
    if cfg.daily_ema_exit_period and cfg.daily_ema_exit_period > 0:
        ema_pivot = daily_close.ewm(
            span=int(cfg.daily_ema_exit_period), adjust=False
        ).mean()

    cash = cfg.capital
    holdings: dict[str, float] = {}
    holdings_log = []
    trade_log = []
    prev_rebal_date: pd.Timestamp | None = None

    if progress_callback:
        progress_callback(3, 5, "running rebalances...")

    for idx, rebal_date in enumerate(rebal_dates):
        if rebal_date not in mom_pivot.index:
            continue

        # --- DAILY EMA EXIT: walk inter-rebalance days, exit any held name
        #     whose close fell below its EMA. Re-entry happens (if at all) at
        #     the very next rebalance below, via the normal target logic.
        if ema_pivot is not None and prev_rebal_date is not None:
            interim_mask = (daily_close.index > prev_rebal_date) & (daily_close.index < rebal_date)
            interim_dates = daily_close.index[interim_mask]
            for dt in interim_dates:
                if not holdings:
                    break
                for sym in list(holdings.keys()):
                    if sym not in daily_close.columns:
                        continue
                    px = daily_close.loc[dt, sym]
                    ema = ema_pivot.loc[dt, sym] if sym in ema_pivot.columns else np.nan
                    if pd.isna(px) or pd.isna(ema) or px >= ema:
                        continue
                    sh = holdings[sym]
                    proceeds = sh * px * (1 - cfg.cost_round_trip / 2)
                    cash += proceeds
                    trade_log.append({
                        "date": dt, "action": "EMA_EXIT", "symbol": sym,
                        "shares": int(sh), "price": float(px), "value": float(sh * px),
                    })
                    del holdings[sym]

        mom_slice = mom_pivot.loc[rebal_date].dropna()
        max_ret_slice = max_ret_pivot.loc[rebal_date]
        liq_slice = liq_pivot.loc[rebal_date]
        close_slice = daily_close.loc[rebal_date]

        cands = pd.DataFrame({
            "mom_signal": mom_slice,
            "max_abs_ret_20d": max_ret_slice.reindex(mom_slice.index),
            "median_dollar_vol_20d": liq_slice.reindex(mom_slice.index),
            "close": close_slice.reindex(mom_slice.index),
        })

        before_count = len(cands)
        cands = cands[
            (cands["close"] >= cfg.min_price) &
            (cands["median_dollar_vol_20d"] >= cfg.min_dollar_vol_20d) &
            (cands["max_abs_ret_20d"] <= cfg.max_daily_move_pct) &
            (cands["mom_signal"] > 0)
        ]
        if cfg.use_per_stock_sma_filter and sma_pivot is not None and rebal_date in sma_pivot.index:
            above_sma = sma_pivot.loc[rebal_date].reindex(cands.index).fillna(False)
            cands = cands[above_sma.astype(bool)]
        after_count = len(cands)
        cands = cands.sort_values("mom_signal", ascending=False)

        # Apply sector cap before final selection
        target_symbols = []
        sector_counts = {}
        for sym in cands.index:
            if len(target_symbols) >= cfg.n_holdings:
                break
            sec = sector_map.get(sym, "Unknown")
            if sector_counts.get(sec, 0) >= cfg.sector_cap:
                continue
            target_symbols.append(sym)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

        invested = bool(regime_daily.loc[rebal_date])
        if not invested:
            target_symbols = []

        prices_at_rebal = close_slice

        # SELLs (positions not in new target)
        to_sell = set(holdings.keys()) - set(target_symbols)
        for sym in to_sell:
            sh = holdings[sym]
            px = prices_at_rebal.get(sym, np.nan)
            if np.isnan(px):
                continue
            proceeds = sh * px * (1 - cfg.cost_round_trip / 2)
            cash += proceeds
            trade_log.append({
                "date": rebal_date, "action": "SELL", "symbol": sym,
                "shares": int(sh), "price": float(px), "value": float(sh * px),
            })
            del holdings[sym]

        # Mark portfolio value (cash + remaining holdings)
        port_val = cash
        for sym, sh in holdings.items():
            px = prices_at_rebal.get(sym, np.nan)
            if not np.isnan(px):
                port_val += sh * px

        if target_symbols:
            target_per_position = port_val / len(target_symbols)
            for sym in target_symbols:
                px = prices_at_rebal.get(sym, np.nan)
                if np.isnan(px) or px <= 0:
                    continue
                target_shares = int(target_per_position / px)
                current_shares = holdings.get(sym, 0)
                delta = target_shares - current_shares

                if delta > 0:
                    cost = delta * px * (1 + cfg.cost_round_trip / 2)
                    if cost <= cash:
                        cash -= cost
                        holdings[sym] = target_shares
                        trade_log.append({
                            "date": rebal_date, "action": "BUY", "symbol": sym,
                            "shares": int(delta), "price": float(px),
                            "value": float(delta * px),
                        })
                elif delta < 0 and abs(delta) >= 1 and not cfg.disable_trimming:
                    proceeds = abs(delta) * px * (1 - cfg.cost_round_trip / 2)
                    cash += proceeds
                    holdings[sym] = target_shares
                    trade_log.append({
                        "date": rebal_date, "action": "TRIM", "symbol": sym,
                        "shares": int(abs(delta)), "price": float(px),
                        "value": float(abs(delta) * px),
                    })

        holdings_log.append({
            "date": rebal_date,
            "regime_invested": invested,
            "n_target": len(target_symbols),
            "n_held_after": len(holdings),
            "candidates_pre_filter": before_count,
            "candidates_post_filter": after_count,
            "portfolio_value": port_val,
            "cash": cash,
            "top_5": ", ".join(target_symbols[:5]) if target_symbols else "CASH",
        })
        prev_rebal_date = rebal_date

    if progress_callback:
        progress_callback(4, 5, "building daily equity curve...")

    # Reconstruct daily equity by walking forward applying trades
    trades_df = pd.DataFrame(trade_log)
    if not trades_df.empty:
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_by_date = {d: g for d, g in trades_df.groupby("date")}
    else:
        trades_by_date = {}

    sim_cash = cfg.capital
    sim_holdings: dict[str, int] = {}
    portfolio_value_series = {}

    for dt in daily_close.index:
        if dt in trades_by_date:
            for _, tr in trades_by_date[dt].iterrows():
                sym = tr["symbol"]
                if tr["action"] == "BUY":
                    cost = tr["shares"] * tr["price"] * (1 + cfg.cost_round_trip / 2)
                    sim_cash -= cost
                    sim_holdings[sym] = sim_holdings.get(sym, 0) + tr["shares"]
                elif tr["action"] in ("SELL", "EMA_EXIT"):
                    proceeds = tr["shares"] * tr["price"] * (1 - cfg.cost_round_trip / 2)
                    sim_cash += proceeds
                    sim_holdings.pop(sym, None)
                elif tr["action"] == "TRIM":
                    proceeds = tr["shares"] * tr["price"] * (1 - cfg.cost_round_trip / 2)
                    sim_cash += proceeds
                    sim_holdings[sym] = sim_holdings.get(sym, 0) - tr["shares"]

        day_prices = daily_close.loc[dt]
        port_val = sim_cash
        for sym, sh in sim_holdings.items():
            px = day_prices.get(sym, np.nan)
            if not np.isnan(px):
                port_val += sh * px
        portfolio_value_series[dt] = port_val

    equity = pd.Series(portfolio_value_series).sort_index()

    # Build final_positions DataFrame from sim_holdings at end of backtest
    final_positions_rows = []
    last_prices = daily_close.iloc[-1]
    # Reconstruct first-buy date per symbol
    first_buy = {}
    avg_cost = {}
    if not trades_df.empty:
        for sym, g in trades_df.groupby("symbol"):
            buys = g[g["action"] == "BUY"]
            if not buys.empty:
                first_buy[sym] = buys["date"].min()
                # weighted avg buy price
                total_qty = buys["shares"].sum()
                total_value = (buys["shares"] * buys["price"]).sum()
                avg_cost[sym] = total_value / total_qty if total_qty > 0 else np.nan

    for sym, sh in sim_holdings.items():
        if sh <= 0:
            continue
        cur_px = last_prices.get(sym, np.nan)
        ac = avg_cost.get(sym, np.nan)
        if np.isnan(cur_px) or np.isnan(ac):
            continue
        final_positions_rows.append({
            "symbol": sym,
            "first_buy_date": first_buy.get(sym),
            "shares": int(sh),
            "avg_cost": round(ac, 2),
            "current_price": round(cur_px, 2),
            "invested": round(sh * ac, 0),
            "current_value": round(sh * cur_px, 0),
            "unrealized_pnl": round(sh * (cur_px - ac), 0),
            "pnl_pct": round((cur_px / ac - 1) * 100, 2) if ac > 0 else 0,
            "sector": sector_map.get(sym, "Unknown"),
        })
    final_positions = pd.DataFrame(final_positions_rows)

    # ---- Next signal: what would the strategy buy at the next rebalance?
    # Use the most recent date in daily_df as the "would-be" signal date
    last_date = daily_close.index.max()
    next_signal = {"as_of": last_date, "in_cash": False, "target_symbols": [],
                   "regime_invested": True}
    try:
        if last_date in mom_pivot.index:
            mom_now = mom_pivot.loc[last_date].dropna()
            max_now = max_ret_pivot.loc[last_date]
            liq_now = liq_pivot.loc[last_date]
            close_now = daily_close.loc[last_date]
            cands_now = pd.DataFrame({
                "mom_signal": mom_now,
                "max_abs_ret_20d": max_now.reindex(mom_now.index),
                "median_dollar_vol_20d": liq_now.reindex(mom_now.index),
                "close": close_now.reindex(mom_now.index),
            })
            cands_now = cands_now[
                (cands_now["close"] >= cfg.min_price) &
                (cands_now["median_dollar_vol_20d"] >= cfg.min_dollar_vol_20d) &
                (cands_now["max_abs_ret_20d"] <= cfg.max_daily_move_pct) &
                (cands_now["mom_signal"] > 0)
            ]
            if cfg.use_per_stock_sma_filter and sma_pivot is not None and last_date in sma_pivot.index:
                above_sma_now = sma_pivot.loc[last_date].reindex(cands_now.index).fillna(False)
                cands_now = cands_now[above_sma_now.astype(bool)]
            cands_now = cands_now.sort_values("mom_signal", ascending=False)

            target_now = []
            sector_counts_now = {}
            for sym in cands_now.index:
                if len(target_now) >= cfg.n_holdings:
                    break
                sec = sector_map.get(sym, "Unknown")
                if sector_counts_now.get(sec, 0) >= cfg.sector_cap:
                    continue
                target_now.append({
                    "symbol": sym,
                    "mom_signal_pct": round(cands_now.loc[sym, "mom_signal"] * 100, 2),
                    "close": round(cands_now.loc[sym, "close"], 2),
                    "sector": sec,
                })
                sector_counts_now[sec] = sector_counts_now.get(sec, 0) + 1

            invested_now = bool(regime_daily.loc[last_date])
            next_signal = {
                "as_of": last_date,
                "in_cash": not invested_now,
                "regime_invested": invested_now,
                "target_symbols": target_now,
            }
    except Exception as e:
        next_signal["error"] = str(e)

    # Build benchmark equity, rebased to initial capital
    bench_aligned = benchmark.reindex(equity.index).ffill()
    if bench_aligned.notna().any():
        bench_eq = bench_aligned / bench_aligned.dropna().iloc[0] * cfg.capital
    else:
        bench_eq = pd.Series(index=equity.index, dtype=float)

    if progress_callback:
        progress_callback(5, 5, "done.")

    return BacktestResult(
        equity=equity,
        trades=trades_df if not trades_df.empty else pd.DataFrame(
            columns=["date", "action", "symbol", "shares", "price", "value"]
        ),
        holdings_log=pd.DataFrame(holdings_log),
        final_positions=final_positions,
        next_signal=next_signal,
        benchmark=bench_eq,
        config=cfg,
        benchmark_ticker="",  # filled in by caller
    )
