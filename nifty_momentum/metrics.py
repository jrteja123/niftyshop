"""
metrics.py — performance metrics for the backtest results.
Pure functions; no Streamlit deps.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_metrics(equity: pd.Series, label: str = "") -> dict:
    """Standard performance metrics from an equity / price series."""
    p = equity.dropna()
    if len(p) < 5:
        return {
            "label": label, "obs": len(p),
            "total_return_pct": 0, "cagr_pct": 0, "sharpe": 0,
            "max_drawdown_pct": 0, "calmar": 0, "final_value": 0,
        }
    daily_ret = p.pct_change().fillna(0)
    n_years = (p.index.max() - p.index.min()).days / 365.25
    total_ret = p.iloc[-1] / p.iloc[0] - 1
    cagr = (p.iloc[-1] / p.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
    rolling_max = p.cummax()
    max_dd = ((p - rolling_max) / rolling_max).min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0
    return {
        "label": label,
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "final_value": round(p.iloc[-1], 0),
        "start_date": p.index.min().date(),
        "end_date": p.index.max().date(),
        "n_years": round(n_years, 2),
    }


def yearly_breakdown(equity: pd.Series) -> pd.DataFrame:
    """Year-by-year performance breakdown."""
    eq = equity.dropna()
    if len(eq) < 5:
        return pd.DataFrame()
    eq = eq.copy()
    eq.index = pd.to_datetime(eq.index)
    eq_df = eq.to_frame("equity")
    eq_df["year"] = eq_df.index.year
    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["dd"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"]

    rows = []
    for year, g in eq_df.groupby("year"):
        if len(g) < 2:
            continue
        first = g["equity"].iloc[0]
        last = g["equity"].iloc[-1]
        ret = last / first - 1
        daily_ret = g["equity"].pct_change().fillna(0)
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
        max_dd = g["dd"].min()
        rows.append({
            "year": int(year),
            "return_pct": round(ret * 100, 2),
            "sharpe": round(sharpe, 2),
            "max_dd_pct": round(max_dd * 100, 2),
            "n_days": len(g),
        })
    return pd.DataFrame(rows)


def enrich_trades_with_entry_info(trades: pd.DataFrame) -> pd.DataFrame:
    """Add entry_date / entry_price / holding_days / pnl_pct columns.

    Populated for SELL and TRIM rows (the exit side). BUY rows leave these
    blank. Re-entries are tracked correctly: entry_date resets after a full
    SELL closes a position.
    """
    if trades.empty:
        for c in ("entry_date", "entry_price", "holding_days", "pnl_pct"):
            trades[c] = pd.Series(dtype="float64")
        return trades

    df = trades.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)

    entry_dates: list = [pd.NaT] * len(df)
    entry_prices: list = [None] * len(df)
    holding_days: list = [None] * len(df)
    pnl_pcts: list = [None] * len(df)

    state: dict[str, dict] = {}  # symbol -> {shares, cost_basis, entry_date}

    for i, row in df.iterrows():
        sym = row["symbol"]
        act = row["action"]
        sh = float(row["shares"])
        px = float(row["price"])
        dt = row["date"]
        s = state.setdefault(sym, {"shares": 0.0, "cost_basis": 0.0, "entry_date": None})

        if act in ("BUY", "ADD"):
            if s["shares"] <= 0:
                s["entry_date"] = dt
                s["cost_basis"] = 0.0
            s["cost_basis"] += sh * px
            s["shares"] += sh
        elif act in ("SELL", "TRIM", "EMA_EXIT", "SMA_EXIT"):
            if s["shares"] > 0:
                avg_cost = s["cost_basis"] / s["shares"]
                entry_dates[i] = s["entry_date"]
                entry_prices[i] = round(avg_cost, 2)
                if s["entry_date"] is not None:
                    holding_days[i] = (dt - s["entry_date"]).days
                pnl_pcts[i] = round((px / avg_cost - 1) * 100, 2) if avg_cost > 0 else None
                if act in ("SELL", "EMA_EXIT", "SMA_EXIT"):
                    s["shares"] = 0.0
                    s["cost_basis"] = 0.0
                    s["entry_date"] = None
                else:  # TRIM keeps avg_cost stable by scaling cost_basis proportionally
                    s["cost_basis"] -= sh * avg_cost
                    s["shares"] -= sh
                    if s["shares"] <= 0:
                        s["shares"] = 0.0
                        s["cost_basis"] = 0.0
                        s["entry_date"] = None

    df["entry_date"] = entry_dates
    df["entry_price"] = entry_prices
    df["holding_days"] = holding_days
    df["pnl_pct"] = pnl_pcts
    return df


def rolling_annualized_returns(equity: pd.Series, window_years: float) -> pd.Series:
    """For each day d in `equity`, the annualized return over the next
    `window_years` calendar years. NaN where the future endpoint falls past
    the end of the series. Used to answer "if I had started on day d, what
    would my N-year CAGR have been?".
    """
    eq = equity.dropna().sort_index()
    if eq.empty:
        return pd.Series(dtype=float)
    window_days = int(round(window_years * 365.25))
    out: dict[pd.Timestamp, float] = {}
    last_date = eq.index.max()
    for d in eq.index:
        target = d + pd.Timedelta(days=window_days)
        if target > last_date:
            continue
        future_idx = eq.index[eq.index >= target]
        if len(future_idx) == 0:
            continue
        future_d = future_idx[0]
        actual_years = (future_d - d).days / 365.25
        if actual_years <= 0:
            continue
        out[d] = ((eq.loc[future_d] / eq.loc[d]) ** (1 / actual_years) - 1) * 100
    return pd.Series(out).sort_index()


def summarize_rolling_returns(s: pd.Series) -> dict:
    """Percentile + tail-risk summary for a rolling-returns series (in %)."""
    if s.empty:
        return {"n_windows": 0}
    return {
        "n_windows": len(s),
        "min": round(float(s.min()), 2),
        "p10": round(float(s.quantile(0.10)), 2),
        "p25": round(float(s.quantile(0.25)), 2),
        "median": round(float(s.median()), 2),
        "mean": round(float(s.mean()), 2),
        "p75": round(float(s.quantile(0.75)), 2),
        "p90": round(float(s.quantile(0.90)), 2),
        "max": round(float(s.max()), 2),
        "pct_negative": round(float((s < 0).mean() * 100), 1),
        "pct_below_bench_proxy_10pct": round(float((s < 10).mean() * 100), 1),
    }


def capital_deployment_series(
    equity: pd.Series,
    holdings_log: pd.DataFrame,
) -> pd.Series:
    """
    % of capital deployed over time.
    Derived from holdings_log's cash column, forward-filled across daily dates.
    """
    if holdings_log.empty:
        return pd.Series(0.0, index=equity.index)

    hl = holdings_log.copy()
    hl["date"] = pd.to_datetime(hl["date"])
    hl = hl.sort_values("date")
    hl["deployed_pct"] = (1 - hl["cash"] / hl["portfolio_value"]) * 100
    deployed = hl.set_index("date")["deployed_pct"]
    return deployed.reindex(equity.index, method="ffill").fillna(0)
