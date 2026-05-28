"""
charts.py — Plotly figures used by the Streamlit app.
Each function returns a go.Figure for app.py to render.
"""

from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Dark theme colors that match the screenshot's aesthetic
COLOR_STRATEGY = "#4FC3F7"      # light blue
COLOR_BENCHMARK = "#FF9800"     # orange
COLOR_GREEN = "#66BB6A"
COLOR_RED = "#EF5350"


def portfolio_value_chart(
    strategy_eq: pd.Series,
    benchmark_eq: pd.Series,
    universe_name: str,
) -> go.Figure:
    """Portfolio value over time — strategy vs benchmark, both rebased."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strategy_eq.index, y=strategy_eq.values,
        mode="lines", name="Momentum Strategy",
        line=dict(color=COLOR_STRATEGY, width=2),
    ))
    if benchmark_eq is not None and benchmark_eq.notna().any():
        fig.add_trace(go.Scatter(
            x=benchmark_eq.index, y=benchmark_eq.values,
            mode="lines", name=f"{universe_name} Benchmark",
            line=dict(color=COLOR_BENCHMARK, width=2, dash="dash"),
        ))
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date", yaxis_title="Portfolio Value (₹)",
        template="plotly_dark", height=420,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
    )
    return fig


def drawdown_chart(strategy_eq: pd.Series, benchmark_eq: pd.Series) -> go.Figure:
    """Underwater curve — % below all-time-high."""
    def underwater(s: pd.Series) -> pd.Series:
        peak = s.cummax()
        return ((s - peak) / peak) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strategy_eq.index, y=underwater(strategy_eq).values,
        mode="lines", name="Strategy",
        line=dict(color=COLOR_STRATEGY, width=2),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.2)",
    ))
    if benchmark_eq is not None and benchmark_eq.notna().any():
        fig.add_trace(go.Scatter(
            x=benchmark_eq.index, y=underwater(benchmark_eq).values,
            mode="lines", name="Benchmark",
            line=dict(color=COLOR_BENCHMARK, width=2, dash="dash"),
        ))
    fig.update_layout(
        title="Drawdown — How Far Below Peak",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        template="plotly_dark", height=320,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
    )
    return fig


def capital_deployment_chart(deployed_pct: pd.Series) -> go.Figure:
    """% of capital invested over time. Visualizes the regime filter behavior."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=deployed_pct.index, y=deployed_pct.values,
        mode="lines", name="% Capital Deployed",
        line=dict(color=COLOR_GREEN, width=2),
        fill="tozeroy", fillcolor="rgba(102,187,106,0.3)",
    ))
    fig.update_layout(
        title="Capital Deployment Over Time — Cash vs Invested",
        xaxis_title="Date", yaxis_title="% Invested",
        template="plotly_dark", height=280,
        yaxis=dict(range=[0, 105]),
        hovermode="x",
    )
    return fig


def yearly_returns_chart(yearly_df: pd.DataFrame) -> go.Figure:
    """Year-by-year returns as a bar chart."""
    if yearly_df.empty:
        return go.Figure()
    colors = [COLOR_GREEN if r >= 0 else COLOR_RED for r in yearly_df["return_pct"]]
    fig = go.Figure(go.Bar(
        x=yearly_df["year"].astype(str),
        y=yearly_df["return_pct"],
        marker_color=colors,
        text=[f"{r:+.1f}%" for r in yearly_df["return_pct"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Yearly Returns (%)",
        xaxis_title="Year", yaxis_title="Return (%)",
        template="plotly_dark", height=320,
    )
    return fig
