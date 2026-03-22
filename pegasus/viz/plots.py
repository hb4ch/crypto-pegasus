from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pegasus.engine.backtest import BacktestResult


def plot_equity_curve(result: BacktestResult) -> go.Figure:
    """Equity curve with drawdown subplot."""
    eq = result.equity_curve
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Equity Curve", "Drawdown"),
    )

    fig.add_trace(
        go.Scatter(x=eq.index, y=eq["equity"], name="Equity", line=dict(width=1.5)),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=eq.index,
            y=eq["drawdown"],
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="crimson", width=1),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{result.symbol} – {result.strategy_name if hasattr(result, 'strategy_name') else ''} ({result.timeframe})",
        height=600,
        showlegend=False,
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
    return fig


def plot_monthly_returns(result: BacktestResult) -> go.Figure:
    """Heatmap of monthly returns."""
    equity = result.equity_curve["equity"]
    monthly = equity.resample("ME").last().pct_change().dropna()

    if monthly.empty:
        return go.Figure()

    df = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        }
    )
    pivot = df.pivot_table(index="year", columns="month", values="return")

    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    # Reindex to ensure all 12 months are present
    pivot = pivot.reindex(columns=range(1, 13))

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values * 100,
            x=month_labels,
            y=pivot.index.astype(str),
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"{v:.1f}%" if pd.notna(v) else "" for v in row] for row in pivot.values * 100],
            texttemplate="%{text}",
            colorbar=dict(title="Return %"),
        )
    )
    fig.update_layout(
        title=f"{result.symbol} – Monthly Returns",
        height=300,
        template="plotly_white",
    )
    return fig


def plot_signals(result: BacktestResult, max_points: int = 5000) -> go.Figure:
    """Price chart with buy/sell signal markers."""
    bars = result.bars
    if bars is None:
        raise ValueError("BacktestResult has no bars attached")

    signals = result.signals

    # Downsample for performance if too many points
    if len(bars) > max_points:
        step = len(bars) // max_points
        bars = bars.iloc[::step]
        signals = signals.iloc[::step]

    fig = go.Figure()

    # Price line
    fig.add_trace(
        go.Scatter(x=bars.index, y=bars["close"], name="Close", line=dict(width=1))
    )

    # Buy signals
    buy_mask = signals > 0
    if buy_mask.any():
        fig.add_trace(
            go.Scatter(
                x=bars.index[buy_mask],
                y=bars["close"][buy_mask],
                mode="markers",
                name="Long",
                marker=dict(color="green", size=4, symbol="triangle-up"),
            )
        )

    # Sell signals
    sell_mask = signals < 0
    if sell_mask.any():
        fig.add_trace(
            go.Scatter(
                x=bars.index[sell_mask],
                y=bars["close"][sell_mask],
                mode="markers",
                name="Short",
                marker=dict(color="red", size=4, symbol="triangle-down"),
            )
        )

    fig.update_layout(
        title=f"{result.symbol} – Signals",
        yaxis_title="Price",
        height=400,
        template="plotly_white",
    )
    return fig


def save_report(result: BacktestResult, output_path: str | Path) -> Path:
    """Generate a standalone HTML report with all plots and metrics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figs = [
        plot_equity_curve(result),
        plot_monthly_returns(result),
    ]
    if result.bars is not None:
        figs.append(plot_signals(result))

    # Build HTML
    html_parts = [
        "<html><head>",
        '<meta charset="utf-8">',
        f"<title>Backtest Report – {result.symbol}</title>",
        "<style>body{font-family:sans-serif;margin:40px;} table{border-collapse:collapse;} "
        "td,th{padding:6px 14px;border:1px solid #ddd;text-align:right;} "
        "th{background:#f5f5f5;text-align:left;}</style>",
        "</head><body>",
        f"<h1>Backtest Report – {result.symbol} ({result.timeframe})</h1>",
    ]

    # Metrics table
    if result.metrics:
        html_parts.append("<h2>Performance Metrics</h2><table>")
        pct_keys = {
            "total_return", "cagr", "max_drawdown", "volatility",
            "win_rate", "avg_win", "avg_loss", "avg_trade_return",
            "best_trade", "worst_trade",
        }
        for k, v in result.metrics.items():
            if k in pct_keys:
                html_parts.append(f"<tr><th>{k}</th><td>{v:.2%}</td></tr>")
            elif isinstance(v, int):
                html_parts.append(f"<tr><th>{k}</th><td>{v}</td></tr>")
            elif isinstance(v, float):
                html_parts.append(f"<tr><th>{k}</th><td>{v:.4f}</td></tr>")
            else:
                html_parts.append(f"<tr><th>{k}</th><td>{v}</td></tr>")
        html_parts.append("</table>")

    # Charts
    for fig in figs:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    html_parts.append("</body></html>")

    output_path.write_text("\n".join(html_parts))
    return output_path
