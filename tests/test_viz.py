"""Smoke tests for pegasus.viz.plots."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from pegasus.engine.backtest import BacktestResult
from pegasus.viz.plots import (
    plot_equity_curve,
    plot_monthly_returns,
    plot_signals,
    save_report,
)


class TestPlotEquityCurve:
    def test_renders(self, mock_backtest_result: BacktestResult) -> None:
        fig = plot_equity_curve(mock_backtest_result)
        assert isinstance(fig, go.Figure)


class TestPlotMonthlyReturns:
    def test_renders(self, mock_backtest_result: BacktestResult) -> None:
        fig = plot_monthly_returns(mock_backtest_result)
        assert isinstance(fig, go.Figure)

    def test_empty_equity(self) -> None:
        empty_idx = pd.DatetimeIndex([], dtype="datetime64[ns, UTC]")
        result = BacktestResult(
            symbol="TEST",
            timeframe="5min",
            equity_curve=pd.DataFrame(
                {"equity": pd.Series(dtype=float, index=empty_idx),
                 "returns": pd.Series(dtype=float, index=empty_idx),
                 "drawdown": pd.Series(dtype=float, index=empty_idx)}
            ),
            positions=pd.Series(dtype=float),
            signals=pd.Series(dtype=float),
        )
        fig = plot_monthly_returns(result)
        assert isinstance(fig, go.Figure)


class TestPlotSignals:
    def test_renders(self, mock_backtest_result: BacktestResult) -> None:
        fig = plot_signals(mock_backtest_result)
        assert isinstance(fig, go.Figure)

    def test_no_bars_raises(self) -> None:
        result = BacktestResult(
            symbol="TEST",
            timeframe="5min",
            equity_curve=pd.DataFrame(),
            positions=pd.Series(dtype=float),
            signals=pd.Series(dtype=float),
            bars=None,
        )
        with pytest.raises(ValueError, match="no bars"):
            plot_signals(result)


class TestSaveReport:
    def test_creates_file(self, mock_backtest_result: BacktestResult, tmp_path: Path) -> None:
        mock_backtest_result.metrics = {"total_return": 0.1, "sharpe": 1.5, "total_trades": 10}
        out = save_report(mock_backtest_result, tmp_path / "report.html")
        assert out.exists()
        content = out.read_text()
        assert "<html>" in content
        assert "Backtest Report" in content
        assert len(content) > 1000  # non-trivial HTML
