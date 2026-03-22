"""Tests for pegasus.metrics.report."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pegasus.engine.backtest import BacktestResult, Trade
from pegasus.metrics.report import (
    compute_metrics, print_report, _trade_metrics, _omega_ratio, _tail_ratio,
    _max_consecutive, _max_dd_duration, _exposure_pct, _rolling_sharpe,
)


@pytest.fixture
def mock_result(sample_bars: pd.DataFrame) -> BacktestResult:
    """Create a mock BacktestResult with known returns."""
    n = len(sample_bars)
    rng = np.random.default_rng(99)
    returns = pd.Series(rng.standard_normal(n) * 0.001, index=sample_bars.index)
    equity = (1 + returns).cumprod() * 100_000
    drawdown = (equity - equity.cummax()) / equity.cummax()

    equity_df = pd.DataFrame(
        {"equity": equity, "returns": returns, "drawdown": drawdown},
        index=sample_bars.index,
    )

    trades = [
        Trade(
            entry_time=sample_bars.index[10],
            exit_time=sample_bars.index[20],
            direction=1.0,
            entry_price=2000.0,
            exit_price=2050.0,
            pnl=50.0,
            return_pct=0.025,
        ),
        Trade(
            entry_time=sample_bars.index[30],
            exit_time=sample_bars.index[40],
            direction=-1.0,
            entry_price=2100.0,
            exit_price=2120.0,
            pnl=-20.0,
            return_pct=-0.0095,
        ),
    ]

    return BacktestResult(
        symbol="TEST",
        timeframe="5min",
        equity_curve=equity_df,
        positions=pd.Series(0.0, index=sample_bars.index),
        signals=pd.Series(0.0, index=sample_bars.index),
        trades=trades,
    )


class TestComputeMetrics:
    def test_returns_expected_keys(self, mock_result: BacktestResult) -> None:
        metrics = compute_metrics(mock_result)
        expected = {
            # Returns
            "total_return", "cagr", "avg_return", "cumulative_pnl",
            # Risk-adjusted
            "sharpe", "sortino", "calmar", "omega", "tail_ratio", "payoff_ratio",
            # Risk
            "volatility", "downside_volatility", "max_drawdown",
            "max_drawdown_duration_bars", "var_95", "cvar_95", "skew", "kurtosis",
            # Exposure
            "exposure_pct", "long_exposure_pct", "short_exposure_pct", "avg_position_size",
            # Rolling
            "rolling_sharpe_mean", "rolling_sharpe_std",
            # Trades
            "total_trades", "win_rate", "profit_factor",
            "max_consecutive_wins", "max_consecutive_losses",
            "median_trade_return", "trade_return_std",
        }
        assert expected.issubset(set(metrics.keys()))

    def test_sharpe_is_finite(self, mock_result: BacktestResult) -> None:
        metrics = compute_metrics(mock_result)
        assert np.isfinite(metrics["sharpe"])

    def test_max_drawdown_is_negative(self, mock_result: BacktestResult) -> None:
        metrics = compute_metrics(mock_result)
        assert metrics["max_drawdown"] <= 0

    def test_win_rate_between_0_and_1(self, mock_result: BacktestResult) -> None:
        metrics = compute_metrics(mock_result)
        assert 0 <= metrics["win_rate"] <= 1

    def test_var_cvar_relationship(self, mock_result: BacktestResult) -> None:
        metrics = compute_metrics(mock_result)
        # CVaR (expected shortfall) should be <= VaR (further into the tail)
        assert metrics["cvar_95"] <= metrics["var_95"]

    def test_exposure_metrics(self, mock_result: BacktestResult) -> None:
        metrics = compute_metrics(mock_result)
        assert 0 <= metrics["exposure_pct"] <= 1
        assert 0 <= metrics["long_exposure_pct"] <= 1
        assert 0 <= metrics["short_exposure_pct"] <= 1

    def test_omega_positive(self, mock_result: BacktestResult) -> None:
        metrics = compute_metrics(mock_result)
        assert metrics["omega"] > 0

    def test_empty_returns(self) -> None:
        result = BacktestResult(
            symbol="TEST",
            timeframe="5min",
            equity_curve=pd.DataFrame(
                {"equity": pd.Series(dtype=float), "returns": pd.Series(dtype=float), "drawdown": pd.Series(dtype=float)}
            ),
            positions=pd.Series(dtype=float),
            signals=pd.Series(dtype=float),
        )
        assert compute_metrics(result) == {}

    def test_all_positive_returns(self) -> None:
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        returns = pd.Series(0.001, index=idx)  # constant positive
        equity = (1 + returns).cumprod() * 100_000
        drawdown = (equity - equity.cummax()) / equity.cummax()
        result = BacktestResult(
            symbol="TEST", timeframe="5min",
            equity_curve=pd.DataFrame({"equity": equity, "returns": returns, "drawdown": drawdown}, index=idx),
            positions=pd.Series(1.0, index=idx),
            signals=pd.Series(1.0, index=idx),
        )
        m = compute_metrics(result)
        assert m["max_drawdown"] == 0.0


class TestHelpers:
    def test_omega_ratio_balanced(self) -> None:
        # Symmetric returns around 0 → omega ≈ 1
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.standard_normal(10000) * 0.01)
        omega = _omega_ratio(returns)
        assert 0.8 < omega < 1.2

    def test_tail_ratio_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.standard_normal(10000) * 0.01)
        tr = _tail_ratio(returns)
        assert 0.8 < tr < 1.2  # Symmetric distribution → ratio ≈ 1

    def test_max_consecutive(self) -> None:
        is_win = np.array([True, True, True, False, False, True, False])
        wins, losses = _max_consecutive(is_win)
        assert wins == 3
        assert losses == 2

    def test_max_consecutive_empty(self) -> None:
        wins, losses = _max_consecutive(np.array([]))
        assert wins == 0 and losses == 0

    def test_max_dd_duration(self) -> None:
        # Equity: rises, drops, stays low, recovers
        equity = pd.Series([100, 110, 105, 100, 95, 90, 100, 115])
        duration = _max_dd_duration(equity)
        assert duration == 5  # bars 2-6 inclusive (below peak of 110)

    def test_exposure_pct_all_flat(self) -> None:
        positions = pd.Series([0.0, 0.0, 0.0])
        assert _exposure_pct(positions) == 0.0

    def test_exposure_pct_all_in(self) -> None:
        positions = pd.Series([1.0, -1.0, 1.0])
        assert _exposure_pct(positions) == 1.0

    def test_rolling_sharpe_short_data(self) -> None:
        returns = pd.Series([0.01, -0.01, 0.02])  # 3 bars < 252 window
        mean, std = _rolling_sharpe(returns)
        assert mean == 0.0
        assert std == 0.0


class TestPrintReport:
    def test_smoke(self, mock_result: BacktestResult, capsys) -> None:
        metrics = compute_metrics(mock_result)
        print_report(metrics)
        captured = capsys.readouterr()
        assert "BACKTEST REPORT" in captured.out
        assert "RETURNS" in captured.out
        assert "TRADES" in captured.out


class TestTradeMetrics:
    def test_empty_trades(self) -> None:
        m = _trade_metrics([])
        assert m["total_trades"] == 0
        assert m["win_rate"] == 0.0
        assert m["max_consecutive_wins"] == 0
        assert m["max_consecutive_losses"] == 0

    def test_all_winners(self) -> None:
        trades = [
            Trade(
                entry_time=pd.Timestamp("2024-01-01"),
                exit_time=pd.Timestamp("2024-01-02"),
                direction=1.0,
                entry_price=100,
                exit_price=110,
                pnl=10,
                return_pct=0.1,
            )
        ]
        m = _trade_metrics(trades)
        assert m["win_rate"] == 1.0
        assert m["profit_factor"] == float("inf")

    def test_mixed_trades(self) -> None:
        trades = [
            Trade(
                entry_time=pd.Timestamp("2024-01-01"),
                exit_time=pd.Timestamp("2024-01-02"),
                direction=1.0,
                entry_price=100,
                exit_price=110,
                pnl=10,
                return_pct=0.1,
            ),
            Trade(
                entry_time=pd.Timestamp("2024-01-03"),
                exit_time=pd.Timestamp("2024-01-04"),
                direction=-1.0,
                entry_price=110,
                exit_price=115,
                pnl=-5,
                return_pct=-0.05,
            ),
        ]
        m = _trade_metrics(trades)
        assert m["total_trades"] == 2
        assert m["win_rate"] == 0.5
        assert m["profit_factor"] == 0.1 / 0.05  # 2.0

    def test_all_losers(self) -> None:
        trades = [
            Trade(
                entry_time=pd.Timestamp("2024-01-01"),
                exit_time=pd.Timestamp("2024-01-02"),
                direction=1.0,
                entry_price=100,
                exit_price=90,
                pnl=-10,
                return_pct=-0.1,
            ),
        ]
        m = _trade_metrics(trades)
        assert m["win_rate"] == 0.0
        assert m["profit_factor"] == 0.0
        assert m["max_consecutive_losses"] == 1
