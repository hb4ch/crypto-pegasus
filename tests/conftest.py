from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """Synthetic OHLCV bars for testing (1000 5-min bars)."""
    rng = np.random.default_rng(42)
    n = 1000
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")

    # Random walk price
    close = 2000.0 + np.cumsum(rng.standard_normal(n) * 2.0)
    noise = rng.standard_normal(n) * 0.5

    return pd.DataFrame(
        {
            "open": close + noise,
            "high": close + np.abs(rng.standard_normal(n)) * 3.0,
            "low": close - np.abs(rng.standard_normal(n)) * 3.0,
            "close": close,
            "volume": rng.integers(100, 5000, size=n).astype(float),
            "buy_volume": rng.integers(50, 2500, size=n).astype(float),
            "vwap": close + noise * 0.1,
            "trade_count": rng.integers(10, 500, size=n),
        },
        index=dates,
    )


@pytest.fixture
def constant_bars() -> pd.DataFrame:
    """Bars with constant price — useful for fee/slippage tests."""
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    price = 1000.0
    return pd.DataFrame(
        {
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 100.0,
            "buy_volume": 50.0,
            "vwap": price,
            "trade_count": 10,
        },
        index=dates,
    )


@pytest.fixture
def trending_bars() -> pd.DataFrame:
    """Bars with a clear uptrend — useful for signal direction tests."""
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = np.linspace(1000, 2000, n)  # Linear uptrend
    return pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": 100.0,
            "buy_volume": 60.0,
            "vwap": close,
            "trade_count": 10,
        },
        index=dates,
    )


@pytest.fixture
def downtrend_bars() -> pd.DataFrame:
    """Bars with a clear downtrend."""
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = np.linspace(2000, 1000, n)
    return pd.DataFrame(
        {
            "open": close + 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": 100.0,
            "buy_volume": 40.0,
            "vwap": close,
            "trade_count": 10,
        },
        index=dates,
    )


@pytest.fixture
def mock_backtest_result(sample_bars: pd.DataFrame) -> "BacktestResult":
    """A reusable BacktestResult for viz/metrics tests."""
    from pegasus.engine.backtest import BacktestResult, Trade

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

    signals = pd.Series(0.0, index=sample_bars.index)
    signals.iloc[10:20] = 1.0
    signals.iloc[30:40] = -1.0

    return BacktestResult(
        symbol="TEST",
        timeframe="5min",
        equity_curve=equity_df,
        positions=signals.shift(1),
        signals=signals,
        trades=trades,
        bars=sample_bars,
    )
