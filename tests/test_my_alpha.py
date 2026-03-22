"""Tests for the example strategy in strategies/my_alpha.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.my_alpha import MyAlpha


class TestMyAlpha:
    def test_signals_shape(self, sample_bars: pd.DataFrame) -> None:
        strat = MyAlpha()
        signals = strat.generate_signals(sample_bars)
        assert len(signals) == len(sample_bars)
        assert signals.index.equals(sample_bars.index)

    def test_breakout_long(self) -> None:
        """When close equals the rolling high, signal should be 1.0."""
        n = 30
        dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        # Flat then spike up on last bar
        close = np.array([100.0] * (n - 1) + [110.0])
        bars = pd.DataFrame(
            {
                "open": close,
                "high": close,
                "low": close - 1,
                "close": close,
                "volume": 100.0,
                "buy_volume": 50.0,
                "vwap": close,
                "trade_count": 10,
            },
            index=dates,
        )
        strat = MyAlpha(params={"period": 20})
        signals = strat.generate_signals(bars)
        assert signals.iloc[-1] == 1.0

    def test_breakdown_short(self) -> None:
        """When close equals the rolling low, signal should be -1.0."""
        n = 30
        dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        close = np.array([100.0] * (n - 1) + [90.0])
        bars = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close,
                "close": close,
                "volume": 100.0,
                "buy_volume": 50.0,
                "vwap": close,
                "trade_count": 10,
            },
            index=dates,
        )
        strat = MyAlpha(params={"period": 20})
        signals = strat.generate_signals(bars)
        assert signals.iloc[-1] == -1.0

    def test_default_period_is_20(self) -> None:
        strat = MyAlpha()
        assert strat.params.get("period", 20) == 20

    def test_custom_params_used(self) -> None:
        strat = MyAlpha(params={"period": 7})
        assert strat.params["period"] == 7
