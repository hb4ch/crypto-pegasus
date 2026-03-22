"""Tests for strategy interface and examples."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pegasus.strategy.base import Strategy
from pegasus.strategy.examples.sma_cross import SMACrossStrategy


class TestStrategyABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            Strategy()  # type: ignore[abstract]

    def test_concrete_implementation(self, sample_bars: pd.DataFrame) -> None:
        class Dummy(Strategy):
            def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
                return pd.Series(1.0, index=bars.index)

        strat = Dummy()
        signals = strat.generate_signals(sample_bars)
        assert len(signals) == len(sample_bars)
        assert strat.name == "Dummy"

    def test_params_default(self) -> None:
        class Dummy(Strategy):
            def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
                return pd.Series(0.0, index=bars.index)

        assert Dummy().params == {}
        assert Dummy(params={"x": 1}).params == {"x": 1}


class TestSMACross:
    def test_signals_shape(self, sample_bars: pd.DataFrame) -> None:
        strat = SMACrossStrategy(params={"fast": 10, "slow": 50})
        signals = strat.generate_signals(sample_bars)
        assert len(signals) == len(sample_bars)

    def test_signals_valid_values(self, sample_bars: pd.DataFrame) -> None:
        strat = SMACrossStrategy(params={"fast": 10, "slow": 50})
        signals = strat.generate_signals(sample_bars)
        valid = signals.dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_warmup_period_is_nan(self, sample_bars: pd.DataFrame) -> None:
        slow = 50
        strat = SMACrossStrategy(params={"fast": 10, "slow": slow})
        signals = strat.generate_signals(sample_bars)
        # First (slow-1) bars should be NaN
        assert signals.iloc[: slow - 1].isna().all()

    def test_uptrend_goes_long(self, trending_bars: pd.DataFrame) -> None:
        strat = SMACrossStrategy(params={"fast": 5, "slow": 20})
        signals = strat.generate_signals(trending_bars)
        # In a pure uptrend, fast SMA > slow SMA eventually → signals should be +1
        valid = signals.dropna()
        # At least the latter half should be long
        latter_half = valid.iloc[len(valid) // 2 :]
        assert (latter_half == 1.0).all()
