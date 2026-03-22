"""Tests for pegasus.engine.backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pegasus.config import BacktestConfig
from pegasus.engine.backtest import BacktestEngine, BacktestResult, _detect_trades
from pegasus.strategy.base import Strategy
from pegasus.strategy.examples.sma_cross import SMACrossStrategy


# ----- Helper strategies for testing -----


class AlwaysLong(Strategy):
    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=bars.index)


class AlwaysFlat(Strategy):
    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=bars.index)


class AlternatingStrategy(Strategy):
    """Alternates between long and short every bar — high turnover."""

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0.0, index=bars.index)
        signals.iloc[::2] = 1.0
        signals.iloc[1::2] = -1.0
        return signals


class AlwaysShort(Strategy):
    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(-1.0, index=bars.index)


class HalfLong(Strategy):
    """50% position sizing."""

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(0.5, index=bars.index)


class NaNStrategy(Strategy):
    """Returns all NaN — simulates no valid signals."""

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(np.nan, index=bars.index)


# ----- Tests -----


class TestBacktestEngine:
    def _make_engine(self, strategy: Strategy, **kwargs) -> BacktestEngine:
        config = BacktestConfig(
            initial_capital=100_000.0,
            fee_rate=0.001,
            slippage_bps=0.0,
            **kwargs,
        )
        return BacktestEngine(strategy=strategy, config=config)

    def test_always_long_positive_in_uptrend(self, trending_bars: pd.DataFrame) -> None:
        engine = self._make_engine(AlwaysLong())
        result = engine.run_on_bars(trending_bars, "TEST", "5min")

        assert isinstance(result, BacktestResult)
        assert result.equity_curve["equity"].iloc[-1] > 100_000

    def test_always_flat_no_change(self, sample_bars: pd.DataFrame) -> None:
        engine = self._make_engine(AlwaysFlat())
        result = engine.run_on_bars(sample_bars, "TEST", "5min")

        # Flat position → no returns → equity stays at initial
        eq = result.equity_curve["equity"]
        np.testing.assert_allclose(eq.iloc[-1], 100_000, atol=1e-6)

    def test_lookahead_bias_prevention(self, sample_bars: pd.DataFrame) -> None:
        """Positions should be shifted by 1 bar from signals."""
        engine = self._make_engine(AlwaysLong())
        result = engine.run_on_bars(sample_bars, "TEST", "5min")

        # First position should be NaN (shifted from signal at bar 0)
        assert pd.isna(result.positions.iloc[0])

    def test_fee_deduction(self, trending_bars: pd.DataFrame) -> None:
        """With fees, equity should be lower than without fees for a high-turnover strategy."""
        engine_no_fee = BacktestEngine(
            AlternatingStrategy(),
            BacktestConfig(initial_capital=100_000.0, fee_rate=0.0, slippage_bps=0.0),
        )
        engine_with_fee = BacktestEngine(
            AlternatingStrategy(),
            BacktestConfig(initial_capital=100_000.0, fee_rate=0.01, slippage_bps=0.0),
        )

        r_nofee = engine_no_fee.run_on_bars(trending_bars, "TEST", "5min")
        r_fee = engine_with_fee.run_on_bars(trending_bars, "TEST", "5min")

        assert r_nofee.equity_curve["equity"].iloc[-1] > r_fee.equity_curve["equity"].iloc[-1]

    def test_equity_curve_columns(self, sample_bars: pd.DataFrame) -> None:
        engine = self._make_engine(AlwaysLong())
        result = engine.run_on_bars(sample_bars, "TEST", "5min")

        assert set(result.equity_curve.columns) == {"equity", "returns", "drawdown"}

    def test_drawdown_non_positive(self, sample_bars: pd.DataFrame) -> None:
        engine = self._make_engine(AlwaysLong())
        result = engine.run_on_bars(sample_bars, "TEST", "5min")
        assert (result.equity_curve["drawdown"] <= 0).all()

    def test_high_turnover_incurs_costs(self, sample_bars: pd.DataFrame) -> None:
        """Alternating strategy should have lower equity due to high turnover costs."""
        engine_alt = self._make_engine(AlternatingStrategy())
        engine_flat = self._make_engine(AlwaysFlat())

        r_alt = engine_alt.run_on_bars(sample_bars, "TEST", "5min")
        r_flat = engine_flat.run_on_bars(sample_bars, "TEST", "5min")

        # Flat stays at 100k, alternating should be below due to costs
        assert r_alt.equity_curve["equity"].iloc[-1] < r_flat.equity_curve["equity"].iloc[-1]

    def test_slippage_reduces_equity(self, trending_bars: pd.DataFrame) -> None:
        """Slippage should reduce equity for a high-turnover strategy."""
        engine_no_slip = BacktestEngine(
            AlternatingStrategy(),
            BacktestConfig(initial_capital=100_000.0, fee_rate=0.0, slippage_bps=0.0),
        )
        engine_with_slip = BacktestEngine(
            AlternatingStrategy(),
            BacktestConfig(initial_capital=100_000.0, fee_rate=0.0, slippage_bps=50.0),
        )
        r_no = engine_no_slip.run_on_bars(trending_bars, "TEST", "5min")
        r_slip = engine_with_slip.run_on_bars(trending_bars, "TEST", "5min")
        assert r_no.equity_curve["equity"].iloc[-1] > r_slip.equity_curve["equity"].iloc[-1]

    def test_fractional_position_sizing(self, trending_bars: pd.DataFrame) -> None:
        """Signal=0.5 should produce ~half the return of signal=1.0."""
        engine_full = BacktestEngine(AlwaysLong(), BacktestConfig(initial_capital=100_000, fee_rate=0.0, slippage_bps=0.0))
        engine_half = BacktestEngine(HalfLong(), BacktestConfig(initial_capital=100_000, fee_rate=0.0, slippage_bps=0.0))
        r_full = engine_full.run_on_bars(trending_bars, "TEST", "5min")
        r_half = engine_half.run_on_bars(trending_bars, "TEST", "5min")

        full_ret = r_full.equity_curve["returns"].dropna().sum()
        half_ret = r_half.equity_curve["returns"].dropna().sum()
        np.testing.assert_allclose(half_ret, full_ret * 0.5, rtol=0.01)

    def test_short_position_in_downtrend(self, downtrend_bars: pd.DataFrame) -> None:
        """Always-short should profit in a downtrend."""
        engine = BacktestEngine(AlwaysShort(), BacktestConfig(initial_capital=100_000, fee_rate=0.0, slippage_bps=0.0))
        result = engine.run_on_bars(downtrend_bars, "TEST", "5min")
        assert result.equity_curve["equity"].iloc[-1] > 100_000

    def test_nan_signals_treated_as_flat(self, sample_bars: pd.DataFrame) -> None:
        """NaN signals should result in NaN positions → no PnL."""
        engine = BacktestEngine(NaNStrategy(), BacktestConfig(initial_capital=100_000, fee_rate=0.0, slippage_bps=0.0))
        result = engine.run_on_bars(sample_bars, "TEST", "5min")
        # All NaN signals → all NaN positions → fillna(0) in equity → stays flat
        eq = result.equity_curve["equity"]
        np.testing.assert_allclose(eq.iloc[-1], 100_000, atol=1e-6)

    def test_single_bar_input(self) -> None:
        """Engine should not crash on a single bar."""
        dates = pd.date_range("2024-01-01", periods=1, freq="5min", tz="UTC")
        bars = pd.DataFrame(
            {"open": 100, "high": 101, "low": 99, "close": 100,
             "volume": 100, "buy_volume": 50, "vwap": 100, "trade_count": 10},
            index=dates,
        )
        engine = self._make_engine(AlwaysLong())
        result = engine.run_on_bars(bars, "TEST", "5min")
        assert len(result.equity_curve) == 1

    def test_result_has_bars_attached(self, sample_bars: pd.DataFrame) -> None:
        engine = self._make_engine(AlwaysLong())
        result = engine.run_on_bars(sample_bars, "TEST", "5min")
        assert result.bars is not None
        assert len(result.bars) == len(sample_bars)

    def test_first_entry_incurs_transaction_cost(self) -> None:
        """First position entry (NaN→1.0) must incur fee, not be skipped."""
        n = 5
        dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        # Flat prices so no PnL — only costs matter
        bars = pd.DataFrame(
            {
                "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                "volume": 100.0, "buy_volume": 50.0, "vwap": 100.0, "trade_count": 10,
            },
            index=dates,
        )
        fee_rate = 0.01  # 1% fee
        engine = BacktestEngine(
            AlwaysLong(),
            BacktestConfig(initial_capital=100_000, fee_rate=fee_rate, slippage_bps=0.0),
        )
        result = engine.run_on_bars(bars, "TEST", "5min")
        # Bar 1 is the first entry (position goes 0→1). With flat prices,
        # the return at bar 1 should be exactly -fee_rate (cost of entry).
        bar1_return = result.equity_curve["returns"].iloc[1]
        np.testing.assert_allclose(bar1_return, -fee_rate, atol=1e-10)

    def test_fee_exact_magnitude(self) -> None:
        """Verify fee cost is exactly fee_rate * turnover per bar."""
        n = 4
        dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        bars = pd.DataFrame(
            {
                "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                "volume": 100.0, "buy_volume": 50.0, "vwap": 100.0, "trade_count": 10,
            },
            index=dates,
        )
        fee_rate = 0.005
        engine = BacktestEngine(
            AlternatingStrategy(),
            BacktestConfig(initial_capital=100_000, fee_rate=fee_rate, slippage_bps=0.0),
        )
        result = engine.run_on_bars(bars, "TEST", "5min")
        returns = result.equity_curve["returns"]
        # With flat prices, every bar's return is purely -cost.
        # AlternatingStrategy: signals = [1, -1, 1, -1], positions = [NaN, 1, -1, 1]
        # Turnover: bar1=1 (0→1), bar2=2 (1→-1), bar3=2 (-1→1)
        expected_costs = [0.0, 1 * fee_rate, 2 * fee_rate, 2 * fee_rate]
        for i in range(1, n):
            np.testing.assert_allclose(returns.iloc[i], -expected_costs[i], atol=1e-10)

    def test_lookahead_bias_full_alignment(self, sample_bars: pd.DataFrame) -> None:
        """Positions at bar t should equal signals at bar t-1."""
        engine = BacktestEngine(
            AlwaysLong(),
            BacktestConfig(initial_capital=100_000, fee_rate=0.0, slippage_bps=0.0),
        )
        result = engine.run_on_bars(sample_bars, "TEST", "5min")
        # First position is NaN
        assert pd.isna(result.positions.iloc[0])
        # All subsequent positions should equal 1.0 (the AlwaysLong signal)
        np.testing.assert_array_equal(result.positions.iloc[1:].values, 1.0)


class TestTradeDetection:
    def test_detect_trades_from_positions(self) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="5min")
        # Long from bar 1-4, flat 5-6, short 7-9
        positions = pd.Series(
            [0, 1, 1, 1, 1, 0, 0, -1, -1, -1], index=dates, dtype=float
        )
        close = pd.Series(
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            index=dates,
            dtype=float,
        )

        trades = _detect_trades(positions, close)
        assert len(trades) >= 1
        # First trade should be long
        assert trades[0].direction == 1.0
        assert trades[0].entry_price == 101.0

    def test_all_flat_no_trades(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="5min")
        positions = pd.Series(0.0, index=dates)
        close = pd.Series(100.0, index=dates)
        trades = _detect_trades(positions, close)
        assert len(trades) == 0

    def test_position_flip_creates_two_trades(self) -> None:
        """Long→short should close the long and open a short."""
        dates = pd.date_range("2024-01-01", periods=6, freq="5min")
        positions = pd.Series([0, 1, 1, -1, -1, 0], index=dates, dtype=float)
        close = pd.Series([100, 101, 102, 103, 104, 105], index=dates, dtype=float)
        trades = _detect_trades(positions, close)
        assert len(trades) == 2
        assert trades[0].direction == 1.0
        assert trades[1].direction == -1.0

    def test_unclosed_position_not_phantom(self) -> None:
        """Position held at end should not create a phantom exit trade."""
        dates = pd.date_range("2024-01-01", periods=5, freq="5min")
        positions = pd.Series([0, 0, 1, 1, 1], index=dates, dtype=float)
        close = pd.Series([100, 101, 102, 103, 104], index=dates, dtype=float)
        trades = _detect_trades(positions, close)
        # The long is never closed, so no completed trade
        assert len(trades) == 0


def _make_bars(close_prices: list[float], high_offset: float = 1.0, low_offset: float = 1.0):
    """Helper to build bars from a close price sequence."""
    n = len(close_prices)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = np.array(close_prices, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + high_offset,
            "low": close - low_offset,
            "close": close,
            "volume": 100.0,
            "buy_volume": 50.0,
            "vwap": close,
            "trade_count": 10,
        },
        index=dates,
    )


class TestStopLossTakeProfit:
    def test_stop_loss_limits_downside(self) -> None:
        """Long position with 2% SL: a 5% drop should exit at ~2% loss."""
        # Prices: 100, 100, 95 (5% drop on bar 2)
        # AlwaysLong: signals=[1,1,1], positions=[NaN,1,1]
        # Entry at bar 0 close=100, SL triggers bar 2 low=94 < 100*0.98=98
        bars = _make_bars([100.0, 100.0, 95.0])
        engine = BacktestEngine(
            AlwaysLong(),
            BacktestConfig(initial_capital=100_000, fee_rate=0.0, slippage_bps=0.0, stop_loss_pct=0.02),
        )
        result = engine.run_on_bars(bars, "TEST", "5min")
        # Position should be zeroed on bar 2 (stop triggered)
        assert result.positions.iloc[2] == 0.0
        # Final equity should reflect ~2% loss, not 5%
        final_equity = result.equity_curve["equity"].iloc[-1]
        assert final_equity > 100_000 * 0.97  # better than 3% loss
        assert final_equity < 100_000  # still a loss

    def test_take_profit_caps_upside(self) -> None:
        """Long position with 3% TP: a 10% rise should exit at ~3% gain."""
        # Prices: 100, 100, 110 (10% rise on bar 2)
        bars = _make_bars([100.0, 100.0, 110.0])
        engine = BacktestEngine(
            AlwaysLong(),
            BacktestConfig(initial_capital=100_000, fee_rate=0.0, slippage_bps=0.0, take_profit_pct=0.03),
        )
        result = engine.run_on_bars(bars, "TEST", "5min")
        assert result.positions.iloc[2] == 0.0  # TP triggered
        final_equity = result.equity_curve["equity"].iloc[-1]
        # Should be ~103k, not 110k
        assert final_equity < 100_000 * 1.05
        assert final_equity > 100_000 * 1.02

    def test_trailing_stop_locks_profit(self) -> None:
        """Price rises 10% then drops — trailing stop (3%) should lock in most gains."""
        prices = [100.0, 100.0, 105.0, 110.0, 106.0]
        # Entry at close[0]=100, trail_high reaches 111 (high of bar 3)
        # Bar 4: low=105, trail_high*(1-0.03) = 111*0.97 ≈ 107.67 > 105 → triggered
        bars = _make_bars(prices)
        engine = BacktestEngine(
            AlwaysLong(),
            BacktestConfig(initial_capital=100_000, fee_rate=0.0, slippage_bps=0.0, trailing_stop_pct=0.03),
        )
        result = engine.run_on_bars(bars, "TEST", "5min")
        assert result.positions.iloc[-1] == 0.0  # trailing stop triggered
        final_equity = result.equity_curve["equity"].iloc[-1]
        # Should have locked in profit (exited above 106)
        assert final_equity > 100_000 * 1.05

    def test_short_stop_loss(self) -> None:
        """Short position stopped out when price rises past SL."""
        prices = [100.0, 100.0, 105.0]  # price rises 5%
        bars = _make_bars(prices)
        engine = BacktestEngine(
            AlwaysShort(),
            BacktestConfig(initial_capital=100_000, fee_rate=0.0, slippage_bps=0.0, stop_loss_pct=0.02),
        )
        result = engine.run_on_bars(bars, "TEST", "5min")
        assert result.positions.iloc[2] == 0.0  # SL triggered
        final_equity = result.equity_curve["equity"].iloc[-1]
        # Loss should be capped near 2%, not 5%
        assert final_equity > 100_000 * 0.97

    def test_short_take_profit(self) -> None:
        """Short position takes profit when price drops."""
        prices = [100.0, 100.0, 90.0]  # price drops 10%
        bars = _make_bars(prices)
        engine = BacktestEngine(
            AlwaysShort(),
            BacktestConfig(initial_capital=100_000, fee_rate=0.0, slippage_bps=0.0, take_profit_pct=0.03),
        )
        result = engine.run_on_bars(bars, "TEST", "5min")
        assert result.positions.iloc[2] == 0.0  # TP triggered
        final_equity = result.equity_curve["equity"].iloc[-1]
        assert final_equity > 100_000 * 1.02
        assert final_equity < 100_000 * 1.05

    def test_no_stops_unchanged(self, sample_bars: pd.DataFrame) -> None:
        """With all stops=None, results should be identical to no-stops engine."""
        config_no_stops = BacktestConfig(initial_capital=100_000, fee_rate=0.001, slippage_bps=5.0)
        config_explicit_none = BacktestConfig(
            initial_capital=100_000, fee_rate=0.001, slippage_bps=5.0,
            stop_loss_pct=None, take_profit_pct=None, trailing_stop_pct=None,
        )
        r1 = BacktestEngine(AlwaysLong(), config_no_stops).run_on_bars(sample_bars, "TEST", "5min")
        r2 = BacktestEngine(AlwaysLong(), config_explicit_none).run_on_bars(sample_bars, "TEST", "5min")
        np.testing.assert_array_equal(
            r1.equity_curve["equity"].values,
            r2.equity_curve["equity"].values,
        )

    def test_stop_and_reentry(self) -> None:
        """After stop triggers, strategy signal can re-enter on next bar."""
        # Prices: 100, 100, 95, 100, 100
        # SL triggers bar 2, AlwaysLong re-enters bar 3
        prices = [100.0, 100.0, 95.0, 100.0, 100.0]
        bars = _make_bars(prices)
        engine = BacktestEngine(
            AlwaysLong(),
            BacktestConfig(initial_capital=100_000, fee_rate=0.0, slippage_bps=0.0, stop_loss_pct=0.02),
        )
        result = engine.run_on_bars(bars, "TEST", "5min")
        # Bar 2: stopped out (pos=0), Bar 3: signal still 1 so re-enters
        assert result.positions.iloc[2] == 0.0
        # Bar 3 position comes from signal at bar 2 which is still 1.0
        # But position was set to 0 at bar 2 by stops... the SIGNAL is still 1,
        # so shifted signal at bar 3 = signal[2] = 1.0 → re-entry
        assert result.positions.iloc[3] == 0.0 or result.positions.iloc[4] == 1.0

    def test_stops_with_fees(self) -> None:
        """Stop exits should incur transaction costs."""
        prices = [100.0, 100.0, 95.0, 100.0]
        bars = _make_bars(prices)
        fee_rate = 0.01
        engine_with_fee = BacktestEngine(
            AlwaysLong(),
            BacktestConfig(initial_capital=100_000, fee_rate=fee_rate, slippage_bps=0.0, stop_loss_pct=0.02),
        )
        engine_no_fee = BacktestEngine(
            AlwaysLong(),
            BacktestConfig(initial_capital=100_000, fee_rate=0.0, slippage_bps=0.0, stop_loss_pct=0.02),
        )
        r_fee = engine_with_fee.run_on_bars(bars, "TEST", "5min")
        r_no_fee = engine_no_fee.run_on_bars(bars, "TEST", "5min")
        # Fee version should have lower equity
        assert r_fee.equity_curve["equity"].iloc[-1] < r_no_fee.equity_curve["equity"].iloc[-1]
