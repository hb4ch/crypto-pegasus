from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numba
import numpy as np
import pandas as pd

from pegasus.config import BacktestConfig
from pegasus.data.provider import DataProvider
from pegasus.strategy.base import Strategy


@dataclass
class Trade:
    """A single round-trip or position change."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: float  # +1 long, -1 short
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""

    symbol: str
    timeframe: str
    equity_curve: pd.DataFrame  # columns: equity, returns, drawdown
    positions: pd.Series
    signals: pd.Series
    trades: list[Trade] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    bars: pd.DataFrame | None = None


def _detect_trades(
    positions: pd.Series,
    close: pd.Series,
) -> list[Trade]:
    """Detect trades from position changes."""
    trades: list[Trade] = []
    pos = positions.dropna()
    if pos.empty:
        return trades

    # Find where position changes
    changes = pos.diff().fillna(pos)
    change_idx = changes[changes != 0].index

    current_entry: pd.Timestamp | None = None
    current_dir: float = 0.0
    current_entry_price: float = 0.0

    for ts in change_idx:
        new_pos = pos.loc[ts]
        price = close.loc[ts]

        # Close existing position if direction changed
        if current_entry is not None and current_dir != 0:
            if new_pos != current_dir:
                pnl = current_dir * (price - current_entry_price) / current_entry_price
                trades.append(
                    Trade(
                        entry_time=current_entry,
                        exit_time=ts,
                        direction=current_dir,
                        entry_price=current_entry_price,
                        exit_price=price,
                        pnl=pnl * current_entry_price,
                        return_pct=pnl,
                    )
                )
                current_entry = None
                current_dir = 0.0

        # Open new position
        if new_pos != 0 and (current_dir == 0 or current_entry is None):
            current_entry = ts
            current_dir = float(np.sign(new_pos))
            current_entry_price = price

    return trades


@numba.njit
def _apply_stops_numba(
    state_positions: np.ndarray,
    return_positions: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    exit_prices: np.ndarray,
    sl_pct: float,
    tp_pct: float,
    trail_pct: float,
) -> None:
    """Apply stop loss, take profit, and trailing stop to positions.

    Outputs two position arrays:
    - state_positions: zeroed on stop bars (for turnover / state tracking)
    - return_positions: keeps position on stop bars (for return calculation)

    Also writes the actual exit price into exit_prices.
    """
    entry_price = 0.0
    trail_high = 0.0
    trail_low = np.inf
    stopped_prev = False

    for i in range(len(state_positions)):
        pos = state_positions[i]

        # If previous bar was stopped, zero this bar's position
        # (the stop exited, so we're flat now unless a new entry signal)
        if stopped_prev:
            state_positions[i] = 0.0
            return_positions[i] = 0.0
            stopped_prev = False
            entry_price = 0.0
            trail_high = 0.0
            trail_low = np.inf
            exit_prices[i] = close[i]
            continue

        # No position — reset tracking state
        if np.isnan(pos) or pos == 0.0:
            entry_price = 0.0
            trail_high = 0.0
            trail_low = np.inf
            exit_prices[i] = close[i]
            return_positions[i] = pos
            continue

        # Detect new entry (position changed from flat or flipped direction)
        prev_pos = state_positions[i - 1] if i > 0 else 0.0
        if np.isnan(prev_pos):
            prev_pos = 0.0

        if prev_pos == 0.0 or np.sign(prev_pos) != np.sign(pos):
            # Entry price is the previous bar's close (position is shifted)
            entry_price = close[i - 1] if i > 0 else close[i]
            trail_high = high[i]
            trail_low = low[i]

        # Update trailing extremes
        if high[i] > trail_high:
            trail_high = high[i]
        if low[i] < trail_low:
            trail_low = low[i]

        triggered = False
        exit_p = close[i]

        if pos > 0:  # Long position
            if sl_pct > 0 and low[i] <= entry_price * (1 - sl_pct):
                exit_p = entry_price * (1 - sl_pct)
                triggered = True
            elif tp_pct > 0 and high[i] >= entry_price * (1 + tp_pct):
                exit_p = entry_price * (1 + tp_pct)
                triggered = True
            elif trail_pct > 0 and low[i] <= trail_high * (1 - trail_pct):
                exit_p = trail_high * (1 - trail_pct)
                triggered = True
        elif pos < 0:  # Short position
            if sl_pct > 0 and high[i] >= entry_price * (1 + sl_pct):
                exit_p = entry_price * (1 + sl_pct)
                triggered = True
            elif tp_pct > 0 and low[i] <= entry_price * (1 - tp_pct):
                exit_p = entry_price * (1 - tp_pct)
                triggered = True
            elif trail_pct > 0 and high[i] >= trail_low * (1 + trail_pct):
                exit_p = trail_low * (1 + trail_pct)
                triggered = True

        if triggered:
            state_positions[i] = 0.0    # Zero for turnover/state
            return_positions[i] = pos   # Keep for return calculation
            exit_prices[i] = exit_p
            stopped_prev = True
        else:
            return_positions[i] = pos
            exit_prices[i] = close[i]


def _apply_stops(
    positions: pd.Series,
    bars: pd.DataFrame,
    sl_pct: float | None,
    tp_pct: float | None,
    trail_pct: float | None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Apply stop loss / take profit / trailing stop to positions.

    Returns (state_positions, return_positions, effective_close):
    - state_positions: zeroed on stop bars (for turnover tracking)
    - return_positions: keeps position on stop bars (for return calculation)
    - effective_close: uses stop price on bars where a stop triggered
    """
    state_arr = positions.values.copy().astype(np.float64)
    return_arr = np.empty_like(state_arr)
    close_arr = bars["close"].values.astype(np.float64)
    high_arr = bars["high"].values.astype(np.float64)
    low_arr = bars["low"].values.astype(np.float64)
    exit_prices = np.empty_like(close_arr)

    _apply_stops_numba(
        state_arr, return_arr, close_arr, high_arr, low_arr, exit_prices,
        sl_pct or 0.0, tp_pct or 0.0, trail_pct or 0.0,
    )

    idx = positions.index
    return (
        pd.Series(state_arr, index=idx),
        pd.Series(return_arr, index=idx),
        pd.Series(exit_prices, index=idx),
    )


class BacktestEngine:
    """Vectorized backtest engine.

    Converts strategy signals into an equity curve using fully vectorized
    operations — no bar-by-bar loops. Optional stop loss / take profit /
    trailing stop are applied via a numba-accelerated pass.
    """

    def __init__(
        self,
        strategy: Strategy,
        config: BacktestConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()

    def run(
        self,
        symbol: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        timeframe: str | None = None,
    ) -> BacktestResult:
        """Run backtest for a single symbol."""
        start = start or self.config.start_date
        end = end or self.config.end_date
        timeframe = timeframe or self.config.timeframe

        with DataProvider(self.config) as dp:
            bars = dp.get_bars(symbol, start, end, timeframe)

        if bars.empty:
            raise ValueError(f"No data for {symbol} in [{start}, {end})")

        return self._run_on_bars(bars, symbol, timeframe)

    def run_on_bars(
        self,
        bars: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "5min",
    ) -> BacktestResult:
        """Run backtest on pre-loaded bars (useful for testing)."""
        return self._run_on_bars(bars, symbol, timeframe)

    def _run_on_bars(
        self,
        bars: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> BacktestResult:
        fee_rate = self.config.fee_rate
        slippage_bps = self.config.slippage_bps
        initial_capital = self.config.initial_capital

        # 1. Generate signals
        signals = self.strategy.generate_signals(bars)

        # 2. Positions: shift signals by 1 to avoid lookahead bias
        positions = signals.shift(1)

        # 2b. Apply stop loss / take profit / trailing stop (if configured)
        sl = self.config.stop_loss_pct
        tp = self.config.take_profit_pct
        trail = self.config.trailing_stop_pct
        if sl is not None or tp is not None or trail is not None:
            state_positions, return_positions, effective_close = _apply_stops(
                positions, bars, sl, tp, trail,
            )
        else:
            state_positions = positions
            return_positions = positions
            effective_close = bars["close"]

        # 3. Price returns (uses effective_close which accounts for stop prices)
        price_returns = effective_close.pct_change()

        # 4. Strategy returns: use return_positions (keeps position on stop bars)
        strategy_returns = return_positions * price_returns

        # 5. Transaction costs: use state_positions (zeroed on stop bars) for turnover
        # fillna(0) before diff so first entry (NaN→position) incurs cost
        prev_positions = state_positions.shift(1).fillna(0)
        curr_positions = state_positions.fillna(0)
        turnover = (curr_positions - prev_positions).abs()
        cost_per_unit = fee_rate + slippage_bps / 10_000
        costs = turnover * cost_per_unit

        # 6. Net returns
        net_returns = strategy_returns - costs

        # 7. Equity curve
        equity = (1 + net_returns.fillna(0)).cumprod() * initial_capital

        # 8. Drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        equity_df = pd.DataFrame(
            {
                "equity": equity,
                "returns": net_returns,
                "drawdown": drawdown,
            },
            index=bars.index,
        )

        # 9. Detect trades (use effective_close for accurate stop exit prices)
        trades = _detect_trades(state_positions, effective_close)

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            equity_curve=equity_df,
            positions=state_positions,
            signals=signals,
            trades=trades,
            bars=bars,
        )

    def run_multi(
        self,
        symbols: list[str] | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        timeframe: str | None = None,
    ) -> dict[str, BacktestResult]:
        """Run backtest across multiple symbols independently."""
        symbols = symbols or self.config.symbols
        results: dict[str, BacktestResult] = {}
        for sym in symbols:
            results[sym] = self.run(sym, start, end, timeframe)
        return results
