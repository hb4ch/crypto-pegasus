"""Example strategy file — AI agent edits THIS file.

Implement ``generate_signals`` and the platform handles the rest:
data loading, execution, fees, metrics, and reporting.
"""
from __future__ import annotations

import pandas as pd

from pegasus.strategy.base import Strategy


class MyAlpha(Strategy):
    """Momentum strategy: go long when close is above 20-bar high, short when below 20-bar low."""

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 20)
        upper = bars["high"].rolling(period).max()
        lower = bars["low"].rolling(period).min()

        signals = pd.Series(0.0, index=bars.index)
        signals[bars["close"] >= upper] = 1.0
        signals[bars["close"] <= lower] = -1.0
        return signals
