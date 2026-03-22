from __future__ import annotations

import numpy as np
import pandas as pd

from pegasus.strategy.base import Strategy


class SMACrossStrategy(Strategy):
    """Simple moving-average crossover.

    Goes long when the fast SMA crosses above the slow SMA, short when below.

    Parameters
    ----------
    params : dict
        ``fast`` – fast window (default 10)
        ``slow`` – slow window (default 50)
    """

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        fast = bars["close"].rolling(self.params.get("fast", 10)).mean()
        slow = bars["close"].rolling(self.params.get("slow", 50)).mean()

        signals = pd.Series(0.0, index=bars.index)
        signals[fast > slow] = 1.0
        signals[fast < slow] = -1.0
        # Warmup period: no signal until slow SMA is valid.
        signals.iloc[: self.params.get("slow", 50) - 1] = np.nan
        return signals
