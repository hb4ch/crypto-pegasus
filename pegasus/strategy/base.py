from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Minimal strategy interface for AI alpha research.

    Subclasses implement *only* ``generate_signals``.  The backtest engine
    handles data loading, position tracking, cost modelling, and reporting.
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params: dict = params or {}

    @abstractmethod
    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        """Generate trading signals from OHLCV bars.

        Parameters
        ----------
        bars : pd.DataFrame
            DataFrame indexed by datetime with columns:
            open, high, low, close, volume, buy_volume, vwap, trade_count.

        Returns
        -------
        pd.Series
            Signal per bar, same index as *bars*:
            * ``1.0``  – fully long
            * ``-1.0`` – fully short
            * ``0.0``  – flat
            Fractional values (e.g. 0.5) control position sizing.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
