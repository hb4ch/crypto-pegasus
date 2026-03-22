"""Tests for pegasus.data.provider.DataProvider."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pegasus.config import BacktestConfig
from pegasus.data.provider import DataProvider


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    """Create a small parquet dataset mimicking the solana-pegasus layout."""
    symbol = "TESTUSDT"
    base = tmp_path / symbol / "year=2024" / "month=01"
    base.mkdir(parents=True)

    n = 1000
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1s", tz="UTC")
    df = pd.DataFrame(
        {
            "agg_trade_id": range(n),
            "price": 2000.0 + rng.standard_normal(n) * 10,
            "quantity": rng.uniform(0.01, 1.0, n).astype(np.float32),
            "first_trade_id": range(n),
            "last_trade_id": range(n),
            "timestamp": timestamps.astype(np.int64) // 1000,  # μs
            "is_buyer_maker": rng.choice([True, False], n),
            "is_best_match": [True] * n,
            "datetime": timestamps,
            "source_date": pd.to_datetime("2024-01-01").date(),
        }
    )
    df.to_parquet(base / "data.parquet", index=False)
    return tmp_path


@pytest.fixture
def provider(sample_parquet: Path) -> DataProvider:
    config = BacktestConfig(data_root=sample_parquet)
    return DataProvider(config)


class TestDataProvider:
    def test_get_symbols(self, provider: DataProvider) -> None:
        symbols = provider.get_symbols()
        assert "TESTUSDT" in symbols

    def test_get_bars_returns_ohlcv(self, provider: DataProvider) -> None:
        bars = provider.get_bars(
            "TESTUSDT", "2024-01-01", "2024-01-02", timeframe="5min"
        )
        assert not bars.empty
        expected_cols = {"open", "high", "low", "close", "volume", "buy_volume", "vwap", "trade_count"}
        assert expected_cols.issubset(set(bars.columns))
        assert bars.index.name == "datetime"

    def test_ohlcv_invariants(self, provider: DataProvider) -> None:
        bars = provider.get_bars(
            "TESTUSDT", "2024-01-01", "2024-01-02", timeframe="5min"
        )
        # high >= open, close, low
        assert (bars["high"] >= bars["open"]).all()
        assert (bars["high"] >= bars["close"]).all()
        assert (bars["high"] >= bars["low"]).all()
        # low <= open, close
        assert (bars["low"] <= bars["open"]).all()
        assert (bars["low"] <= bars["close"]).all()
        # volume > 0
        assert (bars["volume"] > 0).all()

    def test_date_filtering(self, provider: DataProvider) -> None:
        bars = provider.get_bars(
            "TESTUSDT", "2024-01-01 00:00:00", "2024-01-01 00:05:00", timeframe="1min"
        )
        assert len(bars) <= 5

    def test_invalid_timeframe_raises(self, provider: DataProvider) -> None:
        with pytest.raises(ValueError, match="Unknown timeframe"):
            provider.get_bars("TESTUSDT", "2024-01-01", "2024-01-02", timeframe="3min")

    def test_empty_result_for_out_of_range(self, provider: DataProvider) -> None:
        bars = provider.get_bars(
            "TESTUSDT", "2025-06-01", "2025-07-01", timeframe="5min"
        )
        assert bars.empty

    def test_context_manager(self, sample_parquet: Path) -> None:
        config = BacktestConfig(data_root=sample_parquet)
        with DataProvider(config) as dp:
            assert dp.get_symbols() == ["TESTUSDT"]

    def test_get_date_range(self, provider: DataProvider) -> None:
        mn, mx = provider.get_date_range("TESTUSDT")
        assert mn < mx
        # Our sample data starts at 2024-01-01
        assert mn.year == 2024

    def test_get_bars_with_datetime_objects(self, provider: DataProvider) -> None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)
        bars = provider.get_bars("TESTUSDT", start, end, timeframe="5min")
        assert not bars.empty

    def test_get_bars_1h_timeframe(self, provider: DataProvider) -> None:
        bars = provider.get_bars("TESTUSDT", "2024-01-01", "2024-01-02", timeframe="1h")
        assert not bars.empty
        # 1000 seconds of data → should be 1 bar at 1h resolution
        assert len(bars) <= 1

    def test_default_config(self, sample_parquet: Path) -> None:
        """DataProvider with explicit config should not crash."""
        config = BacktestConfig(data_root=sample_parquet)
        dp = DataProvider(config)
        assert dp.config is not None
        dp.close()
