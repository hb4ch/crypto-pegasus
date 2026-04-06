from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class BacktestConfig(BaseModel):
    """Centralized backtest configuration."""

    # Data source
    data_root: Path = Path.home() / "solana-pegasus/data/parquet/spot/aggTrades"
    symbols: list[str] = ["ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT"]

    # Backtest parameters
    start_date: str = "2024-01-01"
    end_date: str = "2026-03-01"
    timeframe: str = "5min"

    # Portfolio
    initial_capital: float = 100_000.0
    fee_rate: float = 0.001  # 10 bps per side
    slippage_bps: float = 5.0

    # Risk management (None = disabled)
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None

    # DuckDB resource limits
    duckdb_threads: int = 16
    duckdb_memory_limit: str = "16GB"
