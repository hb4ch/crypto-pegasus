from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from pegasus.config import BacktestConfig

# Mapping from user-friendly timeframe strings to DuckDB interval syntax.
def _hive_partition_filter(start: str | datetime, end: str | datetime) -> str:
    """SQL predicate on hive year/month columns for partition pruning."""
    s = datetime.fromisoformat(str(start))
    e = datetime.fromisoformat(str(end))
    pairs = []
    cur = s.replace(day=1)
    while cur < e:
        pairs.append((cur.year, cur.month))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    if not pairs:
        return "TRUE"
    clauses = ", ".join(f"({y}, '{m:02d}')" for y, m in pairs)
    return f"(year, month) IN ({clauses})"


_TIMEFRAME_MAP = {
    "1min": "1 minute",
    "5min": "5 minutes",
    "15min": "15 minutes",
    "30min": "30 minutes",
    "1h": "1 hour",
    "4h": "4 hours",
    "1d": "1 day",
}


class DataProvider:
    """DuckDB-backed provider that reads aggTrades parquet and returns OHLCV bars."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.data_root = self.config.data_root
        self.conn = duckdb.connect()
        self.conn.execute(f"SET threads TO {self.config.duckdb_threads}")
        self.conn.execute(f"SET memory_limit = '{self.config.duckdb_memory_limit}'")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> DataProvider:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
        timeframe: str = "5min",
    ) -> pd.DataFrame:
        """Load aggTrades parquet for *symbol* and resample to OHLCV bars.

        Returns a DataFrame indexed by ``bar_time`` with columns:
        open, high, low, close, volume, buy_volume, vwap, trade_count.

        When ``bar_cache_dir`` is configured, bars are cached as parquet
        files keyed by symbol and timeframe.  The first call aggregates
        from raw ticks and writes the cache; subsequent calls read from
        the cache and slice by [start, end).
        """
        interval = _TIMEFRAME_MAP.get(timeframe)
        if interval is None:
            raise ValueError(
                f"Unknown timeframe {timeframe!r}. Choose from {list(_TIMEFRAME_MAP)}"
            )

        cache_dir = self.config.bar_cache_dir
        if cache_dir is not None:
            cache_path = cache_dir / f"{symbol}_{timeframe}.parquet"
            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                df.index.name = "datetime"
                return df.loc[str(start):str(end)]

        # Aggregate from raw ticks
        df = self._aggregate_bars(symbol, start, end, interval, timeframe)

        return df

    def materialize(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
        timeframe: str = "5min",
    ) -> Path:
        """Aggregate bars from raw ticks and write to the bar cache.

        Returns the cache file path.  Subsequent ``get_bars()`` calls
        for the same symbol/timeframe will read from cache.
        """
        interval = _TIMEFRAME_MAP.get(timeframe)
        if interval is None:
            raise ValueError(
                f"Unknown timeframe {timeframe!r}. Choose from {list(_TIMEFRAME_MAP)}"
            )

        cache_dir = self.config.bar_cache_dir
        if cache_dir is None:
            raise ValueError("bar_cache_dir is not configured")

        df = self._aggregate_bars(symbol, start, end, interval, timeframe)

        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{symbol}_{timeframe}.parquet"
        df.to_parquet(cache_path)
        return cache_path

    def _aggregate_bars(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
        interval: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """Run the DuckDB aggregation query on raw tick parquet."""
        parquet_glob = str(self.data_root / symbol / "year=*/month=*/data.parquet")

        partition_filter = _hive_partition_filter(start, end)

        query = f"""
            SELECT
                time_bucket(INTERVAL '{interval}', datetime) AS bar_time,
                first(price ORDER BY datetime)                AS open,
                max(price)                                    AS high,
                min(price)                                    AS low,
                last(price ORDER BY datetime)                 AS close,
                sum(quantity)                                  AS volume,
                sum(CASE WHEN NOT is_buyer_maker
                         THEN quantity ELSE 0 END)            AS buy_volume,
                sum(price * quantity) / sum(quantity)          AS vwap,
                count(*)::INTEGER                             AS trade_count
            FROM read_parquet('{parquet_glob}', hive_partitioning=true)
            WHERE {partition_filter}
              AND datetime >= $start AND datetime < $end
            GROUP BY bar_time
            ORDER BY bar_time
        """

        df: pd.DataFrame = self.conn.execute(
            query, {"start": str(start), "end": str(end)}
        ).fetchdf()

        if df.empty:
            return df

        df = df.set_index("bar_time")
        df.index.name = "datetime"
        return df

    def get_symbols(self) -> list[str]:
        """Discover available symbols from the directory structure."""
        return sorted(
            p.name
            for p in self.data_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    def get_date_range(self, symbol: str) -> tuple[datetime, datetime]:
        """Return (min, max) datetime for a symbol."""
        parquet_glob = str(self.data_root / symbol / "year=*/month=*/data.parquet")
        row = self.conn.execute(
            f"""
            SELECT min(datetime) AS mn, max(datetime) AS mx
            FROM read_parquet('{parquet_glob}', hive_partitioning=true)
            """
        ).fetchone()
        if row is None:
            raise FileNotFoundError(f"No data found for {symbol}")
        return row[0], row[1]
