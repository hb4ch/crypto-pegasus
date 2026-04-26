"""Readers for the multi-source ETL emitted by solana-pegasus.

Layout (post-2026-04 ETL refactor)::

    {root}/{provider}/{dataset}/{source}/{key}/year=*/month=*/data.parquet
    {root}/{provider}/openInterest/{source}/{symbol}/interval=*/year=*/month=*/data.parquet

Each method returns a pandas ``DataFrame`` indexed by ``datetime`` (UTC).
Source schemas are documented per method.

These datasets are small enough (KB to low MB per month) that we read them
directly through DuckDB on every call without an intermediate cache layer.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from pegasus.data.provider import _hive_partition_filter

_DEFAULT_ROOT = Path.home() / "solana-pegasus/data/parquet"

# Symbols supported per data source. Funding rate / open interest are only
# emitted for the major USD-M futures pairs; raise loudly for unsupported keys.
_FUTURES_SYMBOLS = {"BTCUSDT", "ETHUSDT"}


class MultiSourceProvider:
    """DuckDB-backed reader for the non-OHLCV datasets.

    Bars (aggTrades) live in ``DataProvider``. Everything else — funding rate,
    open interest, DefiLlama TVL/volume/stablecoin — lives here.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else _DEFAULT_ROOT
        self.conn = duckdb.connect()
        self.conn.execute("SET threads TO 8")
        self.conn.execute("SET memory_limit = '4GB'")
        # Force UTC for all timestamp arithmetic so callers get consistent
        # timezones regardless of the host's local zone.
        self.conn.execute("SET TimeZone = 'UTC'")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MultiSourceProvider":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Binance USD-M futures
    # ------------------------------------------------------------------

    def get_funding_rate(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
    ) -> pd.DataFrame:
        """Funding rate snapshots (8h cadence on Binance USD-M).

        Available symbols: BTCUSDT, ETHUSDT.
        Coverage: 2022-04 → present.

        Returns a DataFrame indexed by ``datetime`` (UTC) with columns:
        ``last_funding_rate`` (float), ``funding_interval_hours`` (int).
        """
        if symbol not in _FUTURES_SYMBOLS:
            raise ValueError(
                f"Funding rate is only available for {sorted(_FUTURES_SYMBOLS)}; "
                f"got {symbol!r}."
            )
        glob = (
            f"{self.root}/binance/fundingRate/binance_funding_rate/"
            f"{symbol}/year=*/month=*/data.parquet"
        )
        partition = _hive_partition_filter(start, end)
        query = f"""
            SELECT datetime,
                   last_funding_rate,
                   funding_interval_hours
            FROM read_parquet('{glob}', hive_partitioning=true)
            WHERE {partition}
              AND datetime >= $start AND datetime < $end
            ORDER BY datetime
        """
        df = self.conn.execute(
            query, {"start": str(start), "end": str(end)}
        ).fetchdf()
        return _index_by_datetime(df)

    def get_open_interest(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """Open interest snapshots from Binance USD-M (1h cadence).

        Available symbols: BTCUSDT, ETHUSDT.

        IMPORTANT: open interest is fetched via REST pagination, so history
        is bounded by Binance's API window — typically only the last ~30 days
        are available. Strategies that need long history will fail here.

        Returns a DataFrame indexed by ``datetime`` (UTC) with columns:
        ``sum_open_interest`` (contracts), ``sum_open_interest_value`` (USD).
        """
        if symbol not in _FUTURES_SYMBOLS:
            raise ValueError(
                f"Open interest is only available for {sorted(_FUTURES_SYMBOLS)}; "
                f"got {symbol!r}."
            )
        glob = (
            f"{self.root}/binance/openInterest/binance_open_interest/"
            f"{symbol}/interval={interval}/year=*/month=*/data.parquet"
        )
        partition = _hive_partition_filter(start, end)
        query = f"""
            SELECT datetime,
                   sum_open_interest,
                   sum_open_interest_value
            FROM read_parquet('{glob}', hive_partitioning=true)
            WHERE {partition}
              AND datetime >= $start AND datetime < $end
            ORDER BY datetime
        """
        df = self.conn.execute(
            query, {"start": str(start), "end": str(end)}
        ).fetchdf()
        return _index_by_datetime(df)

    # ------------------------------------------------------------------
    # DefiLlama
    # ------------------------------------------------------------------

    def get_chain_tvl(
        self,
        chain: str,
        start: str | datetime,
        end: str | datetime,
    ) -> pd.DataFrame:
        """Daily total value locked per chain.

        Available chains: Arbitrum, Base, Ethereum, Solana.
        Coverage: 2022-04 → present.

        Returns a DataFrame indexed by ``datetime`` (UTC, daily) with column
        ``tvl`` (USD).
        """
        glob = (
            f"{self.root}/defillama/chainTvl/defillama_chain_tvl/"
            f"{chain}/year=*/month=*/data.parquet"
        )
        partition = _hive_partition_filter(start, end)
        query = f"""
            SELECT datetime, tvl
            FROM read_parquet('{glob}', hive_partitioning=true)
            WHERE {partition}
              AND datetime >= $start AND datetime < $end
            ORDER BY datetime
        """
        df = self.conn.execute(
            query, {"start": str(start), "end": str(end)}
        ).fetchdf()
        return _index_by_datetime(df)

    def get_protocol_tvl(
        self,
        protocol: str,
        start: str | datetime,
        end: str | datetime,
        chain: str = "Total",
    ) -> pd.DataFrame:
        """Daily TVL for a protocol on a specific chain.

        ``chain`` is a value INSIDE the parquet (column), not a partition key.
        Use ``chain="Total"`` for the protocol's aggregate TVL across all chains.

        Available protocols: aave, lido, uniswap.
        """
        glob = (
            f"{self.root}/defillama/protocolTvl/defillama_protocol_tvl/"
            f"{protocol}/year=*/month=*/data.parquet"
        )
        partition = _hive_partition_filter(start, end)
        query = f"""
            SELECT datetime, tvl
            FROM read_parquet('{glob}', hive_partitioning=true)
            WHERE {partition}
              AND chain = $chain
              AND datetime >= $start AND datetime < $end
            ORDER BY datetime
        """
        df = self.conn.execute(
            query,
            {"start": str(start), "end": str(end), "chain": chain},
        ).fetchdf()
        return _index_by_datetime(df)

    def get_dex_volume(
        self,
        chain: str,
        start: str | datetime,
        end: str | datetime,
        protocol: str = "Total",
    ) -> pd.DataFrame:
        """Daily DEX trading volume per chain.

        ``protocol`` is a value inside the parquet; ``"Total"`` = aggregate.
        Available chains: Arbitrum, Base, Ethereum, Solana.
        """
        glob = (
            f"{self.root}/defillama/dexVolume/defillama_dex_volume/"
            f"{chain}/year=*/month=*/data.parquet"
        )
        partition = _hive_partition_filter(start, end)
        query = f"""
            SELECT datetime, volume_usd
            FROM read_parquet('{glob}', hive_partitioning=true)
            WHERE {partition}
              AND protocol = $protocol
              AND datetime >= $start AND datetime < $end
            ORDER BY datetime
        """
        df = self.conn.execute(
            query,
            {"start": str(start), "end": str(end), "protocol": protocol},
        ).fetchdf()
        return _index_by_datetime(df)

    def get_stablecoin(
        self,
        chain: str,
        start: str | datetime,
        end: str | datetime,
        stablecoin: str = "peggedUSD",
    ) -> pd.DataFrame:
        """Daily stablecoin supply on a chain, filtered to one peg type.

        ``stablecoin`` is a column value, e.g. "peggedUSD" (default — by far
        the largest), "peggedEUR", "peggedJPY", etc.
        """
        glob = (
            f"{self.root}/defillama/stablecoin/defillama_stablecoin/"
            f"{chain}/year=*/month=*/data.parquet"
        )
        partition = _hive_partition_filter(start, end)
        query = f"""
            SELECT datetime,
                   circulating_usd,
                   minted_usd,
                   bridged_usd
            FROM read_parquet('{glob}', hive_partitioning=true)
            WHERE {partition}
              AND stablecoin = $stablecoin
              AND datetime >= $start AND datetime < $end
            ORDER BY datetime
        """
        df = self.conn.execute(
            query,
            {"start": str(start), "end": str(end), "stablecoin": stablecoin},
        ).fetchdf()
        return _index_by_datetime(df)


def _index_by_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.set_index("datetime")
    df.index.name = "datetime"
    return df
