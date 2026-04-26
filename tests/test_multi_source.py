"""Tests for pegasus.data.multi_source.MultiSourceProvider.

Hermetic tests build synthetic parquet datasets matching the solana-pegasus
multi-source ETL layout. The final block runs lightweight smoke checks
against the real on-disk data when present.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from pegasus.data.multi_source import MultiSourceProvider, _DEFAULT_ROOT


# ---------------------------------------------------------------------------
# Synthetic parquet builders (one per dataset schema)
# ---------------------------------------------------------------------------


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _write_funding_rate(root: Path, symbol: str, year: int, month: int) -> None:
    n = 90
    times = pd.date_range(
        f"{year}-{month:02d}-01", periods=n, freq="8h", tz="UTC"
    )
    df = pd.DataFrame({
        "calc_time": times.astype("int64") // 10**6,
        "funding_interval_hours": [8] * n,
        "last_funding_rate": [0.0001 * (i % 5 - 2) for i in range(n)],
        "datetime": times,
        "source_date": [str(date(year, month, 1))] * n,
    })
    p = root / "binance" / "fundingRate" / "binance_funding_rate" / symbol \
        / f"year={year}" / f"month={month:02d}" / "data.parquet"
    _write_parquet(p, df)


def _write_open_interest(root: Path, symbol: str, year: int, month: int) -> None:
    n = 24 * 5
    times = pd.date_range(
        f"{year}-{month:02d}-01", periods=n, freq="1h", tz="UTC"
    )
    df = pd.DataFrame({
        "symbol": [symbol] * n,
        "sum_open_interest": [50000.0 + i for i in range(n)],
        "sum_open_interest_value": [3.5e9 + i * 1e6 for i in range(n)],
        "cmc_circulating_supply": [2e7] * n,
        "timestamp": times.astype("int64") // 10**6,
        "datetime": times,
        "source_date": [str(date(year, month, 1))] * n,
    })
    p = root / "binance" / "openInterest" / "binance_open_interest" / symbol \
        / "interval=1h" / f"year={year}" / f"month={month:02d}" / "data.parquet"
    _write_parquet(p, df)


def _write_chain_tvl(root: Path, chain: str, year: int, month: int) -> None:
    n = 30
    times = pd.date_range(
        f"{year}-{month:02d}-01", periods=n, freq="1D", tz="UTC"
    )
    df = pd.DataFrame({
        "date": [str(d) for d in times.date],
        "chain": [chain] * n,
        "tvl": [1e11 + i * 1e8 for i in range(n)],
        "datetime": times,
        "source_date": [str(date(year, month, 1))] * n,
    })
    p = root / "defillama" / "chainTvl" / "defillama_chain_tvl" / chain \
        / f"year={year}" / f"month={month:02d}" / "data.parquet"
    _write_parquet(p, df)


def _write_protocol_tvl(root: Path, protocol: str, year: int, month: int) -> None:
    rows = []
    times = pd.date_range(
        f"{year}-{month:02d}-01", periods=30, freq="1D", tz="UTC"
    )
    for chain_val in ("Total", "Ethereum", "Arbitrum"):
        for i, t in enumerate(times):
            rows.append({
                "date": str(t.date()),
                "protocol": protocol,
                "chain": chain_val,
                "tvl": 1e10 if chain_val == "Total" else 5e9,
                "datetime": t,
                "source_date": str(date(year, month, 1)),
            })
    df = pd.DataFrame(rows)
    p = root / "defillama" / "protocolTvl" / "defillama_protocol_tvl" / protocol \
        / f"year={year}" / f"month={month:02d}" / "data.parquet"
    _write_parquet(p, df)


def _write_dex_volume(root: Path, chain: str, year: int, month: int) -> None:
    rows = []
    times = pd.date_range(
        f"{year}-{month:02d}-01", periods=30, freq="1D", tz="UTC"
    )
    for proto_val in ("Total", "uniswap"):
        for i, t in enumerate(times):
            rows.append({
                "date": str(t.date()),
                "chain": chain,
                "protocol": proto_val,
                "volume_usd": 1e9 if proto_val == "Total" else 4e8,
                "datetime": t,
                "source_date": str(date(year, month, 1)),
            })
    df = pd.DataFrame(rows)
    p = root / "defillama" / "dexVolume" / "defillama_dex_volume" / chain \
        / f"year={year}" / f"month={month:02d}" / "data.parquet"
    _write_parquet(p, df)


def _write_stablecoin(root: Path, chain: str, year: int, month: int) -> None:
    rows = []
    times = pd.date_range(
        f"{year}-{month:02d}-01", periods=30, freq="1D", tz="UTC"
    )
    for sc_kind in ("peggedUSD", "peggedEUR"):
        for i, t in enumerate(times):
            rows.append({
                "date": str(t.date()),
                "chain": chain,
                "stablecoin": sc_kind,
                "circulating_usd": 1e11 if sc_kind == "peggedUSD" else 1e8,
                "minted_usd": 1e11 if sc_kind == "peggedUSD" else 1e8,
                "bridged_usd": 0.0,
                "datetime": t,
                "source_date": str(date(year, month, 1)),
            })
    df = pd.DataFrame(rows)
    p = root / "defillama" / "stablecoin" / "defillama_stablecoin" / chain \
        / f"year={year}" / f"month={month:02d}" / "data.parquet"
    _write_parquet(p, df)


# ---------------------------------------------------------------------------
# Synthetic-data fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    """Build a complete synthetic ETL tree for 2024-06."""
    root = tmp_path / "parquet"
    _write_funding_rate(root, "BTCUSDT", 2024, 6)
    _write_funding_rate(root, "ETHUSDT", 2024, 6)
    _write_open_interest(root, "BTCUSDT", 2024, 6)
    _write_chain_tvl(root, "Ethereum", 2024, 6)
    _write_protocol_tvl(root, "aave", 2024, 6)
    _write_dex_volume(root, "Ethereum", 2024, 6)
    _write_stablecoin(root, "Ethereum", 2024, 6)
    return root


# ---------------------------------------------------------------------------
# Hermetic tests
# ---------------------------------------------------------------------------


def test_funding_rate_returns_indexed_df(synthetic_root: Path) -> None:
    with MultiSourceProvider(root=synthetic_root) as p:
        df = p.get_funding_rate("BTCUSDT", "2024-06-01", "2024-07-01")
    assert df.index.name == "datetime"
    assert set(df.columns) == {"last_funding_rate", "funding_interval_hours"}
    assert len(df) > 0
    assert df.index.is_monotonic_increasing


def test_funding_rate_rejects_unsupported_symbol(synthetic_root: Path) -> None:
    with MultiSourceProvider(root=synthetic_root) as p:
        with pytest.raises(ValueError, match="Funding rate is only available"):
            p.get_funding_rate("SOLUSDT", "2024-06-01", "2024-07-01")


def test_funding_rate_filters_by_date_range(synthetic_root: Path) -> None:
    with MultiSourceProvider(root=synthetic_root) as p:
        full = p.get_funding_rate("BTCUSDT", "2024-06-01", "2024-07-01")
        partial = p.get_funding_rate("BTCUSDT", "2024-06-15", "2024-06-20")
    assert len(partial) < len(full)
    assert partial.index.min() >= pd.Timestamp("2024-06-15", tz="UTC")
    assert partial.index.max() < pd.Timestamp("2024-06-20", tz="UTC")


def test_open_interest_returns_indexed_df(synthetic_root: Path) -> None:
    with MultiSourceProvider(root=synthetic_root) as p:
        df = p.get_open_interest("BTCUSDT", "2024-06-01", "2024-07-01")
    assert df.index.name == "datetime"
    assert "sum_open_interest" in df.columns
    assert "sum_open_interest_value" in df.columns
    assert len(df) > 0


def test_open_interest_rejects_unsupported_symbol(synthetic_root: Path) -> None:
    with MultiSourceProvider(root=synthetic_root) as p:
        with pytest.raises(ValueError, match="Open interest is only available"):
            p.get_open_interest("BNBUSDT", "2024-06-01", "2024-07-01")


def test_chain_tvl_returns_indexed_df(synthetic_root: Path) -> None:
    with MultiSourceProvider(root=synthetic_root) as p:
        df = p.get_chain_tvl("Ethereum", "2024-06-01", "2024-07-01")
    assert df.index.name == "datetime"
    assert list(df.columns) == ["tvl"]
    assert len(df) == 30


def test_protocol_tvl_filters_by_chain(synthetic_root: Path) -> None:
    with MultiSourceProvider(root=synthetic_root) as p:
        total = p.get_protocol_tvl("aave", "2024-06-01", "2024-07-01", chain="Total")
        eth = p.get_protocol_tvl("aave", "2024-06-01", "2024-07-01", chain="Ethereum")
    # Both chains should yield 30 daily rows in the synthetic data
    assert len(total) == 30
    assert len(eth) == 30
    # Synthetic Total uses 1e10, Ethereum uses 5e9 — confirm filter applied
    assert total["tvl"].iloc[0] != eth["tvl"].iloc[0]


def test_dex_volume_filters_by_protocol(synthetic_root: Path) -> None:
    with MultiSourceProvider(root=synthetic_root) as p:
        total = p.get_dex_volume("Ethereum", "2024-06-01", "2024-07-01", protocol="Total")
        uni = p.get_dex_volume("Ethereum", "2024-06-01", "2024-07-01", protocol="uniswap")
    assert len(total) == 30
    assert len(uni) == 30
    assert total["volume_usd"].iloc[0] != uni["volume_usd"].iloc[0]


def test_stablecoin_filters_by_peg_type(synthetic_root: Path) -> None:
    with MultiSourceProvider(root=synthetic_root) as p:
        usd = p.get_stablecoin("Ethereum", "2024-06-01", "2024-07-01", stablecoin="peggedUSD")
        eur = p.get_stablecoin("Ethereum", "2024-06-01", "2024-07-01", stablecoin="peggedEUR")
    assert len(usd) == 30
    assert len(eur) == 30
    assert usd["circulating_usd"].iloc[0] != eur["circulating_usd"].iloc[0]


def test_empty_range_returns_empty_dataframe(synthetic_root: Path) -> None:
    with MultiSourceProvider(root=synthetic_root) as p:
        df = p.get_chain_tvl("Ethereum", "2099-01-01", "2099-02-01")
    assert df.empty


# ---------------------------------------------------------------------------
# Smoke test against real on-disk ETL output (skipped when not present)
# ---------------------------------------------------------------------------


_REAL_DATA_EXISTS = (_DEFAULT_ROOT / "binance" / "aggTrades").exists()


@pytest.mark.skipif(not _REAL_DATA_EXISTS, reason="solana-pegasus output not on disk")
def test_smoke_against_real_etl() -> None:
    """Verify the readers work against the real ETL output."""
    with MultiSourceProvider() as p:
        fr = p.get_funding_rate("BTCUSDT", "2025-06-01", "2025-07-01")
        assert len(fr) > 0
        assert fr["last_funding_rate"].notna().all()

        ct = p.get_chain_tvl("Ethereum", "2025-06-01", "2025-07-01")
        assert len(ct) > 0
        assert (ct["tvl"] > 0).all()
