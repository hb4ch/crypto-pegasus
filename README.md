<p align="center">
  <img src="crypto-pegasus.png" alt="Crypto Pegasus" width="200">
</p>

# Crypto Pegasus — Alpha Mining Backtest Platform

A high-performance backtest platform for crypto trading strategies on Binance spot pairs. Designed for minimal edit surface so an AI alpha research agent can evolve strategies by modifying a single file.

## Supported Pairs

`ETHUSDT`, `BTCUSDT`, `SOLUSDT`, `BNBUSDT`

## Data Source

Tick-level aggTrades in Parquet format, sourced from `~/solana-pegasus/data/parquet/spot/aggTrades/`. Data is partitioned by `year=YYYY/month=MM/data.parquet` per symbol (~28GB total). See the [solana-pegasus README](~/solana-pegasus/README.md) for the ETL pipeline and schema.

The platform uses DuckDB to query parquet files directly (zero-copy, hive partitioning, automatic memory management capped at 6 threads / 8GB). Tick data is resampled to OHLCV bars in a single SQL query — no manual chunking required.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Writing a Strategy

**Edit only `strategies/my_alpha.py`.** Implement one method:

```python
from pegasus.strategy.base import Strategy
import pandas as pd

class MyAlpha(Strategy):
    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 20)
        upper = bars["high"].rolling(period).max()
        lower = bars["low"].rolling(period).min()

        signals = pd.Series(0.0, index=bars.index)
        signals[bars["close"] >= upper] = 1.0
        signals[bars["close"] <= lower] = -1.0
        return signals
```

### Input: `bars` DataFrame

DataFrame indexed by datetime (UTC) with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `open` | float | First trade price in bar |
| `high` | float | Max price in bar |
| `low` | float | Min price in bar |
| `close` | float | Last trade price in bar |
| `volume` | float | Total quantity traded |
| `buy_volume` | float | Taker buy quantity (where `is_buyer_maker=False`) |
| `vwap` | float | Volume-weighted average price |
| `trade_count` | int | Number of aggTrades in bar |

### Output: Signal Series

Return a `pd.Series` with the same index as `bars`:

| Value | Meaning |
|-------|---------|
| `1.0` | Fully long |
| `-1.0` | Fully short |
| `0.0` | Flat (no position) |
| `0.5` | 50% long (fractional sizing) |
| `NaN` | No signal (warmup period) |

### Strategy Parameters

Access via `self.params` dict. Pass from CLI with `--params '{"period": 20}'` or in Python:

```python
strategy = MyAlpha(params={"period": 20, "threshold": 0.01})
```

## Running a Backtest

### CLI

```bash
python scripts/run_backtest.py \
  --strategy strategies.my_alpha.MyAlpha \
  --symbols ETHUSDT \
  --start 2024-01-01 \
  --end 2025-01-01 \
  --timeframe 5min \
  --capital 100000 \
  --output results/ \
  --params '{"period": 20}'
```

### Python

```python
from pegasus.config import BacktestConfig
from pegasus.engine.backtest import BacktestEngine
from pegasus.metrics.report import compute_metrics, print_report
from pegasus.viz.plots import save_report
from strategies.my_alpha import MyAlpha

strategy = MyAlpha(params={"period": 20})
config = BacktestConfig()
engine = BacktestEngine(strategy=strategy, config=config)

# Single symbol
result = engine.run("ETHUSDT", "2024-01-01", "2025-01-01", "5min")
result.metrics = compute_metrics(result)
print_report(result.metrics)
save_report(result, "results/ETHUSDT_5min.html")

# Multiple symbols
results = engine.run_multi(
    symbols=["ETHUSDT", "SOLUSDT"],
    start="2024-01-01",
    end="2025-01-01",
    timeframe="5min",
)
```

### Available Timeframes

`1min`, `5min`, `15min`, `30min`, `1h`, `4h`, `1d`

## How the Backtest Engine Works

Fully vectorized — no bar-by-bar loops:

```
signals    = strategy.generate_signals(bars)
positions  = signals.shift(1)                        # avoid lookahead bias
positions  = apply_stops(positions, bars, config)     # optional SL/TP/trailing
returns    = positions * effective_close.pct_change() # position * market return
costs      = positions.diff().abs() * (fee_rate + slippage_bps / 10000)
net_returns = returns - costs
equity     = (1 + net_returns).cumprod() * initial_capital
```

Key properties:
- **No lookahead bias**: positions are shifted by 1 bar from signals
- **Transaction costs**: fee (default 10bps) + slippage (default 5bps) applied on every position change
- **Vectorized**: entire backtest runs in one pass over numpy arrays
- **Stop loss / take profit**: optional engine-level stops check intra-bar high/low against entry price

## Stop Loss / Take Profit

Engine-level risk management via config — no strategy code changes needed:

```python
config = BacktestConfig(
    stop_loss_pct=0.02,       # 2% stop loss
    take_profit_pct=0.05,     # 5% take profit
    trailing_stop_pct=0.03,   # 3% trailing stop
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stop_loss_pct` | `None` | Exit when unrealized loss exceeds this % from entry |
| `take_profit_pct` | `None` | Exit when unrealized gain exceeds this % from entry |
| `trailing_stop_pct` | `None` | Exit when price retraces this % from highest point since entry |

How it works:
- Stops are checked against intra-bar **high/low** (not just close), so a 2% stop won't report a 5% loss
- Exit price is the exact stop level, not the bar's close
- After a stop triggers, the strategy can **re-enter** on the next bar if its signal is still active
- All stops are `None` by default — existing behavior is unchanged
- Uses a numba-accelerated loop for performance
- Stops apply after signal generation, before return calculation

## Metrics

`compute_metrics(result)` returns a dict with 30+ metrics across 6 categories:

### Returns
- `total_return` — cumulative return
- `cagr` — compound annual growth rate
- `avg_return` — mean per-bar return
- `cumulative_pnl` — absolute P&L in dollars

### Risk-Adjusted
- `sharpe` — annualized Sharpe ratio
- `sortino` — Sharpe penalizing only downside vol
- `calmar` — CAGR / max drawdown
- `omega` — probability-weighted gains / losses (>1 is good)
- `tail_ratio` — 95th percentile / |5th percentile| (>1 = fatter right tail)
- `payoff_ratio` — avg win size / avg loss size

### Risk
- `volatility` — annualized standard deviation
- `downside_volatility` — annualized std of negative returns only
- `max_drawdown` — worst peak-to-trough decline
- `max_drawdown_duration_bars` — longest drawdown in bars
- `var_95` — 5th percentile return (Value at Risk)
- `cvar_95` — expected loss in worst 5% (Expected Shortfall)
- `skew` — return distribution skewness
- `kurtosis` — tail heaviness

### Exposure
- `exposure_pct` — % of bars with non-zero position
- `long_exposure_pct` / `short_exposure_pct` — directional breakdown
- `avg_position_size` — mean absolute position

### Rolling
- `rolling_sharpe_mean` / `rolling_sharpe_std` — stability of Sharpe over time

### Trades
- `total_trades` — number of round-trips
- `win_rate` — % profitable trades
- `profit_factor` — gross wins / gross losses
- `avg_win` / `avg_loss` — per winning/losing trade
- `avg_trade_return` / `median_trade_return`
- `best_trade` / `worst_trade`
- `trade_return_std` — dispersion of outcomes
- `max_consecutive_wins` / `max_consecutive_losses`
- `avg_holding_seconds` / `median_holding_seconds`

## Visualization

HTML reports with interactive Plotly charts are saved via `save_report(result, path)`:

- **Equity curve** with drawdown subplot (zoomable, hover tooltips)
- **Monthly returns heatmap** (year × month, red/green)
- **Signal overlay** — price chart with long/short entry markers
- **Metrics table** — all metrics in the HTML header

Individual plot functions are also available:

```python
from pegasus.viz.plots import plot_equity_curve, plot_monthly_returns, plot_signals

fig = plot_equity_curve(result)   # returns plotly.graph_objects.Figure
fig.show()
```

## Configuration

`pegasus/config.py` — `BacktestConfig` (Pydantic model):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_root` | `~/solana-pegasus/data/parquet/spot/aggTrades` | Path to parquet data |
| `symbols` | `["ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT"]` | Trading pairs |
| `start_date` | `"2024-01-01"` | Default backtest start |
| `end_date` | `"2026-03-01"` | Default backtest end |
| `timeframe` | `"5min"` | Default bar timeframe |
| `initial_capital` | `100,000` | Starting capital (USD) |
| `fee_rate` | `0.001` | Fee per side (10 bps) |
| `slippage_bps` | `5.0` | Slippage in basis points |
| `stop_loss_pct` | `None` | Fixed stop loss (e.g. `0.02` = 2%) |
| `take_profit_pct` | `None` | Fixed take profit (e.g. `0.05` = 5%) |
| `trailing_stop_pct` | `None` | Trailing stop (e.g. `0.03` = 3%) |
| `duckdb_threads` | `6` | Max DuckDB threads |
| `duckdb_memory_limit` | `"8GB"` | Max DuckDB memory |

Override in Python:

```python
config = BacktestConfig(timeframe="1h", fee_rate=0.0005, initial_capital=50_000)
```

## Project Structure

```
crypto-pegasus/
├── pegasus/                       # Core library (DO NOT EDIT for strategy work)
│   ├── config.py                  #   Pydantic configuration
│   ├── data/
│   │   └── provider.py            #   DuckDB parquet→OHLCV provider
│   ├── strategy/
│   │   ├── base.py                #   Strategy ABC (one method: generate_signals)
│   │   └── examples/
│   │       └── sma_cross.py       #   SMA crossover reference
│   ├── engine/
│   │   └── backtest.py            #   Vectorized backtest engine
│   ├── metrics/
│   │   └── report.py              #   30+ metrics via quantstats
│   └── viz/
│       └── plots.py               #   Plotly charts + HTML report
├── strategies/                    # ← EDIT HERE: strategy files
│   └── my_alpha.py                #   Your strategy
├── scripts/
│   └── run_backtest.py            #   CLI entry point
├── tests/                         #   82 unit tests
│   ├── conftest.py
│   ├── test_provider.py
│   ├── test_engine.py
│   ├── test_metrics.py
│   └── test_strategy.py
├── results/                       #   Generated HTML reports
├── pyproject.toml
└── README.md
```

## Tests

```bash
pytest tests/ -v          # 82 tests
pytest tests/ -v --cov    # with coverage
```

## Performance

Benchmarked on real data (ETHUSDT, 5min bars):

| Range | Bars | Time |
|-------|------|------|
| 1 month | 8,640 | 0.5s |
| 6 months | 52,416 | 3.6s |
| 1 year | ~105,000 | ~8s |

## Dependencies

| Library | Purpose |
|---------|---------|
| `duckdb` | Parquet queries, OHLCV resampling |
| `pandas` | Strategy interface, data manipulation |
| `numpy` | Vectorized backtest math |
| `numba` | JIT-compiled stop loss/take profit loop |
| `quantstats` | Performance metrics |
| `plotly` | Interactive charts |
| `pydantic` | Configuration validation |
| `click` | CLI |
| `pyarrow` | Parquet format support |
