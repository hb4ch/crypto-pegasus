#!/usr/bin/env python3
"""CLI entry point for running backtests."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import click

# Ensure project root is on sys.path so strategies/ can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pegasus.config import BacktestConfig
from pegasus.engine.backtest import BacktestEngine
from pegasus.metrics.report import compute_metrics, print_report
from pegasus.viz.plots import save_report


def _import_strategy(dotted_path: str):
    """Dynamically import a strategy class from a dotted path like
    ``strategies.my_alpha.MyAlpha``."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@click.command()
@click.option(
    "--strategy",
    required=True,
    help="Dotted path to strategy class (e.g. strategies.my_alpha.MyAlpha)",
)
@click.option("--symbols", default="ETHUSDT", help="Comma-separated symbols")
@click.option("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
@click.option("--end", default="2025-01-01", help="End date YYYY-MM-DD")
@click.option("--timeframe", default="5min", help="Bar timeframe (1min,5min,15min,1h,1d)")
@click.option("--capital", default=100_000.0, help="Initial capital")
@click.option("--output", default="results", help="Output directory for reports")
@click.option("--params", default=None, help="Strategy params as JSON string")
def main(
    strategy: str,
    symbols: str,
    start: str,
    end: str,
    timeframe: str,
    capital: float,
    output: str,
    params: str | None,
) -> None:
    """Run a backtest and generate an HTML report."""
    import json

    # Parse params
    strategy_params = json.loads(params) if params else {}

    # Import & instantiate strategy
    StrategyClass = _import_strategy(strategy)
    strat = StrategyClass(params=strategy_params)
    click.echo(f"Strategy: {strat.name}")

    # Config
    config = BacktestConfig(
        timeframe=timeframe,
        initial_capital=capital,
    )

    engine = BacktestEngine(strategy=strat, config=config)
    output_dir = Path(output)

    symbol_list = [s.strip() for s in symbols.split(",")]
    for sym in symbol_list:
        click.echo(f"\nRunning {sym} [{start} → {end}] @ {timeframe} ...")
        result = engine.run(sym, start, end, timeframe)

        # Compute and attach metrics
        result.metrics = compute_metrics(result)
        print_report(result.metrics)

        # Save HTML report
        report_path = save_report(result, output_dir / f"{sym}_{timeframe}.html")
        click.echo(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
