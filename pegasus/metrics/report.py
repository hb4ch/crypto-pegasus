from __future__ import annotations

import numpy as np
import pandas as pd
import quantstats as qs

from pegasus.engine.backtest import BacktestResult, Trade


def compute_metrics(result: BacktestResult) -> dict:
    """Compute comprehensive performance metrics from a BacktestResult.

    Returns a dict with portfolio-level, risk, and trade-level metrics.
    """
    returns = result.equity_curve["returns"].dropna()
    if returns.empty:
        return {}

    equity = result.equity_curve["equity"]
    positions = result.positions

    metrics: dict = {}

    # ── Return metrics ──
    metrics["total_return"] = float(qs.stats.comp(returns))
    metrics["cagr"] = float(qs.stats.cagr(returns))
    metrics["avg_return"] = float(returns.mean())
    metrics["cumulative_pnl"] = float(equity.iloc[-1] - equity.iloc[0])

    # ── Risk-adjusted metrics ──
    metrics["sharpe"] = float(qs.stats.sharpe(returns))
    metrics["sortino"] = float(qs.stats.sortino(returns))
    metrics["calmar"] = float(qs.stats.calmar(returns))
    metrics["omega"] = _omega_ratio(returns)
    metrics["tail_ratio"] = _tail_ratio(returns)
    metrics["payoff_ratio"] = _payoff_ratio(result.trades)

    # ── Risk metrics ──
    metrics["volatility"] = float(qs.stats.volatility(returns))
    metrics["downside_volatility"] = _downside_vol(returns)
    metrics["max_drawdown"] = float(qs.stats.max_drawdown(returns))
    metrics["max_drawdown_duration_bars"] = _max_dd_duration(equity)
    metrics["var_95"] = float(returns.quantile(0.05))
    metrics["cvar_95"] = float(returns[returns <= returns.quantile(0.05)].mean()) if len(returns) > 20 else 0.0
    metrics["skew"] = float(returns.skew())
    metrics["kurtosis"] = float(returns.kurtosis())

    # ── Exposure metrics ──
    metrics["exposure_pct"] = _exposure_pct(positions)
    metrics["long_exposure_pct"] = _directional_exposure(positions, direction=1)
    metrics["short_exposure_pct"] = _directional_exposure(positions, direction=-1)
    metrics["avg_position_size"] = float(positions.dropna().abs().mean())

    # ── Rolling metrics ──
    metrics["rolling_sharpe_mean"], metrics["rolling_sharpe_std"] = _rolling_sharpe(returns)

    # ── Trade-level ──
    metrics.update(_trade_metrics(result.trades))

    return metrics


# ── Helper functions ──


def _omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio: probability-weighted gains / probability-weighted losses."""
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    return float(gains / losses) if losses > 0 else float("inf")


def _tail_ratio(returns: pd.Series) -> float:
    """95th percentile / abs(5th percentile). >1 means fatter right tail."""
    p95 = returns.quantile(0.95)
    p05 = abs(returns.quantile(0.05))
    return float(p95 / p05) if p05 > 0 else float("inf")


def _payoff_ratio(trades: list[Trade]) -> float:
    """Average win size / average loss size."""
    wins = [t.return_pct for t in trades if t.return_pct > 0]
    losses = [t.return_pct for t in trades if t.return_pct < 0]
    if not wins or not losses:
        return float("inf") if wins else 0.0
    return float(np.mean(wins) / abs(np.mean(losses)))


def _downside_vol(returns: pd.Series) -> float:
    """Annualized downside deviation (only negative returns)."""
    neg = returns[returns < 0]
    if neg.empty:
        return 0.0
    # Assume 252 trading days, but bars may be intraday — use quantstats convention
    return float(neg.std() * np.sqrt(252))


def _max_dd_duration(equity: pd.Series) -> int:
    """Maximum drawdown duration in bars."""
    running_max = equity.cummax()
    in_drawdown = equity < running_max
    if not in_drawdown.any():
        return 0
    # Label contiguous drawdown regions and find the longest
    groups = (~in_drawdown).cumsum()
    dd_groups = groups[in_drawdown]
    if dd_groups.empty:
        return 0
    return int(dd_groups.value_counts().max())


def _exposure_pct(positions: pd.Series) -> float:
    """Percentage of bars with a non-zero position."""
    valid = positions.dropna()
    if valid.empty:
        return 0.0
    return float((valid != 0).sum() / len(valid))


def _directional_exposure(positions: pd.Series, direction: int) -> float:
    """Percentage of bars with position in given direction (+1 or -1)."""
    valid = positions.dropna()
    if valid.empty:
        return 0.0
    if direction > 0:
        return float((valid > 0).sum() / len(valid))
    return float((valid < 0).sum() / len(valid))


def _rolling_sharpe(returns: pd.Series, window: int = 252) -> tuple[float, float]:
    """Mean and std of rolling Sharpe (window bars). Returns (mean, std)."""
    if len(returns) < window:
        return 0.0, 0.0
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_sr = (rolling_mean / rolling_std) * np.sqrt(252)
    rolling_sr = rolling_sr.dropna()
    if rolling_sr.empty:
        return 0.0, 0.0
    return float(rolling_sr.mean()), float(rolling_sr.std())


def _trade_metrics(trades: list[Trade]) -> dict:
    """Compute trade-level statistics."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_trade_return": 0.0,
            "median_trade_return": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "trade_return_std": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "avg_holding_seconds": 0.0,
            "median_holding_seconds": 0.0,
        }

    pnls = np.array([t.return_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0

    holding = np.array([(t.exit_time - t.entry_time).total_seconds() for t in trades])

    # Consecutive wins/losses
    is_win = pnls > 0
    max_con_wins, max_con_losses = _max_consecutive(is_win)

    return {
        "total_trades": len(trades),
        "win_rate": float(len(wins) / len(trades)),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "avg_trade_return": float(pnls.mean()),
        "median_trade_return": float(np.median(pnls)),
        "best_trade": float(pnls.max()),
        "worst_trade": float(pnls.min()),
        "trade_return_std": float(pnls.std()),
        "max_consecutive_wins": max_con_wins,
        "max_consecutive_losses": max_con_losses,
        "avg_holding_seconds": float(holding.mean()),
        "median_holding_seconds": float(np.median(holding)),
    }


def _max_consecutive(is_win: np.ndarray) -> tuple[int, int]:
    """Return (max consecutive wins, max consecutive losses)."""
    if len(is_win) == 0:
        return 0, 0
    max_wins = max_losses = 0
    cur_wins = cur_losses = 0
    for w in is_win:
        if w:
            cur_wins += 1
            cur_losses = 0
        else:
            cur_losses += 1
            cur_wins = 0
        max_wins = max(max_wins, cur_wins)
        max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses


def print_report(metrics: dict) -> None:
    """Pretty-print metrics to stdout."""
    sections = {
        "RETURNS": [
            "total_return", "cagr", "avg_return", "cumulative_pnl",
        ],
        "RISK-ADJUSTED": [
            "sharpe", "sortino", "calmar", "omega", "tail_ratio", "payoff_ratio",
        ],
        "RISK": [
            "volatility", "downside_volatility", "max_drawdown",
            "max_drawdown_duration_bars", "var_95", "cvar_95", "skew", "kurtosis",
        ],
        "EXPOSURE": [
            "exposure_pct", "long_exposure_pct", "short_exposure_pct", "avg_position_size",
        ],
        "ROLLING": [
            "rolling_sharpe_mean", "rolling_sharpe_std",
        ],
        "TRADES": [
            "total_trades", "win_rate", "profit_factor",
            "avg_win", "avg_loss", "avg_trade_return", "median_trade_return",
            "best_trade", "worst_trade", "trade_return_std",
            "max_consecutive_wins", "max_consecutive_losses",
            "avg_holding_seconds", "median_holding_seconds",
        ],
    }

    fmt_pct = {
        "total_return", "cagr", "max_drawdown", "volatility", "downside_volatility",
        "win_rate", "avg_win", "avg_loss", "avg_trade_return", "median_trade_return",
        "best_trade", "worst_trade", "trade_return_std",
        "var_95", "cvar_95", "exposure_pct", "long_exposure_pct", "short_exposure_pct",
    }
    fmt_float = {
        "sharpe", "sortino", "calmar", "omega", "tail_ratio", "payoff_ratio",
        "skew", "kurtosis", "avg_return", "avg_position_size",
        "rolling_sharpe_mean", "rolling_sharpe_std",
    }
    fmt_int = {"total_trades", "max_drawdown_duration_bars", "max_consecutive_wins", "max_consecutive_losses"}
    fmt_money = {"cumulative_pnl"}
    fmt_time = {"avg_holding_seconds", "median_holding_seconds"}

    print("\n" + "=" * 55)
    print("  BACKTEST REPORT")
    print("=" * 55)

    for section_name, keys in sections.items():
        print(f"\n  ── {section_name} ──")
        for key in keys:
            if key not in metrics:
                continue
            val = metrics[key]
            if key in fmt_pct:
                print(f"    {key:<30s} {val:>10.2%}")
            elif key in fmt_float:
                print(f"    {key:<30s} {val:>10.4f}")
            elif key in fmt_int:
                print(f"    {key:<30s} {int(val):>10d}")
            elif key in fmt_money:
                print(f"    {key:<30s} ${val:>10,.2f}")
            elif key in fmt_time:
                hours = val / 3600
                label = key.replace("_seconds", "_hours")
                print(f"    {label:<30s} {hours:>10.1f}")
            else:
                print(f"    {key:<30s} {val:>10.6f}")

    print("\n" + "=" * 55 + "\n")
