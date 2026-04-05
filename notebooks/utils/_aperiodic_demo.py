"""Lightweight backtest helpers for Aperiodic demo notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_position_backtest(
    timestamps: pd.Series,
    position: np.ndarray,
    forward_return: np.ndarray,
    cost_bps_one_way: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """Run a simple position-weighted backtest and return an equity frame and summary stats.

    Parameters
    ----------
    timestamps : pd.Series
        Timestamp for each bar.
    position : np.ndarray
        Position size at each bar (clipped to [-1, 1]).
    forward_return : np.ndarray
        The one-bar-ahead simple return for the underlying asset.
    cost_bps_one_way : float
        One-way transaction cost in basis points (e.g. 1.0 = 1 bp).

    Returns
    -------
    bt_frame : pd.DataFrame
        DataFrame with ``timestamp`` and ``equity_curve`` columns.
    bt_summary : dict
        Dictionary with ``annualized_sharpe``, ``net_return_pct``, and
        ``max_drawdown_pct``.
    """
    position = np.asarray(position, dtype=np.float64)
    forward_return = np.asarray(forward_return, dtype=np.float64)

    # Per-bar PnL: position * forward return
    gross_pnl = position * forward_return

    # Transaction costs from position changes
    turnover = np.abs(np.diff(position, prepend=0.0))
    cost = turnover * cost_bps_one_way / 1e4
    net_pnl = gross_pnl - cost

    # Equity curve (cumulative product of 1 + per-bar return)
    equity = np.cumprod(1.0 + net_pnl)

    # Running maximum for drawdown calculation
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_drawdown_pct = float(np.min(drawdowns)) * 100.0

    # Annualized Sharpe (assume 5-min bars, 288 bars/day x 365 days)
    bars_per_year = 288 * 365
    mean_ret = float(np.mean(net_pnl))
    std_ret = float(np.std(net_pnl, ddof=1)) if len(net_pnl) > 1 else 1.0
    annualized_sharpe = (mean_ret / std_ret) * np.sqrt(bars_per_year) if std_ret > 0 else 0.0

    net_return_pct = float((equity[-1] / equity[0] - 1.0) * 100.0) if len(equity) > 0 else 0.0

    bt_frame = pd.DataFrame(
        {
            "timestamp": timestamps.to_numpy(),
            "equity_curve": equity,
        }
    )

    bt_summary = {
        "annualized_sharpe": float(annualized_sharpe),
        "net_return_pct": net_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
    }

    return bt_frame, bt_summary
