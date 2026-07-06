# ---
# jupyter:
#   aperiodic:
#     uses_preview_data: true
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: aperiodic
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Alpha Discovery
#
# We define **Directional Alpha** as a signal that exhibits a consistent positive
# or negative relationship with forward returns.
#
# The core task in alpha discovery is to transform a raw time series into a signal
# that:
# - ranges from `-1` to `1`
# - has a reasonable distribution, ideally close to uniform
# - is slow enough to survive transaction costs
#
# With no API key it runs end to end on preview data (single month - May 2025).
# Supply your own key (below or via `APERIODIC_API_KEY`) to query your
# subscription's full download window — just widen the `start_date`/`end_date`.
#
# ```text
# +------------------+     +--------------+     +-----------------------------+
# | Raw Metric       | --> | Smoothing    | --> | Rolling Rank Transformation |
# +------------------+     +--------------+     +-----------------------------+
#                                                   |
#                                                   v
#                                   +---------------------+     +----------------+
#                                   | Long/Short Position | --> | Backtest       |
#                                   +---------------------+     +----------------+
# ```
#
# ---
# ## Setup — Configuration
#
# - **`rank_windows`** — look-back length for the rolling percentile-rank transform
#   (longer windows produce slower, more stable signals).
# - **`smooth_windows`** — moving-average lengths applied to the ranked signal;
#   `None` leaves it unsmoothed. This is the second search dimension.
# - **`cost_bps`** — one-way transaction cost, in basis points, applied by the backtest.

# %%
import datetime
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

API_KEY = "..."  # Set via APERIODIC_API_KEY env var or .env file
if API_KEY == "...":
    API_KEY = os.getenv("APERIODIC_API_KEY", "...")
if API_KEY == "...":
    API_KEY = "DEMO-KEY"

USE_PREVIEW = API_KEY == "DEMO-KEY"


@dataclass(frozen=True)
class AlphaDiscoveryConfig:
    """All parameters for the alpha-discovery notebook in one place."""

    api_key: str | None
    preview: bool
    symbol: str
    exchange: str
    interval: str
    timestamp: str
    start_date: datetime.date
    end_date: datetime.date
    metrics: list[tuple[str, str]]
    rank_windows: list[int]
    smooth_windows: list[int | None]
    max_smoothing_rank_ratio: float
    cost_bps: float


config = AlphaDiscoveryConfig(
    api_key=API_KEY,
    preview=USE_PREVIEW,
    symbol="perpetual-BTC-USDT:USDT",
    exchange="binance-futures",
    interval="5m",
    timestamp="exchange",  # local timestamp or "true"
    start_date=datetime.date(2025, 5, 1),
    end_date=datetime.date(2025, 5, 31),
    metrics=[
        ("basis", "derivative"),
        ("funding", "derivative"),
        ("open_interest", "derivative"),
        ("flow", "regular"),
        ("impact", "regular"),
        ("l1_price", "regular"),
        ("l1_imbalance", "regular"),
        ("l1_liquidity", "regular"),
        ("returns", "regular"),
        ("slippage", "regular"),
        ("trade_size", "regular"),
        ("updownticks", "regular"),
        ("run_structure", "regular"),
        ("vtwap", "regular"),
        ("range", "regular"),
    ],
    rank_windows=[300, 600, 1200, 2400, 3600, 4800],
    smooth_windows=[None, 50, 100, 200, 400],
    max_smoothing_rank_ratio=1 / 3,
    cost_bps=1.0,
)

# Display parameters for the tables and plots below.
TOP_STRATEGY_COUNT = 10
TOP_PLOT_COUNT = 5

# %% [markdown]
# ---
# ## Setup — Helper functions
#
# The data retrieval, signal construction, search, and plotting routines are
# defined below in this notebook so the analysis stays self-contained.

# %% {"jupyter": {"source_hidden": true}}
from aperiodic import get_derivative_metrics, get_metrics, get_ohlcv


def run_position_backtest(
    timestamps: pd.Series,
    position: np.ndarray,
    forward_return: np.ndarray,
    cost_bps_one_way: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    position = np.asarray(position, dtype=np.float64)
    forward_return = np.asarray(forward_return, dtype=np.float64)

    gross_pnl = position * forward_return
    turnover = np.abs(np.diff(position, prepend=0.0))
    cost = turnover * cost_bps_one_way / 1e4
    net_pnl = gross_pnl - cost
    equity = np.cumprod(1.0 + net_pnl)

    if equity.size == 0:
        bt_frame = pd.DataFrame(
            {"timestamp": timestamps.to_numpy(), "equity_curve": equity}
        )
        bt_summary = {
            "annualized_sharpe": 0.0,
            "net_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }
        return bt_frame, bt_summary

    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_drawdown_pct = float(np.min(drawdowns)) * 100.0

    bars_per_year = 288 * 365
    mean_ret = float(np.mean(net_pnl))
    std_ret = float(np.std(net_pnl, ddof=1)) if len(net_pnl) > 1 else 1.0
    annualized_sharpe = (
        (mean_ret / std_ret) * np.sqrt(bars_per_year) if std_ret > 0 else 0.0
    )
    net_return_pct = (
        float((equity[-1] / equity[0] - 1.0) * 100.0) if len(equity) > 0 else 0.0
    )

    bt_frame = pd.DataFrame(
        {"timestamp": timestamps.to_numpy(), "equity_curve": equity}
    )
    bt_summary = {
        "annualized_sharpe": float(annualized_sharpe),
        "net_return_pct": net_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
    }
    return bt_frame, bt_summary


def get_numeric_metric_frame(
    config: AlphaDiscoveryConfig, metric: str, kind: str
) -> pd.DataFrame | None:
    fetcher = get_derivative_metrics if kind == "derivative" else get_metrics
    raw_frame = fetcher(
        api_key=config.api_key,
        metric=metric,
        timestamp=config.timestamp,
        interval=config.interval,
        exchange=config.exchange,
        symbol=config.symbol,
        start_date=config.start_date,
        end_date=config.end_date,
        output="pandas",
        show_progress=True,
        preview=config.preview,
    )

    frame = (
        raw_frame.to_pandas()
        if hasattr(raw_frame, "to_pandas")
        else pd.DataFrame(raw_frame)
    )
    if frame.empty or "time" not in frame.columns:
        print(f"Skipping {metric}: no rows returned.")
        return None

    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    if "time" in numeric_cols:
        numeric_cols.remove("time")

    if not numeric_cols:
        print(f"Skipping {metric}: no numeric columns returned.")
        return None

    return frame.sort_values("time").drop_duplicates(subset=["time"], keep="last")[
        ["time", *numeric_cols]
    ]


def build_panel(
    config: AlphaDiscoveryConfig,
) -> tuple[pd.DataFrame, list[str], dict[str, tuple[str, str]]]:
    raw_ohlcv = get_ohlcv(
        api_key=config.api_key,
        timestamp=config.timestamp,
        interval=config.interval,
        exchange=config.exchange,
        symbol=config.symbol,
        start_date=config.start_date,
        end_date=config.end_date,
        output="pandas",
        show_progress=True,
        preview=config.preview,
    )

    panel = (
        raw_ohlcv.to_pandas()
        if hasattr(raw_ohlcv, "to_pandas")
        else pd.DataFrame(raw_ohlcv)
    )
    panel = panel.sort_values("time")[["time", "close"]]

    feature_sources: dict[str, tuple[str, str]] = {}
    for metric, kind in config.metrics:
        frame = get_numeric_metric_frame(config, metric, kind)
        if frame is None:
            continue

        feature_values = [col for col in frame.columns if col != "time"]
        if frame[feature_values].notna().sum().sum() == 0:
            print(f"Skipping {metric}: all feature values are null.")
            continue

        panel = panel.merge(frame, on="time", how="left")
        for col in feature_values:
            feature_sources[col] = (metric, kind)

    panel = panel.sort_values("time")
    feature_cols = [
        col
        for col in panel.columns
        if col not in {"time", "close"} and pd.api.types.is_numeric_dtype(panel[col])
    ]

    panel[feature_cols] = panel[feature_cols].ffill()
    panel["fwd_ret"] = panel["close"].pct_change().shift(-1)
    panel = panel.dropna(subset=["fwd_ret"])

    feature_sources = {
        col: feature_sources[col] for col in feature_cols if col in feature_sources
    }
    return panel, feature_cols, feature_sources


def make_rank_signal(panel_df: pd.DataFrame, feature: str, window: int) -> np.ndarray:
    values = panel_df[feature]
    rolling_rank = values.rolling(window, min_periods=10).rank(method="average")
    effective_window = values.rolling(window, min_periods=10).count()
    signal = ((rolling_rank - 1.0) / (effective_window - 1.0)) * 2.0 - 1.0
    return signal.to_numpy().astype(np.float64)


def smooth_signal(signal: np.ndarray, window: int | None) -> np.ndarray:
    if window is None or pd.isna(window) or int(window) == 1:
        return signal.astype(np.float64, copy=True)

    return pd.Series(signal).rolling(int(window)).mean().to_numpy().astype(np.float64)


def allowed_smooth_windows(
    config: AlphaDiscoveryConfig, rank_window: int
) -> list[int | None]:
    max_smooth_window = rank_window * config.max_smoothing_rank_ratio
    allowed: list[int | None] = []

    for smooth_window in config.smooth_windows:
        if smooth_window is None or pd.isna(smooth_window):
            allowed.append(None)
            continue

        if int(smooth_window) <= max_smooth_window:
            allowed.append(int(smooth_window))

    return allowed


def evaluate_strategies(
    config: AlphaDiscoveryConfig, panel_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    forward_returns = panel_df["fwd_ret"].to_numpy().astype(np.float64)
    results: list[dict[str, float | int | str | None]] = []

    for feature in feature_cols:
        for rank_window in config.rank_windows:
            raw_signal = make_rank_signal(panel_df, feature, rank_window)

            for smooth_window in allowed_smooth_windows(config, rank_window):
                signal = smooth_signal(raw_signal, smooth_window)
                mask = np.isfinite(signal) & np.isfinite(forward_returns)

                signal_valid = signal[mask]
                returns_valid = forward_returns[mask]
                if (
                    signal_valid.size < 2
                    or np.std(signal_valid) == 0.0
                    or np.std(returns_valid) == 0.0
                ):
                    continue

                fit_corr = float(np.corrcoef(signal_valid, returns_valid)[0, 1])
                if not np.isfinite(fit_corr):
                    continue

                direction = 1 if fit_corr >= 0 else -1
                _, bt_summary = run_position_backtest(
                    timestamps=panel_df.loc[mask, "time"],
                    position=np.nan_to_num(
                        np.clip(signal_valid * direction, -1.0, 1.0), nan=0.0
                    ),
                    forward_return=returns_valid,
                    cost_bps_one_way=config.cost_bps,
                )

                results.append(
                    {
                        "feature": feature,
                        "rank_window": rank_window,
                        "smooth_window": smooth_window,
                        "direction": direction,
                        "fit_corr": fit_corr,
                        "sharpe": bt_summary["annualized_sharpe"],
                        "return_pct": bt_summary["net_return_pct"],
                        "drawdown_pct": bt_summary["max_drawdown_pct"],
                    }
                )

    if not results:
        raise RuntimeError(
            "No valid feature/window combinations were produced. "
            "Check the fetched metric coverage for the selected date range."
        )

    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    return results_df, forward_returns


def summarize_top_strategies(results_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    summary = results_df.head(top_n).copy()
    return format_strategy_summary(summary)


def summarize_diverse_top_strategies(
    results_df: pd.DataFrame,
    feature_sources: dict[str, tuple[str, str]],
    top_n: int,
) -> pd.DataFrame:
    summary = results_df.copy()
    summary["metric_type"] = summary["feature"].map(
        lambda feature: feature_sources.get(str(feature), (str(feature), "regular"))[0]
    )
    summary = summary.drop_duplicates(subset=["metric_type"], keep="first").head(top_n)
    return format_strategy_summary(summary, include_metric_type=True)


def format_strategy_summary(
    summary: pd.DataFrame, include_metric_type: bool = False
) -> pd.DataFrame:
    summary = summary.copy()
    summary["smooth_window"] = summary["smooth_window"].map(
        lambda value: "None" if value is None or pd.isna(value) else int(value)
    )
    summary["direction"] = summary["direction"].map({1: "long", -1: "short"})
    columns = [
        "feature",
        "rank_window",
        "smooth_window",
        "direction",
        "fit_corr",
        "sharpe",
        "return_pct",
        "drawdown_pct",
    ]
    if include_metric_type:
        columns = ["metric_type", *columns]
    return summary[columns]


def build_strategy_artifacts(
    config: AlphaDiscoveryConfig,
    panel_df: pd.DataFrame,
    forward_returns: np.ndarray,
    row: pd.Series,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, float], np.ndarray]:
    raw_signal = make_rank_signal(
        panel_df, str(row["feature"]), int(row["rank_window"])
    )
    signal = smooth_signal(raw_signal, row["smooth_window"]) * int(row["direction"])
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    bt_frame, bt_summary = run_position_backtest(
        timestamps=panel_df.loc[mask, "time"],
        position=np.nan_to_num(np.clip(signal[mask], -1.0, 1.0), nan=0.0),
        forward_return=forward_returns[mask],
        cost_bps_one_way=config.cost_bps,
    )
    return raw_signal, signal, bt_frame, bt_summary, mask


def describe_strategy(
    row: pd.Series, feature_sources: dict[str, tuple[str, str]]
) -> str:
    feature = str(row["feature"])
    metric_type, metric_category = feature_sources.get(feature, (feature, "regular"))
    return f"{feature} | metric={metric_type} | category={metric_category}"


def plot_strategy_overview(
    config: AlphaDiscoveryConfig,
    panel_df: pd.DataFrame,
    forward_returns: np.ndarray,
    results_df: pd.DataFrame,
    row: pd.Series,
    title_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw_signal, signal, bt_frame, _, mask = build_strategy_artifacts(
        config, panel_df, forward_returns, row
    )
    deciles_df = build_decile_summary(signal, forward_returns)
    feature_results = results_df.loc[
        results_df["feature"] == row["feature"],
        ["rank_window", "smooth_window", "sharpe"],
    ].copy()
    heatmap = feature_results.pivot(
        index="rank_window", columns="smooth_window", values="sharpe"
    ).sort_index()
    heatmap = heatmap.reindex(index=config.rank_windows, columns=config.smooth_windows)
    heatmap = heatmap.rename(columns={None: "None"})
    heatmap_plot = heatmap.copy()
    heatmap_plot.columns = [str(col) for col in heatmap_plot.columns]
    masked_heatmap = np.ma.masked_invalid(heatmap_plot.to_numpy(dtype=float))
    smooth_label = (
        "None"
        if row["smooth_window"] is None or pd.isna(row["smooth_window"])
        else int(row["smooth_window"])
    )

    fig = plt.figure(figsize=(14, 12))
    grid = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)
    ax_raw = fig.add_subplot(grid[0, :])
    ax_signal = fig.add_subplot(grid[1, :], sharex=ax_raw)
    ax_equity = fig.add_subplot(grid[2, :], sharex=ax_raw)
    ax_deciles = fig.add_subplot(grid[3, 0])
    ax_heatmap = fig.add_subplot(grid[3, 1])

    ax_raw.plot(panel_df["time"], raw_signal, linewidth=0.8, color="tab:orange")
    ax_raw.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax_raw.set_title(f"{title_prefix} raw rank signal | {row['feature']}")
    ax_raw.grid(alpha=0.2)

    ax_signal.plot(panel_df["time"], signal, linewidth=0.9, color="tab:red")
    ax_signal.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax_signal.set_title(
        f"{title_prefix} smoothed signal"
        f" | rank={int(row['rank_window'])}"
        f" | smooth={smooth_label}"
        f" | dir={int(row['direction'])}"
    )
    ax_signal.grid(alpha=0.2)

    ax_equity.plot(
        bt_frame["timestamp"],
        bt_frame["equity_curve"],
        linewidth=1.1,
        color="tab:green",
    )
    ax_equity.set_title(
        f"{title_prefix} equity"
        f" | Sharpe={float(row['sharpe']):.3f}"
        f" | Ret={float(row['return_pct']):.3f}"
    )
    ax_equity.grid(alpha=0.2)

    ax_deciles.bar(
        deciles_df["decile"],
        deciles_df["mean_fwd_ret"],
        color="tab:blue",
        alpha=0.85,
    )
    ax_deciles.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax_deciles.set_title("Mean forward returns by decile")
    ax_deciles.set_xlabel("Decile")
    ax_deciles.set_ylabel("Mean next-bar return")
    ax_deciles.grid(alpha=0.2, axis="y")

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#f3f4f6")
    image = ax_heatmap.imshow(masked_heatmap, aspect="auto", cmap=cmap)
    ax_heatmap.set_title("Robustness by window combination")
    ax_heatmap.set_xlabel("Smooth window")
    ax_heatmap.set_ylabel("Rank window")
    ax_heatmap.set_xticks(
        np.arange(len(heatmap_plot.columns)), labels=heatmap_plot.columns
    )
    ax_heatmap.set_yticks(np.arange(len(heatmap_plot.index)), labels=heatmap_plot.index)

    for row_idx, row_values in enumerate(heatmap_plot.to_numpy(dtype=float)):
        for col_idx, value in enumerate(row_values):
            if not np.isfinite(value):
                continue
            ax_heatmap.text(
                col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8
            )

    fig.colorbar(image, ax=ax_heatmap, label="Annualized Sharpe")

    fig.tight_layout()
    plt.show()
    return signal, mask


def build_decile_summary(
    signal: np.ndarray, forward_returns: np.ndarray
) -> pd.DataFrame:
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    signal_valid = signal[mask]
    returns_valid = forward_returns[mask]

    if signal_valid.size == 0:
        raise RuntimeError(
            "Best strategy produced no valid observations for decile analysis."
        )

    order = np.argsort(signal_valid)
    deciles = np.empty(signal_valid.shape[0], dtype=np.int64)
    deciles[order] = (
        np.arange(signal_valid.shape[0]) * 10 // signal_valid.shape[0]
    ) + 1

    return pd.DataFrame(
        [
            {
                "decile": decile,
                "count": int((decile_mask := deciles == decile).sum()),
                "mean_signal": float(np.nanmean(signal_valid[decile_mask])),
                "mean_fwd_ret": float(np.nanmean(returns_valid[decile_mask])),
            }
            for decile in range(1, 11)
        ]
    )


# %% [markdown]
# ---
# ## Step 1 — Load and inspect the data
#
# `build_panel` retrieves close price together with every numeric column from the
# configured metric families, forward-fills sparse series, and constructs the
# prediction target — the one-bar-ahead return (`fwd_ret`).

# %%
panel, feature_cols, feature_sources = build_panel(config)

print(f"Rows: {len(panel):,}")
print(f"Feature count: {len(feature_cols)}")
print(f"First features: {feature_cols[:12]}")
print(panel[["time", "close", "fwd_ret"]].head().to_markdown(index=False))

# %% [markdown]
# ---
# ## Step 2 — Construct the signal
#
# A raw feature is not directly tradeable: its level is non-stationary and its units
# are arbitrary. Two transforms address this:
#
# 1. **Smoothing (optional).** A moving average of `smooth_window` bars applied to
#    the ranked signal trades responsiveness for stability; `None` leaves it
#    unsmoothed.
# 2. **Percentile rank.** Within a rolling window of `rank_window` bars, each value
#    is replaced by its percentile rank and rescaled to `[-1, +1]`. A high value
#    means "elevated relative to recent history," a low value the opposite — a
#    scale-free measure that is comparable across features.
#
# Smoothing is essential for alpha to survive even minimal transaction costs, when
# one works with market microstructure / liquidity / flow metrics. They're simply
# too high frequency to be traded as-is.
#
# The position direction (long or short) is set by the sign of the in-sample
# correlation between the signal and the one-bar-ahead return.

# %% [markdown]
# ---
# ## Step 3 — Target Signal Turnover
#
# `evaluate_strategies` traverses the constrained window schedule: for each allowed
# combination it constructs the signal, assigns a direction, and runs an identical
# transaction-cost-aware position backtest. Each valid candidate is scored on
# annualized Sharpe, net return, and maximum drawdown, then ranked by Sharpe.
#
# The ranking identifies the strongest candidates; the plots characterize their
# behavior. To keep this comparison diverse, we plot the best candidate from each
# metric family rather than multiple near-duplicates from the same source. For the
# top five diversified candidates we show the raw ranked signal, the smoothed
# signal, the equity curve, and two bottom diagnostics: mean forward returns by
# decile and the feature's robustness across the tested window combinations.

# %%
results_df, forward_returns = evaluate_strategies(config, panel, feature_cols)
top_strategies = summarize_top_strategies(results_df, TOP_STRATEGY_COUNT)
diverse_top_strategies = summarize_diverse_top_strategies(
    results_df, feature_sources, TOP_STRATEGY_COUNT
)

print("\nTop strategies by Sharpe, one per metric family:")
print(diverse_top_strategies.to_markdown(index=False))

print("Top strategies by Sharpe:")
print(top_strategies.to_markdown(index=False))

# %%
diverse_plot_rows = (
    results_df.assign(
        metric_type=lambda df: df["feature"].map(
            lambda feature: feature_sources.get(str(feature), (str(feature), "regular"))[0]
        )
    )
    .drop_duplicates(subset=["metric_type"], keep="first")
    .head(TOP_PLOT_COUNT)
)

for _, row in diverse_plot_rows.iterrows():
    strategy_label = describe_strategy(row, feature_sources)
    plot_strategy_overview(
        config, panel, forward_returns, results_df, row, strategy_label
    )

# %%
best = results_df.iloc[0]
best_signal, best_mask = plot_strategy_overview(
    config,
    panel,
    forward_returns,
    results_df,
    best,
    describe_strategy(best, feature_sources),
)

best_summary = pd.DataFrame(
    [
        {
            "feature": str(best["feature"]),
            "rank_window": int(best["rank_window"]),
            "smooth_window": (
                "None"
                if best["smooth_window"] is None or pd.isna(best["smooth_window"])
                else int(best["smooth_window"])
            ),
            "direction": "long" if int(best["direction"]) == 1 else "short",
            "fit_corr": float(best["fit_corr"]),
            "sharpe": float(best["sharpe"]),
            "return_pct": float(best["return_pct"]),
            "drawdown_pct": float(best["drawdown_pct"]),
            "valid_bars": int(best_mask.sum()),
        }
    ]
)
print(best_summary.to_markdown(index=False))

# %% [markdown]
# ---
# ## Recap and Next Steps
#
# The exercise is using the preview data available, which is a single month in
# 2025, and naturally it's easy to find a strategy that worked well for a short
# period, especially when trying a hundred potential candidates, in sample.
#
# Try out the subscriptions to get access to 4 years of historical data!
#
# The next improvements to pursue are straightforward:
#
# - **Out-of-sample validation or walk-forward testing.** Re-run the selected
#   configuration on later periods, or use a rolling train/test schedule, to check
#   whether the edge survives beyond the in-sample window.
# - **Using the full history.** Expand the sample beyond the current preview slice
#   so the search covers multiple market regimes rather than one month of behavior.
# - **Signal ensembling.** The most obvious model improvement is to combine several
#   strong signals into an ensemble, for example by averaging normalized signal
#   values across complementary features instead of relying on one series alone.
