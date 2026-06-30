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

# %%
# aperiodic: uses_preview_data

# %% [markdown]
# # Alpha Discovery
#
# A structured, end-to-end alpha-discovery notebook. The research logic is broken
# into discrete, documented stages so each part of the pipeline can be examined on
# its own.
#
# **What this notebook covers**
#
# - converting raw microstructure metrics into scale-free, bounded signals
# - a systematic search over the feature and parameter space, scoring each candidate
#   on a transaction-cost-aware backtest
# - diagnostics that distinguish a robust edge from parameter-specific overfitting
# - a standalone snippet for reproducing the selected configuration out-of-sample
#
# The notebook runs end to end on preview data.
#
# ### The pipeline
#
# The workflow spans three phases — **construction**, **search**, and
# **validation** — across six steps.
#
# ```text
# +------------------+     +------------------+     +----------------------+
# | Construction     | --> | Search           | --> | Validation           |
# | - Load           |     | - Rank           |     | - Compare            |
# | - Inspect        |     | - Smooth         |     | - Check              |
# +------------------+     +------------------+     +----------------------+
# ```
#
# Each step that follows maps to one node in that pipeline.
#
# ---
# ## Setup — Configuration
#
# All parameters for the study are consolidated here: the instrument and sample
# period, the metric families to retrieve, and the two search dimensions. Edit this
# cell and re-run; every downstream step reads from it.
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
    raise RuntimeError("Set APERIODIC_API_KEY in the environment or in .env.")


@dataclass(frozen=True)
class AlphaDiscoveryConfig:
    """All parameters for the alpha-discovery notebook in one place."""

    api_key: str
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
        preview=True,
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
        preview=True,
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


def summarize_panel(panel_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": feature,
                "non_null_pct": float(panel_df[feature].notna().mean() * 100.0),
                "std": float(panel_df[feature].std()),
            }
            for feature in feature_cols[:10]
        ]
    )


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
    row: pd.Series,
    title_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw_signal, signal, bt_frame, _, mask = build_strategy_artifacts(
        config, panel_df, forward_returns, row
    )
    smooth_label = (
        "None"
        if row["smooth_window"] is None or pd.isna(row["smooth_window"])
        else int(row["smooth_window"])
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(panel_df["time"], raw_signal, linewidth=0.8, color="tab:orange")
    axes[0].axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    axes[0].set_title(f"{title_prefix} raw rank signal | {row['feature']}")
    axes[0].grid(alpha=0.2)

    axes[1].plot(panel_df["time"], signal, linewidth=0.9, color="tab:red")
    axes[1].axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    axes[1].set_title(
        f"{title_prefix} smoothed signal"
        f" | rank={int(row['rank_window'])}"
        f" | smooth={smooth_label}"
        f" | dir={int(row['direction'])}"
    )
    axes[1].grid(alpha=0.2)

    axes[2].plot(
        bt_frame["timestamp"],
        bt_frame["equity_curve"],
        linewidth=1.1,
        color="tab:green",
    )
    axes[2].set_title(
        f"{title_prefix} equity"
        f" | Sharpe={float(row['sharpe']):.3f}"
        f" | Ret={float(row['return_pct']):.3f}"
    )
    axes[2].grid(alpha=0.2)

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
# prediction target — the one-bar-ahead return (`fwd_ret`). We inspect the panel's
# dimensions, a sample of rows, and per-feature coverage before building anything on
# top of it.

# %%
panel, feature_cols, feature_sources = build_panel(config)
panel_summary = summarize_panel(panel, feature_cols)

print(f"Rows: {len(panel):,}")
print(f"Feature count: {len(feature_cols)}")
print(f"First features: {feature_cols[:12]}")
print(panel[["time", "close", "fwd_ret"]].head().to_markdown(index=False))

print("\nFeature coverage snapshot:")
print(panel_summary.to_markdown(index=False))

# %% [markdown]
# ---
# ## Step 2 — Construct the signal
#
# A raw feature is not directly tradeable: its level is non-stationary and its units
# are arbitrary. Two transforms address this:
#
# 1. **Percentile rank.** Within a rolling window of `rank_window` bars, each value
#    is replaced by its percentile rank and rescaled to `[-1, +1]`. A high value
#    means "elevated relative to recent history," a low value the opposite — a
#    scale-free measure that is comparable across features.
# 2. **Smoothing (optional).** A moving average of `smooth_window` bars applied to
#    the ranked signal trades responsiveness for stability; `None` leaves it
#    unsmoothed.
#
# The position direction (long or short) is set by the sign of the in-sample
# correlation between the signal and the one-bar-ahead return.
#
# We do not test the full Cartesian product. Instead, smoothing is constrained by
# ranking horizon: `None` is always allowed, and numeric smoothers are only tested
# when `smooth_window <= rank_window / 3`. This keeps long smoothers attached to
# long ranking windows and removes the least plausible short-rank / long-smoother
# combinations.

# %%
window_pairs_per_feature = sum(
    len(allowed_smooth_windows(config, rank_window))
    for rank_window in config.rank_windows
)
full_grid_pairs = len(config.rank_windows) * len(config.smooth_windows)
search_space = pd.DataFrame(
    [
        {
            "features": len(feature_cols),
            "rank_windows": len(config.rank_windows),
            "smooth_windows": len(config.smooth_windows),
            "window_pairs_per_feature": window_pairs_per_feature,
            "full_grid_pairs_per_feature": full_grid_pairs,
            "reduction_vs_full_grid_pct": (
                100.0 * (1.0 - window_pairs_per_feature / full_grid_pairs)
            ),
            "total_combinations": len(feature_cols) * window_pairs_per_feature,
        }
    ]
)
print(search_space.to_markdown(index=False))

# %% [markdown]
# ---
# ## Step 3 — Search the parameter grid
#
# `evaluate_strategies` traverses the constrained window schedule: for each allowed
# combination it constructs the signal, assigns a direction, and runs an identical
# transaction-cost-aware position backtest. Each valid candidate is scored on
# annualized Sharpe, net return, and maximum drawdown, then ranked by Sharpe.
#
# **Decision.** We report two leaderboards:
# - the raw top-Sharpe list, which may contain many variants from the same source
#   metric family
# - a diversified list that keeps only the best strategy per metric family
#
# This second view makes the results easier to interpret because it answers a
# different question: which *data sources* are strongest, rather than which small
# parameter variations dominate the same source repeatedly.

# %%
results_df, forward_returns = evaluate_strategies(config, panel, feature_cols)
top_strategies = summarize_top_strategies(results_df, TOP_STRATEGY_COUNT)
diverse_top_strategies = summarize_diverse_top_strategies(
    results_df, feature_sources, TOP_STRATEGY_COUNT
)

print("Top strategies by Sharpe:")
print(top_strategies.to_markdown(index=False))

print("\nTop strategies by Sharpe, one per metric family:")
print(diverse_top_strategies.to_markdown(index=False))

# %% [markdown]
# ---
# ## Step 4 — Compare the leading candidates
#
# The ranking identifies the strongest candidates; the plots characterize their
# behavior. To keep this comparison diverse, we plot the best candidate from each
# metric family rather than multiple near-duplicates from the same source. For the
# top five diversified candidates we show the raw ranked signal, the smoothed
# signal, and the resulting equity curve.

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
    plot_strategy_overview(config, panel, forward_returns, row, strategy_label)

# %% [markdown]
# ---
# ## Step 5 — Examine the best candidate
#
# We isolate the highest-ranked candidate and inspect its signal and equity curve,
# followed by a decile analysis: the signal is partitioned into ten buckets and the
# mean one-bar-ahead return is computed for each. A monotonic progression across
# deciles is stronger evidence of a genuine relationship than a single outlier
# bucket driving the result.

# %%
best = results_df.iloc[0]
best_signal, best_mask = plot_strategy_overview(
    config, panel, forward_returns, best, describe_strategy(best, feature_sources)
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

# %%
deciles_df = build_decile_summary(best_signal, forward_returns)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(deciles_df["decile"], deciles_df["mean_fwd_ret"], color="tab:blue", alpha=0.85)
ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
ax.set_title("Best strategy: mean next-bar return by signal decile")
ax.set_xlabel("Decile")
ax.set_ylabel("Mean next-bar return")
ax.grid(alpha=0.2, axis="y")
fig.tight_layout()
plt.show()

print(deciles_df.to_markdown(index=False))

# %% [markdown]
# ---
# ## Step 6 — Robustness check
#
# Is the measured edge genuine, or an artifact of a single parameter combination?
# The heatmap shows the top feature's Sharpe across the tested `rank_window ×
# smooth_window` combinations. Blank cells were intentionally not evaluated because
# the smoother was too long relative to the ranking horizon. An isolated
# high-Sharpe cell among weak neighbors indicates overfitting; a contiguous
# high-Sharpe region indicates the edge degrades gracefully under parameter
# perturbation, which is characteristic of a robust signal.
#
# > **Scope.** This assesses robustness to parameter selection in-sample. It is a
# > diagnostic, not a substitute for out-of-sample validation — see *Next steps*.

# %%
best_feature_results = results_df.loc[
    results_df["feature"] == best["feature"],
    ["rank_window", "smooth_window", "sharpe"],
].copy()
heatmap = best_feature_results.pivot(
    index="rank_window", columns="smooth_window", values="sharpe"
).sort_index()
heatmap = heatmap.reindex(index=config.rank_windows, columns=config.smooth_windows)
heatmap = heatmap.rename(columns={None: "None"})
heatmap_plot = heatmap.copy()
heatmap_plot.columns = [str(col) for col in heatmap_plot.columns]
masked_heatmap = np.ma.masked_invalid(heatmap_plot.to_numpy(dtype=float))

fig, ax = plt.subplots(figsize=(8, 4))
cmap = plt.get_cmap("RdYlGn").copy()
cmap.set_bad(color="#f3f4f6")
image = ax.imshow(masked_heatmap, aspect="auto", cmap=cmap)
ax.set_title(f"Sharpe by window combination for {best['feature']}")
ax.set_xlabel("Smooth window")
ax.set_ylabel("Rank window")
ax.set_xticks(np.arange(len(heatmap_plot.columns)), labels=heatmap_plot.columns)
ax.set_yticks(np.arange(len(heatmap_plot.index)), labels=heatmap_plot.index)

for row_idx, row_values in enumerate(heatmap_plot.to_numpy(dtype=float)):
    for col_idx, value in enumerate(row_values):
        if not np.isfinite(value):
            continue
        ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=9)

fig.colorbar(image, ax=ax, label="Annualized Sharpe")
fig.tight_layout()
plt.show()

# %%
top_feature_summary = (
    results_df.groupby("feature", as_index=False)["sharpe"]
    .max()
    .sort_values("sharpe", ascending=False)
    .head(10)
)
print("Top features by best Sharpe:")
print(top_feature_summary.to_markdown(index=False))

# %% [markdown]
# ---
# ## Recap
#
# This notebook executed the complete workflow:
#
# - **Loaded** price and microstructure features and constructed a one-bar-ahead
#   return target.
# - **Constructed** scale-free ranked signals, with optional smoothing.
# - **Searched** a horizon-aligned feature × window schedule, scoring every tested
#   candidate on a cost-aware backtest.
# - **Validated** the leading candidates via equity curve, decile monotonicity, and
#   parameter-robustness diagnostics.
#
# Note that smoothing constitutes a second search dimension: a feature can rank
# highly either because its raw ranked signal is predictive or because a slower
# variant is cleaner.

# %% [markdown]
# ---
# ## Next steps
#
# The natural follow-up is **out-of-sample validation**: re-run the selected
# configuration on a later period, a different instrument, or another venue. An edge
# that persists out-of-sample warrants further study.
#
# The cell below prints a **standalone snippet**, pre-filled with this run's selected
# configuration. It depends only on `aperiodic`, `numpy`, and `pandas` — no code from
# this repository — so it can be copied into any environment and run as-is. It
# defaults to the in-sample window on preview data and reports the configuration's
# annualized Sharpe, net return, and maximum drawdown; change the dates (and set
# `PREVIEW = False` with a full API key) to validate out-of-sample.

# %%
best_feature = str(best["feature"])
best_metric, best_kind = feature_sources.get(best_feature, (best_feature, "regular"))
best_rank = int(best["rank_window"])
best_smooth = (
    None
    if best["smooth_window"] is None or pd.isna(best["smooth_window"])
    else int(best["smooth_window"])
)
fetch_fn = "get_derivative_metrics" if best_kind == "derivative" else "get_metrics"

snippet = f'''# Standalone reproduction of the selected configuration.
# Dependencies: aperiodic, numpy, pandas.
import datetime

import numpy as np
import pandas as pd
from aperiodic import get_ohlcv, {fetch_fn}

API_KEY = "YOUR_KEY"
SYMBOL, EXCHANGE = "{config.symbol}", "{config.exchange}"
INTERVAL, TIMESTAMP = "{config.interval}", "{config.timestamp}"
START = datetime.date.fromisoformat("{config.start_date.isoformat()}")
END = datetime.date.fromisoformat("{config.end_date.isoformat()}")
PREVIEW = True  # set False with a full API key (and shift START/END) for out-of-sample

FEATURE = "{best_feature}"          # selected feature column
METRIC = "{best_metric}"            # its source metric family
RANK_WINDOW = {best_rank}
SMOOTH_WINDOW = {best_smooth}        # None disables smoothing
DIRECTION = {int(best["direction"])}            # +1 long, -1 short
COST_BPS = {config.cost_bps}

# 1. Retrieve close price and the source metric, de-duplicate, and align on time.
price = get_ohlcv(api_key=API_KEY, symbol=SYMBOL, exchange=EXCHANGE, interval=INTERVAL,
                  timestamp=TIMESTAMP, start_date=START, end_date=END,
                  output="pandas", preview=PREVIEW)
price = price.to_pandas() if hasattr(price, "to_pandas") else pd.DataFrame(price)
price = price.sort_values("time")[["time", "close"]]

metric = {fetch_fn}(api_key=API_KEY, metric=METRIC, symbol=SYMBOL, exchange=EXCHANGE,
                  interval=INTERVAL, timestamp=TIMESTAMP, start_date=START, end_date=END,
                  output="pandas", preview=PREVIEW)
metric = metric.to_pandas() if hasattr(metric, "to_pandas") else pd.DataFrame(metric)
metric = metric.sort_values("time").drop_duplicates(subset=["time"], keep="last")

panel = price.merge(metric[["time", FEATURE]], on="time", how="left").sort_values("time")
panel[FEATURE] = panel[FEATURE].ffill()
panel["fwd_ret"] = panel["close"].pct_change().shift(-1)
panel = panel.dropna(subset=["fwd_ret"])

# 2. Rebuild the ranked (and optionally smoothed) signal, then form the position.
rolling_rank = panel[FEATURE].rolling(RANK_WINDOW, min_periods=10).rank(method="average")
effective_window = panel[FEATURE].rolling(RANK_WINDOW, min_periods=10).count()
signal = ((rolling_rank - 1.0) / (effective_window - 1.0)) * 2.0 - 1.0
if SMOOTH_WINDOW:
    signal = signal.rolling(SMOOTH_WINDOW).mean()
position = (signal * DIRECTION).clip(-1.0, 1.0)

# 3. Transaction-cost-aware position backtest.
fwd_ret = panel["fwd_ret"].to_numpy()
pos = position.to_numpy()
valid = np.isfinite(pos) & np.isfinite(fwd_ret)
pos, fwd_ret = pos[valid], fwd_ret[valid]
turnover = np.abs(np.diff(pos, prepend=0.0))
net = pos * fwd_ret - turnover * COST_BPS / 1e4
equity = np.cumprod(1.0 + net)

bars_per_year = 288 * 365  # 5-minute bars
std = net.std(ddof=1)
sharpe = net.mean() / std * np.sqrt(bars_per_year) if std > 0 else 0.0
net_return_pct = (equity[-1] / equity[0] - 1.0) * 100.0
running_max = np.maximum.accumulate(equity)
max_drawdown_pct = ((equity - running_max) / running_max).min() * 100.0

print("feature =", FEATURE, "| rank =", RANK_WINDOW, "| smooth =", SMOOTH_WINDOW,
      "| direction =", DIRECTION)
print("annualized_sharpe = %.3f | net_return_pct = %.3f | max_drawdown_pct = %.3f"
      % (sharpe, net_return_pct, max_drawdown_pct))
'''

display(
    Markdown(
        "#### Standalone snippet — reproduce this configuration\n\n"
        "```python\n" + snippet + "\n```"
    )
)

print(
    "Selected configuration ·",
    f"feature={best_feature} · metric={best_metric} · rank={best_rank} · "
    f"smooth={best_smooth} · direction={'long' if int(best['direction']) == 1 else 'short'}",
)
