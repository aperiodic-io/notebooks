# ---
# jupyter:
#   aperiodic:
#     uses_preview_data: true
#   jupytext:
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
# # Alpha Discovery Walkthrough
#
# This notebook is a guided version of the alpha-discovery workflow.
# It keeps the same research logic as the main notebook, but breaks the work into
# smaller steps so it is easier to follow.
#
# The pipeline is:
# 1. load BTC perpetual price and feature data
# 2. forward fill sparse feature columns
# 3. convert each feature into a rolling percentile-rank signal
# 4. optionally smooth that signal
# 5. backtest every feature/window combination
# 6. compare the strongest strategies and inspect the best one in detail

# %% [markdown]
# ## Configuration
#
# `RANK_WINDOWS` control how much history is used when turning a feature into a
# percentile-rank signal.
#
# `SMOOTH_WINDOWS` are applied after the rank signal is built. `None` means
# "use the raw ranked signal without smoothing."
#
# `COST_BPS` is the one-way transaction cost used in the simple position backtest.

# %%

from __future__ import annotations

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aperiodic import get_derivative_metrics, get_metrics, get_ohlcv
from utils._aperiodic_demo import run_position_backtest

SYMBOL = "perpetual-BTC-USDT:USDT"
EXCHANGE = "binance-futures"
INTERVAL = "5m"
TIMESTAMP = "exchange"  # local timestamp or "true"

START_DATE = datetime.date(2025, 5, 1)
END_DATE = datetime.date(2025, 5, 31)

METRICS = [
    ("basis", "derivative"),
    ("funding", "derivative"),
    ("open_interest", "derivative"),
    ("flow", "regular"),
    ("impact", "regular"),
    ("returns", "regular"),
    ("slippage", "regular"),
    ("trade_size", "regular"),
    ("updownticks", "regular"),
    ("run_structure", "regular"),
    ("vtwap", "regular"),
    ("range", "regular"),
]

RANK_WINDOWS = [100, 300, 600, 1200, 2400]
SMOOTH_WINDOWS = [None, 10, 50, 100, 200, 400]
COST_BPS = 1.0
TOP_STRATEGY_COUNT = 10
TOP_PLOT_COUNT = 3

API_KEY = "..."  # Set via APERIODIC_API_KEY env var or .env file
if API_KEY == "...":
    API_KEY = os.getenv("APERIODIC_API_KEY", "...")
if API_KEY == "...":
    raise RuntimeError("Set APERIODIC_API_KEY in the environment or in .env.")


def get_numeric_metric_frame(metric: str, kind: str) -> pd.DataFrame | None:
    fetcher = get_derivative_metrics if kind == "derivative" else get_metrics
    raw_df = fetcher(
        api_key=API_KEY,
        metric=metric,
        timestamp=TIMESTAMP,
        interval=INTERVAL,
        exchange=EXCHANGE,
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        output="pandas",
        show_progress=True,
        preview=True,
    )

    df = raw_df.to_pandas() if hasattr(raw_df, "to_pandas") else pd.DataFrame(raw_df)
    if df.empty or "time" not in df.columns:
        print(f"Skipping {metric}: no rows returned.")
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "time" in numeric_cols:
        numeric_cols.remove("time")

    if not numeric_cols:
        print(f"Skipping {metric}: no numeric columns returned.")
        return None

    return df.sort_values("time").drop_duplicates(subset=["time"], keep="last")[
        ["time", *numeric_cols]
    ]


def build_panel() -> tuple[pd.DataFrame, list[str]]:
    raw_ohlcv = get_ohlcv(
        api_key=API_KEY,
        timestamp=TIMESTAMP,
        interval=INTERVAL,
        exchange=EXCHANGE,
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        output="pandas",
        show_progress=True,
        preview=True,
    )

    panel = (
        raw_ohlcv.to_pandas() if hasattr(raw_ohlcv, "to_pandas") else pd.DataFrame(raw_ohlcv)
    )
    panel = panel.sort_values("time")[["time", "close"]]

    for metric, kind in METRICS:
        frame = get_numeric_metric_frame(metric, kind)
        if frame is None:
            continue

        feature_values = [col for col in frame.columns if col != "time"]
        if frame[feature_values].notna().sum().sum() == 0:
            print(f"Skipping {metric}: all feature values are null.")
            continue

        panel = panel.merge(frame, on="time", how="left")

    panel = panel.sort_values("time")
    feature_cols = [
        col
        for col in panel.columns
        if col not in {"time", "close"}
        and pd.api.types.is_numeric_dtype(panel[col])
    ]

    panel[feature_cols] = panel[feature_cols].ffill()
    panel["fwd_ret"] = panel["close"].pct_change().shift(-1)
    panel = panel.dropna(subset=["fwd_ret"])
    return panel, feature_cols


def summarize_panel(panel_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    coverage_rows = []
    for feature in feature_cols[:10]:
        coverage_rows.append(
            {
                "feature": feature,
                "non_null_pct": float(panel_df[feature].notna().mean() * 100.0),
                "std": float(panel_df[feature].std()),
            }
        )
    return pd.DataFrame(coverage_rows)


def make_rank_signal(panel_df: pd.DataFrame, feature: str, window: int) -> np.ndarray:
    rank = panel_df[feature].rolling(window).rank(method="average")
    signal = ((rank - 1.0) / (window - 1)) * 2.0 - 1.0
    return signal.to_numpy().astype(np.float64)


def smooth_signal(signal: np.ndarray, window: int | None) -> np.ndarray:
    if window is None or pd.isna(window) or int(window) == 1:
        return signal.astype(np.float64, copy=True)

    return pd.Series(signal).rolling(int(window)).mean().to_numpy().astype(np.float64)


def evaluate_strategies(
    panel_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    forward_returns = panel_df["fwd_ret"].to_numpy().astype(np.float64)
    results: list[dict[str, float | int | str | None]] = []

    for feature in feature_cols:
        for rank_window in RANK_WINDOWS:
            raw_signal = make_rank_signal(panel_df, feature, rank_window)

            for smooth_window in SMOOTH_WINDOWS:
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
                    cost_bps_one_way=COST_BPS,
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
    summary["smooth_window"] = summary["smooth_window"].map(
        lambda value: "None" if value is None or pd.isna(value) else int(value)
    )
    summary["direction"] = summary["direction"].map({1: "long", -1: "short"})
    return summary[
        [
            "feature",
            "rank_window",
            "smooth_window",
            "direction",
            "fit_corr",
            "sharpe",
            "return_pct",
            "drawdown_pct",
        ]
    ]


def build_strategy_artifacts(
    panel_df: pd.DataFrame, forward_returns: np.ndarray, row: pd.Series
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, float], np.ndarray]:
    raw_signal = make_rank_signal(panel_df, str(row["feature"]), int(row["rank_window"]))
    signal = smooth_signal(raw_signal, row["smooth_window"]) * int(row["direction"])
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    bt_frame, bt_summary = run_position_backtest(
        timestamps=panel_df.loc[mask, "time"],
        position=np.nan_to_num(np.clip(signal[mask], -1.0, 1.0), nan=0.0),
        forward_return=forward_returns[mask],
        cost_bps_one_way=COST_BPS,
    )
    return raw_signal, signal, bt_frame, bt_summary, mask


def plot_strategy_overview(
    panel_df: pd.DataFrame,
    forward_returns: np.ndarray,
    row: pd.Series,
    title_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw_signal, signal, bt_frame, _, mask = build_strategy_artifacts(
        panel_df, forward_returns, row
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

    axes[2].plot(bt_frame["timestamp"], bt_frame["equity_curve"], linewidth=1.1, color="tab:green")
    axes[2].set_title(
        f"{title_prefix} equity"
        f" | Sharpe={float(row['sharpe']):.3f}"
        f" | Ret={float(row['return_pct']):.3f}"
    )
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    plt.show()
    return signal, mask


def build_decile_summary(signal: np.ndarray, forward_returns: np.ndarray) -> pd.DataFrame:
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    signal_valid = signal[mask]
    returns_valid = forward_returns[mask]

    if signal_valid.size == 0:
        raise RuntimeError("Best strategy produced no valid observations for decile analysis.")

    order = np.argsort(signal_valid)
    deciles = np.empty(signal_valid.shape[0], dtype=np.int64)
    deciles[order] = (np.arange(signal_valid.shape[0]) * 10 // signal_valid.shape[0]) + 1

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
# ## Load And Inspect Data
#
# The panel combines close price with all numeric feature columns returned by the
# selected metric set. Feature columns are forward filled before the one-bar-ahead
# return target is created.

# %%
panel, feature_cols = build_panel()
panel_summary = summarize_panel(panel, feature_cols)

print(f"Rows: {len(panel):,}")
print(f"Feature count: {len(feature_cols)}")
print(f"First features: {feature_cols[:12]}")
print(panel[["time", "close", "fwd_ret"]].head().to_markdown(index=False))

print("\nFeature coverage snapshot:")
print(panel_summary.to_markdown(index=False))


# %% [markdown]
# ## Signal Construction
#
# A raw feature is first turned into a rolling percentile-rank signal in `[-1, 1]`.
# Smoothing is then applied to that ranked signal, not to the raw feature itself.
#
# The strategy direction is chosen from the sign of the in-sample correlation between
# the signal and the next-bar return.

# %%
search_space = pd.DataFrame(
    [
        {
            "features": len(feature_cols),
            "rank_windows": len(RANK_WINDOWS),
            "smooth_windows": len(SMOOTH_WINDOWS),
            "total_combinations": len(feature_cols) * len(RANK_WINDOWS) * len(SMOOTH_WINDOWS),
        }
    ]
)
print(search_space.to_markdown(index=False))


# %% [markdown]
# ## Run The Parameter Search
#
# Every feature is tested across all rank-window and smoothing-window combinations.
# Each valid combination is backtested with the same transaction-cost assumption.

# %%
results_df, forward_returns = evaluate_strategies(panel, feature_cols)
top_strategies = summarize_top_strategies(results_df, TOP_STRATEGY_COUNT)

print("Top strategies by Sharpe:")
print(top_strategies.to_markdown(index=False))


# %% [markdown]
# ## Compare The Strongest Candidates
#
# Start with a compact table, then plot a few of the strongest strategies to get an
# intuition for how different winning candidates behave.

# %%
for plot_idx, (_, row) in enumerate(results_df.head(TOP_PLOT_COUNT).iterrows(), start=1):
    plot_strategy_overview(panel, forward_returns, row, f"Candidate {plot_idx}")


# %% [markdown]
# ## Best Strategy Deep Dive
#
# Now focus on the single best strategy and inspect its signal shape, equity curve,
# decile monotonicity, and sensitivity to the two search dimensions.

# %%
best = results_df.iloc[0]
best_signal, best_mask = plot_strategy_overview(panel, forward_returns, best, "Best strategy")

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
# ## Feature And Window Diagnostics
#
# The heatmap below shows whether the best feature is only strong for one exact
# parameter pair or whether its edge is more stable across the search grid.

# %%
best_feature_results = results_df.loc[
    results_df["feature"] == best["feature"],
    ["rank_window", "smooth_window", "sharpe"],
].copy()
heatmap = best_feature_results.pivot(
    index="rank_window", columns="smooth_window", values="sharpe"
).sort_index()
heatmap = heatmap.rename(columns={None: "None"})

fig, ax = plt.subplots(figsize=(8, 4))
image = ax.imshow(heatmap.to_numpy(), aspect="auto", cmap="RdYlGn")
ax.set_title(f"Sharpe by window combination for {best['feature']}")
ax.set_xlabel("Smooth window")
ax.set_ylabel("Rank window")
ax.set_xticks(np.arange(len(heatmap.columns)), labels=heatmap.columns)
ax.set_yticks(np.arange(len(heatmap.index)), labels=heatmap.index)

for row_idx, row_values in enumerate(heatmap.to_numpy()):
    for col_idx, value in enumerate(row_values):
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
# ## Takeaways
#
# - The notebook searches for features whose ranked microstructure behavior predicts
#   the next 5-minute return.
# - Smoothing is a second dimension in the search, so a good feature can win either
#   because the raw ranked signal works or because a slower version of it is cleaner.
# - The most useful follow-up is usually out-of-sample validation on a longer range
#   or on a different exchange.
