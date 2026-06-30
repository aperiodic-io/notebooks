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
# A structured, end-to-end walk through the alpha-discovery workflow. The research
# logic matches the main notebook; here it is decomposed into discrete, documented
# stages so each part of the pipeline can be examined on its own.
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
# **validation** — across six steps. Run the cell below for the color-coded
# roadmap; each step that follows maps to one node.

# %%
from utils._alpha_discovery import pipeline_diagram

pipeline_diagram()

# %% [markdown]
# **Using this notebook.** Run the two setup cells (configuration, then the helper
# import), then proceed through Steps 1–6 in order. Each step pairs a brief rationale
# with a single code cell. The closing section provides a standalone snippet for
# reproducing the selected configuration.

# %% [markdown]
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from utils._alpha_discovery import WalkthroughConfig

API_KEY = "..."  # Set via APERIODIC_API_KEY env var or .env file
if API_KEY == "...":
    API_KEY = os.getenv("APERIODIC_API_KEY", "...")
if API_KEY == "...":
    raise RuntimeError("Set APERIODIC_API_KEY in the environment or in .env.")

config = WalkthroughConfig(
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
        ("returns", "regular"),
        ("slippage", "regular"),
        ("trade_size", "regular"),
        ("updownticks", "regular"),
        ("run_structure", "regular"),
        ("vtwap", "regular"),
        ("range", "regular"),
    ],
    rank_windows=[100, 300, 600, 1200, 2400],
    smooth_windows=[None, 10, 50, 100, 200, 400],
    cost_bps=1.0,
)

# Display parameters for the tables and plots below.
TOP_STRATEGY_COUNT = 10
TOP_PLOT_COUNT = 3

# %% [markdown]
# ---
# ## Setup — Helper functions
#
# The data retrieval, signal construction, search, and plotting routines live in
# [`utils/_alpha_discovery.py`](utils/_alpha_discovery.py) so the notebook stays
# focused on method rather than plumbing. They are imported below; each step
# describes the routine it calls, and the module holds the full implementation.

# %%
from utils._alpha_discovery import (
    build_decile_summary,
    build_panel,
    evaluate_strategies,
    plot_strategy_overview,
    summarize_panel,
    summarize_top_strategies,
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
# Because every feature is evaluated against every `(rank_window, smooth_window)`
# pair, the search space has cardinality `features × rank_windows × smooth_windows`:

# %%
search_space = pd.DataFrame(
    [
        {
            "features": len(feature_cols),
            "rank_windows": len(config.rank_windows),
            "smooth_windows": len(config.smooth_windows),
            "total_combinations": (
                len(feature_cols) * len(config.rank_windows) * len(config.smooth_windows)
            ),
        }
    ]
)
print(search_space.to_markdown(index=False))

# %% [markdown]
# ---
# ## Step 3 — Search the parameter grid
#
# `evaluate_strategies` traverses the full grid: for each combination it constructs
# the signal, assigns a direction, and runs an identical transaction-cost-aware
# position backtest. Each valid candidate is scored on annualized Sharpe, net
# return, and maximum drawdown, then ranked by Sharpe.

# %%
results_df, forward_returns = evaluate_strategies(config, panel, feature_cols)
top_strategies = summarize_top_strategies(results_df, TOP_STRATEGY_COUNT)

print("Top strategies by Sharpe:")
print(top_strategies.to_markdown(index=False))

# %% [markdown]
# ---
# ## Step 4 — Compare the leading candidates
#
# The ranking identifies the strongest candidates; the plots characterize their
# behavior. For the leading candidates we show the raw ranked signal, the smoothed
# signal, and the resulting equity curve.

# %%
for plot_idx, (_, row) in enumerate(results_df.head(TOP_PLOT_COUNT).iterrows(), start=1):
    plot_strategy_overview(config, panel, forward_returns, row, f"Candidate {plot_idx}")

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
    config, panel, forward_returns, best, "Best strategy"
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
# The heatmap shows the top feature's Sharpe across the full `rank_window ×
# smooth_window` grid. An isolated high-Sharpe cell among weak neighbors indicates
# overfitting; a contiguous high-Sharpe region indicates the edge degrades
# gracefully under parameter perturbation, which is characteristic of a robust
# signal.
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
# ---
# ## Recap
#
# This walkthrough executed the complete workflow:
#
# - **Loaded** price and microstructure features and constructed a one-bar-ahead
#   return target.
# - **Constructed** scale-free ranked signals, with optional smoothing.
# - **Searched** the full feature × window grid, scoring every candidate on a
#   cost-aware backtest.
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
rank = panel[FEATURE].rolling(RANK_WINDOW).rank(method="average")
signal = ((rank - 1.0) / (RANK_WINDOW - 1)) * 2.0 - 1.0
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
