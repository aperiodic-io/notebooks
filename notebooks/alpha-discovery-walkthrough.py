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
# # Alpha Discovery Walkthrough — a mini course
#
# Welcome. This is a **guided, step-by-step** version of the alpha-discovery
# workflow — the same research logic as the main notebook, broken into bite-sized
# lessons you can read top to bottom (or jump around using the roadmap below).
#
# **What you'll learn**
#
# - how raw microstructure metrics become tradeable, scale-free **signals**
# - how to search a feature × parameter **grid** and score every candidate
# - how to tell a genuinely **robust** edge from a lucky parameter pair
# - how to **grab the winning configuration** and reuse it in your own code
#
# *Reading time ≈ 10 min · runs end to end on preview data.*
#
# ### The pipeline
#
# Six steps take you from raw market data to a validated, reusable signal. Run the
# cell below for the roadmap — every step that follows maps to one box.

# %%
from utils._alpha_discovery import pipeline_diagram

pipeline_diagram()

# %% [markdown]
# **How to take this course.** Run the two **Setup** cells (Configuration, then
# Toolbox), then work through Steps 1–6 in order. Each step is a short "what & why"
# note followed by a single code cell. The final section hands you a copy-paste
# snippet so you can take the result with you.

# %% [markdown]
# ---
# ## Setup — Configuration
#
# Everything you might want to tweak lives in one place: the instrument and date
# range, which metric families to pull, and the two search dimensions. Edit this
# cell, re-run the notebook, and every step downstream picks up your changes.
#
# - **`rank_windows`** — how much history to use when turning a feature into a
#   percentile-rank signal (longer = slower, steadier).
# - **`smooth_windows`** — moving-average lengths applied *after* ranking.
#   `None` means "use the raw ranked signal." This is the second search dimension.
# - **`cost_bps`** — one-way transaction cost charged by the backtest.

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

# Display knobs for the tables and plots below.
TOP_STRATEGY_COUNT = 10
TOP_PLOT_COUNT = 3

# %% [markdown]
# ---
# ## Setup — Toolbox
#
# The data-loading, signal-building, search, and plotting helpers live in
# [`utils/_alpha_discovery.py`](utils/_alpha_discovery.py) so this notebook can
# stay focused on the *ideas*. We import them here in one line; each step below
# explains what the helper it uses does, and you can open the module any time to
# see exactly how it works.

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
# ## Step 1 — Load & inspect the data
#
# `build_panel` pulls close price plus every numeric column from the configured
# metric families, **forward-fills** the sparse ones, and builds the prediction
# target: the **next bar's return** (`fwd_ret`).
#
# Before trusting any result built on top of it, sanity-check the data — its
# shape, a few example rows, and how well each feature is populated.

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
# ## Step 2 — Build the signal
#
# A raw feature isn't tradeable on its own — its scale drifts and its units are
# arbitrary. Two transforms fix that:
#
# 1. **Rank.** Inside a rolling window of `rank_window` bars, replace each value
#    with its percentile rank, then rescale to `[-1, +1]`. High means "unusually
#    high versus recent history," low means "unusually low" — scale-free and
#    comparable across features.
# 2. **Smooth** *(optional)*. Average the ranked signal over `smooth_window` bars
#    to trade responsiveness for stability. `None` keeps the raw ranked signal.
#
# The strategy **direction** (long or short) is taken from the sign of the
# in-sample correlation between the signal and the next-bar return.
#
# Because every feature is tried against every `(rank_window, smooth_window)`
# pair, the search space is `features × rank_windows × smooth_windows`:

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
# ## Step 3 — Search the grid
#
# `evaluate_strategies` walks the whole grid: for each combination it builds the
# signal, picks a direction, and runs the same cost-aware position backtest. Every
# valid candidate gets a Sharpe, a net return, and a max drawdown. We sort by
# Sharpe and look at the strongest few.

# %%
results_df, forward_returns = evaluate_strategies(config, panel, feature_cols)
top_strategies = summarize_top_strategies(results_df, TOP_STRATEGY_COUNT)

print("Top strategies by Sharpe:")
print(top_strategies.to_markdown(index=False))

# %% [markdown]
# ---
# ## Step 4 — Compare the strongest candidates
#
# A table tells you *which* candidates won; the plots tell you *how* they behave.
# For the top few we show the raw ranked signal, the smoothed signal, and the
# resulting equity curve, so you can build intuition for what a "good" candidate
# actually looks like.

# %%
for plot_idx, (_, row) in enumerate(results_df.head(TOP_PLOT_COUNT).iterrows(), start=1):
    plot_strategy_overview(config, panel, forward_returns, row, f"Candidate {plot_idx}")

# %% [markdown]
# ---
# ## Step 5 — Deep-dive the best strategy
#
# Now zoom in on the single best candidate: its signal shape and equity curve,
# then a **decile test**. Bucket the signal into ten groups and check that the
# mean next-bar return climbs (or falls) steadily across buckets. A clean,
# monotonic staircase is far more convincing than one extreme bucket doing all
# the work.

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
# *Is the edge real, or did we just get lucky with one parameter pair?*
#
# The heatmap shows the best feature's Sharpe across the whole `rank_window` ×
# `smooth_window` grid. A single hot cell surrounded by cold ones screams
# **overfit** — the result hinges on one exact setting. A broad warm region means
# the edge **degrades gracefully** as you change the knobs, which is what a
# trustworthy signal looks like.
#
# > **Scope:** this tests robustness to *parameter choice* on in-sample data. It
# > is a sanity check, not a substitute for true out-of-sample testing — see
# > *Next steps* below.

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
# You ran the whole loop end to end:
#
# - **Loaded** price and microstructure features and built a next-bar return target.
# - **Turned** raw features into scale-free ranked signals, optionally smoothed.
# - **Searched** the full feature × window grid and scored every candidate.
# - **Inspected** the winners — equity curve, decile monotonicity, and parameter
#   robustness.
#
# The key idea: smoothing is a *second search dimension*, so a feature can win
# either because its raw ranked signal works or because a slower version of it is
# cleaner.

# %% [markdown]
# ---
# ## Next steps — grab these metrics with this snippet
#
# The most valuable follow-up is **out-of-sample validation**: rerun the winning
# configuration on a later date range, a different symbol, or another exchange. If
# the edge survives, it is worth a closer look.
#
# Run the cell below to print a **copy-paste-ready snippet**, pre-filled with this
# run's winning feature and parameters, that re-fetches the source metric and
# rebuilds the exact signal on a fresh window.

# %%
best_feature = str(best["feature"])
new_metric, new_kind = feature_sources.get(best_feature, (best_feature, "regular"))
new_rank = int(best["rank_window"])
new_smooth = (
    None
    if best["smooth_window"] is None or pd.isna(best["smooth_window"])
    else int(best["smooth_window"])
)

snippet = f'''import datetime

from utils._alpha_discovery import WalkthroughConfig, build_panel, evaluate_strategies

# This run's winner: {best_feature!r} (from the {new_metric!r} metric family).
# Re-fetch it on a fresh, OUT-OF-SAMPLE window and rebuild the exact signal.
config = WalkthroughConfig(
    api_key="YOUR_KEY",
    symbol="{config.symbol}",
    exchange="{config.exchange}",
    interval="{config.interval}",
    timestamp="{config.timestamp}",
    start_date=datetime.date(2025, 6, 1),
    end_date=datetime.date(2025, 6, 30),
    metrics=[("{new_metric}", "{new_kind}")],
    rank_windows=[{new_rank}],
    smooth_windows=[{new_smooth}],
    cost_bps={config.cost_bps},
)

panel, feature_cols, _ = build_panel(config)
results, _ = evaluate_strategies(config, panel, feature_cols)

winner = results[results["feature"] == "{best_feature}"].iloc[0]
print(winner[["feature", "rank_window", "smooth_window", "sharpe", "return_pct", "drawdown_pct"]])
'''

display(Markdown("#### Grab these metrics with this snippet\n\n```python\n" + snippet + "\n```"))

print(
    "This run's winner ·",
    f"feature={best_feature} · metric={new_metric} · rank={new_rank} · "
    f"smooth={new_smooth} · sharpe={float(best['sharpe']):.3f}",
)
