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
# # Reading positioning and crowding risk
#
# **Use case: [Read positioning & risk](https://aperiodic.io/use-cases/positioning-and-risk)**
#
# Funding, basis, and open interest each describe a different part of derivatives
# positioning pressure. This notebook combines them into a transparent **crowding
# score** built from trailing rolling percentile ranks, then examines forward
# returns, realised volatility, and in-window drawdowns when the leverage
# indicators point the same way.
#
# We use **Binance BTC perpetuals** (`perpetual-BTC-USDT:USDT`) at **5-minute**
# resolution over **May 2025**, the window served by the shared `DEMO-KEY` preview
# slice.
#
# > **One month is an illustration, not evidence.** Conditional returns, Sharpe-like
# > summaries, and drawdown thresholds below come from a single month of 5-minute
# > data. The observed thresholds will not necessarily generalise; a high score is
# > a **risk-state diagnostic**, not a crash forecast.
#
# **Background reading**
# - Aperiodic derivatives metrics: https://aperiodic.io/metrics/derivatives
# - Binance Academy, funding rates: https://www.binance.com/en/academy/articles/what-are-funding-rates-in-crypto-markets

# %% [markdown]
# ## 1. Crowded leverage builds before it unwinds
#
# Aperiodic's derivatives metrics describe positioning from three angles:
#
# - **Funding** shows financing pressure — persistently positive funding means longs
#   are paying to stay long.
# - **Basis** shows the futures-versus-spot dislocation — a rich basis is a crowded,
#   levered-long tell.
# - **Open interest** shows how much leverage is outstanding, and whether it is
#   building or unwinding.
#
# No single series is decisive. Agreement across all three — financing, dislocation,
# and outstanding leverage all stretched the same way — is more informative than any
# one on its own, because that is when a crowded trade is most vulnerable to a
# sharp unwind.

# %%
from __future__ import annotations

import os
from datetime import date

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aperiodic import get_derivative_metrics, get_ohlcv

sns.set_theme(style="whitegrid", context="talk", palette="deep")
pd.options.display.float_format = "{:,.4f}".format
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 140
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# %% [markdown]
# ## 2. Configuration
#
# Uppercase constants hold the public parameters. The forward horizons are
# expressed in 5-minute bars; the helper below prints their human-readable
# durations. The API key resolves **inline value → `APERIODIC_API_KEY` →
# `DEMO-KEY`**.

# %%
EXCHANGE = "binance-futures"
SYMBOL = "perpetual-BTC-USDT:USDT"
INTERVAL = "5m"
START_DATE = date(2025, 5, 1)
END_DATE = date(2025, 5, 31)
TIMESTAMP = "exchange"
START_TS = pd.Timestamp(START_DATE)
END_TS = pd.Timestamp(END_DATE) + pd.Timedelta(days=1)

# Trailing window (bars) for rolling percentile ranks: 7 days of 5-minute bars.
RANK_WINDOW = 7 * 24 * 12
# Forward horizons (bars) for the event study: 1 hour, 6 hours, 1 day.
FORWARD_HORIZONS = [12, 72, 288]
# Pre-declared crowding-score thresholds for the calibration table.
SCORE_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]

API_KEY = "..."  # Set your key here, or via the APERIODIC_API_KEY env var
if API_KEY == "...":
    API_KEY = os.environ.get("APERIODIC_API_KEY", "DEMO-KEY")

# Only the shared "DEMO-KEY" runs against the preview endpoint. Provide your own
# key (above or via APERIODIC_API_KEY) to use the standard endpoint.
USE_PREVIEW = API_KEY == "DEMO-KEY"


def horizon_label(bars: int) -> str:
    minutes = bars * 5
    if minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"


print(f"Exchange / symbol : {EXCHANGE} / {SYMBOL}")
print(f"Interval / window : {INTERVAL} | {START_DATE} → {END_DATE}")
print(f"Rank window       : {RANK_WINDOW} bars ({horizon_label(RANK_WINDOW)})")
print(f"Forward horizons  : {[f'{h} bars ({horizon_label(h)})' for h in FORWARD_HORIZONS]}")
print(f"Mode              : {'preview (DEMO-KEY)' if USE_PREVIEW else 'standard endpoint'}")

# %% [markdown]
# ## Helper functions
#
# `clip_window` and `format_time_axis` match the introductory notebooks;
# `rolling_rank` is the trailing percentile-rank transform from
# `alpha-discovery.py`; `forward_return`, `forward_vol`, and `forward_max_drawdown`
# summarise the path over the next *h* bars using only future prices for the
# *outcome* (never for the score); and `audit_columns` prints the fetched schema
# and fails loudly if a required field is absent.

# %% {"jupyter": {"source_hidden": true}}
def clip_window(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out = out.loc[(out["time"] >= START_TS) & (out["time"] < END_TS)]
    return out.sort_values("time").reset_index(drop=True)


def format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def audit_columns(frames: dict[str, tuple[pd.DataFrame, list[str]]]) -> None:
    lines, problems = [], []
    for name, (frame, required) in frames.items():
        cols = sorted(map(str, frame.columns))
        missing = [c for c in required if c not in frame.columns]
        lines.append(f"- {name:<16} {len(frame):>5} rows | columns={cols}")
        if missing:
            problems.append(f"{name} missing {missing}")
    report = "Fetched schema:\n" + "\n".join(lines)
    print(report)
    if problems:
        raise KeyError("; ".join(problems) + "\n" + report)


def rolling_rank(series: pd.Series, window: int) -> pd.Series:
    """Trailing rolling percentile rank in [0, 1] (alpha-discovery transform).
    Uses only data up to and including the current bar — no look-ahead."""
    ranks = series.rolling(window, min_periods=10).rank(method="average")
    counts = series.rolling(window, min_periods=10).count()
    return (ranks - 1.0) / (counts - 1.0)


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return from bar t to bar t+horizon."""
    return close.pct_change(horizon).shift(-horizon)


def forward_vol(ret: pd.Series, horizon: int) -> pd.Series:
    """Std of per-bar returns over the next `horizon` bars (reverse-rolling)."""
    rev = ret[::-1]
    fv = rev.rolling(horizon, min_periods=max(2, horizon // 4)).std()
    return fv[::-1]


def forward_max_drawdown(close: pd.Series, horizon: int) -> pd.Series:
    """Most negative close-to-min return over the next `horizon` bars (<= 0)."""
    rev = close[::-1]
    fwd_min = rev.rolling(horizon, min_periods=1).min()[::-1]
    return fwd_min / close - 1.0


# %% [markdown]
# ## 3. Fetch and align derivative data
#
# Five requests share the common arguments (`api_key`, `preview`, `timestamp`,
# `interval`, `exchange`, `symbol`, `start_date`, `end_date`, `output="pandas"`,
# `show_progress=False`): price, funding, basis, open interest, and derivative
# price. We align on `time` without look-ahead, forward-filling sparse derivative
# observations only forward. Derivative price is kept as a **cross-check** on the
# OHLCV return source, not a replacement.

# %%
COMMON = dict(
    api_key=API_KEY,
    preview=USE_PREVIEW,
    timestamp=TIMESTAMP,
    interval=INTERVAL,
    exchange=EXCHANGE,
    symbol=SYMBOL,
    start_date=START_DATE,
    end_date=END_DATE,
    output="pandas",
    show_progress=False,
)

ohlcv = clip_window(get_ohlcv(**COMMON))
funding = clip_window(get_derivative_metrics(metric="funding", **COMMON))
basis = clip_window(get_derivative_metrics(metric="basis", **COMMON))
open_interest = clip_window(get_derivative_metrics(metric="open_interest", **COMMON))
derivative_price = clip_window(get_derivative_metrics(metric="derivative_price", **COMMON))

audit_columns(
    {
        "ohlcv": (ohlcv, ["time", "close"]),
        "funding": (funding, ["time", "funding_rate"]),
        "basis": (basis, ["time", "basis_bps"]),
        "open_interest": (open_interest, ["time", "open_interest", "open_interest_pct_change"]),
        "derivative_price": (
            derivative_price,
            [
                "time",
                "last_price",
                "index_price",
                "mark_price",
                "last_price_pct_change",
                "index_price_pct_change",
                "mark_price_pct_change",
            ],
        ),
    }
)

# %%
df = (
    ohlcv[["time", "close"]]
    .merge(funding[["time", "funding_rate"]], on="time", how="left")
    .merge(basis[["time", "basis_bps"]], on="time", how="left")
    .merge(open_interest[["time", "open_interest", "open_interest_pct_change"]], on="time", how="left")
    .merge(
        derivative_price[["time", "mark_price", "mark_price_pct_change"]],
        on="time",
        how="left",
    )
    .sort_values("time")
    .reset_index(drop=True)
)

# Align sparse derivative series forward only (never pull later values backwards).
for col in ["funding_rate", "basis_bps", "open_interest", "open_interest_pct_change"]:
    df[col] = df[col].ffill()

df["ret"] = df["close"].pct_change()
df["funding_rate_bps"] = df["funding_rate"] * 1e4

# Cross-check: derivative mark-price return vs OHLCV close return (we keep close).
cross_check = df[["ret", "mark_price_pct_change"]].dropna()
mark_corr = cross_check["ret"].corr(cross_check["mark_price_pct_change"]) if len(cross_check) > 2 else np.nan

coverage = pd.Series(
    {
        "Rows (5m bars)": len(df),
        "Start": df["time"].min(),
        "End": df["time"].max(),
        "Mean funding (bps)": df["funding_rate_bps"].mean(),
        "Mean basis (bps)": df["basis_bps"].mean(),
        "Median open interest": df["open_interest"].median(),
        "corr(close ret, mark-price ret)": mark_corr,
    }
)
coverage

# %% [markdown]
# ## Chart 1 — Positioning components
#
# Funding, basis, and open interest with price context. These are the raw
# ingredients of the crowding score; the strongest coincident moves in May 2025 are
# visible, but they are features of this window, not universal regimes.

# %%
fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
axes[0].plot(df["time"], df["close"], color="#2563eb", linewidth=1.2)
axes[0].set_title("BTC perpetual close price")
axes[0].set_ylabel("Price (USDT)")

axes[1].plot(df["time"], df["funding_rate_bps"], color="#dc2626", linewidth=1.0)
axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
axes[1].set_title("Funding rate (bps)")
axes[1].set_ylabel("bps")

axes[2].plot(df["time"], df["basis_bps"], color="#7c3aed", linewidth=1.0)
axes[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
axes[2].set_title("Basis (bps)")
axes[2].set_ylabel("bps")

axes[3].plot(df["time"], df["open_interest"], color="#059669", linewidth=1.1)
axes[3].set_title("Open interest")
axes[3].set_ylabel("OI")
format_time_axis(axes[3])
plt.tight_layout()

# %% [markdown]
# ## 5. Build the crowding score
#
# Each component is mapped to a trailing rolling percentile rank over `RANK_WINDOW`
# bars (7 days), so a value near **1** means "unusually stretched *long*" and near
# **0** means "unusually stretched *short*" relative to the recent past:
#
# - `funding_rate` — higher rank = longs paying more.
# - `basis_bps` — higher rank = richer futures premium.
# - `open_interest_pct_change` — higher rank = leverage building fastest.
#
# The composite is the **simple mean** of the three ranks (a transparent,
# fixed-weight combination). Because ranks preserve direction, the composite sits
# in [0, 1] with **> 0.5 = long crowding, < 0.5 = short crowding** — the sign is
# kept, not collapsed to magnitude. Ranks and the composite use trailing data only.

# %%
df["rank_funding"] = rolling_rank(df["funding_rate"], RANK_WINDOW)
df["rank_basis"] = rolling_rank(df["basis_bps"], RANK_WINDOW)
df["rank_oi_change"] = rolling_rank(df["open_interest_pct_change"], RANK_WINDOW)
df["crowding_score"] = df[["rank_funding", "rank_basis", "rank_oi_change"]].mean(axis=1)

# Forward outcomes for the event study (future prices used only for the outcome).
for h in FORWARD_HORIZONS:
    df[f"fwd_ret_{h}"] = forward_return(df["close"], h)
    df[f"fwd_vol_{h}"] = forward_vol(df["ret"], h)
    df[f"fwd_maxdd_{h}"] = forward_max_drawdown(df["close"], h)

fig, (ax_comp, ax_score) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
ax_comp.plot(df["time"], df["rank_funding"], label="funding rank", color="#dc2626", linewidth=0.9, alpha=0.8)
ax_comp.plot(df["time"], df["rank_basis"], label="basis rank", color="#7c3aed", linewidth=0.9, alpha=0.8)
ax_comp.plot(df["time"], df["rank_oi_change"], label="OI-change rank", color="#059669", linewidth=0.9, alpha=0.8)
ax_comp.set_title("Component percentile ranks (trailing 7-day)")
ax_comp.set_ylabel("Rank [0, 1]")
ax_comp.legend(frameon=True, fontsize=10, ncol=3)

ax_score.plot(df["time"], df["crowding_score"], color="#111827", linewidth=1.3)
ax_score.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="neutral (0.5)")
ax_score.fill_between(df["time"], df["crowding_score"], 0.5, where=df["crowding_score"] > 0.5, color="#dc2626", alpha=0.15, label="long crowding")
ax_score.fill_between(df["time"], df["crowding_score"], 0.5, where=df["crowding_score"] < 0.5, color="#16a34a", alpha=0.15, label="short crowding")
ax_score.set_title("Composite crowding score (mean of the three ranks)")
ax_score.set_ylabel("Score [0, 1]")
ax_score.legend(frameon=True, fontsize=10, ncol=3)
format_time_axis(ax_score)
plt.tight_layout()

# %% [markdown]
# ## Chart 2 — Event study by crowding decile
#
# Bucket every bar by its composite-score decile and compare forward returns and
# realised volatility across the horizons. Sample counts are shown because the
# windows **overlap** — these are not independent observations, so the decile means
# are indicative, not statistically independent estimates.

# %%
event = df.dropna(subset=["crowding_score"]).copy()
event["score_decile"] = pd.qcut(event["crowding_score"], 10, labels=list(range(1, 11)), duplicates="drop")

ret_by_decile = event.groupby("score_decile", observed=True)[[f"fwd_ret_{h}" for h in FORWARD_HORIZONS]].mean() * 100
vol_by_decile = event.groupby("score_decile", observed=True)[[f"fwd_vol_{h}" for h in FORWARD_HORIZONS]].mean() * 1e4
counts = event.groupby("score_decile", observed=True)["crowding_score"].count()

print("Sample count per decile:")
print(counts.to_markdown())

fig, (ax_r, ax_v) = plt.subplots(1, 2, figsize=(16, 6))
width = 0.25
x = np.arange(len(ret_by_decile.index))
for i, h in enumerate(FORWARD_HORIZONS):
    ax_r.bar(x + (i - 1) * width, ret_by_decile[f"fwd_ret_{h}"], width=width, label=horizon_label(h))
ax_r.axhline(0, color="black", linewidth=0.8)
ax_r.set_title("Mean forward return by crowding decile")
ax_r.set_xlabel("Crowding-score decile (1 = short-crowded, 10 = long-crowded)")
ax_r.set_ylabel("Mean forward return (%)")
ax_r.set_xticks(x, ret_by_decile.index.astype(str))
ax_r.legend(frameon=True, title="Horizon")

for i, h in enumerate(FORWARD_HORIZONS):
    ax_v.plot(ret_by_decile.index.astype(int), vol_by_decile[f"fwd_vol_{h}"], marker="o", label=horizon_label(h))
ax_v.set_title("Mean forward realised volatility by decile")
ax_v.set_xlabel("Crowding-score decile")
ax_v.set_ylabel("Forward per-bar vol (bps)")
ax_v.legend(frameon=True, title="Horizon")
plt.tight_layout()

# %% [markdown]
# ## Chart 3 — Sharpest in-window moves
#
# The largest adverse forward moves at the 1-day horizon, annotated with the
# crowding score **known immediately before** each move (a trailing input). This
# asks whether the score was already elevated ahead of the sharpest unwinds — for
# this window only.

# %%
h_long = FORWARD_HORIZONS[-1]
moves = df.dropna(subset=[f"fwd_ret_{h_long}", "crowding_score"]).copy()
sharp = moves.nsmallest(6, f"fwd_ret_{h_long}")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df["time"], df["close"], color="#2563eb", linewidth=1.1)
for _, r in sharp.iterrows():
    ax.scatter(r["time"], r["close"], color="#dc2626", s=60, zorder=5)
    ax.annotate(
        f"{r['fwd_ret_' + str(h_long)] * 100:.1f}% / {horizon_label(h_long)}\nscore={r['crowding_score']:.2f}",
        (r["time"], r["close"]),
        textcoords="offset points",
        xytext=(0, 14),
        fontsize=9,
        ha="center",
        color="#991b1b",
    )
ax.set_title(f"Sharpest {horizon_label(h_long)} forward drops and the crowding score just before")
ax.set_ylabel("Price (USDT)")
format_time_axis(ax)
plt.tight_layout()

print(
    sharp[["time", "crowding_score", f"fwd_ret_{h_long}", f"fwd_maxdd_{h_long}"]]
    .assign(**{f"fwd_ret_{h_long}": lambda d: d[f"fwd_ret_{h_long}"] * 100, f"fwd_maxdd_{h_long}": lambda d: d[f"fwd_maxdd_{h_long}"] * 100})
    .to_markdown(index=False)
)

# %% [markdown]
# ## 8. Threshold calibration table
#
# For a small, pre-declared set of crowding-score thresholds, we report the
# observation count, subsequent return, realised volatility, and maximum subsequent
# drawdown at the 1-day horizon. This is **in-window calibration** — a description
# of what happened in May 2025 above each threshold — not a production threshold or
# an out-of-sample validation.

# %%
h = FORWARD_HORIZONS[-1]
cal = df.dropna(subset=["crowding_score", f"fwd_ret_{h}", f"fwd_vol_{h}", f"fwd_maxdd_{h}"])
rows = []
for thr in SCORE_THRESHOLDS:
    sub = cal[cal["crowding_score"] >= thr]
    rows.append(
        {
            "score >= ": thr,
            "count": int(len(sub)),
            "share_of_sample_%": 100.0 * len(sub) / len(cal) if len(cal) else np.nan,
            f"mean_fwd_ret_{horizon_label(h)}_%": sub[f"fwd_ret_{h}"].mean() * 100,
            f"mean_fwd_vol_{horizon_label(h)}_bps": sub[f"fwd_vol_{h}"].mean() * 1e4,
            f"mean_fwd_maxdd_{horizon_label(h)}_%": sub[f"fwd_maxdd_{h}"].mean() * 100,
        }
    )
calibration_table = pd.DataFrame(rows)
print(f"In-window calibration at the {horizon_label(h)} horizon (illustrative):")
print(calibration_table.to_markdown(index=False))

# %% [markdown]
# ## Takeaways
#
# - **Three angles, one score.** Funding, basis, and open-interest change agree only
#   sometimes; the composite highlights when financing, dislocation, and leverage are
#   stretched together — the state most exposed to a sharp unwind.
# - **A diagnostic, not a forecast.** The decile event study and threshold table
#   describe what followed high scores *in this window*; they are in-window
#   calibration, and the overlapping forward windows are not independent samples.
# - **Sign preserved.** The score keeps long-vs-short crowding (above/below 0.5)
#   rather than collapsing to magnitude, so short-side crowding is visible too.
# - **One month proves nothing.** Every conditional return, volatility, and drawdown
#   here is illustrative; ongoing monitoring would need Live/Institutional data.
#
# ## Further reading
# - [Read positioning & risk](https://aperiodic.io/use-cases/positioning-and-risk)
# - [Derivatives metrics](https://aperiodic.io/metrics/derivatives)
# - Related notebooks: `notebooks/intro-derivatives.py`, `notebooks/alpha-discovery.py`
