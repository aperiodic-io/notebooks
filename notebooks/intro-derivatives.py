# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Intro to Aperiodic Derivatives Metrics
#
# This notebook introduces a simple **derivatives regime dashboard** using Aperiodic's
# `funding`, `open_interest`, and `basis` metrics for **Binance BTC perpetuals** over
# **September 1, 2025 → February 28, 2026** at **5-minute** frequency.
#
# ## Why derivatives metrics matter
#
# Derivatives data helps answer a different question than trade flow: **how crowded is positioning, and how expensive is leverage?**
#
# - **Funding** tracks the periodic transfer that keeps perpetuals aligned with spot.
# - **Open interest** helps show whether leverage is building or being unwound.
# - **Basis** measures the futures/perpetual premium or discount relative to a reference price.
#
# Together, these metrics are useful for spotting crowded bullish or bearish regimes, squeezes, and leverage resets.
#
# **Background reading**
# - Aperiodic derivatives market data: https://aperiodic.io/metrics/derivatives_market_data
# - Binance Academy, funding rates: https://www.binance.com/en/academy/articles/what-are-funding-rates-in-crypto-markets
# - CME Group, futures basis overview: https://www.cmegroup.com/education/courses/introduction-to-basis.html

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

EXCHANGE = "binance-futures"
SYMBOL = "perpetual-BTC-USDT:USDT"
INTERVAL = "5m"
START_DATE = date(2025, 9, 1)
END_DATE = date(2026, 2, 28)
START_TS = pd.Timestamp(START_DATE)
END_TS = pd.Timestamp(END_DATE) + pd.Timedelta(days=1)

API_KEY = "..."  # Set via APERIODIC_API_KEY env var or .env file
if API_KEY == "...":
    API_KEY = os.getenv("APERIODIC_API_KEY", "...")
if API_KEY == "...":
    raise RuntimeError("Set APERIODIC_API_KEY in the environment or in .env.")


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



def to_bps(series: pd.Series) -> pd.Series:
    return series * 10_000


# %% [markdown]
# ## Fetch price, funding, open interest, and basis

# %%
price = clip_window(
    get_ohlcv(
        api_key=API_KEY,
        timestamp="true",
        interval=INTERVAL,
        exchange=EXCHANGE,
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        output="pandas",
        show_progress=False,
    )
)[["time", "close"]]

funding = clip_window(
    get_derivative_metrics(
        api_key=API_KEY,
        metric="funding",
        timestamp="true",
        interval=INTERVAL,
        exchange=EXCHANGE,
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        output="pandas",
        show_progress=False,
    )
)

oi = clip_window(
    get_derivative_metrics(
        api_key=API_KEY,
        metric="open_interest",
        timestamp="true",
        interval=INTERVAL,
        exchange=EXCHANGE,
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        output="pandas",
        show_progress=False,
    )
)

basis = clip_window(
    get_derivative_metrics(
        api_key=API_KEY,
        metric="basis",
        timestamp="true",
        interval=INTERVAL,
        exchange=EXCHANGE,
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        output="pandas",
        show_progress=False,
    )
)

# %%
derivs = (
    price.merge(funding, on="time", how="inner")
    .merge(oi, on="time", how="inner")
    .merge(basis, on="time", how="inner")
)

derivs["price_return_bps_1h"] = derivs["close"].pct_change(12) * 10_000

derivs["funding_rate_bps"] = derivs["funding_rate"] * 10_000
for col in ["funding_rate_bps", "basis_bps", "open_interest"]:
    derivs[f"{col}_z_1d"] = (derivs[col] - derivs[col].rolling(288).mean()) / derivs[col].rolling(288).std()

derivs["oi_change_1h_pct"] = derivs["open_interest"].pct_change(12) * 100

derivs["next_1h_return_bps"] = derivs["close"].pct_change(12).shift(-12) * 10_000

derivs["stress_regime"] = (
    (derivs["funding_rate_bps_z_1d"] > 1.5)
    & (derivs["basis_bps_z_1d"] > 1.5)
    & (derivs["open_interest_z_1d"] > 1.0)
)

derivs["crowding_score"] = derivs[["funding_rate_bps_z_1d", "basis_bps_z_1d", "open_interest_z_1d"]].mean(axis=1)

summary = pd.Series(
    {
        "Rows": len(derivs),
        "Start": derivs["time"].min(),
        "End": derivs["time"].max(),
        "Mean funding (bps)": derivs["funding_rate_bps"].mean(),
        "Mean basis (bps)": derivs["basis_bps"].mean(),
        "Median open interest": derivs["open_interest"].median(),
        "Stress regime share": derivs["stress_regime"].mean(),
    }
)
summary

# %% [markdown]
# ## Charts 1-2 — BTC price and funding rate

# %%
fig, (ax_price, ax_funding) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
ax_price.plot(derivs["time"], derivs["close"], color="#2563eb", linewidth=1.2)
ax_price.set_title("BTC perpetual close price")
ax_price.set_ylabel("Price (USDT)")
format_time_axis(ax_price)

ax_funding.plot(derivs["time"], derivs["funding_rate_bps"], color="#dc2626", linewidth=1)
ax_funding.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax_funding.set_title("Funding rate (bps)")
ax_funding.set_ylabel("bps")
format_time_axis(ax_funding)
plt.tight_layout()

# %% [markdown]
# ## Chart 3 — Funding distribution

# %%
fig, ax = plt.subplots()
ax.hist(derivs["funding_rate_bps"].dropna(), bins=80, color="#f97316", alpha=0.8, edgecolor="white")
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_title("Distribution of 5-minute funding readings")
ax.set_xlabel("Funding rate (bps)")
plt.tight_layout()

# %% [markdown]
# ## Chart 4 — Price vs open interest
#
# A rising market with rising open interest often suggests leverage is building alongside the move.

# %%
fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(derivs["time"], derivs["close"], color="#2563eb", linewidth=1.1, label="Close")
ax1.set_ylabel("Price (USDT)", color="#2563eb")
ax2 = ax1.twinx()
ax2.plot(derivs["time"], derivs["open_interest"], color="#059669", linewidth=1, alpha=0.8, label="Open interest")
ax2.set_ylabel("Open interest", color="#059669")
ax1.set_title("Price and open interest")
format_time_axis(ax)
plt.tight_layout()

# %% [markdown]
# ## Chart 5 — Basis in basis points

# %%
fig, ax = plt.subplots()
ax.plot(derivs["time"], derivs["basis_bps"], color="#7c3aed", linewidth=1)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_title("Basis (bps)")
ax.set_ylabel("bps")
format_time_axis(ax)
plt.tight_layout()

# %% [markdown]
# ## Chart 6 — Crowding score and highlighted stress regimes

# %%
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(derivs["time"], derivs["crowding_score"], color="#111827", linewidth=1.2)
ax.fill_between(
    derivs["time"],
    derivs["crowding_score"],
    where=derivs["stress_regime"],
    color="#ef4444",
    alpha=0.2,
    label="Stress regime",
)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_title("Composite derivatives crowding score")
ax.set_ylabel("Average z-score")
ax.legend(frameon=True)
format_time_axis(ax)
plt.tight_layout()

# %% [markdown]
# ## Chart 7 — Funding by weekday and hour

# %%
seasonality = derivs.assign(
    weekday=derivs["time"].dt.day_name().str[:3],
    hour=derivs["time"].dt.hour,
).pivot_table(index="weekday", columns="hour", values="funding_rate_bps", aggfunc="mean")
seasonality = seasonality.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(seasonality, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Average funding rate by weekday and hour (bps)")
ax.set_xlabel("Hour of day")
ax.set_ylabel("")
plt.tight_layout()

# %% [markdown]
# ## Takeaways
#
# - Funding, basis, and open interest are complementary measures of **leverage, crowding, and carry**.
# - Persistent positive funding plus rich basis often signals an increasingly expensive long trade.
# - Rising open interest during a price move can indicate fresh positioning rather than mere price drift.
# - The most useful regime read usually comes from **the combination** of these metrics, not any one series in isolation.
#
# ## Further reading
# - Aperiodic derivatives market data: https://aperiodic.io/metrics/derivatives_market_data
# - Funding rates: https://www.binance.com/en/academy/articles/what-are-funding-rates-in-crypto-markets
# - Basis overview: https://www.cmegroup.com/education/courses/introduction-to-basis.html
