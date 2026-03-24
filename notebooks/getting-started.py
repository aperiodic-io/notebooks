# %% [markdown]
# # Getting Started with Aperiodic
#
# Welcome! This notebook walks you through the basics of fetching market data
# with the [Aperiodic Python SDK](https://docs.aperiodic.io).
#
# **Paste your API key** in the first cell below, then run all cells.

# %%
try:
    import marimo as mo
    _params = mo.query_params()
    API_KEY = _params.get("apiKey", "...")  # Auto-filled in the playground
    BASE_URL = _params.get("siteUrl", "https://aperiodic.io")
except Exception:
    API_KEY = "..."  # Paste your Aperiodic API key here
    BASE_URL = "https://aperiodic.io"

API_BASE = f"{BASE_URL}/api/v1"

# %%
from datetime import date

import pandas as pd
from aperiodic import get_ohlcv_async

df = await get_ohlcv_async(
    api_key=API_KEY,
    timestamp="true",
    interval="1d",
    exchange="binance-futures",
    symbol="perpetual-BTC-USDT:USDT",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 3, 1),
    base_url=API_BASE,
)
df.head(10)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pd.to_datetime(df["time"]), df["close"], color="#6366f1", linewidth=1.5)
ax.fill_between(pd.to_datetime(df["time"]), df["close"], alpha=0.1, color="#6366f1")
ax.set_title("BTC-USDT Perpetual \u2014 Daily Close")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USDT)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Order Flow
#
# Fetch taker buy/sell volume and compute the net delta.

# %%
from aperiodic import get_metrics_async

flow = await get_metrics_async(
    api_key=API_KEY,
    metric="flow",
    timestamp="true",
    interval="1h",
    exchange="binance-futures",
    symbol="perpetual-BTC-USDT:USDT",
    start_date=date(2024, 1, 15),
    end_date=date(2024, 1, 16),
    base_url=API_BASE,
)
flow["net_delta"] = flow["taker_buy_volume"] - flow["taker_sell_volume"]
flow[["time", "taker_buy_volume", "taker_sell_volume", "net_delta"]].head(10)
