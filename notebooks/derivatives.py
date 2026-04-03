# %% [markdown]
# # Derivatives Analytics
#
# Explore funding rates, open interest, and basis data for perpetual futures contracts.

# %%
API_KEY = "..."  # Paste your Aperiodic API key here
BASE_URL = "https://aperiodic.io"
API_BASE = f"{BASE_URL}/api/v1"

# %%
from datetime import date

import pandas as pd
from aperiodic import get_derivative_metrics_async

funding = await get_derivative_metrics_async(
    api_key=API_KEY,
    metric="funding",
    timestamp="true",
    interval="1h",
    exchange="binance-futures",
    symbol="perpetual-BTC-USDT:USDT",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 2, 1),
    base_url=API_BASE,
    output="pandas",
)
funding.head(10)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(funding["funding_rate"], bins=40, color="#6366f1", alpha=0.7, edgecolor="white")
ax.axvline(x=0, color="#ef4444", linestyle="--", alpha=0.5)
ax.set_title("BTC-USDT Funding Rate Distribution (Jan 2024)")
ax.set_xlabel("Funding Rate")
ax.set_ylabel("Frequency")
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %%
oi = await get_derivative_metrics_async(
    api_key=API_KEY,
    metric="open_interest",
    timestamp="true",
    interval="1h",
    exchange="binance-futures",
    symbol="perpetual-BTC-USDT:USDT",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 2, 1),
    base_url=API_BASE,
    output="pandas",
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pd.to_datetime(oi["time"]), oi["open_interest"], color="#10b981", linewidth=1.2)
ax.fill_between(pd.to_datetime(oi["time"]), oi["open_interest"], alpha=0.08, color="#10b981")
ax.set_title("BTC-USDT Open Interest (Jan 2024)")
ax.set_xlabel("Date")
ax.set_ylabel("Open Interest")
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %%
basis = await get_derivative_metrics_async(
    api_key=API_KEY,
    metric="basis",
    timestamp="true",
    interval="1h",
    exchange="binance-futures",
    symbol="perpetual-BTC-USDT:USDT",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 2, 1),
    base_url=API_BASE,
    output="pandas",
)

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(pd.to_datetime(basis["time"]), basis["basis"], color="#f59e0b", linewidth=1)
ax.set_title("BTC-USDT Basis (Jan 2024)")
ax.set_ylabel("Basis")
ax.set_xlabel("Date")
ax.axhline(y=0, color="#94a3b8", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
