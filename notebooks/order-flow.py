# %% [markdown]
# # Order Flow & Microstructure
#
# Analyze taker flow, buy/sell imbalances, and volume profiles using
# Aperiodic's microstructure metrics.

# %%
try:
    import marimo as mo
    _params = mo.query_params
    API_KEY = _params.get("apiKey", "...")  # Auto-filled in the playground
    BASE_URL = _params.get("siteUrl", "https://aperiodic.io")
except Exception:
    API_KEY = "..."  # Paste your Aperiodic API key here
    BASE_URL = "https://aperiodic.io"

API_BASE = f"{BASE_URL}/api/v1"

# %%
from datetime import date

import pandas as pd
from aperiodic import get_metrics_async

flow = await get_metrics_async(
    api_key=API_KEY,
    metric="flow",
    timestamp="true",
    interval="1h",
    exchange="binance-futures",
    symbol="perpetual-BTC-USDT:USDT",
    start_date=date(2024, 1, 15),
    end_date=date(2024, 1, 17),
    base_url=API_BASE,
)
flow["net_delta"] = flow["taker_buy_volume"] - flow["taker_sell_volume"]
flow.head(10)

# %%
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

times = pd.to_datetime(flow["time"])

ax1.bar(
    times[flow["net_delta"] >= 0],
    flow["net_delta"][flow["net_delta"] >= 0],
    width=0.03,
    color="#10b981",
    alpha=0.7,
    label="Net Buy",
)
ax1.bar(
    times[flow["net_delta"] < 0],
    flow["net_delta"][flow["net_delta"] < 0],
    width=0.03,
    color="#ef4444",
    alpha=0.7,
    label="Net Sell",
)
ax1.set_title("BTC-USDT Taker Net Delta (Jan 15-17, 2024)")
ax1.set_ylabel("Net Delta (USDT)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

cumulative = np.cumsum(flow["net_delta"])
ax2.plot(times, cumulative, color="#6366f1", linewidth=1.5)
ax2.fill_between(times, cumulative, alpha=0.1, color="#6366f1")
ax2.set_title("Cumulative Net Delta")
ax2.set_ylabel("Cumulative Delta (USDT)")
ax2.set_xlabel("Time")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 4))

buy = flow["taker_buy_volume"]
sell = flow["taker_sell_volume"]

ax.bar(times, buy, width=0.03, color="#10b981", alpha=0.6, label="Taker Buy")
ax.bar(times, -sell, width=0.03, color="#ef4444", alpha=0.6, label="Taker Sell")
ax.set_title("Buy vs Sell Volume")
ax.set_ylabel("Volume (USDT)")
ax.set_xlabel("Time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="#94a3b8", linewidth=0.5)

fig.tight_layout()
plt.show()

# %% [markdown]
# ### Flow Statistics

# %%
stats = {
    "Mean Buy Volume": f"{np.mean(flow['taker_buy_volume']):,.2f}",
    "Mean Sell Volume": f"{np.mean(flow['taker_sell_volume']):,.2f}",
    "Mean Net Delta": f"{np.mean(flow['net_delta']):,.2f}",
    "Std Net Delta": f"{np.std(flow['net_delta']):,.2f}",
    "Max Buy Volume": f"{np.max(flow['taker_buy_volume']):,.2f}",
    "Max Sell Volume": f"{np.max(flow['taker_sell_volume']):,.2f}",
    "Buy/Sell Ratio": f"{np.sum(flow['taker_buy_volume']) / np.sum(flow['taker_sell_volume']):.4f}",
}

for k, v in stats.items():
    print(f"{k}: {v}")
