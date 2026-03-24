# %% [markdown]
# # Derivatives Analytics
#
# Explore funding rates, open interest, and basis data for perpetual futures.

# %%
try:
    import marimo as mo
    _params = mo.query_params()
    BASE_URL = _params.get("siteUrl", "https://aperiodic.io")
    # Read API key from cookie (set by parent page on .aperiodic.io domain)
    from js import document as _doc
    API_KEY = dict(c.strip().split("=", 1) for c in _doc.cookie.split(";") if "=" in c).get("apiKey", "...")
except Exception:
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
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.plot(pd.to_datetime(basis["time"]), basis["basis"], color="#f59e0b", linewidth=1)
ax1.set_title("BTC-USDT Basis (Jan 2024)")
ax1.set_ylabel("Basis")
ax1.axhline(y=0, color="#94a3b8", linestyle="--", alpha=0.5)
ax1.grid(True, alpha=0.3)

ax2.plot(pd.to_datetime(basis["time"]), basis["basis_rate"], color="#8b5cf6", linewidth=1)
ax2.set_ylabel("Basis Rate")
ax2.set_xlabel("Date")
ax2.axhline(y=0, color="#94a3b8", linestyle="--", alpha=0.5)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
