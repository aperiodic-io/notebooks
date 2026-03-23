import marimo

app = marimo.App(css_file="theme.css")


@app.cell
def _():
    import marimo as mo

    try:
        import micropip

        await micropip.install(["aperiodic[pandas]", "pyarrow"])
    except ImportError:
        pass  # Running outside WASM — install deps via pip

    API_KEY = "..."  # Paste your Aperiodic API key here
    BASE_URL = "https://aperiodic.io"

    mo.md(
        """
        # Order Flow & Microstructure

        Analyze taker flow, buy/sell imbalances, and volume profiles using
        Aperiodic's microstructure metrics.
        """
    )
    return API_KEY, BASE_URL


@app.cell
def _(API_KEY, BASE_URL):
    import pandas as pd
    from datetime import date
    from aperiodic import get_metrics_async

    API_BASE = f"{BASE_URL}/api/v1"

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
    return flow, pd


@app.cell
def _(flow, pd):
    import matplotlib.pyplot as _plt
    import numpy as np

    _fig, (_ax1, _ax2) = _plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    _times = pd.to_datetime(flow["time"])

    _ax1.bar(
        _times[flow["net_delta"] >= 0],
        flow["net_delta"][flow["net_delta"] >= 0],
        width=0.03,
        color="#10b981",
        alpha=0.7,
        label="Net Buy",
    )
    _ax1.bar(
        _times[flow["net_delta"] < 0],
        flow["net_delta"][flow["net_delta"] < 0],
        width=0.03,
        color="#ef4444",
        alpha=0.7,
        label="Net Sell",
    )
    _ax1.set_title("BTC-USDT Taker Net Delta (Jan 15-17, 2024)")
    _ax1.set_ylabel("Net Delta (USDT)")
    _ax1.legend(fontsize=8)
    _ax1.grid(True, alpha=0.3)

    _cumulative = np.cumsum(flow["net_delta"])
    _ax2.plot(_times, _cumulative, color="#6366f1", linewidth=1.5)
    _ax2.fill_between(_times, _cumulative, alpha=0.1, color="#6366f1")
    _ax2.set_title("Cumulative Net Delta")
    _ax2.set_ylabel("Cumulative Delta (USDT)")
    _ax2.set_xlabel("Time")
    _ax2.grid(True, alpha=0.3)

    _fig.tight_layout()
    _plt.show()
    return ()


@app.cell
def _(flow, pd):
    import matplotlib.pyplot as _plt

    _fig, _ax = _plt.subplots(figsize=(10, 4))

    _times = pd.to_datetime(flow["time"])
    _buy = flow["taker_buy_volume"]
    _sell = flow["taker_sell_volume"]

    _ax.bar(_times, _buy, width=0.03, color="#10b981", alpha=0.6, label="Taker Buy")
    _ax.bar(_times, -_sell, width=0.03, color="#ef4444", alpha=0.6, label="Taker Sell")
    _ax.set_title("Buy vs Sell Volume")
    _ax.set_ylabel("Volume (USDT)")
    _ax.set_xlabel("Time")
    _ax.legend(fontsize=8)
    _ax.grid(True, alpha=0.3)
    _ax.axhline(y=0, color="#94a3b8", linewidth=0.5)

    _fig.tight_layout()
    _plt.show()
    return ()


@app.cell
def _(flow):
    import numpy as np
    import marimo as mo

    _stats = {
        "Mean Buy Volume": f"{np.mean(flow['taker_buy_volume']):,.2f}",
        "Mean Sell Volume": f"{np.mean(flow['taker_sell_volume']):,.2f}",
        "Mean Net Delta": f"{np.mean(flow['net_delta']):,.2f}",
        "Std Net Delta": f"{np.std(flow['net_delta']):,.2f}",
        "Max Buy Volume": f"{np.max(flow['taker_buy_volume']):,.2f}",
        "Max Sell Volume": f"{np.max(flow['taker_sell_volume']):,.2f}",
        "Buy/Sell Ratio": f"{np.sum(flow['taker_buy_volume']) / np.sum(flow['taker_sell_volume']):.4f}",
    }

    mo.md(
        "### Flow Statistics\n\n"
        + "\n".join(f"- **{k}**: {v}" for k, v in _stats.items())
    )
    return ()


if __name__ == "__main__":
    app.run()
