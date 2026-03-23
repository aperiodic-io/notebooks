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
        f"""
        # Getting Started with Aperiodic

        Welcome! This notebook walks you through the basics of fetching market data
        with the [Aperiodic Python SDK](https://docs.aperiodic.io).

        **Paste your API key** in the first cell above, then run all cells.
        """
    )
    return API_KEY, BASE_URL


@app.cell
def _(API_KEY, BASE_URL):
    import pandas as pd
    from datetime import date
    from aperiodic import get_ohlcv_async

    API_BASE = f"{BASE_URL}/api/v1"

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
    return df, pd


@app.cell
def _(df, pd):
    import matplotlib.pyplot as _plt

    _fig, _ax = _plt.subplots(figsize=(10, 4))
    _ax.plot(pd.to_datetime(df["time"]), df["close"], color="#6366f1", linewidth=1.5)
    _ax.fill_between(pd.to_datetime(df["time"]), df["close"], alpha=0.1, color="#6366f1")
    _ax.set_title("BTC-USDT Perpetual — Daily Close")
    _ax.set_xlabel("Date")
    _ax.set_ylabel("Price (USDT)")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _plt.show()
    return ()


@app.cell
def _(API_KEY, BASE_URL):
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
        end_date=date(2024, 1, 16),
        base_url=API_BASE,
    )
    flow["net_delta"] = flow["taker_buy_volume"] - flow["taker_sell_volume"]
    flow[["time", "taker_buy_volume", "taker_sell_volume", "net_delta"]].head(10)
    return ()


if __name__ == "__main__":
    app.run()
