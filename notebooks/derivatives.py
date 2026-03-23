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
        # Derivatives Analytics

        Explore funding rates, open interest, and basis data for perpetual futures.
        """
    )
    return API_KEY, BASE_URL


@app.cell
def _(API_KEY, BASE_URL):
    import pandas as pd
    from datetime import date
    from aperiodic import get_derivative_metrics_async

    API_BASE = f"{BASE_URL}/api/v1"

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
    return funding, pd


@app.cell
def _(funding):
    import matplotlib.pyplot as _plt

    _fig, _ax = _plt.subplots(figsize=(10, 4))
    _ax.hist(funding["funding_rate"], bins=40, color="#6366f1", alpha=0.7, edgecolor="white")
    _ax.axvline(x=0, color="#ef4444", linestyle="--", alpha=0.5)
    _ax.set_title("BTC-USDT Funding Rate Distribution (Jan 2024)")
    _ax.set_xlabel("Funding Rate")
    _ax.set_ylabel("Frequency")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _plt.show()
    return ()


@app.cell
def _(API_KEY, BASE_URL):
    import pandas as pd
    from datetime import date
    from aperiodic import get_derivative_metrics_async

    API_BASE = f"{BASE_URL}/api/v1"

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

    import matplotlib.pyplot as _plt

    _fig, _ax = _plt.subplots(figsize=(10, 4))
    _ax.plot(pd.to_datetime(oi["time"]), oi["open_interest"], color="#10b981", linewidth=1.2)
    _ax.fill_between(pd.to_datetime(oi["time"]), oi["open_interest"], alpha=0.08, color="#10b981")
    _ax.set_title("BTC-USDT Open Interest (Jan 2024)")
    _ax.set_xlabel("Date")
    _ax.set_ylabel("Open Interest")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _plt.show()
    return ()


@app.cell
def _(API_KEY, BASE_URL):
    import pandas as pd
    from datetime import date
    from aperiodic import get_derivative_metrics_async

    API_BASE = f"{BASE_URL}/api/v1"

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

    import matplotlib.pyplot as _plt

    _fig, (_ax1, _ax2) = _plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    _ax1.plot(pd.to_datetime(basis["time"]), basis["basis"], color="#f59e0b", linewidth=1)
    _ax1.set_title("BTC-USDT Basis (Jan 2024)")
    _ax1.set_ylabel("Basis")
    _ax1.axhline(y=0, color="#94a3b8", linestyle="--", alpha=0.5)
    _ax1.grid(True, alpha=0.3)

    _ax2.plot(pd.to_datetime(basis["time"]), basis["basis_rate"], color="#8b5cf6", linewidth=1)
    _ax2.set_ylabel("Basis Rate")
    _ax2.set_xlabel("Date")
    _ax2.axhline(y=0, color="#94a3b8", linestyle="--", alpha=0.5)
    _ax2.grid(True, alpha=0.3)

    _fig.tight_layout()
    _plt.show()
    return ()


if __name__ == "__main__":
    app.run()
