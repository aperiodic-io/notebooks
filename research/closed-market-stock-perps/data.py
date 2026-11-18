"""The study's input, pulled with the public Aperiodic Python client (`pip install "aperiodic[polars]"`,
key in APERIODIC_API_KEY): 5-minute bars of Hyperliquid's trade.xyz stock perps and their Binance twins.

- `l1_price`: the last best bid and ask of each bar and the sizes resting there;
- `derivative_price`: the last oracle (index) price and last trade price of each bar.

Each symbol's pull is cached under `<root>/<exchange>/<metric>/<ticker>.parquet`.
"""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from aperiodic import get_derivative_metrics, get_metrics

START, END = dt.date(2026, 1, 1), dt.date(2026, 11, 11)
HL, BN = "hyperliquid-perps", "binance-futures"
# US-listed stocks, ETFs and the two index perps
UNIVERSE = (
    "SNDK", "MU", "NVDA", "SOXL", "INTC", "GOOGL", "META", "CRCL", "TSLA", "AMD", "NBIS",
    "AAPL", "MSFT", "MRVL", "MSTR", "CBRS", "AMZN", "HOOD", "CRWV", "MRNA", "PLTR", "ORCL",
    "LITE", "COIN", "RKLB", "SPCX", "SP500", "XYZ100",
)  # fmt: skip
INDEX = "XYZ100"
# SK Hynix, Samsung, Kioxia: home markets in Seoul and Tokyo, open during the US night
ASIA = ("SKHX", "SMSN", "KIOXIA")
# the US names that also list on Binance
BINANCE = tuple(t for t in UNIVERSE if t not in ("SP500", "XYZ100"))


def symbol(exchange: str, ticker: str) -> str:
    return (
        f"perpetual-XYZ:{ticker}-USDC:USDC"
        if exchange == HL
        else f"perpetual-{ticker}-USDT:USDT"
    )


def _pull(exchange: str, metric: str, ticker: str) -> pl.DataFrame:
    kw = {
        "timestamp": "true",
        "interval": "5m",
        "exchange": exchange,
        "symbol": symbol(exchange, ticker),
        "start_date": START,
        "end_date": END,
        "show_progress": False,
    }
    if metric == "l1_price":
        df = get_metrics(metric="l1_price", **kw)
        return df.select(
            "time", bid="bid_price", ask="ask_price", bsz="bid_amount", asz="ask_amount"
        )
    df = get_derivative_metrics(metric="derivative_price", **kw)
    return df.select("time", index="index_price", last="last_price")


def download(root: Path, workers: int = 8) -> None:
    jobs = [
        (HL, m, t) for m in ("l1_price", "derivative_price") for t in UNIVERSE + ASIA
    ]
    jobs += [(BN, "l1_price", t) for t in BINANCE]
    jobs = [j for j in jobs if not (root / j[0] / j[1] / f"{j[2]}.parquet").exists()]

    def one(job: tuple[str, str, str]) -> None:
        exchange, metric, ticker = job
        try:
            df = (
                _pull(exchange, metric, ticker).unique("time", keep="last").sort("time")
            )
        except Exception as e:  # a symbol the venue does not list for the whole span
            print(f"{exchange} {metric} {ticker}: {e}")
            return
        out = root / exchange / metric / f"{ticker}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)
        print(f"{exchange} {metric} {ticker}: {df.height} bars")

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, jobs))


def bars(root: Path, ticker: str) -> pl.DataFrame:
    """One Hyperliquid symbol's bars on a regular 5-minute grid, quotes and oracle carried forward."""
    l1 = pl.read_parquet(root / HL / "l1_price" / f"{ticker}.parquet")
    ix = pl.read_parquet(root / HL / "derivative_price" / f"{ticker}.parquet")
    grid = pl.DataFrame(
        {
            "time": pl.datetime_range(
                l1["time"].min(), l1["time"].max(), "5m", eager=True, time_unit="us"
            )
        }
    )
    return (
        grid.join(l1, on="time", how="left")
        .join(
            ix.with_columns(pl.col("time").cast(pl.Datetime("us"))),
            on="time",
            how="left",
        )
        .with_columns(
            pl.col("bid", "ask", "index").forward_fill(), pl.lit(ticker).alias("sym")
        )
    )


def binance_mid(root: Path, ticker: str) -> pl.DataFrame | None:
    p = root / BN / "l1_price" / f"{ticker}.parquet"
    if not p.exists():
        return None
    return pl.read_parquet(p).select(
        pl.col("time").cast(pl.Datetime("us")),
        bn_mid=(pl.col("bid") + pl.col("ask")) / 2,
    )
