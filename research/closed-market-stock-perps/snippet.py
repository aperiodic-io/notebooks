from datetime import date

import polars as pl
from aperiodic import get_metrics

# NVDA's perp on Hyperliquid (trade.xyz), 5-minute top of book
q = get_metrics(
    metric="l1_price",
    timestamp="true",
    interval="5m",
    exchange="hyperliquid-perps",
    symbol="perpetual-XYZ:NVDA-USDC:USDC",
    start_date=date(2026, 9, 1),
    end_date=date(2026, 10, 31),
)

et = pl.col("time").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York")
mid = ((pl.col("bid_price") + pl.col("ask_price")) / 2).log()
print(
    q.with_columns(
        imbalance=(pl.col("bid_amount") - pl.col("ask_amount"))
        / (pl.col("bid_amount") + pl.col("ask_amount")),
        next_15m=(mid.shift(-3) - mid) * 1e4,
        weekend=(et.dt.weekday() >= 6),
    )
    .group_by("weekend")
    .agg(ic=pl.corr(pl.col("imbalance").rank(), pl.col("next_15m").rank()))
)
