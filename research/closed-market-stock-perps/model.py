"""The simplest model that shows the effect, and its backtest.

Four features, one per mechanism, each known at the close of a 5-minute bar:

- `premium`: the perp's mid over its oracle price, in bp, less its 1-day mean (reverts);
- `book_imbalance`: (bid size - ask size) / (bid size + ask size) at the best quotes;
- `return_15m`: the mid's last 15-minute return, in bp (short-term reversal);
- `index_return_15m`: the XYZ100 index perp's last 15-minute return, in bp (the index leads).

The target is the mid's next 15-minute return, in bp. The model is a small gradient-boosted tree
ensemble (7 leaves, 150 trees, Huber loss) retrained each month on the months before, with a
2-day embargo, on the bars of one session group only: the hours the US cash market is closed, or
its regular hours as the control. The backtest crosses the spread when the forecast is among the
most extreme `TOP` of the training month's forecasts: a threshold set by rate rather than in bp,
because the forecast's scale drifts from one monthly refit to the next.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

FEATURES = ("premium", "book_imbalance", "return_15m", "index_return_15m")
ET = ZoneInfo("America/New_York")
# NYSE full-day closures in 2026
HOLIDAYS = tuple(
    dt.date(2026, m, d)
    for m, d in (
        (1, 1),
        (1, 19),
        (2, 16),
        (4, 3),
        (5, 25),
        (6, 19),
        (7, 3),
        (9, 7),
        (11, 26),
    )
)
CLOSED = ("overnight", "weekend", "holiday")
REGULAR = ("regular",)
BARS_15M = 3
BARS_1D = 288
# the fraction of bars traded: the threshold is this quantile of |in-sample forecast|
TOP = 0.001


def session(time: pl.Expr) -> pl.Expr:
    """The US cash-market session of a bar (bar open time in UTC), in New York time."""
    et = time.dt.replace_time_zone("UTC").dt.convert_time_zone(str(ET))
    minute = et.dt.hour().cast(pl.Int32) * 60 + et.dt.minute().cast(pl.Int32)
    weekday = et.dt.weekday()  # Monday = 1
    weekend = (
        (weekday == 6)
        | ((weekday == 5) & (minute >= 20 * 60))
        | ((weekday == 7) & (minute < 20 * 60))
    )
    return (
        pl.when(weekend)
        .then(pl.lit("weekend"))
        .when(et.dt.date().is_in(HOLIDAYS))
        .then(pl.lit("holiday"))
        .when(minute.is_between(9 * 60 + 30, 16 * 60, closed="left"))
        .then(pl.lit("regular"))
        .when(minute.is_between(4 * 60, 9 * 60 + 30, closed="left"))
        .then(pl.lit("premarket"))
        .when(minute.is_between(16 * 60, 20 * 60, closed="left"))
        .then(pl.lit("afterhours"))
        .otherwise(pl.lit("overnight"))
    )


def features(panel: pl.DataFrame, index: str) -> pl.DataFrame:
    """The four features and the target, per symbol; `panel` is every symbol's bars."""
    lm = ((pl.col("bid") + pl.col("ask")) / 2).log()
    df = panel.sort("sym", "time").with_columns(
        mid=(pl.col("bid") + pl.col("ask")) / 2,
        spread=(pl.col("ask") / pl.col("bid") - 1) * 1e4,
        premium_raw=(lm - pl.col("index").log()) * 1e4,
        book_imbalance=(pl.col("bsz") - pl.col("asz"))
        / (pl.col("bsz") + pl.col("asz")),
        return_15m=((lm - lm.shift(BARS_15M)) * 1e4).over("sym"),
        target=((lm.shift(-BARS_15M) - lm) * 1e4).over("sym").clip(-300, 300),
        session=session(pl.col("time")),
    )
    df = df.with_columns(
        premium=(
            pl.col("premium_raw")
            - pl.col("premium_raw").rolling_mean(BARS_1D, min_samples=50)
        ).over("sym")
    )
    idx = df.filter(pl.col("sym") == index).select(
        "time", index_return_15m=pl.col("return_15m")
    )
    return df.join(idx, on="time", how="left").drop("premium_raw")


def month_start(year: int, month: int) -> dt.datetime:
    """Midnight UTC on the 1st, naive like the bars' timestamps (which are UTC)."""
    year, month = year + (month - 1) // 12, (month - 1) % 12 + 1
    return dt.datetime(year, month, 1, tzinfo=dt.UTC).replace(tzinfo=None)


@dataclass(frozen=True)
class Model:
    leaves: int = 7
    trees: int = 150
    learning_rate: float = 0.05
    min_leaf: int = 2000

    def fit_predict(
        self, train: pl.DataFrame, test: pl.DataFrame
    ) -> tuple[np.ndarray, float]:
        """The test rows' forecasts (bp), and the entry threshold: the `TOP` quantile of the
        absolute forecast on the training rows."""
        import lightgbm as lgb

        params = {
            "objective": "huber",
            "alpha": 30.0,
            "num_leaves": self.leaves,
            "min_data_in_leaf": self.min_leaf,
            "learning_rate": self.learning_rate,
            "seed": 1,
            "deterministic": True,
            "verbose": -1,
        }
        x = train.select(FEATURES).to_numpy()
        booster = lgb.train(
            params, lgb.Dataset(x, train["target"].to_numpy()), self.trees
        )
        threshold = float(np.quantile(np.abs(np.asarray(booster.predict(x))), 1 - TOP))
        return np.asarray(
            booster.predict(test.select(FEATURES).to_numpy()), dtype=float
        ), threshold


def walk_forward(
    df: pl.DataFrame, sessions: tuple[str, ...], months: range, model: Model
) -> pl.DataFrame:
    """Out-of-sample forecasts for each test month, from a model fit on the earlier months, with
    that model's entry threshold."""
    rows = df.filter(pl.col("session").is_in(sessions), pl.col("bid").is_not_null())
    out = []
    for m in months:
        start, end = month_start(2026, m), month_start(2026, m + 1)
        train = rows.filter(
            pl.col("time") < start - dt.timedelta(days=2),
            pl.col("target").is_not_null(),
        )
        test = rows.filter(pl.col("time") >= start, pl.col("time") < end)
        if test.height:
            forecast, threshold = model.fit_predict(train, test)
            out.append(
                test.with_columns(
                    forecast=pl.Series(forecast), threshold=pl.lit(threshold)
                )
            )
    return pl.concat(out)


def backtest(
    forecasts: pl.DataFrame,
    fee: float = 1.5,
    max_spread: float = 5.0,
    max_hold_bars: int = 72,
    delay_bars: int = 0,
) -> pl.DataFrame:
    """Trades: enter across the spread when |forecast| > threshold (both bp, per row) and the
    spread is under `max_spread`; exit across the spread after 15 minutes, unless the forecast still points the
    same way at half the threshold, then hold another 15 minutes, up to `max_hold_bars`. `fee` is
    charged on each side. One position per symbol; a trade that would span a data gap is skipped."""
    trades = []
    for (sym,), g in forecasts.sort("time").group_by("sym", maintain_order=True):
        f = g["forecast"].to_numpy()
        thr = g["threshold"].to_numpy()
        bid = g["bid"].to_numpy()
        ask = g["ask"].to_numpy()
        spread = g["spread"].to_numpy()
        t = g["time"].dt.epoch("us").to_numpy()
        sess = g["session"].to_list()
        n = len(g)
        i = 0
        while i < n - BARS_15M - delay_bars - 1:
            if abs(f[i]) > thr[i] and spread[i] < max_spread:
                side = np.sign(f[i])
                e = i + delay_bars
                x = e + BARS_15M
                while (
                    x < n - BARS_15M
                    and side * f[x] > thr[x] / 2
                    and x - e < max_hold_bars
                ):
                    x += BARS_15M
                if t[x] - t[i] > (x - i) * 300e6 * 1.01:  # bars missing in between
                    i += 1
                    continue
                gross = (bid[x] / ask[e] - 1) if side > 0 else (bid[e] / ask[x] - 1)
                trades.append((sym, t[i], sess[i], side, gross * 1e4 - 2 * fee, x - e))
                i = x + 1
            else:
                i += 1
    schema = {
        "sym": pl.Utf8,
        "time": pl.Int64,
        "session": pl.Utf8,
        "side": pl.Float64,
        "net_bp": pl.Float64,
        "bars": pl.Int64,
    }
    return pl.DataFrame(trades, schema=schema, orient="row").with_columns(
        pl.from_epoch("time", time_unit="us")
    )


def daily(trades: pl.DataFrame, start: dt.date, end: dt.date) -> np.ndarray:
    """Net bp per calendar day (one unit of notional per trade), zero on days without trades."""
    days = pl.DataFrame({"day": pl.date_range(start, end, "1d", eager=True)})
    pnl = trades.group_by(day=pl.col("time").dt.date()).agg(pl.col("net_bp").sum())
    return days.join(pnl, on="day", how="left").fill_null(0)["net_bp"].to_numpy()


def summary(trades: pl.DataFrame, start: dt.date, end: dt.date) -> dict[str, float]:
    """Trades, mean net bp per trade with its day-clustered t-statistic, and the annualised
    Sharpe ratio of the daily series."""
    x = daily(trades, start, end)
    if trades.height < 2:
        return {
            "trades": float(trades.height),
            "net_bp": float("nan"),
            "t": float("nan"),
            "sharpe": 0.0,
        }
    g = trades.group_by(pl.col("time").dt.date()).agg(
        s=pl.col("net_bp").sum(), n=pl.len()
    )
    s, n = g["s"].to_numpy(), g["n"].to_numpy()
    mu = s.sum() / n.sum()
    se = np.sqrt(((s - mu * n) ** 2).sum()) / n.sum()
    return {
        "trades": float(trades.height),
        "net_bp": float(mu),
        "t": float(mu / se),
        "sharpe": float(x.mean() / x.std() * np.sqrt(365)),
    }


def rank_ic(forecasts: pl.DataFrame) -> float:
    """Rank correlation between the forecast and the realised target."""
    d = forecasts.filter(pl.col("target").is_not_null())
    return float(
        np.corrcoef(d["forecast"].rank().to_numpy(), d["target"].rank().to_numpy())[
            0, 1
        ]
    )
