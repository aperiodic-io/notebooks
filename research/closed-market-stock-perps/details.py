"""The method details post.md reports beyond study.py's headline numbers, written to details.json:

- the panel: symbols, first bar per symbol, bars per session;
- each monthly refit: training rows, test rows, the entry threshold in bp, the out-of-sample IC,
  trades and net bp per trade (closed market and the regular-hours control);
- the gain importance of each feature in every closed-market refit;
- the closed-market trades: win rate, longs and shorts, holding time, spread and |forecast| at entry;
- the average next-15-minute return over all closed-market test bars, the drift a long-only rule would earn.

    python details.py --root ~/closed_market
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import data
import numpy as np
import polars as pl
from model import (
    CLOSED,
    FEATURES,
    REGULAR,
    Model,
    backtest,
    features,
    month_start,
    rank_ic,
)

TEST_MONTHS = range(4, 12)
END = dt.date(2026, 11, 11)


class Logged(Model):
    """The study's model, also keeping each refit's training size and gain importance."""

    log: list

    def fit_predict(self, train, test):
        import lightgbm as lgb

        forecast, threshold = super().fit_predict(train, test)
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
        booster = lgb.train(
            params,
            lgb.Dataset(train.select(FEATURES).to_numpy(), train["target"].to_numpy()),
            self.trees,
        )
        gain = booster.feature_importance("gain")
        self.log.append(
            {
                "train_rows": train.height,
                "test_rows": test.height,
                "threshold_bp": threshold,
                "gain_share": dict(
                    zip(FEATURES, (gain / gain.sum()).tolist(), strict=True)
                ),
            }
        )
        return forecast, threshold


def refits(
    df: pl.DataFrame, sessions: tuple[str, ...]
) -> tuple[pl.DataFrame, list[dict]]:
    rows = df.filter(pl.col("session").is_in(sessions), pl.col("bid").is_not_null())
    model, out, log = Logged(), [], []
    object.__setattr__(model, "log", log)
    for m in TEST_MONTHS:
        start, end = month_start(2026, m), month_start(2026, m + 1)
        train = rows.filter(
            pl.col("time") < start - dt.timedelta(days=2),
            pl.col("target").is_not_null(),
        )
        test = rows.filter(pl.col("time") >= start, pl.col("time") < end)
        forecast, threshold = model.fit_predict(train, test)
        out.append(
            test.with_columns(forecast=pl.Series(forecast), threshold=pl.lit(threshold))
        )
        log[-1]["month"] = m
    return pl.concat(out), log


def monthly(
    forecasts: pl.DataFrame, trades: pl.DataFrame, log: list[dict]
) -> list[dict]:
    rows = []
    for entry in log:
        m = entry["month"]
        t = trades.filter(pl.col("time").dt.month() == m)
        rows.append(
            {
                **{k: v for k, v in entry.items() if k != "gain_share"},
                "ic": rank_ic(forecasts.filter(pl.col("time").dt.month() == m)),
                "trades": t.height,
                "net_bp": float(t["net_bp"].mean()) if t.height else None,
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root.expanduser()

    panel = pl.concat(
        [data.bars(root, t) for t in data.UNIVERSE], how="diagonal_relaxed"
    )
    df = features(panel, data.INDEX)
    live = df.filter(pl.col("bid").is_not_null(), pl.col("time").dt.date() < END)

    closed, closed_log = refits(df, CLOSED)
    regular, regular_log = refits(df, REGULAR)
    tc, tr = backtest(closed), backtest(regular)

    entry = tc.join(
        closed.select("sym", "time", "forecast", "spread", "threshold", *FEATURES),
        on=["sym", "time"],
        how="left",
    )
    long, short = tc.filter(pl.col("side") > 0), tc.filter(pl.col("side") < 0)
    minutes = tc["bars"].to_numpy() * 5

    gain = pl.DataFrame([e["gain_share"] for e in closed_log])
    out = {
        "symbols": sorted(data.UNIVERSE),
        "first_bar": {
            t: str(d)
            for t, d in live.group_by("sym")
            .agg(pl.col("time").min().dt.date())
            .sort("sym")
            .iter_rows()
        },
        "bars_by_session": dict(
            live["session"].value_counts().sort("session").iter_rows()
        ),
        "bars_total": live.height,
        "closed_test_bars_with_target": closed.filter(
            pl.col("target").is_not_null()
        ).height,
        "regular_test_bars_with_target": regular.filter(
            pl.col("target").is_not_null()
        ).height,
        "refits_closed": monthly(closed, tc, closed_log),
        "refits_regular": monthly(regular, tr, regular_log),
        "gain_share_closed_mean": {f: float(gain[f].mean()) for f in FEATURES},
        "gain_share_closed_range": {
            f: [float(gain[f].min()), float(gain[f].max())] for f in FEATURES
        },
        "closed_test_mean_target_bp": float(closed["target"].drop_nulls().mean()),
        "closed_test_share_target_positive": float(
            (closed["target"].drop_nulls() > 0).mean()
        ),
        "trades_closed": {
            "n": tc.height,
            "win_rate": float((tc["net_bp"] > 0).mean()),
            "long": {"n": long.height, "net_bp": float(long["net_bp"].mean())},
            "short": {"n": short.height, "net_bp": float(short["net_bp"].mean())},
            "hold_minutes_median": float(np.median(minutes)),
            "hold_minutes_mean": float(minutes.mean()),
            "share_held_past_15m": float((minutes > 15).mean()),
            "share_hit_6h_cap": float((minutes >= 360).mean()),
            "spread_bp_at_entry_median": float(entry["spread"].median()),
            "abs_forecast_bp_at_entry_median": float(entry["forecast"].abs().median()),
            "symbols_traded": tc["sym"].n_unique(),
            "days_with_trades": tc["time"].dt.date().n_unique(),
            "features_at_entry_mean": {
                side: {
                    f: float(entry.filter(pl.col("side") == v)[f].mean())
                    for f in FEATURES
                }
                for side, v in (("long", 1.0), ("short", -1.0))
            },
        },
    }
    (Path(__file__).parent / "details.json").write_text(
        json.dumps(out, indent=1, default=str)
    )
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
