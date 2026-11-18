"""Every number in post.md, computed from data pulled with the public Aperiodic Python client.

    pip install "aperiodic[polars]" lightgbm matplotlib
    export APERIODIC_API_KEY=...
    python study.py --root ~/closed_market          # pulls once, then caches

Writes results.json (every figure the post quotes) and the two charts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import data
import numpy as np
import polars as pl
from model import (
    BARS_15M,
    CLOSED,
    REGULAR,
    Model,
    backtest,
    features,
    month_start,
    rank_ic,
    session,
    summary,
    walk_forward,
)

TEST_MONTHS = range(4, 12)
START, END = dt.date(2026, 4, 1), dt.date(2026, 11, 11)
HALF = dt.date(2026, 8, 1)
HERE = Path(__file__).resolve().parent


def halves(trades: pl.DataFrame) -> dict:
    s = summary(trades, START, END)
    a = summary(
        trades.filter(pl.col("time").dt.date() < HALF),
        START,
        HALF - dt.timedelta(days=1),
    )
    b = summary(trades.filter(pl.col("time").dt.date() >= HALF), HALF, END)
    return {**s, "sharpe_apr_jul": a["sharpe"], "sharpe_aug_nov": b["sharpe"]}


def ic_by_symbol(
    df: pl.DataFrame, x: str, y: str = "target"
) -> tuple[float, float, int]:
    """Mean over symbols of the per-symbol rank correlation, its t-statistic across symbols."""
    per = (
        (
            df.filter(
                pl.col(x).is_not_null()
                & pl.col(y).is_not_null()
                & pl.col(x).is_finite()
            )
            .group_by("sym")
            .agg(ic=pl.corr(pl.col(x).rank(), pl.col(y).rank()), n=pl.len())
            .filter(pl.col("n") > 200)
        )["ic"]
        .drop_nulls()
        .to_numpy()
    )
    return (
        float(per.mean()),
        float(per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))),
        len(per),
    )


def single_features(df: pl.DataFrame) -> dict:
    out = {}
    for sess in ("regular", "overnight", "weekend"):
        d = df.filter(pl.col("session") == sess, pl.col("time").dt.date() < END)
        out[sess] = {
            f: ic_by_symbol(d, f)
            for f in ("book_imbalance", "premium", "return_15m", "index_return_15m")
        }
    return out


def bounce(panel: pl.DataFrame) -> dict:
    """Weekend 15-minute reversal measured on last-trade prices and on mid prices."""
    lt = pl.col("last").forward_fill().over("sym").log()
    d = (
        panel.sort("sym", "time")
        .with_columns(
            last_ret=((lt - lt.shift(BARS_15M)) * 1e4).over("sym"),
            last_fwd=((lt.shift(-BARS_15M) - lt) * 1e4).over("sym"),
            session=session(pl.col("time")),
        )
        .filter(pl.col("session") == "weekend")
    )
    lm = ((pl.col("bid") + pl.col("ask")) / 2).log()
    d = d.with_columns(
        mid_ret=((lm - lm.shift(BARS_15M)) * 1e4).over("sym"),
        mid_fwd=((lm.shift(-BARS_15M) - lm) * 1e4).over("sym"),
    )
    return {
        "last": ic_by_symbol(d, "last_ret", "last_fwd"),
        "mid": ic_by_symbol(d, "mid_ret", "mid_fwd"),
    }


def binance_gap(root: Path, df: pl.DataFrame) -> dict:
    """IC of log(Binance mid / Hyperliquid mid) with Binance's next 15-minute mid return."""
    rows = []
    for t in data.BINANCE:
        bn = data.binance_mid(root, t)
        if bn is None:
            continue
        lb = pl.col("bn_mid").log()
        g = (
            df.filter(pl.col("sym") == t)
            .select("sym", "time", "mid", "session")
            .join(bn, on="time", how="inner")
            .sort("time")
            .with_columns(
                gap=(lb - pl.col("mid").log()) * 1e4,
                bn_fwd=(lb.shift(-BARS_15M) - lb) * 1e4,
            )
        )
        rows.append(g)
    g = pl.concat(rows)
    return {
        "symbols": g["sym"].n_unique(),
        "closed": ic_by_symbol(
            g.filter(pl.col("session").is_in(CLOSED)), "gap", "bn_fwd"
        ),
        "regular": ic_by_symbol(
            g.filter(pl.col("session") == "regular"), "gap", "bn_fwd"
        ),
    }


def depth(df: pl.DataFrame) -> float:
    """Median USD resting at the best quote on the side a closed-market trade takes."""
    d = df.filter(pl.col("session").is_in(CLOSED), pl.col("bid").is_not_null())
    usd = pl.concat([(d["ask"] * d["asz"]), (d["bid"] * d["bsz"])])
    return float(usd.median())


def ols(df: pl.DataFrame) -> dict:
    """The same four features, walk-forward, with a linear least-squares forecast."""
    from model import FEATURES

    class OLS(Model):
        def fit_predict(self, train, test):
            tr = train.drop_nulls([*FEATURES, "target"])
            x = np.c_[np.ones(tr.height), tr.select(FEATURES).to_numpy()]
            beta = np.linalg.lstsq(x, tr["target"].to_numpy(), rcond=None)[0]
            xt = np.c_[
                np.ones(test.height), test.select(FEATURES).fill_null(0).to_numpy()
            ]
            return xt @ beta, 6.0

    f = walk_forward(df, CLOSED, TEST_MONTHS, OLS())
    a = np.abs(f["forecast"].to_numpy())
    return {
        "ic": rank_ic(f),
        "abs_forecast_p999": float(np.quantile(a, 0.999)),
        "abs_forecast_max": float(a.max()),
        "at_6bp": halves(backtest(f)),
    }


def thresholds(df: pl.DataFrame) -> dict:
    """The entry rate, chosen on April-July only, and a fixed 8 bp threshold."""
    import model as m

    out = {}
    for top in (0.001, 0.002, 0.003, 0.005):
        m.TOP = top
        f = walk_forward(df, CLOSED, range(4, 8), Model())
        out[f"rate_{top}"] = summary(backtest(f), START, HALF - dt.timedelta(days=1))
    m.TOP = 0.001
    f = walk_forward(df, CLOSED, TEST_MONTHS, Model()).with_columns(
        threshold=pl.lit(8.0)
    )
    out["fixed_8bp"] = halves(backtest(f))
    return out


def capacity(df: pl.DataFrame) -> dict:
    out = {}
    for leaves in (3, 7, 15):
        for trees in (150, 300, 600):
            f = walk_forward(df, CLOSED, TEST_MONTHS, Model(leaves=leaves, trees=trees))
            out[f"{leaves}x{trees}"] = summary(backtest(f), START, END)
    return out


def asia(root: Path, df_us: pl.DataFrame) -> dict:
    """The US closed-market model's forecast on SK Hynix, Samsung and Kioxia, split by whether
    their home market (Seoul, Tokyo: 00:00-06:30 UTC on weekdays) and the US market are open."""
    panel = pl.concat(
        [data.bars(root, t) for t in data.ASIA] + [data.bars(root, data.INDEX)],
        how="diagonal_relaxed",
    )
    a = features(panel, data.INDEX).filter(pl.col("sym") != data.INDEX)
    both = pl.concat([df_us, a], how="diagonal_relaxed")
    rows = both.filter(pl.col("bid").is_not_null())
    out = []
    for mth in TEST_MONTHS:
        s, e = month_start(2026, mth), month_start(2026, mth + 1)
        train = rows.filter(
            pl.col("session").is_in(CLOSED),
            pl.col("sym").is_in(data.UNIVERSE),
            pl.col("time") < s - dt.timedelta(days=2),
            pl.col("target").is_not_null(),
        )
        test = rows.filter(
            pl.col("sym").is_in(data.ASIA), pl.col("time") >= s, pl.col("time") < e
        )
        if test.height:
            fc, _ = Model().fit_predict(train, test)
            out.append(test.with_columns(forecast=pl.Series(fc)))
    f = pl.concat(out)
    minute = f["time"].dt.hour().cast(pl.Int32) * 60 + f["time"].dt.minute().cast(
        pl.Int32
    )
    home_open = (f["time"].dt.weekday() <= 5) & (minute < 6 * 60 + 30)
    us_open = f["session"].is_in(["regular", "premarket", "afterhours"])
    f = f.with_columns(home_open=home_open, us_open=us_open)
    return {
        "both_closed_weekday": rank_ic(
            f.filter(
                ~pl.col("home_open")
                & ~pl.col("us_open")
                & (pl.col("session") != "weekend")
            )
        ),
        "weekend": rank_ic(f.filter(pl.col("session") == "weekend")),
        "home_open": rank_ic(f.filter(pl.col("home_open"))),
        "us_open": rank_ic(f.filter(pl.col("us_open"))),
    }


class Plain:
    """Plain matplotlib styling, used outside the Aperiodic marketing repo (where `brand` lives)."""

    INK, MUTED, SIGNAL, RULE = "#111111", "#6b6b6b", "#e8541f", "#d0d0d0"

    @staticmethod
    def apply():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "figure.figsize": (12.8, 7.2),
                "figure.dpi": 125,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )
        return plt

    @staticmethod
    def frame(fig, title, source, subtitle=None):
        fig.suptitle(
            title + (f"\n{subtitle}" if subtitle else ""),
            x=0.06,
            ha="left",
            fontsize=16,
        )
        fig.text(0.06, 0.02, source, fontsize=9, color=Plain.MUTED)
        fig.subplots_adjust(top=0.82, bottom=0.14)
        return fig

    @staticmethod
    def save(fig, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)


def style():
    """The Aperiodic brand look inside the marketing repo, plain matplotlib anywhere else."""
    try:
        sys.path.insert(0, str(HERE.parents[3]))
        from scripts.visuals import brand

        return brand
    except Exception:
        return Plain


def chart_dir() -> Path:
    assets = HERE.parent / "assets"
    return assets if (HERE.parent / "post.md").exists() else HERE / "charts"


def charts(feat: dict, closed_tr: pl.DataFrame, regular_tr: pl.DataFrame) -> None:
    brand = style()
    plt = brand.apply()
    src = "Source: Aperiodic l1_price and derivative_price, Hyperliquid trade.xyz perps, 5m bars, Jan-Nov 2026"

    names = {
        "book_imbalance": "Book imbalance",
        "premium": "Premium vs oracle",
        "return_15m": "Own 15m return",
        "index_return_15m": "Index 15m return",
    }
    fig, ax = plt.subplots()
    sess = ("regular", "overnight", "weekend")
    colors = (brand.MUTED, brand.INK, brand.SIGNAL)
    w = 0.26
    for k, (s, c) in enumerate(zip(sess, colors, strict=True)):
        vals = [abs(feat[s][f][0]) * 100 for f in names]
        ax.bar(
            np.arange(4) + (k - 1) * w,
            vals,
            w,
            color=c,
            label={
                "regular": "Regular hours",
                "overnight": "Overnight",
                "weekend": "Weekend",
            }[s],
        )
    ax.set_xticks(np.arange(4), list(names.values()))
    ax.set_ylabel("|rank IC| with next 15m mid return, %")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.12), ncol=3)
    ax.set_ylim(0, 7.5)
    brand.frame(
        fig,
        "Every signal is stronger when Nasdaq is shut",
        src,
        subtitle="28 stock perps on trade.xyz, mean over symbols, sign dropped",
    )
    brand.save(fig, chart_dir() / "ic-by-session.png")

    fig, ax = plt.subplots()
    for tr, c, label in (
        (closed_tr, brand.SIGNAL, "Closed market"),
        (regular_tr, brand.MUTED, "Regular hours (control)"),
    ):
        t = tr.sort("time")
        ax.plot(
            t["time"].to_list(), np.cumsum(t["net_bp"].to_numpy()), color=c, label=label
        )
    ax.axhline(0, color=brand.RULE, lw=1)
    ax.set_ylabel("Cumulative net bp, one unit per trade")
    ax.legend(frameon=False, loc="upper left")
    brand.frame(
        fig,
        "Same model: it only works while Nasdaq is closed",
        src,
        subtitle="Walk-forward, Apr 1 to Nov 11 2026, 1.5 bp per side",
    )
    brand.save(fig, chart_dir() / "pnl-closed-vs-regular.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--skip-download", action="store_true")
    root = p.parse_args().root.expanduser()
    if not p.parse_args().skip_download:
        data.download(root)

    panel = pl.concat(
        [data.bars(root, t) for t in data.UNIVERSE], how="diagonal_relaxed"
    )
    df = features(panel, data.INDEX)
    r: dict = {}
    sess = df.filter(pl.col("time").dt.date() < END)["session"].value_counts(
        normalize=True
    )
    r["session_share"] = dict(
        zip(sess["session"].to_list(), sess["proportion"].to_list(), strict=True)
    )
    r["closed_share"] = sum(v for k, v in r["session_share"].items() if k in CLOSED)
    r["symbols"] = df["sym"].n_unique()
    r["bounce_weekend"] = bounce(panel)
    r["single_feature_ic"] = single_features(df)
    r["binance_gap"] = binance_gap(root, df)
    r["median_best_usd_closed"] = depth(df)

    closed = walk_forward(df, CLOSED, TEST_MONTHS, Model())
    regular = walk_forward(df, REGULAR, TEST_MONTHS, Model())
    r["rank_ic"] = {"closed": rank_ic(closed), "regular": rank_ic(regular)}
    r["monthly_ic_closed"] = {
        m: rank_ic(closed.filter(pl.col("time").dt.month() == m)) for m in TEST_MONTHS
    }
    tr_closed, tr_regular = backtest(closed), backtest(regular)
    r["backtest"] = {
        "closed": halves(tr_closed),
        "regular": halves(tr_regular),
        "closed_3bp": halves(backtest(closed, fee=3.0)),
        "closed_late": halves(backtest(closed, delay_bars=1)),
        "closed_3bp_late": halves(backtest(closed, fee=3.0, delay_bars=1)),
        "closed_4.5bp": halves(backtest(closed, fee=4.5)),
    }
    by = tr_closed.group_by("session").agg(
        trades=pl.len(), net_bp=pl.col("net_bp").mean()
    )
    r["by_session"] = {s: {"trades": n, "net_bp": b} for s, n, b in by.iter_rows()}
    by = (
        tr_closed.group_by(m=pl.col("time").dt.month())
        .agg(trades=pl.len(), net_bp=pl.col("net_bp").mean())
        .sort("m")
    )
    r["by_month"] = {m: {"trades": n, "net_bp": b} for m, n, b in by.iter_rows()}
    r["ols"] = ols(df)
    r["thresholds"] = thresholds(df)
    r["capacity"] = capacity(df)
    r["asia"] = asia(root, df)
    d = (
        tr_regular.group_by(pl.col("time").dt.date())
        .agg(pl.col("net_bp").sum(), n=pl.len())
        .sort("net_bp")
    )
    r["regular_worst_days"] = {
        "days": [str(x) for x in d["time"][:3]],
        "trades": int(d["n"][:3].sum()),
        "bp": float(d["net_bp"][:3].sum()),
        "total_bp": float(tr_regular["net_bp"].sum()),
        "without_them": summary(
            tr_regular.filter(~pl.col("time").dt.date().is_in(d["time"][:3].implode())),
            START,
            END,
        ),
    }

    (HERE / "results.json").write_text(json.dumps(r, indent=1, default=str))
    tr_closed.write_parquet(HERE / "trades_closed.parquet")
    charts(r["single_feature_ic"], tr_closed, tr_regular)
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()
