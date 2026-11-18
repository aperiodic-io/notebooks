"""Builds and runs notebook.ipynb: study.py's steps one per cell, outputs saved.

python make_notebook.py --root ~/closed_market
"""

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--out", type=Path, default=None)
args = p.parse_args()
root = args.root  # kept as typed, so the notebook shows no local path
out = args.out or (
    HERE.parent / "notebook.ipynb"
    if (HERE.parent / "post.md").exists()
    else HERE / "notebook.ipynb"
)

cells = [
    (
        "md",
        (
            "# When the exchange closes: microstructure in stock perps\n\nEvery number in `post.md` comes from a cell below. "
            'Data: the public Aperiodic Python client (`pip install "aperiodic[polars]"`, key in `APERIODIC_API_KEY`).'
        ),
    ),
    (
        "code",
        f"import json, datetime as dt, polars as pl\nimport data, study\nfrom model import *\nROOT = __import__('pathlib').Path({root!r}).expanduser()\ndata.download(ROOT)   # cached after the first run",
    ),
    (
        "code",
        (
            "panel = pl.concat([data.bars(ROOT, t) for t in data.UNIVERSE], how='diagonal_relaxed')\n"
            "df = features(panel, data.INDEX)\n"
            "s = df.filter(pl.col('time').dt.date() < study.END)['session'].value_counts(normalize=True).sort('session')\n"
            "print(df['sym'].n_unique(), 'symbols'); print(s); print('closed share', s.filter(pl.col('session').is_in(CLOSED))['proportion'].sum())"
        ),
    ),
    ("md", "## Trade prices bounce; mid prices do not"),
    (
        "code",
        "study.bounce(panel)   # (mean IC over symbols, t across symbols, symbols)",
    ),
    ("md", "## Each feature alone, by session"),
    ("code", "feat = study.single_features(df); feat"),
    ("code", "study.binance_gap(ROOT, df)"),
    ("code", "study.median_usd = study.depth(df); study.median_usd"),
    ("md", "## The four-feature model, walked forward"),
    (
        "code",
        (
            "closed = walk_forward(df, CLOSED, study.TEST_MONTHS, Model())\nregular = walk_forward(df, REGULAR, study.TEST_MONTHS, Model())\n"
            "print('rank IC closed', rank_ic(closed), 'regular', rank_ic(regular))\n"
            "{m: rank_ic(closed.filter(pl.col('time').dt.month() == m)) for m in study.TEST_MONTHS}"
        ),
    ),
    (
        "code",
        (
            "tc, tr = backtest(closed), backtest(regular)\n"
            "rows = {'closed': tc, 'regular': tr, 'closed 3bp': backtest(closed, fee=3.0), 'closed 5m late': backtest(closed, delay_bars=1),\n"
            "        'closed 3bp + late': backtest(closed, fee=3.0, delay_bars=1), 'closed 4.5bp': backtest(closed, fee=4.5)}\n"
            "pl.DataFrame([{'run': k, **study.halves(v)} for k, v in rows.items()])"
        ),
    ),
    (
        "code",
        (
            "print(tc.group_by('session').agg(trades=pl.len(), net_bp=pl.col('net_bp').mean()).sort('session'))\n"
            "tc.group_by(m=pl.col('time').dt.month()).agg(trades=pl.len(), net_bp=pl.col('net_bp').mean()).sort('m')"
        ),
    ),
    ("md", "## The simpler alternatives"),
    ("code", "study.ols(df)"),
    ("code", "study.thresholds(df)"),
    ("code", "study.capacity(df)"),
    ("md", "## Asian underlyings"),
    ("code", "study.asia(ROOT, df)"),
    ("md", "## Charts"),
    ("code", "study.charts(feat, tc, tr)"),
]
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s)
    for k, s in cells
]
NotebookClient(
    nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}
).execute()
nbf.write(nb, out)
print(f"wrote {out}")
