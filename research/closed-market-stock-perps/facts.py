"""Two facts the post's opening rests on, written to facts.json:

- the NVDA perp's book keeps moving at 3am New York time on a Sunday (raw quote bars, not filled);
- the oracle (index) price keeps updating while the cash market is closed (share of bars where it changed).

    python facts.py --root ~/closed_market
"""

import argparse
import json
from pathlib import Path

import data
import polars as pl
from model import session

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
root = p.parse_args().root.expanduser()

et = pl.col("time").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York")
l1 = pl.read_parquet(root / data.HL / "l1_price" / "NVDA.parquet")
sun3 = l1.with_columns(t=et).filter(
    pl.col("t").dt.weekday() == 7, pl.col("t").dt.hour() == 3
)

rows = []
for t in data.UNIVERSE:
    ix = pl.read_parquet(root / data.HL / "derivative_price" / f"{t}.parquet").sort(
        "time"
    )
    rows.append(
        ix.with_columns(s=session(pl.col("time")), changed=pl.col("index").diff() != 0)
    )
ix = pl.concat(rows).drop_nulls("changed")
oracle = {
    s: float(v) for s, v in ix.group_by("s").agg(pl.col("changed").mean()).iter_rows()
}

out = {
    "nvda_sunday_3am_et": {
        "bars": sun3.height,
        "distinct_mids": ((sun3["bid"] + sun3["ask"]) / 2).n_unique(),
    },
    "oracle_changed_share_by_session": oracle,
}
(Path(__file__).parent / "facts.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
