# Stock perps are predictable only while Nasdaq is closed

The code behind the study of that name. It asks whether short-horizon order-book signals in
stock perpetuals on trade.xyz (Hyperliquid) predict the next 15 minutes better when the US cash
market is closed than when it is open. They do: four signals that barely work during regular
hours work several times better overnight and at weekends, and a small LightGBM model built on
them is profitable out of sample in closed hours and loses money in regular hours.

Every number in the write-up comes from these scripts, run on data from the public
[Aperiodic API](https://aperiodic.io) through its Python client.

## What's here

| File | What it does |
|---|---|
| `data.py` | Pulls 5-minute `l1_price` (best bid, ask and sizes) and `derivative_price` (oracle) for 28 US stock perps on trade.xyz, three Asian names, and the Binance versions of the same stocks; caches them; builds each symbol's 5-minute panel |
| `model.py` | The New York session calendar, the four features and the target, the monthly walk-forward refits, the backtest and the statistics |
| `study.py` | Every table in the write-up and both charts, into `results.json` and `charts/` |
| `details.py` | Per-refit training size, threshold and IC, feature importance and trade statistics, into `details.json` |
| `facts.py` | The two facts the write-up opens with, into `facts.json` |
| `snippet.py` | The one-signal version: book imbalance on NVDA's perp, weekdays against weekends |
| `make_notebook.py` | Builds `notebook.ipynb`, the same steps one per cell with outputs |
| `results.json`, `details.json`, `facts.json`, `notebook.ipynb`, `charts/` | The outputs, as committed |

## Run it

You need an Aperiodic API key whose plan covers 2026 data, in `APERIODIC_API_KEY` (or in `.env`).

```bash
uv venv .venv
uv pip install -p .venv/bin/python "aperiodic[polars]" lightgbm matplotlib
cd research/closed-market-stock-perps
../../.venv/bin/python snippet.py                          # one signal, one symbol, about a minute
../../.venv/bin/python study.py --root ~/closed_market     # the full study
../../.venv/bin/python details.py --root ~/closed_market   # refit, feature and trade details
```

The first `study.py` run downloads about 90 series (January 1 to November 11, 2026) into
`~/closed_market` and takes a while; later runs read the cache with `--skip-download`. The model
and backtest then run in a few minutes on a laptop.

## The method in brief

- **Sessions** in New York time: overnight (20:00 to 04:00), weekends (Friday 20:00 to Sunday
  20:00) and NYSE holidays count as closed; regular hours (09:30 to 16:00) are the control.
- **Features**, each known at the end of its 5-minute bar: book imbalance at the best quotes, the
  premium of the mid over the oracle less its 1-day mean, the mid's last 15-minute return, and the
  `XYZ100` index perp's last 15-minute return. **Target:** the next 15-minute mid return in bp,
  clipped at ±300.
- **Model:** LightGBM, Huber loss (δ = 30 bp), 7 leaves, at least 2,000 bars per leaf, learning
  rate 0.05, 150 trees. One model pooled across symbols, refit each month from April on an
  expanding window, with a two-day gap before each test month.
- **Trading rule:** enter across the spread when the forecast is among the top 0.1% of the model's
  in-sample forecasts and the spread is under 5 bp; exit after 15 minutes unless the forecast still
  points the same way at half the threshold (up to 6 hours); 1.5 bp fees per side.
- **Statistics:** Sharpe from daily PnL over every calendar day, annualized with √365;
  t-statistics clustered by day.

The backtest fills at the last bid and ask of each 5-minute bar with no latency and leaves funding
out, so it is optimistic. It is research, not investment advice.

## License

MIT, as the rest of this repository.
