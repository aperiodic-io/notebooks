# Backtest slippage

| Field | Value |
|---|---|
| Slug | `backtest-slippage` |
| Use-case group | Improve your backtest |
| Priority | P0 |
| Status | Outline — not implemented |
| Target file | `notebooks/backtest-slippage.py` |
| Preview-runnable | Required: run end-to-end on `DEMO-KEY` and the May 2025 preview slice |

## Goal

Build a transaction-cost model from spread, impact and illiquidity, then show how dynamic costs change a simple momentum backtest. The central comparison is between fantasy mid-price fills, a flat cost assumption and costs calibrated from point-in-time market data.

## Reader takeaway

Execution costs vary with market state and order size. A backtest that models half-spread and impact, and validates those estimates against measured slippage, gives a more credible view of deployable performance than a price-only simulation.

One month of 5-minute data makes every Sharpe ratio and drawdown in this notebook illustrative, not statistically significant. The prose and chart captions must not present the example as evidence of a durable strategy.

## Configuration

Use the standard preview-safe configuration and API-key resolution pattern:

```python
import os
from datetime import date

API_KEY = "..."
if API_KEY == "...":
    API_KEY = os.environ.get("APERIODIC_API_KEY", "DEMO-KEY")

USE_PREVIEW = API_KEY == "DEMO-KEY"
EXCHANGE = "binance-futures"
SYMBOL = "perpetual-BTC-USDT:USDT"
INTERVAL = "5m"
START_DATE = date(2025, 5, 1)
END_DATE = date(2025, 5, 31)
TIMESTAMP = "exchange"

ORDER_NOTIONALS = [10_000, 50_000, 100_000]
FLAT_COST_BPS = 5.0
```

Do not use `load_dotenv` in notebook code and never print `API_KEY`.

Apply the intro notebooks' visual defaults: `sns.set_theme(style="whitegrid", context="talk", palette="deep")` plus their `rcParams`. Keep configuration constants uppercase.

## Data required

Import `get_ohlcv` and `get_metrics` from `aperiodic`. Every call supplies `api_key=API_KEY`, `preview=USE_PREVIEW`, `timestamp=TIMESTAMP`, `interval=INTERVAL`, `exchange=EXCHANGE`, `symbol=SYMBOL`, `start_date=START_DATE`, `end_date=END_DATE`, `output="pandas"` and `show_progress=False`.

Make these exact requests:

```python
ohlcv = get_ohlcv(...)
l1_price = get_metrics(metric="l1_price", ...)
impact = get_metrics(metric="impact", ...)
slippage = get_metrics(metric="slippage", ...)
```

Required fields:

- OHLCV: timestamps and close prices for returns, realised volatility and the toy momentum signal.
- `l1_price`: bid and ask fields, `midprice`, and `weighted_midprice`; derive spread and half-spread in basis points from bid and ask.
- `impact`: `amihud_like`, `kyle_like_lambda`, and `impact_per_notional`.
- `slippage`: `slippage_bps_mean`, `slippage_bps_std`, `slippage_bps_p95`, and `slippage_bps_vwap`.

Normalise timestamps, retain only the intersection needed for comparisons, and state any imputation or forward filling next to the transformation.

## Helpers to reuse

- Copy `clip_window` and `format_time_axis` from the intro notebooks rather than creating incompatible variants.
- Reuse `run_position_backtest(timestamps, position, forward_return, cost_bps_one_way)` from `most-predictive.py` or `alpha-discovery.py`. It returns an equity frame and `{annualized_sharpe, net_return_pct, max_drawdown_pct}`; its annualisation assumes 5-minute bars, matching this notebook.
- Add one local `expected_cost_bps` helper with documented units. It should combine half-spread with an interval-specific impact coefficient multiplied by order notional, without silently clipping negative or missing inputs.

## Cell-by-cell outline

### 1. Why mid-price fills overstate PnL

Introduce half-spread, market impact and opportunity cost. Add a compact Markdown or Matplotlib diagram contrasting a naive mid-price fill with a cost-aware fill; do not add an external image asset.

### 2. Imports, theme and configuration

Set the standard seaborn theme and `rcParams`, declare the configuration above, resolve the API key, and print only the public dataset parameters and preview/full-data mode.

### 3. Fetch and align data

Run the four exact API requests, inspect shapes and coverage, normalise timestamps, and construct a joined analysis frame. Use `clip_window` so every chart uses a comparable period.

### 4. Chart 1 — Spread distribution by hour

Derive spread in basis points from bid and ask. Plot the hourly distribution and a median-by-hour summary so intraday variation is visible.

### 5. Chart 2 — Spread, volatility and measured slippage

Compare spread with rolling realised volatility, then plot the distribution of `slippage_bps_mean`, `slippage_bps_std`, `slippage_bps_p95` and `slippage_bps_vwap`. Explain that measured slippage is the catalogue ground truth used to test the model.

### 6. Build and validate the cost model

Define `expected_cost_bps(order_notional) = half_spread_bps + lambda * order_notional` for each interval, deriving the impact coefficient from the available impact fields with explicit unit conversion. Plot expected-cost curves for `ORDER_NOTIONALS`.

Create a scatter or binned calibration plot of modelled cost against `slippage_bps_mean`. Report calibration error and describe material bias or dispersion. Do not tune the model on the full sample and then imply out-of-sample accuracy.

### 7. Chart 3 — Cost assumptions change the backtest

Build a transparent toy momentum position from lagged returns. Use `run_position_backtest` for three versions: zero costs, flat `FLAT_COST_BPS`, and modelled dynamic one-way costs. Overlay equity curves and show a summary table with annualised Sharpe, net return and maximum drawdown.

### 8. Takeaways

Summarise where the naive backtest is most misleading, how modelled costs compare with measured slippage and which assumptions require calibration for a real order size. Repeat that the one-month results are illustrative.

### 9. Further reading

Link the resources below and point readers to the funding-aware follow-up.

## Acceptance criteria

- Author the notebook as percent-format `.py` and keep it jupytext-synced; never hand-edit `.ipynb`. CI generates `.ipynb` files.
- Do not create or commit an `html/` export.
- Use the literal `API_KEY = "..."` → `APERIODIC_API_KEY` environment variable → `"DEMO-KEY"` resolution shown above; set `USE_PREVIEW = API_KEY == "DEMO-KEY"`, use no `load_dotenv`, and never print secrets.
- Run end-to-end with the exact preview exchange, symbol, interval and May 2025 window.
- Use only `"..."` and `DEMO-KEY` in key examples; do not add UUID-shaped values to source or documentation.
- Use uppercase configuration constants, the shared theme, `clip_window`, `format_time_axis`, numbered `## Chart N — …` sections, `## Takeaways` and `## Further reading`.
- Call `get_ohlcv` and `get_metrics` with every common argument listed under Data required.
- Validate modelled costs against `slippage_bps_mean` and show curves for three order sizes.
- Use `run_position_backtest` for comparable performance metrics and label one-month Sharpe/drawdown results as illustrative.
- Run `ruff format .` and `ruff check .` before opening the implementation PR.

## Cross-links

- [Improve your backtest](https://aperiodic.io/use-cases/improve-your-backtest)
- [Trade metrics](https://aperiodic.io/metrics/trades)
- Planned follow-up: [`backtest-funding`](backtest-funding.md)
- Related notebook: `notebooks/most-predictive.py`
- Related notebook: `notebooks/alpha-discovery.py`
