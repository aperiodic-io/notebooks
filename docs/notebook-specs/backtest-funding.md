# Backtest funding

| Field | Value |
|---|---|
| Slug | `backtest-funding` |
| Use-case group | Improve your backtest |
| Priority | P0 |
| Status | Outline — not implemented |
| Target file | `notebooks/backtest-funding.py` |
| Preview-runnable | Required: run end-to-end on `DEMO-KEY` and the May 2025 preview slice |

## Goal

Build a funding-aware perpetual-futures backtest and isolate how realised funding changes strategy PnL. Compare price-only performance with price plus funding and with a final version that also includes simplified execution costs.

## Reader takeaway

Perpetual positions pay or receive realised funding at discrete prints. Ignoring those cash flows can misattribute carry to trading alpha, especially for a persistently long strategy in a bullish regime.

One month of 5-minute data makes every Sharpe ratio and drawdown in this notebook illustrative, not statistically significant. Takeaway copy must not overclaim persistence or predictive value.

## Configuration

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
FLAT_COST_BPS = 5.0
```

Do not use `load_dotenv` or print the key. Apply `sns.set_theme(style="whitegrid", context="talk", palette="deep")`, the intro notebooks' `rcParams`, and uppercase configuration constants.

## Data required

Import `get_ohlcv` and `get_derivative_metrics` from `aperiodic`. Every call supplies `api_key=API_KEY`, `preview=USE_PREVIEW`, `timestamp=TIMESTAMP`, `interval=INTERVAL`, `exchange=EXCHANGE`, `symbol=SYMBOL`, `start_date=START_DATE`, `end_date=END_DATE`, `output="pandas"` and `show_progress=False`.

Required calls:

```python
ohlcv = get_ohlcv(...)
funding = get_derivative_metrics(metric="funding", ...)
basis = get_derivative_metrics(metric="basis", ...)  # optional context
open_interest = get_derivative_metrics(metric="open_interest", ...)  # optional context
```

Required fields:

- OHLCV timestamps and close prices for forward returns and the toy signal.
- `funding`: `funding_rate`. The catalogue has no predicted-funding metric; the notebook must use realised funding only and say so explicitly.
- Optional context: `basis_bps` from `basis`, and `open_interest` plus `open_interest_pct_change` from `open_interest`.

Keep the funding events at their actual timestamps. Accrue each realised funding print against the position held at that print, forward-filling the position between prints rather than inventing a funding rate on every 5-minute row.

## Helpers to reuse

- Reuse `clip_window` and `format_time_axis` from the intro notebooks.
- Reuse `run_position_backtest(timestamps, position, forward_return, cost_bps_one_way)` from `most-predictive.py` or `alpha-discovery.py`. Its 5-minute annualisation matches the preview interval.
- Add a small, unit-tested-in-notebook helper that maps discrete realised funding prints to held positions and returns signed funding PnL. State the sign convention next to the function.
- Keep the simplified cost model local and clearly labelled. It may use half-spread or a documented flat proxy based on the slippage notebook, but must not imply a full recreation of N1.

## Cell-by-cell outline

### 1. Funding is part of perp PnL

Explain when longs pay or receive funding, why discrete prints matter and how omitting funding can inflate a long-biased backtest. Distinguish realised funding from predicted funding.

### 2. Imports, theme and configuration

Set the standard theme, resolve the key with the preview-safe pattern, and display only non-sensitive public configuration.

### 3. Fetch price and realised funding

Run the exact calls above. Inspect timestamp coverage and retain funding events at their original timestamps. Optionally fetch basis and open interest for context without making them dependencies of the core analysis.

### 4. Chart 1 — Funding regimes

Plot `funding_rate` through May 2025 with positive, negative and near-zero regime shading. If cadence is irregular, show the actual print markers rather than interpolating a continuous series.

### 5. Chart 2 — Cumulative funding for a constant long

Accrue realised funding against a unit long held at each funding print. Plot cumulative funding paid or received and annotate the sign convention.

### 6. Compare three backtests

Build the same transparent toy momentum signal used in the slippage outline. Compare price-only, price plus realised funding, and price plus funding plus simplified costs. Use `run_position_backtest` for price/cost accounting and add the discrete funding cash flows without double-counting turnover.

Overlay equity curves and report annualised Sharpe, net return and maximum drawdown in a summary table. Add a separate column for cumulative funding contribution.

### 7. Chart 3 — Funding as a signal

As a bonus diagnostic, test `position = -sign(funding_rate)` using only information available at each timestamp. Forward-fill the latest observed rate after its print and avoid look-ahead. Frame this as a bridge to alpha research, not as a tradable conclusion.

### 8. Takeaways

State how much of the toy strategy's price-only result came from omitted funding, when the sign changed, and which production details remain. Repeat the illustrative one-month limitation.

### 9. Further reading

Link to the slippage specification, derivatives metrics and the alpha-discovery notebook.

## Acceptance criteria

- Author as percent-format `.py`, jupytext-synced; never hand-edit `.ipynb`, and do not create or commit `html/` exports.
- Use `API_KEY = "..."` → `APERIODIC_API_KEY` → `"DEMO-KEY"`; set `USE_PREVIEW = API_KEY == "DEMO-KEY"`, avoid `load_dotenv`, and never print secrets.
- Use no UUID-shaped example keys in any source or Markdown.
- Run end-to-end with the exact preview exchange, symbol, 5-minute interval and May 2025 dates.
- Use the shared seaborn theme and `rcParams`, uppercase configuration constants, `clip_window`, `format_time_axis`, numbered `## Chart N — …` sections, `## Takeaways` and `## Further reading`.
- Pass all common call arguments listed above to `get_ohlcv` and `get_derivative_metrics`.
- Use realised `funding_rate` only; preserve funding-print timing and explain the cash-flow sign convention.
- Compare price-only, price plus funding, and price plus funding plus costs using the reusable backtest helper.
- Label all one-month Sharpe/drawdown statistics as illustrative.
- Run `ruff format .` and `ruff check .` before opening the implementation PR.

## Cross-links

- [Improve your backtest](https://aperiodic.io/use-cases/improve-your-backtest)
- [Derivatives metrics](https://aperiodic.io/metrics/derivatives)
- Planned prerequisite: [`backtest-slippage`](backtest-slippage.md)
- Related notebook: `notebooks/alpha-discovery.py`
