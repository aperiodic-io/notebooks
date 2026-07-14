# Execution liquidity timing

| Field | Value |
|---|---|
| Slug | `execution-liquidity-timing` |
| Use-case group | Execute smarter |
| Priority | P1 |
| Status | Outline — not implemented |
| Target file | `notebooks/execution-liquidity-timing.py` |
| Preview-runnable | Required: run end-to-end on `DEMO-KEY` and the May 2025 preview slice |

## Goal

Measure when liquidity is deep, spreads are narrow and order flow is least adverse, then compare a simple time-of-day execution schedule with uniform and deliberately poor schedules.

## Reader takeaway

Execution cost is time-varying. For moderate-frequency order sizes, scheduling around recurring liquidity and toxicity conditions can matter more than submitting immediately, while historical calibration still needs Live data for operation.

One month of 5-minute data makes all cost differences, Sharpe-like summaries and drawdown figures illustrative, not statistically significant. The notebook must not claim that May 2025 seasonality will persist.

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

ORDER_NOTIONAL = 50_000
BEST_K_HOURS = 4
```

Do not use `load_dotenv` or print the key. Apply `sns.set_theme(style="whitegrid", context="talk", palette="deep")`, the intro notebooks' `rcParams`, and uppercase configuration constants.

## Data required

Import `get_ohlcv` and `get_metrics` from `aperiodic`. Every request supplies `api_key=API_KEY`, `preview=USE_PREVIEW`, `timestamp=TIMESTAMP`, `interval=INTERVAL`, `exchange=EXCHANGE`, `symbol=SYMBOL`, `start_date=START_DATE`, `end_date=END_DATE`, `output="pandas"` and `show_progress=False`.

Make these exact requests:

```python
ohlcv = get_ohlcv(...)
l1_price = get_metrics(metric="l1_price", ...)
l2_liquidity = get_metrics(metric="l2_liquidity", ...)
l2_imbalance = get_metrics(metric="l2_imbalance", ...)
flow = get_metrics(metric="flow", ...)
impact = get_metrics(metric="impact", ...)
```

Required fields and derivations:

- OHLCV: timestamps, close and volume for realised volatility and short-horizon adverse moves.
- `l1_price`: bid and ask fields used to calculate spread in basis points.
- `l2_liquidity`: `bid_agg_5`, `bid_agg_10`, `bid_agg_20`, `bid_agg_25`, `ask_agg_5`, `ask_agg_10`, `ask_agg_20`, `ask_agg_25`, and their `_avg` counterparts.
- `l2_imbalance`: `imbalance_5`, `imbalance_10`, `imbalance_20`, `imbalance_25`, `imbalance_ratio_5`, `imbalance_ratio_10`, `imbalance_ratio_20`, `imbalance_ratio_25`, and their `_avg` counterparts.
- `flow`: `flow_toxicity_score`, `volume_delta`, `volume_delta_notional`, and `taker_buy_sell_ratio`.
- `impact`: `amihud_like`, `kyle_like_lambda` and `impact_per_notional` for the cost model.

The L2 datasets are tier 3. They are included in the `DEMO-KEY` preview slice, so the notebook must run in preview mode; full historical access requires a Prime subscription, which closing Markdown must state.

## Helpers to reuse

- Reuse `clip_window` and `format_time_axis` from the intro notebooks.
- Reuse `run_position_backtest(timestamps, position, forward_return, cost_bps_one_way)` only if a comparable equity diagnostic is useful. Its annualisation assumes 5-minute observations.
- Adapt the explicitly unit-labelled cost model from [`backtest-slippage`](backtest-slippage.md): half-spread plus an impact coefficient multiplied by notional.
- Add small helpers for hour/day grouping and for ranking candidate execution hours. Fit the ranking from historical bins only; do not use future values for a row-level decision.

## Cell-by-cell outline

### 1. Execution cost changes through time

Explain why spread, depth, volatility and informed flow move together around stressed periods. Define the exercise as historical scheduling for moderate-frequency execution, not low-latency routing.

### 2. Imports, theme and configuration

Set the shared theme, resolve the API key safely, and display only public configuration. State that all timestamps are converted to one documented timezone before hour/day aggregation.

### 3. Fetch and align market data

Run the six exact requests, report coverage, normalise timestamps and join only fields used below. Make missing-data handling visible.

### 4. Chart 1 — Spread seasonality

Plot median spread as an hour-of-day by day-of-week heatmap. Add observation counts or mask under-populated bins so the chart does not give false precision.

### 5. Chart 2 — Multi-level depth seasonality

Plot the chosen L2 depth measure at 5, 10, 20 and 25 levels, including an hour/day heatmap and a compact comparison of depth profiles. Explain how deeper levels change the available notional picture.

### 6. Chart 3 — Volatility and adverse flow

Overlay realised volatility with spread/depth summaries. Relate L2 imbalance and selected flow features to short-horizon adverse moves using lagged features only. Present the relationship as a toxicity-style diagnostic, not a causal claim.

### 7. Build a liquidity score and scheduler

Combine normalised spread, depth, impact and toxicity inputs into a transparent score. Rank hours and define three schedules for a fixed `ORDER_NOTIONAL`: best `BEST_K_HOURS`, uniform across the day, and worst `BEST_K_HOURS`.

Use the slippage cost model to calculate expected basis-point cost per schedule. Show the absolute cost and the difference versus uniform. Keep selection and cost units visible.

### 8. Chart 4 — When to stand aside

Overlay scheduled hours with the toxicity diagnostic and highlight intervals where nominal depth is high but adverse-flow risk is also elevated. Explain why a single liquidity measure is insufficient.

### 9. Takeaways

Summarise the in-window schedule differences, the sensitivity to order notional and the historical-versus-Live distinction. State that full L2 history requires Prime, while all demonstrated calls run with the preview key.

### 10. Further reading

Link to execution use-case and metric pages, the slippage model and the L1 introduction.

## Acceptance criteria

- Author as percent-format `.py`, jupytext-synced; never hand-edit `.ipynb`, and do not create or commit `html/` exports.
- Use `API_KEY = "..."` → `APERIODIC_API_KEY` → `"DEMO-KEY"`; set `USE_PREVIEW = API_KEY == "DEMO-KEY"`, use no `load_dotenv`, and never print secrets.
- Use no UUID-shaped key examples in source or documentation.
- Run end-to-end with the exact preview exchange, symbol, 5-minute interval and May 2025 dates, including tier-3 L2 calls.
- Use the shared theme and `rcParams`, uppercase configuration constants, `clip_window`, `format_time_axis`, numbered `## Chart N — …` sections, `## Takeaways` and `## Further reading`.
- Pass every common argument listed above to `get_ohlcv` and all five `get_metrics` calls.
- Compare best-hour, uniform and worst-hour schedules for a fixed notional using the spread-plus-impact cost model.
- Include a toxicity overlay and avoid look-ahead in feature/forward-move comparisons.
- State that full L2 history requires Prime and that one-month figures are illustrative.
- Run `ruff format .` and `ruff check .` before opening the implementation PR.

## Cross-links

- [Execute smarter](https://aperiodic.io/use-cases/execute-smarter)
- [L1 metrics](https://aperiodic.io/metrics/l1)
- [L2 metrics](https://aperiodic.io/metrics/l2)
- [Trade metrics](https://aperiodic.io/metrics/trades)
- Cost-model prerequisite: [`backtest-slippage`](backtest-slippage.md)
- Related notebook: `notebooks/intro-l1-price.py`
