# Risk positioning

| Field | Value |
|---|---|
| Slug | `risk-positioning` |
| Use-case group | Read positioning & risk |
| Priority | P2 |
| Status | Outline — not implemented |
| Target file | `notebooks/risk-positioning.py` |
| Preview-runnable | Required: run end-to-end on `DEMO-KEY` and the May 2025 preview slice |

## Goal

Combine funding, basis and open interest into a transparent crowding score, then examine forward returns, volatility and in-window drawdowns when leverage indicators point in the same direction.

## Reader takeaway

Funding, basis and open interest describe different parts of positioning pressure. Rolling percentile ranks make them comparable, but a high composite score is a risk-state diagnostic rather than a deterministic crash forecast.

One month of 5-minute data makes conditional returns, Sharpe ratios and drawdown thresholds illustrative, not statistically significant. The notebook must avoid claiming that the observed thresholds generalise beyond the preview window.

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

RANK_WINDOW = 7 * 24 * 12
FORWARD_HORIZONS = [12, 72, 288]
```

Do not use `load_dotenv` or print the key. Use `sns.set_theme(style="whitegrid", context="talk", palette="deep")`, the intro notebooks' `rcParams`, and uppercase configuration constants.

## Data required

Import `get_ohlcv` and `get_derivative_metrics` from `aperiodic`. Every request supplies `api_key=API_KEY`, `preview=USE_PREVIEW`, `timestamp=TIMESTAMP`, `interval=INTERVAL`, `exchange=EXCHANGE`, `symbol=SYMBOL`, `start_date=START_DATE`, `end_date=END_DATE`, `output="pandas"` and `show_progress=False`.

Make these exact requests:

```python
ohlcv = get_ohlcv(...)
funding = get_derivative_metrics(metric="funding", ...)
basis = get_derivative_metrics(metric="basis", ...)
open_interest = get_derivative_metrics(metric="open_interest", ...)
derivative_price = get_derivative_metrics(metric="derivative_price", ...)
```

Required fields:

- OHLCV timestamps and close prices for returns, realised volatility and drawdowns.
- `funding`: `funding_rate`.
- `basis`: `basis_bps`.
- `open_interest`: `open_interest` and `open_interest_pct_change`.
- `derivative_price`: `last_price`, `index_price`, `mark_price`, `last_price_pct_change`, `index_price_pct_change` and `mark_price_pct_change`.

Align sparse derivative observations without pulling later values backwards. Preserve the source timestamps and make any forward-fill horizon explicit.

## Helpers to reuse

- Reuse `clip_window` and `format_time_axis` from the intro notebooks.
- Reuse the rolling rank transform from `alpha-discovery.py` to express funding, basis and open-interest change on comparable percentile scales. Calculate ranks from trailing data only.
- Reuse `run_position_backtest(timestamps, position, forward_return, cost_bps_one_way)` only for an optional risk-overlay diagnostic; its annualisation assumes 5-minute bars.
- Add compact helpers for forward-return/forward-volatility horizons and maximum subsequent drawdown, each named with units or bar counts.

## Cell-by-cell outline

### 1. Crowded leverage builds before it unwinds

Introduce the funding-plus-basis-plus-open-interest framing used by Aperiodic's derivatives metrics: funding shows financing pressure, basis shows futures-versus-spot dislocation and OI shows outstanding leverage. Explain why agreement is more informative than any single series.

### 2. Imports, theme and configuration

Apply the shared visual defaults, resolve the key safely and display only public configuration. Translate each forward horizon from 5-minute bars into a human-readable duration.

### 3. Fetch and align derivative data

Run the five exact requests, inspect response fields and coverage, and create one analysis frame without look-ahead. Use derivative price as a cross-check against OHLCV rather than silently replacing the return source.

### 4. Chart 1 — Positioning components

Plot funding, basis and open interest with price context. Annotate the strongest coincident moves in the May 2025 window without labelling them as universal regimes.

### 5. Build the crowding score

Calculate trailing rolling percentile ranks for funding, basis and open-interest change using the alpha-discovery rank transform. Document directionality, missing-value policy and whether the composite is a simple mean or another transparent fixed-weight combination.

Plot the component ranks and composite score. If both long and short crowding are represented, preserve the sign instead of collapsing everything to magnitude.

### 6. Chart 2 — Event study by crowding decile

Bucket observations by composite-score decile. Compare forward returns and realised volatility across `FORWARD_HORIZONS`, with sample counts and uncertainty summaries. Avoid overlapping-window language that implies independent observations.

### 7. Chart 3 — Sharpest in-window moves

Identify and annotate the largest forward moves and drawdowns in May 2025. Show the crowding score known immediately before each move, using only trailing inputs.

### 8. Threshold calibration table

For a small, predeclared set of crowding-score thresholds, report observation count, subsequent return, realised volatility and maximum subsequent drawdown. Describe this as in-window calibration, not a production threshold or out-of-sample validation.

### 9. Takeaways

Summarise which components agreed before the sharpest moves, where the composite failed to warn and how Live/Institutional data could support ongoing monitoring. Repeat the one-month limitation.

### 10. Further reading

Link to the positioning use-case, derivatives metrics and the introductory derivatives notebook.

## Acceptance criteria

- Author as percent-format `.py`, jupytext-synced; never hand-edit `.ipynb`, and do not create or commit `html/` exports.
- Use `API_KEY = "..."` → `APERIODIC_API_KEY` → `"DEMO-KEY"`; set `USE_PREVIEW = API_KEY == "DEMO-KEY"`, use no `load_dotenv`, and never print secrets.
- Use no UUID-shaped example keys in source or documentation.
- Run end-to-end with the exact preview exchange, symbol, 5-minute interval and May 2025 dates.
- Use the shared theme and `rcParams`, uppercase configuration constants, `clip_window`, `format_time_axis`, numbered `## Chart N — …` sections, `## Takeaways` and `## Further reading`.
- Pass every common argument listed above to `get_ohlcv` and all four `get_derivative_metrics` calls.
- Build the score from trailing rolling percentile ranks of funding, basis and open-interest change without look-ahead.
- Include the decile event study, sharp-move annotations and a threshold calibration table with sample counts.
- Frame the score as a risk diagnostic, the thresholds as in-window calibration and all one-month statistics as illustrative.
- Run `ruff format .` and `ruff check .` before opening the implementation PR.

## Cross-links

- [Read positioning & risk](https://aperiodic.io/use-cases/positioning-and-risk)
- [Derivatives metrics](https://aperiodic.io/metrics/derivatives)
- Related notebook: `notebooks/intro-derivatives.py`
- Related notebook: `notebooks/alpha-discovery.py`
