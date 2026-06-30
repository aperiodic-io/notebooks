"""Pipeline helpers for the guided alpha-discovery walkthrough notebook.

These functions power the ``alpha-discovery-walkthrough`` notebook. They live in a
module (rather than inline in the notebook) so the walkthrough itself stays short
and skimmable: each step calls one helper and explains *what* it does, while the
*how* lives here for anyone who wants to read it.

The research logic is identical to the original walkthrough — it has only been
parameterized through :class:`WalkthroughConfig` so every parameter sits in one place.
"""

from __future__ import annotations

import base64
import datetime
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aperiodic import get_derivative_metrics, get_metrics, get_ohlcv
from IPython.display import HTML

from ._aperiodic_demo import run_position_backtest

# The roadmap groups the six steps into three phases, each with its own color.
# Colors track the Aperiodic brand accent triad — blue #5a9fd4, terracotta
# #d4845a, green #72b866. The SVG ships only light-theme colors; the export
# template (templates/aperiodic) dark-inverts output images via
# filter: invert()/hue-rotate(), so we use deepened variants that read on the
# warm cream background (--ap-bg #faf8f5) and invert back toward the lighter
# brand accents in dark mode. Each step is (number, line 1, line 2, icon key).
_PIPELINE_PHASES = [
    {"name": "CONSTRUCTION", "color": "#3f78ad", "steps": (0, 1)},
    {"name": "SEARCH", "color": "#c0703f", "steps": (2, 3)},
    {"name": "VALIDATION", "color": "#4f8a50", "steps": (4, 5)},
]
_PIPELINE_STEPS = [
    ("1", "Load &amp;", "inspect", "table"),
    ("2", "Construct", "signal", "wave"),
    ("3", "Search", "grid", "grid"),
    ("4", "Compare", "candidates", "bars"),
    ("5", "Examine", "best", "lens"),
    ("6", "Robustness", "check", "shield"),
]
# Neutral tokens use the canonical Aperiodic warm-neutral ramp (the deck/catalog
# REFERENCE_PALETTE that apps/data ships): body text fg70 #4a463e and a muted
# fg40 #9b958b rule, on the Mona Sans stack (falls back to system fonts in the
# standalone image, matching the template's own fallback).
_DIAGRAM_TEXT = "#4a463e"
_DIAGRAM_ARROW = "#9b958b"
_DIAGRAM_FONT = (
    "'Mona Sans', -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', "
    "Helvetica, Arial, sans-serif"
)


def _phase_color(step_idx: int) -> str:
    for phase in _PIPELINE_PHASES:
        if step_idx in phase["steps"]:
            return phase["color"]
    return _PIPELINE_PHASES[0]["color"]


def _step_icon(kind: str, cx: int, color: str) -> str:
    """Return a small line-art glyph for a step, centered horizontally on ``cx``."""
    stroke = f'fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"'
    if kind == "table":
        icon = (
            f'<rect x="{cx - 13}" y="82" width="26" height="20" rx="3" {stroke}/>'
            f'<line x1="{cx - 13}" y1="89" x2="{cx + 13}" y2="89" stroke="{color}" stroke-width="2"/>'
            f'<line x1="{cx}" y1="89" x2="{cx}" y2="102" stroke="{color}" stroke-width="1.5"/>'
        )
    elif kind == "wave":
        icon = (
            f'<polyline points="{cx - 13},97 {cx - 7},85 {cx - 1},93 {cx + 5},82 '
            f'{cx + 13},90" {stroke} stroke-linecap="round"/>'
        )
    elif kind == "grid":
        icon = (
            f'<rect x="{cx - 12}" y="80" width="24" height="24" rx="2" {stroke}/>'
            f'<line x1="{cx - 4}" y1="80" x2="{cx - 4}" y2="104" stroke="{color}" stroke-width="1.4"/>'
            f'<line x1="{cx + 4}" y1="80" x2="{cx + 4}" y2="104" stroke="{color}" stroke-width="1.4"/>'
            f'<line x1="{cx - 12}" y1="88" x2="{cx + 12}" y2="88" stroke="{color}" stroke-width="1.4"/>'
            f'<line x1="{cx - 12}" y1="96" x2="{cx + 12}" y2="96" stroke="{color}" stroke-width="1.4"/>'
        )
    elif kind == "bars":
        icon = (
            f'<rect x="{cx - 12}" y="94" width="6" height="10" rx="1" fill="{color}"/>'
            f'<rect x="{cx - 3}" y="84" width="6" height="20" rx="1" fill="{color}"/>'
            f'<rect x="{cx + 6}" y="90" width="6" height="14" rx="1" fill="{color}"/>'
        )
    elif kind == "lens":
        icon = (
            f'<circle cx="{cx - 3}" cy="89" r="8" {stroke}/>'
            f'<line x1="{cx + 3}" y1="95" x2="{cx + 11}" y2="103" stroke="{color}" '
            'stroke-width="2.2" stroke-linecap="round"/>'
        )
    else:  # shield
        icon = (
            f'<path d="M {cx} 80 L {cx + 11} 84 L {cx + 11} 92 C {cx + 11} 99 {cx + 6} 103 '
            f'{cx} 105 C {cx - 6} 103 {cx - 11} 99 {cx - 11} 92 L {cx - 11} 84 Z" {stroke}/>'
            f'<polyline points="{cx - 5},91 {cx - 1},96 {cx + 6},87" {stroke} '
            'stroke-linecap="round"/>'
        )
    return icon


def pipeline_diagram() -> HTML:
    """Return a responsive SVG roadmap of the six walkthrough steps.

    The steps are grouped into three color-coded phases (construction, search,
    validation), each node carrying a small glyph. Call it from a code cell so the
    image lands in an output area and inherits the same dark-mode handling as the
    notebook's charts; colors target the light theme and the export template
    inverts output-area images for the dark theme.

    The SVG is delivered as a base64 ``<img>`` because nbconvert lower-cases inline
    SVG attribute names, which would corrupt the case-sensitive ``viewBox``.
    """
    node_w, gap = 132, 33

    def node_x(idx: int) -> int:
        return 2 + idx * (node_w + gap)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 180" '
        f'role="img" font-family="{_DIAGRAM_FONT}" '
        f'aria-label="Alpha discovery pipeline: construction, search, validation">',
        "<title>Alpha discovery pipeline</title>",
    ]

    # Phase headers: a color bar spanning each pair of nodes, with a caption above.
    for phase in _PIPELINE_PHASES:
        first, last = phase["steps"][0], phase["steps"][-1]
        left = node_x(first)
        right = node_x(last) + node_w
        center = (left + right) // 2
        parts.append(
            f'<rect x="{left}" y="32" width="{right - left}" height="3" rx="1.5" '
            f'fill="{phase["color"]}"/>'
        )
        parts.append(
            f'<text x="{center}" y="23" text-anchor="middle" font-size="12" '
            f'font-weight="700" letter-spacing="1.5" fill="{phase["color"]}">'
            f"{phase['name']}</text>"
        )

    # Connecting arrows between consecutive nodes.
    for i in range(len(_PIPELINE_STEPS) - 1):
        arrow_x = node_x(i) + node_w + 4
        parts.append(
            f'<line x1="{arrow_x}" y1="106" x2="{arrow_x + 17}" y2="106" '
            f'stroke="{_DIAGRAM_ARROW}" stroke-width="2"/>'
        )
        parts.append(
            f'<polygon points="{arrow_x + 17},101 {arrow_x + 17},111 '
            f'{arrow_x + 27},106" fill="{_DIAGRAM_ARROW}"/>'
        )

    # Step cards.
    for i, (number, line1, line2, icon_kind) in enumerate(_PIPELINE_STEPS):
        x = node_x(i)
        cx = x + node_w // 2
        color = _phase_color(i)
        parts.append(
            f'<rect x="{x}" y="46" width="{node_w}" height="120" rx="6" '
            f'fill="{color}" fill-opacity="0.06" stroke="{color}" '
            'stroke-opacity="0.55" stroke-width="1.5"/>'
        )
        parts.append(_step_icon(icon_kind, cx, color))
        parts.append(f'<circle cx="{x + 17}" cy="64" r="11" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 17}" y="68" text-anchor="middle" font-size="13" '
            f'font-weight="700" fill="#ffffff">{number}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="130" text-anchor="middle" font-size="12.5" '
            f'font-weight="600" fill="{_DIAGRAM_TEXT}">{line1}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="147" text-anchor="middle" font-size="12.5" '
            f'font-weight="600" fill="{_DIAGRAM_TEXT}">{line2}</text>'
        )

    parts.append("</svg>")

    encoded = base64.b64encode("".join(parts).encode()).decode()
    return HTML(
        f'<img src="data:image/svg+xml;base64,{encoded}" '
        'alt="Alpha discovery pipeline roadmap" '
        'style="width:100%;max-width:940px;height:auto;display:block;'
        'margin:0.75rem auto"/>'
    )


@dataclass(frozen=True)
class WalkthroughConfig:
    """All parameters for the alpha-discovery walkthrough in one place.

    Attributes
    ----------
    api_key:
        Aperiodic API key.
    symbol, exchange, interval, timestamp:
        The instrument and resolution to research.
    start_date, end_date:
        Inclusive date range pulled from the preview dataset.
    metrics:
        ``(metric, kind)`` pairs to fetch, where ``kind`` is ``"derivative"`` or
        ``"regular"``.
    rank_windows:
        Look-back lengths used to turn a feature into a rolling percentile-rank
        signal.
    smooth_windows:
        Moving-average windows applied to the ranked signal. ``None`` means "use
        the raw ranked signal without smoothing."
    cost_bps:
        One-way transaction cost (basis points) used by the position backtest.
    """

    api_key: str
    symbol: str
    exchange: str
    interval: str
    timestamp: str
    start_date: datetime.date
    end_date: datetime.date
    metrics: list[tuple[str, str]]
    rank_windows: list[int]
    smooth_windows: list[int | None]
    cost_bps: float


def get_numeric_metric_frame(
    config: WalkthroughConfig, metric: str, kind: str
) -> pd.DataFrame | None:
    """Fetch one metric family and keep its numeric columns, de-duplicated by time."""
    fetcher = get_derivative_metrics if kind == "derivative" else get_metrics
    raw_frame = fetcher(
        api_key=config.api_key,
        metric=metric,
        timestamp=config.timestamp,
        interval=config.interval,
        exchange=config.exchange,
        symbol=config.symbol,
        start_date=config.start_date,
        end_date=config.end_date,
        output="pandas",
        show_progress=True,
        preview=True,
    )

    frame = (
        raw_frame.to_pandas()
        if hasattr(raw_frame, "to_pandas")
        else pd.DataFrame(raw_frame)
    )
    if frame.empty or "time" not in frame.columns:
        print(f"Skipping {metric}: no rows returned.")
        return None

    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    if "time" in numeric_cols:
        numeric_cols.remove("time")

    if not numeric_cols:
        print(f"Skipping {metric}: no numeric columns returned.")
        return None

    return frame.sort_values("time").drop_duplicates(subset=["time"], keep="last")[
        ["time", *numeric_cols]
    ]


def build_panel(
    config: WalkthroughConfig,
) -> tuple[pd.DataFrame, list[str], dict[str, tuple[str, str]]]:
    """Assemble the close-price + feature panel and the one-bar-ahead return target.

    Returns the panel, the list of usable feature columns, and a mapping from each
    feature column back to the ``(metric, kind)`` it came from (handy for showing a
    reader exactly which API call produced a winning feature).
    """
    raw_ohlcv = get_ohlcv(
        api_key=config.api_key,
        timestamp=config.timestamp,
        interval=config.interval,
        exchange=config.exchange,
        symbol=config.symbol,
        start_date=config.start_date,
        end_date=config.end_date,
        output="pandas",
        show_progress=True,
        preview=True,
    )

    panel = (
        raw_ohlcv.to_pandas()
        if hasattr(raw_ohlcv, "to_pandas")
        else pd.DataFrame(raw_ohlcv)
    )
    panel = panel.sort_values("time")[["time", "close"]]

    feature_sources: dict[str, tuple[str, str]] = {}
    for metric, kind in config.metrics:
        frame = get_numeric_metric_frame(config, metric, kind)
        if frame is None:
            continue

        feature_values = [col for col in frame.columns if col != "time"]
        if frame[feature_values].notna().sum().sum() == 0:
            print(f"Skipping {metric}: all feature values are null.")
            continue

        panel = panel.merge(frame, on="time", how="left")
        for col in feature_values:
            feature_sources[col] = (metric, kind)

    panel = panel.sort_values("time")
    feature_cols = [
        col
        for col in panel.columns
        if col not in {"time", "close"} and pd.api.types.is_numeric_dtype(panel[col])
    ]

    panel[feature_cols] = panel[feature_cols].ffill()
    panel["fwd_ret"] = panel["close"].pct_change().shift(-1)
    panel = panel.dropna(subset=["fwd_ret"])

    feature_sources = {
        col: feature_sources[col] for col in feature_cols if col in feature_sources
    }
    return panel, feature_cols, feature_sources


def summarize_panel(panel_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Coverage and dispersion snapshot for the first handful of feature columns."""
    coverage_rows = [
        {
            "feature": feature,
            "non_null_pct": float(panel_df[feature].notna().mean() * 100.0),
            "std": float(panel_df[feature].std()),
        }
        for feature in feature_cols[:10]
    ]
    return pd.DataFrame(coverage_rows)


def make_rank_signal(panel_df: pd.DataFrame, feature: str, window: int) -> np.ndarray:
    """Turn a raw feature into a rolling percentile-rank signal scaled to ``[-1, 1]``."""
    rank = panel_df[feature].rolling(window).rank(method="average")
    signal = ((rank - 1.0) / (window - 1)) * 2.0 - 1.0
    return signal.to_numpy().astype(np.float64)


def smooth_signal(signal: np.ndarray, window: int | None) -> np.ndarray:
    """Optionally smooth a ranked signal with a trailing moving average."""
    if window is None or pd.isna(window) or int(window) == 1:
        return signal.astype(np.float64, copy=True)

    return pd.Series(signal).rolling(int(window)).mean().to_numpy().astype(np.float64)


def evaluate_strategies(
    config: WalkthroughConfig, panel_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Backtest every feature x rank-window x smooth-window combination.

    Returns a results frame sorted by Sharpe (best first) and the forward-return
    array the search was scored against.
    """
    forward_returns = panel_df["fwd_ret"].to_numpy().astype(np.float64)
    results: list[dict[str, float | int | str | None]] = []

    for feature in feature_cols:
        for rank_window in config.rank_windows:
            raw_signal = make_rank_signal(panel_df, feature, rank_window)

            for smooth_window in config.smooth_windows:
                signal = smooth_signal(raw_signal, smooth_window)
                mask = np.isfinite(signal) & np.isfinite(forward_returns)

                signal_valid = signal[mask]
                returns_valid = forward_returns[mask]
                if (
                    signal_valid.size < 2
                    or np.std(signal_valid) == 0.0
                    or np.std(returns_valid) == 0.0
                ):
                    continue

                fit_corr = float(np.corrcoef(signal_valid, returns_valid)[0, 1])
                if not np.isfinite(fit_corr):
                    continue

                direction = 1 if fit_corr >= 0 else -1
                _, bt_summary = run_position_backtest(
                    timestamps=panel_df.loc[mask, "time"],
                    position=np.nan_to_num(
                        np.clip(signal_valid * direction, -1.0, 1.0), nan=0.0
                    ),
                    forward_return=returns_valid,
                    cost_bps_one_way=config.cost_bps,
                )

                results.append(
                    {
                        "feature": feature,
                        "rank_window": rank_window,
                        "smooth_window": smooth_window,
                        "direction": direction,
                        "fit_corr": fit_corr,
                        "sharpe": bt_summary["annualized_sharpe"],
                        "return_pct": bt_summary["net_return_pct"],
                        "drawdown_pct": bt_summary["max_drawdown_pct"],
                    }
                )

    if not results:
        raise RuntimeError(
            "No valid feature/window combinations were produced. "
            "Check the fetched metric coverage for the selected date range."
        )

    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    return results_df, forward_returns


def summarize_top_strategies(results_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Readable top-``n`` table with smooth-window and direction prettified."""
    summary = results_df.head(top_n).copy()
    summary["smooth_window"] = summary["smooth_window"].map(
        lambda value: "None" if value is None or pd.isna(value) else int(value)
    )
    summary["direction"] = summary["direction"].map({1: "long", -1: "short"})
    return summary[
        [
            "feature",
            "rank_window",
            "smooth_window",
            "direction",
            "fit_corr",
            "sharpe",
            "return_pct",
            "drawdown_pct",
        ]
    ]


def build_strategy_artifacts(
    config: WalkthroughConfig,
    panel_df: pd.DataFrame,
    forward_returns: np.ndarray,
    row: pd.Series,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, float], np.ndarray]:
    """Rebuild one strategy's raw/smoothed signal, backtest frame, summary, and mask."""
    raw_signal = make_rank_signal(
        panel_df, str(row["feature"]), int(row["rank_window"])
    )
    signal = smooth_signal(raw_signal, row["smooth_window"]) * int(row["direction"])
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    bt_frame, bt_summary = run_position_backtest(
        timestamps=panel_df.loc[mask, "time"],
        position=np.nan_to_num(np.clip(signal[mask], -1.0, 1.0), nan=0.0),
        forward_return=forward_returns[mask],
        cost_bps_one_way=config.cost_bps,
    )
    return raw_signal, signal, bt_frame, bt_summary, mask


def plot_strategy_overview(
    config: WalkthroughConfig,
    panel_df: pd.DataFrame,
    forward_returns: np.ndarray,
    row: pd.Series,
    title_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Three-panel view: raw rank signal, smoothed signal, and equity curve."""
    raw_signal, signal, bt_frame, _, mask = build_strategy_artifacts(
        config, panel_df, forward_returns, row
    )
    smooth_label = (
        "None"
        if row["smooth_window"] is None or pd.isna(row["smooth_window"])
        else int(row["smooth_window"])
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(panel_df["time"], raw_signal, linewidth=0.8, color="tab:orange")
    axes[0].axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    axes[0].set_title(f"{title_prefix} raw rank signal | {row['feature']}")
    axes[0].grid(alpha=0.2)

    axes[1].plot(panel_df["time"], signal, linewidth=0.9, color="tab:red")
    axes[1].axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    axes[1].set_title(
        f"{title_prefix} smoothed signal"
        f" | rank={int(row['rank_window'])}"
        f" | smooth={smooth_label}"
        f" | dir={int(row['direction'])}"
    )
    axes[1].grid(alpha=0.2)

    axes[2].plot(
        bt_frame["timestamp"],
        bt_frame["equity_curve"],
        linewidth=1.1,
        color="tab:green",
    )
    axes[2].set_title(
        f"{title_prefix} equity"
        f" | Sharpe={float(row['sharpe']):.3f}"
        f" | Ret={float(row['return_pct']):.3f}"
    )
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    plt.show()
    return signal, mask


def build_decile_summary(
    signal: np.ndarray, forward_returns: np.ndarray
) -> pd.DataFrame:
    """Bucket the signal into deciles and report mean next-bar return per decile."""
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    signal_valid = signal[mask]
    returns_valid = forward_returns[mask]

    if signal_valid.size == 0:
        raise RuntimeError(
            "Best strategy produced no valid observations for decile analysis."
        )

    order = np.argsort(signal_valid)
    deciles = np.empty(signal_valid.shape[0], dtype=np.int64)
    deciles[order] = (
        np.arange(signal_valid.shape[0]) * 10 // signal_valid.shape[0]
    ) + 1

    return pd.DataFrame(
        [
            {
                "decile": decile,
                "count": int((decile_mask := deciles == decile).sum()),
                "mean_signal": float(np.nanmean(signal_valid[decile_mask])),
                "mean_fwd_ret": float(np.nanmean(returns_valid[decile_mask])),
            }
            for decile in range(1, 11)
        ]
    )
