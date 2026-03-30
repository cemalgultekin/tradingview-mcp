"""
Seasonality Detector for tradingview-mcp

Pure Python — no pandas, no numpy.

Analyzes historical patterns to find statistically significant seasonal effects:
  1. Day-of-week patterns (Monday effect, weekend rally, etc.)
  2. Month-of-year patterns (January effect, sell-in-May, etc.)
  3. Intra-month patterns (beginning vs end of month)
  4. Statistical significance testing via bootstrap
"""
from __future__ import annotations

import math
import random
import statistics
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.data_fetcher import fetch_ohlcv


_DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string from candle data."""
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _bootstrap_significance(values: list[float], n_bootstrap: int = 1000) -> float:
    """
    Test if the mean of values is significantly different from zero.
    Returns p-value estimate via bootstrap.
    """
    if len(values) < 5:
        return 1.0
    observed_mean = statistics.mean(values)
    if observed_mean == 0:
        return 1.0

    # Center the data
    centered = [v - observed_mean for v in values]

    count_extreme = 0
    for _ in range(n_bootstrap):
        sample = random.choices(centered, k=len(values))
        sample_mean = sum(sample) / len(sample)
        if abs(sample_mean) >= abs(observed_mean):
            count_extreme += 1

    return round(count_extreme / n_bootstrap, 4)


def detect_seasonality(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    significance_threshold: float = 0.10,
) -> dict:
    """
    Detect day-of-week and month-of-year seasonal patterns.

    Args:
        symbol:                  Yahoo Finance symbol
        period:                  Historical period (recommend '2y' for monthly patterns)
        interval:                Must be '1d' for meaningful seasonal analysis
        significance_threshold:  P-value threshold for flagging patterns (default 0.10)
    """
    try:
        candles = fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < 60:
        return {"error": f"Not enough data ({len(candles)} bars). Need at least 60 trading days."}

    # Compute daily returns with dates
    returns_by_dow = {i: [] for i in range(7)}
    returns_by_month = {i: [] for i in range(1, 13)}
    returns_by_month_half = {"first_half": [], "second_half": []}

    all_returns = []

    for i in range(1, len(candles)):
        if candles[i - 1]["close"] == 0:
            continue
        ret = (candles[i]["close"] - candles[i - 1]["close"]) / candles[i - 1]["close"] * 100
        dt = _parse_date(candles[i]["date"])
        if dt is None:
            continue

        all_returns.append(ret)
        returns_by_dow[dt.weekday()].append(ret)
        returns_by_month[dt.month].append(ret)
        if dt.day <= 15:
            returns_by_month_half["first_half"].append(ret)
        else:
            returns_by_month_half["second_half"].append(ret)

    overall_avg = statistics.mean(all_returns) if all_returns else 0

    # ── Day-of-Week Analysis ─────────────────────────────────────────────────
    dow_results = []
    for dow in range(7):
        rets = returns_by_dow[dow]
        if len(rets) < 5:
            continue
        avg = statistics.mean(rets)
        std = statistics.stdev(rets) if len(rets) > 1 else 0
        win_rate = sum(1 for r in rets if r > 0) / len(rets) * 100
        p_value = _bootstrap_significance(rets)
        significant = p_value < significance_threshold

        dow_results.append({
            "day": _DOW_NAMES[dow],
            "day_index": dow,
            "avg_return_pct": round(avg, 4),
            "std_pct": round(std, 4),
            "win_rate_pct": round(win_rate, 1),
            "sample_size": len(rets),
            "p_value": p_value,
            "significant": significant,
            "edge": round(avg - overall_avg, 4),
        })

    # ── Month-of-Year Analysis ───────────────────────────────────────────────
    month_results = []
    for month in range(1, 13):
        rets = returns_by_month[month]
        if len(rets) < 3:
            continue
        avg = statistics.mean(rets)
        std = statistics.stdev(rets) if len(rets) > 1 else 0
        win_rate = sum(1 for r in rets if r > 0) / len(rets) * 100
        p_value = _bootstrap_significance(rets)
        significant = p_value < significance_threshold

        month_results.append({
            "month": _MONTH_NAMES[month - 1],
            "month_index": month,
            "avg_return_pct": round(avg, 4),
            "std_pct": round(std, 4),
            "win_rate_pct": round(win_rate, 1),
            "sample_size": len(rets),
            "p_value": p_value,
            "significant": significant,
            "edge": round(avg - overall_avg, 4),
        })

    # ── First Half vs Second Half of Month ───────────────────────────────────
    fh = returns_by_month_half["first_half"]
    sh = returns_by_month_half["second_half"]
    month_half_analysis = None
    if len(fh) >= 10 and len(sh) >= 10:
        fh_avg = statistics.mean(fh)
        sh_avg = statistics.mean(sh)
        month_half_analysis = {
            "first_half_avg_return_pct": round(fh_avg, 4),
            "second_half_avg_return_pct": round(sh_avg, 4),
            "first_half_win_rate_pct": round(sum(1 for r in fh if r > 0) / len(fh) * 100, 1),
            "second_half_win_rate_pct": round(sum(1 for r in sh if r > 0) / len(sh) * 100, 1),
            "stronger_half": "first" if fh_avg > sh_avg else "second",
        }

    # ── Best / Worst patterns ────────────────────────────────────────────────
    significant_dow = [d for d in dow_results if d["significant"]]
    significant_months = [m for m in month_results if m["significant"]]

    best_day = max(dow_results, key=lambda x: x["avg_return_pct"]) if dow_results else None
    worst_day = min(dow_results, key=lambda x: x["avg_return_pct"]) if dow_results else None
    best_month = max(month_results, key=lambda x: x["avg_return_pct"]) if month_results else None
    worst_month = min(month_results, key=lambda x: x["avg_return_pct"]) if month_results else None

    # ── Actionable patterns ──────────────────────────────────────────────────
    patterns = []
    for d in significant_dow:
        direction = "bullish" if d["avg_return_pct"] > 0 else "bearish"
        patterns.append(f"{d['day']}s tend to be {direction} ({d['avg_return_pct']:+.3f}% avg, p={d['p_value']:.3f})")
    for m in significant_months:
        direction = "bullish" if m["avg_return_pct"] > 0 else "bearish"
        patterns.append(f"{m['month']} tends to be {direction} ({m['avg_return_pct']:+.3f}% avg, p={m['p_value']:.3f})")

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "trading_days_analyzed": len(all_returns),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "overall_avg_daily_return_pct": round(overall_avg, 4),
        "day_of_week": dow_results,
        "month_of_year": month_results,
        "month_half": month_half_analysis,
        "best_day": best_day["day"] if best_day else None,
        "worst_day": worst_day["day"] if worst_day else None,
        "best_month": best_month["month"] if best_month else None,
        "worst_month": worst_month["month"] if worst_month else None,
        "significant_patterns": patterns,
        "significance_threshold": significance_threshold,
        "disclaimer": "Seasonality patterns are historical and may not persist. Sample sizes for monthly "
                      "patterns are small — treat with caution. For educational use only.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
