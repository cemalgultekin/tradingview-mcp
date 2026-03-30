"""
Correlation / Beta Detector for tradingview-mcp

Pure Python — no pandas, no numpy.

Analyzes:
  1. Pearson correlation between a symbol and benchmark(s)
  2. Rolling correlation to detect regime changes
  3. Beta (sensitivity to benchmark moves)
  4. Alpha (excess return above benchmark-explained return)
  5. Decoupling events (sudden correlation breakdown)
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.data_fetcher import fetch_ohlcv


def _returns(closes: list[float]) -> list[float]:
    """Compute percentage returns from close prices."""
    return [(closes[i] - closes[i - 1]) / closes[i - 1] * 100
            for i in range(1, len(closes)) if closes[i - 1] != 0]


def _pearson(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    mx, my = sum(x[:n]) / n, sum(y[:n]) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    dx = math.sqrt(sum((x[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((y[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return round(num / (dx * dy), 4)


def _beta_alpha(asset_returns: list[float], bench_returns: list[float]) -> tuple[float, float]:
    """Compute beta and annualized alpha via OLS regression."""
    n = min(len(asset_returns), len(bench_returns))
    if n < 5:
        return 0.0, 0.0
    ar, br = asset_returns[:n], bench_returns[:n]
    mb = sum(br) / n
    ma = sum(ar) / n
    cov = sum((ar[i] - ma) * (br[i] - mb) for i in range(n)) / n
    var_b = sum((br[i] - mb) ** 2 for i in range(n)) / n
    if var_b == 0:
        return 0.0, round(ma * 252, 2)
    beta = round(cov / var_b, 4)
    alpha = round((ma - beta * mb) * 252, 4)  # annualized
    return beta, alpha


def _rolling_correlation(x: list[float], y: list[float], window: int = 30) -> list[Optional[float]]:
    n = min(len(x), len(y))
    result = [None] * n
    for i in range(window - 1, n):
        wx = x[i - window + 1:i + 1]
        wy = y[i - window + 1:i + 1]
        result[i] = _pearson(wx, wy)
    return result


def detect_correlation(
    symbol: str,
    benchmarks: Optional[list[str]] = None,
    period: str = "1y",
    interval: str = "1d",
    rolling_window: int = 30,
) -> dict:
    """
    Analyze correlation, beta, and alpha of a symbol vs benchmark(s).

    Args:
        symbol:         Yahoo Finance symbol to analyze
        benchmarks:     List of benchmark symbols (default: BTC-USD, ETH-USD, SPY)
        period:         Historical period
        interval:       1d or 1h
        rolling_window: Window size for rolling correlation (default 30)
    """
    if benchmarks is None:
        benchmarks = ["BTC-USD", "ETH-USD", "SPY"]

    try:
        candles = fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < rolling_window + 10:
        return {"error": f"Not enough data ({len(candles)} bars)."}

    asset_closes = [c["close"] for c in candles]
    asset_ret = _returns(asset_closes)

    bench_results = []

    for bench_sym in benchmarks:
        try:
            bench_candles = fetch_ohlcv(bench_sym, period, interval)
        except Exception:
            bench_results.append({
                "benchmark": bench_sym,
                "error": "Failed to fetch benchmark data",
            })
            continue

        bench_closes = [c["close"] for c in bench_candles]
        bench_ret = _returns(bench_closes)

        # Align lengths
        min_len = min(len(asset_ret), len(bench_ret))
        ar = asset_ret[-min_len:]
        br = bench_ret[-min_len:]

        corr = _pearson(ar, br)
        beta, alpha = _beta_alpha(ar, br)

        # Rolling correlation
        rolling = _rolling_correlation(ar, br, rolling_window)
        valid_rolling = [r for r in rolling if r is not None]

        # Detect decoupling events (rolling corr drops below 0 when avg > 0.5)
        decoupling_events = []
        if valid_rolling and statistics.mean(valid_rolling) > 0.3:
            for i, r in enumerate(rolling):
                if r is not None and r < 0.0:
                    idx_in_candles = len(candles) - min_len + i
                    if 0 <= idx_in_candles < len(candles):
                        decoupling_events.append({
                            "date": candles[idx_in_candles]["date"],
                            "rolling_correlation": r,
                        })

        # Current rolling correlation
        current_rolling = valid_rolling[-1] if valid_rolling else None

        # Correlation trend (recent vs early)
        if len(valid_rolling) > 10:
            seg = len(valid_rolling) // 3
            early_corr = statistics.mean(valid_rolling[:seg])
            late_corr = statistics.mean(valid_rolling[-seg:])
            corr_trend = round(late_corr - early_corr, 4)
        else:
            corr_trend = 0.0

        # Classification
        if abs(corr) >= 0.8:
            relationship = "HIGHLY CORRELATED" if corr > 0 else "HIGHLY INVERSE"
        elif abs(corr) >= 0.5:
            relationship = "MODERATELY CORRELATED" if corr > 0 else "MODERATELY INVERSE"
        elif abs(corr) >= 0.2:
            relationship = "WEAKLY CORRELATED" if corr > 0 else "WEAKLY INVERSE"
        else:
            relationship = "UNCORRELATED"

        bench_results.append({
            "benchmark": bench_sym,
            "correlation": corr,
            "beta": beta,
            "alpha_annualized": alpha,
            "relationship": relationship,
            "current_rolling_correlation": current_rolling,
            "correlation_trend": corr_trend,
            "decoupling_events": decoupling_events[:5],
            "rolling_stats": {
                "mean": round(statistics.mean(valid_rolling), 4) if valid_rolling else None,
                "min": round(min(valid_rolling), 4) if valid_rolling else None,
                "max": round(max(valid_rolling), 4) if valid_rolling else None,
                "std": round(statistics.stdev(valid_rolling), 4) if len(valid_rolling) > 1 else None,
            },
        })

    # Overall independence score
    correlations = [b["correlation"] for b in bench_results if "correlation" in b]
    avg_abs_corr = statistics.mean(abs(c) for c in correlations) if correlations else 0
    independence_score = round(1.0 - avg_abs_corr, 2)

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "independence_score": independence_score,
        "benchmark_analysis": bench_results,
        "disclaimer": "Correlation is not causation. Past correlations may not persist. For educational use only.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
