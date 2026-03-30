"""
Wash Trading / Fake Volume Detector for tradingview-mcp

Pure Python — no pandas, no numpy.

Detects artificial volume by analyzing:
  1. Volume-price correlation — real markets show correlation; wash trading doesn't move price
  2. Volume clustering — repeated identical or near-identical volume bars
  3. Volume-volatility mismatch — high volume with low ATR is suspicious
  4. Round-number volume — bots often trade in exact round lots
  5. Volume distribution — natural volume follows log-normal; fake follows uniform
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.data_fetcher import fetch_ohlcv
from tradingview_mcp.core.services.indicators_calc import calc_atr


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient between two lists."""
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    mx = sum(x[:n]) / n
    my = sum(y[:n]) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    dx = math.sqrt(sum((x[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((y[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return round(num / (dx * dy), 4)


def detect_wash_trading(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
) -> dict:
    """
    Detect potential wash trading / fake volume in a symbol.

    Returns a wash trading probability score (0-100) with detailed evidence.
    """
    try:
        candles = fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < 30:
        return {"error": f"Not enough data ({len(candles)} bars). Need at least 30."}

    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    valid_volumes = [v for v in volumes if v > 0]
    if len(valid_volumes) < 10:
        return {"error": "Insufficient volume data."}

    score = 0
    evidence = []

    # ── 1. Volume-Price Correlation ──────────────────────────────────────────
    # In real markets, large volume bars tend to have larger price moves
    abs_returns = []
    vol_for_corr = []
    for i in range(1, len(candles)):
        if closes[i - 1] == 0 or volumes[i] == 0:
            continue
        abs_returns.append(abs(closes[i] - closes[i - 1]) / closes[i - 1] * 100)
        vol_for_corr.append(float(volumes[i]))

    vol_price_corr = _pearson_correlation(vol_for_corr, abs_returns)

    if vol_price_corr < 0.05:
        score += 30
        evidence.append(f"Volume-price correlation extremely low ({vol_price_corr}) — volume doesn't move price")
    elif vol_price_corr < 0.15:
        score += 15
        evidence.append(f"Volume-price correlation weak ({vol_price_corr})")

    # ── 2. Volume Clustering (repeated identical volumes) ────────────────────
    vol_str = [str(v) for v in valid_volumes]
    unique_ratio = len(set(vol_str)) / len(vol_str) if vol_str else 1.0
    if unique_ratio < 0.5:
        score += 20
        evidence.append(f"Only {round(unique_ratio * 100, 1)}% unique volume values — possible bot activity")
    elif unique_ratio < 0.7:
        score += 10
        evidence.append(f"Low volume uniqueness ({round(unique_ratio * 100, 1)}%)")

    # Near-identical volumes (within 0.1%)
    near_dupes = 0
    for i in range(1, len(valid_volumes)):
        if valid_volumes[i - 1] > 0:
            diff = abs(valid_volumes[i] - valid_volumes[i - 1]) / valid_volumes[i - 1]
            if diff < 0.001:
                near_dupes += 1
    near_dupe_ratio = near_dupes / max(len(valid_volumes) - 1, 1)
    if near_dupe_ratio > 0.1:
        score += 15
        evidence.append(f"{round(near_dupe_ratio * 100, 1)}% consecutive bars have near-identical volume")

    # ── 3. Volume-Volatility Mismatch ───────────────────────────────────────
    atr = calc_atr(highs, lows, closes, 14)
    valid_atr = [a for a in atr if a is not None and a > 0]
    if valid_atr and valid_volumes:
        avg_atr_pct = statistics.mean(valid_atr) / statistics.mean(closes) * 100 if statistics.mean(closes) > 0 else 0
        avg_vol = statistics.mean(valid_volumes)
        # High volume but very low volatility is suspicious
        vol_rank = sorted(valid_volumes)
        vol_75th = vol_rank[int(len(vol_rank) * 0.75)]

        if avg_vol > vol_75th * 0.8 and avg_atr_pct < 0.5:
            score += 15
            evidence.append(f"Consistently high volume ({avg_vol:.0f}) with very low volatility (ATR {avg_atr_pct:.2f}%)")

    # ── 4. Round-Number Volume Ratio ────────────────────────────────────────
    round_count = sum(1 for v in valid_volumes if v > 0 and v % 100 == 0)
    round_ratio = round_count / len(valid_volumes)
    if round_ratio > 0.3:
        score += 15
        evidence.append(f"{round(round_ratio * 100, 1)}% of bars have perfectly round volume — bot signature")
    elif round_ratio > 0.15:
        score += 8
        evidence.append(f"Elevated round-number volume ({round(round_ratio * 100, 1)}%)")

    # ── 5. Volume Distribution Analysis ─────────────────────────────────────
    # Natural volume ~ log-normal; compute coefficient of variation of log-volumes
    log_vols = [math.log(v) for v in valid_volumes if v > 0]
    if len(log_vols) > 5:
        log_cv = statistics.stdev(log_vols) / abs(statistics.mean(log_vols)) if statistics.mean(log_vols) != 0 else 0
        if log_cv < 0.1:
            score += 15
            evidence.append(f"Volume distribution unnaturally uniform (log CV = {round(log_cv, 3)})")
        elif log_cv < 0.2:
            score += 8
            evidence.append(f"Volume distribution suspiciously tight (log CV = {round(log_cv, 3)})")

    score = min(score, 100)

    if score >= 60:
        verdict = "HIGH PROBABILITY — multiple wash trading indicators detected"
        risk = "CRITICAL"
    elif score >= 40:
        verdict = "MODERATE PROBABILITY — some artificial volume patterns found"
        risk = "HIGH"
    elif score >= 20:
        verdict = "LOW PROBABILITY — minor anomalies present but likely natural"
        risk = "MODERATE"
    else:
        verdict = "CLEAN — volume patterns appear natural"
        risk = "LOW"

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "wash_trade_score": score,
        "risk": risk,
        "verdict": verdict,
        "evidence": evidence,
        "metrics": {
            "volume_price_correlation": vol_price_corr,
            "volume_uniqueness_pct": round(unique_ratio * 100, 1),
            "consecutive_near_dupes_pct": round(near_dupe_ratio * 100, 1),
            "round_number_volume_pct": round(round_ratio * 100, 1),
            "avg_daily_volume": round(statistics.mean(valid_volumes), 0),
        },
        "disclaimer": "Heuristic analysis only. Low-liquidity assets may trigger false positives. Not financial advice.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
