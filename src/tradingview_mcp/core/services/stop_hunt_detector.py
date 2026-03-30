"""
Stop Hunt / Liquidity Trap Detector for tradingview-mcp

Pure Python — no pandas, no numpy.

Detects manipulation patterns where price spikes past support/resistance
levels to trigger stop-losses, then reverses immediately:
  1. Wick-to-body ratio analysis (long wicks = stop hunts)
  2. S/R level sweeps followed by reversals
  3. Volume confirmation on wick bars
  4. Cluster analysis of stop hunt zones
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.data_fetcher import fetch_ohlcv
from tradingview_mcp.core.services.indicators_calc import calc_atr, calc_sma


def _candle_metrics(c: dict) -> dict:
    """Compute wick and body metrics for a candle."""
    o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
    body = abs(cl - o)
    full_range = h - l
    upper_wick = h - max(o, cl)
    lower_wick = min(o, cl) - l
    is_bullish = cl >= o

    return {
        "body": body,
        "range": full_range,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "is_bullish": is_bullish,
        "wick_to_body": (upper_wick + lower_wick) / body if body > 0 else float("inf"),
        "upper_wick_ratio": upper_wick / full_range if full_range > 0 else 0,
        "lower_wick_ratio": lower_wick / full_range if full_range > 0 else 0,
    }


def detect_stop_hunts(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    wick_threshold: float = 2.0,
    min_wick_pct: float = 1.0,
) -> dict:
    """
    Detect stop hunt / liquidity trap patterns.

    Args:
        symbol:         Yahoo Finance symbol
        period:         Historical period
        interval:       1d or 1h
        wick_threshold: Minimum wick-to-body ratio to flag (default 2.0)
        min_wick_pct:   Minimum wick size as % of close to filter noise (default 1.0%)
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

    vol_sma = calc_sma([float(v) for v in volumes], 20)
    atr = calc_atr(highs, lows, closes, 14)

    # ── Find recent S/R levels (pivot points from lookback) ──────────────────
    sr_levels = []
    lookback = 10
    for i in range(lookback, len(candles) - lookback):
        # Swing high
        if all(highs[i] >= highs[j] for j in range(i - lookback, i + lookback + 1) if j != i):
            sr_levels.append({"type": "resistance", "price": highs[i], "bar": i, "date": candles[i]["date"]})
        # Swing low
        if all(lows[i] <= lows[j] for j in range(i - lookback, i + lookback + 1) if j != i):
            sr_levels.append({"type": "support", "price": lows[i], "bar": i, "date": candles[i]["date"]})

    # ── Detect stop hunt candles ─────────────────────────────────────────────
    stop_hunts = []

    for i in range(1, len(candles)):
        cm = _candle_metrics(candles[i])

        # Skip if wicks are too small (noise)
        if candles[i]["close"] == 0:
            continue
        upper_pct = cm["upper_wick"] / candles[i]["close"] * 100
        lower_pct = cm["lower_wick"] / candles[i]["close"] * 100

        # Volume context
        vol_ratio = volumes[i] / vol_sma[i] if vol_sma[i] and vol_sma[i] > 0 else 1.0

        # Check upper wick stop hunt (bearish trap — spikes above resistance, closes below)
        if cm["wick_to_body"] >= wick_threshold and upper_pct >= min_wick_pct:
            # Did the wick sweep a resistance level?
            swept_level = None
            for sr in sr_levels:
                if sr["type"] == "resistance" and sr["bar"] < i:
                    if candles[i]["high"] > sr["price"] > max(candles[i]["open"], candles[i]["close"]):
                        swept_level = sr
                        break

            stop_hunts.append({
                "bar": i,
                "date": candles[i]["date"],
                "direction": "upside_trap",
                "description": "Price spiked above resistance then reversed — bearish stop hunt",
                "wick_pct": round(upper_pct, 2),
                "wick_to_body": round(cm["wick_to_body"], 2),
                "volume_ratio": round(vol_ratio, 2),
                "high": candles[i]["high"],
                "close": candles[i]["close"],
                "swept_level": swept_level["price"] if swept_level else None,
                "confirmed": vol_ratio >= 1.5,
            })

        # Check lower wick stop hunt (bullish trap — spikes below support, closes above)
        if cm["wick_to_body"] >= wick_threshold and lower_pct >= min_wick_pct:
            swept_level = None
            for sr in sr_levels:
                if sr["type"] == "support" and sr["bar"] < i:
                    if candles[i]["low"] < sr["price"] < min(candles[i]["open"], candles[i]["close"]):
                        swept_level = sr
                        break

            stop_hunts.append({
                "bar": i,
                "date": candles[i]["date"],
                "direction": "downside_trap",
                "description": "Price spiked below support then reversed — bullish stop hunt",
                "wick_pct": round(lower_pct, 2),
                "wick_to_body": round(cm["wick_to_body"], 2),
                "volume_ratio": round(vol_ratio, 2),
                "low": candles[i]["low"],
                "close": candles[i]["close"],
                "swept_level": swept_level["price"] if swept_level else None,
                "confirmed": vol_ratio >= 1.5,
            })

    # ── Cluster analysis — zones with repeated stop hunts ────────────────────
    hunt_prices = []
    for sh in stop_hunts:
        if sh["direction"] == "upside_trap":
            hunt_prices.append(sh["high"])
        else:
            hunt_prices.append(sh["low"])

    clusters = []
    if hunt_prices and closes[-1] > 0:
        # Group hunts within 2% price bands
        hunt_prices.sort()
        current_cluster = [hunt_prices[0]]
        for p in hunt_prices[1:]:
            if (p - current_cluster[0]) / current_cluster[0] < 0.02:
                current_cluster.append(p)
            else:
                if len(current_cluster) >= 2:
                    clusters.append({
                        "zone_low": round(min(current_cluster), 4),
                        "zone_high": round(max(current_cluster), 4),
                        "hunt_count": len(current_cluster),
                    })
                current_cluster = [p]
        if len(current_cluster) >= 2:
            clusters.append({
                "zone_low": round(min(current_cluster), 4),
                "zone_high": round(max(current_cluster), 4),
                "hunt_count": len(current_cluster),
            })

    # Summary
    confirmed_hunts = [sh for sh in stop_hunts if sh["confirmed"]]
    upside = [sh for sh in stop_hunts if sh["direction"] == "upside_trap"]
    downside = [sh for sh in stop_hunts if sh["direction"] == "downside_trap"]

    # Frequency score
    if len(candles) > 0:
        hunt_frequency = round(len(stop_hunts) / len(candles) * 100, 2)
    else:
        hunt_frequency = 0

    if hunt_frequency > 15:
        risk = "HIGH"
        verdict = "Frequent stop hunts — highly manipulated, use wider stops or avoid"
    elif hunt_frequency > 8:
        risk = "MODERATE"
        verdict = "Regular stop hunt activity — widen stops beyond obvious S/R levels"
    elif hunt_frequency > 3:
        risk = "LOW"
        verdict = "Occasional stop hunts — normal market behavior"
    else:
        risk = "MINIMAL"
        verdict = "Rare stop hunt activity — clean price action"

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "total_stop_hunts": len(stop_hunts),
        "confirmed_stop_hunts": len(confirmed_hunts),
        "upside_traps": len(upside),
        "downside_traps": len(downside),
        "hunt_frequency_pct": hunt_frequency,
        "risk": risk,
        "verdict": verdict,
        "hunt_clusters": clusters[:5],
        "recent_hunts": stop_hunts[-10:],
        "support_resistance_levels": sr_levels[-10:],
        "disclaimer": "Stop hunt detection is heuristic-based. Not all long wicks are manipulation. For educational use only.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
