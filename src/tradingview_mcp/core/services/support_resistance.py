"""
Support / Resistance Mapping for tradingview-mcp

Pure Python — no pandas, no numpy.

Identifies key S/R levels by:
  1. Multi-touch pivot detection (price bounces from same level multiple times)
  2. Zone clustering (groups nearby levels into zones, not single price points)
  3. Strength scoring (more touches + volume = stronger level)
  4. Breakout/breakdown validation (did the level hold or fail?)
  5. Current price position relative to nearest S/R zones
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.data_fetcher import fetch_ohlcv
from tradingview_mcp.core.services.indicators_calc import calc_atr, calc_sma


def _find_pivot_points(candles: list[dict], lookback: int = 3) -> list[dict]:
    """Find all pivot highs and lows with given lookback."""
    pivots = []
    for i in range(lookback, len(candles) - lookback):
        # Pivot high
        is_ph = all(candles[i]["high"] >= candles[j]["high"]
                     for j in range(i - lookback, i + lookback + 1) if j != i)
        if is_ph:
            pivots.append({
                "bar": i,
                "date": candles[i]["date"],
                "price": candles[i]["high"],
                "type": "resistance",
                "volume": candles[i]["volume"],
            })

        # Pivot low
        is_pl = all(candles[i]["low"] <= candles[j]["low"]
                     for j in range(i - lookback, i + lookback + 1) if j != i)
        if is_pl:
            pivots.append({
                "bar": i,
                "date": candles[i]["date"],
                "price": candles[i]["low"],
                "type": "support",
                "volume": candles[i]["volume"],
            })

    return pivots


def _cluster_levels(pivots: list[dict], tolerance_pct: float = 1.0) -> list[dict]:
    """Group pivot points into S/R zones based on price proximity."""
    if not pivots:
        return []

    sorted_pivots = sorted(pivots, key=lambda p: p["price"])
    clusters = []
    current_cluster = [sorted_pivots[0]]

    for p in sorted_pivots[1:]:
        cluster_avg = statistics.mean(pp["price"] for pp in current_cluster)
        if abs(p["price"] - cluster_avg) / cluster_avg * 100 <= tolerance_pct:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]
    clusters.append(current_cluster)

    return clusters


def _score_zone(cluster: list[dict], candles: list[dict], avg_volume: float) -> dict:
    """Score an S/R zone based on touches, volume, recency, and type."""
    prices = [p["price"] for p in cluster]
    zone_low = min(prices)
    zone_high = max(prices)
    zone_mid = statistics.mean(prices)
    touches = len(cluster)

    # Volume score: average volume at touch points vs overall average
    touch_volumes = [p["volume"] for p in cluster if p["volume"] > 0]
    avg_touch_vol = statistics.mean(touch_volumes) if touch_volumes else 0
    volume_score = min(avg_touch_vol / avg_volume, 3.0) if avg_volume > 0 else 1.0

    # Recency score: more recent touches score higher
    max_bar = max(p["bar"] for p in cluster)
    total_bars = max(1, candles[-1]["bar"] if "bar" in candles[-1] else len(candles) - 1)
    # Use the index of the last candle
    last_idx = len(candles) - 1
    recency_score = 1.0 + (max_bar / last_idx) if last_idx > 0 else 1.0

    # Type classification
    support_count = sum(1 for p in cluster if p["type"] == "support")
    resistance_count = sum(1 for p in cluster if p["type"] == "resistance")
    if support_count > resistance_count:
        zone_type = "support"
    elif resistance_count > support_count:
        zone_type = "resistance"
    else:
        zone_type = "support_resistance"  # Flip zone

    # Composite strength score (0-10)
    strength = min(10, round(
        touches * 1.5 +          # More touches = stronger
        volume_score * 1.5 +     # Higher volume = stronger
        recency_score * 1.0,     # More recent = more relevant
    1))

    # Dates of touches
    touch_dates = sorted(set(p["date"] for p in cluster))

    return {
        "zone_low": round(zone_low, 4),
        "zone_high": round(zone_high, 4),
        "zone_mid": round(zone_mid, 4),
        "zone_type": zone_type,
        "touches": touches,
        "strength": strength,
        "volume_score": round(volume_score, 2),
        "recency_score": round(recency_score, 2),
        "first_touch": touch_dates[0],
        "last_touch": touch_dates[-1],
        "touch_dates": touch_dates[-5:],
    }


def _check_breakouts(zones: list[dict], candles: list[dict], lookback_bars: int = 5) -> list[dict]:
    """Check recent candles for breakouts/breakdowns of S/R zones."""
    breakouts = []
    recent = candles[-lookback_bars:]

    for zone in zones:
        for c in recent:
            # Breakout above resistance
            if zone["zone_type"] in ("resistance", "support_resistance"):
                if c["close"] > zone["zone_high"] and c["open"] <= zone["zone_high"]:
                    breakouts.append({
                        "date": c["date"],
                        "type": "breakout",
                        "direction": "up",
                        "zone_mid": zone["zone_mid"],
                        "zone_strength": zone["strength"],
                        "close": c["close"],
                        "volume": c["volume"],
                    })

            # Breakdown below support
            if zone["zone_type"] in ("support", "support_resistance"):
                if c["close"] < zone["zone_low"] and c["open"] >= zone["zone_low"]:
                    breakouts.append({
                        "date": c["date"],
                        "type": "breakdown",
                        "direction": "down",
                        "zone_mid": zone["zone_mid"],
                        "zone_strength": zone["strength"],
                        "close": c["close"],
                        "volume": c["volume"],
                    })

    return breakouts


def detect_support_resistance(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    lookback: int = 3,
    zone_tolerance_pct: float = 1.5,
    min_touches: int = 2,
) -> dict:
    """
    Map key support and resistance zones with strength scoring.

    Args:
        symbol:             Yahoo Finance symbol
        period:             Historical period
        interval:           1d or 1h
        lookback:           Pivot detection sensitivity (default 3)
        zone_tolerance_pct: Max % difference to group pivots into a zone (default 1.5%)
        min_touches:        Minimum touches for a zone to be reported (default 2)
    """
    try:
        candles = fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < 30:
        return {"error": f"Not enough data ({len(candles)} bars). Need at least 30."}

    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]
    current_price = closes[-1]

    avg_volume = statistics.mean(v for v in volumes if v > 0) if any(v > 0 for v in volumes) else 0

    # Find pivots
    pivots = _find_pivot_points(candles, lookback)

    if not pivots:
        return {
            "symbol": symbol.upper(),
            "error": "No pivot points found. Try a different lookback or longer period.",
        }

    # Cluster into zones
    clusters = _cluster_levels(pivots, zone_tolerance_pct)

    # Score each zone
    zones = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        zone = _score_zone(cluster, candles, avg_volume)
        zones.append(zone)

    # Sort by strength
    zones.sort(key=lambda z: z["strength"], reverse=True)

    # Find nearest support and resistance to current price
    support_zones = [z for z in zones if z["zone_mid"] < current_price]
    resistance_zones = [z for z in zones if z["zone_mid"] > current_price]

    nearest_support = max(support_zones, key=lambda z: z["zone_mid"]) if support_zones else None
    nearest_resistance = min(resistance_zones, key=lambda z: z["zone_mid"]) if resistance_zones else None

    # Distance to nearest levels
    dist_to_support = None
    dist_to_resistance = None
    if nearest_support:
        dist_to_support = round((current_price - nearest_support["zone_mid"]) / current_price * 100, 2)
    if nearest_resistance:
        dist_to_resistance = round((nearest_resistance["zone_mid"] - current_price) / current_price * 100, 2)

    # Check for recent breakouts
    breakouts = _check_breakouts(zones, candles)

    # ATR for context
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    atr_raw = calc_atr(highs, lows, closes, 14)
    current_atr = next((a for a in reversed(atr_raw) if a is not None), 0)
    atr_pct = round(current_atr / current_price * 100, 2) if current_price > 0 else 0

    # Position assessment
    if nearest_support and nearest_resistance:
        total_range = nearest_resistance["zone_mid"] - nearest_support["zone_mid"]
        if total_range > 0:
            position_pct = (current_price - nearest_support["zone_mid"]) / total_range
        else:
            position_pct = 0.5
    else:
        position_pct = 0.5

    if position_pct > 0.8:
        position_desc = "Near resistance — potential reversal zone"
    elif position_pct < 0.2:
        position_desc = "Near support — potential bounce zone"
    else:
        position_desc = "Mid-range — between key levels"

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "current_price": current_price,
        "atr_pct": atr_pct,
        "pivot_points_found": len(pivots),
        "zones_identified": len(zones),
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "distance_to_support_pct": dist_to_support,
        "distance_to_resistance_pct": dist_to_resistance,
        "position_in_range": round(position_pct, 4),
        "position_assessment": position_desc,
        "recent_breakouts": breakouts,
        "key_zones": zones[:15],
        "support_zones": [z for z in zones if z["zone_type"] in ("support", "support_resistance")][:8],
        "resistance_zones": [z for z in zones if z["zone_type"] in ("resistance", "support_resistance")][:8],
        "disclaimer": "S/R levels are historical and may not hold. For educational use only.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
