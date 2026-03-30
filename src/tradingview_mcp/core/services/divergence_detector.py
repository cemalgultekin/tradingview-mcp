"""
Divergence Detector — RSI, MACD, OBV divergence from price.

Pure Python — no pandas, no numpy.

Detects bullish and bearish divergences:
  - Bullish divergence: price makes lower low but indicator makes higher low
  - Bearish divergence: price makes higher high but indicator makes lower high
  - Hidden bullish: price makes higher low but indicator makes lower low (trend continuation)
  - Hidden bearish: price makes lower high but indicator makes higher high

Supports RSI, MACD histogram, and OBV as divergence sources.
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.data_fetcher import fetch_ohlcv
from tradingview_mcp.core.services.indicators_calc import calc_rsi, calc_macd, calc_sma


# ─── OBV Calculation ─────────────────────────────────────────────────────────

def _calc_obv(closes: list[float], volumes: list[int]) -> list[float]:
    obv = [0.0] * len(closes)
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv


# ─── Swing Detection ─────────────────────────────────────────────────────────

def _find_swing_lows(data: list[Optional[float]], lookback: int = 5) -> list[tuple[int, float]]:
    """Find local minima with lookback window."""
    swings = []
    for i in range(lookback, len(data) - lookback):
        if data[i] is None:
            continue
        window = [data[j] for j in range(i - lookback, i + lookback + 1) if data[j] is not None]
        if window and data[i] == min(window):
            swings.append((i, data[i]))
    return swings


def _find_swing_highs(data: list[Optional[float]], lookback: int = 5) -> list[tuple[int, float]]:
    """Find local maxima with lookback window."""
    swings = []
    for i in range(lookback, len(data) - lookback):
        if data[i] is None:
            continue
        window = [data[j] for j in range(i - lookback, i + lookback + 1) if data[j] is not None]
        if window and data[i] == max(window):
            swings.append((i, data[i]))
    return swings


# ─── Divergence Matching ─────────────────────────────────────────────────────

def _find_divergences(
    candles: list[dict],
    indicator: list[Optional[float]],
    indicator_name: str,
    lookback: int = 5,
    max_distance: int = 50,
) -> list[dict]:
    """Compare price swings vs indicator swings to find divergences."""
    closes = [c["close"] for c in candles]

    price_lows = _find_swing_lows(closes, lookback)
    price_highs = _find_swing_highs(closes, lookback)
    ind_lows = _find_swing_lows(indicator, lookback)
    ind_highs = _find_swing_highs(indicator, lookback)

    divergences = []

    # Bullish divergence: price lower low + indicator higher low
    for i in range(1, len(price_lows)):
        pl_idx, pl_val = price_lows[i]
        pp_idx, pp_val = price_lows[i - 1]

        if pl_val >= pp_val:
            continue
        if pl_idx - pp_idx > max_distance:
            continue

        # Find indicator lows near these price lows
        il_near_curr = _nearest_swing(ind_lows, pl_idx, lookback)
        il_near_prev = _nearest_swing(ind_lows, pp_idx, lookback)

        if il_near_curr is None or il_near_prev is None:
            continue

        if il_near_curr[1] > il_near_prev[1]:
            divergences.append({
                "type": "bullish",
                "indicator": indicator_name,
                "start_bar": pp_idx,
                "end_bar": pl_idx,
                "start_date": candles[pp_idx]["date"],
                "end_date": candles[pl_idx]["date"],
                "price_prev": pp_val,
                "price_curr": pl_val,
                "indicator_prev": round(il_near_prev[1], 4),
                "indicator_curr": round(il_near_curr[1], 4),
                "strength": round(abs(il_near_curr[1] - il_near_prev[1]) / max(abs(il_near_prev[1]), 0.001), 4),
            })

    # Bearish divergence: price higher high + indicator lower high
    for i in range(1, len(price_highs)):
        ph_idx, ph_val = price_highs[i]
        pp_idx, pp_val = price_highs[i - 1]

        if ph_val <= pp_val:
            continue
        if ph_idx - pp_idx > max_distance:
            continue

        ih_near_curr = _nearest_swing(ind_highs, ph_idx, lookback)
        ih_near_prev = _nearest_swing(ind_highs, pp_idx, lookback)

        if ih_near_curr is None or ih_near_prev is None:
            continue

        if ih_near_curr[1] < ih_near_prev[1]:
            divergences.append({
                "type": "bearish",
                "indicator": indicator_name,
                "start_bar": pp_idx,
                "end_bar": ph_idx,
                "start_date": candles[pp_idx]["date"],
                "end_date": candles[ph_idx]["date"],
                "price_prev": pp_val,
                "price_curr": ph_val,
                "indicator_prev": round(ih_near_prev[1], 4),
                "indicator_curr": round(ih_near_curr[1], 4),
                "strength": round(abs(ih_near_curr[1] - ih_near_prev[1]) / max(abs(ih_near_prev[1]), 0.001), 4),
            })

    # Hidden bullish: price higher low + indicator lower low (trend continuation)
    for i in range(1, len(price_lows)):
        pl_idx, pl_val = price_lows[i]
        pp_idx, pp_val = price_lows[i - 1]

        if pl_val <= pp_val:
            continue
        if pl_idx - pp_idx > max_distance:
            continue

        il_near_curr = _nearest_swing(ind_lows, pl_idx, lookback)
        il_near_prev = _nearest_swing(ind_lows, pp_idx, lookback)

        if il_near_curr is None or il_near_prev is None:
            continue

        if il_near_curr[1] < il_near_prev[1]:
            divergences.append({
                "type": "hidden_bullish",
                "indicator": indicator_name,
                "start_bar": pp_idx,
                "end_bar": pl_idx,
                "start_date": candles[pp_idx]["date"],
                "end_date": candles[pl_idx]["date"],
                "price_prev": pp_val,
                "price_curr": pl_val,
                "indicator_prev": round(il_near_prev[1], 4),
                "indicator_curr": round(il_near_curr[1], 4),
                "strength": round(abs(il_near_curr[1] - il_near_prev[1]) / max(abs(il_near_prev[1]), 0.001), 4),
            })

    # Hidden bearish: price lower high + indicator higher high
    for i in range(1, len(price_highs)):
        ph_idx, ph_val = price_highs[i]
        pp_idx, pp_val = price_highs[i - 1]

        if ph_val >= pp_val:
            continue
        if ph_idx - pp_idx > max_distance:
            continue

        ih_near_curr = _nearest_swing(ind_highs, ph_idx, lookback)
        ih_near_prev = _nearest_swing(ind_highs, pp_idx, lookback)

        if ih_near_curr is None or ih_near_prev is None:
            continue

        if ih_near_curr[1] > ih_near_prev[1]:
            divergences.append({
                "type": "hidden_bearish",
                "indicator": indicator_name,
                "start_bar": pp_idx,
                "end_bar": ph_idx,
                "start_date": candles[pp_idx]["date"],
                "end_date": candles[ph_idx]["date"],
                "price_prev": pp_val,
                "price_curr": ph_val,
                "indicator_prev": round(ih_near_prev[1], 4),
                "indicator_curr": round(ih_near_curr[1], 4),
                "strength": round(abs(ih_near_curr[1] - ih_near_prev[1]) / max(abs(ih_near_prev[1]), 0.001), 4),
            })

    return divergences


def _nearest_swing(swings: list[tuple[int, float]], target_idx: int, tolerance: int) -> Optional[tuple[int, float]]:
    """Find the swing point nearest to target_idx within tolerance."""
    best = None
    best_dist = float("inf")
    for idx, val in swings:
        dist = abs(idx - target_idx)
        if dist <= tolerance and dist < best_dist:
            best = (idx, val)
            best_dist = dist
    return best


# ─── Public API ──────────────────────────────────────────────────────────────

def detect_divergences(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    lookback: int = 5,
    indicators: Optional[list[str]] = None,
) -> dict:
    """
    Detect RSI, MACD, and OBV divergences from price.

    Args:
        symbol:     Yahoo Finance symbol
        period:     Historical data period
        interval:   1d or 1h
        lookback:   Swing detection window size (default 5)
        indicators: List of indicators to check: 'rsi', 'macd', 'obv' (default: all)
    """
    if indicators is None:
        indicators = ["rsi", "macd", "obv"]

    try:
        candles = fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < 50:
        return {"error": f"Not enough data ({len(candles)} bars). Need at least 50."}

    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]

    all_divergences = []

    if "rsi" in indicators:
        rsi = calc_rsi(closes, 14)
        all_divergences.extend(_find_divergences(candles, rsi, "RSI", lookback))

    if "macd" in indicators:
        macd = calc_macd(closes)
        all_divergences.extend(_find_divergences(candles, macd["histogram"], "MACD", lookback))

    if "obv" in indicators:
        obv = _calc_obv(closes, volumes)
        all_divergences.extend(_find_divergences(candles, obv, "OBV", lookback))

    # Sort by recency
    all_divergences.sort(key=lambda d: d["end_bar"], reverse=True)

    # Summary counts
    bullish = [d for d in all_divergences if d["type"] == "bullish"]
    bearish = [d for d in all_divergences if d["type"] == "bearish"]
    hidden_bull = [d for d in all_divergences if d["type"] == "hidden_bullish"]
    hidden_bear = [d for d in all_divergences if d["type"] == "hidden_bearish"]

    # Recent divergences (last 20% of data)
    recent_cutoff = int(len(candles) * 0.8)
    recent = [d for d in all_divergences if d["end_bar"] >= recent_cutoff]

    # Active signal
    if recent:
        recent_types = [d["type"] for d in recent[:3]]
        if recent_types.count("bullish") + recent_types.count("hidden_bullish") > recent_types.count("bearish") + recent_types.count("hidden_bearish"):
            active_signal = "BULLISH"
        elif recent_types.count("bearish") + recent_types.count("hidden_bearish") > recent_types.count("bullish") + recent_types.count("hidden_bullish"):
            active_signal = "BEARISH"
        else:
            active_signal = "MIXED"
    else:
        active_signal = "NONE"

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "total_divergences": len(all_divergences),
        "summary": {
            "bullish": len(bullish),
            "bearish": len(bearish),
            "hidden_bullish": len(hidden_bull),
            "hidden_bearish": len(hidden_bear),
        },
        "active_signal": active_signal,
        "recent_divergences": recent[:10],
        "all_divergences": all_divergences[:25],
        "disclaimer": "Divergences are probabilistic signals, not guarantees. For educational use only.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
