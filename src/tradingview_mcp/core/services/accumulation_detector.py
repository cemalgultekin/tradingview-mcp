"""
Accumulation / Distribution Detector for tradingview-mcp

Pure Python — no pandas, no numpy.

Detects smart money accumulation and distribution phases:
  1. OBV (On-Balance Volume) trend vs price trend divergence
  2. Accumulation/Distribution Line (Chaikin)
  3. Volume-weighted price analysis
  4. Quiet accumulation (price flat + rising OBV = smart money buying)
  5. Stealth distribution (price flat + falling OBV = smart money selling)
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.data_fetcher import fetch_ohlcv
from tradingview_mcp.core.services.indicators_calc import calc_sma


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


def _calc_ad_line(candles: list[dict]) -> list[float]:
    """Chaikin Accumulation/Distribution Line."""
    ad = [0.0] * len(candles)
    for i in range(len(candles)):
        h, l, c, v = candles[i]["high"], candles[i]["low"], candles[i]["close"], candles[i]["volume"]
        hl = h - l
        if hl > 0:
            mfm = ((c - l) - (h - c)) / hl  # Money Flow Multiplier
            mfv = mfm * v  # Money Flow Volume
        else:
            mfv = 0
        ad[i] = (ad[i - 1] if i > 0 else 0) + mfv
    return ad


def _linear_slope(values: list[float]) -> float:
    """Simple linear regression slope (normalized)."""
    n = len(values)
    if n < 3:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    slope = num / den
    # Normalize by mean to get % change per bar
    if y_mean != 0:
        return slope / abs(y_mean) * 100
    return 0.0


def detect_accumulation(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    analysis_window: int = 20,
) -> dict:
    """
    Detect accumulation and distribution phases.

    Args:
        symbol:          Yahoo Finance symbol
        period:          Historical period
        interval:        1d or 1h
        analysis_window: Window for trend comparison (default 20 bars)
    """
    try:
        candles = fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < analysis_window * 2:
        return {"error": f"Not enough data ({len(candles)} bars). Need at least {analysis_window * 2}."}

    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]

    obv = _calc_obv(closes, volumes)
    ad_line = _calc_ad_line(candles)

    # ── Rolling phase detection ──────────────────────────────────────────────
    phases = []
    for i in range(analysis_window, len(candles)):
        window_closes = closes[i - analysis_window:i + 1]
        window_obv = obv[i - analysis_window:i + 1]
        window_ad = ad_line[i - analysis_window:i + 1]
        window_vols = volumes[i - analysis_window:i + 1]

        price_slope = _linear_slope(window_closes)
        obv_slope = _linear_slope(window_obv)
        ad_slope = _linear_slope(window_ad)

        avg_vol = statistics.mean(window_vols) if window_vols else 0

        # Classify phase
        price_flat = abs(price_slope) < 0.2  # < 0.2% per bar
        price_up = price_slope > 0.2
        price_down = price_slope < -0.2

        obv_rising = obv_slope > 0.3
        obv_falling = obv_slope < -0.3

        if price_flat and obv_rising:
            phase = "ACCUMULATION"
            confidence = min(abs(obv_slope) / 1.0, 1.0)
        elif price_flat and obv_falling:
            phase = "DISTRIBUTION"
            confidence = min(abs(obv_slope) / 1.0, 1.0)
        elif price_up and obv_rising:
            phase = "MARKUP"
            confidence = 0.7
        elif price_down and obv_falling:
            phase = "MARKDOWN"
            confidence = 0.7
        elif price_up and obv_falling:
            phase = "DISTRIBUTION_DIVERGENCE"
            confidence = 0.8
        elif price_down and obv_rising:
            phase = "ACCUMULATION_DIVERGENCE"
            confidence = 0.8
        else:
            phase = "NEUTRAL"
            confidence = 0.3

        phases.append({
            "bar": i,
            "date": candles[i]["date"],
            "phase": phase,
            "confidence": round(confidence, 2),
            "price_slope": round(price_slope, 4),
            "obv_slope": round(obv_slope, 4),
            "ad_slope": round(ad_slope, 4),
        })

    # ── Current phase (last window) ──────────────────────────────────────────
    current = phases[-1] if phases else None

    # ── Phase duration ───────────────────────────────────────────────────────
    current_phase_duration = 0
    if phases:
        current_phase_name = phases[-1]["phase"]
        for p in reversed(phases):
            if p["phase"] == current_phase_name:
                current_phase_duration += 1
            else:
                break

    # ── Accumulation/Distribution zones ──────────────────────────────────────
    acc_zones = []
    dist_zones = []
    zone_start = None
    zone_phase = None

    for p in phases:
        is_acc = p["phase"] in ("ACCUMULATION", "ACCUMULATION_DIVERGENCE")
        is_dist = p["phase"] in ("DISTRIBUTION", "DISTRIBUTION_DIVERGENCE")

        if is_acc or is_dist:
            current_type = "accumulation" if is_acc else "distribution"
            if zone_phase == current_type:
                continue  # extend zone
            else:
                # Close previous zone
                if zone_start and zone_phase:
                    zone = {
                        "start_date": zone_start["date"],
                        "end_date": p["date"],
                        "duration_bars": p["bar"] - zone_start["bar"],
                    }
                    if zone_phase == "accumulation":
                        acc_zones.append(zone)
                    else:
                        dist_zones.append(zone)
                zone_start = p
                zone_phase = current_type
        else:
            if zone_start and zone_phase:
                zone = {
                    "start_date": zone_start["date"],
                    "end_date": p["date"],
                    "duration_bars": p["bar"] - zone_start["bar"],
                }
                if zone_phase == "accumulation":
                    acc_zones.append(zone)
                else:
                    dist_zones.append(zone)
                zone_start = None
                zone_phase = None

    # ── Smart money signal ───────────────────────────────────────────────────
    if current and current["phase"] in ("ACCUMULATION", "ACCUMULATION_DIVERGENCE"):
        smart_money_signal = "BUYING"
        action = "Smart money appears to be accumulating — potential upside ahead"
    elif current and current["phase"] in ("DISTRIBUTION", "DISTRIBUTION_DIVERGENCE"):
        smart_money_signal = "SELLING"
        action = "Smart money appears to be distributing — potential downside ahead"
    elif current and current["phase"] == "MARKUP":
        smart_money_signal = "CONFIRMED_UPTREND"
        action = "Volume confirms uptrend — trend likely to continue"
    elif current and current["phase"] == "MARKDOWN":
        smart_money_signal = "CONFIRMED_DOWNTREND"
        action = "Volume confirms downtrend — trend likely to continue"
    else:
        smart_money_signal = "NEUTRAL"
        action = "No clear accumulation or distribution signal"

    # Phase distribution summary
    phase_counts = {}
    for p in phases:
        phase_counts[p["phase"]] = phase_counts.get(p["phase"], 0) + 1

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "current_phase": current["phase"] if current else "UNKNOWN",
        "current_confidence": current["confidence"] if current else 0,
        "phase_duration_bars": current_phase_duration,
        "smart_money_signal": smart_money_signal,
        "action": action,
        "phase_distribution": phase_counts,
        "accumulation_zones": acc_zones[-5:],
        "distribution_zones": dist_zones[-5:],
        "recent_phases": [{"date": p["date"], "phase": p["phase"], "confidence": p["confidence"]}
                          for p in phases[-20:]],
        "disclaimer": "Accumulation/distribution analysis is probabilistic. For educational use only.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
