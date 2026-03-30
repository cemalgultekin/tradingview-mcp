"""
Dead Cat Bounce Detector for tradingview-mcp

Pure Python — no pandas, no numpy.

After a major crash, detects relief rallies that fail:
  1. Identifies crash events (>X% drop over N bars)
  2. Detects subsequent bounces
  3. Checks if bounce fails (lower highs, declining volume)
  4. Classifies bounce as dead cat vs genuine recovery
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.data_fetcher import fetch_ohlcv
from tradingview_mcp.core.services.indicators_calc import calc_rsi, calc_sma


def detect_dead_cat_bounce(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    crash_threshold: float = -20.0,
    crash_window: int = 10,
    bounce_min_pct: float = 5.0,
) -> dict:
    """
    Detect dead cat bounces — relief rallies after crashes that fail to sustain.

    Args:
        symbol:          Yahoo Finance symbol
        period:          Historical period
        interval:        1d or 1h
        crash_threshold: Minimum % drop over crash_window to qualify as crash (default -20%)
        crash_window:    Number of bars for crash measurement (default 10)
        bounce_min_pct:  Minimum % bounce from crash low to register (default 5%)
    """
    try:
        candles = fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < 50:
        return {"error": f"Not enough data ({len(candles)} bars). Need at least 50."}

    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]
    highs = [c["high"] for c in candles]

    rsi = calc_rsi(closes, 14)
    vol_sma = calc_sma([float(v) for v in volumes], 20)

    # ── Find crash events ────────────────────────────────────────────────────
    crashes = []
    i = crash_window
    while i < len(candles) - 5:
        start_price = closes[i - crash_window]
        if start_price == 0:
            i += 1
            continue

        # Find lowest point in window
        low_idx = i - crash_window
        low_price = closes[low_idx]
        for j in range(i - crash_window, i + 1):
            if closes[j] < low_price:
                low_price = closes[j]
                low_idx = j

        drop_pct = (low_price - start_price) / start_price * 100
        if drop_pct <= crash_threshold:
            crashes.append({
                "crash_start_bar": i - crash_window,
                "crash_low_bar": low_idx,
                "crash_start_date": candles[i - crash_window]["date"],
                "crash_low_date": candles[low_idx]["date"],
                "crash_start_price": start_price,
                "crash_low_price": low_price,
                "crash_pct": round(drop_pct, 2),
            })
            i = low_idx + 2  # Skip past this crash
        else:
            i += 1

    # ── Analyze bounces after each crash ─────────────────────────────────────
    bounce_events = []

    for crash in crashes:
        low_bar = crash["crash_low_bar"]
        low_price = crash["crash_low_price"]

        if low_bar >= len(candles) - 3:
            continue

        # Find the bounce peak after crash low
        search_end = min(low_bar + crash_window * 3, len(candles))
        peak_price = low_price
        peak_bar = low_bar

        for j in range(low_bar + 1, search_end):
            if closes[j] > peak_price:
                peak_price = closes[j]
                peak_bar = j

        bounce_pct = (peak_price - low_price) / low_price * 100 if low_price > 0 else 0

        if bounce_pct < bounce_min_pct:
            continue

        # Check what happened after the bounce peak
        post_peak_end = min(peak_bar + crash_window, len(candles))
        post_peak_prices = closes[peak_bar:post_peak_end]

        if len(post_peak_prices) < 3:
            continue

        post_peak_low = min(post_peak_prices)
        post_peak_drop = (post_peak_low - peak_price) / peak_price * 100 if peak_price > 0 else 0

        # Volume analysis: declining volume on bounce = weak
        bounce_volumes = volumes[low_bar:peak_bar + 1]
        post_volumes = volumes[peak_bar:post_peak_end]

        bounce_avg_vol = statistics.mean(bounce_volumes) if bounce_volumes else 0
        post_avg_vol = statistics.mean(post_volumes) if post_volumes else 0
        vol_declining = post_avg_vol < bounce_avg_vol * 0.7

        # RSI check at bounce peak
        peak_rsi = rsi[peak_bar] if peak_bar < len(rsi) and rsi[peak_bar] is not None else None

        # Recovery check: did price recover above pre-crash level?
        recovered = closes[min(post_peak_end - 1, len(closes) - 1)] >= crash["crash_start_price"] * 0.9

        # Retracement of the crash
        crash_range = crash["crash_start_price"] - low_price
        if crash_range > 0:
            fib_retracement = (peak_price - low_price) / crash_range
        else:
            fib_retracement = 0

        # Dead cat classification
        is_dead_cat = False
        reasons = []

        if post_peak_drop <= -bounce_pct * 0.5:
            is_dead_cat = True
            reasons.append(f"Price gave back {abs(post_peak_drop):.1f}% after bouncing {bounce_pct:.1f}%")

        if vol_declining:
            reasons.append("Volume declined during bounce — weak buying")
            if not is_dead_cat and post_peak_drop < -3:
                is_dead_cat = True

        if fib_retracement < 0.382:
            reasons.append(f"Bounce only retraced {fib_retracement:.1%} of crash (below 38.2% Fib)")
            is_dead_cat = True

        if peak_rsi is not None and peak_rsi < 45:
            reasons.append(f"RSI at bounce peak was only {peak_rsi:.1f} — weak momentum")

        if not recovered:
            reasons.append("Failed to recover pre-crash levels")

        if not reasons:
            reasons.append("Bounce showed healthy recovery characteristics")

        bounce_events.append({
            "crash_date": crash["crash_low_date"],
            "crash_pct": crash["crash_pct"],
            "bounce_peak_date": candles[peak_bar]["date"],
            "bounce_pct": round(bounce_pct, 2),
            "post_bounce_drop_pct": round(post_peak_drop, 2),
            "fib_retracement": round(fib_retracement, 4),
            "volume_declining": vol_declining,
            "peak_rsi": round(peak_rsi, 1) if peak_rsi else None,
            "is_dead_cat": is_dead_cat,
            "classification": "DEAD CAT BOUNCE" if is_dead_cat else "GENUINE RECOVERY",
            "reasons": reasons,
            "recovered": recovered,
        })

    # ── Current state assessment ─────────────────────────────────────────────
    # Check if we're currently in a potential dead cat bounce
    current_assessment = None
    if crashes:
        last_crash = crashes[-1]
        bars_since_crash = len(candles) - 1 - last_crash["crash_low_bar"]
        if bars_since_crash < crash_window * 3:
            bounce_from_low = (closes[-1] - last_crash["crash_low_price"]) / last_crash["crash_low_price"] * 100 if last_crash["crash_low_price"] > 0 else 0
            recovery_of_crash = (closes[-1] - last_crash["crash_low_price"]) / (last_crash["crash_start_price"] - last_crash["crash_low_price"]) if last_crash["crash_start_price"] != last_crash["crash_low_price"] else 0

            recent_vol_trend = "declining" if len(volumes) > 10 and statistics.mean(volumes[-5:]) < statistics.mean(volumes[-10:-5]) * 0.8 else "stable_or_rising"

            current_assessment = {
                "status": "IN POTENTIAL BOUNCE",
                "bars_since_crash": bars_since_crash,
                "bounce_from_low_pct": round(bounce_from_low, 2),
                "crash_recovery_ratio": round(recovery_of_crash, 4),
                "volume_trend": recent_vol_trend,
                "warning": "Monitor closely — bounce may fail" if recovery_of_crash < 0.5 else "Recovery looking constructive",
            }

    dead_cats = [b for b in bounce_events if b["is_dead_cat"]]
    recoveries = [b for b in bounce_events if not b["is_dead_cat"]]

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "crashes_found": len(crashes),
        "bounces_analyzed": len(bounce_events),
        "dead_cat_bounces": len(dead_cats),
        "genuine_recoveries": len(recoveries),
        "current_assessment": current_assessment,
        "bounce_events": bounce_events[:10],
        "disclaimer": "Dead cat bounce detection is retrospective. Current bounces cannot be classified with certainty until resolved. For educational use only.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
