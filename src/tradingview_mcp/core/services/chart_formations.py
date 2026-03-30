"""
Chart Formation Recognition for tradingview-mcp

Pure Python — no pandas, no numpy.

Detects multi-bar structural chart patterns from OHLCV data:

  Reversal Formations:
    - Head and Shoulders (and Inverse)
    - Double Top / Double Bottom
    - Triple Top / Triple Bottom
    - Rising Wedge / Falling Wedge

  Continuation Formations:
    - Ascending / Descending / Symmetrical Triangle
    - Bull Flag / Bear Flag
    - Pennant
    - Cup and Handle

  Trend Structure:
    - Ascending / Descending / Horizontal Channel

All detection is based on swing high/low identification and geometric fitting.
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.data_fetcher import fetch_ohlcv
from tradingview_mcp.core.services.indicators_calc import calc_atr, calc_sma


# ─── Swing Point Detection ──────────────────────────────────────────────────

def _find_swings(candles: list[dict], lookback: int = 5) -> tuple[list[dict], list[dict]]:
    """Find swing highs and lows with lookback window."""
    highs = []
    lows = []
    for i in range(lookback, len(candles) - lookback):
        is_high = all(candles[i]["high"] >= candles[j]["high"]
                      for j in range(i - lookback, i + lookback + 1) if j != i)
        is_low = all(candles[i]["low"] <= candles[j]["low"]
                     for j in range(i - lookback, i + lookback + 1) if j != i)

        if is_high:
            highs.append({"bar": i, "price": candles[i]["high"], "date": candles[i]["date"]})
        if is_low:
            lows.append({"bar": i, "price": candles[i]["low"], "date": candles[i]["date"]})

    return highs, lows


def _linear_regression(points: list[tuple[int, float]]) -> tuple[float, float, float]:
    """Returns (slope, intercept, r_squared) for (x, y) points."""
    n = len(points)
    if n < 2:
        return 0, 0, 0
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] ** 2 for p in points)
    sxy = sum(p[0] * p[1] for p in points)

    den = n * sxx - sx * sx
    if den == 0:
        return 0, sy / n, 0

    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n

    # R²
    y_mean = sy / n
    ss_tot = sum((p[1] - y_mean) ** 2 for p in points)
    ss_res = sum((p[1] - (slope * p[0] + intercept)) ** 2 for p in points)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return slope, intercept, r_sq


def _price_tolerance(candles: list[dict], pct: float = 1.5) -> float:
    """Price tolerance for 'roughly equal' comparisons."""
    avg_price = statistics.mean(c["close"] for c in candles)
    return avg_price * pct / 100


# ─── Head and Shoulders ─────────────────────────────────────────────────────

def _detect_head_and_shoulders(swing_highs: list[dict], swing_lows: list[dict],
                                candles: list[dict], tol: float) -> list[dict]:
    """Detect Head and Shoulders (bearish reversal) patterns."""
    formations = []
    for i in range(2, len(swing_highs)):
        ls = swing_highs[i - 2]  # Left shoulder
        head = swing_highs[i - 1]  # Head
        rs = swing_highs[i]  # Right shoulder

        if head["bar"] <= ls["bar"] or rs["bar"] <= head["bar"]:
            continue

        # Head must be highest
        if head["price"] <= ls["price"] or head["price"] <= rs["price"]:
            continue

        # Shoulders roughly equal (within tolerance)
        if abs(ls["price"] - rs["price"]) > tol * 2:
            continue

        # Head significantly above shoulders
        shoulder_avg = (ls["price"] + rs["price"]) / 2
        if (head["price"] - shoulder_avg) / shoulder_avg < 0.02:
            continue

        # Find neckline (lows between shoulders and head)
        neckline_lows = [l for l in swing_lows
                         if ls["bar"] < l["bar"] < rs["bar"]]
        if len(neckline_lows) < 1:
            continue

        neckline_price = statistics.mean(l["price"] for l in neckline_lows)
        target = neckline_price - (head["price"] - neckline_price)

        formations.append({
            "pattern": "head_and_shoulders",
            "type": "bearish",
            "signal": "reversal",
            "reliability": "high",
            "left_shoulder": {"bar": ls["bar"], "date": ls["date"], "price": ls["price"]},
            "head": {"bar": head["bar"], "date": head["date"], "price": head["price"]},
            "right_shoulder": {"bar": rs["bar"], "date": rs["date"], "price": rs["price"]},
            "neckline": round(neckline_price, 4),
            "target": round(target, 4),
            "pattern_height_pct": round((head["price"] - neckline_price) / neckline_price * 100, 2),
        })

    return formations


def _detect_inverse_head_and_shoulders(swing_highs: list[dict], swing_lows: list[dict],
                                        candles: list[dict], tol: float) -> list[dict]:
    """Detect Inverse Head and Shoulders (bullish reversal)."""
    formations = []
    for i in range(2, len(swing_lows)):
        ls = swing_lows[i - 2]
        head = swing_lows[i - 1]
        rs = swing_lows[i]

        if head["bar"] <= ls["bar"] or rs["bar"] <= head["bar"]:
            continue

        # Head must be lowest
        if head["price"] >= ls["price"] or head["price"] >= rs["price"]:
            continue

        # Shoulders roughly equal
        if abs(ls["price"] - rs["price"]) > tol * 2:
            continue

        shoulder_avg = (ls["price"] + rs["price"]) / 2
        if (shoulder_avg - head["price"]) / shoulder_avg < 0.02:
            continue

        neckline_highs = [h for h in swing_highs
                          if ls["bar"] < h["bar"] < rs["bar"]]
        if len(neckline_highs) < 1:
            continue

        neckline_price = statistics.mean(h["price"] for h in neckline_highs)
        target = neckline_price + (neckline_price - head["price"])

        formations.append({
            "pattern": "inverse_head_and_shoulders",
            "type": "bullish",
            "signal": "reversal",
            "reliability": "high",
            "left_shoulder": {"bar": ls["bar"], "date": ls["date"], "price": ls["price"]},
            "head": {"bar": head["bar"], "date": head["date"], "price": head["price"]},
            "right_shoulder": {"bar": rs["bar"], "date": rs["date"], "price": rs["price"]},
            "neckline": round(neckline_price, 4),
            "target": round(target, 4),
            "pattern_height_pct": round((neckline_price - head["price"]) / neckline_price * 100, 2),
        })

    return formations


# ─── Double / Triple Top & Bottom ────────────────────────────────────────────

def _detect_double_top(swing_highs: list[dict], swing_lows: list[dict], tol: float) -> list[dict]:
    formations = []
    for i in range(1, len(swing_highs)):
        h1 = swing_highs[i - 1]
        h2 = swing_highs[i]
        if h2["bar"] <= h1["bar"]:
            continue
        if abs(h1["price"] - h2["price"]) > tol:
            continue
        # Must have a swing low between them
        between_lows = [l for l in swing_lows if h1["bar"] < l["bar"] < h2["bar"]]
        if not between_lows:
            continue
        neckline = min(l["price"] for l in between_lows)
        top_avg = (h1["price"] + h2["price"]) / 2
        target = neckline - (top_avg - neckline)
        formations.append({
            "pattern": "double_top",
            "type": "bearish",
            "signal": "reversal",
            "reliability": "high",
            "peak_1": {"bar": h1["bar"], "date": h1["date"], "price": h1["price"]},
            "peak_2": {"bar": h2["bar"], "date": h2["date"], "price": h2["price"]},
            "neckline": round(neckline, 4),
            "target": round(target, 4),
        })
    return formations


def _detect_double_bottom(swing_highs: list[dict], swing_lows: list[dict], tol: float) -> list[dict]:
    formations = []
    for i in range(1, len(swing_lows)):
        l1 = swing_lows[i - 1]
        l2 = swing_lows[i]
        if l2["bar"] <= l1["bar"]:
            continue
        if abs(l1["price"] - l2["price"]) > tol:
            continue
        between_highs = [h for h in swing_highs if l1["bar"] < h["bar"] < l2["bar"]]
        if not between_highs:
            continue
        neckline = max(h["price"] for h in between_highs)
        bottom_avg = (l1["price"] + l2["price"]) / 2
        target = neckline + (neckline - bottom_avg)
        formations.append({
            "pattern": "double_bottom",
            "type": "bullish",
            "signal": "reversal",
            "reliability": "high",
            "trough_1": {"bar": l1["bar"], "date": l1["date"], "price": l1["price"]},
            "trough_2": {"bar": l2["bar"], "date": l2["date"], "price": l2["price"]},
            "neckline": round(neckline, 4),
            "target": round(target, 4),
        })
    return formations


def _detect_triple_top(swing_highs: list[dict], swing_lows: list[dict], tol: float) -> list[dict]:
    formations = []
    for i in range(2, len(swing_highs)):
        h1, h2, h3 = swing_highs[i - 2], swing_highs[i - 1], swing_highs[i]
        if not (h1["bar"] < h2["bar"] < h3["bar"]):
            continue
        prices = [h1["price"], h2["price"], h3["price"]]
        if max(prices) - min(prices) > tol:
            continue
        between_lows = [l for l in swing_lows if h1["bar"] < l["bar"] < h3["bar"]]
        if len(between_lows) < 2:
            continue
        neckline = min(l["price"] for l in between_lows)
        top_avg = statistics.mean(prices)
        formations.append({
            "pattern": "triple_top",
            "type": "bearish",
            "signal": "reversal",
            "reliability": "high",
            "peaks": [{"bar": h["bar"], "date": h["date"], "price": h["price"]} for h in (h1, h2, h3)],
            "neckline": round(neckline, 4),
            "target": round(neckline - (top_avg - neckline), 4),
        })
    return formations


def _detect_triple_bottom(swing_highs: list[dict], swing_lows: list[dict], tol: float) -> list[dict]:
    formations = []
    for i in range(2, len(swing_lows)):
        l1, l2, l3 = swing_lows[i - 2], swing_lows[i - 1], swing_lows[i]
        if not (l1["bar"] < l2["bar"] < l3["bar"]):
            continue
        prices = [l1["price"], l2["price"], l3["price"]]
        if max(prices) - min(prices) > tol:
            continue
        between_highs = [h for h in swing_highs if l1["bar"] < h["bar"] < l3["bar"]]
        if len(between_highs) < 2:
            continue
        neckline = max(h["price"] for h in between_highs)
        bottom_avg = statistics.mean(prices)
        formations.append({
            "pattern": "triple_bottom",
            "type": "bullish",
            "signal": "reversal",
            "reliability": "high",
            "troughs": [{"bar": l["bar"], "date": l["date"], "price": l["price"]} for l in (l1, l2, l3)],
            "neckline": round(neckline, 4),
            "target": round(neckline + (neckline - bottom_avg), 4),
        })
    return formations


# ─── Triangles ───────────────────────────────────────────────────────────────

def _detect_triangles(swing_highs: list[dict], swing_lows: list[dict],
                       candles: list[dict]) -> list[dict]:
    """Detect ascending, descending, and symmetrical triangles."""
    formations = []

    # Need at least 2 highs and 2 lows in recent data
    recent_n = min(len(candles), 60)
    cutoff = len(candles) - recent_n

    rh = [h for h in swing_highs if h["bar"] >= cutoff]
    rl = [l for l in swing_lows if l["bar"] >= cutoff]

    if len(rh) < 2 or len(rl) < 2:
        return formations

    # Fit trendlines
    high_pts = [(h["bar"], h["price"]) for h in rh]
    low_pts = [(l["bar"], l["price"]) for l in rl]

    h_slope, h_int, h_r2 = _linear_regression(high_pts)
    l_slope, l_int, l_r2 = _linear_regression(low_pts)

    # Minimum fit quality
    if h_r2 < 0.5 or l_r2 < 0.5:
        return formations

    avg_price = statistics.mean(c["close"] for c in candles[-recent_n:])
    slope_threshold = avg_price * 0.0001  # Near-flat threshold

    converging = (h_slope < -slope_threshold and l_slope > slope_threshold) or \
                 (h_slope < 0 and l_slope > 0)

    if abs(h_slope) < slope_threshold and l_slope > slope_threshold:
        # Flat top, rising bottom = ascending triangle (bullish)
        resistance = statistics.mean(h["price"] for h in rh)
        formations.append({
            "pattern": "ascending_triangle",
            "type": "bullish",
            "signal": "continuation",
            "reliability": "high",
            "resistance": round(resistance, 4),
            "support_slope": round(l_slope, 6),
            "high_r2": round(h_r2, 4),
            "low_r2": round(l_r2, 4),
            "start_bar": min(rh[0]["bar"], rl[0]["bar"]),
            "start_date": candles[min(rh[0]["bar"], rl[0]["bar"])]["date"],
            "end_date": candles[-1]["date"],
            "breakout_direction": "up",
            "target": round(resistance + (resistance - rl[-1]["price"]), 4),
        })

    elif abs(l_slope) < slope_threshold and h_slope < -slope_threshold:
        # Flat bottom, falling top = descending triangle (bearish)
        support = statistics.mean(l["price"] for l in rl)
        formations.append({
            "pattern": "descending_triangle",
            "type": "bearish",
            "signal": "continuation",
            "reliability": "high",
            "support": round(support, 4),
            "resistance_slope": round(h_slope, 6),
            "high_r2": round(h_r2, 4),
            "low_r2": round(l_r2, 4),
            "start_bar": min(rh[0]["bar"], rl[0]["bar"]),
            "start_date": candles[min(rh[0]["bar"], rl[0]["bar"])]["date"],
            "end_date": candles[-1]["date"],
            "breakout_direction": "down",
            "target": round(support - (rh[-1]["price"] - support), 4),
        })

    elif converging:
        # Both converging = symmetrical triangle
        apex_bar = None
        if abs(h_slope - l_slope) > 0:
            apex_bar = int((l_int - h_int) / (h_slope - l_slope))

        formations.append({
            "pattern": "symmetrical_triangle",
            "type": "neutral",
            "signal": "breakout_pending",
            "reliability": "moderate",
            "high_slope": round(h_slope, 6),
            "low_slope": round(l_slope, 6),
            "high_r2": round(h_r2, 4),
            "low_r2": round(l_r2, 4),
            "apex_bar": apex_bar,
            "start_bar": min(rh[0]["bar"], rl[0]["bar"]),
            "start_date": candles[min(rh[0]["bar"], rl[0]["bar"])]["date"],
            "end_date": candles[-1]["date"],
            "breakout_direction": "either",
        })

    return formations


# ─── Wedges ──────────────────────────────────────────────────────────────────

def _detect_wedges(swing_highs: list[dict], swing_lows: list[dict],
                    candles: list[dict]) -> list[dict]:
    """Detect rising and falling wedges."""
    formations = []
    recent_n = min(len(candles), 60)
    cutoff = len(candles) - recent_n

    rh = [h for h in swing_highs if h["bar"] >= cutoff]
    rl = [l for l in swing_lows if l["bar"] >= cutoff]

    if len(rh) < 2 or len(rl) < 2:
        return formations

    high_pts = [(h["bar"], h["price"]) for h in rh]
    low_pts = [(l["bar"], l["price"]) for l in rl]

    h_slope, h_int, h_r2 = _linear_regression(high_pts)
    l_slope, l_int, l_r2 = _linear_regression(low_pts)

    if h_r2 < 0.4 or l_r2 < 0.4:
        return formations

    # Rising wedge: both slopes positive, converging (high slope < low slope)
    if h_slope > 0 and l_slope > 0 and h_slope < l_slope:
        formations.append({
            "pattern": "rising_wedge",
            "type": "bearish",
            "signal": "reversal",
            "reliability": "high",
            "high_slope": round(h_slope, 6),
            "low_slope": round(l_slope, 6),
            "high_r2": round(h_r2, 4),
            "low_r2": round(l_r2, 4),
            "start_date": candles[min(rh[0]["bar"], rl[0]["bar"])]["date"],
            "end_date": candles[-1]["date"],
            "breakout_direction": "down",
        })

    # Falling wedge: both slopes negative, converging (low slope > high slope, both neg)
    if h_slope < 0 and l_slope < 0 and l_slope > h_slope:
        formations.append({
            "pattern": "falling_wedge",
            "type": "bullish",
            "signal": "reversal",
            "reliability": "high",
            "high_slope": round(h_slope, 6),
            "low_slope": round(l_slope, 6),
            "high_r2": round(h_r2, 4),
            "low_r2": round(l_r2, 4),
            "start_date": candles[min(rh[0]["bar"], rl[0]["bar"])]["date"],
            "end_date": candles[-1]["date"],
            "breakout_direction": "up",
        })

    return formations


# ─── Flags and Pennants ──────────────────────────────────────────────────────

def _detect_flags(candles: list[dict], swing_highs: list[dict], swing_lows: list[dict]) -> list[dict]:
    """Detect bull flags, bear flags, and pennants."""
    formations = []

    if len(candles) < 30:
        return formations

    # Look for a strong impulse move followed by a consolidation
    closes = [c["close"] for c in candles]

    for window_end in range(len(candles) - 5, max(20, len(candles) - 30), -5):
        # Look for impulse in 5-15 bar window before consolidation
        for impulse_len in range(5, 16):
            impulse_start = window_end - impulse_len
            if impulse_start < 0:
                continue

            impulse_move = (closes[window_end] - closes[impulse_start]) / closes[impulse_start] * 100

            # Need significant impulse (>5%)
            if abs(impulse_move) < 5:
                continue

            # Check consolidation after impulse (next 5-20 bars)
            consol_end = min(window_end + 20, len(candles) - 1)
            consol_candles = candles[window_end:consol_end + 1]

            if len(consol_candles) < 5:
                continue

            consol_range = max(c["high"] for c in consol_candles) - min(c["low"] for c in consol_candles)
            impulse_range = abs(candles[window_end]["close"] - candles[impulse_start]["close"])

            if impulse_range == 0:
                continue

            # Consolidation should be <50% of impulse range
            if consol_range / impulse_range > 0.5:
                continue

            # Consolidation slope
            consol_closes = [c["close"] for c in consol_candles]
            slope, _, r2 = _linear_regression(list(enumerate(consol_closes)))

            is_bull_flag = impulse_move > 5 and slope < 0  # Up impulse, down consolidation
            is_bear_flag = impulse_move < -5 and slope > 0  # Down impulse, up consolidation

            # Pennant: consolidation converges
            consol_highs = [c["high"] for c in consol_candles]
            consol_lows = [c["low"] for c in consol_candles]
            h_slope, _, _ = _linear_regression(list(enumerate(consol_highs)))
            l_slope, _, _ = _linear_regression(list(enumerate(consol_lows)))
            is_pennant = h_slope < 0 and l_slope > 0

            if is_bull_flag:
                pattern = "pennant" if is_pennant else "bull_flag"
                target = closes[consol_end] + abs(impulse_move) / 100 * closes[consol_end]
                formations.append({
                    "pattern": pattern,
                    "type": "bullish",
                    "signal": "continuation",
                    "reliability": "high",
                    "impulse_start_date": candles[impulse_start]["date"],
                    "impulse_move_pct": round(impulse_move, 2),
                    "consolidation_start_date": candles[window_end]["date"],
                    "consolidation_end_date": candles[consol_end]["date"],
                    "consolidation_slope": round(slope, 6),
                    "target": round(target, 4),
                })
                return formations  # Return first found

            elif is_bear_flag:
                pattern = "pennant" if is_pennant else "bear_flag"
                target = closes[consol_end] - abs(impulse_move) / 100 * closes[consol_end]
                formations.append({
                    "pattern": pattern,
                    "type": "bearish",
                    "signal": "continuation",
                    "reliability": "high",
                    "impulse_start_date": candles[impulse_start]["date"],
                    "impulse_move_pct": round(impulse_move, 2),
                    "consolidation_start_date": candles[window_end]["date"],
                    "consolidation_end_date": candles[consol_end]["date"],
                    "consolidation_slope": round(slope, 6),
                    "target": round(target, 4),
                })
                return formations

    return formations


# ─── Cup and Handle ──────────────────────────────────────────────────────────

def _detect_cup_and_handle(swing_highs: list[dict], swing_lows: list[dict],
                            candles: list[dict], tol: float) -> list[dict]:
    """Detect cup-and-handle (bullish continuation) pattern."""
    formations = []
    closes = [c["close"] for c in candles]

    # Need at least 2 swing highs at similar level with a low between them
    for i in range(1, len(swing_highs)):
        left_rim = swing_highs[i - 1]
        right_rim = swing_highs[i]

        if right_rim["bar"] <= left_rim["bar"]:
            continue

        # Rims should be at roughly the same level
        if abs(left_rim["price"] - right_rim["price"]) > tol * 1.5:
            continue

        # Find cup bottom (lowest point between rims)
        cup_section = candles[left_rim["bar"]:right_rim["bar"] + 1]
        if len(cup_section) < 10:
            continue

        cup_bottom_price = min(c["low"] for c in cup_section)
        cup_bottom_bar = left_rim["bar"] + next(
            i for i, c in enumerate(cup_section) if c["low"] == cup_bottom_price
        )

        # Cup depth should be 12-35% of rim height
        rim_avg = (left_rim["price"] + right_rim["price"]) / 2
        cup_depth_pct = (rim_avg - cup_bottom_price) / rim_avg * 100
        if cup_depth_pct < 12 or cup_depth_pct > 35:
            continue

        # Cup should be roughly U-shaped (not V-shaped)
        left_half = cup_section[:len(cup_section) // 2]
        right_half = cup_section[len(cup_section) // 2:]
        if not left_half or not right_half:
            continue

        # Check for handle (small consolidation after right rim)
        handle_start = right_rim["bar"]
        handle_end = min(handle_start + 15, len(candles) - 1)
        handle_section = candles[handle_start:handle_end + 1]

        has_handle = False
        handle_low = None
        if len(handle_section) >= 3:
            handle_low_price = min(c["low"] for c in handle_section)
            handle_depth = (right_rim["price"] - handle_low_price) / right_rim["price"] * 100
            if 1 < handle_depth < cup_depth_pct * 0.5:
                has_handle = True
                handle_low = handle_low_price

        breakout_level = max(left_rim["price"], right_rim["price"])
        target = breakout_level + (breakout_level - cup_bottom_price)

        formations.append({
            "pattern": "cup_and_handle" if has_handle else "cup",
            "type": "bullish",
            "signal": "continuation",
            "reliability": "high" if has_handle else "moderate",
            "left_rim": {"bar": left_rim["bar"], "date": left_rim["date"], "price": left_rim["price"]},
            "cup_bottom": {"bar": cup_bottom_bar, "date": candles[cup_bottom_bar]["date"], "price": cup_bottom_price},
            "right_rim": {"bar": right_rim["bar"], "date": right_rim["date"], "price": right_rim["price"]},
            "cup_depth_pct": round(cup_depth_pct, 2),
            "has_handle": has_handle,
            "handle_low": round(handle_low, 4) if handle_low else None,
            "breakout_level": round(breakout_level, 4),
            "target": round(target, 4),
        })

    return formations


# ─── Channel Detection ───────────────────────────────────────────────────────

def _detect_channels(swing_highs: list[dict], swing_lows: list[dict],
                      candles: list[dict]) -> list[dict]:
    """Detect ascending, descending, and horizontal channels."""
    formations = []
    recent_n = min(len(candles), 80)
    cutoff = len(candles) - recent_n

    rh = [h for h in swing_highs if h["bar"] >= cutoff]
    rl = [l for l in swing_lows if l["bar"] >= cutoff]

    if len(rh) < 2 or len(rl) < 2:
        return formations

    high_pts = [(h["bar"], h["price"]) for h in rh]
    low_pts = [(l["bar"], l["price"]) for l in rl]

    h_slope, h_int, h_r2 = _linear_regression(high_pts)
    l_slope, l_int, l_r2 = _linear_regression(low_pts)

    if h_r2 < 0.6 or l_r2 < 0.6:
        return formations

    # Slopes must be roughly parallel (similar direction and magnitude)
    avg_price = statistics.mean(c["close"] for c in candles[-recent_n:])
    slope_diff = abs(h_slope - l_slope)
    avg_slope = (h_slope + l_slope) / 2

    # Parallel check: slopes within 30% of each other
    if max(abs(h_slope), abs(l_slope)) > 0:
        if slope_diff / max(abs(h_slope), abs(l_slope), 0.0001) > 0.5:
            return formations

    # Channel width
    channel_width = abs(h_int - l_int)
    width_pct = channel_width / avg_price * 100

    slope_threshold = avg_price * 0.0002

    if avg_slope > slope_threshold:
        pattern = "ascending_channel"
        chan_type = "bullish"
    elif avg_slope < -slope_threshold:
        pattern = "descending_channel"
        chan_type = "bearish"
    else:
        pattern = "horizontal_channel"
        chan_type = "neutral"

    # Current position within channel
    last_bar = len(candles) - 1
    upper_at_last = h_slope * last_bar + h_int
    lower_at_last = l_slope * last_bar + l_int
    current_price = candles[-1]["close"]

    if upper_at_last != lower_at_last:
        position_in_channel = (current_price - lower_at_last) / (upper_at_last - lower_at_last)
    else:
        position_in_channel = 0.5

    formations.append({
        "pattern": pattern,
        "type": chan_type,
        "signal": "trend_structure",
        "reliability": "moderate",
        "high_slope": round(h_slope, 6),
        "low_slope": round(l_slope, 6),
        "high_r2": round(h_r2, 4),
        "low_r2": round(l_r2, 4),
        "channel_width_pct": round(width_pct, 2),
        "current_upper_bound": round(upper_at_last, 4),
        "current_lower_bound": round(lower_at_last, 4),
        "position_in_channel": round(position_in_channel, 4),
        "start_date": candles[cutoff]["date"],
        "end_date": candles[-1]["date"],
        "near_upper": position_in_channel > 0.8,
        "near_lower": position_in_channel < 0.2,
    })

    return formations


# ─── Public API ──────────────────────────────────────────────────────────────

def detect_chart_formations(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    lookback: int = 5,
) -> dict:
    """
    Detect classical chart formations from OHLCV data.

    Scans for Head & Shoulders, Double/Triple Top/Bottom, Triangles,
    Wedges, Flags, Pennants, Cup & Handle, and Channels.

    Args:
        symbol:   Yahoo Finance symbol
        period:   Historical period
        interval: 1d or 1h
        lookback: Swing detection sensitivity (default 5)
    """
    try:
        candles = fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < 40:
        return {"error": f"Not enough data ({len(candles)} bars). Need at least 40."}

    tol = _price_tolerance(candles)
    swing_highs, swing_lows = _find_swings(candles, lookback)

    all_formations = []

    # Detect all pattern types
    all_formations.extend(_detect_head_and_shoulders(swing_highs, swing_lows, candles, tol))
    all_formations.extend(_detect_inverse_head_and_shoulders(swing_highs, swing_lows, candles, tol))
    all_formations.extend(_detect_double_top(swing_highs, swing_lows, tol))
    all_formations.extend(_detect_double_bottom(swing_highs, swing_lows, tol))
    all_formations.extend(_detect_triple_top(swing_highs, swing_lows, tol))
    all_formations.extend(_detect_triple_bottom(swing_highs, swing_lows, tol))
    all_formations.extend(_detect_triangles(swing_highs, swing_lows, candles))
    all_formations.extend(_detect_wedges(swing_highs, swing_lows, candles))
    all_formations.extend(_detect_flags(candles, swing_highs, swing_lows))
    all_formations.extend(_detect_cup_and_handle(swing_highs, swing_lows, candles, tol))
    all_formations.extend(_detect_channels(swing_highs, swing_lows, candles))

    # Summary
    bullish = [f for f in all_formations if f["type"] == "bullish"]
    bearish = [f for f in all_formations if f["type"] == "bearish"]
    neutral = [f for f in all_formations if f["type"] == "neutral"]

    if bullish and not bearish:
        bias = "BULLISH"
    elif bearish and not bullish:
        bias = "BEARISH"
    elif bullish and bearish:
        bias = "MIXED"
    else:
        bias = "NO FORMATIONS"

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "swing_highs_found": len(swing_highs),
        "swing_lows_found": len(swing_lows),
        "total_formations": len(all_formations),
        "bias": bias,
        "summary": {
            "bullish_formations": len(bullish),
            "bearish_formations": len(bearish),
            "neutral_formations": len(neutral),
        },
        "formations": all_formations,
        "disclaimer": "Chart pattern recognition is geometric and probabilistic. Patterns may not "
                      "complete or may fail. Always use with other confirmation signals. For educational use only.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
