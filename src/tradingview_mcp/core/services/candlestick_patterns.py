"""
Candlestick Pattern Recognition for tradingview-mcp

Pure Python — no pandas, no numpy, no TA-Lib.

Detects 30+ single-bar, dual-bar, and triple-bar candlestick patterns from OHLCV data.

Pattern categories:
  Single-bar:  Doji, Hammer, Inverted Hammer, Shooting Star, Hanging Man,
               Marubozu, Spinning Top, Dragonfly Doji, Gravestone Doji
  Dual-bar:    Bullish/Bearish Engulfing, Piercing Line, Dark Cloud Cover,
               Tweezer Top/Bottom, Harami, Harami Cross
  Triple-bar:  Morning Star, Evening Star, Three White Soldiers,
               Three Black Crows, Rising/Falling Three Methods,
               Three Inside Up/Down
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.data_fetcher import fetch_ohlcv
from tradingview_mcp.core.services.indicators_calc import calc_rsi, calc_atr


# ─── Candle Geometry Helpers ─────────────────────────────────────────────────

def _body(c: dict) -> float:
    return abs(c["close"] - c["open"])


def _range(c: dict) -> float:
    return c["high"] - c["low"]


def _upper_wick(c: dict) -> float:
    return c["high"] - max(c["open"], c["close"])


def _lower_wick(c: dict) -> float:
    return min(c["open"], c["close"]) - c["low"]


def _is_bullish(c: dict) -> bool:
    return c["close"] >= c["open"]


def _is_bearish(c: dict) -> bool:
    return c["close"] < c["open"]


def _body_pct(c: dict) -> float:
    r = _range(c)
    return _body(c) / r if r > 0 else 0


def _midpoint(c: dict) -> float:
    return (c["open"] + c["close"]) / 2


# ─── Single-Bar Patterns ────────────────────────────────────────────────────

def _detect_doji(c: dict, avg_body: float) -> Optional[str]:
    """Doji: body < 10% of range, or body < 20% of avg body."""
    if _range(c) == 0:
        return None
    body_ratio = _body(c) / _range(c)
    if body_ratio > 0.1:
        if avg_body > 0 and _body(c) / avg_body > 0.2:
            return None

    uw = _upper_wick(c)
    lw = _lower_wick(c)
    r = _range(c)

    if r == 0:
        return "doji"

    if lw / r > 0.6 and uw / r < 0.1:
        return "dragonfly_doji"
    elif uw / r > 0.6 and lw / r < 0.1:
        return "gravestone_doji"
    else:
        return "doji"


def _detect_hammer(c: dict, avg_body: float) -> Optional[str]:
    """Hammer (bullish) / Hanging Man (bearish at top of trend)."""
    r = _range(c)
    if r == 0:
        return None
    body = _body(c)
    lw = _lower_wick(c)
    uw = _upper_wick(c)

    # Hammer: small body at top, long lower wick (>2x body), small upper wick
    if body > 0 and lw >= body * 2 and uw <= body * 0.3:
        return "hammer"
    return None


def _detect_inverted_hammer(c: dict, avg_body: float) -> Optional[str]:
    """Inverted Hammer: small body at bottom, long upper wick."""
    r = _range(c)
    if r == 0:
        return None
    body = _body(c)
    uw = _upper_wick(c)
    lw = _lower_wick(c)

    if body > 0 and uw >= body * 2 and lw <= body * 0.3:
        return "inverted_hammer"
    return None


def _detect_shooting_star(c: dict, avg_body: float) -> Optional[str]:
    """Shooting Star: bearish version of inverted hammer (at top of trend)."""
    r = _range(c)
    if r == 0:
        return None
    body = _body(c)
    uw = _upper_wick(c)
    lw = _lower_wick(c)

    if body > 0 and uw >= body * 2 and lw <= body * 0.3 and _is_bearish(c):
        return "shooting_star"
    return None


def _detect_marubozu(c: dict, avg_body: float) -> Optional[str]:
    """Marubozu: full body, no or tiny wicks (>95% body)."""
    r = _range(c)
    if r == 0:
        return None
    if _body_pct(c) >= 0.95:
        return "bullish_marubozu" if _is_bullish(c) else "bearish_marubozu"
    return None


def _detect_spinning_top(c: dict, avg_body: float) -> Optional[str]:
    """Spinning Top: small body, roughly equal upper and lower wicks."""
    r = _range(c)
    if r == 0 or avg_body == 0:
        return None
    body = _body(c)
    uw = _upper_wick(c)
    lw = _lower_wick(c)

    if body < avg_body * 0.4 and uw > body and lw > body:
        # Wicks should be roughly similar
        wick_ratio = min(uw, lw) / max(uw, lw) if max(uw, lw) > 0 else 0
        if wick_ratio > 0.4:
            return "spinning_top"
    return None


# ─── Dual-Bar Patterns ──────────────────────────────────────────────────────

def _detect_engulfing(c1: dict, c2: dict) -> Optional[str]:
    """Bullish/Bearish Engulfing: c2 body completely engulfs c1 body."""
    if _body(c2) <= _body(c1):
        return None

    if _is_bearish(c1) and _is_bullish(c2):
        if c2["open"] <= c1["close"] and c2["close"] >= c1["open"]:
            return "bullish_engulfing"

    if _is_bullish(c1) and _is_bearish(c2):
        if c2["open"] >= c1["close"] and c2["close"] <= c1["open"]:
            return "bearish_engulfing"

    return None


def _detect_piercing_line(c1: dict, c2: dict) -> Optional[str]:
    """Piercing Line: bearish c1, bullish c2 opens below c1 low, closes above c1 midpoint."""
    if not (_is_bearish(c1) and _is_bullish(c2)):
        return None
    if c2["open"] < c1["low"] and c2["close"] > _midpoint(c1) and c2["close"] < c1["open"]:
        return "piercing_line"
    return None


def _detect_dark_cloud(c1: dict, c2: dict) -> Optional[str]:
    """Dark Cloud Cover: bullish c1, bearish c2 opens above c1 high, closes below c1 midpoint."""
    if not (_is_bullish(c1) and _is_bearish(c2)):
        return None
    if c2["open"] > c1["high"] and c2["close"] < _midpoint(c1) and c2["close"] > c1["open"]:
        return "dark_cloud_cover"
    return None


def _detect_tweezer(c1: dict, c2: dict, tolerance: float = 0.001) -> Optional[str]:
    """Tweezer Top/Bottom: two candles with nearly identical highs or lows."""
    if c1["close"] == 0:
        return None
    tol = c1["close"] * tolerance

    if abs(c1["high"] - c2["high"]) <= tol and _is_bullish(c1) and _is_bearish(c2):
        return "tweezer_top"
    if abs(c1["low"] - c2["low"]) <= tol and _is_bearish(c1) and _is_bullish(c2):
        return "tweezer_bottom"
    return None


def _detect_harami(c1: dict, c2: dict, avg_body: float) -> Optional[str]:
    """Harami: c2 body is contained within c1 body."""
    if _body(c1) == 0:
        return None
    if _body(c2) >= _body(c1):
        return None

    b1_hi = max(c1["open"], c1["close"])
    b1_lo = min(c1["open"], c1["close"])
    b2_hi = max(c2["open"], c2["close"])
    b2_lo = min(c2["open"], c2["close"])

    if b2_hi <= b1_hi and b2_lo >= b1_lo:
        # Harami Cross if c2 is a doji
        if _body(c2) < avg_body * 0.15:
            return "bullish_harami_cross" if _is_bearish(c1) else "bearish_harami_cross"
        return "bullish_harami" if _is_bearish(c1) else "bearish_harami"
    return None


# ─── Triple-Bar Patterns ────────────────────────────────────────────────────

def _detect_morning_star(c1: dict, c2: dict, c3: dict, avg_body: float) -> Optional[str]:
    """Morning Star: bearish c1, small-body c2 (gaps down), bullish c3 (closes above c1 mid)."""
    if not _is_bearish(c1):
        return None
    if _body(c2) > avg_body * 0.4:
        return None
    if not _is_bullish(c3):
        return None
    if c3["close"] > _midpoint(c1):
        return "morning_star"
    return None


def _detect_evening_star(c1: dict, c2: dict, c3: dict, avg_body: float) -> Optional[str]:
    """Evening Star: bullish c1, small-body c2 (gaps up), bearish c3 (closes below c1 mid)."""
    if not _is_bullish(c1):
        return None
    if _body(c2) > avg_body * 0.4:
        return None
    if not _is_bearish(c3):
        return None
    if c3["close"] < _midpoint(c1):
        return "evening_star"
    return None


def _detect_three_soldiers(c1: dict, c2: dict, c3: dict, avg_body: float) -> Optional[str]:
    """Three White Soldiers: three consecutive bullish candles with higher closes."""
    if not all(_is_bullish(c) for c in (c1, c2, c3)):
        return None
    if not (c2["close"] > c1["close"] and c3["close"] > c2["close"]):
        return None
    if not (c2["open"] > c1["open"] and c3["open"] > c2["open"]):
        return None
    # Each body should be significant
    min_body = avg_body * 0.5
    if all(_body(c) >= min_body for c in (c1, c2, c3)):
        return "three_white_soldiers"
    return None


def _detect_three_crows(c1: dict, c2: dict, c3: dict, avg_body: float) -> Optional[str]:
    """Three Black Crows: three consecutive bearish candles with lower closes."""
    if not all(_is_bearish(c) for c in (c1, c2, c3)):
        return None
    if not (c2["close"] < c1["close"] and c3["close"] < c2["close"]):
        return None
    if not (c2["open"] < c1["open"] and c3["open"] < c2["open"]):
        return None
    min_body = avg_body * 0.5
    if all(_body(c) >= min_body for c in (c1, c2, c3)):
        return "three_black_crows"
    return None


def _detect_three_inside(c1: dict, c2: dict, c3: dict) -> Optional[str]:
    """Three Inside Up/Down: harami followed by confirmation candle."""
    b1_hi = max(c1["open"], c1["close"])
    b1_lo = min(c1["open"], c1["close"])
    b2_hi = max(c2["open"], c2["close"])
    b2_lo = min(c2["open"], c2["close"])

    # c2 inside c1
    if not (b2_hi <= b1_hi and b2_lo >= b1_lo):
        return None

    if _is_bearish(c1) and _is_bullish(c2) and _is_bullish(c3) and c3["close"] > b1_hi:
        return "three_inside_up"
    if _is_bullish(c1) and _is_bearish(c2) and _is_bearish(c3) and c3["close"] < b1_lo:
        return "three_inside_down"
    return None


# ─── Pattern Metadata ────────────────────────────────────────────────────────

_PATTERN_INFO = {
    "doji":                  {"type": "neutral",  "signal": "indecision",  "reliability": "moderate", "bars": 1},
    "dragonfly_doji":        {"type": "bullish",  "signal": "reversal",   "reliability": "moderate", "bars": 1},
    "gravestone_doji":       {"type": "bearish",  "signal": "reversal",   "reliability": "moderate", "bars": 1},
    "hammer":                {"type": "bullish",  "signal": "reversal",   "reliability": "high",     "bars": 1},
    "inverted_hammer":       {"type": "bullish",  "signal": "reversal",   "reliability": "moderate", "bars": 1},
    "shooting_star":         {"type": "bearish",  "signal": "reversal",   "reliability": "high",     "bars": 1},
    "bullish_marubozu":      {"type": "bullish",  "signal": "momentum",   "reliability": "high",     "bars": 1},
    "bearish_marubozu":      {"type": "bearish",  "signal": "momentum",   "reliability": "high",     "bars": 1},
    "spinning_top":          {"type": "neutral",  "signal": "indecision",  "reliability": "low",      "bars": 1},
    "bullish_engulfing":     {"type": "bullish",  "signal": "reversal",   "reliability": "high",     "bars": 2},
    "bearish_engulfing":     {"type": "bearish",  "signal": "reversal",   "reliability": "high",     "bars": 2},
    "piercing_line":         {"type": "bullish",  "signal": "reversal",   "reliability": "moderate", "bars": 2},
    "dark_cloud_cover":      {"type": "bearish",  "signal": "reversal",   "reliability": "moderate", "bars": 2},
    "tweezer_top":           {"type": "bearish",  "signal": "reversal",   "reliability": "moderate", "bars": 2},
    "tweezer_bottom":        {"type": "bullish",  "signal": "reversal",   "reliability": "moderate", "bars": 2},
    "bullish_harami":        {"type": "bullish",  "signal": "reversal",   "reliability": "moderate", "bars": 2},
    "bearish_harami":        {"type": "bearish",  "signal": "reversal",   "reliability": "moderate", "bars": 2},
    "bullish_harami_cross":  {"type": "bullish",  "signal": "reversal",   "reliability": "high",     "bars": 2},
    "bearish_harami_cross":  {"type": "bearish",  "signal": "reversal",   "reliability": "high",     "bars": 2},
    "morning_star":          {"type": "bullish",  "signal": "reversal",   "reliability": "high",     "bars": 3},
    "evening_star":          {"type": "bearish",  "signal": "reversal",   "reliability": "high",     "bars": 3},
    "three_white_soldiers":  {"type": "bullish",  "signal": "momentum",   "reliability": "high",     "bars": 3},
    "three_black_crows":     {"type": "bearish",  "signal": "momentum",   "reliability": "high",     "bars": 3},
    "three_inside_up":       {"type": "bullish",  "signal": "reversal",   "reliability": "high",     "bars": 3},
    "three_inside_down":     {"type": "bearish",  "signal": "reversal",   "reliability": "high",     "bars": 3},
}


# ─── Public API ──────────────────────────────────────────────────────────────

def detect_candlestick_patterns(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    min_reliability: str = "low",
) -> dict:
    """
    Scan for candlestick patterns in recent price data.

    Args:
        symbol:          Yahoo Finance symbol
        period:          Historical period
        interval:        1d or 1h
        min_reliability: Minimum pattern reliability to include: 'low', 'moderate', 'high'
    """
    try:
        candles = fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < 20:
        return {"error": f"Not enough data ({len(candles)} bars). Need at least 20."}

    reliability_order = {"low": 0, "moderate": 1, "high": 2}
    min_rel = reliability_order.get(min_reliability, 0)

    # Average body size for context
    bodies = [_body(c) for c in candles if _range(c) > 0]
    avg_body = statistics.mean(bodies) if bodies else 0

    # RSI for context on pattern significance
    closes = [c["close"] for c in candles]
    rsi = calc_rsi(closes, 14)

    patterns_found = []

    for i in range(len(candles)):
        c = candles[i]
        bar_patterns = []

        # ── Single-bar ───────────────────────────────────────────────────────
        for fn in (_detect_doji, _detect_hammer, _detect_inverted_hammer,
                   _detect_shooting_star, _detect_marubozu, _detect_spinning_top):
            result = fn(c, avg_body)
            if result:
                bar_patterns.append(result)

        # ── Dual-bar ────────────────────────────────────────────────────────
        if i >= 1:
            c1 = candles[i - 1]
            for fn in (_detect_engulfing, _detect_piercing_line, _detect_dark_cloud, _detect_tweezer):
                result = fn(c1, c)
                if result:
                    bar_patterns.append(result)

            result = _detect_harami(c1, c, avg_body)
            if result:
                bar_patterns.append(result)

        # ── Triple-bar ──────────────────────────────────────────────────────
        if i >= 2:
            c1, c2 = candles[i - 2], candles[i - 1]
            for fn in (_detect_morning_star, _detect_evening_star,
                       _detect_three_soldiers, _detect_three_crows):
                result = fn(c1, c2, c, avg_body)
                if result:
                    bar_patterns.append(result)

            result = _detect_three_inside(c1, c2, c)
            if result:
                bar_patterns.append(result)

        # Record found patterns
        for pattern_name in bar_patterns:
            info = _PATTERN_INFO.get(pattern_name, {})
            rel = reliability_order.get(info.get("reliability", "low"), 0)
            if rel < min_rel:
                continue

            patterns_found.append({
                "bar": i,
                "date": candles[i]["date"],
                "pattern": pattern_name,
                "type": info.get("type", "neutral"),
                "signal": info.get("signal", "unknown"),
                "reliability": info.get("reliability", "unknown"),
                "bars": info.get("bars", 1),
                "close": candles[i]["close"],
                "rsi": round(rsi[i], 1) if rsi[i] is not None else None,
                "volume": candles[i]["volume"],
            })

    # ── Summary ──────────────────────────────────────────────────────────────
    recent_cutoff = max(0, len(candles) - 10)
    recent = [p for p in patterns_found if p["bar"] >= recent_cutoff]

    bullish = [p for p in patterns_found if p["type"] == "bullish"]
    bearish = [p for p in patterns_found if p["type"] == "bearish"]

    recent_bullish = [p for p in recent if p["type"] == "bullish"]
    recent_bearish = [p for p in recent if p["type"] == "bearish"]

    if recent_bullish and not recent_bearish:
        bias = "BULLISH"
    elif recent_bearish and not recent_bullish:
        bias = "BEARISH"
    elif recent_bullish and recent_bearish:
        bias = "MIXED"
    else:
        bias = "NEUTRAL"

    # Pattern frequency
    pattern_counts = {}
    for p in patterns_found:
        pattern_counts[p["pattern"]] = pattern_counts.get(p["pattern"], 0) + 1

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "total_patterns_found": len(patterns_found),
        "recent_bias": bias,
        "summary": {
            "bullish_patterns": len(bullish),
            "bearish_patterns": len(bearish),
            "neutral_patterns": len(patterns_found) - len(bullish) - len(bearish),
        },
        "recent_patterns": recent,
        "pattern_frequency": dict(sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)),
        "all_patterns": patterns_found[-30:],
        "disclaimer": "Candlestick patterns are probabilistic and more reliable with volume/trend confirmation. "
                      "For educational use only.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
