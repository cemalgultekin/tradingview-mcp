"""
Volatility Regime Detector for tradingview-mcp

Pure Python — no pandas, no numpy.

Classifies the current market regime based on:
  1. ATR percentile rank (vs historical ATR distribution)
  2. Bollinger Band Width percentile
  3. Daily range (high-low) as % of close
  4. Returns standard deviation over rolling windows
  5. Regime transition detection (low→high volatility breakouts)

Regimes: LOW / NORMAL / HIGH / EXTREME
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.data_fetcher import fetch_ohlcv
from tradingview_mcp.core.services.indicators_calc import calc_atr, calc_bollinger


def _percentile_rank(value: float, distribution: list[float]) -> float:
    """What percentile is `value` in the given distribution? 0-100."""
    if not distribution:
        return 50.0
    count_below = sum(1 for v in distribution if v < value)
    return round(count_below / len(distribution) * 100, 1)


def _rolling_std(values: list[float], window: int) -> list[Optional[float]]:
    result = [None] * len(values)
    for i in range(window - 1, len(values)):
        w = values[i - window + 1:i + 1]
        if len(w) >= 2:
            result[i] = statistics.stdev(w)
    return result


def detect_volatility_regime(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    lookback: int = 20,
) -> dict:
    """
    Classify the current volatility regime and detect regime transitions.

    Args:
        symbol:   Yahoo Finance symbol
        period:   Historical period
        interval: 1d or 1h
        lookback: Window for current regime measurement (default 20)
    """
    try:
        candles = fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < 60:
        return {"error": f"Not enough data ({len(candles)} bars). Need at least 60."}

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    # ── ATR analysis ─────────────────────────────────────────────────────────
    atr_raw = calc_atr(highs, lows, closes, 14)
    valid_atr = [a for a in atr_raw if a is not None]
    current_atr = valid_atr[-1] if valid_atr else 0
    atr_pct = round(current_atr / closes[-1] * 100, 4) if closes[-1] > 0 else 0
    atr_percentile = _percentile_rank(current_atr, valid_atr)

    # ── Bollinger Band Width ─────────────────────────────────────────────────
    bb = calc_bollinger(closes, 20, 2.0)
    bbw_values = []
    for i in range(len(closes)):
        if bb["upper"][i] is not None and bb["middle"][i] and bb["middle"][i] > 0:
            bbw_values.append((bb["upper"][i] - bb["lower"][i]) / bb["middle"][i] * 100)
    current_bbw = bbw_values[-1] if bbw_values else 0
    bbw_percentile = _percentile_rank(current_bbw, bbw_values)

    # ── Daily range ──────────────────────────────────────────────────────────
    ranges = [(c["high"] - c["low"]) / c["close"] * 100 for c in candles if c["close"] > 0]
    current_range = ranges[-1] if ranges else 0
    avg_range = statistics.mean(ranges) if ranges else 0
    range_percentile = _percentile_rank(current_range, ranges)

    # ── Returns volatility ───────────────────────────────────────────────────
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1] * 100
               for i in range(1, len(closes)) if closes[i - 1] != 0]
    rolling_vol = _rolling_std(returns, lookback)
    valid_rvol = [v for v in rolling_vol if v is not None]
    current_rvol = valid_rvol[-1] if valid_rvol else 0
    rvol_percentile = _percentile_rank(current_rvol, valid_rvol)

    # ── Composite regime score ───────────────────────────────────────────────
    composite_percentile = round(
        (atr_percentile * 0.3 + bbw_percentile * 0.25 +
         range_percentile * 0.2 + rvol_percentile * 0.25), 1
    )

    if composite_percentile >= 85:
        regime = "EXTREME"
        description = "Extremely high volatility — expect large swings, widen stops, reduce position size"
    elif composite_percentile >= 65:
        regime = "HIGH"
        description = "Elevated volatility — trending strategies (Supertrend, MACD) tend to perform well"
    elif composite_percentile >= 35:
        regime = "NORMAL"
        description = "Normal volatility — balanced conditions, most strategies viable"
    else:
        regime = "LOW"
        description = "Low volatility / squeeze — mean-reversion (RSI, Bollinger) works; anticipate breakout"

    # ── Strategy recommendations ─────────────────────────────────────────────
    if regime == "LOW":
        recommended = ["rsi", "bollinger"]
        avoid = ["supertrend", "donchian"]
    elif regime == "EXTREME":
        recommended = ["supertrend"]
        avoid = ["rsi", "bollinger"]
    elif regime == "HIGH":
        recommended = ["supertrend", "macd", "donchian"]
        avoid = ["rsi"]
    else:
        recommended = ["ema_cross", "macd", "rsi"]
        avoid = []

    # ── Regime transitions ───────────────────────────────────────────────────
    transitions = []
    if len(valid_rvol) > lookback:
        for i in range(lookback, len(valid_rvol)):
            prev = valid_rvol[i - 1]
            curr = valid_rvol[i]
            if prev > 0 and curr / prev > 2.0:
                idx = len(candles) - len(valid_rvol) + i
                if 0 <= idx < len(candles):
                    transitions.append({
                        "date": candles[idx]["date"],
                        "type": "volatility_spike",
                        "prev_vol": round(prev, 4),
                        "curr_vol": round(curr, 4),
                        "ratio": round(curr / prev, 2),
                    })
            elif prev > 0 and curr / prev < 0.4:
                idx = len(candles) - len(valid_rvol) + i
                if 0 <= idx < len(candles):
                    transitions.append({
                        "date": candles[idx]["date"],
                        "type": "volatility_collapse",
                        "prev_vol": round(prev, 4),
                        "curr_vol": round(curr, 4),
                        "ratio": round(curr / prev, 2),
                    })

    # Squeeze detection (BBW at bottom 10%)
    squeeze = bbw_percentile < 10
    squeeze_duration = 0
    if squeeze and bbw_values:
        threshold = sorted(bbw_values)[max(0, int(len(bbw_values) * 0.1))]
        for v in reversed(bbw_values):
            if v <= threshold:
                squeeze_duration += 1
            else:
                break

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "regime": regime,
        "description": description,
        "composite_percentile": composite_percentile,
        "components": {
            "atr_pct": atr_pct,
            "atr_percentile": atr_percentile,
            "bbw": round(current_bbw, 4),
            "bbw_percentile": bbw_percentile,
            "daily_range_pct": round(current_range, 4),
            "range_percentile": range_percentile,
            "returns_volatility": round(current_rvol, 4),
            "returns_vol_percentile": rvol_percentile,
        },
        "squeeze": {
            "active": squeeze,
            "duration_bars": squeeze_duration,
            "note": "Bollinger squeeze — low volatility compression, breakout imminent" if squeeze else None,
        },
        "strategy_recommendations": {
            "recommended": recommended,
            "avoid": avoid,
        },
        "recent_transitions": transitions[-10:],
        "disclaimer": "Regime classification is probabilistic. For educational use only.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
