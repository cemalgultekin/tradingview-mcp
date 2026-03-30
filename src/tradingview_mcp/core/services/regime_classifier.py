"""
Market Regime Classifier for tradingview-mcp

Pure Python — no pandas, no numpy.

Classifies the current market into one of 4 regimes:
  1. TRENDING UP   — clear uptrend with momentum confirmation
  2. TRENDING DOWN — clear downtrend with momentum confirmation
  3. RANGING       — price oscillating within a band (mean-reversion territory)
  4. CHOPPY        — no clear direction, high noise, frequent reversals

Uses ADX for trend strength, directional indicators for direction,
and efficiency ratio for noise measurement.
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.data_fetcher import fetch_ohlcv
from tradingview_mcp.core.services.indicators_calc import calc_ema, calc_atr, calc_sma


def _calc_adx(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> dict:
    """
    Compute ADX, +DI, and -DI.
    Returns dict with 'adx', 'plus_di', 'minus_di' lists.
    """
    n = len(closes)
    plus_di = [None] * n
    minus_di = [None] * n
    adx = [None] * n

    if n < period * 2:
        return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}

    # True Range, +DM, -DM
    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm = up if up > down and up > 0 else 0
        minus_dm = down if down > up and down > 0 else 0
        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    # Smoothed averages (Wilder's)
    atr_s = sum(tr_list[:period]) / period
    plus_dm_s = sum(plus_dm_list[:period]) / period
    minus_dm_s = sum(minus_dm_list[:period]) / period

    dx_values = []

    for i in range(period, len(tr_list)):
        atr_s = (atr_s * (period - 1) + tr_list[i]) / period
        plus_dm_s = (plus_dm_s * (period - 1) + plus_dm_list[i]) / period
        minus_dm_s = (minus_dm_s * (period - 1) + minus_dm_list[i]) / period

        pdi = (plus_dm_s / atr_s * 100) if atr_s > 0 else 0
        mdi = (minus_dm_s / atr_s * 100) if atr_s > 0 else 0

        idx = i + 1  # offset because tr_list starts at index 1
        if idx < n:
            plus_di[idx] = round(pdi, 2)
            minus_di[idx] = round(mdi, 2)

        di_sum = pdi + mdi
        dx = abs(pdi - mdi) / di_sum * 100 if di_sum > 0 else 0
        dx_values.append((idx, dx))

    # ADX = smoothed average of DX
    if len(dx_values) >= period:
        adx_val = sum(d for _, d in dx_values[:period]) / period
        if dx_values[period - 1][0] < n:
            adx[dx_values[period - 1][0]] = round(adx_val, 2)

        for j in range(period, len(dx_values)):
            adx_val = (adx_val * (period - 1) + dx_values[j][1]) / period
            if dx_values[j][0] < n:
                adx[dx_values[j][0]] = round(adx_val, 2)

    return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}


def _efficiency_ratio(closes: list[float], period: int = 10) -> list[Optional[float]]:
    """
    Kaufman Efficiency Ratio: net price change / sum of absolute changes.
    1.0 = perfectly trending, 0.0 = perfectly choppy.
    """
    result = [None] * len(closes)
    for i in range(period, len(closes)):
        net = abs(closes[i] - closes[i - period])
        volatility = sum(abs(closes[j] - closes[j - 1]) for j in range(i - period + 1, i + 1))
        result[i] = round(net / volatility, 4) if volatility > 0 else 0.0
    return result


def classify_regime(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    adx_trend_threshold: float = 25.0,
    er_trend_threshold: float = 0.4,
) -> dict:
    """
    Classify the current market regime.

    Args:
        symbol:              Yahoo Finance symbol
        period:              Historical period
        interval:            1d or 1h
        adx_trend_threshold: ADX above this = trending (default 25)
        er_trend_threshold:  Efficiency ratio above this = trending (default 0.4)
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

    # Compute indicators
    adx_data = _calc_adx(highs, lows, closes)
    er = _efficiency_ratio(closes, 10)
    ema20 = calc_ema(closes, 20)
    ema50 = calc_ema(closes, 50)
    sma200 = calc_sma(closes, 200) if len(closes) >= 200 else [None] * len(closes)

    # Current values
    current_adx = next((adx_data["adx"][i] for i in range(len(closes) - 1, -1, -1) if adx_data["adx"][i] is not None), None)
    current_plus_di = next((adx_data["plus_di"][i] for i in range(len(closes) - 1, -1, -1) if adx_data["plus_di"][i] is not None), None)
    current_minus_di = next((adx_data["minus_di"][i] for i in range(len(closes) - 1, -1, -1) if adx_data["minus_di"][i] is not None), None)
    current_er = next((er[i] for i in range(len(closes) - 1, -1, -1) if er[i] is not None), None)
    current_ema20 = ema20[-1] if ema20[-1] is not None else closes[-1]
    current_ema50 = ema50[-1] if ema50[-1] is not None else closes[-1]

    # ── Regime classification ────────────────────────────────────────────────
    is_trending = (current_adx is not None and current_adx >= adx_trend_threshold) or \
                  (current_er is not None and current_er >= er_trend_threshold)

    is_bullish = (current_plus_di or 0) > (current_minus_di or 0) and current_ema20 > current_ema50
    is_bearish = (current_minus_di or 0) > (current_plus_di or 0) and current_ema20 < current_ema50

    # Choppiness index proxy: count direction changes in last 20 bars
    reversals = 0
    for i in range(2, min(21, len(closes))):
        idx = len(closes) - i
        if idx < 1:
            break
        curr_dir = 1 if closes[idx] > closes[idx - 1] else -1
        prev_dir = 1 if closes[idx - 1] > closes[idx - 2] else -1
        if curr_dir != prev_dir:
            reversals += 1
    choppiness = reversals / min(19, len(closes) - 2) if len(closes) > 2 else 0

    if is_trending and is_bullish:
        regime = "TRENDING_UP"
        description = "Clear uptrend — trend-following strategies (MACD, Supertrend, EMA Cross) recommended"
        confidence = min((current_adx or 0) / 40 + (current_er or 0), 1.0)
    elif is_trending and is_bearish:
        regime = "TRENDING_DOWN"
        description = "Clear downtrend — short strategies or stay out; trend-following on short side"
        confidence = min((current_adx or 0) / 40 + (current_er or 0), 1.0)
    elif choppiness > 0.6:
        regime = "CHOPPY"
        description = "Choppy/noisy market — most strategies will whipsaw; reduce position size or sit out"
        confidence = min(choppiness, 1.0)
    else:
        regime = "RANGING"
        description = "Range-bound market — mean-reversion strategies (RSI, Bollinger) recommended"
        confidence = 1.0 - (current_adx or 15) / 30

    confidence = round(max(0, min(confidence, 1.0)), 2)

    # ── Strategy fitness scores ──────────────────────────────────────────────
    strategy_fitness = {}
    if regime == "TRENDING_UP" or regime == "TRENDING_DOWN":
        strategy_fitness = {
            "supertrend": 0.9,
            "macd": 0.85,
            "ema_cross": 0.85,
            "donchian": 0.8,
            "rsi": 0.3,
            "bollinger": 0.3,
        }
    elif regime == "RANGING":
        strategy_fitness = {
            "rsi": 0.9,
            "bollinger": 0.85,
            "ema_cross": 0.4,
            "macd": 0.4,
            "supertrend": 0.2,
            "donchian": 0.2,
        }
    else:  # CHOPPY
        strategy_fitness = {
            "rsi": 0.3,
            "bollinger": 0.4,
            "ema_cross": 0.2,
            "macd": 0.2,
            "supertrend": 0.2,
            "donchian": 0.2,
        }

    # ── Rolling regime history ───────────────────────────────────────────────
    regime_history = []
    window = 20
    for i in range(max(30, window), len(candles), window):
        seg_closes = closes[i - window:i + 1]
        seg_er_vals = [er[j] for j in range(i - window, i + 1) if j < len(er) and er[j] is not None]
        seg_adx_vals = [adx_data["adx"][j] for j in range(i - window, i + 1) if j < len(adx_data["adx"]) and adx_data["adx"][j] is not None]

        seg_er_avg = statistics.mean(seg_er_vals) if seg_er_vals else 0
        seg_adx_avg = statistics.mean(seg_adx_vals) if seg_adx_vals else 0
        seg_return = (seg_closes[-1] - seg_closes[0]) / seg_closes[0] * 100 if seg_closes[0] != 0 else 0

        if seg_adx_avg >= adx_trend_threshold or seg_er_avg >= er_trend_threshold:
            seg_regime = "TRENDING_UP" if seg_return > 0 else "TRENDING_DOWN"
        else:
            seg_regime = "RANGING"

        regime_history.append({
            "date": candles[i]["date"],
            "regime": seg_regime,
            "adx_avg": round(seg_adx_avg, 2),
            "er_avg": round(seg_er_avg, 4),
            "return_pct": round(seg_return, 2),
        })

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "regime": regime,
        "description": description,
        "confidence": confidence,
        "indicators": {
            "adx": current_adx,
            "plus_di": current_plus_di,
            "minus_di": current_minus_di,
            "efficiency_ratio": current_er,
            "choppiness": round(choppiness, 4),
            "ema20": round(current_ema20, 4),
            "ema50": round(current_ema50, 4),
            "ema20_above_ema50": current_ema20 > current_ema50,
            "price_above_ema20": closes[-1] > current_ema20,
        },
        "strategy_fitness": strategy_fitness,
        "regime_history": regime_history[-10:],
        "disclaimer": "Regime classification is based on historical data and may change rapidly. For educational use only.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
