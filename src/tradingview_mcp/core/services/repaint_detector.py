"""
Repaint Detector — Signal Stability Validation for tradingview-mcp

Pure Python — no pandas, no numpy.

Detects indicator repainting by:
  1. Simulating real-time bar-by-bar signal generation (only seeing data up to bar N)
  2. Comparing those "live" signals to signals computed with full hindsight
  3. Measuring how often signals flip, appear late, or vanish

A strategy that repaints produces signals in backtest that could never have been
acted on in real-time — making backtest results unreliable.
"""
from __future__ import annotations

import json
import math
import statistics
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.indicators_calc import (
    calc_rsi, calc_bollinger, calc_macd, calc_ema, calc_supertrend, calc_donchian,
)


_UA = "tradingview-mcp/0.7.0 repaint-detector"
_YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"


# ─── Data Fetching ────────────────────────────────────────────────────────────

def _fetch_ohlcv(symbol: str, period: str, interval: str = "1d") -> list[dict]:
    url = f"{_YF_BASE}/{symbol}?interval={interval}&range={period}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})

    data = None
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass

    if data is None:
        try:
            from tradingview_mcp.core.services.proxy_manager import build_opener_with_proxy
            opener = build_opener_with_proxy(_UA)
            with opener.open(url, timeout=18) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Both direct and proxy connections failed: {e}")

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    q = result["indicators"]["quote"][0]
    date_fmt = "%Y-%m-%d %H:%M" if interval in ("1h", "5m", "15m") else "%Y-%m-%d"

    candles = []
    for i, ts in enumerate(timestamps):
        o, h, l, c, v = q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i]
        if None in (o, h, l, c):
            continue
        candles.append({
            "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime(date_fmt),
            "open": round(o, 4),
            "high": round(h, 4),
            "low": round(l, 4),
            "close": round(c, 4),
            "volume": v or 0,
        })
    return candles


# ─── Signal Generators (return +1 buy, -1 sell, 0 neutral per bar) ───────────

def _signals_rsi(candles: list[dict], oversold=40, overbought=60, period=14) -> list[int]:
    closes = [c["close"] for c in candles]
    rsi = calc_rsi(closes, period)
    signals = [0] * len(candles)
    for i in range(len(candles)):
        if rsi[i] is None:
            continue
        if rsi[i] < oversold:
            signals[i] = 1
        elif rsi[i] > overbought:
            signals[i] = -1
    return signals


def _signals_bollinger(candles: list[dict], period=20, std_mult=2.0) -> list[int]:
    closes = [c["close"] for c in candles]
    bb = calc_bollinger(closes, period, std_mult)
    signals = [0] * len(candles)
    for i in range(len(candles)):
        if bb["lower"][i] is None:
            continue
        if closes[i] < bb["lower"][i]:
            signals[i] = 1
        elif closes[i] > bb["middle"][i]:
            signals[i] = -1
    return signals


def _signals_macd(candles: list[dict], fast=12, slow=26, signal=9) -> list[int]:
    closes = [c["close"] for c in candles]
    macd = calc_macd(closes, fast, slow, signal)
    signals = [0] * len(candles)
    for i in range(1, len(candles)):
        m, s = macd["macd"][i], macd["signal"][i]
        mp, sp = macd["macd"][i - 1], macd["signal"][i - 1]
        if None in (m, s, mp, sp):
            continue
        if mp < sp and m >= s:
            signals[i] = 1
        elif mp > sp and m <= s:
            signals[i] = -1
    return signals


def _signals_ema_cross(candles: list[dict], fast_period=20, slow_period=50) -> list[int]:
    closes = [c["close"] for c in candles]
    ema_fast = calc_ema(closes, fast_period)
    ema_slow = calc_ema(closes, slow_period)
    signals = [0] * len(candles)
    for i in range(1, len(candles)):
        f, s = ema_fast[i], ema_slow[i]
        fp, sp = ema_fast[i - 1], ema_slow[i - 1]
        if None in (f, s, fp, sp):
            continue
        if fp < sp and f >= s:
            signals[i] = 1
        elif fp > sp and f <= s:
            signals[i] = -1
    return signals


def _signals_supertrend(candles: list[dict], atr_period=10, multiplier=3.0) -> list[int]:
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    st = calc_supertrend(highs, lows, closes, atr_period, multiplier)
    signals = [0] * len(candles)
    for i in range(1, len(candles)):
        d, dp = st["direction"][i], st["direction"][i - 1]
        if d is None or dp is None:
            continue
        if dp == -1 and d == 1:
            signals[i] = 1
        elif dp == 1 and d == -1:
            signals[i] = -1
    return signals


def _signals_donchian(candles: list[dict], period=20) -> list[int]:
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    dc = calc_donchian(highs, lows, period)
    signals = [0] * len(candles)
    for i in range(1, len(candles)):
        if dc["upper"][i] is None or dc["upper"][i - 1] is None:
            continue
        if highs[i - 1] > dc["upper"][i - 1]:
            signals[i] = 1
        elif dc["lower"][i] is not None and closes[i] < dc["lower"][i]:
            signals[i] = -1
    return signals


_SIGNAL_MAP = {
    "rsi": _signals_rsi,
    "bollinger": _signals_bollinger,
    "macd": _signals_macd,
    "ema_cross": _signals_ema_cross,
    "supertrend": _signals_supertrend,
    "donchian": _signals_donchian,
}

_STRATEGY_LABELS = {
    "rsi":        "RSI Oversold/Overbought",
    "bollinger":  "Bollinger Band Mean Reversion",
    "macd":       "MACD Crossover",
    "ema_cross":  "EMA 20/50 Golden/Death Cross",
    "supertrend": "Supertrend (ATR-based Trend Following)",
    "donchian":   "Donchian Channel Breakout",
}


# ─── Repaint Detection Engine ────────────────────────────────────────────────

def detect_repaint(
    symbol: str,
    strategy: str,
    period: str = "1y",
    interval: str = "1d",
    warmup: int = 60,
) -> dict:
    """
    Detect signal repainting by comparing incremental vs hindsight signals.

    For each bar from `warmup` onward:
      - Compute signals using data up to bar N only (simulates live trading)
      - Compute signals using full dataset (hindsight/backtest view)
      - Compare: if signal at bar N differs, that's a repaint

    Args:
        symbol:   Yahoo Finance symbol
        strategy: rsi | bollinger | macd | ema_cross | supertrend | donchian
        period:   Historical data period
        interval: 1d or 1h
        warmup:   Minimum bars before starting comparison (indicator warm-up)

    Returns:
        Repaint analysis: stability score, repaint frequency, affected bars, verdict
    """
    strategy = strategy.lower().strip()
    if strategy not in _SIGNAL_MAP:
        return {"error": f"Unknown strategy '{strategy}'. Choose: {', '.join(_SIGNAL_MAP)}"}

    try:
        candles = _fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < warmup + 20:
        return {"error": f"Not enough data ({len(candles)} bars). Need at least {warmup + 20}."}

    sig_fn = _SIGNAL_MAP[strategy]

    # Full-hindsight signals (computed once with all data)
    hindsight_signals = sig_fn(candles)

    # Incremental signals (simulate bar-by-bar)
    incremental_signals = [0] * len(candles)
    for n in range(warmup, len(candles)):
        partial = candles[:n + 1]
        partial_sigs = sig_fn(partial)
        incremental_signals[n] = partial_sigs[n]

    # Compare signals from warmup onward
    total_compared = 0
    repaints = 0
    repaint_details = []
    signal_flips = 0
    phantom_signals = 0  # signal exists in hindsight but not incrementally
    late_signals = 0      # signal exists incrementally but not in hindsight

    for i in range(warmup, len(candles)):
        inc = incremental_signals[i]
        hind = hindsight_signals[i]

        # Only compare bars where at least one produced a non-neutral signal
        if inc == 0 and hind == 0:
            continue

        total_compared += 1

        if inc != hind:
            repaints += 1
            if inc == 0 and hind != 0:
                phantom_signals += 1
                kind = "phantom"
            elif inc != 0 and hind == 0:
                late_signals += 1
                kind = "vanished"
            else:
                signal_flips += 1
                kind = "flipped"

            if len(repaint_details) < 20:
                repaint_details.append({
                    "bar": i,
                    "date": candles[i]["date"],
                    "incremental_signal": _sig_label(inc),
                    "hindsight_signal": _sig_label(hind),
                    "type": kind,
                    "close": candles[i]["close"],
                })

    # Stability score: 1.0 = perfect (no repaints), 0.0 = all signals repaint
    if total_compared == 0:
        stability_score = 1.0
        repaint_rate = 0.0
    else:
        repaint_rate = round(repaints / total_compared * 100, 2)
        stability_score = round(1.0 - (repaints / total_compared), 4)

    # Verdict
    if stability_score >= 0.95:
        verdict = "CLEAN — signals are stable and do not repaint"
        risk = "LOW"
    elif stability_score >= 0.85:
        verdict = "MOSTLY CLEAN — minor signal instability detected, acceptable for most use cases"
        risk = "LOW"
    elif stability_score >= 0.70:
        verdict = "MODERATE REPAINT — some signals change with new data, use caution in live trading"
        risk = "MODERATE"
    elif stability_score >= 0.50:
        verdict = "SIGNIFICANT REPAINT — many signals are unreliable, backtest results inflated"
        risk = "HIGH"
    else:
        verdict = "SEVERE REPAINT — strategy signals are fundamentally unstable, do not trust backtest"
        risk = "CRITICAL"

    # Count total buy/sell signals
    total_buy_hindsight = sum(1 for s in hindsight_signals[warmup:] if s == 1)
    total_sell_hindsight = sum(1 for s in hindsight_signals[warmup:] if s == -1)
    total_buy_incremental = sum(1 for s in incremental_signals[warmup:] if s == 1)
    total_sell_incremental = sum(1 for s in incremental_signals[warmup:] if s == -1)

    return {
        "symbol": symbol.upper(),
        "strategy": strategy,
        "strategy_label": _STRATEGY_LABELS.get(strategy, strategy),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "bars_compared": total_compared,
        "warmup_bars": warmup,
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "stability_score": stability_score,
        "repaint_rate_pct": repaint_rate,
        "total_repaints": repaints,
        "phantom_signals": phantom_signals,
        "vanished_signals": late_signals,
        "flipped_signals": signal_flips,
        "risk": risk,
        "verdict": verdict,
        "signal_counts": {
            "hindsight_buys": total_buy_hindsight,
            "hindsight_sells": total_sell_hindsight,
            "incremental_buys": total_buy_incremental,
            "incremental_sells": total_sell_incremental,
        },
        "repaint_details": repaint_details,
        "disclaimer": "Repaint detection simulates bar-by-bar signal generation. "
                      "Some minor differences are normal due to indicator warm-up. "
                      "For educational use only.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _sig_label(sig: int) -> str:
    if sig == 1:
        return "BUY"
    elif sig == -1:
        return "SELL"
    return "NEUTRAL"
