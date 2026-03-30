"""
Security Checks — Rug Pull Detection for tradingview-mcp

Pure Python — no pandas, no numpy.

Detects potential rug pull indicators by analyzing:
  - Sudden price crashes with abnormal volume spikes
  - Liquidity health (bid-ask spread proxies, volume consistency)
  - Whale activity (abnormal single-candle volume)
  - Pump-and-dump patterns (rapid rise followed by sharp decline)
"""
from __future__ import annotations

import json
import math
import statistics
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.indicators_calc import calc_atr, calc_rsi, calc_sma


_UA = "tradingview-mcp/0.7.0 security-bot"
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


# ─── Rug Pull Detection ──────────────────────────────────────────────────────

def detect_rug_pull(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    volume_spike_threshold: float = 3.0,
    price_crash_threshold: float = -15.0,
    pump_dump_rise_pct: float = 50.0,
    pump_dump_window: int = 14,
) -> dict:
    """
    Detect potential rug pull indicators for a given symbol.

    Checks for:
      1. Crash events:     Single-candle drops > price_crash_threshold with volume spike
      2. Whale activity:   Candles with volume > volume_spike_threshold × avg volume
      3. Pump-and-dump:    Rapid price rise (>pump_dump_rise_pct) within pump_dump_window
                           candles followed by a crash
      4. Liquidity health: Volume consistency, average spread, volume trend

    Returns a risk assessment with severity: CRITICAL / HIGH / MODERATE / LOW
    """
    try:
        candles = _fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < 30:
        return {"error": f"Not enough data ({len(candles)} bars). Try a longer period."}

    volumes = [c["volume"] for c in candles]
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    vol_sma = calc_sma(volumes, 20)
    rsi = calc_rsi(closes, 14)

    # ── 1. Crash Events ──────────────────────────────────────────────────────
    crash_events = []
    for i in range(1, len(candles)):
        if candles[i - 1]["close"] == 0:
            continue
        pct_change = (candles[i]["close"] - candles[i - 1]["close"]) / candles[i - 1]["close"] * 100
        vol_ratio = volumes[i] / vol_sma[i] if vol_sma[i] and vol_sma[i] > 0 else 0

        if pct_change <= price_crash_threshold and vol_ratio >= volume_spike_threshold:
            crash_events.append({
                "date": candles[i]["date"],
                "price_change_pct": round(pct_change, 2),
                "volume_ratio": round(vol_ratio, 2),
                "close": candles[i]["close"],
                "rsi": round(rsi[i], 1) if rsi[i] is not None else None,
            })

    # ── 2. Whale Activity (abnormal volume spikes) ───────────────────────────
    whale_events = []
    for i in range(20, len(candles)):
        if vol_sma[i] is None or vol_sma[i] == 0:
            continue
        vol_ratio = volumes[i] / vol_sma[i]
        if vol_ratio >= volume_spike_threshold * 1.5:
            pct_change = 0.0
            if candles[i - 1]["close"] != 0:
                pct_change = (candles[i]["close"] - candles[i - 1]["close"]) / candles[i - 1]["close"] * 100
            whale_events.append({
                "date": candles[i]["date"],
                "volume_ratio": round(vol_ratio, 2),
                "price_change_pct": round(pct_change, 2),
                "direction": "dump" if pct_change < -3 else ("pump" if pct_change > 3 else "neutral"),
            })

    # ── 3. Pump-and-Dump Detection ───────────────────────────────────────────
    pump_dump_events = []
    for i in range(pump_dump_window, len(candles)):
        window_start = candles[i - pump_dump_window]["close"]
        if window_start == 0:
            continue

        # Find peak within window
        peak_price = max(c["close"] for c in candles[i - pump_dump_window:i + 1])
        peak_idx = None
        for j in range(i - pump_dump_window, i + 1):
            if candles[j]["close"] == peak_price:
                peak_idx = j
                break

        if peak_idx is None:
            continue

        rise_pct = (peak_price - window_start) / window_start * 100
        if rise_pct < pump_dump_rise_pct:
            continue

        # Check for crash after peak
        if peak_idx < len(candles) - 1:
            drop_from_peak = (candles[i]["close"] - peak_price) / peak_price * 100
            if drop_from_peak <= price_crash_threshold:
                pump_dump_events.append({
                    "pump_start": candles[i - pump_dump_window]["date"],
                    "peak_date": candles[peak_idx]["date"],
                    "dump_date": candles[i]["date"],
                    "rise_pct": round(rise_pct, 2),
                    "drop_from_peak_pct": round(drop_from_peak, 2),
                    "peak_price": peak_price,
                    "current_price": candles[i]["close"],
                })

    # ── 4. Liquidity Health ──────────────────────────────────────────────────
    valid_volumes = [v for v in volumes if v > 0]
    avg_volume = statistics.mean(valid_volumes) if valid_volumes else 0
    vol_std = statistics.stdev(valid_volumes) if len(valid_volumes) > 1 else 0
    vol_cv = round(vol_std / avg_volume, 2) if avg_volume > 0 else 0

    # Volume trend: compare last 20% of volume to first 20%
    seg = max(1, len(valid_volumes) // 5)
    early_vol = statistics.mean(valid_volumes[:seg]) if valid_volumes[:seg] else 0
    late_vol = statistics.mean(valid_volumes[-seg:]) if valid_volumes[-seg:] else 0
    vol_trend_pct = round((late_vol - early_vol) / early_vol * 100, 2) if early_vol > 0 else 0

    # Average candle spread as liquidity proxy
    spreads = [(c["high"] - c["low"]) / c["close"] * 100 for c in candles if c["close"] > 0]
    avg_spread = round(statistics.mean(spreads), 2) if spreads else 0

    # Zero-volume days ratio
    zero_vol_ratio = round(sum(1 for v in volumes if v == 0) / len(volumes) * 100, 2)

    liquidity = {
        "avg_daily_volume": round(avg_volume, 0),
        "volume_coefficient_of_variation": vol_cv,
        "volume_trend_pct": vol_trend_pct,
        "avg_spread_pct": avg_spread,
        "zero_volume_days_pct": zero_vol_ratio,
        "health": (
            "CRITICAL" if zero_vol_ratio > 20 or vol_trend_pct < -70 else
            "POOR" if zero_vol_ratio > 10 or vol_trend_pct < -50 or vol_cv > 2.0 else
            "FAIR" if vol_cv > 1.5 or vol_trend_pct < -30 else
            "GOOD"
        ),
    }

    # ── Risk Score Calculation ───────────────────────────────────────────────
    risk_score = 0
    risk_flags = []

    if crash_events:
        risk_score += min(len(crash_events) * 25, 50)
        risk_flags.append(f"{len(crash_events)} crash event(s) with volume spike detected")

    if whale_events:
        dumps = [e for e in whale_events if e["direction"] == "dump"]
        risk_score += min(len(dumps) * 15, 30)
        if dumps:
            risk_flags.append(f"{len(dumps)} whale dump(s) detected")

    if pump_dump_events:
        risk_score += min(len(pump_dump_events) * 30, 60)
        risk_flags.append(f"{len(pump_dump_events)} pump-and-dump pattern(s) detected")

    if liquidity["health"] == "CRITICAL":
        risk_score += 30
        risk_flags.append("Critical liquidity issues")
    elif liquidity["health"] == "POOR":
        risk_score += 15
        risk_flags.append("Poor liquidity health")

    if vol_trend_pct < -50:
        risk_score += 20
        risk_flags.append(f"Volume declining sharply ({vol_trend_pct}%)")

    risk_score = min(risk_score, 100)

    if risk_score >= 70:
        severity = "CRITICAL"
        recommendation = "HIGH RISK — multiple rug pull indicators detected. Avoid or exit position."
    elif risk_score >= 45:
        severity = "HIGH"
        recommendation = "Elevated risk — suspicious patterns found. Exercise extreme caution."
    elif risk_score >= 20:
        severity = "MODERATE"
        recommendation = "Some warning signs present. Monitor closely and use stop-losses."
    else:
        severity = "LOW"
        recommendation = "No significant rug pull indicators detected at this time."

    # Price context
    current_price = closes[-1]
    ath = max(closes)
    atl = min(c for c in closes if c > 0)
    from_ath_pct = round((current_price - ath) / ath * 100, 2) if ath > 0 else 0

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "current_price": current_price,
        "all_time_high": ath,
        "from_ath_pct": from_ath_pct,
        "risk_score": risk_score,
        "severity": severity,
        "recommendation": recommendation,
        "risk_flags": risk_flags,
        "crash_events": crash_events[:10],
        "whale_activity": whale_events[:10],
        "pump_dump_patterns": pump_dump_events[:5],
        "liquidity": liquidity,
        "disclaimer": "This is a heuristic analysis tool for educational purposes only. "
                      "Not financial advice. Always do your own research.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
