"""
Shared OHLCV data fetcher for tradingview-mcp detector modules.

Pure Python — no pandas, no numpy.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone


_UA = "tradingview-mcp/0.7.0 detector-bot"
_YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

_VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y"}
_VALID_INTERVALS = {"1d", "1h"}


def fetch_ohlcv(symbol: str, period: str, interval: str = "1d") -> list[dict]:
    """Fetch OHLCV candles from Yahoo Finance with proxy fallback."""
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


def fetch_ohlcv_multi(symbols: list[str], period: str, interval: str = "1d") -> dict[str, list[dict]]:
    """Fetch OHLCV for multiple symbols. Returns {symbol: candles} dict."""
    result = {}
    for sym in symbols:
        sym = sym.strip().upper()
        try:
            result[sym] = fetch_ohlcv(sym, period, interval)
        except Exception:
            result[sym] = []
    return result
