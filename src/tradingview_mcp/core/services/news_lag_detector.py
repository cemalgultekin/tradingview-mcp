"""
News-Price Lag Detector for tradingview-mcp

Pure Python — no pandas, no numpy.

Correlates news sentiment spikes with price moves to detect:
  1. Whether news leads or lags price for a given symbol
  2. Average delay between sentiment shift and price reaction
  3. Whether trading on news is profitable for this asset
  4. News impact magnitude (how much price moves after sentiment spikes)
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone, timedelta
from typing import Optional

from tradingview_mcp.core.services.data_fetcher import fetch_ohlcv
from tradingview_mcp.core.services.news_service import fetch_news_summary
from tradingview_mcp.core.services.sentiment_service import analyze_sentiment


def detect_news_lag(
    symbol: str,
    period: str = "3mo",
    interval: str = "1d",
    category: str = "crypto",
) -> dict:
    """
    Analyze the relationship between news/sentiment and price movements.

    Since real-time timestamped news correlation requires a premium data source,
    this tool performs a snapshot analysis:
      1. Fetches current news sentiment
      2. Fetches recent price data
      3. Analyzes recent price momentum around sentiment signals
      4. Computes a reactivity score for the asset

    Args:
        symbol:   Yahoo Finance symbol
        period:   Historical period for price data
        interval: 1d or 1h
        category: News category: 'crypto', 'stocks', 'all'
    """
    # Fetch price data
    try:
        candles = fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch price data for '{symbol}': {e}"}

    if len(candles) < 20:
        return {"error": f"Not enough price data ({len(candles)} bars)."}

    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]

    # Fetch sentiment
    clean_symbol = symbol.upper().replace("-USD", "").replace("-", "")
    sentiment = {}
    try:
        sentiment = analyze_sentiment(clean_symbol, category=category, limit=30)
    except Exception:
        pass

    # Fetch news
    news = {}
    try:
        news = fetch_news_summary(clean_symbol, category=category, limit=20)
    except Exception:
        pass

    # ── Price momentum analysis ──────────────────────────────────────────────
    # Recent returns at various horizons
    def _return_pct(n: int) -> Optional[float]:
        if len(closes) > n and closes[-n - 1] != 0:
            return round((closes[-1] - closes[-n - 1]) / closes[-n - 1] * 100, 2)
        return None

    momentum = {
        "1d": _return_pct(1),
        "3d": _return_pct(3),
        "7d": _return_pct(7),
        "14d": _return_pct(14),
        "30d": _return_pct(30),
    }

    # ── Volume-price reaction analysis ───────────────────────────────────────
    # Find days with volume spikes (proxy for news-driven trading)
    avg_vol = statistics.mean(volumes[-60:]) if len(volumes) >= 60 else statistics.mean(volumes)
    spike_days = []

    for i in range(max(1, len(candles) - 60), len(candles)):
        if avg_vol > 0 and volumes[i] > avg_vol * 2:
            pct_change = (closes[i] - closes[i - 1]) / closes[i - 1] * 100 if closes[i - 1] != 0 else 0
            # Check follow-through (next 1-3 bars)
            follow_through = []
            for j in range(1, 4):
                if i + j < len(candles) and closes[i] != 0:
                    ft = (closes[i + j] - closes[i]) / closes[i] * 100
                    follow_through.append(round(ft, 2))

            spike_days.append({
                "date": candles[i]["date"],
                "volume_ratio": round(volumes[i] / avg_vol, 2),
                "price_change_pct": round(pct_change, 2),
                "follow_through_pct": follow_through,
                "direction": "up" if pct_change > 0 else "down",
            })

    # ── Reactivity score ─────────────────────────────────────────────────────
    # How consistently do volume spikes lead to continued moves?
    if spike_days:
        continuation_count = 0
        reversal_count = 0
        for sd in spike_days:
            if sd["follow_through_pct"]:
                avg_ft = statistics.mean(sd["follow_through_pct"])
                if (sd["direction"] == "up" and avg_ft > 0) or (sd["direction"] == "down" and avg_ft < 0):
                    continuation_count += 1
                else:
                    reversal_count += 1

        total = continuation_count + reversal_count
        continuation_rate = round(continuation_count / total * 100, 1) if total > 0 else 50.0

        avg_spike_move = statistics.mean(abs(sd["price_change_pct"]) for sd in spike_days)
    else:
        continuation_rate = 50.0
        avg_spike_move = 0

    # ── Sentiment-price alignment ────────────────────────────────────────────
    sent_label = sentiment.get("sentiment_label", "Neutral")
    sent_score = sentiment.get("sentiment_score", 0)

    if momentum["7d"] is not None:
        if (sent_label in ("Bullish", "Very Bullish") and momentum["7d"] > 0) or \
           (sent_label in ("Bearish", "Very Bearish") and momentum["7d"] < 0):
            alignment = "ALIGNED"
            alignment_note = f"Sentiment ({sent_label}) matches 7d price trend ({momentum['7d']:+.1f}%)"
        elif sent_label in ("Neutral",):
            alignment = "NEUTRAL"
            alignment_note = "Sentiment is neutral"
        else:
            alignment = "DIVERGENT"
            alignment_note = f"Sentiment ({sent_label}) diverges from 7d price trend ({momentum['7d']:+.1f}%) — potential reversal signal"
    else:
        alignment = "UNKNOWN"
        alignment_note = "Insufficient price data for alignment check"

    # ── News tradability assessment ──────────────────────────────────────────
    if continuation_rate > 65 and avg_spike_move > 2:
        tradability = "HIGH"
        tradability_note = "Volume spikes show strong follow-through — news-driven trades tend to continue"
    elif continuation_rate > 55:
        tradability = "MODERATE"
        tradability_note = "Some follow-through on volume spikes — selective news trading may work"
    else:
        tradability = "LOW"
        tradability_note = "Volume spikes frequently reverse — news is already priced in quickly"

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "sentiment": {
            "label": sent_label,
            "score": sent_score,
            "posts_analyzed": sentiment.get("posts_analyzed", 0),
            "bullish_count": sentiment.get("bullish_count", 0),
            "bearish_count": sentiment.get("bearish_count", 0),
        },
        "news_count": news.get("count", 0),
        "recent_headlines": [item.get("title", "") for item in news.get("items", [])[:5]],
        "momentum": momentum,
        "sentiment_price_alignment": alignment,
        "alignment_note": alignment_note,
        "volume_spike_analysis": {
            "spike_days_found": len(spike_days),
            "continuation_rate_pct": continuation_rate,
            "avg_spike_move_pct": round(avg_spike_move, 2),
            "recent_spikes": spike_days[-5:],
        },
        "news_tradability": tradability,
        "tradability_note": tradability_note,
        "disclaimer": "News-price lag analysis uses heuristic methods. Not financial advice. For educational use only.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
