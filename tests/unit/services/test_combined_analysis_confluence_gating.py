"""Regression tests: combined_analysis must not claim confluence with absent sentiment.

Fork patch. ``sent_bullish = sentiment.get("sentiment_score", 0) > 0.1`` reads
0.0 when no sentiment was fetched, so ``sent_bullish`` is False. Paired with a
non-bullish technical read that made ``False == False`` -> ``signals_agree``
True -> ``confidence`` "HIGH" — a high-confidence agreement between technicals
and sentiment that was never obtained.

Three inputs reach that state, all with ``posts_analyzed == 0``: no
MARKETAUX_API_TOKEN, no articles for the symbol, and the error envelope
substituted when the sentiment fetch raises.

The bug needs a non-bullish technical to bite, which is why the existing
fan-out test in test_async_handlers.py never caught it — it feeds a Bullish
technical alongside 12 analysed articles.
"""
from __future__ import annotations

import pytest

from tradingview_mcp import server

_BEARISH_TECH = {"market_sentiment": {"momentum": "Bearish", "buy_sell_signal": "SELL"}}
_BULLISH_TECH = {"market_sentiment": {"momentum": "Bullish", "buy_sell_signal": "BUY"}}
_NEWS = {"count": 0, "items": []}


def _no_token_sentiment():
    return {
        "symbol": "AAPL", "sentiment_score": 0.0, "sentiment_label": "Unavailable",
        "posts_analyzed": 0, "bullish_count": 0, "bearish_count": 0,
        "error": "MARKETAUX_API_TOKEN not configured",
    }


def _empty_feed_sentiment():
    return {
        "symbol": "AAPL", "sentiment_score": 0.0, "sentiment_label": "Neutral",
        "posts_analyzed": 0, "bullish_count": 0, "bearish_count": 0,
    }


def _raised_envelope():
    """Shape exception_to_envelope leaves behind when the fetch raises."""
    return {"error": {"code": "UPSTREAM_ERROR", "message": "marketaux 503", "retryable": True}}


@pytest.fixture
def run(monkeypatch):
    def _apply(tech, sentiment, news=None):
        monkeypatch.setattr(server, "analyze_coin", lambda *a, **k: tech)
        monkeypatch.setattr(server, "analyze_sentiment", lambda *a, **k: sentiment)
        monkeypatch.setattr(server, "fetch_news_summary", lambda *a, **k: news or _NEWS)
        return server.combined_analysis("AAPL", exchange="NASDAQ", timeframe="1D")
    return _apply


class TestNoSentimentData:
    @pytest.mark.parametrize(
        "sentiment",
        [_no_token_sentiment(), _empty_feed_sentiment(), _raised_envelope()],
        ids=["no_token", "empty_feed", "fetch_raised"],
    )
    @pytest.mark.asyncio
    async def test_no_high_confidence_without_sentiment(self, run, sentiment):
        """The exact bug: bearish tech + absent sentiment used to score HIGH."""
        r = await run(_BEARISH_TECH, sentiment)
        c = r["confluence"]

        assert c["confidence"] != "HIGH", "claimed HIGH confidence from zero articles"
        assert c["confidence"] == "TECHNICAL_ONLY"
        assert c["signals_agree"] is None, "asserted agreement with sentiment that was never fetched"
        assert c["sentiment_available"] is False

    @pytest.mark.parametrize(
        "tech", [_BEARISH_TECH, _BULLISH_TECH], ids=["bearish", "bullish"],
    )
    @pytest.mark.asyncio
    async def test_gating_holds_for_either_technical_direction(self, run, tech):
        c = (await run(tech, _no_token_sentiment()))["confluence"]
        assert c["signals_agree"] is None
        assert c["confidence"] == "TECHNICAL_ONLY"

    @pytest.mark.asyncio
    async def test_recommendation_does_not_claim_confirmation(self, run):
        rec = (await run(_BEARISH_TECH, _no_token_sentiment()))["confluence"]["recommendation"].lower()
        assert "confirmed by" not in rec
        assert "conflicts with" not in rec
        assert "no news sentiment available" in rec

    @pytest.mark.asyncio
    async def test_technical_section_still_returned(self, run):
        """Gating confluence must not suppress the analysis that did succeed."""
        r = await run(_BEARISH_TECH, _no_token_sentiment())
        assert r["technical"]["market_sentiment"]["buy_sell_signal"] == "SELL"
        assert "SELL" in r["confluence"]["recommendation"]


class TestMeasuredSentimentUnchanged:
    @pytest.mark.asyncio
    async def test_agreement_still_scores_high(self, run):
        s = {"sentiment_score": 0.3, "sentiment_label": "Bullish", "posts_analyzed": 12}
        c = (await run(_BULLISH_TECH, s))["confluence"]
        assert c["signals_agree"] is True
        assert c["confidence"] == "HIGH"
        assert c["sentiment_available"] is True
        assert "confirmed by" in c["recommendation"]

    @pytest.mark.asyncio
    async def test_disagreement_still_scores_mixed(self, run):
        s = {"sentiment_score": 0.3, "sentiment_label": "Bullish", "posts_analyzed": 12}
        c = (await run(_BEARISH_TECH, s))["confluence"]
        assert c["signals_agree"] is False
        assert c["confidence"] == "MIXED"
        assert "conflicts with" in c["recommendation"]

    @pytest.mark.asyncio
    async def test_measured_neutral_is_not_treated_as_missing(self, run):
        """A real 0.0 reading from analysed articles is data, not absence."""
        s = {"sentiment_score": 0.0, "sentiment_label": "Neutral", "posts_analyzed": 8}
        c = (await run(_BEARISH_TECH, s))["confluence"]
        assert c["sentiment_available"] is True
        assert c["signals_agree"] is True          # bearish tech, non-bullish sentiment
        assert c["confidence"] == "HIGH"
