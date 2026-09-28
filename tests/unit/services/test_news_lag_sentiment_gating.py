"""Regression tests: the news-lag detector must not invent a sentiment verdict.

Upstream replaced the RSS/Reddit feed with Marketaux, which needs
MARKETAUX_API_TOKEN. That made "no sentiment data" the default state rather
than an edge case, and exposed a catch-all branch in the alignment logic.

Two shapes produced a confident verdict from nothing:

  * No token -> ``sentiment_label`` is "Unavailable", which matches neither the
    bullish/bearish nor the "Neutral" branch, so it fell through to the final
    ``else`` and reported DIVERGENT with a "potential reversal signal" note.
    Observed 2026-09-28 on AAPL: news_count 0, posts_analyzed 0, yet
    "Sentiment (Unavailable) diverges from 7d price trend (+0.4%)".
  * Token present but zero articles -> the score defaults to 0.0, which labels
    as "Neutral", asserting a neutral reading that was never measured.

Both must now report UNAVAILABLE. The price-derived fields stay populated.
"""
from __future__ import annotations

import pytest

from tradingview_mcp.core.services import news_lag_detector as nld

# 70 daily bars: enough to clear the 20-bar minimum and fill every momentum
# horizon (1d/3d/7d/14d/30d). Gently rising so 7d momentum is positive and
# non-zero, which is what the old code paired with the bogus sentiment label.
_CANDLES = [
    {
        "date": f"2026-01-{i + 1:02d}",
        "open": 100.0 + i,
        "high": 101.0 + i,
        "low": 99.0 + i,
        "close": 100.0 + i,
        "volume": 1_000_000,
    }
    for i in range(70)
]


def _no_token_sentiment():
    """Exact shape marketaux_service returns when MARKETAUX_API_TOKEN is unset."""
    return {
        "symbol": "AAPL", "sentiment_score": 0.0,
        "sentiment_label": "Unavailable", "posts_analyzed": 0,
        "bullish_count": 0, "bearish_count": 0, "neutral_count": 0,
        "top_posts": [], "sources": ["Marketaux news"], "provider": "marketaux",
        "error": "MARKETAUX_API_TOKEN not configured",
    }


def _empty_feed_sentiment():
    """Token present, but the feed had no articles for the symbol."""
    return {
        "symbol": "AAPL", "sentiment_score": 0.0,
        "sentiment_label": "Neutral", "posts_analyzed": 0,
        "bullish_count": 0, "bearish_count": 0, "neutral_count": 0,
        "top_posts": [], "sources": ["Marketaux news"], "provider": "marketaux",
    }


def _live_sentiment():
    return {
        "symbol": "AAPL", "sentiment_score": 0.42,
        "sentiment_label": "Bullish", "posts_analyzed": 12,
        "bullish_count": 9, "bearish_count": 1, "neutral_count": 2,
        "top_posts": [], "sources": ["Marketaux news"], "provider": "marketaux",
    }


@pytest.fixture
def patched(monkeypatch):
    """Patch out the network so only the alignment logic is under test."""
    def _apply(sentiment, news=None):
        monkeypatch.setattr(nld, "fetch_ohlcv", lambda *a, **k: list(_CANDLES))
        monkeypatch.setattr(nld, "analyze_sentiment", lambda *a, **k: sentiment)
        monkeypatch.setattr(
            nld, "fetch_news_summary",
            lambda *a, **k: news if news is not None else {"count": 0, "items": []},
        )
        return nld.detect_news_lag("AAPL")
    return _apply


class TestNoSentimentData:
    @pytest.mark.parametrize(
        "sentiment, label",
        [(_no_token_sentiment(), "missing token"), (_empty_feed_sentiment(), "empty feed")],
        ids=["no_token", "empty_feed"],
    )
    def test_alignment_is_unavailable_not_a_verdict(self, patched, sentiment, label):
        r = patched(sentiment)
        assert r["sentiment_price_alignment"] == "UNAVAILABLE", (
            f"{label}: reported a verdict from zero articles"
        )
        assert r["sentiment"]["available"] is False
        assert r["sentiment"]["label"] == "Unavailable"
        # 0.0 is indistinguishable from a real neutral reading.
        assert r["sentiment"]["score"] is None
        assert r["sentiment"]["posts_analyzed"] == 0

    @pytest.mark.parametrize(
        "sentiment",
        [_no_token_sentiment(), _empty_feed_sentiment()],
        ids=["no_token", "empty_feed"],
    )
    def test_note_never_claims_a_reversal_signal(self, patched, sentiment):
        note = patched(sentiment)["alignment_note"].lower()
        assert "reversal signal" not in note
        assert "diverge" not in note
        assert "no news sentiment available" in note
        assert "not assessed" in note

    def test_missing_token_reason_is_surfaced(self, patched):
        r = patched(_no_token_sentiment())
        assert "MARKETAUX_API_TOKEN" in r["sentiment"]["unavailable_reason"]
        assert "MARKETAUX_API_TOKEN" in r["alignment_note"]

    def test_price_fields_still_computed(self, patched):
        """Absent sentiment must not suppress the price-derived analysis."""
        r = patched(_no_token_sentiment())
        assert r["candles_analyzed"] == len(_CANDLES)
        assert r["momentum"]["7d"] is not None
        assert r["news_tradability"] in ("HIGH", "MODERATE", "LOW")

    def test_sentiment_fetch_raising_is_treated_as_unavailable(self, monkeypatch):
        """An exception leaves sentiment={} — it must not default to Neutral."""
        def _boom(*a, **k):
            raise RuntimeError("marketaux 503")
        monkeypatch.setattr(nld, "fetch_ohlcv", lambda *a, **k: list(_CANDLES))
        monkeypatch.setattr(nld, "analyze_sentiment", _boom)
        monkeypatch.setattr(nld, "fetch_news_summary", lambda *a, **k: {"count": 0, "items": []})

        r = nld.detect_news_lag("AAPL")
        assert r["sentiment_price_alignment"] == "UNAVAILABLE"
        assert r["sentiment"]["available"] is False


class TestRealSentimentStillWorks:
    def test_bullish_with_rising_price_is_aligned(self, patched):
        r = patched(_live_sentiment())
        assert r["sentiment_price_alignment"] == "ALIGNED"
        assert r["sentiment"]["available"] is True
        assert r["sentiment"]["score"] == 0.42
        assert r["sentiment"]["posts_analyzed"] == 12

    def test_bearish_against_rising_price_still_diverges(self, patched):
        """The DIVERGENT branch must survive — it is only wrong without data."""
        bearish = dict(_live_sentiment(), sentiment_label="Bearish", sentiment_score=-0.4)
        r = patched(bearish)
        assert r["sentiment_price_alignment"] == "DIVERGENT"
        assert "reversal signal" in r["alignment_note"]

    def test_measured_neutral_is_distinct_from_unavailable(self, patched):
        neutral = dict(_live_sentiment(), sentiment_label="Neutral", sentiment_score=0.0)
        r = patched(neutral)
        assert r["sentiment_price_alignment"] == "NEUTRAL"
        assert r["sentiment"]["available"] is True
        assert r["sentiment"]["score"] == 0.0
