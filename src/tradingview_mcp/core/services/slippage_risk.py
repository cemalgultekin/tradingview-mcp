"""
Slippage Risk Detector for tradingview-mcp

Pure Python — no pandas, no numpy.

Estimates realistic slippage for a given symbol by analyzing:
  1. Average daily volume vs typical trade sizes
  2. Bid-ask spread proxy (high-low range on low-volume bars)
  3. Volume profile — what % of bars can absorb a given trade size
  4. Impact estimation for various position sizes
  5. Comparison of realistic slippage vs the backtester's default assumption
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.data_fetcher import fetch_ohlcv


def detect_slippage_risk(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    trade_sizes_usd: Optional[list[float]] = None,
) -> dict:
    """
    Estimate realistic slippage risk for various position sizes.

    Args:
        symbol:          Yahoo Finance symbol
        period:          Historical period
        interval:        1d or 1h
        trade_sizes_usd: List of trade sizes in USD to estimate impact for
                         (default: [1000, 5000, 10000, 50000, 100000])
    """
    if trade_sizes_usd is None:
        trade_sizes_usd = [1000, 5000, 10000, 50000, 100000]

    try:
        candles = fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    if len(candles) < 20:
        return {"error": f"Not enough data ({len(candles)} bars). Need at least 20."}

    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]
    current_price = closes[-1]

    valid_bars = [(c, v) for c, v in zip(candles, volumes) if v > 0 and c["close"] > 0]
    if len(valid_bars) < 10:
        return {"error": "Insufficient volume data."}

    valid_volumes = [v for _, v in valid_bars]

    # ── Volume metrics ───────────────────────────────────────────────────────
    avg_volume = statistics.mean(valid_volumes)
    median_volume = statistics.median(valid_volumes)
    min_volume = min(valid_volumes)
    p10_volume = sorted(valid_volumes)[max(0, int(len(valid_volumes) * 0.1))]

    avg_dollar_volume = avg_volume * current_price
    median_dollar_volume = median_volume * current_price

    # ── Spread proxy (high-low range) ────────────────────────────────────────
    spreads = [(c["high"] - c["low"]) / c["close"] * 100 for c, _ in valid_bars]
    avg_spread = statistics.mean(spreads)
    min_spread = min(spreads)

    # Low-volume bar spread (proxy for thin liquidity)
    low_vol_bars = sorted(valid_bars, key=lambda x: x[1])[:max(1, len(valid_bars) // 5)]
    low_vol_spreads = [(c["high"] - c["low"]) / c["close"] * 100 for c, _ in low_vol_bars]
    thin_liquidity_spread = statistics.mean(low_vol_spreads)

    # ── Position size impact estimation ──────────────────────────────────────
    # Simple market impact model: slippage ≈ spread/2 + k * (trade_size / avg_dollar_volume)
    # where k is a market impact constant (empirically ~0.1 for liquid, ~0.5 for illiquid)

    if avg_dollar_volume > 100_000_000:
        impact_k = 0.05  # Very liquid
        liquidity_tier = "VERY HIGH"
    elif avg_dollar_volume > 10_000_000:
        impact_k = 0.1
        liquidity_tier = "HIGH"
    elif avg_dollar_volume > 1_000_000:
        impact_k = 0.2
        liquidity_tier = "MODERATE"
    elif avg_dollar_volume > 100_000:
        impact_k = 0.4
        liquidity_tier = "LOW"
    else:
        impact_k = 0.8
        liquidity_tier = "VERY LOW"

    half_spread = avg_spread / 2

    size_analysis = []
    for size in trade_sizes_usd:
        participation_rate = size / avg_dollar_volume if avg_dollar_volume > 0 else float("inf")
        market_impact = impact_k * math.sqrt(participation_rate) * 100 if participation_rate < 1 else 99.0
        estimated_slippage = half_spread + market_impact
        bars_to_fill = max(1, math.ceil(size / (median_dollar_volume * 0.1))) if median_dollar_volume > 0 else float("inf")

        # What % of bars could absorb this trade as < 10% of volume?
        trade_shares = size / current_price if current_price > 0 else 0
        absorbable_pct = sum(1 for v in valid_volumes if trade_shares < v * 0.1) / len(valid_volumes) * 100

        # Compare to backtester's default
        backtest_default = 0.05  # Default slippage_pct in backtester
        underestimation_factor = round(estimated_slippage / backtest_default, 1) if backtest_default > 0 else 0

        size_analysis.append({
            "trade_size_usd": size,
            "participation_rate_pct": round(participation_rate * 100, 4),
            "estimated_slippage_pct": round(estimated_slippage, 4),
            "market_impact_pct": round(market_impact, 4),
            "bars_to_fill": bars_to_fill if bars_to_fill != float("inf") else "N/A",
            "absorbable_bars_pct": round(absorbable_pct, 1),
            "vs_backtest_default": f"{underestimation_factor}x" if underestimation_factor > 1 else "within default",
            "feasible": participation_rate < 0.5,
        })

    # ── Risk assessment ──────────────────────────────────────────────────────
    # For a $10k trade (typical retail)
    retail_entry = next((s for s in size_analysis if s["trade_size_usd"] == 10000), size_analysis[0])
    retail_slippage = retail_entry["estimated_slippage_pct"]

    if retail_slippage > 2.0:
        risk = "CRITICAL"
        verdict = "Extremely high slippage risk — backtester results are unreliable for this asset"
    elif retail_slippage > 0.5:
        risk = "HIGH"
        verdict = "Significant slippage — adjust backtester slippage_pct upward for realistic results"
    elif retail_slippage > 0.15:
        risk = "MODERATE"
        verdict = "Moderate slippage — default backtester settings may be slightly optimistic"
    else:
        risk = "LOW"
        verdict = "Low slippage risk — default backtester slippage settings are reasonable"

    return {
        "symbol": symbol.upper(),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(candles),
        "current_price": current_price,
        "liquidity_tier": liquidity_tier,
        "risk": risk,
        "verdict": verdict,
        "volume_metrics": {
            "avg_daily_volume": round(avg_volume, 0),
            "median_daily_volume": round(median_volume, 0),
            "avg_dollar_volume": round(avg_dollar_volume, 0),
            "median_dollar_volume": round(median_dollar_volume, 0),
            "min_daily_volume": round(min_volume, 0),
            "p10_daily_volume": round(p10_volume, 0),
        },
        "spread_metrics": {
            "avg_spread_pct": round(avg_spread, 4),
            "min_spread_pct": round(min_spread, 4),
            "thin_liquidity_spread_pct": round(thin_liquidity_spread, 4),
        },
        "position_size_analysis": size_analysis,
        "recommended_backtest_slippage_pct": round(max(retail_slippage, 0.05), 4),
        "disclaimer": "Slippage estimates are modeled approximations. Actual slippage depends on order type, "
                      "market conditions, and exchange. For educational use only.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
