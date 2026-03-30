"""
Cross-Exchange Arbitrage Detector for tradingview-mcp

Pure Python — no pandas, no numpy.

Compares the same asset across different exchanges to detect:
  1. Price discrepancies between exchanges
  2. Historical spread analysis
  3. Arbitrage opportunity windows
  4. Exchange premium/discount persistence
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.data_fetcher import fetch_ohlcv


# Common crypto pairs mapped to Yahoo Finance symbols per exchange
# Yahoo Finance uses {COIN}-USD for most crypto
# For exchange-specific, we compare TradingView symbols via screener
_CRYPTO_YAHOO_PAIRS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
    "ADA": "ADA-USD",
    "DOGE": "DOGE-USD",
    "AVAX": "AVAX-USD",
    "DOT": "DOT-USD",
    "MATIC": "MATIC-USD",
    "LINK": "LINK-USD",
    "UNI": "UNI-USD",
    "ATOM": "ATOM-USD",
    "LTC": "LTC-USD",
}


def detect_arbitrage(
    base_symbol: str,
    compare_symbols: Optional[list[str]] = None,
    period: str = "1mo",
    interval: str = "1d",
) -> dict:
    """
    Detect cross-exchange arbitrage opportunities by comparing price feeds.

    Since Yahoo Finance provides composite prices, this tool compares:
    - A symbol against its stablecoin pairs (e.g., BTC-USD vs BTC-EUR converted)
    - A crypto token against related ETFs or trusts
    - User-specified symbol pairs that should track each other

    Args:
        base_symbol:     Primary Yahoo Finance symbol (e.g., 'BTC-USD')
        compare_symbols: Symbols to compare against (e.g., ['GBTC', 'BITO', 'BTC-EUR'])
                         If None, auto-selects based on base_symbol
        period:          Historical period
        interval:        1d or 1h
    """
    # Auto-select comparison pairs if not provided
    if compare_symbols is None:
        base_upper = base_symbol.upper().replace("-USD", "")
        if base_upper == "BTC":
            compare_symbols = ["GBTC", "BITO", "IBIT"]
        elif base_upper == "ETH":
            compare_symbols = ["ETHE", "ETHA"]
        else:
            return {"error": f"No default comparison pairs for '{base_symbol}'. Provide compare_symbols explicitly."}

    try:
        base_candles = fetch_ohlcv(base_symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch base symbol '{base_symbol}': {e}"}

    if len(base_candles) < 10:
        return {"error": f"Not enough data for '{base_symbol}' ({len(base_candles)} bars)."}

    base_closes = [c["close"] for c in base_candles]
    base_dates = [c["date"] for c in base_candles]

    comparisons = []

    for comp_sym in compare_symbols:
        try:
            comp_candles = fetch_ohlcv(comp_sym, period, interval)
        except Exception:
            comparisons.append({
                "symbol": comp_sym,
                "error": "Failed to fetch data",
            })
            continue

        if len(comp_candles) < 5:
            comparisons.append({
                "symbol": comp_sym,
                "error": "Insufficient data",
            })
            continue

        comp_closes = [c["close"] for c in comp_candles]

        # Align by date
        comp_date_map = {c["date"]: c["close"] for c in comp_candles}
        aligned_base = []
        aligned_comp = []
        aligned_dates = []

        for i, d in enumerate(base_dates):
            if d in comp_date_map:
                aligned_base.append(base_closes[i])
                aligned_comp.append(comp_date_map[d])
                aligned_dates.append(d)

        if len(aligned_base) < 5:
            comparisons.append({
                "symbol": comp_sym,
                "error": "Insufficient overlapping dates",
            })
            continue

        # Compute normalized spread (% difference)
        # Normalize both to index 100 at start
        base_norm = [p / aligned_base[0] * 100 for p in aligned_base]
        comp_norm = [p / aligned_comp[0] * 100 for p in aligned_comp]

        spreads = [round(comp_norm[i] - base_norm[i], 4) for i in range(len(base_norm))]
        abs_spreads = [abs(s) for s in spreads]

        avg_spread = statistics.mean(spreads)
        avg_abs_spread = statistics.mean(abs_spreads)
        max_spread = max(spreads)
        min_spread = min(spreads)
        spread_std = statistics.stdev(spreads) if len(spreads) > 1 else 0
        current_spread = spreads[-1]

        # Detect arbitrage windows (spread > 2 std from mean)
        threshold = avg_spread + 2 * spread_std
        neg_threshold = avg_spread - 2 * spread_std
        arb_windows = []

        for i, s in enumerate(spreads):
            if s > threshold or s < neg_threshold:
                arb_windows.append({
                    "date": aligned_dates[i],
                    "spread_pct": round(s, 4),
                    "direction": "premium" if s > 0 else "discount",
                    "base_price": aligned_base[i],
                    "comp_price": aligned_comp[i],
                })

        # Current opportunity
        if abs(current_spread - avg_spread) > 1.5 * spread_std and spread_std > 0:
            opportunity = "ACTIVE"
            opp_direction = "premium" if current_spread > avg_spread else "discount"
            opp_zscore = round((current_spread - avg_spread) / spread_std, 2)
        else:
            opportunity = "NONE"
            opp_direction = None
            opp_zscore = round((current_spread - avg_spread) / spread_std, 2) if spread_std > 0 else 0

        comparisons.append({
            "symbol": comp_sym,
            "overlapping_bars": len(aligned_base),
            "current_spread_pct": round(current_spread, 4),
            "avg_spread_pct": round(avg_spread, 4),
            "avg_abs_spread_pct": round(avg_abs_spread, 4),
            "max_spread_pct": round(max_spread, 4),
            "min_spread_pct": round(min_spread, 4),
            "spread_std": round(spread_std, 4),
            "z_score": opp_zscore,
            "opportunity": opportunity,
            "opportunity_direction": opp_direction,
            "arbitrage_windows": arb_windows[-10:],
            "base_price": aligned_base[-1],
            "comp_price": aligned_comp[-1],
        })

    # Overall assessment
    active_opps = [c for c in comparisons if isinstance(c, dict) and c.get("opportunity") == "ACTIVE"]

    return {
        "base_symbol": base_symbol.upper(),
        "period": period,
        "interval": interval,
        "candles_analyzed": len(base_candles),
        "date_from": base_candles[0]["date"],
        "date_to": base_candles[-1]["date"],
        "active_opportunities": len(active_opps),
        "comparisons": comparisons,
        "disclaimer": "Arbitrage analysis compares normalized price series. Actual arbitrage requires accounting "
                      "for fees, withdrawal times, and counterparty risk. For educational use only.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
