## TradingView MCP Server — Full Capability Analysis

### Core Architecture
A **4,000+ line MCP server** ([server.py](src/tradingview_mcp/server.py)) exposing **46 tools** over stdio or HTTP transport, backed by a fully **dependency-free pure-Python indicator/backtest engine** (no pandas, numpy, or scikit-learn) and **19 specialized detector modules**.

---

### 46 MCP Tools in 11 Categories

**Screening & Scanning (6 tools)**
- `top_gainers` / `top_losers` — rank by % change with Bollinger indicators
- `bollinger_scan` — low BBW squeeze detection for breakout setups
- `rating_filter` — filter by strong buy/sell rating (-3 to +3)
- `volume_breakout_scanner` — volume spike + price breakout combos
- `smart_volume_scanner` — volume + RSI + technicals combined

**Deep Analysis (5 tools)**
- `coin_analysis` — full technical breakdown (RSI, MACD, SMA, EMA, ATR, Bollinger, OBV, Stochastic, ADX, S/R levels, market structure)
- `consecutive_candles_scan` / `advanced_candle_pattern` — candle pattern detection with momentum confirmation
- `volume_confirmation_analysis` — volume-price divergence signals
- `multi_agent_analysis` — **3-agent debate system** (Technical, Sentiment, Risk agents argue to consensus)

**Egyptian Exchange Specialized (7 tools)**
- Full EGX ecosystem: market overview, sector scanning (18 sectors with market cap weights), index analysis, stock screener (growth/value/momentum/quality), trade plan generator, Fibonacci retracement — all with hardcoded sector metadata in [egx_sectors.py](src/tradingview_mcp/core/data/egx_sectors.py)

**Multi-Timeframe & Sentiment (4 tools)**
- `multi_timeframe_analysis` — compare across 5m/15m/1h/4h/1D in one call
- `market_sentiment` — **Reddit scraping** via public JSON endpoints with proxy rotation
- `financial_news` — RSS aggregation from CoinDesk, CoinTelegraph, Reuters
- `combined_analysis` — **confluence tool** merging TradingView + Reddit + News

**Backtesting & Validation (5 tools)**
- `backtest_strategy` — full backtest with trade logs, equity curves, commission/slippage modeling
- `compare_strategies` — leaderboard of all 6 strategies vs buy-and-hold
- `walk_forward_backtest_strategy` — **overfitting detection** via train/test splits with robustness scoring
- `batch_walk_forward_test` — **cross-currency walk-forward** validation across up to 50 symbols at once
- `out_of_sample_test` — **pure OOS test** with clean chronological train/test split for forward validation

**Security & Signal Integrity (2 tools)**
- `rug_pull_detector` — detects crash events, whale dumps, pump-and-dump patterns, liquidity health; returns risk score (0-100) with severity rating (CRITICAL/HIGH/MODERATE/LOW)
- `repaint_detector` — validates signal stability by comparing bar-by-bar incremental signals vs full-hindsight signals; detects phantom, vanished, and flipped signals

**Market Intelligence Detectors (12 tools)**
- `divergence_detector` — detects 4 divergence types (bullish divergence, bearish divergence, hidden bullish divergence, hidden bearish divergence) across 3 indicators (RSI, MACD histogram, OBV) with swing-point matching and strength scoring
- `wash_trade_detector` — fake volume detection via 5 metrics: volume-price Pearson correlation, volume clustering (repeated identical bars), consecutive near-duplicate ratio, round-number volume ratio, log-volume distribution uniformity
- `correlation_detector` — Pearson correlation, beta (market sensitivity), annualized alpha (excess return) vs configurable benchmarks; rolling correlation with decoupling event detection; independence score
- `volatility_regime_detector` — classifies LOW/NORMAL/HIGH/EXTREME volatility via composite of ATR percentile, Bollinger Band Width percentile, daily range percentile, returns volatility percentile; recommends strategies per regime; detects Bollinger squeezes and regime transitions
- `stop_hunt_detector` — liquidity trap detection via wick-to-body ratio analysis on candles that sweep swing high/low pivot levels then reverse; volume confirmation; hunt zone clustering within 2% price bands; frequency scoring
- `dead_cat_bounce_detector` — identifies crash events (configurable threshold), detects subsequent bounces, classifies each as dead cat or genuine recovery using: Fibonacci retracement depth, volume trend during bounce, RSI at bounce peak, post-bounce price action
- `accumulation_distribution_detector` — Wyckoff-style phase detection (ACCUMULATION, DISTRIBUTION, MARKUP, MARKDOWN, ACCUMULATION_DIVERGENCE, DISTRIBUTION_DIVERGENCE, NEUTRAL) via OBV slope and Chaikin A/D line slope vs price slope comparison over rolling windows
- `slippage_risk_detector` — realistic slippage estimation for configurable position sizes ($1K/$5K/$10K/$50K/$100K) using market impact model (spread/2 + k × sqrt(participation_rate)); liquidity tier classification (VERY HIGH/HIGH/MODERATE/LOW/VERY LOW); comparison to backtester default assumption
- `market_regime_classifier` — classifies TRENDING_UP, TRENDING_DOWN, RANGING, or CHOPPY via ADX (trend strength), +DI/-DI (direction), Kaufman Efficiency Ratio (noise), EMA20/EMA50 alignment, choppiness index; outputs per-strategy fitness scores (0.0–1.0)
- `arbitrage_detector` — cross-instrument normalized spread analysis; computes z-scores, detects active opportunities when spread exceeds 1.5 standard deviations; tracks historical arbitrage windows; auto-selects comparison pairs (BTC-USD vs GBTC/BITO/IBIT, ETH-USD vs ETHE/ETHA)
- `news_price_lag_detector` — sentiment-price alignment check (ALIGNED/DIVERGENT/NEUTRAL), volume spike follow-through analysis (continuation vs reversal rate), news tradability assessment (HIGH/MODERATE/LOW), multi-horizon momentum (1d/3d/7d/14d/30d)
- `seasonality_detector` — day-of-week analysis (Monday through Sunday avg return, win rate, edge vs overall), month-of-year analysis (January through December), first-half vs second-half of month, bootstrap statistical significance testing (p-values) for each pattern

**Formation Recognition (3 tools)**
- `candlestick_pattern_scanner` — detects 25+ candlestick patterns across three categories:
  - *Single-bar (9):* Doji, Dragonfly Doji, Gravestone Doji, Hammer, Inverted Hammer, Shooting Star, Bullish Marubozu, Bearish Marubozu, Spinning Top
  - *Dual-bar (8):* Bullish Engulfing, Bearish Engulfing, Piercing Line, Dark Cloud Cover, Tweezer Top, Tweezer Bottom, Bullish Harami, Bearish Harami, Bullish Harami Cross, Bearish Harami Cross
  - *Triple-bar (6):* Morning Star, Evening Star, Three White Soldiers, Three Black Crows, Three Inside Up, Three Inside Down
  - Each pattern includes reliability rating (low/moderate/high), RSI context, and bullish/bearish/neutral classification
- `chart_formation_scanner` — detects 17 classical multi-bar structural formations:
  - *Reversal (8):* Head and Shoulders, Inverse Head and Shoulders, Double Top, Double Bottom, Triple Top, Triple Bottom, Rising Wedge, Falling Wedge
  - *Continuation (5):* Ascending Triangle, Descending Triangle, Bull Flag, Bear Flag, Pennant, Cup and Handle
  - *Breakout (1):* Symmetrical Triangle
  - *Trend Structure (3):* Ascending Channel, Descending Channel, Horizontal Channel
  - Each formation includes price targets (measured move), necklines, breakout levels, R² trendline fit quality, and current position within channels
- `support_resistance_mapper` — maps S/R zones via multi-touch pivot clustering with strength scoring (0-10), nearest level detection, position assessment (near support / near resistance / mid-range), breakout/breakdown validation with volume confirmation, and ATR context for stop-loss sizing

**Market Data (2 tools)**
- `yahoo_price` — real-time OHLCV for stocks, crypto, ETFs, FX, indices
- `market_snapshot` — global overview (major indices, top crypto, FX rates, key ETFs)

---

### 6 Built-In Trading Strategies
| Strategy | Logic |
|----------|-------|
| RSI | Buy oversold <30, sell overbought >70 |
| Bollinger | Buy lower band, sell middle |
| MACD | Golden/death cross |
| EMA Cross | EMA20/EMA50 crossover |
| Supertrend | ATR-based trend flip |
| Donchian | N-period high/low breakout |

Each produces: Sharpe ratio, Calmar ratio, max drawdown, profit factor, win rate, expectancy, best/worst trades, full trade log, and equity curve data.

---

### Rug Pull Detection
The `rug_pull_detector` ([security_checks.py](src/tradingview_mcp/core/services/security_checks.py)) analyzes:
- **Crash events** — single-candle drops exceeding threshold with volume spikes
- **Whale activity** — abnormal volume (>4.5x SMA20) with directional classification (pump/dump/neutral)
- **Pump-and-dump patterns** — rapid rises (>50%) within a window followed by sharp declines
- **Liquidity health** — volume consistency, trend, average spread, zero-volume day ratio
- Outputs a composite risk score (0-100) with CRITICAL/HIGH/MODERATE/LOW severity

### Repaint Detection
The `repaint_detector` ([repaint_detector.py](src/tradingview_mcp/core/services/repaint_detector.py)) validates signal integrity by:
- Running each strategy bar-by-bar (only seeing data up to bar N) to simulate live trading
- Comparing those "live" signals to signals computed with full hindsight
- Classifying discrepancies as **phantom** (only in hindsight), **vanished** (disappeared in hindsight), or **flipped** (direction changed)
- Returning a stability score (0.0–1.0) and risk level (CLEAN/MOSTLY CLEAN/MODERATE/SIGNIFICANT/SEVERE)

### Formation Recognition

#### Candlestick Patterns ([candlestick_patterns.py](src/tradingview_mcp/core/services/candlestick_patterns.py))
Pure-Python recognition of 25+ patterns from OHLCV data, each with type/signal/reliability metadata:

| Category | Patterns |
|----------|----------|
| **Single-bar reversal** | Hammer, Inverted Hammer, Shooting Star |
| **Single-bar momentum** | Bullish Marubozu, Bearish Marubozu |
| **Single-bar indecision** | Doji, Dragonfly Doji, Gravestone Doji, Spinning Top |
| **Dual-bar reversal** | Bullish Engulfing, Bearish Engulfing, Piercing Line, Dark Cloud Cover, Tweezer Top, Tweezer Bottom, Bullish Harami, Bearish Harami, Bullish Harami Cross, Bearish Harami Cross |
| **Triple-bar reversal** | Morning Star, Evening Star, Three Inside Up, Three Inside Down |
| **Triple-bar momentum** | Three White Soldiers, Three Black Crows |

Each pattern includes: bar index, date, close price, RSI context, reliability rating (low/moderate/high), and bullish/bearish/neutral classification. Results summarize recent bias and pattern frequency distribution.

#### Chart Formations ([chart_formations.py](src/tradingview_mcp/core/services/chart_formations.py))
Structural pattern detection via swing high/low identification and geometric fitting:

| Formation | Type | Signal | Detection Method |
|-----------|------|--------|-----------------|
| **Head & Shoulders** | Bearish | Reversal | 3 swing highs with middle highest, roughly equal shoulders, neckline from intermediate lows |
| **Inverse H&S** | Bullish | Reversal | Mirror of H&S on swing lows |
| **Double Top** | Bearish | Reversal | 2 swing highs at same level with intervening low |
| **Double Bottom** | Bullish | Reversal | 2 swing lows at same level with intervening high |
| **Triple Top** | Bearish | Reversal | 3 swing highs at same level with intervening lows |
| **Triple Bottom** | Bullish | Reversal | 3 swing lows at same level with intervening highs |
| **Ascending Triangle** | Bullish | Continuation | Flat resistance + rising support trendline (linear regression) |
| **Descending Triangle** | Bearish | Continuation | Flat support + falling resistance trendline |
| **Symmetrical Triangle** | Neutral | Breakout | Converging trendlines (both contracting) |
| **Rising Wedge** | Bearish | Reversal | Both trendlines rising but converging |
| **Falling Wedge** | Bullish | Reversal | Both trendlines falling but converging |
| **Bull Flag** | Bullish | Continuation | Strong impulse up + small downward-sloping consolidation |
| **Bear Flag** | Bearish | Continuation | Strong impulse down + small upward-sloping consolidation |
| **Pennant** | Both | Continuation | Impulse + converging consolidation |
| **Cup & Handle** | Bullish | Continuation | U-shaped base with rims at similar level + small handle pullback |
| **Ascending Channel** | Bullish | Trend | Parallel rising trendlines with current position tracking |
| **Descending Channel** | Bearish | Trend | Parallel falling trendlines |
| **Horizontal Channel** | Neutral | Range | Parallel flat trendlines |

Each formation includes: price targets (measured move), necklines/breakout levels, R² fit quality for trendlines, and current position within channels.

#### Support / Resistance Mapping ([support_resistance.py](src/tradingview_mcp/core/services/support_resistance.py))
Multi-touch zone detection with scoring:
- **Pivot detection** — swing highs/lows with configurable lookback sensitivity
- **Zone clustering** — groups nearby pivots within tolerance % into zones (not single price points)
- **Strength scoring (0-10)** — based on: number of touches (more = stronger), volume at touch points vs average, recency of touches
- **Position assessment** — nearest support/resistance, distance to each, position within range (near support/resistance/mid-range)
- **Breakout validation** — detects recent breakouts/breakdowns of S/R zones with volume confirmation
- **ATR context** — current volatility for stop-loss sizing relative to levels

---

### Batch Walk-Forward & Out-of-Sample Testing
- `batch_walk_forward_test` — tests a strategy across multiple currencies/symbols simultaneously, aggregates cross-symbol robustness with variance analysis
- `out_of_sample_test` — clean chronological split (e.g., 70% train / 30% test on most recent data) with degradation ratio and Sharpe degradation scoring

### Market Intelligence Detectors

| Detector | What It Detects | Key Output |
|----------|----------------|------------|
| [divergence_detector.py](src/tradingview_mcp/core/services/divergence_detector.py) | RSI/MACD/OBV price divergences | Bullish/bearish/hidden divergences with strength scores |
| [wash_trade_detector.py](src/tradingview_mcp/core/services/wash_trade_detector.py) | Fake/artificial volume | Wash trade probability (0-100), evidence list |
| [correlation_detector.py](src/tradingview_mcp/core/services/correlation_detector.py) | Benchmark correlation/beta/alpha | Independence score, decoupling events |
| [volatility_regime.py](src/tradingview_mcp/core/services/volatility_regime.py) | Volatility regime classification | LOW/NORMAL/HIGH/EXTREME + strategy recommendations |
| [stop_hunt_detector.py](src/tradingview_mcp/core/services/stop_hunt_detector.py) | Liquidity traps / stop hunts | Hunt clusters, swept S/R levels, frequency score |
| [dead_cat_detector.py](src/tradingview_mcp/core/services/dead_cat_detector.py) | Failed relief rallies after crashes | Dead cat vs genuine recovery classification |
| [accumulation_detector.py](src/tradingview_mcp/core/services/accumulation_detector.py) | Smart money accumulation/distribution | Wyckoff-style phase detection, A/D zones |
| [slippage_risk.py](src/tradingview_mcp/core/services/slippage_risk.py) | Realistic slippage per position size | Liquidity tier, recommended backtester settings |
| [regime_classifier.py](src/tradingview_mcp/core/services/regime_classifier.py) | Market regime (trending/ranging/choppy) | ADX/ER-based classification, strategy fitness scores |
| [arbitrage_detector.py](src/tradingview_mcp/core/services/arbitrage_detector.py) | Cross-instrument price discrepancies | Spread z-scores, arbitrage windows |
| [news_lag_detector.py](src/tradingview_mcp/core/services/news_lag_detector.py) | News-price relationship | Sentiment alignment, tradability assessment |
| [seasonality_detector.py](src/tradingview_mcp/core/services/seasonality_detector.py) | Day/month seasonal patterns | Bootstrap-tested significant patterns |
| [candlestick_patterns.py](src/tradingview_mcp/core/services/candlestick_patterns.py) | 25+ candlestick patterns | Pattern type, signal, reliability, RSI context |
| [chart_formations.py](src/tradingview_mcp/core/services/chart_formations.py) | 17 classical chart formations | Targets, necklines, breakout levels, R² fit |
| [support_resistance.py](src/tradingview_mcp/core/services/support_resistance.py) | Multi-touch S/R zones | Strength scores, position assessment, breakouts |

---

### Non-Obvious Capabilities

1. **Pure-Python indicator library** ([indicators_calc.py](src/tradingview_mcp/core/services/indicators_calc.py)) — EMA, SMA, RSI (Wilder's smoothing), Bollinger, MACD, ATR, Supertrend, Donchian, ADX — all from scratch with zero dependencies
2. **Proxy rotation system** ([proxy_manager.py](src/tradingview_mcp/core/services/proxy_manager.py)) — Webshare residential proxy integration with session pooling (1-250 sessions) for Reddit/Yahoo rate-limit bypass
3. **Auto market-type detection** — automatically distinguishes stock vs crypto symbols and routes to correct screener
4. **Batch processing** — handles 200-500 symbols per API call with smart fallbacks
5. **22 exchange symbol lists** in [coinlist/](src/tradingview_mcp/coinlist/) covering 11 crypto + 11 stock exchanges across Egypt, Turkey, USA, Malaysia, Hong Kong
6. **OpenClaw CLI wrapper** ([openclaw/trading.py](openclaw/trading.py)) — standalone Python CLI for using tools outside MCP context
7. **Walk-forward robustness scoring** — flags strategies as ROBUST/MODERATE/WEAK/OVERFITTED based on out-of-sample vs in-sample performance ratio
8. **Multi-agent debate** — the `multi_agent_analysis` tool simulates 3 AI personas (Technical Analyst, Sentiment Analyst, Risk Manager) that independently evaluate and then synthesize a consensus
9. **Dual transport** — runs as stdio (for Claude Code) or streamable-http (for Docker/remote) with health check endpoint
10. **Zero required API keys** — everything works out of the box using public APIs; proxy credentials are optional for heavy usage

---

### Market Coverage
**Crypto:** Binance, Bybit, Bitget, OKX, Coinbase, KuCoin, Gate.io, Huobi, Bitfinex
**Stocks:** NASDAQ, NYSE, EGX (Egypt), BIST (Turkey), BURSA/KLSE/MYX/ACE/LEAP (Malaysia), HKEX (Hong Kong)
**Other:** ETFs, Forex pairs, major indices (S&P 500, Dow, NASDAQ Composite)
