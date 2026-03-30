## TradingView MCP Server — Full Capability Analysis

### Core Architecture
A **3,800+ line MCP server** ([server.py](src/tradingview_mcp/server.py)) exposing **43 tools** over stdio or HTTP transport, backed by a fully **dependency-free pure-Python indicator/backtest engine** (no pandas, numpy, or scikit-learn) and **16 specialized detector modules**.

---

### 43 MCP Tools in 10 Categories

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
- `divergence_detector` — RSI/MACD/OBV divergence from price (bullish, bearish, hidden); one of the most reliable reversal signals
- `wash_trade_detector` — fake volume detection via volume-price correlation, clustering, round-number analysis, distribution shape
- `correlation_detector` — correlation, beta, alpha vs benchmarks; rolling correlation; decoupling event detection
- `volatility_regime_detector` — classifies LOW/NORMAL/HIGH/EXTREME volatility; recommends which strategies to use per regime
- `stop_hunt_detector` — liquidity trap detection (wicks sweeping S/R then reversing); cluster analysis of manipulation zones
- `dead_cat_bounce_detector` — identifies relief rallies after crashes that fail (declining volume, shallow retracement, weak RSI)
- `accumulation_distribution_detector` — smart money phase detection via OBV/A-D line trend vs price trend
- `slippage_risk_detector` — realistic slippage estimation per position size; validates backtester assumptions
- `market_regime_classifier` — TRENDING UP/DOWN/RANGING/CHOPPY classification with ADX, Efficiency Ratio, per-strategy fitness scores
- `arbitrage_detector` — cross-instrument spread analysis (e.g., BTC-USD vs GBTC/BITO/IBIT); z-score opportunities
- `news_price_lag_detector` — sentiment-price alignment, volume spike follow-through, news tradability assessment
- `seasonality_detector` — day-of-week and month-of-year patterns with bootstrap statistical significance testing

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
