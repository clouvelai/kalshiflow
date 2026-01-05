# LSD MODE: Market Discovery Wild Ideas

**Session Date**: 2026-01-05
**Mode**: LSD (Lateral Strategy Discovery)
**Focus**: Finding BETTER markets to trade BEFORE they become obvious

## Current State Analysis

The V3 trader uses these discovery mechanisms:
1. **ApiDiscoverySyncer**: Fetches status="open" markets, filters by category (sports, crypto, entertainment, media_mentions)
2. **EventLifecycleService**: Listens to lifecycle WebSocket for newly created markets
3. **TrackedMarketsSyncer**: Periodic REST sync for price/volume updates
4. **Priority scoring**: Volume * 1000 + (has_trade_history ? 500 : 0) + (hot_keyword ? 100 : 0)

**What's Missing**: The current system is REACTIVE. It finds active markets but doesn't PREDICT which will become hot or find hidden edge.

---

## WILD IDEAS

### 1. Event Lifecycle Velocity Detector

**Craziness Level**: 3/10
**Core Concept**: Markets that transition quickly through lifecycle stages have urgent resolution = more informed trading = more edge.

**Data Source**: Kalshi lifecycle WebSocket (created -> determined events)
**Implementation Difficulty**: Easy
**Expected Impact**: Medium - could filter out "boring" long-dated markets

**Why It Might Work**:
- Fast-resolving markets (sports during games, breaking news) attract attention
- Time pressure creates emotional/irrational trading
- Our RLM strategy thrives on price drops from panicked sellers

**Why It Might Fail**:
- Fast markets might be TOO efficient (sharp money already there)
- May correlate with what we already capture via volume

**Implementation Sketch**:
```python
# Track creation timestamps, score markets by expected velocity
velocity_score = 1.0 / max(1.0, hours_to_settlement)
# High velocity (settling soon) = higher priority
```

---

### 2. Cross-Market Correlation Mapper

**Craziness Level**: 5/10
**Core Concept**: When Market A moves, related Market B often follows with a lag. Find the leaders.

**Data Source**: Our own orderbook deltas database + trade stream
**Implementation Difficulty**: Medium
**Expected Impact**: High - could front-run correlated markets

**Why It Might Work**:
- Kalshi has many correlated markets (NFL game winner vs player props)
- Event-level correlation (same game) creates lead-lag relationships
- First-mover advantage on correlated trades

**Why It Might Fail**:
- Correlations may be too obvious (already priced in)
- Lag might be <1 second (faster than we can act)

**Implementation Sketch**:
```python
# Group markets by event_ticker
# Track which market in a group moves first
# Subscribe to "leader" markets, wait for signal, trade "laggards"
correlation_matrix = compute_rolling_correlation(orderbook_deltas, window=100)
leader_markets = identify_granger_causality(trade_stream)
```

---

### 3. Whale Tracker / Follow the Money

**Craziness Level**: 4/10
**Core Concept**: Detect when large traders enter a market, ride their coattails.

**Data Source**: Public trades stream (we have this!)
**Implementation Difficulty**: Medium
**Expected Impact**: High - piggyback on informed traders

**Why It Might Work**:
- Large trades often precede price moves
- Whales have research budgets we don't
- Even 10c move after whale = profitable for us

**Why It Might Fail**:
- Whales might be dumb money too
- Public trades are delayed - might be too late
- Whales might be market makers (neutral signal)

**Implementation Sketch**:
```python
# Track trade sizes, identify outlier large trades
WHALE_THRESHOLD = 1000  # contracts
def detect_whale(trade):
    if trade.count >= WHALE_THRESHOLD:
        # New market enters our watchlist
        prioritize_market(trade.ticker)
        # If whale bought YES, we might follow
        emit_whale_signal(trade.ticker, trade.side, trade.count)
```

**Validated Hypothesis**: S013 already uses this! But we could extend:
- Track whale IDENTITY (repeated patterns from same trading style)
- Whale divergence (whale buys YES but price drops = stronger RLM signal)

---

### 4. News Sentiment Pre-Scanner

**Craziness Level**: 6/10
**Core Concept**: Scan news/Twitter for events BEFORE Kalshi creates the market.

**Data Source**: Twitter API, News APIs (NewsAPI, Google News), RSS feeds
**Implementation Difficulty**: Hard
**Expected Impact**: Very High - first mover advantage

**Why It Might Work**:
- Breaking news creates new markets
- Early detection = time to research before market opens
- Social media leads traditional news by minutes

**Why It Might Fail**:
- API costs, rate limits
- False positives (noise)
- Kalshi might already monitor same sources
- NLP/sentiment analysis is hard

**Implementation Sketch**:
```python
# Monitor Twitter for trending topics in our categories
# Score tweets by engagement velocity
# Cross-reference with Kalshi API for new market listings
# Queue pre-research on likely upcoming markets
upcoming_topics = twitter_client.get_trending(categories=["sports", "politics"])
for topic in upcoming_topics:
    likely_markets = predict_kalshi_markets(topic)
    await pre_research_queue.add(likely_markets)
```

---

### 5. Spread Decay Predictor

**Craziness Level**: 4/10
**Core Concept**: Markets with wide spreads that are tightening = liquidity arriving = opportunity.

**Data Source**: Our orderbook snapshots/deltas
**Implementation Difficulty**: Easy
**Expected Impact**: Medium - better execution

**Why It Might Work**:
- Wide spreads = market makers haven't arrived
- Tightening spread = competition arriving = better prices coming
- First to tight spread = best fills

**Why It Might Fail**:
- Wide spreads might stay wide (illiquid market)
- Tightening might mean opportunity is GONE

**Implementation Sketch**:
```python
# Track spread history per market
spread_velocity = (current_spread - spread_10min_ago) / 10
# Negative velocity = tightening = prioritize
if spread_velocity < -0.1:  # 1c tightening per minute
    upgrade_priority(ticker)
```

---

### 6. Volume Momentum Scanner

**Craziness Level**: 3/10
**Core Concept**: Markets with accelerating volume (not just high volume) are heating up.

**Data Source**: Kalshi REST API volume fields, our trade stream
**Implementation Difficulty**: Easy
**Expected Impact**: Medium - catch markets "going viral"

**Why It Might Work**:
- Volume acceleration precedes price moves
- Social momentum is self-reinforcing
- Early detection of "hot" markets

**Why It Might Fail**:
- Volume might be manipulation/wash trades
- Momentum might reverse quickly

**Implementation Sketch**:
```python
# Track volume over time windows
volume_1h = get_volume(ticker, window="1h")
volume_24h = get_volume(ticker, window="24h")
volume_momentum = volume_1h / max(1, volume_24h / 24)  # Ratio to hourly average
# >2x normal hourly volume = heating up
if volume_momentum > 2.0:
    prioritize_market(ticker)
```

---

### 7. New Market Alpha Hunter

**Craziness Level**: 5/10
**Core Concept**: Brand new markets have the most mispricing before efficient pricing arrives.

**Data Source**: Lifecycle WebSocket "created" events
**Implementation Difficulty**: Easy
**Expected Impact**: High - first mover on pricing inefficiency

**Why It Might Work**:
- New markets have no historical pricing reference
- Initial prices set by first few traders (often wrong)
- Sharp money takes time to arrive

**Why It Might Fail**:
- Low liquidity makes execution hard
- We might be the dumb money setting initial price
- Spreads are usually wide at launch

**Implementation Sketch**:
```python
# Aggressively subscribe to ALL new markets
# Track initial price vs first hour average
# Markets with >5c initial mispricing = opportunity
async def on_market_created(event):
    await immediately_subscribe(event.ticker)
    await capture_initial_orderbook(event.ticker)
    # Compare to fundamental analysis if available
    fair_value = estimate_fair_value(event)
    if abs(event.yes_price - fair_value) > 5:
        emit_new_market_edge_signal(event.ticker, direction)
```

---

### 8. Category Rotation / Sector Momentum

**Craziness Level**: 5/10
**Core Concept**: Attention rotates between categories (sports seasons, political cycles, crypto hype).

**Data Source**: Category-level volume aggregation
**Implementation Difficulty**: Medium
**Expected Impact**: Medium - portfolio allocation improvement

**Why It Might Work**:
- NFL season = sports category hot
- Election cycle = politics hot
- Crypto bull run = crypto category hot
- Rotating capital to hot sectors improves opportunity

**Why It Might Fail**:
- Obvious (everyone knows NFL is popular in fall)
- Our RLM edge might not vary by category

**Implementation Sketch**:
```python
# Aggregate volume by category
# Track 7-day rolling volume per category
# Allocate subscription slots proportionally
category_momentum = {}
for category in CATEGORIES:
    recent_vol = sum_volume(category, days=7)
    prior_vol = sum_volume(category, days=7, offset=7)
    category_momentum[category] = recent_vol / max(1, prior_vol)

# Reallocate tracked market capacity to hot categories
```

---

### 9. Inefficiency Hour Detector

**Craziness Level**: 6/10
**Core Concept**: Trading is most irrational at specific times (late night, drinking hours, lunch breaks).

**Data Source**: Time-series analysis of our trade outcomes
**Implementation Difficulty**: Easy
**Expected Impact**: Medium - timing optimization

**Why It Might Work**:
- Research shows retail trades at night are worse
- "Drunk betting" hypothesis (validated in S011!)
- Lunch hour might have distracted traders

**Why It Might Fail**:
- Effect might be small
- Low volume at these times

**Implementation Sketch**:
```python
# Analyze historical trade outcomes by hour
# Identify hours with worst retail performance = best for us
DRUNK_HOURS = [23, 0, 1, 2, 3]  # 11pm - 3am
LUNCH_HOURS = [12, 13]
def adjust_opportunity_score(ticker, base_score):
    hour = datetime.now().hour
    if hour in DRUNK_HOURS:
        return base_score * 1.5  # 50% boost
    return base_score
```

---

### 10. Contrarian Market Finder

**Craziness Level**: 7/10
**Core Concept**: Markets that EVERYONE ignores might have hidden value.

**Data Source**: Volume ranking (inverse)
**Implementation Difficulty**: Easy
**Expected Impact**: Unknown - could be genius or terrible

**Why It Might Work**:
- Efficient market hypothesis doesn't hold in thin markets
- No attention = no sharp money = mispricing persists
- Proprietary edge if we're the only ones looking

**Why It Might Fail**:
- Markets are ignored for a reason (boring, illiquid)
- No volume = no one to trade against
- Might be the "falling knife" of markets

**Implementation Sketch**:
```python
# Find markets with LOW volume but approaching settlement
# Theory: Last-minute activity as resolution approaches
lonely_markets = filter(lambda m: m.volume < 100 and m.hours_to_close < 4)
# Monitor these for sudden activity spikes
```

---

### 11. Event Clustering / Related Market Bundler

**Craziness Level**: 5/10
**Core Concept**: Group all markets from same event, trade the bundle when one moves.

**Data Source**: event_ticker field in market data
**Implementation Difficulty**: Medium
**Expected Impact**: Medium - better signal aggregation

**Why It Might Work**:
- NFL game has game winner + props + score markets
- Movement in one implies info about others
- Aggregate signal stronger than individual

**Why It Might Fail**:
- Markets might be independent (player props unrelated to game outcome)
- Adds complexity

**Implementation Sketch**:
```python
# Group all markets by event_ticker
event_bundles = group_by(tracked_markets, key="event_ticker")
# When ANY market in bundle signals, evaluate ALL
async def on_rlm_signal(signal):
    bundle = event_bundles.get(signal.event_ticker, [])
    for related_market in bundle:
        if not related_market.has_position:
            evaluate_correlated_opportunity(related_market, signal)
```

---

### 12. Sports Calendar Integration

**Craziness Level**: 4/10
**Core Concept**: Pre-position on markets for known future events.

**Data Source**: ESPN API, Sports-Reference, official league calendars
**Implementation Difficulty**: Medium
**Expected Impact**: High - prepare research before markets open

**Why It Might Work**:
- NFL schedule is known months ahead
- Can research teams/matchups before market opens
- First to fundamental analysis when market lists

**Why It Might Fail**:
- Kalshi might not create market for every game
- Research advantage might not translate to trading edge

**Implementation Sketch**:
```python
# Fetch upcoming sports schedule
nfl_games = espn_client.get_schedule("nfl", days_ahead=7)
# Pre-compute expected markets
expected_tickers = [predict_ticker(game) for game in nfl_games]
# When lifecycle event matches expected, instantly prioritize
```

---

### 13. Order Book Imbalance Screener

**Craziness Level**: 4/10
**Core Concept**: Markets with heavily imbalanced order books (all bids, no asks) are about to move.

**Data Source**: Our orderbook snapshots
**Implementation Difficulty**: Easy (we have the data!)
**Expected Impact**: High - directional prediction

**Why It Might Work**:
- Order flow imbalance predicts short-term price direction
- Classic market microstructure signal
- HFT firms use this (at faster timescales)

**Why It Might Fail**:
- Imbalances might be spoofing
- Signal might be too short-lived for our reaction time

**Implementation Sketch**:
```python
# Compute bid/ask imbalance from orderbook
def compute_imbalance(orderbook):
    total_bid_volume = sum(level.size for level in orderbook.bids[:5])
    total_ask_volume = sum(level.size for level in orderbook.asks[:5])
    imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
    return imbalance  # -1 to +1, positive = buying pressure

# Alert on extreme imbalances
if abs(imbalance) > 0.7:
    emit_imbalance_alert(ticker, imbalance)
```

---

### 14. Market Maker Detection

**Craziness Level**: 6/10
**Core Concept**: Identify markets where no market maker is present = wider spreads = more opportunity.

**Data Source**: Orderbook patterns, trade flow
**Implementation Difficulty**: Hard
**Expected Impact**: Medium - better market selection

**Why It Might Work**:
- Market makers tighten spreads (less opportunity)
- No MM = retail-only = more mispricing
- Can identify MM by symmetric quote patterns

**Why It Might Fail**:
- Hard to distinguish MM from other traders
- No MM = low liquidity = hard to execute

**Implementation Sketch**:
```python
# Heuristics for MM presence
# - Symmetric quotes around mid
# - Quick requotes after fills
# - Consistent size on both sides
def has_market_maker(orderbook, trade_history):
    # Check for symmetric quotes
    bid_ask_symmetry = abs(orderbook.best_bid_size - orderbook.best_ask_size) / (orderbook.best_bid_size + orderbook.best_ask_size)
    # Check for requote patterns
    requote_frequency = count_requotes(trade_history)

    if bid_ask_symmetry < 0.2 and requote_frequency > 10:
        return True  # MM likely present
    return False

# Prioritize markets WITHOUT market makers
```

---

### 15. Price Discovery Phase Detector

**Craziness Level**: 5/10
**Core Concept**: Markets go through phases - discovery (volatile), consensus (stable). Trade during discovery.

**Data Source**: Price/orderbook volatility over time
**Implementation Difficulty**: Medium
**Expected Impact**: High - timing optimization

**Why It Might Work**:
- Discovery phase has wide price swings = more RLM signals
- Consensus phase is efficient = boring
- Can predict phase from volatility patterns

**Why It Might Fail**:
- Phase transitions are hard to detect in real-time
- Discovery might be too chaotic to trade

**Implementation Sketch**:
```python
# Compute rolling price volatility
def get_discovery_score(ticker, window_minutes=30):
    prices = get_price_history(ticker, window_minutes)
    volatility = np.std(prices)
    # High volatility = discovery phase
    return volatility

# Prioritize high-discovery markets
discovery_scores = {t: get_discovery_score(t) for t in tracked_markets}
prioritized = sorted(discovery_scores.items(), key=lambda x: -x[1])
```

---

### 16. Fibonacci Retracement Level Trigger

**Craziness Level**: 8/10
**Core Concept**: Technical traders believe in Fibonacci levels. When price hits 61.8% retracement, they act.

**Data Source**: Our price history
**Implementation Difficulty**: Easy
**Expected Impact**: Unknown - might be nonsense

**Why It Might Work**:
- Self-fulfilling prophecy: enough traders believe it
- Provides clear entry/exit points
- Works in stocks, might work in prediction markets

**Why It Might Fail**:
- Prediction markets are fundamentally different
- No continuous trading history like stocks
- Probably just astrology for traders

**Implementation Sketch**:
```python
# Calculate Fibonacci levels from recent high/low
def fib_levels(high, low):
    diff = high - low
    return {
        "0.236": high - diff * 0.236,
        "0.382": high - diff * 0.382,
        "0.618": high - diff * 0.618,
    }

# Alert when price approaches key level
levels = fib_levels(market_high, market_low)
if abs(current_price - levels["0.618"]) < 2:
    emit_fib_alert(ticker, "0.618_touch")
```

---

### 17. Moon Phase Market Selector

**Craziness Level**: 10/10
**Core Concept**: Full moon = more irrational trading. Select markets during lunar extremes.

**Data Source**: Astronomy API
**Implementation Difficulty**: Easy
**Expected Impact**: Probably zero, but YOLO

**Why It Might Work**:
- Studies show emergency room visits spike on full moons
- If people are generally more emotional, trading might be too
- At minimum, funny story if it works

**Why It Might Fail**:
- Complete nonsense
- No causal mechanism
- Just astrology

**Implementation Sketch**:
```python
# Get current moon phase
from astropy.coordinates import get_moon, get_sun
moon_phase = calculate_moon_phase()  # 0 = new, 0.5 = full

# Boost opportunity during full moon (phase 0.4-0.6)
def moon_adjustment(base_score):
    if 0.4 <= moon_phase <= 0.6:
        return base_score * 1.1  # 10% boost during full moon
    return base_score
```

---

### 18. Weather-Correlated Market Finder

**Craziness Level**: 7/10
**Core Concept**: Weather affects sports outcomes AND trader mood. Double opportunity.

**Data Source**: OpenWeather API, National Weather Service
**Implementation Difficulty**: Medium
**Expected Impact**: Medium for sports, Low for mood

**Why It Might Work**:
- Rain/wind affects outdoor sports outcomes
- Weather affects market sentiment (documented in stock markets)
- Might find edge in weather-sensitive markets

**Why It Might Fail**:
- Weather data might already be priced in
- Mood correlation might not apply to prediction markets

**Implementation Sketch**:
```python
# Fetch weather for sports venue
venue_weather = weather_api.get_forecast(game.venue_location)
# Adjust market priority for weather-sensitive sports
if game.sport in ["NFL", "MLB", "Soccer"] and venue_weather.is_extreme:
    prioritize_market(game.ticker)
    # Also flag for fundamental analysis
    flag_for_review(game.ticker, reason="extreme_weather")
```

---

### 19. Copy-Cat New Market Detector

**Craziness Level**: 5/10
**Core Concept**: When one platform creates a market, others follow. Watch Polymarket/PredictIt for Kalshi previews.

**Data Source**: Polymarket API, PredictIt RSS
**Implementation Difficulty**: Medium
**Expected Impact**: Medium - lead time on new markets

**Why It Might Work**:
- Platforms copy popular markets from each other
- Polymarket might launch first, Kalshi follows
- Can front-run the market creation

**Why It Might Fail**:
- Regulatory differences (CFTC approval needed for Kalshi)
- Different market structures
- API availability varies

**Implementation Sketch**:
```python
# Monitor Polymarket for new markets
polymarket_new = polymarket_api.get_markets(created_after=last_check)
# Check if Kalshi has equivalent
for pm_market in polymarket_new:
    kalshi_match = find_kalshi_equivalent(pm_market)
    if not kalshi_match:
        # Add to watchlist - might come to Kalshi soon
        watchlist.add(pm_market.topic)
```

---

### 20. Social Sentiment Spike Detector

**Craziness Level**: 6/10
**Core Concept**: Reddit/Twitter volume spikes precede market volume spikes.

**Data Source**: Reddit API, Twitter API
**Implementation Difficulty**: Hard
**Expected Impact**: High if done right

**Why It Might Work**:
- Social discussion leads trading activity
- WSB-style sentiment moves markets
- Early detection = position before retail pile-in

**Why It Might Fail**:
- API costs and rate limits
- NLP is hard
- Social sentiment might be wrong

**Implementation Sketch**:
```python
# Monitor r/wallstreetbets, r/sportsbook, Twitter
# Track mention velocity
def get_social_velocity(topic):
    reddit_mentions = reddit_api.search(topic, since="1h")
    twitter_mentions = twitter_api.search(topic, since="1h")
    return len(reddit_mentions) + len(twitter_mentions)

# Alert on spike (>2x normal)
for market in tracked_markets:
    velocity = get_social_velocity(market.title)
    if velocity > 2 * market.avg_social_velocity:
        emit_social_spike_alert(market.ticker)
```

---

### 21. Regulatory Event Calendar

**Craziness Level**: 4/10
**Core Concept**: Known regulatory events create predictable market opportunities.

**Data Source**: Fed calendar, FOMC dates, earnings calendars
**Implementation Difficulty**: Easy
**Expected Impact**: Medium - timing markets around events

**Why It Might Work**:
- Fed meetings move crypto markets
- Earnings move stock-related prediction markets
- Known dates = can prepare

**Why It Might Fail**:
- Events are priced in
- Kalshi might not have relevant markets

**Implementation Sketch**:
```python
# Fetch regulatory calendar
fed_dates = fetch_fomc_calendar()
# Increase crypto market priority around Fed meetings
for date in fed_dates:
    if days_until(date) < 3:
        for market in tracked_markets:
            if market.category == "crypto":
                boost_priority(market.ticker, factor=1.5)
```

---

### 22. Orderbook Toxicity Scorer

**Craziness Level**: 5/10
**Core Concept**: Measure "toxicity" - probability of being adversely selected. Avoid toxic markets.

**Data Source**: Our orderbook + trade flow data
**Implementation Difficulty**: Hard
**Expected Impact**: High - avoid bad trades

**Why It Might Work**:
- Classic HFT concept - VPIN, Kyle's Lambda
- Toxic markets have informed traders picking off stale quotes
- Our RLM might be getting "toxic" fills

**Why It Might Fail**:
- Complex to compute accurately
- Might need more data than we have
- Definition of "toxic" varies

**Implementation Sketch**:
```python
# Compute VPIN (Volume-Synchronized Probability of Informed Trading)
def compute_vpin(trades, buckets=50):
    bucket_size = sum(t.count for t in trades) / buckets
    buy_volume = []
    sell_volume = []
    # ... bucket trades by volume, classify buy/sell
    # VPIN = abs(buy - sell) / (buy + sell)
    return vpin

# Avoid markets with high VPIN
if compute_vpin(recent_trades) > 0.6:
    reduce_priority(ticker)
```

---

### 23. Machine Learning Market Scorer

**Craziness Level**: 6/10
**Core Concept**: Train a model to predict which markets will have positive RLM edge.

**Data Source**: All our historical data
**Implementation Difficulty**: Hard
**Expected Impact**: High - automated market selection

**Why It Might Work**:
- We have labeled data (did RLM work in this market?)
- Can find patterns humans miss
- Continuous improvement as data grows

**Why It Might Fail**:
- Overfitting to historical patterns
- Markets change (concept drift)
- Needs significant data to train

**Implementation Sketch**:
```python
# Feature engineering
features = [
    "volume_momentum",
    "spread_percentile",
    "hours_to_settlement",
    "category_encoded",
    "time_of_day",
    "day_of_week",
    "orderbook_imbalance",
    "trade_velocity",
]

# Train classifier: "Will RLM have positive edge?"
model = XGBClassifier()
model.fit(X_train, y_train)  # y = 1 if RLM profitable, 0 otherwise

# Score new markets
def score_market(market):
    features = extract_features(market)
    probability = model.predict_proba(features)[0][1]
    return probability
```

---

### 24. Arbitrage Opportunity Scanner

**Craziness Level**: 4/10
**Core Concept**: Find markets where YES + NO < 100 (guaranteed profit) or related markets misprice same event.

**Data Source**: All Kalshi markets, cross-check with other platforms
**Implementation Difficulty**: Medium
**Expected Impact**: Low volume but guaranteed profit

**Why It Might Work**:
- Arbitrage is "free money"
- Prediction markets are inefficient
- Even small arbs compound

**Why It Might Fail**:
- Arbs get closed fast
- Execution risk (legs don't fill)
- Fees might eat profit

**Implementation Sketch**:
```python
# Check for YES + NO < 98c (2c profit after spreads)
def scan_internal_arb(market):
    best_yes = market.best_ask_yes
    best_no = market.best_ask_no
    if best_yes + best_no < 98:
        return {"ticker": market.ticker, "profit": 100 - best_yes - best_no}
    return None

# Check for cross-platform arb
def scan_external_arb(kalshi_market):
    polymarket_price = get_polymarket_price(kalshi_market.topic)
    if abs(kalshi_market.yes_price - polymarket_price) > 3:
        return {"opportunity": True, "diff": diff}
```

---

### 25. Execution Quality Feedback Loop

**Craziness Level**: 3/10
**Core Concept**: Track fill rates and slippage per market, avoid markets where execution is bad.

**Data Source**: Our own trading history
**Implementation Difficulty**: Easy
**Expected Impact**: Medium - better P&L through execution

**Why It Might Work**:
- Some markets consistently have bad fills
- Avoiding them improves average execution
- Data-driven market selection

**Why It Might Fail**:
- Might avoid markets with edge due to past bad luck
- Sample size issues

**Implementation Sketch**:
```python
# Track execution quality per market
execution_stats = {}
def on_order_fill(fill):
    ticker = fill.ticker
    slippage = fill.actual_price - fill.intended_price
    execution_stats[ticker]["fills"].append(slippage)

# Compute execution score
def execution_score(ticker):
    stats = execution_stats.get(ticker, {})
    avg_slippage = np.mean(stats.get("fills", [0]))
    fill_rate = stats.get("fill_rate", 1.0)
    return fill_rate * (1 - avg_slippage / 10)  # Penalize slippage

# Factor into market selection
```

---

## META-DISCOVERY INSIGHTS

### What Makes a "Good" Market for Our Strategy?

Based on RLM validation and the ideas above, ideal markets have:

1. **Price Volatility**: Price swings create RLM signals
2. **Sufficient Liquidity**: Can execute without massive slippage
3. **Retail Presence**: Dumb money creates opportunity
4. **Short Time Horizon**: Fast resolution = capital efficiency
5. **Information Asymmetry**: Some traders know more = price drops we can exploit
6. **No Dominant Market Maker**: Wider spreads = more opportunity

### Feature Engineering for Market Quality

Combine the above into a unified "Market Quality Score":

```python
def compute_market_quality_score(market):
    features = {
        "volatility": compute_volatility(market),           # Higher = better
        "liquidity": compute_liquidity_score(market),       # Goldilocks - not too high, not too low
        "retail_ratio": estimate_retail_presence(market),   # Higher = better
        "time_to_settle": market.hours_to_settlement,       # Lower = better
        "spread": market.bid_ask_spread,                    # Mid-range optimal
        "mm_presence": detect_market_maker(market),         # Lower = better
        "imbalance": compute_orderbook_imbalance(market),   # Higher = better
        "social_velocity": get_social_velocity(market),     # Higher = better
        "whale_activity": detect_whale_activity(market),    # Higher = better
    }

    # Weighted combination (weights from backtesting)
    weights = {
        "volatility": 0.20,
        "liquidity": 0.15,
        "retail_ratio": 0.15,
        "time_to_settle": -0.10,  # Negative = shorter is better
        "spread": 0.10,
        "mm_presence": -0.10,     # Negative = less MM is better
        "imbalance": 0.10,
        "social_velocity": 0.05,
        "whale_activity": 0.05,
    }

    return sum(features[k] * weights[k] for k in features)
```

---

## PRIORITIZED IMPLEMENTATION LIST

Based on implementation difficulty, expected impact, and current infrastructure:

### Tier 1: Quick Wins (This Week)

1. **Volume Momentum Scanner** - Easy, we have the data
2. **Order Book Imbalance Screener** - Easy, we have the data
3. **Spread Decay Predictor** - Easy, we have the data
4. **Execution Quality Feedback Loop** - Easy, just track our trades

### Tier 2: Medium Effort (Next 2 Weeks)

5. **Event Clustering / Bundle Trading** - Medium, use event_ticker
6. **Price Discovery Phase Detector** - Medium, compute volatility
7. **Cross-Market Correlation Mapper** - Medium, needs correlation analysis
8. **New Market Alpha Hunter** - Medium, enhance lifecycle listener

### Tier 3: Significant Investment (Future)

9. **Sports Calendar Integration** - Needs external API integration
10. **Machine Learning Market Scorer** - Needs training pipeline
11. **Social Sentiment Spike Detector** - Needs Twitter/Reddit API
12. **Whale Tracker Enhancement** - Extend existing S013 logic

### Tier 4: Moonshots (If Everything Else Works)

13. **News Sentiment Pre-Scanner** - Complex NLP
14. **Cross-Platform Arbitrage** - Regulatory complexity
15. **Orderbook Toxicity Scorer** - Research-heavy
16. **Moon Phase Selector** - Just for laughs

---

## NEXT STEPS

1. **Validate Volume Momentum** on historical data
2. **Build Order Book Imbalance** signal into discovery scoring
3. **Track Execution Quality** starting today
4. **Prototype Event Clustering** using event_ticker field
5. **Research correlation structure** in our orderbook data

---

*Generated in LSD Mode - January 5, 2026*
*"Question everything. Test everything. Deploy what works."*
