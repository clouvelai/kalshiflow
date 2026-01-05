# Market Maker Strategy Hypotheses from External Research

**Generated**: 2026-01-05
**Sources**: Academic papers (SSRN, arXiv), sports betting literature, crypto/DeFi research, market microstructure papers

---

## ACADEMIC: Prediction Market Research

### H-EXT-001: Favorite-Longshot Bias Exploitation
**Source**: [Makers and Takers: The Economics of the Kalshi Prediction Market](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5502658) (SSRN, 2025)

**Core Idea**: Kalshi exhibits systematic favorite-longshot bias where low-price contracts win far less often than implied, while high-price contracts win more often than implied.

**Signal Definition**:
- BUY when contract price > 80 cents (favorites are underpriced)
- SELL/AVOID when contract price < 20 cents (longshots are overpriced)
- Expected edge: ~3-5% on extreme favorites based on traditional betting literature

**Data Required**: Historical trade prices, settlement outcomes, price at entry
**Priority**: HIGH - Direct Kalshi research with transaction-level data

---

### H-EXT-002: Large Trade Order Imbalance Signal
**Source**: [Price Discovery and Trading in Prediction Markets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5331995) (SSRN, 2025)

**Core Idea**: Net order imbalance of large trades significantly predicts subsequent returns in prediction markets.

**Signal Definition**:
- Track cumulative large trade imbalance (trades > $500 or top 10% by size)
- BUY when large trade imbalance is strongly positive
- SELL when large trade imbalance is strongly negative
- Expected edge: Large traders are informed; following their direction yields alpha

**Data Required**: Trade size, trade side (yes/no), rolling imbalance calculation
**Priority**: HIGH - Validated on Kalshi/Polymarket 2024 election data

---

### H-EXT-003: Cross-Platform Arbitrage Signals
**Source**: [Price Discovery and Trading in Prediction Markets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5331995) (SSRN, 2025)

**Core Idea**: Significant price disparities exist between Kalshi, Polymarket, and PredictIt, with Polymarket leading price discovery.

**Signal Definition**:
- When Kalshi price diverges >5% from Polymarket on same event, trade toward Polymarket price
- Polymarket moves predict Kalshi moves (lag relationship)
- Expected edge: 2-5% on convergence trades

**Data Required**: Cross-platform price feeds (would require Polymarket API integration)
**Priority**: MEDIUM - Requires external data source

---

### H-EXT-004: Market Rebalancing Arbitrage
**Source**: [Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets](https://arxiv.org/abs/2508.03474) (arXiv, 2025)

**Core Idea**: Within a single market, related contracts can become mispriced relative to each other, creating riskless arbitrage.

**Signal Definition**:
- For multi-outcome markets, monitor if sum of YES probabilities deviates from 100%
- Buy underpriced outcomes, sell overpriced when sum > 100%
- Expected edge: Variable, but risk-free when identified

**Data Required**: All contract prices within a market, bid/ask spreads
**Priority**: MEDIUM - Requires multi-contract market analysis

---

## SPORTS BETTING: Sharp Money Signals

### H-EXT-005: Closing Line Value (CLV) Tracking
**Source**: [Sports Betting CLV Research](https://www.thelines.com/betting/closing-line-value/)

**Core Idea**: Professional bettors measure success by beating the closing line, not win rate. This can be adapted to prediction markets.

**Signal Definition**:
- Track your entry price vs. price at market close/settlement
- Systematic positive CLV indicates edge
- Focus on markets where your entry beats the final pre-settlement price
- Expected edge: 2% CLV corresponds to ~4% ROI in traditional sports betting

**Data Required**: Entry price, closing price, settlement time
**Priority**: HIGH - Core metric for evaluating any strategy

---

### H-EXT-006: Steam Move Detection
**Source**: [Sports Betting Sharp Money Analysis](https://vsin.com/how-to-bet/the-importance-of-closing-line-value/)

**Core Idea**: When sharp money enters a market, prices move rapidly across multiple books. Early detection allows riding the wave.

**Signal Definition**:
- Detect rapid price movement (>3% in <5 minutes) accompanied by volume spike
- Enter in direction of steam move before full price adjustment
- Exit once price stabilizes at new level
- Expected edge: 1-2% by capturing price momentum

**Data Required**: High-frequency price data, volume time series, velocity of price change
**Priority**: HIGH - Testable with existing trade data

---

### H-EXT-007: Key Number Exploitation
**Source**: [NFL Betting Key Numbers](https://www.boydsbets.com/closing-line-value/)

**Core Idea**: In sports betting, certain numbers (3, 7 in NFL) are "key numbers" where many games land. Crossing these numbers provides outsized value.

**Signal Definition**:
- For spread-based Kalshi markets, identify if price crosses key thresholds
- BUY when price moves from 49 to 51 cents (crossing 50% probability threshold)
- Expected edge: 5-16% value when crossing key psychological barriers

**Data Required**: Price movement around psychological thresholds (25%, 50%, 75%)
**Priority**: MEDIUM - Requires adaptation to prediction market context

---

### H-EXT-008: Market-Maker vs. Retail Flow Identification
**Source**: [Sportsbook Business Models](https://www.oddsshopper.com/articles/betting-101/market-maker-sportsbooks-vs-line-following-books-the-key-to-ev-nfl-betting-y10)

**Core Idea**: Market makers adjust prices based on sharp action; retail books copy with delay. The delay creates opportunity.

**Signal Definition**:
- Identify when sophisticated traders (large size, consistent patterns) enter
- Follow their direction before price fully adjusts
- Expected edge: 1-3% on informed flow detection

**Data Required**: Trader identification (wallet/account patterns), trade timing, size clustering
**Priority**: HIGH - Could combine with large trade imbalance signal

---

## CRYPTO/DEFI: AMM and MEV Research

### H-EXT-009: Inventory-Adjusted Reservation Price (Avellaneda-Stoikov)
**Source**: [Avellaneda-Stoikov Market Making Model](https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf)

**Core Idea**: Optimal bid/ask quotes depend on current inventory position, not just mid-price. Skew quotes based on inventory to manage risk.

**Signal Definition**:
- Reservation Price = Mid - (inventory * gamma * sigma^2 * time_remaining)
- When holding YES contracts, skew bid lower (more willing to sell)
- When holding NO contracts, skew ask higher (more willing to buy)
- Adjust spread based on time to settlement

**Data Required**: Current inventory, price volatility, time to expiration
**Priority**: HIGH - Foundational market making model

---

### H-EXT-010: Impermanent Loss Avoidance (Flash LP Strategy)
**Source**: [Impermanent Loss in Uniswap v3](https://arxiv.org/abs/2111.09192) (arXiv, 2021)

**Core Idea**: Long-term liquidity providers suffer impermanent loss to informed traders. Only "flash LPs" who provide liquidity for very short periods (single blocks) consistently profit.

**Signal Definition**:
- Provide liquidity (quotes) only when volatility is low
- Withdraw/widen quotes when detecting informed flow
- Time liquidity provision around expected low-information periods
- Expected edge: Avoid adverse selection during high-volatility periods

**Data Required**: Volatility measures, trade clustering, informed flow proxies
**Priority**: MEDIUM - Defensive strategy for market making

---

### H-EXT-011: Sandwich Attack Defense Patterns
**Source**: [MEV Protection Research](https://cow.fi/learn/mev-attacks-explained)

**Core Idea**: In crypto, sandwich attacks profit by front-running and back-running victim trades. Understanding this pattern helps avoid being the victim and potentially profit from detecting it.

**Signal Definition**:
- Detect rapid buy-sell-buy or sell-buy-sell patterns in quick succession
- Avoid placing large market orders during high-activity periods
- Use limit orders instead of market orders to avoid being sandwiched
- Expected edge: -1% to -2% saved by avoiding adverse execution

**Data Required**: Trade sequence patterns, trade timing microseconds apart
**Priority**: LOW - Less applicable to Kalshi's latency profile

---

### H-EXT-012: Whale Wallet Following (Copy Trading)
**Source**: [Polymarket Whale Tracking](https://www.polywhaler.com/)

**Core Idea**: Only ~17% of Polymarket wallets are profitable. Following consistently profitable large traders yields positive expected value.

**Signal Definition**:
- Identify traders with >60% win rate over >50 trades
- Mirror their positions with small delay (seconds to minutes)
- Expected edge: Variable, but copy trading has extracted significant profits

**Data Required**: Trader ID (if available), historical trade performance, position sizing
**Priority**: MEDIUM - Requires trader identification capability

---

## TRADITIONAL MICROSTRUCTURE: Toxicity and Informed Trading

### H-EXT-013: VPIN-Based Flow Toxicity
**Source**: [Flow Toxicity and Liquidity](https://www.stern.nyu.edu/sites/default/files/assets/documents/con_035928.pdf) (NYU Stern)

**Core Idea**: VPIN (Volume-Synchronized Probability of Informed Trading) measures order flow toxicity in real-time. High VPIN predicts adverse market events.

**Signal Definition**:
- Calculate VPIN using bulk volume classification
- When VPIN > threshold, widen spreads / reduce position size
- When VPIN is low, tighten spreads aggressively
- Expected edge: Avoid adverse selection during toxic flow periods

**Data Required**: Volume-bucketed trade classification, imbalance calculation
**Priority**: HIGH - Directly applicable to market making risk management

---

### H-EXT-014: Order Flow Imbalance Momentum
**Source**: [Order Flow Imbalance Research](https://www.tandfonline.com/doi/full/10.1080/14697688.2024.2358963) (Quantitative Finance, 2024)

**Core Idea**: There exists a positive relation between lagged order imbalances and subsequent returns. Imbalance-based strategies yield statistically significant profits.

**Signal Definition**:
- Calculate order flow imbalance: (Buy volume - Sell volume) / Total volume
- BUY if previous period's imbalance is positive
- SELL if previous period's imbalance is negative
- Hold position until imbalance reverses
- Expected edge: Monthly returns of 2.18% in extreme cases

**Data Required**: Trade side classification, volume aggregation, lag calculation
**Priority**: HIGH - Well-validated signal across multiple asset classes

---

### H-EXT-015: Kyle's Lambda-Based Adverse Selection Detection
**Source**: [Kyle Lambda Research](https://haas.berkeley.edu/wp-content/uploads/asymeasures21.pdf) (UC Berkeley)

**Core Idea**: Kyle's lambda measures price impact per dollar of order flow. High lambda indicates informed trading; market makers should widen spreads.

**Signal Definition**:
- Estimate lambda = Price change / Order flow
- When lambda is high (price moves a lot per $ traded), widen spreads
- When lambda is low (price stable despite volume), tighten spreads
- Expected edge: Better inventory management, reduced adverse selection

**Data Required**: Trade price, trade size, time series for regression
**Priority**: MEDIUM - Requires sufficient data for regression estimation

---

### H-EXT-016: Time-of-Day Liquidity Patterns
**Source**: [Market Microstructure Literature](https://www.princeton.edu/~markus/teaching/Eco467/05Lecture/04a_MarketMaking.pdf) (Princeton)

**Core Idea**: Liquidity and spreads follow predictable intraday patterns. Market making is more profitable during high-liquidity periods.

**Signal Definition**:
- Map trade volume and spread patterns by hour-of-day
- Provide aggressive liquidity during historically high-volume periods
- Widen spreads or reduce activity during low-volume periods
- Expected edge: 0.5-1% improvement in execution quality

**Data Required**: Hourly trade volume, spread patterns, time-of-day analysis
**Priority**: MEDIUM - Easy to implement, modest expected edge

---

## COMBINED STRATEGIES

### H-EXT-017: Sharp Flow + Favorite Bias Combo
**Source**: Combined academic insights

**Core Idea**: Combine large trade imbalance signal (sharp money) with favorite-longshot bias for higher conviction trades.

**Signal Definition**:
- BUY when: (contract price > 70 cents) AND (large trade imbalance > +20%)
- SELL when: (contract price < 30 cents) AND (large trade imbalance < -20%)
- Expected edge: 5-8% by stacking two independent signals

**Data Required**: All of the above
**Priority**: HIGH - Combines best signals from research

---

### H-EXT-018: Volatility-Adjusted Spread Management
**Source**: [Avellaneda-Stoikov](https://medium.com/hummingbot/a-comprehensive-guide-to-avellaneda-stoikovs-market-making-strategy-102d64bf5df6) + [VPIN](https://medium.com/@kryptonlabs/vpin-the-coolest-market-metric-youve-never-heard-of-e7b3d6cbacf1)

**Core Idea**: Dynamically adjust spread based on both inventory position and flow toxicity.

**Signal Definition**:
- Base spread = f(inventory, volatility, time_to_expiry)
- Spread multiplier = g(VPIN) where high VPIN = wider spread
- Optimal spread = Base spread * VPIN_multiplier
- Expected edge: 1-2% improvement in P&L by avoiding adverse selection

**Data Required**: Inventory tracking, volatility estimation, VPIN calculation
**Priority**: HIGH - Core risk management for market making

---

## SUMMARY: Priority Ranking

### Tier 1 - Immediate Testing (HIGH priority, testable with existing data)
1. **H-EXT-002**: Large Trade Order Imbalance Signal
2. **H-EXT-006**: Steam Move Detection
3. **H-EXT-014**: Order Flow Imbalance Momentum
4. **H-EXT-001**: Favorite-Longshot Bias Exploitation
5. **H-EXT-013**: VPIN-Based Flow Toxicity
6. **H-EXT-017**: Sharp Flow + Favorite Bias Combo

### Tier 2 - Requires Implementation Work (MEDIUM priority)
7. **H-EXT-009**: Inventory-Adjusted Reservation Price
8. **H-EXT-018**: Volatility-Adjusted Spread Management
9. **H-EXT-008**: Market-Maker vs. Retail Flow Identification
10. **H-EXT-005**: CLV Tracking (evaluation metric)

### Tier 3 - Requires External Data or More Research (LOWER priority)
11. **H-EXT-003**: Cross-Platform Arbitrage (needs Polymarket feed)
12. **H-EXT-004**: Market Rebalancing Arbitrage
13. **H-EXT-012**: Whale Wallet Following
14. **H-EXT-007**: Key Number Exploitation
15. **H-EXT-015**: Kyle's Lambda Detection
16. **H-EXT-016**: Time-of-Day Patterns
17. **H-EXT-010**: Flash LP Strategy
18. **H-EXT-011**: Sandwich Attack Defense

---

## Next Steps

1. Start with **H-EXT-002** and **H-EXT-006** - these are directly testable with the ~7.9M trade dataset
2. Implement **H-EXT-014** order flow imbalance as a core signal
3. Build **H-EXT-001** favorite-longshot bias analysis using settlement data
4. Develop **H-EXT-013** VPIN calculator for risk management

---

## Sources

### Academic Papers
- [Makers and Takers: The Economics of the Kalshi Prediction Market](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5502658) - Burgi, Deng, Whelan (SSRN, 2025)
- [Price Discovery and Trading in Prediction Markets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5331995) - Ng, Peng, Tao, Zhou (SSRN, 2025)
- [SoK: Market Microstructure for Decentralized Prediction Markets](https://arxiv.org/abs/2510.15612) - Rahman, Al-Chami, Clark (arXiv, 2025)
- [Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets](https://arxiv.org/abs/2508.03474) - arXiv, 2025
- [Impermanent Loss in Uniswap v3](https://arxiv.org/abs/2111.09192) - arXiv, 2021
- [Flow Toxicity and Liquidity in a High Frequency World](https://www.stern.nyu.edu/sites/default/files/assets/documents/con_035928.pdf) - Easley, Lopez de Prado, O'Hara (NYU Stern)
- [High-frequency trading in a limit order book](https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf) - Avellaneda & Stoikov
- [Order Flow Imbalance Research](https://www.tandfonline.com/doi/full/10.1080/14697688.2024.2358963) - Quantitative Finance, 2024
- [Explaining the Favorite-Longshot Bias](https://www.journals.uchicago.edu/doi/abs/10.1086/655844) - Journal of Political Economy

### Sports Betting
- [Closing Line Value Explained](https://www.thelines.com/betting/closing-line-value/)
- [CLV Importance](https://vsin.com/how-to-bet/the-importance-of-closing-line-value/)
- [Market Maker vs Retail Sportsbooks](https://www.oddsshopper.com/articles/betting-101/market-maker-sportsbooks-vs-line-following-books-the-key-to-ev-nfl-betting-y10)
- [Pinnacle Sharp Book Analysis](https://www.pinnacleoddsdropper.com/blog/sharp-sportsbook)

### Crypto/DeFi
- [MEV Explained](https://a16zcrypto.com/posts/article/mev-explained/) - a16z crypto
- [MEV Protection Guide](https://cow.fi/learn/mev-attacks-explained) - CoW Protocol
- [Polymarket Whale Tracker](https://www.polywhaler.com/)
- [Avellaneda-Stoikov Strategy Guide](https://hummingbot.org/blog/guide-to-the-avellaneda--stoikov-strategy/) - Hummingbot

### Market Microstructure
- [VPIN Introduction](https://www.quantresearch.org/From%20PIN%20to%20VPIN.pdf) - QuantResearch
- [Market Making Lecture Notes](https://www.princeton.edu/~markus/teaching/Eco467/05Lecture/04a_MarketMaking.pdf) - Princeton
