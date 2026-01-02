# AI Context Analyzer Proposal

**Status:** PROPOSAL - Awaiting Approval  
**Priority:** P1 - High-Impact Enhancement  
**Date:** 2025-01-01  
**Author:** Trading System Architecture Team

---

## Executive Summary

The **AI Context Analyzer** is an enhancement layer for the validated RLM_NO trading strategy (S-RLM-001, +24.88% edge). Instead of replacing the proven RLM signal detection, it adds contextual intelligence to improve position sizing, filter edge cases, and enhance overall profitability through adaptive decision-making.

**Core Principle:** Enhance, don't replace. The RLM signal remains deterministic and validated. AI provides contextual reasoning to optimize execution.

**Expected Impact:**
- **Base RLM Edge:** +24.88% (optimal parameters)
- **Target Enhanced Edge:** +28-30% through better filtering and sizing
- **Win Rate Improvement:** +1-2% by avoiding low-quality contexts
- **Capital Efficiency:** +10-15% by avoiding weak signals and optimizing position sizes

---

## 1. Problem Statement

### Current State

The RLM_NO strategy is validated and profitable (+24.88% edge), but has limitations:

1. **Fixed Position Sizing:** Currently uses static sizing based on price_drop tiers (S-001 research):
   - 20c+ drop: 2.0x size
   - 10-20c drop: 1.5x size
   - 5-10c drop: 1.0x size
   - <5c drop: Skip

2. **No Contextual Filtering:** Takes all RLM signals that pass thresholds, regardless of:
   - Market category nuances
   - Time-to-close considerations
   - Liquidity conditions
   - Market title semantics (e.g., "Will X happen by Y date?")

3. **Blind to Edge Cases:** Research shows some contexts have lower edge:
   - Very high price ranges (90-100c): Only +5.1% edge
   - Low-volume markets: Higher slippage risk
   - Late lifecycle markets: Less time for price to move

### Opportunity

AI can add contextual intelligence to:
- **Filter bad contexts** (avoid 10% of trades with near-zero edge)
- **Increase sizing on high-confidence** (boost 5-10% edge through better allocation)
- **Catch edge cases** (prevent 2-3% slippage losses from liquidity issues)
- **Adapt to market semantics** (extract meaning from market titles for better context)

---

## 2. Architecture Overview

### Design Philosophy

```
┌─────────────────────────────────────────────────────────────┐
│                    RLM Signal Detection                      │
│              (Deterministic, Validated Logic)                │
│  ✅ YES ratio > 65%, 15+ trades, 5c+ price drop             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              AI Context Analyzer (Enhancement)               │
│  • Market context analysis                                  │
│  • Risk assessment                                          │
│  • Position sizing recommendation                           │
│  • Confidence multiplier (0.5x to 2.0x)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Final Trading Decision                          │
│  • Synthesize RLM signal + AI context                       │
│  • Apply position sizing with multiplier                    │
│  • Execute or skip based on risk flags                      │
└─────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **RLM Signal is Sacred:** Never override RLM signal detection. Only enhance execution.
2. **AI as Advisor:** AI recommends confidence multipliers and flags risks. Final decision is deterministic.
3. **Explainable:** All AI reasoning is logged for audit and continuous improvement.
4. **Fallback Safety:** If AI service fails, default to RLM-only mode (no degradation).

---

## 3. Detailed Design

### 3.1 Component: `MarketContextAnalyzer`

**Location:** `backend/src/kalshiflow_rl/traderv3/services/ai_context_analyzer.py`

**Responsibilities:**
- Analyze market context (title, category, timing, liquidity)
- Assess RLM signal quality in context
- Recommend confidence multiplier
- Flag risk conditions

**Interface:**

```python
@dataclass
class ContextScore:
    """AI analysis result for a market/signal combination."""
    confidence_multiplier: float  # 0.5x to 2.0x (1.0x = neutral)
    risk_flags: List[str]  # ["low_liquidity", "late_lifecycle", "unvalidated_category"]
    explanation: str  # Human-readable reasoning
    position_recommendation: str  # "skip" | "small" | "normal" | "large" | "maximum"
    confidence_score: float  # 0.0 to 1.0 (internal confidence metric)


class MarketContextAnalyzer:
    """
    AI-powered context analyzer for RLM trading signals.
    
    Uses LLM (GPT-4 or Claude) to assess market context and enhance
    position sizing decisions beyond the base RLM signal.
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4-turbo",
        temperature: float = 0.1,
        enable_ai: bool = True
    ):
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.enable_ai = enable_ai
        self._category_edge_map = self._load_category_performance()
    
    async def analyze(
        self,
        market: TrackedMarket,
        rlm_signal: RLMSignal,
        orderbook: dict,
        recent_trades: List[dict]
    ) -> ContextScore:
        """
        Analyze market context and return confidence score.
        
        Args:
            market: TrackedMarket with metadata
            rlm_signal: Validated RLM signal (already passed thresholds)
            orderbook: Current orderbook snapshot
            recent_trades: Last 50 trades for pattern analysis
            
        Returns:
            ContextScore with multiplier and risk flags
        """
        if not self.enable_ai:
            # Fallback: return neutral score
            return ContextScore(
                confidence_multiplier=1.0,
                risk_flags=[],
                explanation="AI disabled, using neutral multiplier",
                position_recommendation="normal",
                confidence_score=0.5
            )
        
        # Gather context
        context = self._gather_context(market, rlm_signal, orderbook, recent_trades)
        
        # AI analysis
        ai_response = await self._query_ai(context)
        
        # Parse and validate
        return self._parse_ai_response(ai_response, context)
```

### 3.2 Context Gathering

**Data Sources:**

```python
def _gather_context(
    self,
    market: TrackedMarket,
    rlm_signal: RLMSignal,
    orderbook: dict,
    recent_trades: List[dict]
) -> Dict[str, Any]:
    """Gather all relevant context for AI analysis."""
    
    time_to_close = market.close_ts - time.time()
    current_yes_price = rlm_signal.last_yes_price
    current_no_price = 100 - current_yes_price
    
    return {
        # Market metadata
        "market_title": market.title,
        "category": market.category,
        "event_ticker": market.event_ticker,
        "time_to_close_hours": time_to_close / 3600,
        "market_age_hours": (time.time() - market.open_ts) / 3600,
        
        # RLM signal strength
        "signal_strength": {
            "yes_ratio": rlm_signal.yes_ratio,
            "price_drop_cents": rlm_signal.price_drop,
            "n_trades": rlm_signal.n_trades,
            "first_yes_price": rlm_signal.first_yes_price,
            "last_yes_price": rlm_signal.last_yes_price,
        },
        
        # Current market conditions
        "current_prices": {
            "yes_price": current_yes_price,
            "no_price": current_no_price,
            "price_range": self._classify_price_range(current_no_price),
        },
        
        # Liquidity assessment
        "liquidity": self._assess_orderbook_liquidity(orderbook),
        
        # Volume metrics
        "volume": {
            "total_volume": market.volume,
            "volume_24h": market.volume_24h,
            "recent_trade_count": len(recent_trades),
        },
        
        # Category performance (from research)
        "category_edge": self._category_edge_map.get(market.category, None),
        
        # Trade pattern analysis
        "trade_pattern": self._analyze_trade_pattern(recent_trades[-50:]),
        
        # Price history (if available)
        "price_trend": self._get_price_trend(market.ticker, market.open_ts),
    }
```

### 3.3 AI Prompt Template

**Structured Prompt for Consistency:**

```python
RLM_CONTEXT_PROMPT = """
You are a quantitative trading expert analyzing a Reverse Line Movement (RLM) signal in a Kalshi prediction market.

## VALIDATED STRATEGY BACKGROUND

The RLM_NO strategy has been statistically validated with +24.88% expected edge when:
- >65% of trades are YES bets (retail betting on favorite)
- YES price drops (smart money betting NO against retail)
- 15+ trades observed (stable pattern)

**Position Sizing Research (S-001):**
- Price drop 20c+: +30.7% edge → Use 2.0x position size
- Price drop 10-20c: +17-19.5% edge → Use 1.5x position size  
- Price drop 5-10c: +11.9% edge → Use 1.0x position size
- Price drop <5c: Negative/marginal edge → SKIP

**Category Performance:**
- Sports (KXNFL, KXNBA, etc.): +25% edge (validated)
- Crypto (KXBTCD, etc.): +20% edge (validated)
- Entertainment: +18% edge (validated)
- Media Mentions: +15% edge (validated)
- Weather/Economics: Unvalidated (be cautious)

## CURRENT MARKET ANALYSIS

**Market:** {market_title}
**Category:** {category}
**Time to Close:** {time_to_close_hours:.1f} hours
**Market Age:** {market_age_hours:.1f} hours

**RLM Signal:**
- YES trade ratio: {yes_ratio:.1%}
- Price drop: {price_drop_cents}c ({first_yes_price}c → {last_yes_price}c)
- Trades observed: {n_trades}

**Current Prices:**
- YES: {yes_price}c
- NO: {no_price}c (target entry)
- Price range: {price_range}

**Liquidity:** {liquidity_summary}
**Volume (24h):** {volume_24h} contracts
**Recent activity:** {recent_trade_count} trades in window

**Category Historical Edge:** {category_edge}

## YOUR TASK

Assess this RLM signal quality and recommend:

1. **Confidence Multiplier** (0.5x to 2.0x):
   - 2.0x: Exceptional context, maximum size
   - 1.5x: Strong context, increase size
   - 1.0x: Normal context, use base RLM sizing
   - 0.75x: Weak context, reduce size
   - 0.5x: Poor context, minimal size
   - 0.0x: SKIP (flag as high risk)

2. **Risk Flags** (comma-separated if any):
   - `low_liquidity`: Orderbook too thin for entry/exit
   - `late_lifecycle`: Less than 2 hours to close (price may not move)
   - `unvalidated_category`: Category not in validated list
   - `high_price_range`: NO price >90c (marginal edge per research)
   - `low_volume`: Very low trading activity
   - `volatile_pattern`: Recent trades show unusual volatility
   - `correlated_position`: We already have position in related market

3. **Position Recommendation:**
   - `skip`: Don't trade (confidence_multiplier = 0.0)
   - `small`: 0.5x-0.75x multiplier
   - `normal`: 1.0x multiplier (base RLM sizing)
   - `large`: 1.5x multiplier
   - `maximum`: 2.0x multiplier

4. **Explanation:** 1-2 sentence reasoning for your recommendation

## RESPONSE FORMAT (JSON)

{{
    "confidence_multiplier": 0.0-2.0,
    "risk_flags": ["flag1", "flag2"] or [],
    "position_recommendation": "skip|small|normal|large|maximum",
    "explanation": "Brief reasoning"
}}
"""
```

### 3.4 Integration with RLMService

**Modified Decision Flow:**

```python
# In RLMService._execute_signal()

async def _execute_signal(self, signal: RLMSignal) -> None:
    """Execute RLM signal with AI context enhancement."""
    
    # Get market and orderbook
    market = self._tracked_markets.get_market(signal.market_ticker)
    orderbook = await self._get_orderbook(signal.market_ticker)
    
    # AI Context Analysis
    context_score = await self._ai_analyzer.analyze(
        market=market,
        rlm_signal=signal,
        orderbook=orderbook,
        recent_trades=self._get_recent_trades(signal.market_ticker)
    )
    
    # Apply risk flags
    if context_score.position_recommendation == "skip" or "low_liquidity" in context_score.risk_flags:
        logger.info(f"Skipping RLM signal {signal.market_ticker}: {context_score.explanation}")
        self._record_decision(signal, "skipped_ai_filter", context_score.explanation)
        return
    
    # Base position sizing (from S-001 research)
    if signal.price_drop >= 20:
        base_size = self._contracts_per_trade * 2.0
    elif signal.price_drop >= 10:
        base_size = self._contracts_per_trade * 1.5
    else:
        base_size = self._contracts_per_trade * 1.0
    
    # Apply AI confidence multiplier
    final_size = int(base_size * context_score.confidence_multiplier)
    final_size = max(1, min(final_size, self._max_position_size))  # Clamp to bounds
    
    # Execute trade
    decision = TradingDecision(
        action="buy",
        market=signal.market_ticker,
        side="no",
        quantity=final_size,
        price=self._get_best_no_price(orderbook),
        reason=f"rlm_ai:drop={signal.price_drop}c,"
               f"mult={context_score.confidence_multiplier:.2f}x,"
               f"ai={context_score.position_recommendation},"
               f"explanation={context_score.explanation[:50]}",
        confidence=context_score.confidence_multiplier
    )
    
    await self._execute_decision(decision)
    
    # Log AI reasoning for audit
    logger.info(
        f"RLM+AI: {signal.market_ticker} "
        f"base_size={base_size} "
        f"ai_mult={context_score.confidence_multiplier:.2f}x "
        f"final={final_size} "
        f"flags={context_score.risk_flags} "
        f"reasoning={context_score.explanation}"
    )
```

---

## 4. Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

**Tasks:**
1. Create `ai_context_analyzer.py` service
2. Implement context gathering functions
3. Set up LLM client (OpenAI or Anthropic)
4. Create `ContextScore` dataclass
5. Add configuration flags (`ENABLE_AI_CONTEXT`, `AI_MODEL`)

**Deliverables:**
- Service skeleton with fallback (neutral multiplier if AI disabled)
- Unit tests for context gathering

### Phase 2: AI Integration (Week 2)

**Tasks:**
1. Implement AI prompt template
2. Add JSON response parsing and validation
3. Implement response validation (ensure multiplier in range, flags valid)
4. Add error handling (fallback to neutral on AI failure)
5. Integration tests with mock LLM responses

**Deliverables:**
- Working AI analysis with real LLM
- Validation and error handling
- Test suite with various market contexts

### Phase 3: Integration with RLMService (Week 3)

**Tasks:**
1. Modify `RLMService._execute_signal()` to call AI analyzer
2. Apply confidence multipliers to position sizing
3. Implement risk flag filtering
4. Add comprehensive logging (AI reasoning in decision reason)
5. Add metrics tracking (AI multiplier distribution, skip rate)

**Deliverables:**
- End-to-end integration
- Enhanced logging and metrics
- Performance monitoring

### Phase 4: Testing & Optimization (Week 4)

**Tasks:**
1. Paper trading validation (run for 1 week)
2. Compare AI-enhanced vs RLM-only performance
3. Analyze AI decision quality (check if high-confidence = better outcomes)
4. Refine prompt based on results
5. Document findings

**Deliverables:**
- Validation report
- Prompt optimizations
- Production readiness assessment

---

## 5. Expected Outcomes

### Quantitative Improvements

| Metric | Baseline (RLM) | Target (RLM+AI) | Improvement |
|--------|----------------|-----------------|-------------|
| Expected Edge | +24.88% | +28-30% | +3-5% |
| Win Rate | 90.2% | 91-92% | +1-2% |
| Signals Traded | 100% | 85-90% | -10-15% (filtered) |
| Avg Position Size | Fixed tiers | Adaptive | +10-15% efficiency |
| Slippage Losses | ~2-3% | <1% | -1-2% |

### Qualitative Improvements

1. **Risk Reduction:** AI filters low-liquidity, late-lifecycle, and unvalidated contexts
2. **Capital Efficiency:** Better position sizing on high-confidence signals
3. **Adaptability:** AI can learn from market semantics (title analysis)
4. **Explainability:** All decisions have reasoning for audit and learning

---

## 6. Risk Mitigation

### Technical Risks

| Risk | Mitigation |
|------|------------|
| AI service downtime | Fallback to neutral multiplier (1.0x), no degradation |
| LLM hallucination | Strict JSON schema validation, range clamping |
| Latency issues | Async calls, timeout handling (default to neutral if slow) |
| Cost concerns | Cache similar contexts, batch analysis, fallback to cheaper model |

### Strategy Risks

| Risk | Mitigation |
|------|------------|
| AI overrides valid signals | AI never overrides RLM detection, only enhances sizing |
| Over-optimization | Start conservative (multiplier range 0.75x-1.5x), expand gradually |
| False confidence | Log all AI reasoning, validate against outcomes |
| Prompt drift | Version control prompts, A/B test improvements |

### Operational Risks

| Risk | Mitigation |
|------|------------|
| Increased complexity | Clear separation of concerns, comprehensive logging |
| Debugging difficulty | Structured logging, decision audit trail |
| Model updates breaking behavior | Pin model versions, test before updates |

---

## 7. Configuration

### Environment Variables

```bash
# AI Context Analyzer
ENABLE_AI_CONTEXT=true                    # Master switch
AI_CONTEXT_MODEL=gpt-4-turbo             # LLM model to use
AI_CONTEXT_TEMPERATURE=0.1               # Low temperature for consistency
AI_CONTEXT_TIMEOUT_SECONDS=5             # Max wait for AI response
AI_CONTEXT_FALLBACK_ENABLED=true         # Use neutral on failure
AI_CONTEXT_MULTIPLIER_MIN=0.5            # Minimum multiplier (safety)
AI_CONTEXT_MULTIPLIER_MAX=2.0            # Maximum multiplier (safety)

# Cost control
AI_CONTEXT_CACHE_ENABLED=true            # Cache similar contexts
AI_CONTEXT_MAX_REQUESTS_PER_MINUTE=60    # Rate limiting
```

### Feature Flags

```python
# Gradual rollout
AI_CONTEXT_ENABLED_CATEGORIES=["sports", "crypto"]  # Start with validated categories
AI_CONTEXT_PERCENTAGE_ROLLOUT=50                    # 50% of signals get AI analysis
```

---

## 8. Monitoring & Metrics

### Key Metrics to Track

1. **AI Performance:**
   - AI multiplier distribution (histogram)
   - Skip rate (signals filtered by AI)
   - Average confidence multiplier
   - AI response time (p50, p95, p99)

2. **Trading Performance:**
   - Win rate by AI multiplier tier
   - Edge by AI multiplier tier
   - Comparison: AI-enhanced vs RLM-only (A/B test)

3. **Risk Metrics:**
   - Risk flags triggered frequency
   - Losses in high-confidence vs low-confidence trades
   - Liquidity issues caught by AI

### Logging

```python
# Every AI analysis logged with:
{
    "market_ticker": "KXNFL-25JAN05-DET",
    "rlm_signal": {...},
    "context_score": {
        "confidence_multiplier": 1.5,
        "risk_flags": [],
        "explanation": "...",
        "position_recommendation": "large"
    },
    "final_decision": {
        "action": "buy",
        "quantity": 15,
        "base_size": 10,
        "multiplier_applied": 1.5
    },
    "timestamp": 1704067200
}
```

---

## 9. Success Criteria

### Validation Period (4 weeks paper trading)

**Must Achieve:**
1. ✅ Edge improvement: +2% minimum (from +24.88% to +26.88%)
2. ✅ Win rate: No degradation (maintain 90%+)
3. ✅ AI uptime: >99% (fallback works on failures)
4. ✅ Latency: <5 seconds per analysis (async, non-blocking)

**Nice to Have:**
1. Edge improvement: +3-5% (from +24.88% to +28-30%)
2. Win rate improvement: +1-2%
3. Slippage reduction: -1-2%
4. Capital efficiency: +10-15%

### Approval for Production

- ✅ All "Must Achieve" criteria met
- ✅ No systematic edge degradation
- ✅ AI reasoning quality validated (high confidence = better outcomes)
- ✅ Team approval

---

## 10. Future Enhancements

### Phase 2 Ideas (Post-Validation)

1. **Learning from Outcomes:**
   - Store AI decisions and outcomes
   - Fine-tune prompts based on performance
   - Build confidence calibration model

2. **Advanced Context:**
   - Market title semantic analysis (NLP)
   - Event correlation detection (related markets)
   - Time-of-day patterns (drunk betting windows)

3. **Multi-Strategy Support:**
   - Extend to other validated strategies (S010, S013)
   - Strategy-specific context analysis
   - Portfolio-level risk assessment

4. **Self-Improvement:**
   - Automated prompt optimization
   - Performance feedback loops
   - Continuous learning from trades

---

## 11. Dependencies

### External Services
- OpenAI API (or Anthropic Claude API)
- Existing RLMService
- Orderbook service
- TrackedMarketsState

### Internal Dependencies
- `backend/src/kalshiflow_rl/traderv3/services/rlm_service.py`
- `backend/src/kalshiflow_rl/traderv3/state/tracked_markets.py`
- LangChain (for LLM abstraction)
- OpenAI/Anthropic SDK

### Research Dependencies
- Category performance data (`VALIDATED_STRATEGIES.md`)
- S-001 position sizing research
- RLM validation research

---

## 12. References

- **VALIDATED_STRATEGIES.md**: RLM_NO strategy validation (S-RLM-001)
- **RLM_IMPROVEMENTS.md**: S-001 position sizing research
- **research/hypotheses/rlm_enhancement_research.md**: Signal strength analysis
- **trader_v3_architecture.md**: System architecture reference

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2025-01-01 | Initial proposal | Architecture Team |

---

**Next Steps:**
1. Review and approve proposal
2. Assign implementation owner
3. Set up development branch
4. Begin Phase 1 implementation
