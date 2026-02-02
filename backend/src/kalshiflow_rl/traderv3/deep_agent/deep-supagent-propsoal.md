# Deep Agent Sub-Agent Proposal

Honest assessment of sub-agent opportunities for the SelfImprovingAgent system.
Based on analysis of the current architecture: Sonnet 4 backbone, 31+ tools, 120s cycle interval, 40-message history limit, ~$3-6/hour baseline cost.

## Current State

The deep agent already runs one sub-agent successfully:

**News Intelligence (DEPLOYED)** -- Haiku wrapping GDELT article analysis via `get_news_intelligence` / `GdeltNewsAnalyzer`. Cached 15min, ~$0.003/call. This is the textbook sub-agent pattern: massive unstructured data (article text) compressed into structured intelligence by a cheap model. It works because the raw data genuinely needs LLM summarization.

---

## Recommended Sub-Agents (ranked by ROI)

### 1. Async Reflection Sub-Agent (HIGH VALUE)

**Problem**: `reflect()` runs inside the main Sonnet cycle. When trades settle in batch (common at market close), multiple reflections block the trading loop. Reflection doesn't need Sonnet's reasoning depth -- it's pattern matching on outcomes.

**Proposal**:
- Model: Haiku 4.5
- Trigger: `reflection.py::_handle_settlement()` queues async task instead of blocking
- Tools: `append_memory`, `read_memory` (write to same memory files the main agent reads)
- Output: Pattern identified, mistake classification (if loss), strategy update recommendation, confidence level. Max 300 words per trade.
- Cost: ~$0.001-0.002 per settled trade (vs ~$0.01+ for Sonnet)

**Why it works**: Reflection is write-heavy and formulaic. The sub-agent receives structured input (entry reasoning, extraction snapshot at entry, microstructure context, outcome) and produces structured output (memory file updates). Haiku handles this well. The main agent cycle continues trading unblocked.

**Risk**: Low. Reflections are async and non-blocking by design. Bad reflection = slightly worse memory update, caught on next consolidation. No direct trading impact.

**Savings**: ~60% cost reduction per reflection, plus elimination of cycle-blocking during batch settlements.

### 2. Event Research Sub-Agent (MEDIUM VALUE)

**Problem**: `understand_event()` makes a full Sonnet call to build extraction specs for newly discovered events. Runs once per new event but blocks the cycle while it reasons about entity extraction patterns.

**Proposal**:
- Model: Haiku 4.5 (with Sonnet fallback for complex multi-entity events)
- Trigger: `_bootstrap_new_events()` spawns async task
- Input: Event metadata, sample tickers, market descriptions
- Output: Extraction spec (entities to track, sentiment keywords, magnitude thresholds)
- Cost: ~$0.002 per event (one-time)

**Why it works**: Most events follow predictable patterns (binary outcome, 1-3 key entities, standard sentiment dimensions). Haiku can generate extraction specs for straightforward events. Complex events (multi-leg, conditional outcomes) fall back to Sonnet.

**Risk**: Medium. A bad extraction spec means bad signal extraction for the event's lifetime. Mitigation: `evaluate_extractions` and `refine_event` tools already exist for the main agent to course-correct.

**Savings**: ~70% cost per event spec, plus unblocked cycle during event onboarding.

### 3. Microstructure Pattern Detector (NICE-TO-HAVE)

**Problem**: Main agent manually parses L2 orderbook data via `get_microstructure` to detect whale activity, liquidity shifts, and momentum patterns. This is mechanical pattern-matching consuming expensive Sonnet tokens.

**Proposal**:
- Model: Haiku 4.5
- Trigger: Pre-cycle, alongside signal pre-fetch
- Input: Raw orderbook snapshot, recent trade flow
- Output: Structured alert (whale detected Y/N, liquidity shift direction, momentum score)
- Cost: ~$0.002 per market per cycle

**Risk**: Medium. Pattern detection quality directly affects trade decisions. Haiku may miss subtle signals that Sonnet catches. Requires careful threshold calibration.

**Deferral reason**: The main agent currently calls `get_microstructure` selectively (not every cycle, not every market). Making this a sub-agent requires deciding which markets to analyze pre-emptively, which may waste calls on quiet markets.

---

## Explicitly Rejected: Signal Summarizer Sub-Agent

Early analysis ranked a "signal summarizer" (Haiku pre-processing extraction signals before Sonnet) as high-ROI. Deeper investigation proved it **net-negative**. Five reasons:

1. **Signals are already free**. Extraction signals come from a Supabase RPC call (~50ms, zero LLM tokens) in `_build_cycle_context()`. The data arrives pre-formatted as compact markdown. There's nothing expensive to optimize.

2. **Sonnet needs exact values**. The system prompt teaches threshold-based reasoning: `consensus > 0.7` = strong conviction, `first_seen_at < 30 min` = fresh edge, `magnitude < 3` = skip. A Haiku summary rounding "0.67 consensus" to "moderate consensus" would cross the 0.7 threshold boundary incorrectly.

3. **Truncation is mostly irrelevant**. The 4K char truncation applies to tool results during the cycle (when Sonnet re-queries), and most per-ticker queries return <1K. The pre-fetched context message is never truncated.

4. **It adds cost and latency for worse data**. You'd pay for a Haiku call every cycle to produce a lossy summary that Sonnet then interprets, when currently Sonnet just reads exact numbers directly.

5. **The pattern already exists where it's needed**. `get_news_intelligence` wraps GDELT data (massive, unstructured article text) through Haiku -- a genuine case where LLM summarization adds value. Extraction signals are already structured and compact (one line per market, ~2-5K chars for 20 signals).

**Lesson**: Sub-agents add value when raw data is large/unstructured and the consumer needs a compressed representation. They destroy value when the data is already compact/structured and the consumer needs precision.

---

## Non-Sub-Agent Optimizations (Higher ROI Than Most Sub-Agents)

These code-level changes deliver more savings than adding new LLM calls.

### Priority 1: Memory File Caching

**Problem**: `strategy.md`, `golden_rules.md`, and `cycle_journal.md` are read every cycle via disk I/O and injected into context. These files rarely change between cycles.

**Fix**: In-memory cache with write-through invalidation.
```python
class MemoryCache:
    _cache: dict[str, str] = {}
    _write_times: dict[str, float] = {}

    async def read_with_cache(self, filename: str, ttl: int = 120) -> str:
        if filename in self._cache and time.time() - self._write_times.get(filename, 0) < ttl:
            return self._cache[filename]
        content = await self._read_from_disk(filename)
        self._cache[filename] = content
        self._write_times[filename] = time.time()
        return content
```

**Savings**: ~1,200 tokens/cycle eliminated from redundant reads. At 30 cycles/hour = ~36,000 tokens/hour.

### Priority 2: Tool Result Summarization (Code, Not LLM)

**Problem**: Tools like `get_trade_log` and `get_true_performance` return 2-5K+ tokens of structured data. Most of it gets truncated to 4K anyway, losing the useful parts.

**Fix**: Per-tool result formatters that extract the decision-relevant fields before context injection.
```python
def summarize_tool_result(tool_name: str, result: Any) -> str:
    formatters = {
        "get_trade_log": lambda r: f"Trades: {len(r['trades'])}, Win rate: {r['win_rate']*100:.0f}%",
        "get_true_performance": lambda r: f"Positions: {r['position_count']}, P&L: ${r['total_pnl_cents']/100:.0f}",
    }
    formatter = formatters.get(tool_name)
    if formatter and len(str(result)) > 1000:
        return formatter(result)
    return result
```

**Savings**: ~40-50% reduction in tool output context.

### Priority 3: Tool Result Caching (1-Cycle TTL)

**Problem**: Deterministic tools (`get_session_state`, `get_markets`, `get_event_context`) are called multiple times per cycle. `preflight_check` calls `get_event_context` internally, then the agent may call it again.

**Fix**: 1-cycle TTL cache, invalidated on state change (trade execution, new market).

**Savings**: 10-20% reduction in tool calls per cycle.

### Priority 4: Pre-Filter Large Tool Results

**Problem**: News tools return massive payloads that get truncated.

**Fix**: Server-side limits before returning results.
```python
if tool_name == "get_reddit_daily_digest":
    result['posts'] = result.get('posts', [])[:10]
elif tool_name == "query_gdelt_news":
    result['top_articles'] = result.get('top_articles', [])[:5]
```

**Savings**: ~30% context reduction on news-heavy cycles.

---

## Implementation Order

| Phase | Change | Type | Impact |
|-------|--------|------|--------|
| 1 | Memory file caching | Code | ~36K tokens/hr saved |
| 2 | Tool result summarization | Code | ~40% tool context reduction |
| 3 | Async reflection sub-agent | Sub-agent | Unblock batch settlements, 60% reflection cost reduction |
| 4 | Tool result caching | Code | 10-20% fewer tool calls |
| 5 | Event research sub-agent | Sub-agent | ~70% event spec cost reduction |
| 6 | Microstructure detector | Sub-agent | Selective, evaluate after 1-5 |

Phases 1-2 are pure code changes with no new LLM calls, no risk, and the highest token savings. Phase 3 is the first real sub-agent addition. Phases 4+ are incremental.

---

## Cost Model

Current baseline: ~$0.03-0.05/cycle idle, ~$0.12-0.20/cycle active. ~$3-6/hour.

After all optimizations:
- Memory caching: -$0.10/hour (token savings)
- Tool summarization: -$0.15/hour (context reduction)
- Async reflection: -$0.05/hour (Haiku vs Sonnet for reflections) + eliminates batch settlement blocking
- Tool caching: -$0.05/hour (fewer redundant calls)

**Estimated post-optimization: ~$2.50-5/hour** (15-20% reduction), with the primary win being cycle availability (no blocking on reflections or event research).
