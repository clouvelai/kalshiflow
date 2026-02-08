# Kalshi API Architect - Memory

## Key Architecture Patterns

### Module Dependency Injection
- Pattern: Module-level globals (`_index`, `_trading_client`, etc.) set via `set_dependencies()` at startup
- Used in: `tools.py`, `mentions_tools.py`, all tool modules
- Coordinator calls `set_dependencies()` after creating shared services
- Clean separation: tools don't import coordinator, dependencies flow down

### EventUnderstanding System
- Central abstraction for event context across ALL events (not just mentions)
- 4-hour TTL cache at `memory/data/understanding/understanding_{event_ticker}.json`
- Pipeline: Identity (Kalshi API) → Wikipedia → Extensions → LLM Synthesis → Cache
- Extension pattern for domain-specific enrichment (MentionsExtension)
- UnderstandingBuilder is the orchestrator

### Budget Tracking Pattern
- SimulationBudgetManager pattern: track usage WITHOUT hard enforcement
- Per-event tracking with phase-based scheduling
- Soft warnings at milestones, graceful degradation on exhaustion
- Similar pattern works for API credits (Tavily) as well as LLM calls

### Web Search Current State
- DuckDuckGo ONLY (sync via `duckduckgo_search` library)
- Used ~20 times in mentions_context.py for: speaker search, entity resolution, term relevance
- Wikipedia via async httpx (REST API)
- No caching for web search results (only EventContext cached)

## Tavily Integration Design Decisions

### Why Tavily Primary
- Better quality than DDG (scoring, recency, deduplication)
- Structured API (Search/Research/Extract)
- 1000 free credits/month sufficient for demo
- Async-first (tavily-python SDK)

### Where Tavily Goes
- **YES**: EventUnderstanding news enrichment (universal)
- **YES**: Captain ad-hoc news search tool
- **NO**: Mentions simulation context gathering (keep DDG for now)
- **NO**: Research API in Phase 1 (Search API covers 90%)

### Fallback Strategy
- Always have DDG as fallback (no external dependency risk)
- Tavily error → log warning → transparent DDG fallback
- Budget exhausted → log info → transparent DDG fallback
- Never fail loudly - graceful degradation

### Caching Strategy
- Tavily results: 4-hour TTL (same as EventUnderstanding)
- Cache key: `{query}:{params_hash}`
- Cache includes DDG fallback results (reduce redundant calls)
- EventUnderstanding cache includes news (no separate news cache)

## Implementation Notes

### File Locations
- `tavily_service.py` - New unified search abstraction
- `tavily_budget.py` - Budget tracking (pattern from SimulationBudgetManager)
- `event_understanding.py` - Add `news_context` fields, `_enrich_news()` method
- `coordinator.py` - Wire TavilySearchService into UnderstandingBuilder
- `tools.py` - Add `search_event_news()` tool

### Config Pattern
- V3Config dataclass with env var loading
- Naming: `V3_TAVILY_*` prefix for trader-specific config
- `TAVILY_API_KEY` at root (not V3-prefixed, used by SDK)
- Defaults that work without config (fallback to DDG)

### Tool Interface
- Automatic: Captain sees news in EventUnderstanding (via `get_event_snapshot()`)
- Manual: Captain calls `search_event_news(event_ticker, query)` for ad-hoc
- Update: `update_understanding(event_ticker, force_refresh=True)` triggers fresh news fetch

## Lessons Learned

### EventUnderstanding is the right place for news
- Universal context for ALL events (not mentions-specific)
- Already has Wikipedia, LLM synthesis, caching
- News is just another enrichment layer
- Extensions can use news in their enrichment

### Budget tracking should be soft
- Visibility > enforcement
- Graceful degradation better than hard failure
- Log warnings at milestones (50%, 75%, 90%)
- Let user decide whether to continue beyond budget

### Keep mentions separate (for now)
- Mentions already works with DDG
- Don't mix concerns in Phase 1
- Can migrate mentions to Tavily in Phase 4 if needed
- Separation = easier testing and rollback

## News Intelligence System Design (2026-02-07)

### Core Architecture Decision: Extend `agent_memories` vs New Table
**DECISION: Extend existing `agent_memories` table with news-specific columns.**

**Rationale:**
- Already has pgvector HNSW index (sub-10ms similarity search at 100K rows)
- VectorMemoryService handles embedding + deduplication (0.88 threshold)
- Memory consolidation pattern applies to news (cluster similar articles)
- Avoids duplicate embedding costs ($0.02/1M tokens) and dual index maintenance
- Captain tools already use `VectorMemoryService.search()` - no new API surface

**Schema Design Patterns:**
1. **Nullable news columns** - Only populated for `memory_type = 'news'`
2. **JSONB for flexibility** - `price_snapshot` keyed by market_ticker, easy extension
3. **Separate correlation table** - `news_price_impacts` tracks news → price causality
4. **Timeline as JSONB snapshots** - `event_timeline` stores versioned event state

### Performance Characteristics
- **Semantic news search**: ~210ms (200ms OpenAI embedding + 10ms HNSW scan)
- **Price impact query**: ~5ms (indexed JOIN on news_id + market_ticker)
- **Event timeline**: ~3ms (single-table scan with composite index)
- **Impact detection**: ~10-20ms per event (acceptable for 30min background loop)

### Storage Economics
- **Year 1**: 60K articles × 6KB data + 12KB embedding = ~1.08GB total
- **Cost**: $0 database (under 8GB Supabase free tier), $0.36 OpenAI embeddings
- **Scaling**: HNSW is logarithmic, <50ms at 1M articles

### Integration Points
1. **NewsStore** - Persist articles after EventUnderstanding news fetch
2. **ImpactDetector** - Correlate news with candlestick price changes (30min intervals)
3. **TimelineManager** - Snapshot event state on understanding refresh, price moves, news bursts
4. **Captain tools** - `search_news_semantic()`, `get_price_movers()`, `get_event_timeline()`

### Key Design Patterns
- **Async persistence** - News storage NEVER blocks Captain cycle (60s)
- **Graceful degradation** - All news features optional, system works without
- **Correlation heuristics** - Time-based (±30min), confidence by article density
- **Retention policies** - Keep news indefinitely, purge timeline >28 days

### Lessons from Design Process
1. **Reuse > rebuild** - Extending agent_memories is 10× simpler than new table + new indexes
2. **JSONB flexibility** - Price snapshots + timeline data evolve without schema changes
3. **Separate concerns** - NewsStore (persistence), ImpactDetector (correlation), TimelineManager (snapshots)
4. **Captain-first API** - Tools designed for LLM reasoning, not human dashboards

---

Links:
- Tavily design: `/Users/samuelclark/.claude/plans/elegant-splashing-feather-agent-a8b600d.md`
- News Intelligence design: `/Users/samuelclark/.claude/plans/delegated-strolling-sundae-agent-a5b416c.md`
- Tavily docs: https://docs.tavily.com/
- Tavily SDK: https://github.com/assafelovic/tavily-python
