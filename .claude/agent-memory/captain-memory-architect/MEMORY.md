# Captain Memory Architect - Key Learnings

## V2 Memory System (reviewed 2026-02-09, audited 2026-02-11)

V1 fully removed. No feature flags, no conditional paths.

### Architecture
- Two-tier: SessionMemoryStore (FAISS session + pgvector persistent)
- Files: session_store.py, vector_store.py, models.py, context_builder.py
- Module exports: `__init__.py` exports only VectorMemoryService + SessionMemoryStore
- Journal: bounded deque(maxlen=500), injected into cycle prompt via journal_summary()
- Tools: store_insight, recall_memory, search_news (auto-stores)
- 13 tools total, single ToolContext dataclass
- Captain is single-agent (no subagents), uses create_deep_agent with StateBackend only

### Critical Bug Found & Fixed (2026-02-11)
- `rl_db.get_pool()` did NOT exist on RLDatabase class -- 5 callers silently failed
- Impact: news_price_impacts never backfilled, retention policy never ran, task persistence dead
- Fix: Added `get_pool()` method to RLDatabase in data/database.py
- Also fixed: store_insight docstring advertised "rule" memory_type not in DB CHECK constraint

### Key Patterns
- pgvector writes are fire-and-forget via asyncio.create_task()
- FAISS uses asyncio.to_thread() to avoid blocking event loop
- Dedup: FAISS+pgvector results merged by first-100-chars key, sorted by similarity
- pgvector dedup on store: find_similar_memories RPC with 0.88 threshold
- Access-based ranking: touch_memories RPC bumps access_count after each recall

### DB Schema
- Table: agent_memories (migration 20260202100000+)
- Columns: memory_type, content, content_hash, embedding(1536), market_tickers[], event_tickers[], confidence, news_url/title/published_at/source, price_snapshot(jsonb), signal_quality(float)
- Scoring: cosine_sim * exp(-0.01*hours) + ln(access+1)/ln(2)*0.02 + (signal_quality-0.5)*0.15
- memory_type allowed: learning, mistake, pattern, journal, market_knowledge, consolidation, signal, research, thesis, trade, strategy, observation, trade_result, news, news_digest, thesis_archived, settlement_outcome, trade_outcome

### News-Price Impact Pipeline
- search_news stores articles with price_snapshot metadata
- coordinator._backfill_news_impacts() runs every 30min
- find_market_movers RPC returns market-moving articles for deep_scan

### Event Understanding System (planned refactor 2026-02-11)
- Current: disk cache (JSON files, 4h TTL), rebuilt every startup, no DB persistence
- Plan: Supabase table `event_understandings`, one row per event_ticker (UNIQUE)
- Startup: batch-load from DB -> build missing -> gate Captain on completion
- New tool: `update_event_understanding` for incremental updates
- Disk cache to be removed once DB persistence is live
- Key file: `event_understanding.py` (UnderstandingBuilder + EventUnderstanding dataclass)
- News fields being removed from EventUnderstanding (decoupled from search_news)
- Plan doc: `/PLAN-MEMORY.md`

### Fresh Start Requirements (for live switch)
1. Truncate agent_memories, news_price_impacts, captain_task_ledger, event_understandings
2. Delete memory/data/understanding/*.json cache files (will be removed entirely after refactor)
3. Verify all migrations applied (latest: 20260213100000 after understanding migration)

### Gotchas
- deque does NOT support slicing. Must convert to list first.
- RLDatabase has BOTH get_connection() (async ctx mgr) AND get_pool() (returns raw pool)
- FAISS eviction drops extra metadata (only keeps memory_type+timestamp) -- harmless
- FAISS lazy-init: first store creates FAISS.from_texts(), subsequent use add_texts()

### Key Files
- `/backend/src/kalshiflow_rl/data/database.py` - RLDatabase with get_pool() + get_connection()
- `/backend/src/kalshiflow_rl/traderv3/single_arb/memory/` - session_store, vector_store, issues
- `/backend/src/kalshiflow_rl/traderv3/single_arb/tools.py` - 13 tools, ToolContext
- `/backend/src/kalshiflow_rl/traderv3/single_arb/captain.py` - Single agent + 3 modes
- `/backend/src/kalshiflow_rl/traderv3/single_arb/coordinator.py` - Startup wiring

### Test Coverage: 491 tests total in tests/traderv3/, all passing (2026-02-11)

## Sniper Execution Layer
- Captain tools: configure_sniper, get_sniper_status (integrated into ToolContext)
- SniperConfig.update() returns (changed, unknown) tuple
- Capital lifecycle: capital_in_flight + capital_in_positions, NOT cumulative lifetime
