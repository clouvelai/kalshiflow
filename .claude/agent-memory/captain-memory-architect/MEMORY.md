# Captain Memory Architect - Key Learnings

## V2 Memory System (reviewed 2026-02-09)

V1 fully removed. No feature flags, no conditional paths.

### Architecture
- Two-tier: SessionMemoryStore (FAISS session + pgvector persistent)
- Files: session_store.py, vector_store.py, models.py, context_builder.py
- Module exports: `__init__.py` exports only VectorMemoryService + SessionMemoryStore
- Journal: bounded deque(maxlen=500), injected into cycle prompt via journal_summary()
- Tools: store_insight, recall_memory, search_news (auto-stores)
- 10 tools total, single ToolContext dataclass replaces 14 module-level globals
- Captain is single-agent (no subagents), uses create_deep_agent with StateBackend only

### Key Patterns
- pgvector writes are fire-and-forget via asyncio.create_task() -- errors logged, never raised
- FAISS uses asyncio.to_thread() to avoid blocking event loop
- VectorMemoryService._get_embedding() is async (wraps sync OpenAI SDK in to_thread)
- Dedup: FAISS+pgvector results merged by first-100-chars key, sorted by similarity
- pgvector dedup on store: find_similar_memories RPC with 0.88 threshold
- Access-based ranking: touch_memories RPC bumps access_count after each recall
- Session journal injected into cycle prompt (last 5 entries)

### DB Schema
- Table: agent_memories (migration 20260202100000, extended 20260207100000)
- Columns: memory_type, content, content_hash, embedding(1536), market_tickers[], event_tickers[], confidence, news_url/title/published_at/source, price_snapshot(jsonb)
- Indexes: HNSW on embedding, GIN on market_tickers/event_tickers, btree on type/active/created/hash/access
- RPCs: search_agent_memories (temporal decay scoring), find_similar_memories, touch_memories
- search_agent_memories params order: embedding, types[], market_ticker, event_ticker, min_recency_hours, limit, threshold

### V1 Cleanup (completed 2026-02-09)
- Deleted code: auto_curator.py, dual_store.py, file_store.py, DualMemoryStore, FileMemoryStore, MemoryCurator subagent
- Deleted data: AGENTS.md, SIGNALS.md, PLAYBOOK.md, THESES.md, journal.jsonl, distill_state.json, mentions_state.json, context_*.json, skills/, simulations/
- Kept: issues.jsonl (active), understanding/ (active cache)
- No code references remain to V1 modules

### Test Coverage: 23 tests in test_session_store.py, 18 in test_tools.py
- Journal: CRUD, summary, max_entries, size bound, deque-to-list
- pgvector: fire-and-forget called, error non-fatal
- Recall: empty, pgvector-only, dedup, error non-fatal, limit, sort-by-sim
- Hybrid: pgvector-only, pgvector-down, no-vector-store
- Concurrency: parallel stores, store+recall parallel
- Metadata: journal passthrough, pgvector passthrough
- Tools: all 10 tools tested with mocked gateway/memory

### Gotchas
- deque does NOT support slicing ([-N:]). Must convert to list first.
- VectorMemoryService._get_embedding_sync() is the actual sync method; _get_embedding() is async wrapper
- Captain accesses tool_ctx via `from .tools import _ctx as tool_ctx` inside _run_cycle()
- FAISS lazy-init: first store creates FAISS.from_texts(), subsequent use add_texts()

### Key Files
- `/backend/src/kalshiflow_rl/traderv3/single_arb/captain.py` - Single agent + prompt
- `/backend/src/kalshiflow_rl/traderv3/single_arb/tools.py` - 10 tools, ToolContext dataclass
- `/backend/src/kalshiflow_rl/traderv3/single_arb/models.py` - Pydantic models for all tool I/O
- `/backend/src/kalshiflow_rl/traderv3/single_arb/context_builder.py` - MarketState/PortfolioState/SniperStatus builders
- `/backend/src/kalshiflow_rl/traderv3/single_arb/coordinator.py` - Startup wiring, _setup_tools()
- `/backend/src/kalshiflow_rl/traderv3/single_arb/memory/session_store.py` - FAISS + pgvector
- `/backend/src/kalshiflow_rl/traderv3/single_arb/memory/vector_store.py` - pgvector backend
- `/backend/src/kalshiflow_rl/traderv3/single_arb/memory/issues.py` - Self-improvement issues (still active)

## Sniper Execution Layer
- See CLAUDE.md MEMORY.md for full sniper details
- Captain tools: configure_sniper, get_sniper_status (integrated into ToolContext)
- SniperConfig.update() returns (changed, unknown) tuple
- Capital lifecycle: capital_in_flight + capital_in_positions, NOT cumulative lifetime
