# Captain Memory Architect - Key Learnings

## Architecture Understanding (verified 2026-02-07)

### Memory Stack
- **Static files**: AGENTS.md (60L), SIGNALS.md (30L), PLAYBOOK.md (40L), THESES.md (30L) -- loaded via MemoryMiddleware each cycle
- **journal.jsonl**: append-only, FileMemoryStore writes, never curated
- **VectorMemoryService**: OpenAI text-embedding-3-small (1536d) + Supabase pgvector + HNSW index
- **DualMemoryStore**: file (sync) + vector (fire-and-forget async)
- **recall_context tool**: NOW EXISTS -- calls DualMemoryStore.search() which merges keyword + semantic
- See [vector-store-schema.md](vector-store-schema.md) for DB details

### Captain Cycle Structure
- 60s interval, new thread_id each cycle (prevents history accumulation)
- CompositeBackend: /memories/ -> FilesystemBackend (persistent), rest -> StateBackend (ephemeral)
- auto_curator runs at cycle start (pure Python, no LLM)
- Prompt includes strategy playbook (S1-S6), subagent descriptions, memory protocol
- Tools: 22+ total (observation + execution + memory + self-improvement + news intelligence)

### News Intelligence System (added 2026-02-07)
- `recall_context()`: semantic search over DualMemoryStore.search() -- WORKS
- `NewsStore`: persists articles as embeddings via DualMemoryStore.append() with memory_type='news'
- `PriceImpactTracker`: schedule_snapshots() IS called via NewsStore.persist_articles() -> search_event_news auto-persist
- `manage_thesis()`: thesis lifecycle with theses.json + THESES.md memory file
- `LifecycleClassifier`: Haiku-powered event stage classification cached on EventMeta.lifecycle, 30min cache TTL
- Background loops: news refresh (30min), lifecycle reclassify (30min), causal model rebuild (30min) -- all in _news_refresh_loop

### Causal Model System (reviewed 2026-02-07)
- `CausalModel`: per-event, drivers + catalysts + entity_links + consensus + narrative
- `CausalModelBuilder`: Haiku LLM, builds from EventMeta + understanding + lifecycle + candlesticks
- Driver lifecycle: active -> stale (2h no validation) -> invalidated (contradictions > validations)
- Built at startup, rebuilt every 30min by _news_refresh_loop
- Captain tool: update_causal_model() with 8 actions (refresh/add_driver/validate/invalidate/add_catalyst/mark_catalyst/prune/view)
- get_events_summary() includes compact causal model (top 3 drivers, next catalyst, consensus)
- get_event_snapshot() includes full causal model
- Cycle briefing tracks imminent catalysts but NOT driver staleness or consensus changes

### Critical Bugs Found (2026-02-07 review, updated)
1. VectorMemoryService.store() does NOT write news-specific columns (news_url, news_title, etc.)
2. **FIXED**: PriceImpactTracker.schedule_snapshots() IS called via NewsStore.persist_articles()
3. NewsStore.search_news() expects news fields in search results but vector store doesn't return them through metadata
4. Thesis "first 10 cycles" guard mentioned in prompt but NOT implemented in code
5. **NEW**: tools.py:546 add_driver passes Driver object to CausalModel.add_driver() which expects keyword args -- SILENT CORRUPTION
6. **NEW**: coordinator.py _build_gateway_tool_overrides() missing ALL intelligence tools (update_causal_model, recall_context, search_news_history, get_price_movers, manage_thesis, get_event_candlesticks)

### Captain Prompt Gaps (2026-02-07 causal review)
- No WHEN guidance for causal model maintenance (prune/refresh cadence)
- No decision-making integration (how drivers/lifecycle influence trade selection)
- No news-to-causal-model workflow (search news -> validate/add/invalidate drivers)
- Cycle briefing missing: driver staleness changes, consensus shifts, catalyst occurrences
- ChevalDeTroie has no lifecycle/causal context for pattern interpretation
- No tool to trigger on-demand lifecycle reclassification

### Auto-Curator Critical Bug FIXED (2026-02-07)
- LESSONS section in AGENTS.md was NEVER truncated: `keep_limits` dict only had "Current Cycle Notes" and "Hypotheses Under Test"
- Section matching used exact string match on full header text -- fragile, broke when header text varied
- Fixed: switched to substring/keyword matching via `_is_protected()` and `_get_keep_limit()`
- Added `"LESSONS": 10` to TRUNCATABLE_KEEP so lessons get trimmed to 10 most recent
- Added deduplication via `difflib.SequenceMatcher` (threshold 0.82) -- collapsed 50+ identical "Cycle #N: PASS" entries to 1
- Added section-aware curation for PLAYBOOK.md (preserves EXIT_RULES/BANKROLL, truncates narrative)

### Captain Prompt Reduced (2026-02-07)
- 227 lines / ~8600 chars -> 84 lines / ~3864 chars (55% reduction)
- Removed verbose RECALL PROTOCOL, NEWS-PRICE INTELLIGENCE sections (covered by tool descriptions)
- Added anti-narration guardrail: "NEVER write 'Cycle #N: PASS - same issues' -- if nothing changed, write nothing"
- Memory state cleaned: AGENTS.md distilled from 72 lines of noise to 25 lines of high-value rules+lessons

### AGENTS.md Degradation Pattern
- Captain writes cycle-by-cycle narration instead of IF-THEN rules despite prompt saying "write rules not narratives"
- 56 of 72 lines were variations of "Cycle #N: PASS - same issues"
- Deduplication + truncation now catches this pattern automatically
- Root cause persists: LLM ignores soft prompt instructions. Anti-narration guardrail + dedup is the defense.

### Vector Store Schema (Supabase)
- Table: `agent_memories` -- base + news columns (news_url, news_title, news_published_at, news_source, price_snapshot)
- Types: learning, mistake, pattern, journal, market_knowledge, consolidation, signal, research, thesis, trade, strategy, observation, trade_result, news, news_digest, thesis_archived
- RPCs: search_agent_memories (similarity + temporal-decay + access-boost), find_similar_memories (dedup at 0.88), touch_memories, find_market_movers
- Table: `news_price_impacts` -- market_ticker, event_ticker, price changes at 1h/4h/24h
- Retention: auto-expire low/medium confidence unaccessed memories after 14/60/120 days

### Key Files
- `/backend/src/kalshiflow_rl/traderv3/single_arb/captain.py` - Agent + prompts
- `/backend/src/kalshiflow_rl/traderv3/single_arb/tools.py` - Tools, module-level globals
- `/backend/src/kalshiflow_rl/traderv3/single_arb/news_store.py` - NewsStore
- `/backend/src/kalshiflow_rl/traderv3/single_arb/impact_tracker.py` - PriceImpactTracker
- `/backend/src/kalshiflow_rl/traderv3/single_arb/coordinator.py` - Startup wiring
- `/backend/src/kalshiflow_rl/traderv3/single_arb/memory/` - file_store, vector_store, dual_store, auto_curator, issues
- `/backend/supabase/migrations/20260207100000_news_intelligence.sql` - Schema changes

### Design Patterns
- Tools use `set_dependencies()` for DI via module-level globals
- Coordinator now passes DualMemoryStore to tools (not just FileMemoryStore)
- Vector writes happen via DualMemoryStore.append() which fire-and-forgets to vector store
- DualMemoryStore.search() merges semantic (vector) + keyword (file) results
