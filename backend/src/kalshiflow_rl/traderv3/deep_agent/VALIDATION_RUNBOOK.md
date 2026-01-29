# Deep Agent Validation Runbook

Reusable validation process for the V3 Trader deep_agent strategy.
Last validated: 2026-01-28 (sam/agentv2 branch, commit 3d05e45)

---

## Quick Start Validation

```bash
# 1. Ensure clean working state
git status  # should be clean

# 2. Start V3 trader (paper mode)
./scripts/run-v3.sh paper

# 3. Open frontend
open http://localhost:5173/v3-trader/agent

# 4. Monitor logs (in separate terminal)
tail -f backend/*.log  # or check terminal running run-v3.sh
```

---

## Validation Checklist

### Phase 1: Startup (0-30 seconds)

| Check | Expected | How to Verify |
|-------|----------|---------------|
| Environment loads | `Using demo API (safe for paper trading)` | Console output |
| Credentials validated | ANTHROPIC_API_KEY, SUPABASE_URL, SUPABASE_KEY all present | No `Error: Missing` messages |
| Kalshi WebSocket connects | `WebSocket connected` | Log: `kalshi_ws_client` |
| Lifecycle discovery | `Discovered N events with M markets` | Log: `event_lifecycle_service` |
| Orderbook subscriptions | `Subscribing to ticker channel for N tickers` | Log: `market_ticker_listener` |
| Market price sync | `Market price sync complete: N/N tickers` | Log: `market_price_syncer` |
| Trading state sync | Balance, positions, orders fetched | Log: `kalshi_data_sync` |
| Entity Market Index built | `EntityMarketIndex built: N entities` | Log: `coordinator` |
| TRADER V3 STARTED SUCCESSFULLY | Status: `ready` | Log: `coordinator` |

### Phase 2: Deep Agent Initialization (30-60 seconds)

| Check | Expected | How to Verify |
|-------|----------|---------------|
| Agent start | `Agent started successfully` | Log: `deep_agent.agent` |
| Memory files loaded | `mistakes.md`, `patterns.md`, `strategy.md` loaded | Log: `deep_agent.tools` |
| Reflection engine started | `Started reflection engine` | Log: `deep_agent.reflection` |
| Main loop started | `_main_loop() STARTED` | Log: `deep_agent.agent` |
| Cycle 1 begins | `=== Starting cycle 1 ===` | Log: `deep_agent.agent` |
| Claude API working | `Cache CREATED: N tokens cached` | Log: `deep_agent.agent` |

### Phase 3: Entity Pipeline (0-2 minutes)

| Check | Expected | How to Verify |
|-------|----------|---------------|
| Reddit agent healthy | `healthy (praw=True, nlp=True, supabase=True)` | Log: `reddit_entity_agent` |
| Posts fetched | `Fetched N posts from r/politics, r/news, ...` | Log: `reddit_entity_agent` |
| Entities extracted | Entity names matched to KB | Log: `price_impact_agent` |
| Price impacts created | `Created N price impacts` | Log: `price_impact_agent` |
| Supabase insert | `Inserted to market_price_impacts` | Log: `price_impact_agent` |

### Phase 4: Trading Behavior (2-5 minutes)

| Check | Expected | How to Verify |
|-------|----------|---------------|
| Signals detected | `Found N tradeable signals` | Log: `deep_agent.tools` |
| Trade decision made | Agent thinks through signal analysis, risk management | Frontend: Thinking panel |
| Orders placed | `place_order` tool calls with ticker, side, contracts | Log: `deep_agent.agent` |
| Cycle PASS when appropriate | `PASS - Signals persist but already have exposure` | Frontend: Thinking panel |
| Position awareness | Agent notes existing positions before trading | Frontend: Position Status section |

### Phase 5: Frontend UX

| Check | Expected | How to Verify |
|-------|----------|---------------|
| Connection status | `Connected` + `READY` badges | Top right of header |
| Data Pipeline counters | Reddit > 0, Entities > 0, Signals > 0, Index = 42 | Pipeline bar at top |
| Reddit Stream | Live posts with subreddit, score, age | Left panel |
| Entity Extractions | Market-linked entities with sentiment scores | Left panel below Reddit |
| Agent status | `RUNNING` badge, Cycle N, Trades N, P&L | Right panel header |
| Thinking panel | Markdown-formatted reasoning with sections | Right panel |
| Tool Calls | Count badge, expandable list | Below thinking |
| Price Impact Signals | Cards with entity name, sentiment, impact, confidence | Below tool calls |
| Trades section | Trade entries with ticker, side, contracts | Below signals |
| Entity Index | LIVE badge, entity count, search, aliases | Bottom section |

---

## Known Issues (2026-01-28)

### Bugs

| # | Severity | Component | Description |
|---|----------|-----------|-------------|
| B1 | Medium | Entity Pipeline | **Content Extraction 0% success**: Links are discovered (21+) but text extraction returns 0. The OpenAI-powered content extraction may be failing silently for most URLs. |
| B2 | Medium | Entity Extraction | **Low-quality entity names**: Market titles/subtitles extracted as entities (e.g., "Above 2.5%", "Before Jan 31, 2026", "Yes", "I Like It"). Need entity name filtering/validation. |
| B3 | Low | Supabase Schema | **entity_aliases table missing**: 10 failed POST requests per startup to `public.entity_aliases`. Aliases work in-memory but aren't persisted. Create the table to fix. |
| B4 | Low | Truth Social | **403 Authentication failure**: `TruthSocialAuthError` at startup. Auto-retries every 5 min but persists. Non-critical data source. |
| B5 | Low | TMO Fetcher | **36 tickers fail TMO fetch**: Candlestick API returns no valid data for many tickers. Retries exhaust (3 attempts). May be API data availability issue. |
| B6 | Low | Price Impact Schema | **Schema mismatch on insert**: New columns not in Supabase schema. Code retries without optional columns (graceful degradation) but some data is lost. |

### Improvements

| # | Priority | Component | Description |
|---|----------|-----------|-------------|
| I1 | High | Entity Extraction | Filter out market title fragments from entity names. Validate entity names against a minimum quality threshold (length, capitalization, not a number/date). |
| I2 | High | Content Extraction | Investigate 0% success rate. Check if OpenAI API calls are failing, or if URL fetching is blocked. Consider fallback extraction methods. |
| I3 | Medium | Logging | Reduce orderbook_delta DEBUG noise. These messages dominate logs (thousands per minute) making it hard to find meaningful events. |
| I4 | Medium | Supabase | Create `entity_aliases` table and `market_price_impacts` schema migration to include new columns. |
| I5 | Low | TMO Fetcher | Use exponential backoff instead of fixed retries. Consider marking tickers as "no TMO data" to avoid repeated failures. |
| I6 | Low | Deep Agent | Add cycle completion summary to logs (not just start). Log total tokens used, cache hit rate, and trade count per cycle. |
| I7 | Low | Frontend | Trades section should auto-expand when trades exist (currently collapsed by default). |

---

## Validation Results (2026-01-28 22:47-22:53 EST)

### System Health: HEALTHY

- **Environment**: Paper (demo-api.kalshi.co)
- **Session**: 462 (active)
- **Markets**: 42 tracked, 170 tickers subscribed
- **Balance**: $24,520.42 cash, $6,474.54 in positions, $30,994.96 total

### Deep Agent Performance

- **Cycles completed**: 3
- **Trades placed**: 4 (in cycle 2)
  - KXTRUMPSAY-26FEB02-TDS: 15 NO contracts (signal: -40 impact, 100% confidence)
  - KXGOVSHUT-26JAN31: 25 YES contracts (signal: +75 impact, 100% confidence, inverted for OUT market)
- **Cycle 3 decision**: PASS (already have exposure, signals persist)
- **Claude API**: Cache working (80% hit rate after warmup)

### Entity Pipeline Performance

- **Reddit posts processed**: 53
- **Entities extracted**: 13
- **Price impact signals**: 10
- **Entity Index**: 42 entities, 42 markets, 229 aliases
- **Subreddit coverage**: r/politics, r/news + 11 others configured
- **Entity matching**: Working (e.g., "shutdown" -> "Shut down" -> KXGOVSHUT-26JAN31)

### Frontend UX

- **Connection**: Stable (WebSocket connected, READY)
- **All panels rendering**: Data Pipeline, Reddit Stream, Entity Extractions, Thinking, Tool Calls, Price Impact Signals, Trades, Learnings, Entity Index
- **Real-time updates**: All counters incrementing live
- **Markdown rendering**: Working in Thinking panel (headers, bullets, bold)

---

## Monitoring Commands

```bash
# Check trader health
curl -s http://localhost:8005/v3/health | python3 -m json.tool

# Check trader status
curl -s http://localhost:8005/v3/status | python3 -m json.tool

# Watch for errors only
grep -i "ERROR" <log_output> | grep -v "DEBUG"

# Watch deep agent cycles
grep "Starting cycle\|PASS\|place_order\|Cache" <log_output>

# Watch entity pipeline
grep "price_impact\|entity.*match\|Reddit.*fetch" <log_output> | grep -v DEBUG

# Check status updates
grep "STATUS:" <log_output>
```
