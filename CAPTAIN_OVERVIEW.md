# Captain: End-to-End Trading Lifecycle

A complete walkthrough of what happens when you run `./scripts/run-captain.sh` — from cold boot to live trades to cross-session learning.

---

## Phase 1: Startup & Market Discovery (~30-60 seconds)

### Boot Sequence

`coordinator.py` → `single_arb/coordinator.py` runs a 13-step startup:

1. **Infrastructure** — EventBus (pub/sub), WebSocket manager (frontend), state machine all start
2. **Connections** — Orderbook WebSocket connects to `demo-api.kalshi.co`, trading client authenticates with RSA
3. **Model tiers configured** — Claude Sonnet for Captain, Haiku for subagents, Flash for utility, OpenAI embeddings for memory
4. **Lifecycle discovery** — The lifecycle service categorizes active Kalshi markets by domain (sports, politics, crypto, elections, corporate, entertainment). Markets within 90 days of settlement are eligible.
5. **Event loading** — For each discovered event, `GET /trade-api/v2/events/{event_ticker}` fetches the full event structure. Each event has 2-50 markets. Events with >50 markets are skipped.
6. **Orderbook prefetch** — Every market gets its orderbook via `GET /trade-api/v2/markets/{ticker}/orderbook?depth=5`. The index is **fully populated before Captain starts** — no cold-start delay.
7. **WebSocket subscriptions** — All markets subscribe to real-time orderbook deltas, ticker updates, and trades
8. **Subsystems wire up** — Sniper, AttentionRouter, AutoActionManager, SessionMemoryStore, TaskLedger, ContextBuilder all initialize with shared `ToolContext`

### What Markets Does It Find?

Every active Kalshi event with 2+ open markets across all categories. Typical inventory: **50-200 events, 200-1000+ markets**:

- **Fed rate decisions** — Mutually exclusive outcomes, classic arb territory
- **Election markets** — Multi-candidate, probabilities should sum to 100%
- **Sports outcomes** — Game results, player props
- **Crypto price brackets** — BTC above/below X by date
- **Corporate events** — Earnings beats, CEO departures

The system doesn't pick favorites — it loads everything and lets the attention system filter what matters.

---

## Phase 2: Real-Time Data Flow (Continuous)

```
Kalshi WebSocket
    → EventBus broadcasts (ORDERBOOK_SNAPSHOT, TICKER_UPDATE, MARKET_TRADE)
        → EventArbMonitor updates MarketMeta in the index
            → Computes: BBO, spread, edge, VPIN, OFI, microstructure signals
                → If edge >= 0.5c: AttentionRouter scores the signal
                    → If score >= 40: Captain gets notified
```

### What Gets Computed Per Market

| Signal | What It Means |
|--------|--------------|
| **long_edge** | How much cheaper it is to buy all YES outcomes than 100c (arb profit on ME events) |
| **short_edge** | Same but buying all NO outcomes |
| **VPIN** | Volume-synchronized probability of informed trading (toxic flow detector, 20-contract buckets) |
| **OFI** | Order Flow Imbalance — EMA-smoothed, detects directional pressure |
| **book_imbalance** | Bid vs ask depth ratio |
| **whale_trades** | Trades >= 100 contracts |
| **rapid_sequences** | Trades within 100ms of each other (algorithmic activity) |

### Fallback: REST Poller

Markets where WebSocket data is stale (>30 seconds) get polled via REST every 10 seconds. Catches thin markets where WS snapshots are infrequent.

---

## Phase 3: Captain Invocation (Three Modes)

The Captain is a single Claude Sonnet agent with 13 tools. It doesn't run continuously — it wakes up based on signals or timers.

### Mode 1: Reactive (Attention-Driven, ~6-8s typical)

**Trigger:** AttentionRouter detects an urgent signal (score >= 40, urgency "high" or "immediate")

**What triggers it:**
- Edge jumps +2c suddenly (news broke, someone moved the book)
- Position P&L crosses -12c/contract (losing money)
- VPIN spikes to 0.98 (informed traders flooding the book)
- Early bird opportunity appears (newly activated market with complement strategy)

**What Captain sees:** Compact ~300 token briefing — just the attention items, current positions, and sniper status.

**What Captain does:** 1-3 tool calls. Respond to the signal — trade, exit, or note why it's passing. No exploration, no unrelated research.

### Mode 2: Strategic (Every 5 Minutes, ~15-30s)

**Trigger:** Timer (`V3_STRATEGIC_INTERVAL=300s`)

**What Captain sees:** ~500 token briefing — portfolio (all positions + P&L), account health, early bird opportunities, pending attention items, sniper performance, decision accuracy stats from last 24h.

**What Captain does:** 3-8 tool calls. Search news on active events, evaluate early bird opportunities, tune sniper settings, manage positions, update task list.

### Mode 3: Deep Scan (Every 30 Minutes, ~30-60s)

**Trigger:** Timer (`V3_DEEP_SCAN_INTERVAL=1800s`)

**What Captain sees:** ~1000 token briefing — ALL events (compact JSON), ALL positions, sniper lifetime stats, health metrics, trade outcome memories, news impact memories (last 48h of price-moving articles), known news-price patterns.

**What Captain does:** 5-15 tool calls. Full portfolio review, recall cross-session memories, search news on every active event, evaluate all positions against news, store learnings, rebalance.

---

## Phase 4: What Trades Does It Make?

### Trade Type 1: Early Bird Complement (Highest Priority)

When a new market activates, the system computes a deterministic `fair_value` from the complement structure.

**Example:** "Fed holds rates" event has 3 outcomes. New market activates with fair_value=45c.

1. Captain places YES limit at 43c (fair_value - 2c) — maker order, 0% fees
2. Captain places NO limit at 53c (100 - 45 - 2c) — maker order, 0% fees
3. If both fill: 4c spread captured = profit
4. If one fills: cancel the other, hold directional position

**Size:** 100-250 contracts. **Speed:** Acts in seconds, no news research needed.

### Trade Type 2: News-Driven Directional

Captain searches news, finds a probability shift vs current price, and places a resting limit order.

**Example:** News breaks "Fed signals surprise rate cut." Captain searches, estimates this shifts KXFED from 45c to 55c fair value. Current ask is 47c.

1. `search_news("Federal Reserve rate cut signal March 2026")`
2. Assess: 10c shift, high confidence
3. `place_order(KXFED-YES, buy, 75 contracts, 47c)` — edge is ~8c
4. `store_insight("Fed rate cut signal → KXFED fair 55c, bought 75ct @ 47c")`

**Sizing by edge:**

| Edge | Contracts |
|------|-----------|
| >= 10c | 50-100 |
| 5-10c | 25-50 |
| 2-5c | 10-25 |

### Trade Type 3: Sniper Auto-Arb (Autonomous, No LLM)

The Sniper runs on the hot path with zero latency. When EventArbIndex detects `long_edge >= 3.0c` on a mutually exclusive event:

1. **Risk gates check** (~5ms): cycle limit, edge threshold, event cooldown, capital limit, VPIN toxicity
2. **Execute** (<2s): Place orders on all legs simultaneously
3. **Track**: Capital in flight updated, Captain informed next cycle

**Example:** 3-outcome event where all YES asks sum to 96c. Edge = 4c. Sniper buys 30 contracts on each leg at market. Total cost ~2880c, guaranteed profit ~120c minus fees.

Captain doesn't call `execute_arb` for routine arbs — Sniper handles them autonomously.

### Trade Type 4: Position Management (Auto-Actions)

Three deterministic rules fire **before Captain even sees the signal:**

| Rule | Trigger | Action |
|------|---------|--------|
| **stop_loss** | Position P&L <= -12c/contract | Sell immediately at market |
| **time_exit** | < 30 minutes to settlement | Exit position at current bid |
| **regime_gate** | VPIN > 0.98 (toxic flow) | Pause Sniper for 5 minutes |

Captain sees these as `auto_handled` items and can override if it has better information.

---

## Phase 5: What Does It Remember?

### Within a Session (FAISS, <1ms)

- Every `store_insight()` call goes to FAISS immediately
- Recent trade outcomes, news analyses, market observations
- LRU eviction at 2,000 entries (keeps newest 1,500)
- Lost on restart

### Across Sessions (pgvector, persistent)

Every `store_insight()` also writes to Supabase pgvector (fire-and-forget). Survives restarts.

**What gets stored:**
- `"learning"`: "IF Fed hawkish tone THEN KXFED drops 3c in 1h"
- `"trade_outcome"`: "SETTLEMENT: KXFED settled YES, P&L +$2.15"
- `"news"`: Full articles chunked for semantic search
- `"pattern"`: "VPIN > 0.9 correlates with 5min price reversions on crypto markets"
- `"observation"`: "KXFED spread widens to 8c in last 30min before settlement"
- `"mistake"`: "Bought at ask with no edge — edge gate should have blocked"

**What gets blocked (toxic patterns):**
- "never trade", "pause trading", "freeze", "wait for accuracy"
- These create self-reinforcing behavioral loops. Only factual observations are stored.

### Working Memory (Task Ledger)

Persisted to Supabase, tracks priorities across cycles:
```
[>] [HIGH] Execute complement on KXFED-NEW (in_progress)
[ ] [MED]  Monitor KXBTC spread narrowing (pending)
[STALE]    Search news on earnings (unchanged 10+ cycles → pruned)
```

Stale detection triggers replanning when >50% of active tasks are stale.

### Decision Accuracy (24h rolling)

Captain sees its own track record each strategic cycle:
- Direction accuracy: "5/5 filled orders were correct direction"
- Fill rate: "75% of limit orders would have filled at entry price"
- Average P&L: "+3.2c/trade"

This feedback loop calibrates sizing and confidence.

---

## Phase 6: Safety Nets

| Mechanism | What It Does |
|-----------|-------------|
| **Drawdown circuit breaker** | Pauses Captain at -25% drawdown from peak. Resumes at -20%. Auto-resets after 24h. |
| **Edge gate on orders** | Blocks "buy YES at ask" trades (zero edge). Prevents accidental market orders. |
| **Position conflict guard** | Can't buy opposite side of existing position |
| **VPIN regime gate** | Sniper pauses when VPIN > 0.98 (toxic flow) |
| **Capital scaling** | Sniper sizes by available headroom, not total balance |
| **Order TTL** | Resting orders auto-cancel after 90s. Stale orders cleaned every 5 min. |
| **Memory retention** | pgvector cleaned every 30 min via RPC |
| **Settled event cleanup** | Dead markets pruned every 30 min |
| **Atomic file writes** | Issues.jsonl uses tempfile + os.rename() |
| **Graceful shutdown** | `memory.flush(timeout=5s)` awaits pending pgvector writes |
| **Bull/bear discipline** | Before any trade: state bull case, bear case, why bull wins |

---

## Why It Works (Design Thesis)

1. **Attention-driven, not poll-driven.** Captain sleeps until something matters. Reactive mode responds in ~6-8 seconds. No wasted LLM calls on quiet markets.

2. **Separation of fast and slow paths.** Sniper handles mechanical arbs in <2 seconds (no LLM). Captain handles judgment calls (news interpretation, position management). Auto-actions handle risk management deterministically (no LLM latency on stop-losses).

3. **Prescriptive, not open-ended.** The system prompt gives Captain specific decision trees with concrete numbers (4c complement spread, 5c news shift threshold, sizing tables). It's not "think about what to trade" — it's "follow this checklist."

4. **Learning without toxicity.** Cross-session memory stores factual observations but blocks self-imposed behavioral restrictions that create toxic feedback loops.

5. **Progressive context.** Reactive gets 300 tokens (just the signal). Strategic gets 500 tokens (portfolio + opportunities). Deep scan gets 1000 tokens (everything). Captain never wastes reasoning on irrelevant context.

6. **Mandatory bull/bear discipline.** Before any trade, Captain must state the bull case, the bear case, and why bull wins. If it can't rebut the bear case, it doesn't trade.

7. **Feedback calibration.** Captain sees its own 24h decision accuracy every strategic cycle. Good accuracy → scale up. Bad accuracy → the prompt naturally causes more cautious sizing.

8. **Paper trading isolation.** All trading happens on `demo-api.kalshi.co` via a dedicated subaccount. No risk of accidentally touching real money.

---

## Architecture Reference

```
┌─────────────────────────────────────────────────────────────────┐
│                         V3 Trader                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│
│  │ EventBus │──│ Orderbook│──│ Trading  │──│ WebSocket Manager││
│  │ (pub/sub)│  │ Client   │  │ Client   │  │ (frontend)       ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Single-Arb System                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────────┐  │
│  │EventArb  │──│ Monitor  │──│    Captain (single agent)    │  │
│  │Index     │  │          │  │  ┌────────────────────────┐  │  │
│  │(markets) │  │(data in) │  │  │ AttentionRouter        │  │  │
│  └──────────┘  └──────────┘  │  │ AutoActionManager      │  │  │
│                              │  │ Sniper (auto-exec)     │  │  │
│                              │  │ SessionMemoryStore     │  │  │
│                              │  │ TaskLedger             │  │  │
│                              │  └────────────────────────┘  │  │
│                              └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Captain's 13 Tools

| Category | Tool | Purpose |
|----------|------|---------|
| **Observation** | `get_market_state` | Market data, orderbook, microstructure, edge |
| | `get_portfolio` | Positions, P&L, balance |
| | `get_account_health` | Drawdown, risk metrics |
| | `get_resting_orders` | Open orders with fill status |
| | `get_market_movers` | News articles that moved prices |
| **Execution** | `place_order` | Single-leg order with reasoning |
| | `execute_arb` | Multi-leg arb across event |
| | `cancel_order` | Cancel resting order |
| **Configuration** | `configure_sniper` | Tune sniper thresholds |
| | `configure_automation` | Tune auto-action rules |
| **Intelligence** | `search_news` | News search (3 tiers: ultra_fast/fast/advanced) |
| | `recall_memory` | Semantic search across FAISS + pgvector |
| | `store_insight` | Persist learnings to memory |

### Attention Scoring

| Signal | Weight | Captain Triggers At |
|--------|--------|-------------------|
| Edge magnitude > 3.0c | 30 | Immediate (score 70+) |
| Edge delta since last cycle | 15 | If delta > 1.0c |
| Position P&L threshold | 20 | pnl/ct >= +10c OR <= -12c |
| VPIN spike (> 0.92) | 10 | Early alert |
| VPIN critical (> 0.98) | 10 | With position (score 55+) |
| Sweep detected | 10 | Urgent (edge-dependent) |
| Whale trade activity | 5 | Supplementary |
| Time to close < 1h with position | 10 | Settlement pressure |
| Early bird opportunity | 35 | Immediate (score 70+) |

**Threshold to emit:** score >= 40. **Urgency:** immediate (70+) > high (55+) > normal (35+).
