# Deep Agent Data Access Reference

Comprehensive map of what the deep agent can see regarding events, markets, positions, orders, and settlements.

All tools defined in `deep_agent/agent.py:269-900`, implemented in `deep_agent/tools.py`.

---

## 1. COMPLETE TOOL INVENTORY (23 tools)

| # | Tool | Category | Data Source |
|---|------|----------|-------------|
| 1 | `get_extraction_signals` | Signals | Supabase `extractions` table (SQL RPC) |
| 2 | `get_markets` | Markets | `TrackedMarketsState` + Kalshi REST fallback |
| 3 | `get_event_context` | Markets | `TrackedMarketsState` + `StateContainer` + `EventPositionTracker` |
| 4 | `preflight_check` | Safety | Bundles markets + event context + safety checks |
| 5 | `trade` | Execution | `V3TradingClientIntegration` |
| 6 | `get_session_state` | Positions/P&L | `StateContainer` (filtered to deep_agent, includes open_orders) |
| 7 | `get_true_performance` | Positions/P&L | `StateContainer` + settlements (all-time) |
| 8 | `get_microstructure` | Orderbook | `OrderbookIntegration` + `TradeFlowService` + L2 depth |
| 9 | `get_candlesticks` | Price History | `V3TradingClientIntegration` event candlesticks endpoint |
| 10 | `think` | Reasoning | No external data (structured pre-trade analysis) |
| 11 | `reflect` | Learning | No external data (structured post-trade learning) |
| 12 | `write_cycle_summary` | Learning | No external data (end-of-cycle journal) |
| 13 | `understand_event` | Events | Kalshi REST API + LLM → Supabase `event_configs` |
| 14 | `evaluate_extractions` | Learning | Supabase `extractions` table (updates quality scores) |
| 15 | `refine_event` | Learning | Supabase `event_configs` (pushes learnings back) |
| 16 | `get_extraction_quality` | Signals | Supabase `extractions` (quality metrics per event) |
| 17 | `read_memory` / `write_memory` / `append_memory` | Memory | Local filesystem `./memory/` |
| 18 | `read_todos` / `write_todos` | Memory | Local filesystem `./memory/todos.json` |
| 19 | `get_reddit_daily_digest` | Intelligence | Reddit Historic Agent (background) |
| 20 | `query_gdelt_news` | Intelligence | BigQuery GKG (if GDELT_ENABLED) |
| 21 | `query_gdelt_events` | Intelligence | BigQuery GDELT 2.0 Events (if GDELT_ENABLED) |
| 22 | `search_gdelt_articles` | Intelligence | GDELT DOC API (free, no BigQuery needed) |
| 23 | `get_gdelt_volume_timeline` | Intelligence | GDELT DOC API timeline endpoint |

---

## 2. MARKET & EVENT DATA

### `get_markets(event_ticker?, limit=20)` → `tools.py:528`

Returns per-market from `TrackedMarketsState`:

| Field | Type | Source |
|-------|------|--------|
| `ticker` | str | TrackedMarketsState |
| `event_ticker` | str | TrackedMarketsState |
| `title` | str | `yes_sub_title` or market title |
| `yes_bid` | int (cents) | Orderbook integration |
| `yes_ask` | int (cents) | Orderbook integration |
| `spread` | int (cents) | Computed: ask - bid |
| `volume_24h` | int | TrackedMarketsState |
| `status` | str | "active" / "closed" etc. |
| `last_trade_price` | int? (cents) | TradeFlowService |
| `data_quality` | str | "live" / "estimated" / "unknown" |

**Fallback**: If `TrackedMarketsState` unavailable, calls `trading_client.get_markets()` REST API.

### `get_event_context(event_ticker)` → `tools.py:686`

Returns event-level structure with position & risk data:

**Event Summary:**
| Field | Type | Description |
|-------|------|-------------|
| `event_ticker` | str | Event identifier |
| `market_count` | int | Number of markets in event |
| `yes_sum` | int (cents) | Sum of all YES prices - critical for mutual exclusivity |
| `no_sum` | int (cents) | N*100 - yes_sum |
| `risk_level` | str | "ARBITRAGE" / "NORMAL" / "HIGH_RISK" / "GUARANTEED_LOSS" |
| `has_positions` | bool | Do we hold positions in this event |
| `position_count` | int | Markets with positions |
| `total_contracts` | int | Total contracts across positions |
| `mutual_exclusivity_note` | str | Human-readable event structure explanation |

**Per-Market in Event:**
| Field | Type |
|-------|------|
| `ticker` | str |
| `title` | str |
| `yes_price` | int? (cents) |
| `no_price` | int? (cents) |
| `has_position` | bool |
| `position_side` | "yes" / "no" / "none" |
| `position_contracts` | int |

**Source chain**: `TrackedMarketsState.get_markets_by_event()` → prices from orderbook → positions from `StateContainer.get_trading_summary()` → risk from `EventPositionTracker`

### `understand_event(event_ticker, force_refresh?)` → `tools.py:2113`

Builds/refreshes event understanding via LLM. Returns:
- `event_title`, `description`, `primary_entity`
- `key_drivers` - factors affecting outcome
- `markets` - market structure
- `extraction_classes` - custom signal types for the extraction pipeline
- `watchlist` - entities to monitor

Stored in Supabase `event_configs` table (24h cache TTL).

---

## 3. POSITIONS, ORDERS & P&L

### `get_session_state()` → `tools.py:1491`

Returns **current session** data, **filtered to deep_agent trades only**:

**Account Level:**
| Field | Type | Description |
|-------|------|-------------|
| `balance_cents` | int | Cash available |
| `portfolio_value_cents` | int | Total portfolio value |
| `realized_pnl_cents` | int | Closed position P&L |
| `unrealized_pnl_cents` | int | Open position P&L |
| `total_pnl_cents` | int | Realized + unrealized |
| `position_count` | int | Open positions |
| `open_order_count` | int | Active orders |
| `trade_count` | int | Completed trades |
| `win_rate` | float | Win percentage |

**Positions (deep_agent only):**
| Field | Type |
|-------|------|
| `ticker` | str |
| `side` | "yes" / "no" |
| `contracts` | int |
| `avg_price` | int (cents) |
| `current_value` | int (cents) |
| `unrealized_pnl` | int (cents) |

**Recent Fills (last 10):**
| Field | Type |
|-------|------|
| `order_id` | str |
| `ticker` | str |
| `side` | str |
| `contracts` | int |
| `price_cents` | int |
| `timestamp` | float |
| `reasoning` | str (first 200 chars) |

**Open Orders (deep_agent only):**
| Field | Type |
|-------|------|
| `order_id` | str |
| `ticker` | str |
| `side` | str |
| `action` | str |
| `contracts` | int |
| `price_cents` | int |
| `status` | "resting" / "pending" / "partial" |
| `fill_count` | int |
| `placed_at` | float |
| `age_seconds` | float |

**Filtering mechanism**: Primary filter is `strategy_id == "deep_agent"` on TradingAttachment (survives restarts). Fallback: `_traded_tickers` set (current session only).

### `get_true_performance()` → `tools.py:1610`

All-time deep_agent performance. Everything in `get_session_state()` PLUS:

**Settlements (closed trades):**
| Field | Type | Description |
|-------|------|-------------|
| `ticker` | str | Market |
| `event_ticker` | str | Parent event |
| `side` | str | "yes" / "no" |
| `contracts` | int | Position size |
| `net_pnl` | int (cents) | Realized P&L |
| `is_win` | bool | Profitable? |
| `settled_at` | float | Settlement timestamp |

**Per-Event Breakdown:**
| Field | Type |
|-------|------|
| `settled_trades` | int |
| `wins` / `losses` | int |
| `realized_pnl` | int (cents) |
| `unrealized_pnl` | int (cents) |
| `open_positions` | int |

**Source**: Reads `StateContainer.get_trading_summary()` → filters settlements by `strategy_id == "deep_agent"` → groups by event_ticker.

---

## 4. SETTLEMENT & REFLECTION DATA FLOW

### How Settlements Reach the Agent

```
Kalshi REST API (/portfolio/settlements)
    ↓  every 20s via TradingStateSyncer
KalshiDataSync.sync_with_kalshi()
    ↓
TraderState.settlements (list of dicts)
    ↓
V3StateContainer._settled_positions (deque, max 500)
    ↓
get_trading_summary()["settlements"]
    ↓  every 30s via ReflectionEngine._check_settlements()
Match against PendingTrade (by order_id primary, ticker+side+contracts fallback)
    ↓
_handle_settlement() → computes win/loss/P&L
    ↓
generate_reflection_prompt() → LLM reflection
```

### Settlement Fields Available to Reflection

| Field | Type | Description |
|-------|------|-------------|
| `pnl` | float (dollars) | Raw P&L from Kalshi (converted to cents x 100) |
| `price_cents` | int | Exit price |
| `order_id` | str | For precise matching |
| `ticker` | str | Market identifier |
| `side` | str | "yes" / "no" |
| `contracts` | int | Position size |
| `net_pnl` | int (cents) | P&L for strategy filtering |
| `strategy_id` | str | "deep_agent" for filtering |

### PendingTrade (captured at trade time, used for reflection context)

| Field | Type | Description |
|-------|------|-------------|
| `trade_id` | str | Internal ID |
| `ticker` | str | Market |
| `event_ticker` | str | Parent event |
| `side` | str | "yes" / "no" |
| `contracts` | int | Size |
| `entry_price_cents` | int | Entry price |
| `reasoning` | str | Agent's reasoning at trade time |
| `order_id` | str? | Kalshi order ID |
| `extraction_ids` | list[str] | Extraction IDs used for trade decision |
| `extraction_snapshot` | list[dict] | Full extraction data at trade time |
| `gdelt_snapshot` | list[dict] | GDELT queries at trade time |
| `microstructure_snapshot` | dict? | Orderbook + trade flow at trade time |
| `estimated_probability` | int? | Agent's pre-trade probability estimate (0-100) |
| `what_could_go_wrong` | str? | Risk scenario identified |

### What the Reflection Prompt Includes

1. **Trade details**: ticker, side, contracts, entry/exit price, P&L, result
2. **Performance scorecard**: all-time stats, session stats, last-5 trend, per-event breakdown
3. **Open positions**: count, unrealized P&L, winning/losing
4. **True state from Kalshi**: settlements filtered by `strategy_id == "deep_agent"`
5. **Calibration fields**: estimated_probability vs actual outcome, what_could_go_wrong assessment

---

## 5. SAFETY & RISK TOOLS

### `preflight_check(ticker, side, contracts)` → `tools.py:865`

Bundles market data + event context + safety validation:

**Market Data**: ticker, event_ticker, title, yes_bid, yes_ask, spread, intended_side, estimated_limit_price, estimated_cost_cents

**Safety Checks:**
| Check | Description |
|-------|-------------|
| `spread_ok` | Spread within 3-8c range |
| `circuit_breaker_ok` | Ticker not blacklisted |
| `event_exposure_ok` | Won't exceed per-event cap ($1k default) |
| `risk_level_ok` | Event risk acceptable |
| `data_quality_ok` | Have live prices |
| `all_clear` | All checks passed |
| `tradeable` | Overall: can trade execute |
| `blockers` | Array of blocking reasons |
| `warnings` | Non-blocking issues |

### `EventPositionTracker` (injected, shared with coordinator)

Per-event group:
- `yes_sum` / `no_sum` - mutual exclusivity math
- `risk_level` - ARBITRAGE / NORMAL / HIGH_RISK / GUARANTEED_LOSS
- `position_count`, `has_mixed_sides`
- `pnl_estimate` - estimated P&L if all-NO
- `markets_with_positions`

---

## 6. MICROSTRUCTURE DATA

### `get_microstructure(ticker?, window_minutes=5)` → `tools.py:383`

**Trade Flow (per market):**
| Field | Description |
|-------|-------------|
| `yes_trades` / `no_trades` | Side trade counts |
| `total_trades` | Total in window |
| `yes_ratio` | YES / total (0-1.0) |
| `price_drop` | Price change (cents) |
| `last_yes_price` | Most recent YES price |

**Orderbook Signals (if available):**
| Field | Description |
|-------|-------------|
| `imbalance_ratio` | Buy/sell order imbalance |
| `delta_count` | Order update count |
| `spread_open/close/high/low` | Spread evolution |
| `large_order_count` | Whale-sized orders |

**L2 Orderbook Depth (single-market mode only):**
| Field | Description |
|-------|-------------|
| `orderbook_depth.yes_bids` | Top 5 YES bid levels as [[price, qty], ...] |
| `orderbook_depth.yes_asks` | Top 5 YES ask levels |
| `orderbook_depth.no_bids` | Top 5 NO bid levels |
| `orderbook_depth.no_asks` | Top 5 NO ask levels |
| `orderbook_depth.source` | "websocket" or "rest_fallback" |

**Source chain**: `SharedOrderbookState.get_top_levels(5)` via WebSocket → REST `get_orderbook(ticker, depth=5)` fallback.

**Scan mode**: When no ticker specified, returns aggregated activity across all tracked markets sorted by trade volume.

### `get_candlesticks(event_ticker, period?, hours_back?)` → `tools.py`

Fetches historical OHLC data for ALL markets in an event (single API call):

**Per-Market:**
| Field | Type | Description |
|-------|------|-------------|
| `ticker` | str | Market ticker |
| `candle_count` | int | Number of candles returned |
| `open_price` | int? | First candle open (cents) |
| `close_price` | int? | Last candle close (cents) |
| `high` | int? | Period high (cents) |
| `low` | int? | Period low (cents) |
| `total_volume` | int | Total contracts traded |
| `candles` | list | Last 10 candles with full OHLC detail |

**Source**: `V3TradingClientIntegration.get_event_candlesticks()` → Kalshi REST API.

---

## 7. INTELLIGENCE TOOLS

### `get_extraction_signals(market_ticker?, event_ticker?, window_hours=4, limit=20)` → `tools.py:2387`

**PRIMARY DATA SOURCE** - NLP extraction signals from Reddit/news:

| Field | Type | Description |
|-------|------|-------------|
| `market_ticker` | str | Market this signal applies to |
| `event_tickers` | list[str] | Parent events |
| `occurrence_count` | int | Extraction occurrences in window |
| `unique_source_count` | int | Distinct Reddit/news sources |
| `total_engagement` | int | Sum of upvotes + comments |
| `max_engagement` | int | Peak single post engagement |
| `directions` | dict | `{bullish: N, bearish: N}` |
| `consensus` | str | "bullish" / "bearish" / "neutral" |
| `consensus_strength` | float | 0.0-1.0 |
| `avg_magnitude` | float | Price impact magnitude (0-10) |
| `entity_mentions` | list | Named entities with sentiment |
| `context_factors` | list | Environmental signals |
| `recent_extractions` | list | Top 3 by engagement with full text |

### `get_reddit_daily_digest(force_refresh?)` → `tools.py:2682`
Top 25 posts from past 24h across tracked subreddits with comment analysis.

### `query_gdelt_news(search_terms, window_hours, ...)` → `tools.py:2711`
BigQuery GKG: article count, source diversity, tone summary, key themes/persons/orgs, top articles, timeline.

### `query_gdelt_events(search_terms, window_hours, ...)` → `tools.py:2786`
GDELT 2.0 Events: actor-event-actor triples, Goldstein scale, quad class, geo hotspots.

### `search_gdelt_articles(...)` / `get_gdelt_volume_timeline(...)` → DOC API
Free GDELT article search and volume timeline (no BigQuery needed).

---

## 8. MEMORY & PLANNING TOOLS

### Persistent Memory Files (`./memory/`)
| File | Purpose | Access |
|------|---------|--------|
| `learnings.md` | Discovered patterns & insights | read/append |
| `strategy.md` | Current trading strategy (3KB cap, versioned) | read/write |
| `mistakes.md` | Past failures to avoid | read/append |
| `patterns.md` | Market patterns | read/append |
| `golden_rules.md` | Hard constraints | read/write |
| `cycle_journal.md` | Decision log per cycle | append (auto-archive at 10KB) |
| `market_knowledge.md` | Market-specific facts | read/append |
| `gdelt_reference.md` | News intelligence reference | read/append |
| `todos.json` | Task list with priorities | read/write (full replace) |

### Structured Reasoning Tools
- `think()` - Pre-trade analysis (signal_analysis, strategy_check, risk_assessment, decision, estimated_probability, what_could_go_wrong)
- `reflect()` - Post-trade learning (outcome_analysis, reasoning_accuracy, key_learning, mistake, pattern, strategy_update_needed)
- `write_cycle_summary()` - End-of-cycle journal (signals_observed, decisions_made, reasoning_notes, markets_of_interest)
- `evaluate_extractions()` - Score extraction accuracy after settlement, auto-promote accurate extractions as examples
- `refine_event()` - Push learnings back to extraction pipeline (what_works, what_fails, watchlist additions)
- `get_extraction_quality()` - Quality metrics per event

---

## 9. DATA REFRESH FREQUENCIES

| Data | Refresh Rate | Source |
|------|-------------|--------|
| Orderbook (bid/ask) | Real-time via WebSocket | OrderbookIntegration |
| Positions/Orders | Every 20s | TradingStateSyncer → REST API |
| Settlements | Every 20s (sync) + 30s (reflection check) | TradingStateSyncer + ReflectionEngine |
| Market prices | Real-time (WS ticker) + 30s (REST fallback) | MarketTickerListener + MarketPriceSyncer |
| Tracked markets | On lifecycle events + 300s API discovery | EventLifecycleService + ApiDiscoverySyncer |
| Extraction signals | On-demand (tool call queries Supabase) | `get_extraction_signals()` |
| Trade flow | Real-time via trades WebSocket | TradeFlowService |

---

## 10. RESOLVED DATA GAPS (2026-02-01)

1. **~~L2/L3 orderbook depth not surfaced~~** — RESOLVED. `get_microstructure(ticker)` now returns `orderbook_depth` with top 5 price levels per side (yes_bids, yes_asks, no_bids, no_asks) as [price, quantity] pairs via `SharedOrderbookState.get_top_levels()`. REST fallback via `get_orderbook()` when WS data unavailable.
2. **~~No historical candlestick/OHLC data~~** — RESOLVED. New `get_candlesticks(event_ticker, period, hours_back)` tool fetches OHLC data for all markets in an event via the event-level candlesticks endpoint (single API call). Returns open/close/high/low/volume summary + last 10 candles per market.
3. **~~Order lifecycle visibility gap~~** — RESOLVED. `get_session_state()` now includes `open_orders` section showing all resting/pending/partial deep_agent orders with order_id, side, action, contracts, price, status, fill_count, and age_seconds. Uses `TradingAttachment.orders` filtered by `strategy_id == "deep_agent"`.
4. **~~Position view is session-scoped~~** — RESOLVED. `get_session_state()` now uses `strategy_id == "deep_agent"` as primary position filter (survives restarts via TradingAttachment), with `_traded_tickers` as fallback for current session. Matches pattern already used by `get_true_performance()`.
