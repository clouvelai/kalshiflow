# CLAUDE.md

Project guidance for Claude Code. Focus: Captain agent development on Kalshi prediction markets.

## Project Overview

**Kalshi Flow** - A real-time prediction market trading platform with two main products:

1. **Flowboard** (shipped, production) - https://kalshiflow.io/
   - Live trade tape showing public Kalshi trades in real-time
   - Hot markets heatmap based on volume and flow direction
   - Time-series analytics and market insights
   - WebSocket-driven real-time updates

2. **V3 Trader + Captain** (active development)
   - LLM-powered Captain agent for autonomous trading
   - Single-event arbitrage detection with attention-driven invocation
   - Sniper auto-execution layer for arb opportunities
   - Paper trading on demo-api.kalshi.co

## Production

**Live URL**: https://kalshiflow.io/

Railway services:
- `kalshi-flowboard-backend` - Python backend + WebSocket
- `kalshi-flowboard` - Static frontend

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.x + Starlette (ASGI), uv for deps |
| Frontend | React + Vite + Tailwind, npm for deps |
| Database | PostgreSQL via Supabase |
| LLM Framework | deepagents + LangChain tools + Claude Sonnet |
| Deployment | Railway (backend + frontend) |

## Flowboard Backend (Production)

The core Kalshiflow backend at `backend/src/kalshiflow/`:

```
Kalshi WebSocket (public trades) → Backend → PostgreSQL + In-Memory Aggregates
                                      ↓
                              Frontend WebSocket
                                      ↓
                         Trade Tape + Hot Markets + Charts
```

Key modules:
- `kalshi_client.py` - WebSocket connection to Kalshi with RSA auth
- `trade_processor.py` - Processes incoming trades, deduplication
- `aggregator.py` - Hot markets, ticker states, sliding window aggregates
- `websocket_handler.py` - Frontend WebSocket connections
- `database.py` - PostgreSQL via Supabase

## Architecture

### Core Systems

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

### Key Files

| Component | Location |
|-----------|----------|
| V3 Coordinator | `backend/src/kalshiflow_rl/traderv3/core/coordinator.py` |
| Event Bus | `backend/src/kalshiflow_rl/traderv3/core/event_bus.py` |
| **Captain Agent** | `backend/src/kalshiflow_rl/traderv3/single_arb/captain.py` |
| Captain Tools | `backend/src/kalshiflow_rl/traderv3/single_arb/tools.py` |
| Captain Models | `backend/src/kalshiflow_rl/traderv3/single_arb/models.py` |
| Context Builder | `backend/src/kalshiflow_rl/traderv3/single_arb/context_builder.py` |
| AttentionRouter | `backend/src/kalshiflow_rl/traderv3/single_arb/attention.py` |
| AutoActionManager | `backend/src/kalshiflow_rl/traderv3/single_arb/auto_actions.py` |
| Sniper Execution | `backend/src/kalshiflow_rl/traderv3/single_arb/sniper.py` |
| Task Ledger | `backend/src/kalshiflow_rl/traderv3/single_arb/task_ledger.py` |
| Account Health | `backend/src/kalshiflow_rl/traderv3/single_arb/account_health.py` |
| Single-Arb Coordinator | `backend/src/kalshiflow_rl/traderv3/single_arb/coordinator.py` |
| EventArb Index | `backend/src/kalshiflow_rl/traderv3/single_arb/index.py` |
| Memory System | `backend/src/kalshiflow_rl/traderv3/single_arb/memory/` |
| V3 Config | `backend/src/kalshiflow_rl/traderv3/config/environment.py` |

## Quick Commands

### Flowboard (Production Backend)

```bash
# Start Flowboard backend (port 8000)
cd backend && uv run uvicorn kalshiflow.app:app --reload

# Frontend (separate terminal)
cd frontend && npm run dev
```

### V3 Trader (Captain Development)

```bash
# Start Captain (paper trading, auto-discovers markets, port 8005)
./scripts/run-captain.sh

# Start with custom settings: [env] [mode] [market_limit]
./scripts/run-captain.sh paper discovery 20

# Frontend for arb dashboard
cd frontend && npm run dev
# Then visit http://localhost:5173/arb
```

### Access Points

**Flowboard (production):**
- **Production**: https://kalshiflow.io/
- **Local**: http://localhost:5173/ (requires backend on :8000)
- **Backend WebSocket**: ws://localhost:8000/ws

**V3 Trader (development):**
- **Arb Dashboard**: http://localhost:5173/arb
- **Health**: http://localhost:8005/v3/health
- **Status**: http://localhost:8005/v3/status
- **WebSocket**: ws://localhost:8005/v3/ws

### Testing

```bash
# V3 Trader unit tests (214 tests, ~3s, no network/LLM/DB required)
cd backend && uv run pytest tests/traderv3/ -v --tb=short

# Backend E2E (golden standard - MUST pass before deploy)
cd backend && uv run pytest tests/test_backend_e2e_regression.py -v

# Frontend E2E (requires backend on :8000)
cd frontend && npm run test:frontend-regression

# Captain E2E smoke test (~30-60s, requires .env.paper credentials)
cd backend && uv run pytest tests/test_captain_e2e_smoke.py -v

# Against already-running server
V3_SMOKE_EXTERNAL_URL=http://localhost:8005 uv run pytest tests/test_captain_e2e_smoke.py -v

# Run everything (unit + E2E)
cd backend && uv run pytest tests/traderv3/ tests/test_backend_e2e_regression.py tests/test_captain_e2e_smoke.py -v --tb=short
```

**Unit test files** (`backend/tests/traderv3/`):

| File | Tests | Scope |
|------|-------|-------|
| `test_index.py` | ~40 | MarketMeta, EventMeta, arb detection (pure) |
| `test_config.py` | ~15 | V3Config validation (pure) |
| `test_state_machine.py` | ~22 | State transitions (mocked) |
| `test_event_bus.py` | ~17 | Pub/sub, coalescing (mocked) |
| `test_tools.py` | ~60 | V2 Captain tools with ToolContext (mocked) |
| `test_sniper.py` | 30 | Sniper execution layer (mocked) |
| `test_attention.py` | ~50 | AttentionRouter signal scoring (pure) |
| `test_auto_actions.py` | ~40 | AutoActionManager rules (mocked) |
| `test_context_builder.py` | ~40 | Mode-specific context injection (pure) |
| `test_models_v2.py` | ~30 | Pydantic model validation (pure) |
| `test_issues.py` | ~14 | Issue lifecycle (tmp_path) |

### Deployment

```bash
# ONLY deploy when explicitly requested by user
./deploy.sh
```

## Captain Agent Architecture

### Overview

The Captain is a single-agent, attention-driven LLM system using `deepagents` framework with Claude Sonnet. It has 3 invocation modes driven by an AttentionRouter (reactive) and timers (strategic every 5min, deep_scan every 30min).

### Invocation Modes

| Mode | Trigger | Context Size | Purpose |
|------|---------|-------------|---------|
| **Reactive** | AttentionRouter signal | ~200-400 tokens | Respond to urgent signals (edge shifts, fills, regime changes) |
| **Strategic** | Every 5 minutes | ~400-600 tokens | Portfolio review, sniper tuning, task planning |
| **Deep Scan** | Every 30 minutes | ~800-1200 tokens | Full event scan, health check, memory recall, rebalancing |

### Key Subsystems

| System | File | Role |
|--------|------|------|
| **AttentionRouter** | `attention.py` | Scores signals, emits AttentionItems, notifies Captain via asyncio.Event |
| **AutoActionManager** | `auto_actions.py` | Deterministic rules: stop_loss, time_exit, regime_gate (Captain can override) |
| **Sniper** | `sniper.py` | Auto-executes arb opportunities through risk gates (edge, capital, VPIN, cooldown) |
| **TaskLedger** | `task_ledger.py` | Externalized working memory — intercepts write_todos, tracks priority/staleness |
| **ContextBuilder** | `context_builder.py` | Mode-specific prompt construction with token budget awareness |
| **SessionMemoryStore** | `memory/session_store.py` | FAISS (session, <1ms) + pgvector (persistent, fire-and-forget) |

### Memory Architecture

```
┌────────────────────┐   ┌─────────────────┐   ┌────────────────┐
│  FAISS (session)   │   │  Task Ledger    │   │   pgvector     │
│  <1ms retrieval    │   │  (working mem)  │   │  (persistent)  │
│                    │   │                 │   │                │
│  Recent insights,  │   │  [HIGH]/[MED]/  │   │  Cross-session │
│  via recall_memory │   │  [LOW] tasks    │   │  semantic      │
│  LRU @ 2000 items  │   │  Stale detect   │   │  search        │
└────────────────────┘   └─────────────────┘   └────────────────┘
```

### Captain Tools (13 total)

**Observation:**
- `get_market_state(event_ticker)` - Market data with orderbook, microstructure, edge
- `get_portfolio()` - Positions, P&L, balance
- `get_account_health()` - Drawdown, risk metrics, retention policy
- `get_resting_orders()` - Open orders with fill status

**Execution:**
- `place_order(ticker, side, contracts, price, reasoning)` - Single-leg order
- `execute_arb(event_ticker, direction, max_contracts, reasoning)` - Multi-leg arb
- `cancel_order(order_id, reason)` - Cancel resting order

**Configuration:**
- `configure_sniper(settings)` - Tune sniper edge threshold, capital limits, cooldowns
- `configure_automation(settings)` - Tune auto-action rules (stop_loss, time_exit, regime_gate)

**Intelligence:**
- `search_news(query)` - Search news for event context
- `recall_memory(query)` - Semantic search across FAISS + pgvector
- `store_insight(content, tags)` - Persist learnings to memory
- `get_market_movers(event_ticker)` - Find news articles that moved prices (from news_price_impacts)

## Development Patterns

### LangChain Tools Pattern

```python
# ToolContext dataclass replaces module-level globals
@dataclass
class ToolContext:
    index: EventArbIndex
    trading_client: KalshiDemoTradingClient
    order_tracker: OrderTracker
    sniper: Optional[SniperExecutor] = None
    auto_actions: Optional[AutoActionManager] = None
    # ... all dependencies in one place

_ctx: Optional[ToolContext] = None

def set_context(ctx: ToolContext):
    """Called by coordinator._setup_tools() at startup."""
    global _ctx
    _ctx = ctx

@tool
async def get_market_state(event_ticker: str) -> str:
    """Get market data with orderbook, microstructure, edge."""
    # All tools access dependencies via _ctx
```

### EventBus Pattern

```python
# Subscribe to events
event_bus.subscribe(EventType.ORDERBOOK_SNAPSHOT, self._on_orderbook)

# Callback receives market_ticker and metadata
async def _on_orderbook(self, market_ticker: str, metadata: Dict):
    yes_bid = metadata.get("yes_bid")
    yes_ask = metadata.get("yes_ask")
    # ...
```

### Memory Persistence Pattern

```python
# SessionMemoryStore: dual-layer memory
# FAISS (session) — <1ms retrieval, LRU eviction at 2000 entries
# pgvector (persistent) — fire-and-forget writes, semantic search

# Each Captain cycle gets unique thread_id to prevent history accumulation
thread_id = str(uuid.uuid4())

# Tools: recall_memory (search), store_insight (persist)
# Shutdown: SessionMemoryStore.flush() awaits pending pgvector writes
```

## Configuration

### Model Configuration

LLM model selection is centralized into 4 tiers, configured via env vars or `V3Config` fields.
All consumers read from `mentions_models.py` getters after `configure(config)` is called at startup.

| Tier | Env Var | Default | Used By |
|------|---------|---------|---------|
| **captain** | `V3_MODEL_CAPTAIN` | `claude-sonnet-4-20250514` | Captain agent |
| **subagent** | `V3_MODEL_SUBAGENT` | `claude-haiku-4-5-20251001` | EventUnderstanding synthesis |
| **utility** | `V3_MODEL_UTILITY` | `gemini-2.0-flash` | (reserved for future use) |
| **embedding** | `V3_MODEL_EMBEDDING` | `text-embedding-3-small` | VectorMemoryService (pgvector) |

**Why subagent tier for structured output consumers**: Gemini Flash has a known issue dropping list fields in `.with_structured_output()`. EventUnderstanding uses structured output with lists, so it uses the subagent tier (Haiku) instead of utility tier.

### Key Environment Variables

```bash
# V3 Trader
V3_PORT=8005
V3_ORDER_TTL_ENABLED=true
V3_ORDER_TTL_SECONDS=90

# Single-Arb System
V3_SINGLE_ARB_ENABLED=true
V3_SINGLE_ARB_CAPTAIN_ENABLED=true
V3_SINGLE_ARB_CAPTAIN_INTERVAL=60
V3_SINGLE_ARB_MIN_EDGE_CENTS=0.5
V3_SINGLE_ARB_ORDER_TTL=30

# LLM Models (centralized tiers)
V3_MODEL_CAPTAIN=claude-sonnet-4-20250514
V3_MODEL_SUBAGENT=claude-haiku-4-5-20251001
V3_MODEL_UTILITY=gemini-2.0-flash
V3_MODEL_EMBEDDING=text-embedding-3-small
```

### Environment Files

- `.env.paper` - Paper trading (demo-api.kalshi.co)
- `.env.local` - Local development
- `.env.production` - Production

Switch with: `./scripts/switch-env.sh paper`

### Subaccount Management

The V3 trader uses Kalshi subaccounts to isolate Captain's balance from the main account. CLI tool: `scripts/manage_subaccount.py` (must run via `cd backend && uv run python ../scripts/manage_subaccount.py`).

```bash
# Show all subaccount balances
cd backend && uv run python ../scripts/manage_subaccount.py balances

# Create a new subaccount
cd backend && uv run python ../scripts/manage_subaccount.py create

# Transfer $10,000 from subaccount 0 (main) to subaccount 1
cd backend && uv run python ../scripts/manage_subaccount.py transfer 0 1 10000
```

After creating/transferring, update `.env.paper`:
```bash
V3_SUBACCOUNT=1  # Set to the subaccount number the Captain should trade on
```

**API notes** (Kalshi subaccounts endpoint quirks):
- `GET /portfolio/subaccounts/balances` returns `balance` as a **string in dollars** (e.g. `"56.0000"`), not int cents
- The endpoint does **not** return `portfolio_value` -- it's computed from positions' `market_exposure` downstream
- `POST /portfolio/subaccounts/transfer` requires `client_transfer_id` (UUID), `from_subaccount`, `to_subaccount`, `amount_cents`

## Subagent Selection Guide

Use the Task tool with these specialized agents:

| Task | Agent | When to Use |
|------|-------|-------------|
| **Captain debugging** | `kalshi-flow-trader-specialist` | Debug trader issues, state machine, trading client |
| **Prompt improvements** | `prompt-engineer` | Design/fix LLM prompts, schema validation |
| **WebSocket/real-time** | `fullstack-websocket-engineer` | WebSocket features, performance, test fixes |
| **Deployment** | `deployment-engineer` | Railway deployment, Supabase config |

### Quick Reference

- "Debug why Captain isn't trading" → `kalshi-flow-trader-specialist`
- "Fix Captain's prompt for better edge detection" → `prompt-engineer`
- "WebSocket messages not reaching frontend" → `fullstack-websocket-engineer`
- "Deploy to production" → `deployment-engineer`

## Skills

Available skills (invoke with `/skill-name`):

| Skill | Purpose |
|-------|---------|
| `/iterate-captain` | Step-by-step Captain debugging - runs 2 cycles, pauses, analyzes, fixes |

## Code Quality Standards

### Module Docstrings

Every file MUST have a comprehensive docstring:
- Purpose, Key Responsibilities, Architecture Position, Design Principles

### Type Safety

- Use `TYPE_CHECKING` for forward references
- Dataclasses for structured data
- Enums for constants
- Type hints on all functions

### Async Patterns

- NEVER block event loop with sync operations
- Use `asyncio.create_task()` for background work
- Handle `asyncio.CancelledError` in long-running loops
- Use timeouts for external calls

### Logging

```python
logger = logging.getLogger("kalshiflow_rl.traderv3.component_name")
```

## Testing Workflow

### Before Deployment (Mandatory)

1. V3 unit tests pass: `cd backend && uv run pytest tests/traderv3/ --tb=short`
2. Backend E2E test passes: `cd backend && uv run pytest tests/test_backend_e2e_regression.py -v`
3. Frontend E2E test passes
4. Health endpoints responding

### Captain Iteration

Use `/iterate-captain` skill for step-by-step debugging:
1. Starts V3 trader
2. Runs 2 Captain cycles
3. Auto-pauses
4. Analyzes logs for issues
5. Fixes or escalates
6. Resume/restart based on changes

## Critical Constraints

- **NEVER deploy without explicit user request**
- **NEVER modify E2E tests to make them pass** - fix the code
- **Paper trading only** - demo-api.kalshi.co
- **Context management** - Use `/clear` between unrelated tasks

## Railway Deployment

Services:
- `kalshi-flowboard-backend` - Python backend + WebSocket
- `kalshi-flowboard` - Static frontend

```bash
# Check Railway status
railway status

# View logs
railway logs --service kalshi-flowboard-backend

# Manual deploy (prefer ./deploy.sh)
railway up --service kalshi-flowboard-backend
```

## Useful Debugging

### Captain Not Trading?

1. Check health: `curl http://localhost:8005/v3/health`
2. Check status: `curl http://localhost:8005/v3/status | jq '.captain'`
3. Check logs: `grep "SINGLE_ARB" backend/logs/v3-trader.log`
4. Use `/iterate-captain` for systematic debugging

### WebSocket Issues?

1. Check browser console at http://localhost:5173/arb
2. Check backend logs for WebSocket errors
3. Use `fullstack-websocket-engineer` agent

### Deployment Issues?

1. Run E2E tests first
2. Check Railway logs: `railway logs`
3. Use `deployment-engineer` agent
