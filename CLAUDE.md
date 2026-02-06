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
   - Single-event arbitrage detection
   - Mentions market probability estimation
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
│  │EventArb  │──│ Monitor  │──│         Captain              │  │
│  │Index     │  │          │  │  ┌────────────────────────┐  │  │
│  │(markets) │  │(data in) │  │  │ Subagents:             │  │  │
│  └──────────┘  └──────────┘  │  │ - TradeCommando (exec) │  │  │
│                              │  │ - ChevalDeTroie (surv) │  │  │
│                              │  │ - MentionsSpecialist   │  │  │
│                              │  │ - MemoryCurator        │  │  │
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
| Mentions Tools | `backend/src/kalshiflow_rl/traderv3/single_arb/mentions_tools.py` |
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
# Start V3 trader (paper trading, auto-discovers markets, port 8005)
./scripts/run-v3.sh

# Start with custom settings: [env] [mode] [market_limit]
./scripts/run-v3.sh paper discovery 20

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
# Backend E2E (golden standard - MUST pass before deploy)
cd backend && uv run pytest tests/test_backend_e2e_regression.py -v

# Frontend E2E (requires backend on :8000)
cd frontend && npm run test:frontend-regression
```

### Deployment

```bash
# ONLY deploy when explicitly requested by user
./deploy.sh
```

## Captain Agent Architecture

### Overview

The Captain is a multi-agent LLM system using `deepagents` framework with Claude Sonnet. It runs on a 60-second cycle, observing markets, delegating to specialized subagents, and learning from experience.

### Subagents

| Agent | Role | Key Tools |
|-------|------|-----------|
| **TradeCommando** | Execution specialist | `place_order`, `execute_arb`, `cancel_order`, `get_resting_orders` |
| **ChevalDeTroie** | Surveillance/bot detection | `analyze_microstructure`, `analyze_orderbook_patterns` |
| **MentionsSpecialist** | Mentions market edge | `simulate_probability`, `trigger_simulation`, `compute_edge` |
| **MemoryCurator** | Memory maintenance | `memory_store`, file operations |

### Memory Architecture

```
┌────────────────┐   ┌─────────────────┐   ┌────────────────┐
│  AGENTS.md     │   │  journal.jsonl  │   │   pgvector     │
│  (persistent)  │   │  (append-only)  │   │  (semantic)    │
│                │   │                 │   │                │
│  Learnings     │   │  Trade records  │   │  Search across │
│  loaded into   │   │  via memory_    │   │  historical    │
│  system prompt │   │  store tool     │   │  context       │
└────────────────┘   └─────────────────┘   └────────────────┘
```

### Captain Tools (13 total)

**Observation:**
- `get_events_summary()` - All events with edge calculations
- `get_event_snapshot(event_ticker)` - Full orderbook depth for one event
- `get_market_orderbook(ticker)` - 5 levels of orderbook
- `get_trade_history(ticker)` - Fills, settlements, P&L
- `get_positions()` - Current positions (captain vs legacy split)
- `get_balance()` - Account balance

**Execution:**
- `place_order(ticker, side, contracts, price, reasoning)` - Single-leg order
- `execute_arb(event_ticker, direction, max_contracts, reasoning)` - Multi-leg arb
- `cancel_order(order_id, reason)` - Cancel resting order

**Memory:**
- `memory_store(content, memory_type, metadata)` - Persist learnings

### Mentions Strategy System

Blind LLM simulation for mentions markets (e.g., "Will Trump say 'tariff'?"):

1. **Blind transcript generation** - LLM generates realistic speech without knowing target terms
2. **Post-hoc scanning** - Count term appearances in generated text
3. **P(term) estimation** - appearances / n_simulations
4. **Edge calculation** - Blend baseline + informed probabilities

Key tools: `simulate_probability`, `trigger_simulation` (async), `compute_edge`, `query_wordnet`

## Development Patterns

### LangChain Tools Pattern

```python
# Tools use module-level globals for dependency injection
_index: Optional[EventArbIndex] = None
_trading_client: Optional[KalshiDemoTradingClient] = None

def set_dependencies(index, client, order_group_id, order_ttl):
    """Called by coordinator at startup"""
    global _index, _trading_client, _order_group_id, _order_ttl
    _index = index
    _trading_client = client
    # ...

@tool
async def get_events_summary() -> str:
    """Get summary of all events with edge calculations."""
    # Implementation using _index
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
# File-based persistence via FilesystemBackend
# /memories/ and /skills/ routes → FilesystemBackend (persistent)
# Everything else → StateBackend (ephemeral per thread_id)

# Each cycle gets unique thread_id to prevent history accumulation
thread_id = str(uuid.uuid4())
```

## Configuration

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

# Mentions
V3_MENTIONS_ENABLED=true
```

### Environment Files

- `.env.paper` - Paper trading (demo-api.kalshi.co)
- `.env.local` - Local development
- `.env.production` - Production

Switch with: `./scripts/switch-env.sh paper`

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

1. Backend E2E test passes
2. Frontend E2E test passes
3. Health endpoints responding

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
