# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kalshi Flowboard - A real-time web application that displays Kalshi public trades via WebSocket, showing a live tape and heatmap of "hot" markets based on trade volume and flow direction.

## Architecture

### Tech Stack
- **Backend**: Python 3.x + Starlette (ASGI) with uv for dependency management
- **Frontend**: React + Vite + Tailwind CSS with npm for dependency management
- **Database**: PostgreSQL via Supabase (production + development)
- **Authentication**: RSA private key file-based auth for Kalshi API

### Core Constraint
Do not call any Kalshi REST endpoints in the MVP. All data must originate from the WebSocket public trades stream.

### Data Flow
1. Kalshi WebSocket → Backend (auth with RSA signature)
2. Backend processes trades → PostgreSQL (durable) + In-memory (aggregates)
3. Backend broadcasts → Frontend WebSocket
4. Frontend displays → Trade tape + Hot markets + Ticker details

## Common Commands

### Development Setup

#### Quick Start (PostgreSQL/Supabase)
```bash
# Start local Supabase instance
cd backend && supabase start

# Install dependencies and run backend
uv sync
uv run uvicorn kalshiflow.app:app --reload

# In separate terminal: run frontend
cd frontend && npm install && npm run dev
```

#### Alternative Setup
```bash
# Initialize and install all dependencies
./init.sh

# Backend only (using uv)
cd backend
uv sync
uv run uvicorn kalshiflow.app:app --reload

# Frontend only (using npm)
cd frontend
npm install
npm run dev
```

#### Environment Management
```bash
# Switch between local/production environments
./scripts/switch-env.sh local       # Use local Supabase
./scripts/switch-env.sh production  # Use remote Supabase
./scripts/switch-env.sh current     # Show current environment

# PostgreSQL is used in both environments via Supabase
# See SUPABASE_SETUP.md for detailed configuration
```

### V3 Trader (Paper Trading)
**Purpose**: Clean architecture trading system with WebSocket-driven state management
```bash
# Start V3 trader (default: paper trading, discovery mode, 10 markets)
./scripts/run-v3.sh

# Custom configurations: [environment] [mode] [market_limit]
./scripts/run-v3.sh paper discovery 20      # More markets
./scripts/run-v3.sh paper config 5          # Specific tickers from RL_MARKET_TICKERS
```

**Default Configuration:**
- **Environment**: `paper` (demo account)
- **Port**: 8005
- **Mode**: Discovery (auto-discovers active markets)
- **Markets**: 10 orderbook subscriptions

**Access Points:**
- **Console**: http://localhost:5173/v3-trader (requires frontend)
- **Health**: http://localhost:8005/v3/health
- **Status**: http://localhost:8005/v3/status
- **WebSocket**: ws://localhost:8005/v3/ws

**Order Management:**

*Order TTL (Time-to-Live)*: Resting orders are automatically cancelled after a configurable timeout to prevent capital lock-up.
- `V3_ORDER_TTL_ENABLED`: Enable/disable TTL cleanup (default: `true`)
- `V3_ORDER_TTL_SECONDS`: TTL threshold in seconds (default: `300` = 5 minutes)
- Runs after each sync cycle (~30 seconds), only cancels "resting" orders (not partial fills)
- Emits toast notification to frontend, tracks count in "TTL Cancelled" metric

*Startup Cleanup*: On restart, old order groups are automatically cleaned up:
1. **Reset** each old order group (cancels all resting orders)
2. **Delete** the empty order group
3. Current session's order group is preserved
- `cleanup_on_startup: True` (default) - Enable startup cleanup

### Testing
```bash
# Backend tests
cd backend
uv run pytest

# CRITICAL: Backend E2E regression test (golden standard)
# Run this before any deployment to ensure entire backend pipeline works
uv run pytest tests/test_backend_e2e_regression.py -v

# Detailed validation output for debugging
uv run pytest tests/test_backend_e2e_regression.py -v -s --log-cli-level=INFO

# RL Orderbook Collector E2E test (for RL subsystem)
uv run pytest tests/test_rl_orderbook_e2e.py -v
# Or use the test script
./scripts/test_rl_e2e.sh

# Test Kalshi client standalone
uv run backend/scripts/test_kalshi_client.py

# Frontend E2E tests (requires backend running on port 8000)
cd frontend

# CRITICAL: Frontend E2E regression test (golden standard)
# Comprehensive validation of entire frontend against live data
npm run test:frontend-regression

# All frontend E2E tests
npm run test:e2e

# Interactive UI for debugging
npm run test:e2e-ui
```

## Key Implementation Details

### Kalshi Authentication
- Uses RSA private key file (not API secret)
- Environment variables:
  - `KALSHI_API_KEY_ID` - The API key ID
  - `KALSHI_PRIVATE_KEY_CONTENT` - RSA private key content as string (or `KALSHI_PRIVATE_KEY_PATH` for file path)
- Signature process: `timestamp_ms + method + path` → RSA sign with PSS padding → base64 encode
- Reference implementation: https://github.com/clouvelai/prophete/blob/main/backend/app/core/auth.py

### WebSocket Message Protocol

**Frontend receives from backend:**
```json
// Snapshot (on connect)
{
  "type": "snapshot",
  "data": {
    "recent_trades": [...],
    "hot_markets": [...]
  }
}

// Trade update
{
  "type": "trade",
  "data": {
    "trade": {...},
    "ticker_state": {...}
  }
}
```

### In-Memory Aggregation
- Sliding window aggregates (default 10 minutes)
- Per-ticker state tracks: volume, yes/no flow, price points for sparklines
- Periodic pruning of old data to maintain performance

## Implementation Roadmap

The project follows a phased approach documented in `feature_plan.json`:

**Phase 1 (Current)**: End-to-end live public trades feed
- Milestone 1: Project initialization
- Milestone 2: Kalshi WebSocket client with RSA auth
- Milestone 3: Backend integration with Starlette
- Milestone 4: Frontend implementation

**Phase 2 (Future)**: Storage, aggregation, and advanced features

## File Structure

```
backend/
  src/kalshiflow/
    main.py         # Entry point
    app.py          # Starlette app
    auth.py         # RSA authentication
    kalshi_client.py # WebSocket client
    models.py       # Pydantic models
    database.py     # PostgreSQL setup
    aggregator.py   # In-memory aggregation
    trade_processor.py # Trade handling
    websocket_handler.py # Frontend WebSocket
frontend/
  src/
    components/     # React components
    hooks/          # Custom hooks
    context/        # State management
```

## Environment Configuration

The project uses environment-specific `.env` files for credential management:

### Environment Files
- **`.env.local`** - Local development environment (default)
- **`.env.production`** - Production environment
- **`.env.paper`** - Paper trading environment (demo account)

### Environment Loading Pattern
The system loads environment variables based on the `ENVIRONMENT` variable:
1. Loads `.env.{ENVIRONMENT}` first (with override=True)
2. Falls back to `.env` for any missing variables

**Example:**
```bash
# Set environment
export ENVIRONMENT=paper  # or "local" or "production"

# System will load:
# 1. .env.paper (override=True)
# 2. .env (fallback)
```

### Required Variables by Environment

#### Local Development (`.env.local`)
```bash
ENVIRONMENT=local

# Kalshi API Configuration (production API for data collection)
KALSHI_API_KEY_ID=<your_api_key_id>
KALSHI_PRIVATE_KEY_CONTENT=<your_private_key_content>
KALSHI_API_URL=https://api.elections.kalshi.com/trade-api/v2
KALSHI_WS_URL=wss://api.elections.kalshi.com/trade-api/ws/v2

# PostgreSQL Database Configuration (via Supabase)
DATABASE_URL=<postgresql_connection_string>
DATABASE_URL_POOLED=<postgresql_pooled_connection_string>

# Application Settings
WINDOW_MINUTES=10
HOT_MARKETS_LIMIT=20
RECENT_TRADES_LIMIT=200

# Server Configuration
BACKEND_PORT=8000
FRONTEND_PORT=5173
```

#### Production (`.env.production`)
```bash
ENVIRONMENT=production

# Kalshi API Configuration (production API)
KALSHI_API_KEY_ID=<your_production_api_key_id>
KALSHI_PRIVATE_KEY_CONTENT=<your_production_private_key_content>
KALSHI_API_URL=https://api.elections.kalshi.com/trade-api/v2
KALSHI_WS_URL=wss://api.elections.kalshi.com/trade-api/ws/v2

# PostgreSQL Database Configuration
DATABASE_URL=<production_postgresql_connection_string>
DATABASE_URL_POOLED=<production_pooled_connection_string>
```

#### Paper Trading (`.env.paper`)
```bash
ENVIRONMENT=paper

# Kalshi Demo Account API Configuration (demo-api.kalshi.co)
KALSHI_API_KEY_ID=<your_demo_api_key_id>
KALSHI_PRIVATE_KEY_CONTENT=<your_demo_private_key_content>
KALSHI_API_URL=https://demo-api.kalshi.co/trade-api/v2
KALSHI_WS_URL=wss://demo-api.kalshi.co/trade-api/ws/v2

# Database Configuration (shared with other environments)
DATABASE_URL=<postgresql_connection_string>
DATABASE_URL_POOLED=<postgresql_pooled_connection_string>

# RL-Specific Configuration (optional)
RL_MARKET_TICKERS=INXD-25JAN03,OTHER-TICKER
RL_LOG_LEVEL=INFO
```

### Environment Switching

Use the provided script to switch between environments:
```bash
# Switch to local development
./scripts/switch-env.sh local

# Switch to production
./scripts/switch-env.sh production

# Show current environment
./scripts/switch-env.sh current
```

### Paper Trading Safety

The `KalshiDemoTradingClient` includes validation to prevent accidental production trading:
- ✅ **Validates URLs**: Ensures all URLs point to `demo-api.kalshi.co`
- ✅ **Blocks Production**: Raises error if production URLs (`api.elections.kalshi.com`) are detected
- ✅ **Clear Errors**: Provides helpful error messages directing users to use `ENVIRONMENT=paper`

**To use paper trading:**
1. Copy `.env.paper.example` to `.env.paper`
2. Fill in your demo account credentials
3. Set `ENVIRONMENT=paper` or use the environment switcher
4. The demo client will automatically validate the configuration
## Backend E2E Regression Test

**Critical test that MUST pass before any deployment or major changes.**

### What it validates:
- ✅ **Backend Startup**: Complete application starts successfully
- ✅ **Service Integration**: All services (trade processor, aggregator, websocket manager) initialize
- ✅ **Kalshi Connection**: WebSocket client connects to Kalshi public trades stream
- ✅ **Database Functionality**: PostgreSQL database connection and functionality
- ✅ **Frontend WebSocket**: Client connections work and receive valid data
- ✅ **Data Processing**: Trade data flows through complete pipeline (when available)
- ✅ **Clean Shutdown**: All services stop gracefully

### Running the test:
```bash
# Standard run (must pass before deployment)
uv run pytest tests/test_backend_e2e_regression.py -v

# Debug mode with detailed validation steps
uv run pytest tests/test_backend_e2e_regression.py -v -s --log-cli-level=INFO

# What to expect:
# - Test duration: ~10-11 seconds
# - Clear ✅/❌ status indicators for each validation step
# - Detailed failure messages if anything breaks
# - Works with or without live trade data from Kalshi
```

### Understanding test output:
- **✅ PASSED**: Validation step completed successfully
- **❌ FAILED**: Validation step failed (investigate immediately)
- **⚠️ WARNING**: Non-critical issue detected (monitor)
- **ℹ️ INFO**: Status information (normal)
- **📊 STATS**: Final test statistics

### When to run:
- **Before deployment** (mandatory)
- **After backend changes** (highly recommended)
- **When debugging backend issues** (use debug mode)
- **As part of CI pipeline** (automated)

This test serves as the definitive validation that the entire backend is functional and safe to deploy.

## Frontend E2E Regression Test

**Critical test that MUST pass before any deployment or major changes.**

### What it validates:
- ✅ **Backend Connection**: WebSocket connection established with "Live" status
- ✅ **Real Data Flow**: Live trade data flows from Kalshi → Backend → Frontend
- ✅ **Analytics Populated**: Summary statistics show non-zero values (volume, trades)
- ✅ **Market Grid Active**: At least one market displayed (critical failure if empty)
- ✅ **Chart Rendering**: Time-series chart renders with data points
- ✅ **Interactive Features**: Hour/Day mode toggle functions correctly
- ✅ **Real-time Updates**: Data changes over time proving live stream works
- ✅ **Component Stability**: All UI components render and remain functional

### Running the test:
```bash
# Prerequisites: Backend MUST be running on port 8000
# From project root directory
cd backend && uv run uvicorn kalshiflow.app:app --reload --port 8000

# Run the golden frontend test (in separate terminal)
# From project root directory  
cd frontend && npm run test:frontend-regression

# Note: Ensure you're in the kalshiflow root directory before running these commands

# What to expect:
# - Test duration: ~15-20 seconds
# - 5 screenshots captured in test-results/screenshots/
# - Clear ✅/❌ status indicators for each validation step
# - IMMEDIATE FAILURE if backend not running or no data flowing
# - Visual proof via screenshots of working system

```

### Understanding test output:
- **✅ WebSocket connected**: Backend is running and accessible
- **✅ Analytics active**: Real data flowing (Volume: $XXXk, Trades: XXX)
- **✅ Market grid populated**: X active markets displayed
- **✅ Chart rendering**: X data points visible
- **✅ Real-time updates**: Data increased over test duration
- **❌ CRITICAL FAILURES**: Backend not running, no data, or component failures

### Screenshots captured:
1. `01_initial_load.png` - Application startup state
2. `02_connection_established.png` - WebSocket "Live" connection confirmed
3. `03_data_populated.png` - Full view with analytics, markets, charts populated
4. `04_interactive_features.png` - After testing Hour/Day toggle functionality
5. `05_final_state.png` - Final state showing real-time data updates

### When to run:
- **Before deployment** (mandatory)
- **After frontend changes** (highly recommended)
- **When debugging frontend issues** (screenshots help diagnosis)
- **As part of CI pipeline** (automated validation)

### Critical failure conditions:
- Backend not running (WebSocket shows "Disconnected")
- No data flowing (Analytics shows $0 volume)
- Empty market grid (No markets displayed)
- Components not rendering (UI elements missing)
- No real-time updates (Data unchanged over test duration)

This test serves as the definitive validation that the entire frontend is functional and the E2E system works with live data.

## Railway Deployment

The application is deployed on Railway.app for production hosting with automated deployment pipeline.

### Deployment Prerequisites
1. **Railway CLI**: Ensure `railway` CLI is installed and authenticated
2. **Validation**: Run pre-deployment validation script
3. **Tests**: Both E2E regression tests must pass

### Automated Deployment Setup

#### Configuration Files
- **`railway.toml`**: Railway service configuration with health checks
- **`nixpacks.toml`**: Optimized build process configuration
- **`scripts/deploy-setup.sh`**: One-time Railway setup script
- **`scripts/pre-deploy-validation.sh`**: Pre-deployment validation

#### Setup Process
```bash
# 1. One-time Railway setup
./scripts/deploy-setup.sh

# 2. Enable auto-deployment on main branch
railway settings --auto-deploy=main

# 3. Configure missing environment variables
railway variables set PYTHONPATH="/app/backend/src"
railway variables set UVICORN_HOST="0.0.0.0"
railway variables set UVICORN_PORT="$PORT"
railway variables set NODE_ENV="production"
```

### Deployment Workflow

#### Deployment Script (Recommended)
```bash
# IMPORTANT: Only deploy when explicitly requested by the user
# This script handles all validation and deployment steps
./deploy.sh

# What the script does:
# 1. Validates git status (main branch, clean working directory)
# 2. Runs backend E2E regression tests
# 3. Runs frontend E2E regression tests  
# 4. Deploys backend to kalshi-flowboard-backend service
# 5. Deploys frontend to kalshi-flowboard service
# 6. Verifies deployment success
```

#### Manual Deployment (Alternative)
```bash
# 1. Run pre-deployment validation
./scripts/pre-deploy-validation.sh

# 2. Deploy backend and frontend separately
cd backend && railway up --service kalshi-flowboard-backend
cd frontend && railway up --service kalshi-flowboard
```

### Railway Configuration
- **Build**: Optimized via nixpacks.toml
- **Health Check**: `/health` endpoint with 30s timeout
- **Restart Policy**: On failure with max 3 retries
- **Environment Variables**: Managed via Railway dashboard

### Validation Checklist
- ✅ Backend E2E regression test passes
- ✅ Frontend builds successfully  
- ✅ Railway configuration files present
- ✅ Environment variables configured
- ✅ Health endpoints responding

### Production Monitoring
- **Logs**: Available via Railway dashboard
- **Metrics**: CPU, memory, and request metrics tracked
- **Health**: Continuous monitoring with automatic restarts
- **WebSocket**: Connection stability monitoring

- use the planning agent for all planning
- use the fullstack websocket agent for all implementation/coding
- use the deployment agent for Railway.app deployments and production infrastructure
- use the strategy-researcher agent to find novel hypothesis ideas from external sources (academic papers, sports betting, crypto/DeFi)
- use the quant agent with "LSD mode" for rapid lateral hypothesis exploration (user says "LSD mode" to enable)
- IMPORTANT: Only deploy to production when explicitly requested by the user. Never deploy autonomously.

## Multi-Session Coordination

When multiple Claude Code sessions work on the same codebase, use the session tracking system to avoid conflicts and enable visibility.

### Session Registration (REQUIRED)

**At the start of every conversation**, register your session:

```bash
# Generate a session ID and create your session file
SESSION_ID=$(uuidgen | cut -c1-8 | tr '[:upper:]' '[:lower:]')
```

Then create `.claude-sessions/active/{SESSION_ID}.json`:
```json
{
  "session_id": "{SESSION_ID}",
  "started_at": "2025-12-30T10:15:00Z",
  "last_updated": "2025-12-30T10:15:00Z",
  "branch": "current-git-branch",
  "status": "active",
  "current_task": "Initial task description",
  "files_touched": [],
  "sub_agents": []
}
```

### Updating Session Status

Update your session file when:
1. **Switching tasks** - Update `current_task` and `last_updated`
2. **Touching files** - Add to `files_touched` array
3. **Spawning sub-agents** - Add to `sub_agents` array

### Sub-Agent Tracking

When spawning a Task agent, add it to your session's `sub_agents`:
```json
{
  "sub_agents": [
    {
      "type": "fullstack-websocket-engineer",
      "task": "Implementing WebSocket reconnection",
      "started_at": "2025-12-30T10:40:00Z",
      "status": "running"
    }
  ]
}
```

Update `status` to `"completed"` when the agent finishes.

### Viewing Other Sessions

Before making major changes, check what other sessions are doing:
```bash
./scripts/claude-sessions.sh list
```

Or read files directly in `.claude-sessions/active/`.

### Session Lifecycle

- **Active**: Session is working (update `last_updated` regularly)
- **Idle**: Session is waiting for user input
- **Stale**: Session file >4 hours old (auto-cleaned)

Clean up stale sessions:
```bash
./scripts/claude-sessions.sh clean
```

## Strategy Research System

### Strategy Researcher Agent
Use this agent to discover novel trading strategy hypotheses from external sources:
- **Academic**: SSRN, arXiv quant-finance, prediction market papers
- **Sports Betting**: CLV, steam moves, sharp vs square patterns
- **Crypto/DeFi**: DEX microstructure, MEV research, Polymarket analysis
- **Market Microstructure**: Order flow toxicity, informed trading detection

Output: Hypothesis briefs in `research/hypotheses/incoming/`

### Quant LSD Mode (Lateral Strategy Discovery)
When the user says "LSD mode", the quant enters rapid exploration mode:
- **Speed over rigor**: Quick 10-min edge checks, skip full bucket analysis
- **Lower threshold**: Flag anything with raw edge > 5%
- **Absurdity encouraged**: Moon phases, fibonacci trade counts, 4+ signal mega-stacks
- **Volume over quality**: Test 10 ideas to find 1 good one

User says "normal mode" to return to rigorous bucket-matched validation.

See `research/LSD_MODE_INSTRUCTIONS.md` for full details.

## Managing RL Session Data

### Checking Orderbook Sessions
Use `@backend/src/kalshiflow_rl/scripts/fetch_session_data.py` to monitor and analyze session data:

```bash
# List all available sessions
uv run python src/kalshiflow_rl/scripts/fetch_session_data.py --list

# Load and analyze specific session
uv run python src/kalshiflow_rl/scripts/fetch_session_data.py --analyze 9

# Analyze the most recent session
uv run python src/kalshiflow_rl/scripts/fetch_session_data.py --analyze

# Create market-specific view
uv run python src/kalshiflow_rl/scripts/fetch_session_data.py --view 9 --market TICKER
```

### Cleaning Up Empty/Test Sessions
Use `@backend/src/kalshiflow_rl/scripts/cleanup_sessions.py` to identify and remove problematic sessions:

```bash
# Check database statistics
uv run python src/kalshiflow_rl/scripts/cleanup_sessions.py --stats

# Generate cleanup report (shows what can be deleted)
uv run python src/kalshiflow_rl/scripts/cleanup_sessions.py --report

# List empty sessions (0 snapshots and 0 deltas)
uv run python src/kalshiflow_rl/scripts/cleanup_sessions.py --list-empty

# List test sessions (<5 min, ≤5 markets)
uv run python src/kalshiflow_rl/scripts/cleanup_sessions.py --list-test

# Delete specific sessions (with confirmation)
uv run python src/kalshiflow_rl/scripts/cleanup_sessions.py --delete 2,3,8,18-22

# Delete all empty sessions
uv run python src/kalshiflow_rl/scripts/cleanup_sessions.py --delete-empty

# Delete all test sessions
uv run python src/kalshiflow_rl/scripts/cleanup_sessions.py --delete-test
```

### Session Data Quality Indicators
- **Meaningful sessions**: Have snapshots (>0) and deltas (>0)
- **Empty sessions**: 0 snapshots AND 0 deltas (safe to delete)
- **Test sessions**: <5 minutes duration, ≤5 markets (usually safe to delete)
- **Active stuck sessions**: Status='active' but no data for >24 hours (investigate before deleting)

### Best Practices
1. Run `--report` first to understand what will be deleted
2. Check `--stats` after cleanup to verify database state
3. Keep deletion logs for audit trail (automatically saved)
4. Preserve sessions with any meaningful data (snapshots or deltas > 0)
5. Document cleanup actions in `training/reports/` directory

## Managing RL Trained Models

### Current Model Configuration
The active RL model is tracked in `@backend/src/kalshiflow_rl/BEST_MODEL/CURRENT_MODEL.json`:
- **Current model**: `session32_final.zip` (21-action space with 5 contract sizes)
- **Model location**: `@backend/src/kalshiflow_rl/BEST_MODEL/session32_final.zip`
- **Action space**: 21 actions supporting position sizes of [5, 10, 20, 50, 100] contracts
- **Status**: Production-ready with centralized organization
- **Backup model**: `session9_ppo_20251211_221054` (historical reference)

### Cleaning Up Trained Models
Use `@backend/src/kalshiflow_rl/scripts/cleanup_trained_models.py` to manage the trained models directory:

```bash
# Preview what will be deleted (dry run mode - default)
uv run python src/kalshiflow_rl/scripts/cleanup_trained_models.py

# Actually delete old models (keeps only current + historical reference)
uv run python src/kalshiflow_rl/scripts/cleanup_trained_models.py --execute

# Quiet mode (minimal output)
uv run python src/kalshiflow_rl/scripts/cleanup_trained_models.py --execute --quiet
```

### Model Cleanup Features
- **Automatic detection**: Reads BEST_MODEL/CURRENT_MODEL.json to identify models to keep
- **Space recovery**: Removes failed training runs and experiments (typically 400+ MB)
- **Safety**: Always preserves current production model and historical reference
- **Centralized organization**: Works with new BEST_MODEL directory structure
- **Audit trail**: Saves deletion log to `src/kalshiflow_rl/logs/model_cleanup_YYYYMMDD_HHMMSS.json`
- **Dry run default**: Won't delete anything unless `--execute` flag is used

### When to Run Cleanup
- After updating BEST_MODEL/CURRENT_MODEL.json to a new model
- When trained_models directory exceeds 1GB
- Before archiving or backing up the project
- After extensive hyperparameter tuning sessions