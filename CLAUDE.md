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
  - `KALSHI_API_KEY` - The API key ID
  - `KALSHI_PRIVATE_KEY_PATH` - Path to RSA private key file
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

Required `.env` variables:
```
# Kalshi API Configuration
KALSHI_API_KEY=<your_api_key>
KALSHI_PRIVATE_KEY_PATH=<path_to_rsa_key>
KALSHI_WS_URL=wss://api.elections.kalshi.com/trade-api/ws/v2

# Application Settings
WINDOW_MINUTES=10
HOT_MARKETS_LIMIT=20
RECENT_TRADES_LIMIT=200

# PostgreSQL Database Configuration (via Supabase)
DATABASE_URL=<postgresql_connection_string>
DATABASE_URL_POOLED=<postgresql_pooled_connection_string>

# Server Configuration
BACKEND_PORT=8000
FRONTEND_PORT=5173
```
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

The application is deployed on Railway.app for production hosting.

### Deployment Prerequisites
1. **Railway CLI**: Ensure `railway` CLI is installed and authenticated
2. **Backend Tests**: Run `uv run pytest tests/test_backend_e2e_regression.py -v` (must pass)
3. **Frontend Tests**: Run `npm run test:frontend-regression` (must pass)

### Deployment Process
```bash
# Deploy backend to Railway
railway login
railway up

# Environment variables are managed via Railway dashboard:
# - KALSHI_API_KEY
# - KALSHI_PRIVATE_KEY_PATH
# - DATABASE_URL (Supabase PostgreSQL)
# - DATABASE_URL_POOLED
```

### Railway Configuration
- **Build Command**: `cd backend && pip install uv && uv sync`
- **Start Command**: `cd backend && uv run uvicorn kalshiflow.app:app --host 0.0.0.0 --port $PORT`
- **Health Check**: `/health` endpoint
- **Auto-deploy**: Enabled on `main` branch

### Production Monitoring
- **Logs**: Available via Railway dashboard
- **Metrics**: CPU, memory, and request metrics tracked
- **Health**: Continuous health monitoring on `/health`

- use the planning agent for all planning
- use the fullstack websocket agent for all implementation/coding
- use the deployment agent for Railway.app deployments and production infrastructure