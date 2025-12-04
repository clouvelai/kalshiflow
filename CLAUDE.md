# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kalshi Flowboard - A real-time web application that displays Kalshi public trades via WebSocket, showing a live tape and heatmap of "hot" markets based on trade volume and flow direction.

## Architecture

### Tech Stack
- **Backend**: Python 3.x + Starlette (ASGI) with uv for dependency management
- **Frontend**: React + Vite + Tailwind CSS with npm for dependency management
- **Data**: SQLite for durable trade history + in-memory aggregates for performance
- **Authentication**: RSA private key file-based auth for Kalshi API

### Core Constraint
Do not call any Kalshi REST endpoints in the MVP. All data must originate from the WebSocket public trades stream.

### Data Flow
1. Kalshi WebSocket → Backend (auth with RSA signature)
2. Backend processes trades → SQLite (durable) + In-memory (aggregates)
3. Backend broadcasts → Frontend WebSocket
4. Frontend displays → Trade tape + Hot markets + Ticker details

## Common Commands

### Development Setup
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
    database.py     # SQLite setup
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
KALSHI_API_KEY=<your_api_key>
KALSHI_PRIVATE_KEY_PATH=<path_to_rsa_key>
KALSHI_WS_URL=wss://api.elections.kalshi.com/trade-api/ws/v2
WINDOW_MINUTES=10
HOT_MARKETS_LIMIT=12
RECENT_TRADES_LIMIT=200
SQLITE_DB_PATH=./kalshi_trades.db
```
## Backend E2E Regression Test

**Critical test that MUST pass before any deployment or major changes.**

### What it validates:
- ✅ **Backend Startup**: Complete application starts successfully
- ✅ **Service Integration**: All services (trade processor, aggregator, websocket manager) initialize
- ✅ **Kalshi Connection**: WebSocket client connects to Kalshi public trades stream
- ✅ **Database Functionality**: SQLite database creation and accessibility
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

**Critical test that MUST pass before any frontend deployment or major changes.**

### What it validates:
- ✅ **Application Startup**: Complete frontend loads within 15 seconds
- ✅ **WebSocket Connection**: Frontend connects to backend WebSocket successfully
- ✅ **Live Data Flow**: Real-time data populates and updates (trades, analytics, markets)
- ✅ **Component Functionality**: All interactive features work (toggles, drawer, clicks)
- ✅ **Responsive Design**: Layout functions across desktop, tablet, mobile viewports
- ✅ **Real-time Updates**: Confirms live data values increment over time
- ✅ **Visual Validation**: 13 comprehensive screenshots document entire user journey

### Running the test:
```bash
# Prerequisites: Backend must be running on port 8000
cd backend && uv run uvicorn kalshiflow.app:app --reload --port 8000

# Standard run (must pass before deployment)
cd frontend && npm run test:frontend-regression

# Debug mode with interactive UI
npm run test:e2e-ui

# What to expect:
# - Test duration: ~31-33 seconds
# - Real-time data validation against live Kalshi WebSocket
# - 6 validation phases with clear ✅/❌ status indicators
# - 13 screenshots captured in test-results/
# - Works with live market data
```

### Test phases:
1. **Application Startup**: Layout structure, main sections visibility
2. **Connection & Data**: WebSocket connection, live data population 
3. **Component Functions**: Analytics toggle, market selection, drawer interactions
4. **Responsive Design**: Desktop (4-col), tablet (2-col), mobile (1-col) layouts
5. **Real-time Validation**: Capture initial values, wait, compare for live updates
6. **Quality Assurance**: Console errors, live indicators, final state verification

### Understanding test output:
- **✅ PASSED**: Validation step completed successfully
- **❌ FAILED**: Validation step failed (investigate immediately)
- **📊 Data Updates**: Shows initial vs updated values to confirm real-time flow
- **📸 Screenshots**: All 13 validation screenshots saved to test-results/

### When to run:
- **Before deployment** (mandatory)
- **After frontend changes** (highly recommended) 
- **When debugging frontend issues** (use interactive UI mode)
- **As part of CI pipeline** (automated)

This test serves as the definitive validation that the entire frontend works correctly against live data.

- use the planning agent for all planning
- use the fullstack websocket agent for all implementation/coding