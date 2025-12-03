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

# Test Kalshi client standalone
uv run backend/scripts/test_kalshi_client.py
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
- use the planning agent for all planning
- use the fullstack websocket agent for all implementation/coding