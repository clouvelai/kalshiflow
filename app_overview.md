APP_NAME: Kalshi Flowboard

ONE_LINER:
A tiny web app that listens to Kalshi’s public trades WebSocket and shows a real-time tape + heatmap of which markets are “hot” and which way the flow is leaning, using only the incoming trade events.

CORE CONSTRAINT:
Do not call any Kalshi REST endpoints in the first version. All data must originate from the WebSocket public trades stream.

TECH STACK (LOCKED IN):
- Backend: Python 3.x + Starlette (ASGI)
- Frontend: React + Vite + Tailwind CSS
- Data: In-memory aggregates for fast UI + SQLite for durable per-trade storage

--------------------------------
1. SCOPE & NON-GOALS
--------------------------------

IN-SCOPE (MVP):
- Single-page web UI.
- Backend service that:
  - Connects to Kalshi WebSocket API.
  - Subscribes to the "public trades" channel.
  - Parses incoming trade messages into a normalized internal model.
  - Stores trades in:
    - SQLite (append-only history).
    - In-memory structures for hot, recent aggregates.
  - Streams processed trade updates to the frontend in real time via WebSocket.
- Real-time visualizations:
  - Global trade tape (chronological list of recent trades).
  - "Hot markets" panel:
    - Top N tickers in the last 5–15 minutes ranked by volume (sum of "count").
    - For each ticker: last yes probability, recent volume, net taker flow.
  - Per-ticker detail view: small trade history + price sparkline for last X minutes.

OUT-OF-SCOPE (MVP):
- Authentication, user accounts, or preferences.
- REST metadata calls (titles, categories, series names).
- Order book, positions, or user fills.
- Historical backfill from before process start.
- Advanced filters/search (by category, event date, etc.).
- Polished mobile UX (basic responsiveness is enough).

--------------------------------
2. DATA MODEL
--------------------------------

Use SQLite for durable raw trade history + in-memory structures for aggregates.

2.1 SQLite schema (raw trades)

Table: trades
- id               INTEGER PRIMARY KEY AUTOINCREMENT
- market_ticker    TEXT NOT NULL
- yes_price        INTEGER NOT NULL         -- cents, e.g. 36
- no_price         INTEGER NOT NULL
- yes_price_dollars TEXT NOT NULL          -- string from payload
- no_price_dollars  TEXT NOT NULL
- count            INTEGER NOT NULL        -- contracts
- taker_side       TEXT NOT NULL           -- "yes" or "no"
- ts               INTEGER NOT NULL        -- unix seconds from Kalshi
- received_at      INTEGER NOT NULL        -- unix seconds when backend received it

Indexes:
- INDEX trades_ts_idx ON trades(ts);
- INDEX trades_ticker_ts_idx ON trades(market_ticker, ts);

2.2 In-memory aggregates (per ticker)

For each `market_ticker`, maintain sliding window aggregates for the last W minutes (default W = 10):

TickerState:
- market_ticker: string
- last_yes_price: int
- last_ts: int
- volume_window: int           -- Σ count in window
- yes_volume_window: int       -- Σ count where taker_side = "yes"
- no_volume_window: int        -- Σ count where taker_side = "no"
- net_yes_volume_window: int   -- yes_volume_window - no_volume_window
- trade_count_window: int
- price_points: list[(ts:int, yes_price:int)]  -- bounded size for sparkline (e.g. last 100–300 points)

Global state:
- recent_trades: deque of last N trades globally (e.g. N = 200–500).
- ticker_states: dict[str, TickerState]

Window logic:
- Define W (window_minutes, e.g. 10).
- When a new trade arrives:
  - Insert into SQLite immediately.
  - Update in-memory TickerState for that ticker.
  - Append to global recent_trades.
- Periodically (e.g. every 10–30 seconds):
  - Prune old trades out of each TickerState based on ts < now - W.
  - Prune global recent_trades based on max length.

--------------------------------
3. BACKEND ARCHITECTURE (STARLETTE)
--------------------------------

3.1 Overall structure

- ASGI app: Starlette.
- Components:
  - Kalshi WebSocket client task.
  - Trade processor + aggregator.
  - WebSocket endpoint for frontend realtime stream.
  - Optional HTTP REST endpoints for initial snapshot / debugging.
- Use Python asyncio and Starlette’s background tasks / startup events.

3.2 Kalshi WebSocket client

Responsibilities:
- Connect to Kalshi trade WebSocket endpoint:
  - (Use URL from Kalshi docs, e.g. `wss://api.elections.kalshi.com/trade-api/ws/v2`; agent should verify exact URL & auth scheme.)
- Handle authentication headers / signature (from config).
- Subscribe to public trades channel.
- Read incoming messages in an async loop.
- On message:
  - Parse JSON.
  - If `type == "trade"`, send trade payload to the trade processor.

Implementation notes:
- Run as a background task started on app startup.
- Implement reconnection with exponential backoff on errors/closure.
- Optionally send heartbeat/ping if Kalshi requires.

3.3 Trade processor and storage

Responsibilities:
- Normalize the incoming JSON trade payload:
  Input sample:
    {
      "type": "trade",
      "sid": 11,
      "msg": {
        "market_ticker": "HIGHNY-22DEC23-B53.5",
        "yes_price": 36,
        "yes_price_dollars": "0.360",
        "no_price": 64,
        "no_price_dollars": "0.640",
        "count": 136,
        "taker_side": "no",
        "ts": 1669149841
      }
    }

- Convert to internal Trade model (Python dataclass or Pydantic-like model).
- Write one row into SQLite `trades` table (non-blocking via async or threadpool).
- Update in-memory:
  - Global recent_trades deque.
  - Per-ticker TickerState aggregates.
- Emit an update event to WebSocket broadcaster.

3.4 Realtime broadcaster (frontend WebSocket)

Endpoint: `GET /ws/stream`

Behavior:
- On client connect:
  - Send a `snapshot` message with:
    - recent_trades (last N globally).
    - hot_markets (top N TickerState sorted by volume_window).
- Then subscribe the connection to incremental updates.
- On each new processed trade, broadcast a compact message to all connected clients.

Message protocol (JSON):

1) Snapshot:
{
  "type": "snapshot",
  "data": {
    "recent_trades": [
      {
        "market_ticker": "...",
        "yes_price_dollars": "0.360",
        "no_price_dollars": "0.640",
        "count": 136,
        "taker_side": "no",
        "ts": 1669149841
      },
      ...
    ],
    "hot_markets": [
      {
        "market_ticker": "HIGHNY-22DEC23-B53.5",
        "last_yes_price": 36,
        "last_ts": 1669149841,
        "volume_window": 1234,
        "yes_volume_window": 800,
        "no_volume_window": 434,
        "net_yes_volume_window": 366,
        "trade_count_window": 54,
        "price_points": [
          [1669149800, 32],
          [1669149820, 35],
          [1669149841, 36]
        ]
      },
      ...
    ]
  }
}

2) Incremental trade update:
{
  "type": "trade",
  "data": {
    "trade": {
      "market_ticker": "...",
      "yes_price_dollars": "0.360",
      "no_price_dollars": "0.640",
      "count": 136,
      "taker_side": "no",
      "ts": 1669149841
    },
    "ticker_state": {
      "market_ticker": "...",
      "last_yes_price": 36,
      "last_ts": 1669149841,
      "volume_window": 1234,
      "yes_volume_window": 800,
      "no_volume_window": 434,
      "net_yes_volume_window": 366,
      "trade_count_window": 54
      // price_points omitted here, or include minimal slice as needed
    }
  }
}

Implementation:
- Maintain a set of active WebSocket connections.
- On each new trade, broadcast the `trade` message to all.
- Consider throttling / batch updates if trade rate is very high (MVP can send per-trade updates).

3.5 Optional HTTP endpoints (for initial load / debugging)

- `GET /api/markets/hot?window_minutes=10&limit=20`
  - Returns hot_markets list (same shape as snapshot hot_markets).
- `GET /api/trades/recent?limit=200`
  - Returns latest trades globally.
- `GET /api/markets/{ticker}/trades?window_minutes=60&limit=500`
  - Returns trades for a specific ticker (from SQLite or memory).

These can be used by the frontend for initial data fetch if desired, or rely solely on the WebSocket snapshot.

--------------------------------
4. FRONTEND UX (REACT + VITE + TAILWIND)
--------------------------------

Tools:
- React (via Vite).
- Tailwind CSS for styling.
- Simple charting:
  - Either a very lightweight chart library
  - Or hand-rolled SVG for tiny sparklines (MVP can start with minimal SVG).

Page layout: single page, two-column layout on desktop, stacked on mobile.

4.1 Header
- App title: "Kalshi Flowboard".
- Small subtitle: "Live view of Kalshi public trades. No positions, just the public tape."

4.2 Left Panel: Live Trade Tape

Component: `<TradeTape />`

Behavior:
- Displays last N trades globally (ordered newest → oldest).
- Each row shows:
  - Time: formatted from ts (HH:MM:SS).
  - Ticker: market_ticker (truncated if too long).
  - Direction: "YES" or "NO" depending on taker_side. Color:
    - YES: green.
    - NO: red.
  - Price: yes_price_dollars (as "0.360" or "36¢", stylistic choice).
  - Size: count.
- New trades appear at the top with a short highlight (e.g. Tailwind transition + background flash).
- Clicking a row selects that ticker and notifies parent.

4.3 Right Panel: Hot Markets

Component: `<HotMarkets />`

Behavior:
- Shows top N tickers (e.g. 8–12) sorted by volume_window (last 10 min).
- Each TickerCard shows:
  - market_ticker.
  - Last price: convert `last_yes_price` to probability (yes_price/100):
    - e.g., "36% YES".
  - Volume: "Vol (10m): 2.1k" (format volume_window).
  - Net flow: "Flow (10m): +800 YES" or "-300 YES" based on net_yes_volume_window.
  - Mini sparkline of yes_price from `price_points`.
- Clicking a card opens detail drawer for that ticker.

4.4 Detail Drawer / Panel

Component: `<TickerDetailDrawer />`

Behavior:
- Opens when a ticker is selected (from tape or hot markets).
- Fetch data either from:
  - Current in-memory state from WebSocket snapshot + updates, OR
  - HTTP endpoint `/api/markets/{ticker}/trades`.
- Shows:
  - Title: market_ticker.
  - Key stats:
    - "Last price: 64% YES".
    - "Trades (10m): X".
    - "Volume (10m): Y".
    - "Net YES flow (10m): +Z".
  - Larger line chart for yes_price vs time for last 30–60 minutes.
  - Table of last ~20 trades for that ticker.

4.5 Data flow (frontend)

- On mount:
  - Connect to `ws://<backend>/ws/stream`.
  - Wait for initial `snapshot` message.
  - Store:
    - recentTrades in state.
    - hotMarkets in state.
- On `trade` messages:
  - Update recentTrades (prepend, trim).
  - Update hotMarkets (replace or merge ticker_state; resort).
- Selection:
  - User clicks a ticker → set selectedTicker.
  - Optionally fetch additional history from HTTP endpoint.

--------------------------------
5. NON-FUNCTIONAL REQUIREMENTS
--------------------------------

Performance:
- Should handle hundreds of trades per second in backend without crashing.
- Frontend:
  - Limit number of DOM rows (e.g. 200 latest trades).
  - Limit number of tickers in hotMarkets (e.g. 12).
  - Limit number of points in price_points per ticker (e.g. 100–300).

Resilience:
- Kalshi WebSocket reconnection with exponential backoff.
- If backend restarts:
  - In-memory state resets.
  - SQLite still has historical trades, but MVP does not need automatic replay.
- If WebSocket to frontend drops:
  - Frontend should reconnect after a delay and request a new snapshot.

Configuration:
- Environment variables:
  - `KALSHI_API_KEY`
  - `KALSHI_API_SECRET` (if needed for signing)
  - `KALSHI_WS_URL`
  - `WINDOW_MINUTES` (default 10)
  - `HOT_MARKETS_LIMIT` (default 12)
  - `RECENT_TRADES_LIMIT` (default 200)
  - `SQLITE_DB_PATH` (e.g. `./kalshi_trades.db`)

--------------------------------
6. STRETCH GOALS (LATER, NOT MVP)
--------------------------------

- Simple filters: show only tickers matching a substring.
- “Pin” selected tickers to top of hot markets.
- Display cumulative PnL-style “drift” chart for a ticker over the session.
- Use Kalshi REST API to enrich market_ticker with human-readable event names and categories.
- Persist aggregate stats in SQLite and allow multi-hour charts.

--------------------------------
7. Milestones and Validation
--------------------------------
Phase 1
1. Initialize the project. (a) Create skeleton project structure (b) install dependencies, (c) user sets required kalshi auth in .env --> Success criteriea is were ready to develop. There should be an init.sh that installs dependencies and runs the app that future coding agents can run.
2. Implement Kalshi client to listen for public trades (a) auth -> example of working rsa key based here https://github.com/clouvelai/prophete/blob/main/backend/app/core/auth.py (b) async subscribe to public trades https://docs.kalshi.com/websockets/public-trades -> success criteria for this milestone is we can standalone run the kalshi client, auth successfully, subscribe to public trades and see incoming messages in stdout. We should have unit test coverage for the kalshi client. 
3. Starlette + kalshi client public trades integration. We can run our backend app, it successfully auth and subscribes and streams trades coming in over websocket. We'd now be able to implement the frontend feed if we wanted to. 
4. Implement frontend layout + feed. First we should get the frontend skeleton in place, then implement the feed to show incoming events via websocket.
Phase 2
--- Once we get here we'll plan out storage, aggregation trends etc. 

END_OF_SPEC