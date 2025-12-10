APP_NAME: Kalshi Flow RL Trading Subsystem

ONE_LINER:
A reinforcement learning agent that trains on historical Kalshi orderbook data using Gymnasium + Stable Baselines3, and executes paper trades in real-time with full visualization and monitoring capabilities.

CORE CONSTRAINT:
Must operate as a completely separate service (kalshiflow-rl-service) that does not interfere with the existing analytics backend. All RL functionality must be namespaced under /rl/* endpoints.

TECH STACK (LOCKED IN):
- Backend: Python 3.x + Starlette (ASGI) with async-first architecture
- Frontend: React + Vite + Tailwind CSS (existing Kalshi Flow frontend)
- Database: PostgreSQL via Supabase (shared with main app)
- ML Framework: Gymnasium + Stable Baselines3 (SB3)
- Data Source: Kalshi orderbook WebSocket API (for live data) + Historical DB (for training)

--------------------------------
1. SCOPE & NON-GOALS
--------------------------------

IN-SCOPE (MVP):
- Single-market orderbook ingestion via WebSocket with async queue
- Non-blocking orderbook storage in PostgreSQL
- Custom Gymnasium environment for historical data replay ONLY
- SB3 integration with basic algorithms (PPO/A2C)
- Actor loop (inference-only) with hot-reload protocol
- Paper trading client implementation
- WebSocket-based ML visualization tab
- Training exclusively from historical orderbook data
- Clear trainer/actor separation with different data sources
- Minimal model registry and episode logging

OUT-OF-SCOPE (MVP):
- Multi-market simultaneous trading
- Complex portfolio management
- Advanced RL algorithms beyond SB3 defaults
- Backtesting framework
- Risk management beyond position limits
- Market making strategies
- Cross-market arbitrage
- Live trading mode (paper only for MVP)
- Advanced performance metrics/dashboards
- Distributed training
- Custom replay buffers (SB3 handles this)
- Full observation storage in trading_actions

--------------------------------
2. DATA MODEL
--------------------------------

2.1 PostgreSQL Schema (Orderbook Data)

Table: orderbook_snapshots
- id                  BIGSERIAL PRIMARY KEY
- market_ticker       TEXT NOT NULL
- timestamp           BIGINT NOT NULL        -- unix milliseconds
- sequence_number     BIGINT NOT NULL        -- from Kalshi
- yes_bids            JSONB NOT NULL         -- [{price, quantity}, ...]
- yes_asks            JSONB NOT NULL         -- [{price, quantity}, ...]
- no_bids             JSONB NOT NULL         -- [{price, quantity}, ...]
- no_asks             JSONB NOT NULL         -- [{price, quantity}, ...]
- received_at         TIMESTAMP DEFAULT NOW()
- checksum            TEXT                   -- optional validation

Indexes:
- INDEX idx_snapshots_market_time ON orderbook_snapshots(market_ticker, timestamp);
- INDEX idx_snapshots_sequence ON orderbook_snapshots(market_ticker, sequence_number);

Table: orderbook_deltas
- id                  BIGSERIAL PRIMARY KEY
- market_ticker       TEXT NOT NULL
- timestamp           BIGINT NOT NULL
- sequence_number     BIGINT NOT NULL
- delta_type          TEXT NOT NULL          -- 'add', 'remove', 'update'
- side                TEXT NOT NULL          -- 'yes_bid', 'yes_ask', 'no_bid', 'no_ask'
- price               INTEGER NOT NULL       -- in cents
- quantity            INTEGER                -- null for removals
- received_at         TIMESTAMP DEFAULT NOW()

Indexes:
- INDEX idx_deltas_market_seq ON orderbook_deltas(market_ticker, sequence_number);
- INDEX idx_deltas_timestamp ON orderbook_deltas(market_ticker, timestamp);

Table: models
- id                  UUID PRIMARY KEY DEFAULT gen_random_uuid()
- name                TEXT NOT NULL          -- human-readable name (e.g., "PPO_TRUMP_v1")
- version             INTEGER NOT NULL       -- version number for tracking iterations
- architecture        TEXT NOT NULL          -- 'PPO', 'A2C', 'DQN', 'SAC', etc.
- market_ticker       TEXT NOT NULL          -- which market it's trained for
- parent_model_id     UUID REFERENCES models(id)  -- for fine-tuning lineage
- checkpoint_path     TEXT NOT NULL          -- filesystem/S3 path to model file
- hyperparameters     JSONB NOT NULL         -- learning_rate, batch_size, gamma, etc.
- training_config     JSONB NOT NULL         -- env config, observation space, reward version
- training_episodes   INTEGER DEFAULT 0      -- total episodes trained
- training_timesteps  BIGINT DEFAULT 0       -- total timesteps trained
- created_at          TIMESTAMP DEFAULT NOW()
- updated_at          TIMESTAMP DEFAULT NOW()
- performance_summary JSONB                  -- {avg_reward, win_rate, sharpe_ratio, etc.}
- status              TEXT NOT NULL          -- 'training', 'ready', 'deployed', 'archived'
- notes               TEXT                   -- human notes/description

Indexes:
- INDEX idx_models_market ON models(market_ticker);
- INDEX idx_models_status ON models(status);
- UNIQUE INDEX idx_models_name_version ON models(name, version);
- INDEX idx_models_created ON models(created_at DESC);

Table: trading_episodes
- id                  BIGSERIAL PRIMARY KEY
- episode_id          UUID DEFAULT gen_random_uuid()
- model_id            UUID REFERENCES models(id)  -- which model was used
- market_ticker       TEXT NOT NULL
- start_time          TIMESTAMP NOT NULL
- end_time            TIMESTAMP
- mode                TEXT NOT NULL          -- 'training', 'paper'; 'live' reserved for future (unused in MVP)
- total_reward        FLOAT
- total_steps         INTEGER
- final_position      JSONB                  -- final portfolio state
- metadata            JSONB                  -- episode-specific config/overrides

Indexes:
- INDEX idx_episodes_model ON trading_episodes(model_id);
- INDEX idx_episodes_mode ON trading_episodes(mode);
- INDEX idx_episodes_time ON trading_episodes(start_time DESC);

Table: trading_actions
- id                  BIGSERIAL PRIMARY KEY
- episode_id          UUID REFERENCES trading_episodes(episode_id)
- timestamp           BIGINT NOT NULL
- market_ticker       TEXT NOT NULL
- action              INTEGER NOT NULL       -- encoded action (0=hold, 1=buy_yes, 2=sell_yes, etc.)
- action_details      JSONB                  -- {side, price, quantity, etc.}
- state_hash          TEXT                   -- MVP: hash of observation (full state storage is FUTURE)
- reward              FLOAT
- executed            BOOLEAN DEFAULT FALSE
- execution_result    JSONB                  -- order_id, fill_price, etc.

2.2 In-Memory State Management

OrderbookState (per market):
- market_ticker: str
- last_sequence: int
- yes_bids: SortedDict[price, quantity]  -- Using sortedcontainers for O(log n) operations
- yes_asks: SortedDict[price, quantity]
- no_bids: SortedDict[price, quantity]
- no_asks: SortedDict[price, quantity]
- last_update: datetime
- spread_yes: float
- spread_no: float
- mid_price_yes: float
- mid_price_no: float
- total_volume: dict[side, int]

ActorState:
- current_episode: UUID
- position_state: dict[market, position]  -- Consistent naming with observation builder
- model: sb3.Model (inference mode only)
- model_version: int
- model_checkpoint_mtime: float  -- for hot-reload detection
- observation_buffer: deque(maxlen=N)
- action_history: deque(maxlen=M)
- performance_metrics: dict

OrderbookWriteQueue:
- queue: asyncio.Queue
- batch_size: int (configurable, default=100)
- sample_rate: int (keep 1 out of N deltas, default=1)
- flush_interval: float (seconds, default=1.0)
- pending_snapshots: list
- pending_deltas: list

ActionWriteQueue:
- queue: asyncio.Queue
- batch_size: int (configurable, default=50)
- flush_interval: float (seconds, default=2.0)
- pending_actions: list

SharedOrderbookState:
- market_ticker: str
- state: OrderbookState  # Single source of truth
- lock: asyncio.Lock  # For thread-safe updates
- subscribers: list  # Actor, UI, etc. read from this

--------------------------------
3. CRITICAL ARCHITECTURAL REQUIREMENTS
--------------------------------

**These requirements prevent silent failures and ensure system correctness:**

1. **Unified Observation Logic**
   - ALL observation-building logic MUST live in `observation_space.py`
   - Both KalshiTradingEnv and TradingActor MUST call the same function:
     `build_observation_from_orderbook(orderbook_state, position_state)`
   - This ensures identical feature distribution between training and inference

2. **Non-Blocking Database Writes**
   - WebSocket orderbook writes MUST use OrderbookWriteQueue
   - Actor action logs MUST use ActionWriteQueue
   - NO component may write directly to PostgreSQL in hot paths
   - Background tasks handle batch flushing asynchronously

3. **Mode Restrictions**
   - Only 'training' and 'paper' modes allowed for MVP
   - 'live' mode MUST return HTTP 400/501 error
   - Actor MUST reject any attempt to enter 'live' mode

4. **Environment Database Isolation**
   - KalshiTradingEnv MUST preload all data before training
   - NO database queries allowed during reset() or step()
   - All historical data loaded into memory or iterator at init
   - For MVP, preload time-bounded windows (e.g., N days) to manage memory usage

5. **Minimal Reward Logging**
   - Actor logs reward ONLY if trivial to compute (e.g., mark-to-market)
   - Complex reward shaping belongs in training env exclusively
   - Logged rewards NEVER used for training

6. **Pipeline Isolation**
   - Training and inference pipelines MUST NEVER interact
   - Actor NEVER pushes to SB3 buffers
   - Trainer NEVER consumes live actor data
   - Shared artifacts limited to: checkpoints, DB state, observation builder

7. **Single Orderbook State**
   - ONE SharedOrderbookState per market (maintained by WebSocket client)
   - All consumers (Actor, UI) read from this single source
   - NO duplicate state machines allowed

8. **Historical Data Format Consistency**
   - Historical replay MUST normalize to exact same structure as live OrderbookState
   - Both paths feed into same `build_observation_from_orderbook()` function
   - Prevents training/inference distribution mismatch

--------------------------------
4. BACKEND ARCHITECTURE
--------------------------------

4.1 Service Structure

kalshiflow-rl-service/
├── src/
│   ├── kalshiflow_rl/
│   │   ├── __init__.py
│   │   ├── app.py                    # Starlette app + startup
│   │   ├── config.py                 # Settings, env vars
│   │   │
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── orderbook_client.py   # Kalshi orderbook WebSocket
│   │   │   ├── normalizer.py         # Raw → normalized data
│   │   │   ├── storage.py            # Database persistence
│   │   │   └── replay.py             # Historical data replay
│   │   │
│   │   ├── environments/
│   │   │   ├── __init__.py
│   │   │   ├── kalshi_env.py         # Gymnasium environment
│   │   │   ├── observation_space.py  # State representation
│   │   │   ├── action_space.py       # Action encoding/decoding
│   │   │   └── reward.py             # Reward calculation
│   │   │
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── base_agent.py         # Agent interface
│   │   │   ├── sb3_agent.py          # SB3 wrapper
│   │   │   ├── dummy_agent.py        # Random/fixed policies
│   │   │   └── training.py           # Training loops
│   │   │
│   │   ├── trading/
│   │   │   ├── __init__.py
│   │   │   ├── actor.py              # Live trading loop
│   │   │   ├── paper_client.py       # Paper trading impl
│   │   │   ├── kalshi_client.py      # Real trading (future)
│   │   │   └── portfolio.py          # Position tracking
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py             # HTTP endpoints
│   │   │   ├── websocket.py          # WebSocket handlers
│   │   │   └── models.py             # Pydantic schemas
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── logging.py
│   │       ├── metrics.py
│   │       └── serialization.py
│   │
│   ├── scripts/
│   │   ├── train.py                  # Standalone training script
│   │   ├── evaluate.py               # Model evaluation
│   │   └── replay_episode.py         # Replay past episodes
│   │
│   └── tests/
│       ├── test_orderbook_client.py
│       ├── test_environment.py
│       ├── test_agent.py
│       └── test_actor.py

4.2 Orderbook WebSocket Client (Non-Blocking)

**CRITICAL**: This continuously collects orderbook data that becomes the training dataset.

Responsibilities:
- Connect to Kalshi orderbook WebSocket:
  - URL: wss://api.elections.kalshi.com/trade-api/ws/v2
  - Subscribe to orderbook channel for configured market
- Handle authentication (RSA signature, same as trade stream)
- Process messages WITHOUT blocking on DB writes:
  - Initial snapshot
  - Delta updates
  - Sequence number tracking
  - Checksum validation
- Queue messages for async database persistence (builds training dataset)
- Broadcast to actor loop and UI subscribers

Implementation:
```python
class OrderbookClient:
    def __init__(self):
        self.write_queue = OrderbookWriteQueue()
        
    async def connect(self):
        # WebSocket connection with auth
        # Start background write task
        asyncio.create_task(self.write_queue.flush_loop())
    
    async def process_snapshot(self, msg):
        # Update in-memory state immediately
        # Queue for DB write (non-blocking)
        await self.write_queue.enqueue_snapshot(msg)
        # Broadcast to subscribers
        await self.broadcast(msg)
    
    async def process_delta(self, msg):
        # Apply delta to in-memory state immediately
        # Queue for DB write with optional sampling
        if self.should_sample(msg):
            await self.write_queue.enqueue_delta(msg)
        # Always broadcast (regardless of sampling)
        await self.broadcast(msg)
```

4.3 Gymnasium Environment (Historical Data Only)

```python
class KalshiTradingEnv(gym.Env):
    """
    CRITICAL REQUIREMENTS:
    1. This environment ONLY replays historical data
    2. It does NOT connect to live WebSockets or use async
    3. All data must be preloaded - NO DB queries during step() or reset()
    4. Must use IDENTICAL observation builder as actor (from observation_space.py)
    """
    def __init__(self, config):
        # PRELOAD all historical data into memory or iterator
        # NO database access after this point
        # For MVP, preload a time-bounded window (e.g., N days) to avoid memory issues
        self.historical_data = self.preload_historical_data(
            market=config['market'],
            start_time=config['start_time'],
            end_time=config['end_time']
        )
        self.observation_space = spaces.Box(...)
        self.action_space = spaces.Discrete(5)  # hold, buy_yes, sell_yes, buy_no, sell_no
        self.current_step = 0
        
    def reset(self):
        # Reset to beginning of historical data
        # NO DB queries allowed here
        self.current_step = 0
        orderbook_state = self._reconstruct_orderbook_state()
        # CRITICAL: Use shared observation builder
        from kalshiflow_rl.environments.observation_space import build_observation_from_orderbook
        return build_observation_from_orderbook(orderbook_state, self.position_state)
    
    def step(self, action):
        # Advance through preloaded data
        # NO DB queries allowed here
        self.current_step += 1
        orderbook_state = self._reconstruct_orderbook_state()
        
        # Use shared observation builder (SAME as actor)
        from kalshiflow_rl.environments.observation_space import build_observation_from_orderbook
        obs = build_observation_from_orderbook(orderbook_state, self.position_state)
        
        # Calculate reward (complex shaping allowed in training)
        reward = self._calculate_reward(action)
        done = self.current_step >= len(self.historical_data)
        return obs, reward, done, False, {}
    
    def _reconstruct_orderbook_state(self):
        # Normalize historical data to EXACT same format as live OrderbookState
        # This ensures training/inference consistency
        pass
```

Observation Space (normalized features):
- Bid/ask spreads (yes/no)
- Volume at best prices
- Price levels (top 5 each side)
- Historical price movement (rolling window)
- Time features (time to expiry, etc.)
- Current position (if any)

Action Space:
- Discrete: 0=hold, 1=buy_yes, 2=sell_yes, 3=buy_no, 4=sell_no
- Future: Continuous (price, quantity)

4.4 Actor Loop (Inference Only)

```python
class TradingActor:
    """
    CRITICAL REQUIREMENTS:
    1. Actor is inference-only - NEVER trains or updates weights
    2. Uses IDENTICAL observation builder as training env
    3. All DB writes must be non-blocking via queue
    4. Reads from single shared OrderbookState (no duplicate state)
    5. Only supports 'paper' mode for MVP ('live' returns error)
    """
    def __init__(self, model_id, trading_client, config, orderbook_state, write_queue):
        self.model_id = model_id
        self.model = self.load_model(model_id)
        self.client = trading_client  # Paper only for MVP
        self.orderbook_state = orderbook_state  # Shared with WebSocket client
        self.write_queue = write_queue  # Shared or dedicated ActionWriteQueue
        self.state = ActorState()
        self.last_checkpoint_mtime = None
        
        # Validate mode
        if config.get('mode') == 'live':
            raise ValueError("Live trading mode forbidden in MVP - use 'paper' only")
        
    async def run(self):
        while True:
            # Check for model updates (hot-reload)
            await self.check_and_reload_model()
            
            # Get latest LIVE orderbook from SHARED state (no duplicate state machine)
            orderbook_snapshot = self.orderbook_state.get_snapshot()
            
            # CRITICAL: Use shared observation builder (SAME as training env)
            from kalshiflow_rl.environments.observation_space import build_observation_from_orderbook
            obs = build_observation_from_orderbook(orderbook_snapshot, self.state.position_state)
            
            # Get action from model (inference only)
            with torch.no_grad():  # Ensure no gradient computation
                action, _ = self.model.predict(obs, deterministic=True)
            
            # Execute via paper trading client
            result = await self.client.execute(action)
            
            # Calculate simple reward if trivial (e.g., mark-to-market)
            # Complex reward shaping belongs in training env only
            reward = self._calculate_simple_reward(result) if easy else None
            
            # Queue action for NON-BLOCKING database write
            await self.write_queue.enqueue_action({
                'episode_id': self.state.current_episode,
                'action': action,
                'state_hash': hashlib.md5(obs.tobytes()).hexdigest(),
                'reward': reward,  # Optional/minimal
                'result': result
            })
            
            # Broadcast to UI immediately (don't wait for DB)
            await self.broadcast_to_ui(action, obs)
            
            await asyncio.sleep(self.config.tick_interval)
    
    async def check_and_reload_model(self):
        """Hot-reload protocol: Check DB or filesystem for new checkpoint"""
        current_mtime = await self.get_checkpoint_mtime(self.model_id)
        if current_mtime != self.last_checkpoint_mtime:
            # Atomic model swap
            new_model = self.load_model(self.model_id)
            self.model = new_model
            self.last_checkpoint_mtime = current_mtime
            logger.info(f"Hot-reloaded model {self.model_id}")
```

4.5 Trainer/Actor Separation Contract

**CRITICAL DESIGN PRINCIPLE**: Complete isolation between training and inference pipelines.

```python
# Trainer (Historical Data Only)
class Trainer:
    """
    CRITICAL REQUIREMENTS:
    1. Runs SB3 training loops on HISTORICAL data only
    2. NEVER connects to live WebSocket
    3. NEVER consumes live actor data
    4. NEVER does inference on live data
    5. Uses shared observation builder from observation_space.py
    """
    def train(self, model_id, config):
        # Load historical environment with PRELOADED data
        env = KalshiTradingEnv(config)  # Historical data only, no DB access during training
        
        # Initialize or load SB3 model
        if config.get('parent_model_id'):
            model = PPO.load(parent_checkpoint_path)
        else:
            model = PPO('MlpPolicy', env, **hyperparams)
        
        # Train using SB3's internal replay buffer management
        model.learn(total_timesteps=config['timesteps'])
        
        # Save checkpoint (only shared artifact with Actor)
        checkpoint_path = f"models/{model_id}/checkpoint.zip"
        model.save(checkpoint_path)
        
        # Update database (only shared state with Actor)
        update_model_status(model_id, 'ready', checkpoint_path)

# Actor (Live Data Only)
class Actor:
    """
    CRITICAL REQUIREMENTS:
    1. Does inference on live WebSocket data
    2. NEVER trains or updates weights
    3. NEVER pushes data to SB3 buffers or training components
    4. Uses shared observation builder from observation_space.py
    5. All logging is non-blocking via queues
    """
    # See section 3.4 above
```

**Pipeline Isolation Requirements**:
- Training and inference pipelines MUST NEVER interact directly
- Actor MUST NEVER push data into any SB3 buffer or training component
- Trainer MUST NEVER consume live actor data
- They share ONLY:
  - Model checkpoints (read-only for Actor)
  - Database state (models table)
  - Orderbook data (historical snapshots only, not live stream)
  - Observation builder function (from observation_space.py)

**Data Flow Separation**:
- Data Collection: Live WebSocket → Async Queue → PostgreSQL (continuous, always running)
- Training: PostgreSQL historical (preloaded) → Gymnasium Env → SB3 → Checkpoint
- Inference: Live WebSocket → Shared OrderbookState → Actor → Paper Trading → Action Queue → DB

**Important Data Storage Distinction**:
- Orderbook data (orderbook_snapshots/deltas): Continuously collected, becomes training dataset
- Actor actions (trading_actions table): Logged for observability via queue, NOT for training
- Training replay buffer: Managed internally by SB3, NEVER exposed or stored
- Key point: We LOG everything for analysis, but pipelines remain isolated

**Hot-Reload Protocol**:
1. Trainer saves new checkpoint to `checkpoint_path`
2. Trainer updates DB: model version++, status='ready'
3. Actor polls for changes every N seconds
4. Actor detects new version or mtime change
5. Actor loads new checkpoint atomically
6. Actor continues with new model (no downtime)

4.6 API Endpoints

HTTP Routes (/rl/*):
- GET /rl/status - Service health and status
- GET /rl/models - List all models with filters
- GET /rl/models/{id} - Model details and performance
- POST /rl/models - Create new model entry
- PUT /rl/models/{id} - Update model status/notes
- GET /rl/episodes - List training/trading episodes
- GET /rl/episodes/{id} - Episode details
- POST /rl/training/start - Start training session (creates model, mode='training')
- POST /rl/training/stop - Stop training
- POST /rl/actor/start - Start paper trading actor with model_id (mode='paper' only)
- POST /rl/actor/stop - Stop actor
- POST /rl/actor/switch-model - Hot-swap to different model
- POST /rl/actor/set-mode - Set mode ('paper' only for MVP, 'live' returns 501)
- GET /rl/metrics - Performance metrics
- GET /rl/compare - Compare models performance

WebSocket Routes:
- /rl/ws/orderbook - Live orderbook stream
- /rl/ws/actions - Live action stream
- /rl/ws/training - Training progress stream

--------------------------------
5. FRONTEND INTEGRATION
--------------------------------

5.1 ML Tab Component Structure

```
frontend/src/components/ml/
├── MLDashboard.tsx          # Main container
├── OrderbookVisualizer.tsx  # Orderbook depth chart
├── ActionFeed.tsx           # Recent actions list
├── PerformanceMetrics.tsx  # P&L, win rate, etc.
├── ModelControl.tsx         # Start/stop/load model
└── TrainingProgress.tsx    # Training charts

```

5.2 WebSocket Protocol

Orderbook Update:
```json
{
  "type": "orderbook_update",
  "data": {
    "market_ticker": "TRUMPWIN-2024",
    "timestamp": 1669149841000,
    "yes_bids": [[65, 1000], [64, 500]],
    "yes_asks": [[66, 800], [67, 200]],
    "no_bids": [[34, 800], [33, 200]],
    "no_asks": [[35, 1000], [36, 500]],
    "spread_yes": 1,
    "spread_no": 1,
    "mid_yes": 65.5,
    "mid_no": 34.5
  }
}
```

Action Event:
```json
{
  "type": "action",
  "data": {
    "timestamp": 1669149841000,
    "market_ticker": "TRUMPWIN-2024",
    "action": "buy_yes",
    "details": {
      "side": "yes",
      "price": 66,
      "quantity": 100,
      "order_type": "limit"
    },
    "state": {
      "position": 100,
      "avg_price": 66,
      "unrealized_pnl": 0
    },
    "confidence": 0.85
  }
}
```

Training Update:
```json
{
  "type": "training_progress",
  "data": {
    "episode": 150,
    "total_episodes": 1000,
    "avg_reward": 12.5,
    "loss": 0.0023,
    "learning_rate": 0.0001,
    "timesteps_so_far": 15000
  }
}
```

--------------------------------
6. IMPLEMENTATION ROADMAP
--------------------------------

PHASE 1: Data Pipeline (Week 1)
Milestone 1: Non-blocking orderbook ingestion
- Implement orderbook_client.py with Kalshi auth
- Create async write queue with batching/sampling
- Create full database schema (all tables, MVP fields only)
- Store snapshots and deltas without blocking WebSocket
- Success: Can ingest orderbook data without dropping messages

Milestone 2: Data normalization and state management
- Build in-memory orderbook state tracker
- Implement efficient delta application
- Add basic data validation
- Success: Maintain accurate orderbook state for actor use

PHASE 2: RL Foundation (Week 2)
Milestone 3: Historical-only Gymnasium environment
- Implement KalshiTradingEnv that loads from DB/Parquet
- Create observation builder (historical orderbook → features)
- Add simple reward function
- NO async, NO WebSocket connections
- Success: Can run env.reset() and env.step() on historical data

Milestone 4: SB3 integration with model registry
- Create training harness with PPO/A2C
- Let SB3 manage replay buffers (no custom buffering)
- Implement minimal model registry (checkpoint_path, version, status)
- Set up model checkpointing
- Success: Can train on historical data, save checkpoint to registry

PHASE 3: Trading Logic (Week 3)
Milestone 5: Paper trading client
- Implement order matching logic
- Track positions and P&L
- Simulate fills based on live orderbook
- Mode='paper' only (no 'live' for MVP)
- Success: Can execute paper trades with realistic fills

Milestone 6: Inference-only actor with hot-reload
- Build async actor for inference ONLY
- Connect to LIVE WebSocket for real-time data
- Implement hot-reload protocol (poll for new checkpoints)
- Log all actions to trading_actions table for observability (state_hash, action, reward)
- NOTE: WebSocket client continuously saves orderbook data to DB (this becomes training data)
- NOTE: Actor DOES log actions for analysis, but does NOT feed them back for training
- Success: Actor runs continuously, hot-reloads new models, full audit trail in DB

PHASE 4: Visualization (Week 4)
Milestone 7: ML tab frontend
- Create React components for ML dashboard
- Implement WebSocket connections
- Build orderbook visualizer
- Success: Can see live orderbook and agent actions in browser

Milestone 8: Training UI
- Add training controls (start/stop/configure)
- Display training metrics and progress
- Implement model selection/loading
- Success: Can manage training sessions from UI

PHASE 5: MVP Completion (Week 5)
Milestone 9: Basic testing
- Unit tests for critical components
- Integration test for data pipeline
- Simple model evaluation metrics
- Success: Core functionality tested

Milestone 10: Simple deployment
- Docker containerization
- Basic health check endpoints
- Simple logging setup
- Success: Deployed and running

--------------------------------
7. NON-FUNCTIONAL REQUIREMENTS (MVP)
--------------------------------

Performance (MVP):
- Orderbook updates processed without blocking
- Actor decision latency < 100ms
- Handle normal Kalshi orderbook rates
- Training: Run on CPU acceptable

Reliability (MVP):
- Basic WebSocket reconnection
- Simple error logging
- Manual restart acceptable

Data Management (MVP):
- Data retention: 7 days minimum
- Local checkpoint storage only
- Manual cleanup acceptable

Monitoring (MVP):
- Basic health endpoints (/rl/status)
- Console logging
- Manual monitoring

Configuration (MVP):
- Essential environment variables:
  - KALSHI_API_KEY
  - KALSHI_PRIVATE_KEY_PATH
  - RL_MARKET_TICKER (e.g., "TRUMPWIN-2024")
  - RL_MODE (training/paper only)
  - DATABASE_URL
  - TICK_INTERVAL_SECONDS

--------------------------------
8. TESTING STRATEGY (MVP)
--------------------------------

Unit Tests:
- Orderbook state management
- Historical environment loading
- Paper trading logic
- Hot-reload mechanism

Integration Tests:
- Non-blocking WebSocket → Database pipeline
- Historical data → Training → Checkpoint
- Live data → Actor → Paper trading

Basic E2E Test:
- Train model on historical data (10 episodes)
- Load model in actor
- Run paper trading for 5 minutes

--------------------------------
9. SUCCESS CRITERIA (MVP)
--------------------------------

MVP Completion Checklist:
✓ Single market orderbook data flowing to database (non-blocking)
✓ Historical-only Gymnasium environment working with SB3
✓ Minimal model registry with checkpoint tracking
✓ Actor loop executing paper trades on LIVE data
✓ Clear trainer/actor separation (no cross-contamination)
✓ Hot-reload protocol working
✓ ML tab showing orderbook + actions
✓ Training creates model entries with checkpoints
✓ No 'live' trading mode (paper only)
✓ Basic health endpoints
✓ Docker containerized
✓ Core tests passing

Key Metrics (MVP):
- Orderbook ingestion: No dropped messages
- Actor runs: Continuous paper trading
- Training: Completes without errors
- Hot-reload: Works within 30 seconds

--------------------------------
10. KALSHI API COMPATIBILITY & PAPER TRADING ARCHITECTURE
--------------------------------

10.1 WebSocket Stream Architecture

PUBLIC STREAMS (Already Implemented):
- Live trades feed: Real-time public trade data
- Orderbook updates: Full orderbook snapshots and deltas
- Connection: Single WebSocket handles multiple markets
- State Management: Per-market SharedOrderbookState instances

USER-SPECIFIC STREAMS (Paper Trading Simulation):
- User fills stream: Notifications when orders are executed
- Position updates: Real-time position changes
- Order status stream: Order lifecycle notifications
- Implementation: Separate subscriptions, not part of public feed
- Simulation: Paper client generates realistic stream events

10.2 Kalshi Trading API Structure

ORDER MANAGEMENT ENDPOINTS:
- POST /portfolio/orders: Create single or batch orders
- POST /portfolio/orders/{id}/amend: Modify existing order
- POST /portfolio/orders/{id}/decrease: Reduce order size
- DELETE /portfolio/orders/{id}: Cancel single order
- DELETE /portfolio/orders/batched: Batch cancel

PORTFOLIO ENDPOINTS:
- GET /portfolio/balance: Account balance
- GET /portfolio/positions: Current positions
- GET /portfolio/fills: Execution history
- GET /portfolio/settlements: Settlement information

ORDER LIFECYCLE STATES:
- pending: Order submitted but not yet active
- resting: Order active in orderbook
- executed: Order fully or partially filled
- canceled: Order canceled by user
- settled: Position settled after market resolution

10.3 Paper Trading Client Design

INTERFACE COMPATIBILITY:
```python
class KalshiPaperTradingClient:
    """100% API-compatible with future KalshiLiveTradingClient"""
    
    async def create_order(self, market_ticker: str, side: str, 
                          order_type: str, quantity: int, price: Optional[int]):
        # Identical interface to live API
        pass
    
    async def amend_order(self, order_id: str, new_quantity: int, new_price: int):
        # Matches Kalshi API exactly
        pass
    
    async def cancel_order(self, order_id: str):
        # Same as live endpoint
        pass
    
    async def get_positions(self) -> List[Position]:
        # Returns Kalshi-formatted positions
        pass
```

POSITION TRACKING (Kalshi Format):
```python
class Position:
    market_ticker: str
    position: int          # Net position (not separate yes/no)
    total_traded: int      # Volume traded
    settlement_status: str # 'unsettled' or 'settled'
    average_cost: float    # For P&L calculations
```

FILL SIMULATION ENGINE:
- Uses live orderbook data for realistic fills
- Market impact modeling based on order size
- Slippage calculation from market depth
- Latency simulation (10-100ms configurable)
- Partial fills when liquidity insufficient

10.4 Enhanced Action Space

CURRENT LIMITATIONS:
- Simple BUY_YES/SELL_YES/BUY_NO/SELL_NO actions
- No price specification for limit orders
- No order management capabilities

ENHANCED ACTION SPACE:
```python
class EnhancedAction:
    action_type: str  # 'create_order', 'cancel_order', 'amend_order'
    order_type: str   # 'market' or 'limit'
    side: str         # 'yes' or 'no'
    direction: str    # 'buy' or 'sell'
    quantity: int
    price: Optional[int]  # For limit orders
    order_id: Optional[str]  # For cancel/amend
```

10.5 Database Schema Extensions

```sql
-- Orders table for paper trading
CREATE TABLE rl_orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    market_ticker TEXT NOT NULL,
    side TEXT NOT NULL,           -- 'yes' or 'no'
    direction TEXT NOT NULL,       -- 'buy' or 'sell'
    order_type TEXT NOT NULL,      -- 'market' or 'limit'
    status TEXT NOT NULL,          -- Kalshi order states
    quantity INTEGER NOT NULL,
    remaining_quantity INTEGER NOT NULL,
    price INTEGER,                 -- in cents for limit orders
    filled_quantity INTEGER DEFAULT 0,
    average_fill_price FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Fills table for execution tracking
CREATE TABLE rl_fills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID REFERENCES rl_orders(id),
    market_ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    execution_type TEXT NOT NULL   -- 'market' or 'limit'
);

-- Indexes for performance
CREATE INDEX idx_orders_market ON rl_orders(market_ticker);
CREATE INDEX idx_orders_status ON rl_orders(status);
CREATE INDEX idx_fills_order ON rl_fills(order_id);
CREATE INDEX idx_fills_timestamp ON rl_fills(timestamp);
```

10.6 Paper → Live Transition Strategy

SEAMLESS CLIENT SWAP:
```python
def create_trading_client(mode: str, config: dict):
    """Factory pattern for paper/live client creation"""
    if mode == 'paper':
        return KalshiPaperTradingClient(config)
    elif mode == 'live':
        # Future implementation
        return KalshiLiveTradingClient(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")
```

VALIDATION REQUIREMENTS:
- Paper client methods have identical signatures to live
- All return types match Kalshi API responses exactly
- Error codes and exceptions match live behavior
- WebSocket message formats identical to production

SUCCESS METRICS:
- 100% API interface compatibility
- Fill simulation within 5% of live behavior
- Sub-100ms decision latency maintained
- Zero code changes needed for live transition
- Position/P&L calculations match Kalshi exactly

--------------------------------
11. FUTURE ENHANCEMENTS
--------------------------------

Post-MVP Roadmap:

Performance & Scale:
- Support 1000+ orderbook updates per second
- Actor decision latency < 50ms p99
- GPU training support
- Distributed training
- 100k+ training steps per hour
- Full observation storage in trading_actions
- Advanced data compression

Infrastructure:
- Prometheus metrics export
- Structured logging with correlation IDs
- Performance dashboards
- A/B testing framework
- S3 checkpoint storage
- Automatic data archival
- 95% test coverage
- Load testing suite
- Railway deployment automation
- Full observability stack

Risk Management Framework:
- Live trading mode (with safeguards)
- Position limits: MAX_POSITION_SIZE per market
- Loss limits: Daily/session maximum loss thresholds
- Circuit breakers: Automatic halt on anomalous behavior
- Model performance monitoring and automatic rollback
- Emergency stop mechanism with position unwinding
- Risk-adjusted reward functions
- Portfolio risk metrics (Sharpe ratio, max drawdown)

Advanced Trading Features:
- Multi-market simultaneous trading
- Complex portfolio management
- Advanced RL algorithms (SAC, TD3, DQN, custom)
- Custom replay buffer implementations
- Real trading integration with full risk controls
- Backtesting framework with historical data
- Market making strategies
- Cross-market arbitrage detection
- Advanced position sizing algorithms
- Ensemble models
- Production monitoring dashboard
- Algorithm-specific hyperparameter tracking

Research Directions:
- Transformer-based architectures for orderbook
- Multi-agent reinforcement learning
- Meta-learning for market regime changes
- Explainable AI for trade decisions
- Adversarial training for robustness
- Advanced reward shaping
- Hierarchical RL approaches

END_OF_SPEC