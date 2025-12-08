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
- mode                TEXT NOT NULL          -- 'training', 'paper', 'live'
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
- current_position: dict[market, position]
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

--------------------------------
3. BACKEND ARCHITECTURE
--------------------------------

3.1 Service Structure

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

3.2 Orderbook WebSocket Client (Non-Blocking)

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

3.3 Gymnasium Environment (Historical Data Only)

```python
class KalshiTradingEnv(gym.Env):
    """
    CRITICAL: This environment ONLY replays historical data.
    It does NOT connect to live WebSockets or use async.
    All data comes from PostgreSQL or Parquet files.
    """
    def __init__(self, config):
        # Load historical data from DB/Parquet
        self.historical_data = self.load_historical_data(
            market=config['market'],
            start_time=config['start_time'],
            end_time=config['end_time']
        )
        self.observation_space = spaces.Box(...)
        self.action_space = spaces.Discrete(5)  # hold, buy_yes, sell_yes, buy_no, sell_no
        self.current_step = 0
        
    def reset(self):
        # Reset to beginning of historical data
        self.current_step = 0
        return self._get_observation()
    
    def step(self, action):
        # Advance through historical data
        self.current_step += 1
        # Simulate action execution on historical orderbook
        # Calculate reward based on simulated fills
        obs = self._get_observation()
        reward = self._calculate_reward(action)
        done = self.current_step >= len(self.historical_data)
        return obs, reward, done, False, {}
    
    def _get_observation(self):
        # Return features from historical data at current_step
        # NO live data, NO async calls
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

3.4 Actor Loop (Inference Only)

```python
class TradingActor:
    """
    CRITICAL: Actor is inference-only. It NEVER trains or writes training data.
    It connects to live WebSocket for real-time orderbook data.
    Supports hot-reload of new model checkpoints.
    """
    def __init__(self, model_id, trading_client, config):
        self.model_id = model_id
        self.model = self.load_model(model_id)
        self.client = trading_client  # Paper only for MVP
        self.state = ActorState()
        self.last_checkpoint_mtime = None
        
    async def run(self):
        while True:
            # Check for model updates (hot-reload)
            await self.check_and_reload_model()
            
            # Get latest LIVE orderbook state from WebSocket
            obs = await self.get_live_observation()
            
            # Get action from model (inference only)
            with torch.no_grad():  # Ensure no gradient computation
                action, _ = self.model.predict(obs, deterministic=True)
            
            # Execute via paper trading client
            result = await self.client.execute(action)
            
            # Log action to DB for observability/analysis (NOT for training)
            await self.log_action_to_database(action, obs_hash, result, reward)
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

3.5 Trainer/Actor Separation Contract

**CRITICAL DESIGN PRINCIPLE**: Complete separation between training and inference.

```python
# Trainer (Historical Data Only)
class Trainer:
    """
    Runs SB3 training loops on historical data.
    NEVER connects to live WebSocket.
    NEVER does inference on live data.
    """
    def train(self, model_id, config):
        # Load historical environment
        env = KalshiTradingEnv(config)  # Historical data only
        
        # Initialize or load SB3 model
        if config.get('parent_model_id'):
            model = PPO.load(parent_checkpoint_path)
        else:
            model = PPO('MlpPolicy', env, **hyperparams)
        
        # Train using SB3's internal replay buffer management
        model.learn(total_timesteps=config['timesteps'])
        
        # Save checkpoint
        checkpoint_path = f"models/{model_id}/checkpoint.zip"
        model.save(checkpoint_path)
        
        # Update database
        update_model_status(model_id, 'ready', checkpoint_path)

# Actor (Live Data Only)
class Actor:
    """
    Does inference on live WebSocket data.
    NEVER trains or updates weights.
    NEVER writes experience tuples for training.
    """
    # See section 3.4 above
```

**Data Flow Separation**:
- Data Collection: Live WebSocket → Async Queue → PostgreSQL (continuous, always running)
- Training: PostgreSQL historical → Gymnasium Env → SB3 → Checkpoint
- Inference: Live WebSocket → Actor → Paper Trading → Action Logging to DB

**Important Data Storage Distinction**:
- Orderbook data (orderbook_snapshots/deltas): Continuously collected, becomes training dataset
- Actor actions (trading_actions table): Logged for observability, debugging, performance analysis
- Training replay buffer: Managed internally by SB3, NOT stored in DB
- Key point: We LOG everything for analysis, but don't manually manage training data structures

**Hot-Reload Protocol**:
1. Trainer saves new checkpoint to `checkpoint_path`
2. Trainer updates DB: model version++, status='ready'
3. Actor polls for changes every N seconds
4. Actor detects new version or mtime change
5. Actor loads new checkpoint atomically
6. Actor continues with new model (no downtime)

3.6 API Endpoints

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
4. FRONTEND INTEGRATION
--------------------------------

4.1 ML Tab Component Structure

```
frontend/src/components/ml/
├── MLDashboard.tsx          # Main container
├── OrderbookVisualizer.tsx  # Orderbook depth chart
├── ActionFeed.tsx           # Recent actions list
├── PerformanceMetrics.tsx  # P&L, win rate, etc.
├── ModelControl.tsx         # Start/stop/load model
└── TrainingProgress.tsx    # Training charts

```

4.2 WebSocket Protocol

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
5. IMPLEMENTATION ROADMAP
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
6. NON-FUNCTIONAL REQUIREMENTS (MVP)
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
7. TESTING STRATEGY (MVP)
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
8. SUCCESS CRITERIA (MVP)
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
9. FUTURE ENHANCEMENTS
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