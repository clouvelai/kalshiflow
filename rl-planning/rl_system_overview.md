APP_NAME: Kalshi Flow RL Trading Subsystem

ONE_LINER:
A reinforcement learning agent that observes Kalshi orderbook data in real-time, trains trading policies using Gymnasium + Stable Baselines3, and executes paper/live trades with full visualization and monitoring capabilities.

CORE CONSTRAINT:
Must operate as a completely separate service (kalshiflow-rl-service) that does not interfere with the existing analytics backend. All RL functionality must be namespaced under /rl/* endpoints.

TECH STACK (LOCKED IN):
- Backend: Python 3.x + Starlette (ASGI) with async-first architecture
- Frontend: React + Vite + Tailwind CSS (existing Kalshi Flow frontend)
- Database: PostgreSQL via Supabase (shared with main app)
- ML Framework: Gymnasium + Stable Baselines3 (SB3)
- Data Source: Kalshi orderbook WebSocket API

--------------------------------
1. SCOPE & NON-GOALS
--------------------------------

IN-SCOPE (MVP):
- Single-market orderbook ingestion via WebSocket
- Normalized orderbook storage in PostgreSQL
- Custom Gymnasium environment for Kalshi trading
- SB3 integration with basic algorithms (PPO/A2C)
- Live actor loop with hot-swappable policies
- Paper trading client implementation
- WebSocket-based ML visualization tab
- Training from historical orderbook data
- Real-time orderbook → action pipeline
- Non-blocking async architecture throughout

OUT-OF-SCOPE (MVP):
- Multi-market simultaneous trading
- Complex portfolio management
- Advanced RL algorithms beyond SB3 defaults
- Backtesting framework
- Risk management beyond position limits
- Market making strategies
- Cross-market arbitrage
- Production live trading (paper only for MVP)
- Advanced performance metrics/dashboards
- Distributed training

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
- state_before        JSONB                  -- compressed observation
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
- model: sb3.Model
- observation_buffer: deque(maxlen=N)
- action_history: deque(maxlen=M)
- performance_metrics: dict

TrainingBufferState:
- experience_buffer: deque(maxlen=100000)  -- (state, action, reward, next_state, done) tuples
- episode_returns: list[float]
- current_trajectories: dict[UUID, list]

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

3.2 Orderbook WebSocket Client

Responsibilities:
- Connect to Kalshi orderbook WebSocket:
  - URL: wss://api.elections.kalshi.com/trade-api/ws/v2
  - Subscribe to orderbook channel for configured market
- Handle authentication (RSA signature, same as trade stream)
- Process messages:
  - Initial snapshot
  - Delta updates
  - Sequence number tracking
  - Checksum validation
- Normalize and persist to database
- Broadcast to actor loop and UI subscribers

Implementation:
```python
class OrderbookClient:
    async def connect(self):
        # WebSocket connection with auth
        # Subscribe to orderbook channel
        pass
    
    async def process_snapshot(self, msg):
        # Store full orderbook state
        # Update in-memory state
        pass
    
    async def process_delta(self, msg):
        # Apply delta to in-memory state
        # Store delta record
        # Validate sequence continuity
        pass
```

3.3 Gymnasium Environment

```python
class KalshiTradingEnv(gym.Env):
    def __init__(self, config):
        # Define observation/action spaces
        # Initialize data source (DB or stream)
        self.observation_space = spaces.Box(...)
        self.action_space = spaces.Discrete(5)  # hold, buy_yes, sell_yes, buy_no, sell_no
        
    def reset(self):
        # Reset to initial state
        # Return first observation
        pass
    
    def step(self, action):
        # Execute action
        # Calculate reward
        # Return obs, reward, done, truncated, info
        pass
    
    def render(self):
        # Send state to WebSocket for UI
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

3.4 Actor Loop

```python
class TradingActor:
    def __init__(self, model, trading_client, config):
        self.model = model
        self.client = trading_client  # Paper or real
        self.state = ActorState()
        
    async def run(self):
        while True:
            # Get latest orderbook state
            obs = await self.get_observation()
            
            # Get action from model
            action, _ = self.model.predict(obs)
            
            # Execute via trading client
            result = await self.client.execute(action)
            
            # Collect training data
            await self.collect_experience(obs, action, result)
            
            # Log and broadcast
            await self.log_action(action, result)
            await self.broadcast_to_ui(action, obs)
            
            await asyncio.sleep(self.config.tick_interval)
```

3.5 API Endpoints

HTTP Routes (/rl/*):
- GET /rl/status - Service health and status
- GET /rl/models - List all models with filters
- GET /rl/models/{id} - Model details and performance
- POST /rl/models - Create new model entry
- PUT /rl/models/{id} - Update model status/notes
- GET /rl/episodes - List training/trading episodes
- GET /rl/episodes/{id} - Episode details
- POST /rl/training/start - Start training session (creates model)
- POST /rl/training/stop - Stop training
- POST /rl/actor/start - Start live actor with model_id
- POST /rl/actor/stop - Stop actor
- POST /rl/actor/switch-model - Hot-swap to different model
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
    "epsilon": 0.1
  }
}
```

--------------------------------
5. IMPLEMENTATION ROADMAP
--------------------------------

PHASE 1: Data Pipeline (Week 1)
Milestone 1: Orderbook WebSocket ingestion
- Implement orderbook_client.py with Kalshi auth
- Create database models and migrations
- Store snapshots and deltas
- Success: Can connect, subscribe, and persist orderbook data for one market

Milestone 2: Data normalization and state management
- Build in-memory orderbook state tracker
- Implement efficient delta application
- Add data validation and error recovery
- Success: Maintain accurate orderbook state with < 10ms latency

PHASE 2: RL Foundation (Week 2)
Milestone 3: Gymnasium environment
- Implement KalshiTradingEnv with proper spaces
- Create observation builder (orderbook → features)
- Add dummy reward function
- Success: Can run env.reset() and env.step() with random actions

Milestone 4: SB3 integration with model tracking
- Create training harness with PPO
- Implement model registry (create model entry in DB)
- Set up model checkpointing with versioning
- Add tensorboard logging
- Success: Can train a model for 100 episodes, track in models table

PHASE 3: Trading Logic (Week 3)
Milestone 5: Paper trading client
- Implement order matching logic
- Track positions and P&L
- Simulate fills based on orderbook
- Success: Can execute paper trades with realistic fills

Milestone 6: Actor loop with feedback collection
- Build async actor with model inference
- Implement model loading from registry
- Integrate experience collection for training
- Add action logging linked to model_id
- Support hot-swapping models without restart
- Success: Actor runs continuously, collects training data

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

PHASE 5: Production Readiness (Week 5)
Milestone 9: Testing and model evaluation
- Unit tests for all components
- Integration tests for data pipeline
- Model evaluation framework
- A/B testing capability
- Load testing for actor loop
- Success: 95% test coverage, model validation pipeline working

Milestone 10: Deployment and monitoring
- Docker containerization
- Railway deployment config
- Health check endpoints
- Prometheus metrics export
- Data lifecycle automation
- Alerting configuration
- Success: Deployed to staging with full observability

--------------------------------
6. NON-FUNCTIONAL REQUIREMENTS
--------------------------------

Performance:
- Orderbook updates processed within 10ms
- Actor decision latency < 50ms
- Support 1000+ orderbook updates per second
- Training: 100k steps per hour minimum
- GPU optional for MVP, recommended for production

Reliability:
- Automatic WebSocket reconnection with exponential backoff
- Sequence number validation and gap detection
- Graceful degradation on data loss
- Actor loop recovery from crashes
- Market halt/closure handling

Data Management:
- Data retention: 30 days for orderbook snapshots
- Automatic archival of older data to cold storage
- Pruning strategy for training data (keep best episodes)
- Checkpoint storage: Local for MVP, S3 for production

Monitoring & Observability:
- Health endpoints for each component
- Prometheus metrics export
- Structured logging with correlation IDs
- Alerting on critical failures
- Performance dashboards

Configuration:
- Environment variables:
  - KALSHI_API_KEY
  - KALSHI_PRIVATE_KEY_PATH
  - RL_MARKET_TICKER (e.g., "TRUMPWIN-2024")
  - RL_MODE (training/paper/live)
  - MODEL_CHECKPOINT_PATH
  - CHECKPOINT_STORAGE (local/s3)
  - TICK_INTERVAL_SECONDS
  - DATABASE_URL
  - DATA_RETENTION_DAYS

--------------------------------
7. TESTING STRATEGY
--------------------------------

Unit Tests:
- Orderbook state management
- Observation/action encoding
- Reward calculation
- Paper trading logic

Integration Tests:
- WebSocket → Database pipeline
- Environment → Agent → Trading flow
- API endpoint responses

E2E Tests:
- Full training run (10 episodes)
- Actor loop with dummy model
- UI visualization pipeline

Performance Tests:
- Orderbook update throughput
- Actor inference latency
- Database write performance

--------------------------------
8. SUCCESS CRITERIA
--------------------------------

MVP Completion Checklist:
✓ Single market orderbook data flowing to database
✓ Gymnasium environment working with SB3
✓ Model registry tracking all trained models
✓ Actor loop executing paper trades
✓ Experience collection feeding back to training pipeline
✓ ML tab showing orderbook + actions + model info
✓ Training runs creating versioned model entries
✓ Model checkpoints loadable by ID from registry
✓ Model evaluation before deployment
✓ Support for training multiple models on same market
✓ Model comparison and lineage tracking
✓ Data retention and lifecycle management
✓ Health monitoring and alerting
✓ WebSocket reconnection with error handling
✓ All components containerized
✓ Documentation complete
✓ Tests passing with >90% coverage

Key Metrics:
- Data pipeline uptime: >99%
- Actor decision time: <50ms p99
- Training convergence: Positive reward trend
- Paper trading accuracy: Realistic fills
- UI latency: <100ms updates

--------------------------------
9. FUTURE ENHANCEMENTS
--------------------------------

Post-MVP Roadmap:

Risk Management Framework:
- Position limits: MAX_POSITION_SIZE per market
- Loss limits: Daily/session maximum loss thresholds
- Circuit breakers: Automatic halt on anomalous behavior
- Model performance monitoring and automatic rollback
- Emergency stop mechanism with position unwinding
- Risk-adjusted reward functions
- Portfolio risk metrics (Sharpe ratio, max drawdown)

Advanced Trading Features:
- Multi-market support with portfolio optimization
- Advanced RL algorithms (SAC, TD3, custom)
- Real trading integration with full risk controls
- Backtesting framework with historical data
- Market making strategies
- Cross-market arbitrage detection
- Advanced position sizing algorithms
- Ensemble models
- Distributed training on GPU cluster
- Production monitoring dashboard

Research Directions:
- Transformer-based architectures for orderbook
- Multi-agent reinforcement learning
- Meta-learning for market regime changes
- Explainable AI for trade decisions
- Adversarial training for robustness

END_OF_SPEC