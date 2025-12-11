# Kalshi Flow RL Training System

This directory contains the complete Stable Baselines3 (SB3) integration for training reinforcement learning agents on Kalshi market data.

## Overview

The training system provides a production-ready pipeline for training RL agents using real Kalshi orderbook data collected from live markets. It supports multiple algorithms (PPO, A2C), curriculum learning across sessions, and comprehensive metrics tracking.

## Quick Start

### Basic Training

```bash
# Train PPO on session 9 for 10,000 timesteps
uv run python scripts/train_with_sb3.py --session 9 --algorithm ppo --total-timesteps 10000

# Train A2C on session 9 with custom starting cash
uv run python scripts/train_with_sb3.py --session 9 --algorithm a2c --total-timesteps 5000 --cash-start 5000
```

### Curriculum Learning

```bash
# Train across multiple sessions with market rotation
uv run python scripts/train_with_sb3.py --sessions 6,7,8,9 --algorithm ppo --total-timesteps 50000
```

### Environment Validation

```bash
# Validate environment compatibility with SB3
uv run python scripts/validate_sb3_environment.py

# Validate specific session
uv run python scripts/validate_sb3_environment.py 9
```

## Architecture

### Core Components

- **`sb3_wrapper.py`**: Main integration layer between MarketAgnosticKalshiEnv and SB3
  - `SessionBasedEnvironment`: Gym environment that rotates through market views
  - `CurriculumEnvironmentFactory`: Creates environments from session data
  - Configuration dataclasses for training setup

- **`curriculum.py`**: Session-based curriculum learning implementation
  - `SimpleSessionCurriculum`: Manages training across multiple sessions
  - `MarketSessionView`: Single-market view of session data
  - Training result tracking and metrics

### Data Flow

```
Database Session → SessionDataLoader → MarketSessionView → MarketAgnosticKalshiEnv → SB3 Algorithm → Trained Model
       ↑                    ↑                  ↑                      ↑                     ↑
  (Real orderbooks)   (Async loading)   (Market rotation)    (52 features, 5 actions)  (PPO/A2C)
```

### Key Features

1. **Market-Agnostic Training**: Agents learn universal trading strategies across different markets
2. **Session-Based Episodes**: Each episode is a complete market session with natural termination
3. **Automatic Market Rotation**: Curriculum learning cycles through all valid markets in sessions
4. **OrderManager Integration**: Realistic limit order simulation with proper position tracking
5. **Cents Arithmetic**: All monetary values in cents for exact precision (matching Kalshi API)

## Training Scripts

### Main Training Script

`scripts/train_with_sb3.py` - Complete training pipeline with all options:

```bash
Options:
  --session SESSION          Single session ID to train on
  --sessions SESSIONS        Comma-separated session IDs for curriculum learning
  --algorithm {ppo,a2c}      RL algorithm to use
  --total-timesteps N        Total training timesteps
  --learning-rate LR         Custom learning rate (optional)
  --cash-start CENTS         Starting cash in cents (default: 10000)
  --model-save-path PATH     Where to save trained model
  --resume-from PATH         Resume training from checkpoint
  --save-freq N              Checkpoint save frequency
  --eval-freq N              Evaluation frequency
  --log-level {DEBUG,INFO}   Logging verbosity
```

### Validation Script

`scripts/validate_sb3_environment.py` - Validates environment compatibility:

```bash
# Validates:
- 52-feature observation space
- 5-action discrete action space
- Gymnasium compatibility
- SB3 compatibility
- Episode simulation
```

## Session Data

Training uses real orderbook data collected from Kalshi markets:

- **Session Structure**: Each session contains multiple markets with orderbook snapshots and deltas
- **Typical Session**: 500+ markets, 40,000+ timesteps, 30+ minutes of data
- **Market Coverage**: Elections, sports, economics, weather, etc.

### Checking Available Sessions

```bash
# List all available sessions
uv run python scripts/fetch_session_data.py --list

# Analyze specific session
uv run python scripts/fetch_session_data.py --analyze 9
```

## Environment Details

### Observation Space
- **52 features** including:
  - Market features (21): Spreads, depths, imbalances, probabilities
  - Temporal features (14): Time gaps, activity bursts, momentum
  - Portfolio features (12): Position, P&L, cash balance
  - Global features (5): Session progress, market metadata

### Action Space
- **5 discrete actions**:
  - 0: HOLD
  - 1: BUY_YES_LIMIT
  - 2: SELL_YES_LIMIT
  - 3: BUY_NO_LIMIT
  - 4: SELL_NO_LIMIT

### Reward Function
- Simple portfolio value change in cents
- Captures all important signals naturally (P&L, costs, market impact)

## Model Management

Trained models are saved to `backend/src/kalshiflow_rl/trained_models/`:

```bash
trained_models/
├── trained_model.zip          # Latest trained model
├── checkpoints/               # Training checkpoints
│   ├── model_10000_steps.zip
│   ├── model_20000_steps.zip
│   └── ...
├── best_model/                # Best performing model
│   └── best_model.zip
└── eval_logs/                 # Evaluation metrics
```

### Loading Trained Models

```python
from stable_baselines3 import PPO
from kalshiflow_rl.training.sb3_wrapper import SessionBasedEnvironment

# Load trained model
model = PPO.load("backend/src/kalshiflow_rl/trained_models/trained_model.zip")

# Use for inference
obs = env.reset()
action, _ = model.predict(obs)
```

## Programmatic Usage

While the training scripts are the recommended way to train, you can also use the system programmatically:

```python
import asyncio
from kalshiflow_rl.training.sb3_wrapper import (
    SessionBasedEnvironment, 
    SB3TrainingConfig,
    create_env_config,
    create_training_config
)
from stable_baselines3 import PPO

async def train_custom():
    # Configure environment
    env_config = create_env_config(cash_start=10000)
    training_config = create_training_config(min_episode_length=10)
    
    config = SB3TrainingConfig(
        env_config=env_config,
        min_episode_length=training_config.min_episode_length,
        max_episode_steps=training_config.max_episode_steps,
        skip_failed_markets=training_config.skip_failed_markets
    )
    
    # Create environment
    database_url = os.getenv("DATABASE_URL")
    env = SessionBasedEnvironment(database_url, [9], config)
    await env.initialize()  # Load market views
    
    # Train model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("backend/src/kalshiflow_rl/trained_models/my_model.zip")

asyncio.run(train_custom())
```

## Performance

Typical training performance on modern hardware:
- **PPO**: 800-1000 timesteps/second
- **A2C**: 1500-2000 timesteps/second
- **Memory Usage**: ~2-4GB for large sessions
- **Episode Reset**: <100ms with pre-loaded data

## Testing

Comprehensive test coverage in `backend/tests/test_rl/integration/`:

```bash
# Run integration tests
uv run pytest tests/test_rl/integration/test_sb3_training.py -v

# Run end-to-end tests
uv run pytest tests/test_rl/integration/test_end_to_end_training.py -v
```

## Troubleshooting

### Common Issues

1. **"DATABASE_URL not set"**: Ensure environment variables are configured
2. **"No sessions found"**: Run orderbook collector to gather session data
3. **"Market view too short"**: Some markets have insufficient data, system skips them automatically
4. **Memory issues**: Reduce session size or use fewer concurrent sessions

### Debug Mode

Run with DEBUG logging for detailed information:
```bash
uv run python scripts/train_with_sb3.py --session 9 --algorithm ppo --total-timesteps 1000 --log-level DEBUG
```

## Next Steps

1. **Collect More Data**: Run orderbook collector for diverse market sessions
2. **Hyperparameter Tuning**: Experiment with learning rates, network architectures
3. **Advanced Algorithms**: Try SAC, TD3, or custom algorithms
4. **Production Deployment**: Deploy trained models for paper trading
5. **Performance Analysis**: Analyze agent behavior and trading patterns

## Related Documentation

- [RL System Overview](../../rl-planning/rl_system_overview.md)
- [Implementation Plan](../../rl-planning/rl-implementation-plan.json)
- [Market Agnostic Environment](../environments/README.md)
- [Orderbook Collector](../orderbook/README.md)