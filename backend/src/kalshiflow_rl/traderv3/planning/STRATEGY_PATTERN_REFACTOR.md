# Strategy Pattern Refactor Proposal

> Architecture proposal for cleaner strategy extensibility in TRADER V3.
> Status: **Proposed** | Created: 2024-12-29

## 1. Current State

### 1.1 How Strategies Work Today

The V3 Trader supports multiple trading strategies through a combination of:

1. **TradingDecisionService** (`services/trading_decision_service.py`)
   - Central switch statement in `evaluate_market()` routes to strategy methods
   - Strategies enum: `HOLD`, `WHALE_FOLLOWER`, `PAPER_TEST`, `RL_MODEL`, `YES_80_90`, `CUSTOM`
   - Each strategy has inline implementation or delegates to external service

2. **Standalone Strategy Services**
   - `Yes8090Service` - Event-driven, subscribes to `ORDERBOOK_SNAPSHOT/DELTA`
   - `WhaleExecutionService` - Event-driven, subscribes to `WHALE_QUEUE_UPDATED`

3. **Coordinator Integration** (`core/coordinator.py`)
   - Conditionally starts strategy services based on config
   - Manages lifecycle (start/stop) for each service

### 1.2 Current File Locations

```
traderv3/
├── services/
│   ├── trading_decision_service.py  # Central dispatch + some inline strategies
│   ├── yes_80_90_service.py         # YES 80-90c strategy (event-driven)
│   ├── whale_execution_service.py   # Whale follower strategy (event-driven)
│   └── whale_tracker.py             # Whale detection (not a strategy)
├── core/
│   └── coordinator.py               # Starts/stops strategy services
└── config/
    └── environment.py               # V3Config with strategy settings
```

---

## 2. Problems with Current Design

### 2.1 Adding New Strategies Requires Multiple File Edits

To add a new strategy (e.g., "Mean Reversion"), you must:

1. Add to `TradingStrategy` enum in `trading_decision_service.py`
2. Add switch case in `evaluate_market()` method
3. Create new service file in `services/`
4. Add conditional startup in `coordinator.py`
5. Add environment variables in `config/environment.py`
6. Update architecture documentation

**Impact**: High friction for strategy experimentation

### 2.2 No Clear Interface Contract

Current strategies have inconsistent interfaces:

| Method | TradingDecisionService | Yes8090Service | WhaleExecutionService |
|--------|------------------------|----------------|-----------------------|
| `start()` | N/A | ✅ | ✅ |
| `stop()` | N/A | ✅ | ✅ |
| `is_healthy()` | ✅ | ✅ | ✅ |
| `get_stats()` | ✅ | ✅ | ✅ |
| `get_decision_history()` | ✅ (partial) | ✅ | ✅ |
| Event subscriptions | None | `ORDERBOOK_*` | `WHALE_QUEUE_UPDATED` |

**Impact**: Inconsistent monitoring, hard to add generic strategy management

### 2.3 Testing Individual Strategies is Hard

- Strategies are tightly coupled to `EventBus`, `V3StateContainer`, `TradingDecisionService`
- No mock boundaries defined
- Integration tests required even for unit-level logic

### 2.4 Configuration is Inconsistent

- `WHALE_*` env vars for whale strategy
- `YES8090_*` env vars for 80-90c strategy
- `V3_TRADING_STRATEGY` to select active strategy
- No validation that strategy config exists for selected strategy

---

## 3. Proposed Architecture

### 3.1 Core Interface: `BaseStrategy`

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class StrategyConfig:
    """Base configuration for all strategies."""
    name: str
    enabled: bool = True
    max_concurrent_positions: int = 100
    rate_limit_trades_per_minute: int = 10

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Lifecycle: configure() -> start() -> [process events] -> stop()
    """

    def __init__(self, config: StrategyConfig):
        self._config = config
        self._running = False
        self._started_at: Optional[float] = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def is_running(self) -> bool:
        return self._running

    # === Lifecycle Methods ===

    @abstractmethod
    async def start(self) -> None:
        """Start the strategy. Subscribe to events, initialize state."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the strategy. Unsubscribe from events, cleanup."""
        pass

    # === Event Handlers (Optional) ===

    async def on_orderbook_update(self, event: Dict[str, Any]) -> None:
        """Handle orderbook snapshot/delta. Override if needed."""
        pass

    async def on_trade(self, event: Dict[str, Any]) -> None:
        """Handle public trade event. Override if needed."""
        pass

    async def on_position_update(self, event: Dict[str, Any]) -> None:
        """Handle position update from Kalshi. Override if needed."""
        pass

    # === Monitoring (Required) ===

    @abstractmethod
    def is_healthy(self) -> bool:
        """Return True if strategy is functioning correctly."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return strategy-specific statistics."""
        pass

    @abstractmethod
    def get_decision_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent trading decisions for audit display."""
        pass

    # === Execution Interface ===

    @abstractmethod
    async def execute_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Execute a trading signal generated by this strategy.

        Returns True if execution was successful, False otherwise.
        """
        pass
```

### 3.2 Strategy Registry

```python
class StrategyRegistry:
    """
    Central registry for all trading strategies.

    Usage:
        registry = StrategyRegistry(event_bus, trading_service, state_container)
        registry.register("yes_80_90", Yes8090Strategy)
        registry.register("whale_follower", WhaleFollowerStrategy)

        await registry.start_strategy("yes_80_90", config)
        await registry.stop_strategy("yes_80_90")
    """

    def __init__(
        self,
        event_bus: EventBus,
        trading_service: TradingDecisionService,
        state_container: V3StateContainer
    ):
        self._event_bus = event_bus
        self._trading_service = trading_service
        self._state_container = state_container
        self._strategy_classes: Dict[str, Type[BaseStrategy]] = {}
        self._active_strategies: Dict[str, BaseStrategy] = {}

    def register(self, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """Register a strategy class by name."""
        self._strategy_classes[name] = strategy_class

    async def start_strategy(self, name: str, config: StrategyConfig) -> BaseStrategy:
        """Instantiate and start a strategy."""
        if name in self._active_strategies:
            raise ValueError(f"Strategy '{name}' is already active")

        strategy_class = self._strategy_classes[name]
        strategy = strategy_class(
            config=config,
            event_bus=self._event_bus,
            trading_service=self._trading_service,
            state_container=self._state_container
        )
        await strategy.start()
        self._active_strategies[name] = strategy
        return strategy

    async def stop_strategy(self, name: str) -> None:
        """Stop and remove a strategy."""
        if name in self._active_strategies:
            await self._active_strategies[name].stop()
            del self._active_strategies[name]

    async def stop_all(self) -> None:
        """Stop all active strategies."""
        for name in list(self._active_strategies.keys()):
            await self.stop_strategy(name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats from all active strategies."""
        return {
            name: strategy.get_stats()
            for name, strategy in self._active_strategies.items()
        }
```

### 3.3 Configuration via Pydantic

```python
from pydantic import BaseSettings, Field

class WhaleFollowerConfig(StrategyConfig):
    """Configuration for Whale Follower strategy."""
    min_whale_size_cents: int = Field(10000, env="WHALE_MIN_SIZE_CENTS")
    max_age_seconds: int = Field(120, env="WHALE_MAX_AGE_SECONDS")
    contracts_per_trade: int = Field(5, env="WHALE_FOLLOW_CONTRACTS")

class Yes8090Config(StrategyConfig):
    """Configuration for YES 80-90c strategy."""
    min_price_cents: int = Field(80, env="YES8090_MIN_PRICE")
    max_price_cents: int = Field(90, env="YES8090_MAX_PRICE")
    min_liquidity: int = Field(10, env="YES8090_MIN_LIQUIDITY")
    max_spread_cents: int = Field(5, env="YES8090_MAX_SPREAD")
    tier_a_contracts: int = Field(150, env="YES8090_TIER_A_CONTRACTS")
    tier_b_contracts: int = Field(100, env="YES8090_CONTRACTS")

# Registry of all strategy configs
STRATEGY_CONFIGS = {
    "whale_follower": WhaleFollowerConfig,
    "yes_80_90": Yes8090Config,
}
```

---

## 4. Migration Path

### Phase 1: Create Interface (Non-Breaking)

**Goal**: Add `BaseStrategy` and `StrategyRegistry` without changing existing code.

1. Create `traderv3/strategies/base.py` with `BaseStrategy` ABC
2. Create `traderv3/strategies/registry.py` with `StrategyRegistry`
3. Create `traderv3/strategies/__init__.py` with exports

**Files to Create**:
- `traderv3/strategies/base.py`
- `traderv3/strategies/registry.py`
- `traderv3/strategies/__init__.py`

**Estimated Effort**: 2-3 hours

### Phase 2: Wrap Existing Strategies

**Goal**: Create strategy wrappers that delegate to existing services.

1. Create `Yes8090Strategy` that wraps `Yes8090Service`
2. Create `WhaleFollowerStrategy` that wraps `WhaleExecutionService`
3. Both implement `BaseStrategy` interface

**Files to Create**:
- `traderv3/strategies/yes_80_90.py`
- `traderv3/strategies/whale_follower.py`

**Files to Modify**:
- `traderv3/core/coordinator.py` - Use registry instead of direct service management

**Estimated Effort**: 4-6 hours

### Phase 3: Migrate Strategies to New Pattern

**Goal**: Refactor strategy services to directly extend `BaseStrategy`.

1. Move `Yes8090Service` logic into `Yes8090Strategy`
2. Move `WhaleExecutionService` logic into `WhaleFollowerStrategy`
3. Update `TradingDecisionService` to be execution-only (no strategy dispatch)
4. Remove old service files

**Files to Delete**:
- `traderv3/services/yes_80_90_service.py`
- `traderv3/services/whale_execution_service.py`

**Estimated Effort**: 8-12 hours

### Phase 4: Add New Strategy Template

**Goal**: Create a documented template for new strategies.

1. Create `traderv3/strategies/TEMPLATE.py` with example strategy
2. Update documentation with "How to Add a Strategy" guide
3. Add strategy tests with mock boundaries

**Estimated Effort**: 2-3 hours

---

## 5. Benefits After Refactor

| Aspect | Before | After |
|--------|--------|-------|
| Add new strategy | 6+ file edits | 1 file + config |
| Interface contract | Implicit | Explicit `BaseStrategy` |
| Testing | Integration required | Unit-testable with mocks |
| Configuration | Scattered env vars | Centralized Pydantic |
| Runtime management | Hard-coded in coordinator | Dynamic via registry |
| Multiple strategies | Single active | Multiple concurrent |

---

## 6. Risks and Mitigations

### Risk 1: Breaking Existing Functionality
- **Mitigation**: Phase 1-2 are non-breaking, existing code continues to work
- **Mitigation**: Full test coverage before Phase 3

### Risk 2: Over-Engineering for Current Needs
- **Mitigation**: Stop after Phase 2 if only 2-3 strategies needed
- **Mitigation**: Keep interface minimal, extend as needed

### Risk 3: Performance Overhead from Abstraction
- **Mitigation**: Registry lookup is O(1), negligible overhead
- **Mitigation**: Event handlers are async, no blocking

---

## 7. Decision Points for Implementation

Before starting implementation, decide:

1. **Multi-Strategy Support**: Should the system run multiple strategies concurrently?
   - Current: Single strategy active at a time
   - Proposed: Registry supports multiple, but coordinator can enforce single

2. **Hot Reload**: Should strategies be startable/stoppable at runtime?
   - Current: Restart required to change strategy
   - Proposed: Registry enables runtime changes

3. **Configuration Validation**: Should startup fail if strategy config is invalid?
   - Current: Silent fallback to defaults
   - Proposed: Pydantic validation with clear errors

---

## Appendix: Example New Strategy

```python
# traderv3/strategies/mean_reversion.py

from .base import BaseStrategy, StrategyConfig
from dataclasses import dataclass

@dataclass
class MeanReversionConfig(StrategyConfig):
    """Configuration for Mean Reversion strategy."""
    lookback_periods: int = 20
    std_dev_threshold: float = 2.0
    position_size: int = 50

class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy: buy when price is N std devs below mean.
    """

    def __init__(
        self,
        config: MeanReversionConfig,
        event_bus: EventBus,
        trading_service: TradingDecisionService,
        state_container: V3StateContainer
    ):
        super().__init__(config)
        self._event_bus = event_bus
        self._trading_service = trading_service
        self._state_container = state_container
        self._price_history: Dict[str, deque] = {}
        self._decisions: deque = deque(maxlen=100)

    async def start(self) -> None:
        self._running = True
        self._started_at = time.time()
        await self._event_bus.subscribe(EventType.ORDERBOOK_SNAPSHOT, self.on_orderbook_update)
        logger.info(f"MeanReversionStrategy started with lookback={self._config.lookback_periods}")

    async def stop(self) -> None:
        self._running = False
        logger.info("MeanReversionStrategy stopped")

    async def on_orderbook_update(self, event: Dict[str, Any]) -> None:
        ticker = event.get("market_ticker")
        price = event.get("mid_price")

        # Update price history
        if ticker not in self._price_history:
            self._price_history[ticker] = deque(maxlen=self._config.lookback_periods)
        self._price_history[ticker].append(price)

        # Check for signal
        if len(self._price_history[ticker]) == self._config.lookback_periods:
            signal = self._detect_signal(ticker)
            if signal:
                await self.execute_signal(signal)

    def _detect_signal(self, ticker: str) -> Optional[Dict[str, Any]]:
        prices = list(self._price_history[ticker])
        mean = sum(prices) / len(prices)
        std = (sum((p - mean) ** 2 for p in prices) / len(prices)) ** 0.5

        current = prices[-1]
        z_score = (current - mean) / std if std > 0 else 0

        if z_score < -self._config.std_dev_threshold:
            return {"ticker": ticker, "side": "yes", "price": current, "z_score": z_score}
        return None

    async def execute_signal(self, signal: Dict[str, Any]) -> bool:
        decision = TradingDecision(
            action="buy",
            market=signal["ticker"],
            side=signal["side"],
            quantity=self._config.position_size,
            price=signal["price"],
            reason=f"mean_reversion:z={signal['z_score']:.2f}",
            strategy=TradingStrategy.CUSTOM
        )
        success = await self._trading_service.execute_decision(decision)
        self._decisions.append({"signal": signal, "success": success, "timestamp": time.time()})
        return success

    def is_healthy(self) -> bool:
        return self._running

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "running": self._running,
            "markets_tracked": len(self._price_history),
            "decisions_made": len(self._decisions),
            "uptime_seconds": time.time() - self._started_at if self._started_at else 0
        }

    def get_decision_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self._decisions)[-limit:]
```

Usage:
```python
# In coordinator.py
from .strategies import StrategyRegistry, MeanReversionStrategy, MeanReversionConfig

registry = StrategyRegistry(event_bus, trading_service, state_container)
registry.register("mean_reversion", MeanReversionStrategy)

config = MeanReversionConfig(
    name="mean_reversion",
    lookback_periods=20,
    std_dev_threshold=2.0,
    position_size=50
)
await registry.start_strategy("mean_reversion", config)
```
