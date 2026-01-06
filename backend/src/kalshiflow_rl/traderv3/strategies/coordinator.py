"""
Strategy Coordinator for TRADER V3 Plugin System.

This module provides the StrategyCoordinator that manages multiple
concurrent trading strategies. It handles:
- Loading strategy configurations from YAML
- Instantiating and managing strategy instances
- Shared rate limiting across strategies
- Health monitoring and statistics aggregation

Design Principles:
    - **Config-driven**: All strategy params come from YAML files
    - **Multi-strategy**: Run multiple strategies concurrently
    - **Rate limiting**: Token bucket shared across all strategies
    - **Async lifecycle**: Proper async start/stop for all strategies

Architecture Position:
    - Created by V3Coordinator during startup
    - Manages Strategy instances via StrategyRegistry
    - Provides rate_limit_acquire() for strategies to share limits
    - Reports health and stats to health monitor
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

import yaml

from .protocol import Strategy, StrategyContext
from .registry import StrategyRegistry

logger = logging.getLogger("kalshiflow_rl.traderv3.strategies.coordinator")


# Default config directory relative to this file
DEFAULT_CONFIG_DIR = Path(__file__).parent / "config"


@dataclass
class StrategyConfig:
    """
    Configuration for a single strategy loaded from YAML.

    Attributes:
        name: Strategy identifier (must match registry)
        enabled: Whether the strategy should be started
        display_name: Human-readable name
        max_positions: Maximum concurrent positions
        max_exposure_cents: Maximum total exposure in cents
        max_trades_per_minute: Rate limit for this strategy
        params: Strategy-specific parameters
    """
    name: str
    enabled: bool = True
    display_name: str = ""
    max_positions: int = 60
    max_exposure_cents: int = 600000  # $6000 default
    max_trades_per_minute: int = 10
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_data: Dict[str, Any]) -> 'StrategyConfig':
        """
        Create a StrategyConfig from parsed YAML data.

        Args:
            yaml_data: Parsed YAML dictionary

        Returns:
            StrategyConfig instance
        """
        return cls(
            name=yaml_data.get("name", ""),
            enabled=yaml_data.get("enabled", True),
            display_name=yaml_data.get("display_name", yaml_data.get("name", "")),
            max_positions=yaml_data.get("max_positions", 60),
            max_exposure_cents=yaml_data.get("max_exposure_cents", 600000),
            max_trades_per_minute=yaml_data.get("max_trades_per_minute", 10),
            params=yaml_data.get("params", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "display_name": self.display_name,
            "max_positions": self.max_positions,
            "max_exposure_cents": self.max_exposure_cents,
            "max_trades_per_minute": self.max_trades_per_minute,
            "params": self.params,
        }


class TokenBucket:
    """
    Token bucket rate limiter for shared rate limiting.

    Strategies acquire tokens before placing trades. Tokens refill
    at a constant rate up to the maximum capacity.

    Attributes:
        capacity: Maximum tokens in the bucket
        refill_rate: Tokens added per second
        tokens: Current token count
        last_refill: Timestamp of last refill
    """

    def __init__(self, capacity: int = 20, refill_rate: float = 0.333):
        """
        Initialize the token bucket.

        Args:
            capacity: Maximum tokens (default 20 = 20 trades per minute max)
            refill_rate: Tokens per second (default 0.333 = 20 per minute)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False if not enough tokens
        """
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def acquire_blocking(self, tokens: int = 1, timeout: float = 10.0) -> bool:
        """
        Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum seconds to wait

        Returns:
            True if tokens were acquired, False if timeout
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            if await self.acquire(tokens):
                return True
            # Wait for approximately one token to refill
            await asyncio.sleep(1.0 / self.refill_rate)

        return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def get_tokens(self) -> float:
        """Get current token count after refill."""
        self._refill()
        return self.tokens


class StrategyCoordinator:
    """
    Coordinates multiple concurrent trading strategies.

    The coordinator is responsible for:
    - Loading strategy configurations from YAML files
    - Instantiating strategies from the registry
    - Managing strategy lifecycle (start/stop)
    - Providing shared rate limiting
    - Aggregating health and statistics

    Attributes:
        _context: Shared strategy context
        _config_dir: Directory containing YAML configs
        _configs: Dict of strategy name -> StrategyConfig
        _strategies: Dict of strategy name -> Strategy instance
        _rate_limiter: Shared token bucket for rate limiting
        _running: Whether the coordinator is active

    Usage:
        coordinator = StrategyCoordinator(context)
        await coordinator.load_configs()
        await coordinator.start_all()
        ...
        await coordinator.stop_all()
    """

    def __init__(
        self,
        context: StrategyContext,
        config_dir: Optional[Path] = None,
        global_rate_limit: int = 20,  # 20 trades per minute across all strategies
    ):
        """
        Initialize the strategy coordinator.

        Args:
            context: Shared strategy context
            config_dir: Directory containing YAML configs (default: strategies/config/)
            global_rate_limit: Maximum trades per minute across all strategies
        """
        self._context = context
        self._config_dir = config_dir or DEFAULT_CONFIG_DIR
        self._configs: Dict[str, StrategyConfig] = {}
        self._strategies: Dict[str, Strategy] = {}
        self._rate_limiter = TokenBucket(
            capacity=global_rate_limit,
            refill_rate=global_rate_limit / 60.0  # Convert to per-second
        )
        self._running = False
        self._started_at: Optional[float] = None

        logger.info(f"StrategyCoordinator initialized (config_dir={self._config_dir})")

    async def load_configs(self) -> int:
        """
        Load all strategy configurations from YAML files.

        Scans the config directory for .yaml files and loads each one.
        Only loads configs for strategies that are registered.

        Returns:
            Number of configs successfully loaded
        """
        if not self._config_dir.exists():
            logger.warning(f"Config directory does not exist: {self._config_dir}")
            return 0

        loaded = 0

        for config_file in sorted(self._config_dir.glob("*.yaml")):
            try:
                with open(config_file, 'r') as f:
                    yaml_data = yaml.safe_load(f)

                if not yaml_data:
                    logger.warning(f"Empty config file: {config_file}")
                    continue

                config = StrategyConfig.from_yaml(yaml_data)

                # Check if strategy is registered
                if not StrategyRegistry.is_registered(config.name):
                    logger.warning(
                        f"Strategy '{config.name}' not registered, skipping config {config_file}"
                    )
                    continue

                self._configs[config.name] = config
                loaded += 1
                logger.info(
                    f"Loaded config: {config.name} "
                    f"(enabled={config.enabled}, max_positions={config.max_positions})"
                )

            except yaml.YAMLError as e:
                logger.error(f"Failed to parse YAML {config_file}: {e}")
            except Exception as e:
                logger.error(f"Failed to load config {config_file}: {e}")

        logger.info(f"Loaded {loaded} strategy configs from {self._config_dir}")
        return loaded

    async def start_all(self) -> int:
        """
        Start all enabled strategies.

        Instantiates each enabled strategy from the registry and calls
        its start() method with the shared context.

        Returns:
            Number of strategies successfully started
        """
        if self._running:
            logger.warning("StrategyCoordinator already running")
            return 0

        self._running = True
        self._started_at = time.time()
        started = 0

        for name, config in self._configs.items():
            if not config.enabled:
                logger.info(f"Strategy '{name}' disabled, skipping")
                continue

            try:
                # Get strategy class from registry
                strategy_cls = StrategyRegistry.get(name)
                if strategy_cls is None:
                    logger.error(f"Strategy '{name}' not found in registry")
                    continue

                # Instantiate strategy
                strategy = strategy_cls()

                # Create strategy-specific context with config
                strategy_context = replace(self._context, config=config)
                await strategy.start(strategy_context)

                self._strategies[name] = strategy
                started += 1
                logger.info(f"Started strategy: {name} (config: {config.name})")

            except Exception as e:
                logger.error(f"Failed to start strategy '{name}': {e}", exc_info=True)

        logger.info(f"Started {started}/{len(self._configs)} strategies")
        return started

    async def start_strategy(self, name: str) -> bool:
        """
        Start a specific strategy by name.

        Args:
            name: Strategy name to start

        Returns:
            True if strategy was started successfully
        """
        if name in self._strategies:
            logger.warning(f"Strategy '{name}' already running")
            return False

        config = self._configs.get(name)
        if config is None:
            logger.error(f"No config found for strategy '{name}'")
            return False

        try:
            strategy_cls = StrategyRegistry.get(name)
            if strategy_cls is None:
                logger.error(f"Strategy '{name}' not found in registry")
                return False

            strategy = strategy_cls()

            # Create strategy-specific context with config
            strategy_context = replace(self._context, config=config)
            await strategy.start(strategy_context)

            self._strategies[name] = strategy
            logger.info(f"Started strategy: {name} (config loaded: {config.name})")
            return True

        except Exception as e:
            logger.error(f"Failed to start strategy '{name}': {e}", exc_info=True)
            return False

    async def stop_all(self) -> int:
        """
        Stop all running strategies.

        Calls stop() on each strategy in reverse start order.

        Returns:
            Number of strategies successfully stopped
        """
        if not self._running:
            return 0

        stopped = 0

        # Stop in reverse order
        for name in reversed(list(self._strategies.keys())):
            try:
                strategy = self._strategies[name]
                await strategy.stop()
                stopped += 1
                logger.info(f"Stopped strategy: {name}")
            except Exception as e:
                logger.error(f"Failed to stop strategy '{name}': {e}", exc_info=True)

        self._strategies.clear()
        self._running = False
        logger.info(f"Stopped {stopped} strategies")
        return stopped

    async def stop_strategy(self, name: str) -> bool:
        """
        Stop a specific strategy by name.

        Args:
            name: Strategy name to stop

        Returns:
            True if strategy was stopped successfully
        """
        if name not in self._strategies:
            logger.warning(f"Strategy '{name}' not running")
            return False

        try:
            strategy = self._strategies[name]
            await strategy.stop()
            del self._strategies[name]
            logger.info(f"Stopped strategy: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to stop strategy '{name}': {e}", exc_info=True)
            return False

    async def rate_limit_acquire(self, tokens: int = 1) -> bool:
        """
        Acquire rate limit tokens (non-blocking).

        Strategies should call this before placing trades to ensure
        they don't exceed the global rate limit.

        Args:
            tokens: Number of tokens to acquire (usually 1 per trade)

        Returns:
            True if tokens were acquired, False if rate limited
        """
        return await self._rate_limiter.acquire(tokens)

    async def rate_limit_acquire_blocking(
        self,
        tokens: int = 1,
        timeout: float = 10.0
    ) -> bool:
        """
        Acquire rate limit tokens (blocking with timeout).

        Waits for tokens to become available up to the timeout.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum seconds to wait

        Returns:
            True if tokens were acquired, False if timeout
        """
        return await self._rate_limiter.acquire_blocking(tokens, timeout)

    def get_config(self, name: str) -> Optional[StrategyConfig]:
        """
        Get configuration for a specific strategy.

        Args:
            name: Strategy name

        Returns:
            StrategyConfig if found, None otherwise
        """
        return self._configs.get(name)

    def get_all_configs(self) -> Dict[str, StrategyConfig]:
        """
        Get all loaded configurations.

        Returns:
            Dict mapping strategy names to their configs
        """
        return dict(self._configs)

    def list_running(self) -> List[str]:
        """
        List names of currently running strategies.

        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())

    def get_strategy(self, name: str) -> Optional[Strategy]:
        """
        Get a running strategy by name.

        Args:
            name: Strategy name (e.g., "rlm_no")

        Returns:
            Strategy instance if running, None otherwise
        """
        return self._strategies.get(name)

    def is_healthy(self) -> bool:
        """
        Check if coordinator and all strategies are healthy.

        Returns:
            True if running and all strategies are healthy
        """
        if not self._running:
            return False

        for name, strategy in self._strategies.items():
            if not strategy.is_healthy():
                logger.warning(f"Strategy '{name}' is unhealthy")
                return False

        return True

    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics from all strategies.

        Returns:
            Dict with coordinator stats and per-strategy stats
        """
        strategy_stats = {}
        for name, strategy in self._strategies.items():
            try:
                strategy_stats[name] = strategy.get_stats()
            except Exception as e:
                logger.error(f"Failed to get stats from strategy '{name}': {e}")
                strategy_stats[name] = {"error": str(e)}

        return {
            "running": self._running,
            "uptime": time.time() - self._started_at if self._started_at else 0,
            "configs_loaded": len(self._configs),
            "strategies_running": len(self._strategies),
            "rate_limit_tokens": self._rate_limiter.get_tokens(),
            "rate_limit_capacity": self._rate_limiter.capacity,
            "strategies": strategy_stats,
        }

    def get_health_details(self) -> Dict[str, Any]:
        """
        Get detailed health information.

        Returns:
            Dict with health details for coordinator and strategies
        """
        strategy_health = {}
        for name, strategy in self._strategies.items():
            strategy_health[name] = {
                "healthy": strategy.is_healthy(),
                "stats": strategy.get_stats(),
            }

        return {
            "healthy": self.is_healthy(),
            "running": self._running,
            "strategies": strategy_health,
            "rate_limiter": {
                "tokens_available": self._rate_limiter.get_tokens(),
                "capacity": self._rate_limiter.capacity,
                "refill_rate": self._rate_limiter.refill_rate,
            },
        }

    # ========== Trade Processing Aggregation Methods ==========
    # These methods aggregate data across all strategies for WebSocketManager

    def get_trade_processing_stats(self) -> Dict[str, Any]:
        """
        Get aggregated trade processing statistics across all strategies.

        This method is designed for WebSocketManager._broadcast_trade_processing()
        which expects these specific keys:
        - trades_processed: Total trades that passed filters
        - trades_filtered: Total trades filtered out
        - signals_detected: Total signals detected
        - signals_executed: Total signals that resulted in orders
        - signals_skipped: Total signals skipped
        - rate_limited_count: Total rate limited signals
        - reentries: Total re-entry signals

        Returns:
            Dict with aggregated stats from all running strategies
        """
        aggregated = {
            "trades_processed": 0,
            "trades_filtered": 0,
            "signals_detected": 0,
            "signals_executed": 0,
            "signals_skipped": 0,
            "rate_limited_count": 0,
            "reentries": 0,
        }

        for name, strategy in self._strategies.items():
            try:
                stats = strategy.get_stats()
                aggregated["trades_processed"] += stats.get("trades_processed", 0)
                aggregated["trades_filtered"] += stats.get("trades_filtered", 0)
                aggregated["signals_detected"] += stats.get("signals_detected", 0)
                aggregated["signals_executed"] += stats.get("signals_executed", 0)
                aggregated["signals_skipped"] += stats.get("signals_skipped", 0)
                aggregated["rate_limited_count"] += stats.get("rate_limited_count", 0)
                aggregated["reentries"] += stats.get("reentries", 0)
            except Exception as e:
                logger.error(f"Failed to get trade processing stats from '{name}': {e}")

        return aggregated

    def get_recent_tracked_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent tracked trades aggregated across all strategies.

        Combines trades from all strategies, sorts by timestamp (newest first),
        and returns the most recent up to the limit.

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of trade dicts sorted by timestamp (newest first)
        """
        all_trades: List[Dict[str, Any]] = []

        for name, strategy in self._strategies.items():
            try:
                # Check if strategy has get_recent_tracked_trades method
                if hasattr(strategy, 'get_recent_tracked_trades'):
                    trades = strategy.get_recent_tracked_trades(limit=limit)
                    all_trades.extend(trades)
            except Exception as e:
                logger.error(f"Failed to get recent trades from '{name}': {e}")

        # Sort by timestamp (newest first) and limit
        all_trades.sort(key=lambda t: t.get("timestamp", 0), reverse=True)
        return all_trades[:limit]

    def get_decision_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get decision history aggregated across all strategies.

        Combines decisions from all strategies, sorts by timestamp (newest first),
        and returns the most recent up to the limit.

        Args:
            limit: Maximum number of decisions to return

        Returns:
            List of decision dicts sorted by timestamp (newest first)
        """
        all_decisions: List[Dict[str, Any]] = []

        for name, strategy in self._strategies.items():
            try:
                # Check if strategy has get_decision_history method
                if hasattr(strategy, 'get_decision_history'):
                    decisions = strategy.get_decision_history(limit=limit)
                    all_decisions.extend(decisions)
            except Exception as e:
                logger.error(f"Failed to get decision history from '{name}': {e}")

        # Sort by timestamp (newest first) and limit
        all_decisions.sort(key=lambda d: d.get("timestamp", 0), reverse=True)
        return all_decisions[:limit]

    def get_market_states(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get market states aggregated across all strategies.

        Combines market states from all strategies, sorts by trade count
        (most active first), and returns up to the limit.

        Args:
            limit: Maximum number of market states to return

        Returns:
            List of market state dicts sorted by activity
        """
        all_states: List[Dict[str, Any]] = []

        for name, strategy in self._strategies.items():
            try:
                # Check if strategy has get_market_states method
                if hasattr(strategy, 'get_market_states'):
                    states = strategy.get_market_states(limit=limit)
                    all_states.extend(states)
            except Exception as e:
                logger.error(f"Failed to get market states from '{name}': {e}")

        # Sort by total_trades (most active first) and limit
        all_states.sort(key=lambda s: s.get("total_trades", 0), reverse=True)
        return all_states[:limit]
