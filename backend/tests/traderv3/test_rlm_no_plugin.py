"""
Tests for RLM NO strategy plugin config loading.

Critical tests to prevent regression of the config loading bug where
start_all() wasn't passing config to strategies.
"""

import pytest
from dataclasses import replace
from unittest.mock import MagicMock, AsyncMock, patch

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from kalshiflow_rl.traderv3.strategies.coordinator import StrategyCoordinator, StrategyConfig
from kalshiflow_rl.traderv3.strategies.protocol import StrategyContext
from kalshiflow_rl.traderv3.strategies.plugins.rlm_no import RLMNoStrategy


class TestRLMNoConfigLoading:
    """Tests that config is properly passed to RLM NO plugin."""

    def _create_mock_context(self, config=None):
        """Create a real StrategyContext with mocked services."""
        mock_event_bus = AsyncMock()
        mock_event_bus.subscribe_to_public_trade = AsyncMock()
        mock_event_bus.subscribe_to_market_determined = AsyncMock()
        mock_event_bus.subscribe_to_tmo_fetched = AsyncMock()
        mock_event_bus.emit_system_activity = AsyncMock()
        mock_event_bus.unsubscribe = MagicMock()

        mock_state_container = MagicMock()
        mock_state_container.machine_state = "READY"

        mock_orderbook_integration = MagicMock()

        # Create a REAL StrategyContext dataclass (not a mock)
        # This allows replace() to work correctly
        context = StrategyContext(
            event_bus=mock_event_bus,
            trading_service=None,
            state_container=mock_state_container,
            orderbook_integration=mock_orderbook_integration,
            tracked_markets=None,
            config=config,
        )
        return context

    @pytest.mark.asyncio
    async def test_strategy_uses_config_when_provided(self):
        """When config is provided, strategy should use config values not defaults."""
        strategy = RLMNoStrategy()

        # Verify defaults before start
        assert strategy._contracts_per_trade == 100  # hardcoded default
        assert strategy._yes_threshold == 0.65  # hardcoded default

        # Create config with YAML values
        config = StrategyConfig(
            name="rlm_no",
            enabled=True,
            params={
                "contracts_per_trade": 10,
                "yes_threshold": 0.70,
                "min_trades": 25,
                "min_price_drop": 5,
                "max_concurrent": 5,
            }
        )

        mock_context = self._create_mock_context(config=config)
        await strategy.start(mock_context)

        # Verify config values were loaded
        assert strategy._contracts_per_trade == 10, "Should use config value 10, not default 100"
        assert strategy._yes_threshold == 0.70, "Should use config value 0.70, not default 0.65"
        assert strategy._min_trades == 25, "Should use config value 25, not default 15"
        assert strategy._min_price_drop == 5, "Should use config value 5, not default 0"
        assert strategy._max_concurrent == 5, "Should use config value 5, not default 1000"

        await strategy.stop()

    @pytest.mark.asyncio
    async def test_strategy_uses_defaults_when_no_config(self):
        """When no config provided, strategy should use hardcoded defaults."""
        strategy = RLMNoStrategy()

        mock_context = self._create_mock_context(config=None)
        await strategy.start(mock_context)

        # Verify defaults are used
        assert strategy._contracts_per_trade == 100, "Should use default 100"
        assert strategy._yes_threshold == 0.65, "Should use default 0.65"
        assert strategy._min_trades == 15, "Should use default 15"
        assert strategy._min_price_drop == 0, "Should use default 0"
        assert strategy._max_concurrent == 1000, "Should use default 1000"

        await strategy.stop()

    @pytest.mark.asyncio
    async def test_all_yaml_params_are_loaded(self):
        """All YAML params should be loaded, not just some."""
        strategy = RLMNoStrategy()

        config = StrategyConfig(
            name="rlm_no",
            params={
                "yes_threshold": 0.70,
                "min_trades": 25,
                "min_price_drop": 5,
                "contracts_per_trade": 10,
                "max_concurrent": 5,
                "allow_reentry": False,
                "orderbook_timeout": 3.0,
                "tight_spread": 3,
                "normal_spread": 6,
                "max_spread": 12,
                "min_hours_to_settlement": 0.5,
                "max_days_to_settlement": 7,
            }
        )

        mock_context = self._create_mock_context(config=config)
        await strategy.start(mock_context)

        # Verify ALL params loaded
        assert strategy._yes_threshold == 0.70
        assert strategy._min_trades == 25
        assert strategy._min_price_drop == 5
        assert strategy._contracts_per_trade == 10
        assert strategy._max_concurrent == 5
        assert strategy._allow_reentry is False
        assert strategy._orderbook_timeout == 3.0
        assert strategy._tight_spread == 3
        assert strategy._normal_spread == 6
        assert strategy._max_spread == 12
        assert strategy._min_hours_to_settlement == 0.5
        assert strategy._max_days_to_settlement == 7

        await strategy.stop()

    @pytest.mark.asyncio
    async def test_coordinator_start_all_passes_config(self):
        """Critical: start_all() must pass config to strategies."""
        # Create base mock context without config
        base_context = self._create_mock_context(config=None)

        # Create coordinator
        coordinator = StrategyCoordinator(base_context)

        # Manually add a config (simulating load_configs)
        config = StrategyConfig(
            name="rlm_no",
            enabled=True,
            params={"contracts_per_trade": 10, "yes_threshold": 0.70}
        )
        coordinator._configs["rlm_no"] = config

        # Start all strategies
        started = await coordinator.start_all()
        assert started == 1, "Should have started 1 strategy"

        # Verify the strategy received config
        strategy = coordinator.get_strategy("rlm_no")
        assert strategy is not None, "Strategy should be running"
        assert strategy._contracts_per_trade == 10, "Config should have been passed (10 not 100)"
        assert strategy._yes_threshold == 0.70, "Config should have been passed (0.70 not 0.65)"

        await coordinator.stop_all()

    @pytest.mark.asyncio
    async def test_coordinator_start_strategy_passes_config(self):
        """start_strategy() should also pass config to the strategy."""
        base_context = self._create_mock_context(config=None)

        coordinator = StrategyCoordinator(base_context)

        config = StrategyConfig(
            name="rlm_no",
            enabled=True,
            params={"contracts_per_trade": 15, "yes_threshold": 0.75}
        )
        coordinator._configs["rlm_no"] = config

        # Start single strategy
        success = await coordinator.start_strategy("rlm_no")
        assert success is True

        strategy = coordinator.get_strategy("rlm_no")
        assert strategy is not None
        assert strategy._contracts_per_trade == 15
        assert strategy._yes_threshold == 0.75

        await coordinator.stop_all()


class TestRLMNoStats:
    """Tests for stats reporting format compatibility with WebSocketManager."""

    @pytest.mark.asyncio
    async def test_get_stats_returns_required_keys(self):
        """get_stats() must return keys expected by WebSocketManager."""
        strategy = RLMNoStrategy()

        config = StrategyConfig(name="rlm_no", params={})

        mock_event_bus = AsyncMock()
        mock_event_bus.subscribe_to_public_trade = AsyncMock()
        mock_event_bus.subscribe_to_market_determined = AsyncMock()
        mock_event_bus.subscribe_to_tmo_fetched = AsyncMock()
        mock_event_bus.emit_system_activity = AsyncMock()
        mock_event_bus.unsubscribe = MagicMock()

        context = StrategyContext(
            event_bus=mock_event_bus,
            trading_service=None,
            state_container=MagicMock(),
            orderbook_integration=MagicMock(),
            tracked_markets=None,
            config=config,
        )

        await strategy.start(context)

        stats = strategy.get_stats()

        # Required keys for WebSocketManager._broadcast_trade_processing()
        required_keys = [
            "trades_processed",
            "trades_filtered",
            "signals_detected",
            "signals_executed",
            "signals_skipped",
            "rate_limited_count",
            "reentries",
        ]

        for key in required_keys:
            assert key in stats, f"Missing required key: {key}"

        await strategy.stop()
