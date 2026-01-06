"""
Strategy Registry for TRADER V3 Plugin System.

This module provides the central registry for discovering and managing
trading strategy plugins. Strategies register themselves using the
@register decorator.

Usage:
    # In a strategy plugin file:
    from ..registry import StrategyRegistry

    @StrategyRegistry.register("rlm_no")
    class RLMNoStrategy:
        name = "rlm_no"
        display_name = "Reverse Line Movement NO"
        ...

    # In the coordinator:
    strategy_class = StrategyRegistry.get("rlm_no")
    strategy = strategy_class()
    await strategy.start(context)

Design Principles:
    - **Class-level registry**: No instantiation needed
    - **Decorator-based registration**: Simple plugin API
    - **Type-safe**: Validates Strategy protocol compliance
    - **Discovery-friendly**: list_all() for UI display
"""

import logging
from typing import Dict, Type, List, Optional

from .protocol import Strategy

logger = logging.getLogger("kalshiflow_rl.traderv3.strategies.registry")


class StrategyRegistry:
    """
    Central registry for trading strategy plugins.

    Strategies register themselves using the @register decorator.
    The registry maintains a class-level dictionary of strategy
    name -> strategy class mappings.

    Class Attributes:
        _strategies: Dict mapping strategy names to their classes

    Class Methods:
        register(name): Decorator to register a strategy class
        get(name): Get a strategy class by name
        list_all(): List all registered strategy names
        get_all(): Get all registered strategy classes

    Example:
        @StrategyRegistry.register("my_strategy")
        class MyStrategy:
            name = "my_strategy"
            ...

        # Later:
        cls = StrategyRegistry.get("my_strategy")
        strategy = cls()
    """

    _strategies: Dict[str, Type[Strategy]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a strategy class.

        The decorated class must implement the Strategy protocol.

        Args:
            name: Unique strategy identifier

        Returns:
            Decorator function

        Example:
            @StrategyRegistry.register("rlm_no")
            class RLMNoStrategy:
                name = "rlm_no"
                display_name = "Reverse Line Movement NO"
                subscribed_events = {EventType.PUBLIC_TRADE_RECEIVED}
                ...
        """
        def decorator(strategy_cls: Type[Strategy]) -> Type[Strategy]:
            # Validate the class has required attributes
            if not hasattr(strategy_cls, 'name'):
                raise ValueError(
                    f"Strategy class {strategy_cls.__name__} missing required 'name' attribute"
                )
            if not hasattr(strategy_cls, 'display_name'):
                raise ValueError(
                    f"Strategy class {strategy_cls.__name__} missing required 'display_name' attribute"
                )
            if not hasattr(strategy_cls, 'subscribed_events'):
                raise ValueError(
                    f"Strategy class {strategy_cls.__name__} missing required 'subscribed_events' attribute"
                )

            # Validate required methods exist
            required_methods = ['start', 'stop', 'is_healthy', 'get_stats']
            for method_name in required_methods:
                if not hasattr(strategy_cls, method_name):
                    raise ValueError(
                        f"Strategy class {strategy_cls.__name__} missing required method '{method_name}'"
                    )
                if not callable(getattr(strategy_cls, method_name)):
                    raise ValueError(
                        f"Strategy class {strategy_cls.__name__}.{method_name} must be callable"
                    )

            # Check for duplicate registration
            if name in cls._strategies:
                existing = cls._strategies[name]
                logger.warning(
                    f"Strategy '{name}' already registered by {existing.__name__}, "
                    f"overwriting with {strategy_cls.__name__}"
                )

            # Register the strategy
            cls._strategies[name] = strategy_cls
            logger.info(f"Registered strategy: {name} -> {strategy_cls.__name__}")

            return strategy_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[Strategy]]:
        """
        Get a strategy class by name.

        Args:
            name: Strategy identifier

        Returns:
            Strategy class if found, None otherwise
        """
        return cls._strategies.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        """
        List all registered strategy names.

        Returns:
            List of strategy names in registration order
        """
        return list(cls._strategies.keys())

    @classmethod
    def get_all(cls) -> Dict[str, Type[Strategy]]:
        """
        Get all registered strategy classes.

        Returns:
            Dict mapping names to strategy classes
        """
        return dict(cls._strategies)

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered strategies.

        Primarily for testing purposes.
        """
        cls._strategies.clear()
        logger.debug("Strategy registry cleared")

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a strategy is registered.

        Args:
            name: Strategy identifier

        Returns:
            True if strategy is registered
        """
        return name in cls._strategies
