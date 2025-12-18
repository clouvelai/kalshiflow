"""
Service factories for dependency injection.

Contains factory functions to create service instances with proper dependency injection.
Replaces global singleton initialization patterns.
"""

import logging
from typing import Any, Dict, Optional, List

logger = logging.getLogger("kalshiflow_rl.service_factories")


# ==============================================================================
# OrderbookStateRegistry Factory
# ==============================================================================

class OrderbookStateRegistry:
    """
    Registry for SharedOrderbookState instances.
    
    Replaces the global _orderbook_states dictionary with a proper service.
    """
    
    def __init__(self):
        self._states: Dict[str, Any] = {}  # Will hold SharedOrderbookState instances
        self._lock = None  # Will be set during initialize
        
    async def initialize(self):
        """Initialize the registry."""
        import asyncio
        self._lock = asyncio.Lock()
        logger.info("OrderbookStateRegistry initialized")
    
    async def get_shared_orderbook_state(self, market_ticker: str):
        """
        Get or create SharedOrderbookState for a market.
        
        Args:
            market_ticker: Market ticker
            
        Returns:
            SharedOrderbookState instance
        """
        # Import here to avoid circular dependencies
        from ..data.orderbook_state import SharedOrderbookState
        
        async with self._lock:
            if market_ticker not in self._states:
                self._states[market_ticker] = SharedOrderbookState(market_ticker)
                logger.info(f"Created SharedOrderbookState for {market_ticker}")
            
            return self._states[market_ticker]
    
    async def get_all_states(self) -> Dict[str, Any]:
        """Get all registered orderbook states."""
        async with self._lock:
            return dict(self._states)
    
    async def cleanup(self):
        """Clean up all states."""
        async with self._lock:
            self._states.clear()
        logger.info("OrderbookStateRegistry cleanup complete")


async def create_orderbook_state_registry(container: Any) -> OrderbookStateRegistry:
    """
    Factory function to create OrderbookStateRegistry.
    
    Args:
        container: ServiceContainer instance
        
    Returns:
        OrderbookStateRegistry instance
    """
    logger.info("Creating OrderbookStateRegistry...")
    registry = OrderbookStateRegistry()
    # Note: initialize() will be called automatically by ServiceContainer
    return registry


# ==============================================================================
# LiveObservationAdapter Factory  
# ==============================================================================

async def create_live_observation_adapter(
    container: Any,
    orderbook_state_registry: OrderbookStateRegistry
) -> Any:
    """
    Factory function to create LiveObservationAdapter with injected dependencies.
    
    Args:
        container: ServiceContainer instance
        orderbook_state_registry: Injected orderbook registry
        
    Returns:
        LiveObservationAdapter instance with dependencies injected
    """
    logger.info("Creating LiveObservationAdapter with dependency injection...")
    
    # Import here to avoid circular dependencies
    from .live_observation_adapter import LiveObservationAdapter
    
    # Create adapter instance
    adapter = LiveObservationAdapter(
        window_size=10,
        max_markets=1,
        temporal_context_minutes=30
    )
    
    # Inject the orderbook state registry
    adapter._orderbook_state_registry = orderbook_state_registry
    
    logger.info("✅ LiveObservationAdapter created with injected dependencies")
    return adapter


# ==============================================================================
# ActorService Factory
# ==============================================================================

async def create_actor_service(
    container: Any,
    live_observation_adapter: Any,
    event_bus: Any
) -> Any:
    """
    Factory function to create ActorService with injected dependencies.
    
    Args:
        container: ServiceContainer instance
        live_observation_adapter: Injected observation adapter
        event_bus: Injected event bus
        
    Returns:
        ActorService instance with dependencies injected
    """
    logger.info("Creating ActorService with dependency injection...")
    
    # Import here to avoid circular dependencies
    from .actor_service import ActorService
    from ..config import config
    
    # Create actor service instance with strict_validation=False to allow
    # services to be created without all dependencies (they can be set later)
    actor_service = ActorService(
        market_tickers=config.RL_MARKET_TICKERS,
        model_path=None,  # Can be configured later
        queue_size=1000,
        throttle_ms=250,
        strict_validation=False  # Allow creation without all dependencies
    )
    
    # Inject dependencies through setter methods
    if hasattr(live_observation_adapter, 'build_observation'):
        # Create adapter function that wraps the injected adapter
        async def observation_adapter_fn(market_ticker: str):
            return await live_observation_adapter.build_observation(market_ticker)
        
        actor_service.set_observation_adapter(observation_adapter_fn)
    
    # Store references to injected dependencies
    actor_service._injected_adapter = live_observation_adapter
    actor_service._injected_event_bus = event_bus
    
    # Set default action selector if not already set (HardcodedSelector as fallback)
    if not actor_service._action_selector:
        from .hardcoded_policies import HardcodedSelector
        default_selector = HardcodedSelector()
        actor_service.set_action_selector(default_selector)
        
        # Register with order manager if available for mutual exclusivity
        if order_manager and hasattr(order_manager, 'register_action_selector'):
            order_manager.register_action_selector(default_selector)
            logger.debug("Set and registered default HardcodedSelector with order manager")
        else:
            logger.debug("Set default HardcodedSelector for ActorService (order manager not available for registration)")
    
    logger.info("✅ ActorService created with injected dependencies")
    return actor_service


# ==============================================================================
# EventBus Factory
# ==============================================================================

async def create_event_bus(container: Any) -> Any:
    """
    Factory function to create EventBus.
    
    Args:
        container: ServiceContainer instance
        
    Returns:
        EventBus instance
    """
    logger.info("Creating EventBus...")
    
    # Import here to avoid circular dependencies
    from .event_bus import EventBus
    
    # EventBus can remain a singleton since it's naturally stateless
    # and doesn't create the same testing issues as other singletons
    from .event_bus import get_event_bus
    
    event_bus = await get_event_bus()
    
    logger.info("✅ EventBus created")
    return event_bus


# ==============================================================================
# Integration helpers
# ==============================================================================

def create_backward_compatible_accessor(service_name: str):
    """
    Create a backward-compatible accessor function.
    
    This allows gradual migration from global singletons to DI.
    
    Args:
        service_name: Name of service in container
        
    Returns:
        Async function that gets service from default container
    """
    async def accessor():
        from .service_container import get_default_container
        container = await get_default_container()
        return await container.get_service(service_name)
    
    return accessor


# Convenience accessors for backward compatibility during migration
get_injected_orderbook_state_registry = create_backward_compatible_accessor("orderbook_state_registry")
get_injected_live_observation_adapter = create_backward_compatible_accessor("live_observation_adapter") 
get_injected_actor_service = create_backward_compatible_accessor("actor_service")