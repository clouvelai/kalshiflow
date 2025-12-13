"""
Service Container for Kalshi Trading Actor MVP.

Provides dependency injection container to replace global singleton anti-patterns.
Manages service lifecycle, initialization order, and testing isolation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, TypeVar, Generic, Type, Callable, List, Set
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("kalshiflow_rl.service_container")

T = TypeVar('T')


class ServiceLifecycle(Enum):
    """Service lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"
    FAILED = "failed"


@dataclass
class ServiceDefinition:
    """Definition of a service in the container."""
    service_type: Type
    factory: Callable[..., Any]
    dependencies: List[str] = field(default_factory=list)
    singleton: bool = True
    lifecycle: ServiceLifecycle = ServiceLifecycle.UNINITIALIZED
    instance: Optional[Any] = None
    initialization_order: int = 0
    

class ServiceContainer:
    """
    Dependency injection container for trading services.
    
    Features:
    - Service registration with dependency declarations
    - Automatic dependency resolution and initialization order
    - Singleton and transient service lifetimes
    - Graceful shutdown with reverse dependency order
    - Testing isolation via container instances
    - Circular dependency detection
    - Async service lifecycle management
    
    Architecture:
    - Replaces global singleton pattern with explicit DI
    - Makes dependencies visible and testable
    - Proper initialization order management
    - Clean shutdown handling
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize service container.
        
        Args:
            name: Container name for logging and identification
        """
        self.name = name
        self._services: Dict[str, ServiceDefinition] = {}
        self._instances: Dict[str, Any] = {}
        self._initialization_order: List[str] = []
        self._initialized_services: Set[str] = set()
        self._lock = asyncio.Lock()
        self._is_shutting_down = False
        
        logger.info(f"ServiceContainer '{name}' initialized")
    
    def register_singleton(
        self,
        service_name: str,
        service_type: Type[T],
        factory: Callable[..., T],
        dependencies: Optional[List[str]] = None,
        initialization_order: int = 0
    ) -> None:
        """
        Register a singleton service.
        
        Args:
            service_name: Unique service identifier
            service_type: Service class/type
            factory: Factory function to create service instance
            dependencies: List of service names this depends on
            initialization_order: Lower numbers initialize first
        """
        if service_name in self._services:
            raise ValueError(f"Service '{service_name}' already registered")
        
        self._services[service_name] = ServiceDefinition(
            service_type=service_type,
            factory=factory,
            dependencies=dependencies or [],
            singleton=True,
            initialization_order=initialization_order
        )
        
        logger.debug(f"Registered singleton service: {service_name}")
    
    def register_transient(
        self,
        service_name: str,
        service_type: Type[T],
        factory: Callable[..., T],
        dependencies: Optional[List[str]] = None
    ) -> None:
        """
        Register a transient service (new instance each time).
        
        Args:
            service_name: Unique service identifier  
            service_type: Service class/type
            factory: Factory function to create service instance
            dependencies: List of service names this depends on
        """
        if service_name in self._services:
            raise ValueError(f"Service '{service_name}' already registered")
        
        self._services[service_name] = ServiceDefinition(
            service_type=service_type,
            factory=factory,
            dependencies=dependencies or [],
            singleton=False
        )
        
        logger.debug(f"Registered transient service: {service_name}")
    
    async def get_service(self, service_name: str) -> Any:
        """
        Get a service instance by name.
        
        Args:
            service_name: Service identifier
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service not registered
            RuntimeError: If circular dependency detected
        """
        if self._is_shutting_down:
            raise RuntimeError("Container is shutting down, cannot get services")
        
        if service_name not in self._services:
            raise ValueError(f"Service '{service_name}' not registered")
        
        service_def = self._services[service_name]
        
        # For transient services, always create new instance
        if not service_def.singleton:
            return await self._create_service_instance(service_name)
        
        # For singletons, use cached instance if available
        async with self._lock:
            if service_def.instance is not None:
                return service_def.instance
        
        # Create singleton instance outside the lock to avoid deadlock
        instance = await self._create_service_instance(service_name)
        
        async with self._lock:
            # Double-check pattern in case another coroutine created it
            if service_def.instance is not None:
                return service_def.instance
            
            service_def.instance = instance
            self._instances[service_name] = instance
            
            return instance
    
    async def _create_service_instance(self, service_name: str) -> Any:
        """Create service instance with dependency injection."""
        service_def = self._services[service_name]
        
        # Mark as initializing to detect circular dependencies
        if service_def.lifecycle == ServiceLifecycle.INITIALIZING:
            raise RuntimeError(f"Circular dependency detected involving '{service_name}'")
        
        service_def.lifecycle = ServiceLifecycle.INITIALIZING
        
        try:
            # Resolve dependencies first
            dependency_instances = {}
            for dep_name in service_def.dependencies:
                if dep_name not in self._services:
                    raise ValueError(f"Dependency '{dep_name}' not registered for service '{service_name}'")
                
                # Get dependency without holding any locks to avoid deadlock
                dependency_instances[dep_name] = await self.get_service(dep_name)
            
            # Create service instance
            logger.debug(f"Creating service instance: {service_name}")
            
            if asyncio.iscoroutinefunction(service_def.factory):
                instance = await service_def.factory(self, **dependency_instances)
            else:
                instance = service_def.factory(self, **dependency_instances)
            
            # Initialize if the service has an initialize method
            if hasattr(instance, 'initialize') and callable(instance.initialize):
                if asyncio.iscoroutinefunction(instance.initialize):
                    await instance.initialize()
                else:
                    instance.initialize()
            
            service_def.lifecycle = ServiceLifecycle.INITIALIZED
            self._initialized_services.add(service_name)
            
            logger.info(f"✅ Service '{service_name}' created and initialized")
            return instance
            
        except Exception as e:
            service_def.lifecycle = ServiceLifecycle.FAILED
            logger.error(f"❌ Failed to create service '{service_name}': {e}")
            raise
    
    async def initialize_all(self) -> None:
        """
        Initialize all registered services in dependency order.
        
        Services are initialized in order of their initialization_order value,
        with dependencies resolved automatically.
        """
        logger.info(f"Initializing all services in container '{self.name}'...")
        
        # Sort services by initialization order
        service_order = sorted(
            self._services.items(),
            key=lambda item: item[1].initialization_order
        )
        
        # Initialize each service
        for service_name, service_def in service_order:
            if service_def.singleton and service_name not in self._initialized_services:
                await self.get_service(service_name)
        
        logger.info(f"✅ All services initialized in container '{self.name}'")
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown all services in reverse dependency order.
        """
        logger.info(f"Shutting down container '{self.name}'...")
        
        self._is_shutting_down = True
        
        # Shutdown services in reverse initialization order
        shutdown_order = sorted(
            self._initialized_services,
            key=lambda name: self._services[name].initialization_order,
            reverse=True
        )
        
        for service_name in shutdown_order:
            await self._shutdown_service(service_name)
        
        # Clear state
        self._instances.clear()
        self._initialized_services.clear()
        self._is_shutting_down = False
        
        logger.info(f"✅ Container '{self.name}' shutdown complete")
    
    async def _shutdown_service(self, service_name: str) -> None:
        """Shutdown a specific service."""
        try:
            service_def = self._services[service_name]
            service_def.lifecycle = ServiceLifecycle.SHUTTING_DOWN
            
            instance = self._instances.get(service_name)
            if instance and hasattr(instance, 'shutdown') and callable(instance.shutdown):
                logger.debug(f"Shutting down service: {service_name}")
                
                if asyncio.iscoroutinefunction(instance.shutdown):
                    await instance.shutdown()
                else:
                    instance.shutdown()
            
            service_def.lifecycle = ServiceLifecycle.SHUTDOWN
            service_def.instance = None
            
            logger.debug(f"✅ Service '{service_name}' shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error shutting down service '{service_name}': {e}")
    
    def is_registered(self, service_name: str) -> bool:
        """Check if a service is registered."""
        return service_name in self._services
    
    def get_service_info(self, service_name: str) -> Dict[str, Any]:
        """Get information about a service."""
        if service_name not in self._services:
            raise ValueError(f"Service '{service_name}' not registered")
        
        service_def = self._services[service_name]
        return {
            "name": service_name,
            "type": service_def.service_type.__name__,
            "singleton": service_def.singleton,
            "dependencies": service_def.dependencies,
            "lifecycle": service_def.lifecycle.value,
            "initialization_order": service_def.initialization_order,
            "initialized": service_name in self._initialized_services
        }
    
    def get_container_status(self) -> Dict[str, Any]:
        """Get container status and all service information."""
        services_info = {}
        for service_name in self._services:
            try:
                services_info[service_name] = self.get_service_info(service_name)
            except Exception as e:
                services_info[service_name] = {"error": str(e)}
        
        return {
            "container_name": self.name,
            "is_shutting_down": self._is_shutting_down,
            "total_services": len(self._services),
            "initialized_services": len(self._initialized_services),
            "services": services_info
        }


# Global container instance (this is the ONLY acceptable global)
# Unlike service singletons, the container itself should be global for convenience
_default_container: Optional[ServiceContainer] = None
_container_lock = asyncio.Lock()


async def get_default_container() -> ServiceContainer:
    """
    Get the default service container.
    
    This is the only acceptable global singleton - the container itself.
    All services should be managed through explicit dependency injection.
    """
    global _default_container
    
    async with _container_lock:
        if _default_container is None:
            _default_container = ServiceContainer("default")
        
        return _default_container


async def shutdown_default_container() -> None:
    """Shutdown the default container."""
    global _default_container
    
    async with _container_lock:
        if _default_container:
            await _default_container.shutdown()
            _default_container = None


@asynccontextmanager
async def create_test_container(name: str = "test"):
    """
    Create an isolated container for testing.
    
    This provides proper testing isolation without global state pollution.
    
    Args:
        name: Container name for identification
        
    Yields:
        ServiceContainer: Isolated container instance for testing
    """
    container = ServiceContainer(name)
    try:
        yield container
    finally:
        await container.shutdown()


# Convenience functions for default container
async def register_trading_services() -> ServiceContainer:
    """
    Register all trading-related services in the default container.
    
    This replaces the global singleton initialization functions.
    """
    container = await get_default_container()
    
    # Import factories here to avoid circular dependencies
    from .service_factories import (
        create_orderbook_state_registry,
        create_live_observation_adapter,
        create_actor_service,
        create_event_bus
    )
    
    # Register services in dependency order
    # 1. EventBus (no dependencies)
    container.register_singleton(
        service_name="event_bus",
        service_type=object,  # EventBus type from event_bus module
        factory=create_event_bus,
        dependencies=[],
        initialization_order=10
    )
    
    # 2. OrderbookStateRegistry (no dependencies)
    container.register_singleton(
        service_name="orderbook_state_registry", 
        service_type=object,  # OrderbookStateRegistry class
        factory=create_orderbook_state_registry,
        dependencies=[],
        initialization_order=20
    )
    
    # 3. LiveObservationAdapter (depends on orderbook registry)
    container.register_singleton(
        service_name="live_observation_adapter",
        service_type=object,  # LiveObservationAdapter type
        factory=create_live_observation_adapter,
        dependencies=["orderbook_state_registry"],
        initialization_order=30
    )
    
    # 4. ActorService (depends on adapter and event bus)
    container.register_singleton(
        service_name="actor_service",
        service_type=object,  # ActorService type
        factory=create_actor_service,
        dependencies=["live_observation_adapter", "event_bus"],
        initialization_order=40
    )
    
    logger.info("✅ All trading services registered in default container")
    return container