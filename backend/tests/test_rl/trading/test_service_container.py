"""
Tests for ServiceContainer dependency injection.

Tests the removal of global singleton anti-patterns and proper dependency injection.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from kalshiflow_rl.trading.service_container import (
    ServiceContainer, 
    ServiceDefinition,
    ServiceLifecycle,
    create_test_container,
    get_default_container,
    shutdown_default_container
)


class TestServiceContainer:
    """Test dependency injection container functionality."""
    
    @pytest.mark.asyncio
    async def test_container_initialization(self):
        """Test basic container initialization."""
        async with create_test_container("test") as container:
            assert container.name == "test"
            assert len(container._services) == 0
            assert len(container._instances) == 0
            assert not container._is_shutting_down
    
    @pytest.mark.asyncio
    async def test_singleton_service_registration(self):
        """Test singleton service registration."""
        async with create_test_container() as container:
            
            def mock_factory(container, **deps):
                return MagicMock()
            
            container.register_singleton(
                service_name="test_service",
                service_type=MagicMock,
                factory=mock_factory,
                dependencies=[],
                initialization_order=10
            )
            
            assert container.is_registered("test_service")
            
            service_info = container.get_service_info("test_service")
            assert service_info["name"] == "test_service"
            assert service_info["singleton"] is True
            assert service_info["dependencies"] == []
            assert service_info["initialization_order"] == 10
    
    @pytest.mark.asyncio
    async def test_dependency_injection(self):
        """Test dependency injection between services."""
        async with create_test_container() as container:
            
            # Create mock services
            dependency_instance = MagicMock()
            service_instance = MagicMock()
            
            def dependency_factory(container, **deps):
                return dependency_instance
            
            def service_factory(container, dependency_service=None, **deps):
                service_instance.dependency = dependency_service
                return service_instance
            
            # Register dependency first
            container.register_singleton(
                service_name="dependency_service",
                service_type=MagicMock,
                factory=dependency_factory,
                dependencies=[],
                initialization_order=10
            )
            
            # Register service that depends on dependency
            container.register_singleton(
                service_name="main_service", 
                service_type=MagicMock,
                factory=service_factory,
                dependencies=["dependency_service"],
                initialization_order=20
            )
            
            # Get the main service - should trigger dependency injection
            main_service = await container.get_service("main_service")
            
            assert main_service is service_instance
            assert main_service.dependency is dependency_instance
    
    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        async with create_test_container() as container:
            
            def factory_a(container, service_b=None, **deps):
                return MagicMock()
            
            def factory_b(container, service_a=None, **deps):
                return MagicMock()
            
            container.register_singleton(
                service_name="service_a",
                service_type=MagicMock, 
                factory=factory_a,
                dependencies=["service_b"]
            )
            
            container.register_singleton(
                service_name="service_b",
                service_type=MagicMock,
                factory=factory_b, 
                dependencies=["service_a"]
            )
            
            # Should raise RuntimeError for circular dependency
            with pytest.raises(RuntimeError, match="Circular dependency detected"):
                await container.get_service("service_a")
    
    @pytest.mark.asyncio
    async def test_initialization_order(self):
        """Test service initialization order."""
        async with create_test_container() as container:
            
            initialization_order = []
            
            def factory_1(container, **deps):
                initialization_order.append(1)
                return MagicMock()
            
            def factory_2(container, **deps):
                initialization_order.append(2)
                return MagicMock()
            
            def factory_3(container, **deps):
                initialization_order.append(3)
                return MagicMock()
            
            # Register in reverse order but with explicit initialization_order
            container.register_singleton("service_3", MagicMock, factory_3, [], 30)
            container.register_singleton("service_1", MagicMock, factory_1, [], 10)
            container.register_singleton("service_2", MagicMock, factory_2, [], 20)
            
            # Initialize all - should respect initialization_order
            await container.initialize_all()
            
            assert initialization_order == [1, 2, 3]
    
    @pytest.mark.asyncio 
    async def test_transient_services(self):
        """Test transient (non-singleton) services."""
        async with create_test_container() as container:
            
            call_count = 0
            
            def factory(container, **deps):
                nonlocal call_count
                call_count += 1
                return MagicMock(id=call_count)
            
            container.register_transient(
                service_name="transient_service",
                service_type=MagicMock,
                factory=factory
            )
            
            # Get service multiple times
            service1 = await container.get_service("transient_service")
            service2 = await container.get_service("transient_service")
            service3 = await container.get_service("transient_service")
            
            # Should be different instances
            assert service1 is not service2
            assert service2 is not service3
            assert service1.id == 1
            assert service2.id == 2
            assert service3.id == 3
    
    @pytest.mark.asyncio
    async def test_async_service_lifecycle(self):
        """Test async initialize/shutdown lifecycle."""
        async with create_test_container() as container:
            
            mock_service = MagicMock()
            mock_service.initialize = AsyncMock()
            mock_service.shutdown = AsyncMock()
            
            def factory(container, **deps):
                return mock_service
            
            container.register_singleton(
                service_name="lifecycle_service",
                service_type=MagicMock,
                factory=factory
            )
            
            # Get service - should call initialize
            service = await container.get_service("lifecycle_service")
            assert service is mock_service
            mock_service.initialize.assert_called_once()
            
            # Shutdown container - should call shutdown
            await container.shutdown()
            mock_service.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_container_status(self):
        """Test container status reporting."""
        async with create_test_container("status_test") as container:
            
            container.register_singleton(
                service_name="test_service",
                service_type=MagicMock,
                factory=lambda container, **deps: MagicMock()
            )
            
            # Before initialization
            status = container.get_container_status()
            assert status["container_name"] == "status_test"
            assert status["total_services"] == 1
            assert status["initialized_services"] == 0
            assert not status["is_shutting_down"]
            
            # After getting service
            await container.get_service("test_service")
            
            status = container.get_container_status()
            assert status["initialized_services"] == 1
            assert "test_service" in status["services"]


class TestTradingServicesIntegration:
    """Test trading services dependency injection integration."""
    
    @pytest.mark.asyncio
    async def test_trading_services_registration(self):
        """Test registration of all trading services."""
        from kalshiflow_rl.trading.service_container import register_trading_services
        
        # Clean up any existing default container
        await shutdown_default_container()
        
        try:
            # Register all trading services
            container = await register_trading_services()
            
            assert container.is_registered("event_bus")
            assert container.is_registered("orderbook_state_registry")
            assert container.is_registered("live_observation_adapter")
            assert container.is_registered("actor_service")
            
            # Check dependency relationships
            adapter_info = container.get_service_info("live_observation_adapter")
            assert "orderbook_state_registry" in adapter_info["dependencies"]
            
            actor_info = container.get_service_info("actor_service")
            assert "live_observation_adapter" in actor_info["dependencies"]
            assert "event_bus" in actor_info["dependencies"]
            
        finally:
            # Clean up
            await shutdown_default_container()
    
    @pytest.mark.asyncio
    async def test_backward_compatibility_accessors(self):
        """Test backward compatibility with global singleton accessors."""
        from kalshiflow_rl.trading.service_factories import (
            get_injected_orderbook_state_registry,
            get_injected_live_observation_adapter,
            get_injected_actor_service
        )
        from kalshiflow_rl.trading.service_container import register_trading_services
        
        # Clean up any existing default container
        await shutdown_default_container()
        
        try:
            # Register services
            await register_trading_services()
            
            # Initialize all services
            container = await get_default_container()
            await container.initialize_all()
            
            # Test backward compatibility accessors
            registry = await get_injected_orderbook_state_registry()
            assert registry is not None
            
            adapter = await get_injected_live_observation_adapter()
            assert adapter is not None
            
            actor_service = await get_injected_actor_service()
            assert actor_service is not None
            
            # Verify they're the same instances from container
            container_registry = await container.get_service("orderbook_state_registry")
            container_adapter = await container.get_service("live_observation_adapter")
            container_actor = await container.get_service("actor_service")
            
            assert registry is container_registry
            assert adapter is container_adapter
            assert actor_service is container_actor
            
        finally:
            # Clean up
            await shutdown_default_container()


class TestServiceIsolation:
    """Test service isolation for testing."""
    
    @pytest.mark.asyncio
    async def test_isolated_test_containers(self):
        """Test that test containers provide proper isolation."""
        # Create two isolated containers
        async with create_test_container("test1") as container1:
            async with create_test_container("test2") as container2:
                
                # Register same service name in both containers
                container1.register_singleton(
                    "test_service", 
                    MagicMock,
                    lambda c, **deps: MagicMock(container="test1")
                )
                
                container2.register_singleton(
                    "test_service",
                    MagicMock, 
                    lambda c, **deps: MagicMock(container="test2")
                )
                
                # Get services from each container
                service1 = await container1.get_service("test_service")
                service2 = await container2.get_service("test_service")
                
                # Should be different instances
                assert service1 is not service2
                assert service1.container == "test1"
                assert service2.container == "test2"
    
    @pytest.mark.asyncio
    async def test_no_global_state_pollution(self):
        """Test that containers don't pollute global state."""
        from kalshiflow_rl.trading.live_observation_adapter import get_live_observation_adapter
        from kalshiflow_rl.trading.actor_service import get_actor_service
        
        # Ensure clean state
        await shutdown_default_container()
        
        # Create isolated test container
        async with create_test_container("isolated") as container:
            
            # Register services in isolated container
            container.register_singleton(
                "test_adapter",
                MagicMock,
                lambda c, **deps: MagicMock()
            )
            
            # Get service from isolated container
            test_adapter = await container.get_service("test_adapter")
            assert test_adapter is not None
            
            # Global accessors should still return None (no pollution)
            global_adapter = await get_live_observation_adapter()
            global_actor = await get_actor_service()
            
            # Should be None since we didn't initialize global singletons
            assert global_adapter is None
            assert global_actor is None