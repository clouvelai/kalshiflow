"""
Starlette application for RL Trading Subsystem.

Provides ASGI app with startup/shutdown lifecycle management,
background task orchestration, health endpoints, and graceful
service coordination.
"""

import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import aiohttp

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from .config import config, logger
from .data.database import rl_db
from .data.write_queue import get_write_queue
from .data.orderbook_client import OrderbookClient
from .data.orderbook_state import get_all_orderbook_states, get_shared_orderbook_state
from .data.auth import validate_rl_auth
from .data.market_discovery import fetch_active_markets
from .websocket_manager import websocket_manager
from .stats_collector import stats_collector
from .trading.event_bus import get_event_bus
from .trading.actor_service import ActorService
from .trading.kalshi_multi_market_order_manager import KalshiMultiMarketOrderManager
from .trading.live_observation_adapter import LiveObservationAdapter
from .trading.action_selector import create_action_selector
from .trading.initialization_tracker import InitializationTracker

# Background task management
_background_tasks = []
_shutdown_event = asyncio.Event()

# Global orderbook client instance (initialized in lifespan)
orderbook_client: Optional[OrderbookClient] = None

# Global ActorService components (initialized in lifespan)
event_bus = None
actor_service: Optional[ActorService] = None
order_manager: Optional[KalshiMultiMarketOrderManager] = None
observation_adapter: Optional[LiveObservationAdapter] = None

# Global market tickers (determined at startup)
active_market_tickers: Optional[list] = None


async def select_market_tickers() -> list[str]:
    """
    Select market tickers based on configuration mode.
    
    Returns:
        List of market tickers to monitor
    """
    global active_market_tickers
    
    if config.RL_MARKET_MODE == "discovery":
        logger.info(f"Using discovery mode to fetch {config.ORDERBOOK_MARKET_LIMIT} active markets...")
        try:
            discovered_tickers = await fetch_active_markets(limit=config.ORDERBOOK_MARKET_LIMIT)
            if discovered_tickers:
                active_market_tickers = discovered_tickers
                logger.info(f"Discovered {len(discovered_tickers)} active markets")
                return discovered_tickers
            else:
                logger.warning("No active markets discovered, falling back to config mode")
                active_market_tickers = config.RL_MARKET_TICKERS
                return config.RL_MARKET_TICKERS
        except Exception as e:
            logger.error(f"Market discovery failed: {e}, falling back to config mode")
            active_market_tickers = config.RL_MARKET_TICKERS
            return config.RL_MARKET_TICKERS
    else:
        logger.info(f"Using config mode with {len(config.RL_MARKET_TICKERS)} configured markets")
        active_market_tickers = config.RL_MARKET_TICKERS
        return config.RL_MARKET_TICKERS


async def check_exchange_status(api_url: str) -> Dict[str, Any]:
    """
    Check Kalshi exchange status by calling the /exchange/status endpoint.
    
    Args:
        api_url: Base API URL (e.g., https://demo-api.kalshi.co/trade-api/v2)
        
    Returns:
        Dictionary with exchange status information including:
        - exchange_active: bool
        - trading_active: bool
        - exchange_estimated_resume_time: str | None
        - api_status: "healthy" | "unhealthy" | "error"
        - error: str | None (if error occurred)
    """
    status_url = f"{api_url}/exchange/status"
    
    try:
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(status_url) as response:
                if response.status == 200:
                    data = await response.json()
                    exchange_active = data.get("exchange_active", False)
                    trading_active = data.get("trading_active", False)
                    estimated_resume_time = data.get("exchange_estimated_resume_time")
                    
                    # Consider unhealthy if exchange is not active
                    api_status = "healthy" if exchange_active else "unhealthy"
                    
                    return {
                        "exchange_active": exchange_active,
                        "trading_active": trading_active,
                        "exchange_estimated_resume_time": estimated_resume_time,
                        "api_status": api_status,
                        "error": None,
                        "http_status": response.status,
                    }
                else:
                    # Non-200 status codes indicate service issues
                    error_text = await response.text()
                    return {
                        "exchange_active": False,
                        "trading_active": False,
                        "exchange_estimated_resume_time": None,
                        "api_status": "unhealthy",
                        "error": f"HTTP {response.status}: {error_text}",
                        "http_status": response.status,
                    }
    except asyncio.TimeoutError:
        return {
            "exchange_active": False,
            "trading_active": False,
            "exchange_estimated_resume_time": None,
            "api_status": "error",
            "error": "Request timeout",
            "http_status": None,
        }
    except Exception as e:
        logger.error(f"Failed to check exchange status: {e}")
        return {
            "exchange_active": False,
            "trading_active": False,
            "exchange_estimated_resume_time": None,
            "api_status": "error",
            "error": str(e),
            "http_status": None,
        }


async def verify_orderbook_health(orderbook_client, initialization_tracker, max_wait_time: float = 30.0):
    """
    Verify orderbook client health and data reception.
    
    Args:
        orderbook_client: The orderbook client to verify
        initialization_tracker: Tracker for initialization status
        max_wait_time: Maximum time to wait for health verification
        
    Returns:
        True if healthy, False otherwise
    """
    logger.info("Verifying orderbook health...")
    wait_interval = 1.0
    elapsed = 0.0
    
    while elapsed < max_wait_time:
        if orderbook_client and orderbook_client.is_healthy():
            # Verify it has received some data (at least one snapshot or delta)
            stats = orderbook_client.get_stats()
            if stats.get("snapshots_received", 0) > 0 or stats.get("deltas_received", 0) > 0:
                logger.info(f"✅ Orderbook verified healthy after {elapsed:.1f}s (snapshots: {stats.get('snapshots_received', 0)}, deltas: {stats.get('deltas_received', 0)})")
                # Update health status
                health_details = orderbook_client.get_health_details()
                await initialization_tracker.update_component_health("orderbook_client", "healthy", health_details)
                return True
        
        await asyncio.sleep(wait_interval)
        elapsed += wait_interval
    
    # Not healthy after timeout
    if orderbook_client:
        health_details = orderbook_client.get_health_details()
        await initialization_tracker.update_component_health("orderbook_client", "unhealthy", health_details)
    
    logger.warning(f"Orderbook not healthy after {max_wait_time}s wait")
    return False


@asynccontextmanager
async def lifespan(app: Starlette):
    """
    Lifespan context manager for startup and shutdown.
    
    Handles:
    - Database initialization
    - Background service startup
    - Graceful shutdown coordination
    """
    logger.info("Starting RL Trading Subsystem...")
    
    # Initialize module-level variables to avoid UnboundLocalError in finally block
    global actor_service, order_manager, observation_adapter
    actor_service = None
    order_manager = None
    observation_adapter = None
    
    # Create initialization tracker
    initialization_tracker = InitializationTracker(websocket_manager=websocket_manager)
    
    try:
        # Start WebSocket manager early so it can accept connections and broadcast initialization updates
        await websocket_manager.start_early()
        logger.info("WebSocket manager started early for initialization tracking")
        
        # Start initialization tracking (can now broadcast)
        await initialization_tracker.start()
        
        # Check exchange status FIRST before anything else
        await initialization_tracker.mark_step_in_progress("exchange_status_health")
        exchange_status = await check_exchange_status(config.KALSHI_API_URL)
        
        # Determine if exchange is healthy
        # Exchange is healthy if exchange_active is True
        exchange_healthy = exchange_status.get("exchange_active", False) and exchange_status.get("api_status") == "healthy"
        
        if exchange_healthy:
            await initialization_tracker.mark_step_complete("exchange_status_health", {
                "details": exchange_status
            })
            await initialization_tracker.update_component_health("exchange_status", "healthy", exchange_status)
        else:
            error_msg = f"Exchange status unhealthy: {exchange_status.get('error', 'exchange_active=False')}"
            if exchange_status.get("exchange_estimated_resume_time"):
                error_msg += f" (estimated resume: {exchange_status['exchange_estimated_resume_time']})"
            await initialization_tracker.mark_step_failed("exchange_status_health", error_msg, {
                "details": exchange_status
            })
            # FAIL INITIALIZATION if exchange is unhealthy
            raise RuntimeError(f"Kalshi exchange is unavailable: {error_msg}")
        
        
        # Validate authentication first (skip if ActorService enabled - OrderManager will validate)
        # OrderManager initialization will validate credentials, so we can defer validation
        # when ActorService is enabled to allow OrderManager to handle credential validation
        if not config.RL_ACTOR_ENABLED:
            # Only validate auth early if ActorService is disabled (orderbook collector only)
            if not validate_rl_auth():
                logger.error("RL authentication validation failed")
                raise RuntimeError("Authentication validation failed")
        else:
            logger.info("Skipping early auth validation - OrderManager will validate credentials during initialization")
        
        # Initialize database
        await rl_db.initialize()
        logger.info("Database initialized")
        
        # Start write queue
        write_queue = get_write_queue()
        await write_queue.start()
        logger.info("Write queue started")
        
        # Start EventBus (must be before OrderbookClient emits events)
        global event_bus
        event_bus = await get_event_bus()
        await event_bus.start()
        logger.info("EventBus started")
        
        # Report EventBus health
        await initialization_tracker.mark_step_in_progress("event_bus_health")
        if event_bus.is_healthy():
            await initialization_tracker.mark_step_complete("event_bus_health", {
                "details": event_bus.get_health_details()
            })
            await initialization_tracker.update_component_health("event_bus", "healthy", event_bus.get_health_details())
        else:
            await initialization_tracker.mark_step_failed("event_bus_health", "EventBus not healthy")
        
        # Select market tickers (either from discovery or config)
        market_tickers = await select_market_tickers()
        
        # Initialize OrderbookClient with selected markets and stats collector
        # Pass stats_collector directly so it tracks snapshots/deltas immediately
        global orderbook_client
        orderbook_client = OrderbookClient(market_tickers=market_tickers, stats_collector=stats_collector)
        orderbook_client.on_connected(_on_orderbook_connected)
        orderbook_client.on_disconnected(_on_orderbook_disconnected)
        orderbook_client.on_error(_on_orderbook_error)
        
        # Set up stats collector references
        stats_collector.orderbook_client = orderbook_client
        stats_collector.websocket_manager = websocket_manager
        
        # Start stats collector
        await stats_collector.start()
        logger.info("Statistics collector started")
        
        # Update WebSocket manager with discovered markets
        websocket_manager.set_market_tickers(market_tickers)
        websocket_manager.stats_collector = stats_collector
        
        # Start orderbook client as background task
        # Note: Stats tracking is now handled directly inside orderbook_client via passed stats_collector
        orderbook_task = asyncio.create_task(orderbook_client.start())
        _background_tasks.append(orderbook_task)
        
        # Wait for orderbook client to establish WebSocket connection
        await initialization_tracker.mark_step_in_progress("orderbook_connection")
        connection_established = await orderbook_client.wait_for_connection(timeout=30.0)
        
        if connection_established:
            await initialization_tracker.mark_step_complete("orderbook_connection", {
                "status": "connected",
                "markets": len(orderbook_client.market_tickers)
            })
            
            # Subscribe to orderbook states after connection is confirmed
            await websocket_manager.subscribe_to_orderbook_states()
            logger.info("WebSocket manager subscribed to orderbook states")
        else:
            await initialization_tracker.mark_step_failed("orderbook_connection", 
                "Failed to establish WebSocket connection within timeout", {
                    "timeout": 30.0,
                    "markets": orderbook_client.market_tickers
                })
            # Continue anyway - orderbook client will keep retrying
        
        # Verify orderbook health and data reception
        await initialization_tracker.mark_step_in_progress("orderbook_health")
        orderbook_healthy = await verify_orderbook_health(orderbook_client, initialization_tracker, max_wait_time=30.0)
        
        if orderbook_healthy:
            stats = orderbook_client.get_stats()
            await initialization_tracker.mark_step_complete("orderbook_health", {
                "markets_subscribed": stats.get("market_count", 0),
                "snapshots_received": stats.get("snapshots_received", 0),
                "deltas_received": stats.get("deltas_received", 0),
            })
        else:
            await initialization_tracker.mark_step_failed("orderbook_health", "OrderbookClient failed health verification", {
                "details": orderbook_client.get_health_details() if orderbook_client else None
            })
        
        # Initialize ActorService components for trading (only if enabled)
        logger.info("=" * 60)
        logger.info(f"Actor Service: {'ENABLED' if config.RL_ACTOR_ENABLED else 'DISABLED'}")
        if config.RL_ACTOR_ENABLED:
            logger.info("ActorService enabled - initializing trading components...")
            logger.info(f"  Strategy: {config.RL_ACTOR_STRATEGY}")
            logger.info(f"  Model Path: {config.RL_ACTOR_MODEL_PATH or 'None (using stub)'}")
            
            # Create OrderbookStateRegistry wrapper (uses global get_shared_orderbook_state)
            class OrderbookStateRegistryWrapper:
                """Simple wrapper that uses global get_shared_orderbook_state."""
                async def get_shared_orderbook_state(self, market_ticker: str):
                    return await get_shared_orderbook_state(market_ticker)
            
            registry = OrderbookStateRegistryWrapper()
            
            # Create LiveObservationAdapter
            observation_adapter = LiveObservationAdapter(
                window_size=10,
                max_markets=1,
                temporal_context_minutes=30,
                orderbook_state_registry=registry
            )
            logger.info("LiveObservationAdapter created")
            
            # Create KalshiMultiMarketOrderManager
            order_manager = KalshiMultiMarketOrderManager(initial_cash=10000.0)
            
            # Initialize OrderManager (requires credentials - fail fast if missing)
            try:
                await order_manager.initialize(
                    initialization_tracker=initialization_tracker,
                    websocket_manager=websocket_manager
                )
                logger.info("✅ KalshiMultiMarketOrderManager initialized successfully")
                # Cleanup is now handled inside initialize() method before syncing
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize OrderManager: {e}")
                logger.error("ActorService disabled - OrderManager requires valid credentials")
                order_manager = None
                actor_service = None
                observation_adapter = None
                # Continue without ActorService - orderbook collector still works
                logger.info("Continuing without ActorService (orderbook collection only)")
            
            # Only proceed with ActorService if OrderManager initialized successfully
            if order_manager:
                # Create ActorService
                actor_service = ActorService(
                    market_tickers=market_tickers,
                    model_path=config.RL_ACTOR_MODEL_PATH,  # Use config value
                    queue_size=1000,
                    throttle_ms=config.RL_ACTOR_THROTTLE_MS,
                    event_bus=event_bus,
                    observation_adapter=observation_adapter
                )
                
                # Create and set action selector based on config
                # No fallbacks - if selector creation fails, initialization fails
                selector = create_action_selector(
                    strategy=config.RL_ACTOR_STRATEGY,
                    model_path=config.RL_ACTOR_MODEL_PATH
                )
                actor_service.set_action_selector(selector)
                
                # Register selector with order manager for mutual exclusivity protection
                order_manager.register_action_selector(selector)
                
                logger.info(f"Action selector configured: {selector.get_strategy_name()}")
                
                # Set order manager
                actor_service.set_order_manager(order_manager)
                logger.info("Order manager configured")
                
                # Set websocket manager for broadcasting trader actions
                actor_service.set_websocket_manager(websocket_manager)
                logger.info("WebSocket manager configured for ActorService")
                
                # Set order manager reference in websocket manager for initial state broadcast
                websocket_manager.set_order_manager(order_manager)
                websocket_manager.set_actor_service(actor_service)
                
                # WebSocket manager already set in order_manager during initialize()
                # No need to call order_manager.set_websocket_manager() again
                
                # Add state change callback to order manager for UI updates
                async def broadcast_state(state):
                    """Callback to broadcast trader state changes via websocket."""
                    # Include actor metrics if available
                    if actor_service:
                        try:
                            actor_metrics = actor_service.get_metrics()
                            state["actor_metrics"] = actor_metrics
                        except Exception as e:
                            logger.warning(f"Could not get actor metrics: {e}")
                    
                    await websocket_manager.broadcast_trader_state(state)
                
                order_manager.add_state_change_callback(broadcast_state)
                logger.info("State change callback configured for OrderManager")
                
                # Initialize ActorService (subscribes to event bus and starts processing loop)
                # Only start ActorService after all initialization steps are complete
                await actor_service.initialize()
                logger.info("✅ ActorService initialized and ready")
                
                # Final verification: Ensure orderbook is still healthy before completing initialization
                logger.info("Final orderbook health verification before completing initialization...")
                orderbook_healthy_at_end = await verify_orderbook_health(orderbook_client, initialization_tracker, max_wait_time=10.0)
                
                if not orderbook_healthy_at_end:
                    # Orderbook must be healthy and receiving data before initialization completes
                    error_msg = "Orderbook not healthy after final verification - must receive data before initialization can complete"
                    logger.error(error_msg)
                    if orderbook_client:
                        health_details = orderbook_client.get_health_details()
                        await initialization_tracker.update_component_health("orderbook_client", "unhealthy", health_details)
                    raise RuntimeError(error_msg)
                
                # Complete initialization tracking
                await initialization_tracker.complete_initialization({
                    "starting_cash": order_manager.session_start_cash if order_manager else None,
                    "starting_portfolio_value": order_manager.session_start_portfolio_value if order_manager else None,
                    "positions_resumed": len(order_manager.positions) if order_manager else 0,
                    "orders_resumed": len(order_manager.open_orders) if order_manager else 0,
                    "markets_trading": market_tickers,
                })
                
                # Start periodic health broadcasts
                async def periodic_health_broadcast():
                    """Periodically broadcast component health updates."""
                    while True:
                        try:
                            await asyncio.sleep(10.0)  # Broadcast every 10 seconds
                            
                            # Update health for all components
                            if orderbook_client:
                                try:
                                    health_status = "healthy" if orderbook_client.is_healthy() else "unhealthy"
                                except Exception as e:
                                    logger.warning(f"Error checking orderbook health: {e}, assuming healthy based on stats")
                                    # Fallback: check if we're receiving messages
                                    stats = orderbook_client.get_stats()
                                    health_status = "healthy" if stats.get("messages_received", 0) > 0 else "unhealthy"
                                
                                await initialization_tracker.update_component_health(
                                    "orderbook_client",
                                    health_status,
                                    orderbook_client.get_health_details()
                                )
                            
                            if order_manager:
                                health_status = "healthy" if order_manager.is_healthy() else "unhealthy"
                                await initialization_tracker.update_component_health(
                                    "trader_client",
                                    health_status,
                                    order_manager.get_health_details()
                                )
                                
                                if order_manager._fill_listener:
                                    fill_health_status = "healthy" if order_manager._fill_listener.is_healthy() else "unhealthy"
                                    await initialization_tracker.update_component_health(
                                        "fill_listener",
                                        fill_health_status,
                                        order_manager._fill_listener.get_health_details()
                                    )
                                
                                if order_manager._position_listener:
                                    position_health_status = "healthy" if order_manager._position_listener.is_healthy() else "unhealthy"
                                    await initialization_tracker.update_component_health(
                                        "position_listener",
                                        position_health_status,
                                        order_manager._position_listener.get_health_details()
                                    )
                            
                            if event_bus:
                                health_status = "healthy" if event_bus.is_healthy() else "unhealthy"
                                await initialization_tracker.update_component_health(
                                    "event_bus",
                                    health_status,
                                    event_bus.get_health_details()
                                )
                            
                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            logger.error(f"Error in periodic health broadcast: {e}")
                
                health_broadcast_task = asyncio.create_task(periodic_health_broadcast())
                _background_tasks.append(health_broadcast_task)
        else:
            logger.info("ActorService disabled - orderbook collector only mode")
            actor_service = None
            order_manager = None
            observation_adapter = None
            
            # Complete initialization even if ActorService is disabled
            await initialization_tracker.complete_initialization({
                "actor_service_enabled": False,
                "markets_collecting": market_tickers,
            })
        logger.info("=" * 60)
        
        # Log startup summary
        logger.info(f"RL Trading Subsystem started successfully for {len(market_tickers)} markets")
        
        # Show sample of markets if there are many (for production logging)
        if len(market_tickers) > 10:
            sample_markets = ', '.join(market_tickers[:5])
            logger.info(f"Monitoring markets (sample): {sample_markets} ... and {len(market_tickers) - 5} more")
        
        # Wait for shutdown signal
        yield
        
    except Exception as e:
        logger.error(f"Failed to start RL Trading Subsystem: {e}")
        raise
    
    finally:
        # Shutdown sequence
        logger.info("Shutting down RL Trading Subsystem...")
        _shutdown_event.set()
        
        # Shutdown ActorService first (check if it was initialized)
        if actor_service:
            try:
                await actor_service.shutdown()
                logger.info("ActorService shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down ActorService: {e}")
        
        # Shutdown OrderManager
        if order_manager:
            try:
                await order_manager.shutdown()
                logger.info("OrderManager shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down OrderManager: {e}")
        
        # Stop WebSocket manager first (stops broadcasting)
        await websocket_manager.stop()
        
        # Stop orderbook client
        if orderbook_client:
            await orderbook_client.stop()
        
        # Stop EventBus
        if event_bus:
            try:
                await event_bus.stop()
                logger.info("EventBus shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down EventBus: {e}")
        
        # Stop stats collector
        await stats_collector.stop()
        
        # Stop write queue (this flushes remaining messages)
        write_queue = get_write_queue()
        await write_queue.stop()
        
        # Cancel background tasks
        for task in _background_tasks:
            if not task.done():
                task.cancel()
        
        if _background_tasks:
            await asyncio.gather(*_background_tasks, return_exceptions=True)
        
        # Close database connections
        await rl_db.close()
        
        logger.info("RL Trading Subsystem shutdown complete")


# Event handlers for orderbook client
async def _on_orderbook_connected():
    """Handle orderbook client connection."""
    logger.info("Orderbook client connected successfully")


async def _on_orderbook_disconnected():
    """Handle orderbook client disconnection."""
    logger.warning("Orderbook client disconnected")


async def _on_orderbook_error(error: Exception):
    """Handle orderbook client errors."""
    logger.error(f"Orderbook client error: {error}")


# Stats tracking is now handled directly in orderbook_client via passed stats_collector


# API endpoints

async def health_check(request):
    """Health check endpoint for monitoring."""
    try:
        # Check all components
        health_status = {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "service": "kalshiflow_rl",
            "version": "0.1.0",
            "market_tickers": active_market_tickers or config.RL_MARKET_TICKERS,
            "markets_count": len(active_market_tickers or config.RL_MARKET_TICKERS),
            "market_mode": config.RL_MARKET_MODE,
            "kalshi_api_url": config.KALSHI_API_URL,
            "kalshi_ws_url": config.KALSHI_WS_URL,
            "components": {}
        }
        
        # Check database
        try:
            async with rl_db.get_connection() as conn:
                await conn.fetchval('SELECT 1')
            health_status["components"]["database"] = {"status": "healthy"}
        except Exception as e:
            health_status["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check write queue
        write_queue = get_write_queue()
        if write_queue.is_healthy():
            health_status["components"]["write_queue"] = {
                "status": "healthy",
                **write_queue.get_stats()
            }
        else:
            health_status["components"]["write_queue"] = {
                "status": "unhealthy",
                **write_queue.get_stats()
            }
            health_status["status"] = "degraded"
        
        # Check orderbook client
        if orderbook_client and orderbook_client.is_healthy():
            health_status["components"]["orderbook_client"] = {
                "status": "healthy",
                **orderbook_client.get_stats()
            }
        else:
            health_status["components"]["orderbook_client"] = {
                "status": "unhealthy" if orderbook_client else "not_initialized",
                **(orderbook_client.get_stats() if orderbook_client else {})
            }
            health_status["status"] = "degraded"
        
        # Check WebSocket manager
        if websocket_manager.is_healthy():
            health_status["components"]["websocket_manager"] = {
                "status": "healthy",
                **websocket_manager.get_stats()
            }
        else:
            health_status["components"]["websocket_manager"] = {
                "status": "unhealthy",
                **websocket_manager.get_stats()
            }
            health_status["status"] = "degraded"
        
        # Check statistics collector
        if stats_collector.is_healthy():
            health_status["components"]["stats_collector"] = {
                "status": "healthy",
                **stats_collector.get_summary()
            }
        else:
            health_status["components"]["stats_collector"] = {
                "status": "unhealthy",
                **stats_collector.get_summary()
            }
            health_status["status"] = "degraded"
        
        # Check ActorService (if enabled)
        if config.RL_ACTOR_ENABLED:
            if actor_service:
                # Check if actor service is processing (healthy)
                actor_status = actor_service.get_status()
                if actor_status.get("status") == "running":
                    health_status["components"]["actor_service"] = {
                        "status": "healthy",
                        **actor_service.get_metrics()
                    }
                else:
                    health_status["components"]["actor_service"] = {
                        "status": "unhealthy",
                        "status_detail": actor_status.get("status"),
                        **actor_service.get_metrics()
                    }
                    health_status["status"] = "degraded"
            else:
                health_status["components"]["actor_service"] = {
                    "status": "not_initialized",
                    "error": "ActorService enabled but not initialized"
                }
                health_status["status"] = "degraded"
        else:
            health_status["components"]["actor_service"] = {
                "status": "disabled"
            }
        
        # Return appropriate status code
        status_code = 200 if health_status["status"] == "healthy" else 503
        return JSONResponse(health_status, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e),
            "service": "kalshiflow_rl"
        }, status_code=503)


async def status_endpoint(request):
    """Detailed status endpoint with component information."""
    try:
        # Get detailed stats from all components
        orderbook_states = await get_all_orderbook_states()
        
        status = {
            "service": "kalshiflow_rl",
            "timestamp": asyncio.get_event_loop().time(),
            "config": {
                "market_tickers": active_market_tickers or config.RL_MARKET_TICKERS,
                "markets_count": len(active_market_tickers or config.RL_MARKET_TICKERS),
                "market_mode": config.RL_MARKET_MODE,
                "market_limit": config.ORDERBOOK_MARKET_LIMIT,
                "configured_tickers": config.RL_MARKET_TICKERS,
                "environment": config.ENVIRONMENT,
                "debug": config.DEBUG,
                "kalshi_api_url": config.KALSHI_API_URL,
                "kalshi_ws_url": config.KALSHI_WS_URL
            },
            "stats": {
                "write_queue": get_write_queue().get_stats(),
                "orderbook_client": orderbook_client.get_stats() if orderbook_client else {},
                "websocket_manager": websocket_manager.get_stats(),
                "stats_collector": stats_collector.get_stats(),
                "orderbook_states": {
                    ticker: state.get_stats() 
                    for ticker, state in orderbook_states.items()
                }
            }
        }
        
        return JSONResponse(status)
        
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        return JSONResponse({
            "error": str(e),
            "service": "kalshiflow_rl"
        }, status_code=500)


async def orderbook_snapshot_endpoint(request):
    """Get current orderbook snapshots for all configured markets."""
    try:
        orderbook_states = await get_all_orderbook_states()
        
        # Get snapshots for all active markets
        market_tickers = active_market_tickers or config.RL_MARKET_TICKERS
        snapshots = {}
        for market_ticker in market_tickers:
            if market_ticker in orderbook_states:
                state = orderbook_states[market_ticker]
                snapshot = await state.get_snapshot()
                snapshots[market_ticker] = snapshot
        
        if snapshots:
            return JSONResponse({
                "market_tickers": list(snapshots.keys()),
                "snapshots": snapshots,
                "timestamp": asyncio.get_event_loop().time()
            })
        else:
            return JSONResponse({
                "error": "No orderbook states found for active markets",
                "active_markets": market_tickers,
                "configured_markets": config.RL_MARKET_TICKERS,
                "available_markets": list(orderbook_states.keys())
            }, status_code=404)
            
    except Exception as e:
        logger.error(f"Orderbook snapshot error: {e}")
        return JSONResponse({
            "error": str(e)
        }, status_code=500)


async def trader_status_endpoint(request):
    """Get current trader status including positions, orders, and recent actions."""
    try:
        # Check if ActorService is enabled
        if not actor_service:
            return JSONResponse({
                "error": "ActorService not enabled",
                "message": "Trader functionality is disabled. Enable with RL_ACTOR_ENABLED=true"
            }, status_code=503)
        
        # Check if OrderManager is available
        if not order_manager:
            return JSONResponse({
                "error": "OrderManager not initialized",
                "message": "Trader OrderManager is not available"
            }, status_code=503)
        
        # Get current trader state
        trader_state = await order_manager.get_current_state()
        
        # Get actor service metrics
        actor_metrics = actor_service.get_metrics() if actor_service else {}
        
        # Combine into status response
        status = {
            "service": "trader",
            "timestamp": asyncio.get_event_loop().time(),
            "enabled": True,
            "strategy": config.RL_ACTOR_STRATEGY,
            "model_path": config.RL_ACTOR_MODEL_PATH,
            "trader_state": trader_state,
            "actor_metrics": actor_metrics,
            "markets_trading": active_market_tickers or config.RL_MARKET_TICKERS
        }
        
        return JSONResponse(status)
        
    except Exception as e:
        logger.error(f"Trader status error: {e}")
        return JSONResponse({
            "error": str(e),
            "service": "trader"
        }, status_code=500)


async def force_flush_endpoint(request):
    """Force flush of write queue (admin endpoint)."""
    try:
        write_queue = get_write_queue()
        await write_queue.force_flush()
        
        return JSONResponse({
            "message": "Write queue flushed successfully",
            "stats": write_queue.get_stats()
        })
        
    except Exception as e:
        logger.error(f"Force flush error: {e}")
        return JSONResponse({
            "error": str(e)
        }, status_code=500)


async def trader_sync_endpoint(request):
    """Manual sync endpoint for frontend-triggered synchronization."""
    try:
        # Check if ActorService and OrderManager are available
        if not order_manager:
            return JSONResponse({
                "error": "OrderManager not available - ActorService may be disabled"
            }, status_code=503)
        
        logger.info("Manual trader sync requested by frontend")
        
        # Perform order sync
        logger.info("Starting manual order sync...")
        order_stats = await order_manager.sync_orders_with_kalshi()
        
        # Perform position sync
        logger.info("Starting manual position sync...")
        await order_manager._sync_positions_with_kalshi()
        
        # Perform settlement sync
        logger.info("Starting manual settlement sync...")
        settlement_stats = await order_manager.sync_settlements_with_kalshi()
        
        # Get current state for broadcasting
        current_state = await order_manager.get_current_state()
        
        # Broadcast updates via WebSocket with proper source attribution
        if websocket_manager:
            # Broadcast orders update
            await websocket_manager.broadcast_orders_update(
                {"orders": current_state.get("open_orders", [])},
                source="api_sync"
            )
            
            # Broadcast positions update
            positions_data = {
                "positions": current_state.get("positions", {}),
                "total_value": current_state.get("portfolio_value", 0.0)
            }
            await websocket_manager.broadcast_positions_update(
                positions_data,
                source="api_sync"
            )
            
            # Broadcast portfolio update
            portfolio_data = {
                "cash_balance": current_state.get("cash_balance", 0.0),
                "portfolio_value": current_state.get("portfolio_value", 0.0)
            }
            await websocket_manager.broadcast_portfolio_update(portfolio_data)
            
            logger.info("Manual sync completed - WebSocket updates broadcast")
        
        return JSONResponse({
            "message": "Trader sync completed successfully",
            "order_sync": order_stats,
            "settlement_sync": settlement_stats,
            "trader_state": current_state,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"Trader sync error: {e}")
        return JSONResponse({
            "error": str(e),
            "message": "Failed to sync trader state"
        }, status_code=500)


async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time orderbook updates."""
    await websocket_manager.handle_connection(websocket)


async def trades_websocket_endpoint(websocket):
    """WebSocket endpoint for real-time trades and observation space updates."""
    try:
        await websocket.accept()
        logger.info("Trades WebSocket client connected")
        
        # Create a callback function to send updates to this WebSocket
        async def send_trades_update(data):
            if websocket.client_state == websocket.client_state.CONNECTED:
                try:
                    await websocket.send_json(data)
                except Exception as e:
                    logger.error(f"Error sending trades update: {e}")
        
        # Register the callback with the order manager if available
        if order_manager:
            order_manager.add_trade_broadcast_callback(send_trades_update)
            logger.info("Registered trades broadcast callback")
            
            # Send initial state
            initial_data = {
                "type": "trades",
                "data": {
                    "recent_fills": [],
                    "execution_stats": {
                        "total_fills": order_manager.execution_stats.total_fills,
                        "maker_fills": order_manager.execution_stats.maker_fills,
                        "taker_fills": order_manager.execution_stats.taker_fills,
                        "avg_fill_time_ms": round(order_manager.execution_stats.avg_fill_time_ms, 2),
                        "total_volume": round(order_manager.execution_stats.total_volume, 2)
                    },
                    "observation_space": {
                        "orderbook_features": {
                            "spread": {"value": 0.02, "intensity": "medium"},
                            "bid_depth": {"value": 0.5, "intensity": "medium"},
                            "ask_depth": {"value": 0.5, "intensity": "medium"}
                        },
                        "market_dynamics": {
                            "momentum": {"value": 0.0, "intensity": "low"},
                            "volatility": {"value": 0.1, "intensity": "low"},
                            "activity": {"value": 0.3, "intensity": "medium"}
                        },
                        "portfolio_state": {
                            "cash_ratio": {"value": order_manager.cash_balance / (order_manager.cash_balance + order_manager.promised_cash + 1), "intensity": "high"},
                            "exposure": {"value": len(order_manager.positions) / 10.0, "intensity": "medium" if len(order_manager.positions) < 5 else "high"},
                            "risk_level": {"value": min(len(order_manager.open_orders) / 10.0, 1.0), "intensity": "low" if len(order_manager.open_orders) < 3 else "high"}
                        }
                    }
                }
            }
            await send_trades_update(initial_data)
        else:
            logger.warning("Order manager not available for trades WebSocket")
            await websocket.send_json({
                "type": "error",
                "message": "Order manager not available"
            })
        
        # Keep connection alive and handle disconnection
        try:
            while True:
                # Wait for ping or close message
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
        except Exception as e:
            logger.info(f"Trades WebSocket client disconnected: {e}")
        
    except Exception as e:
        logger.error(f"Error in trades WebSocket: {e}")
    finally:
        logger.info("Trades WebSocket client disconnected")


# Routes
routes = [
    Route("/rl/health", health_check, methods=["GET"]),
    Route("/rl/status", status_endpoint, methods=["GET"]),
    Route("/rl/orderbook/snapshot", orderbook_snapshot_endpoint, methods=["GET"]),
    Route("/rl/trader/status", trader_status_endpoint, methods=["GET"]),
    Route("/rl/trader/sync", trader_sync_endpoint, methods=["POST"]),
    Route("/rl/admin/flush", force_flush_endpoint, methods=["POST"]),
    WebSocketRoute("/rl/ws", websocket_endpoint),
    WebSocketRoute("/rl/trades", trades_websocket_endpoint),
]

# Middleware
middleware = [
    Middleware(CORSMiddleware, 
               allow_origins=["*"], 
               allow_methods=["GET", "POST"],
               allow_headers=["*"])
]

# Create Starlette app
app = Starlette(
    debug=config.DEBUG,
    routes=routes,
    middleware=middleware,
    lifespan=lifespan
)


# Signal handlers for graceful shutdown
def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        _shutdown_event.set()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# Initialize signal handlers
if __name__ == "__main__":
    setup_signal_handlers()


# Export for ASGI server
__all__ = ["app"]