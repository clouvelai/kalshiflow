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
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from .config import config, logger
from .data.database import rl_db
from .data.write_queue import write_queue
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
from .trading.action_selector import select_action_stub

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
    
    try:
        # Validate authentication first
        if not validate_rl_auth():
            logger.error("RL authentication validation failed")
            raise RuntimeError("Authentication validation failed")
        
        # Initialize database
        await rl_db.initialize()
        logger.info("Database initialized")
        
        # Start write queue
        await write_queue.start()
        logger.info("Write queue started")
        
        # Start EventBus (must be before OrderbookClient emits events)
        global event_bus
        event_bus = await get_event_bus()
        await event_bus.start()
        logger.info("EventBus started")
        
        # Select market tickers (either from discovery or config)
        market_tickers = await select_market_tickers()
        
        # Initialize OrderbookClient with selected markets
        global orderbook_client
        orderbook_client = OrderbookClient(market_tickers=market_tickers)
        orderbook_client.on_connected(_on_orderbook_connected)
        orderbook_client.on_disconnected(_on_orderbook_disconnected)
        orderbook_client.on_error(_on_orderbook_error)
        
        # Set up stats collector references
        stats_collector.orderbook_client = orderbook_client
        stats_collector.websocket_manager = websocket_manager
        
        # Start stats collector
        await stats_collector.start()
        logger.info("Statistics collector started")
        
        # Start WebSocket manager
        websocket_manager.stats_collector = stats_collector
        await websocket_manager.start()
        logger.info("WebSocket manager started")
        
        # Start orderbook client as background task
        orderbook_task = asyncio.create_task(orderbook_client.start())
        _background_tasks.append(orderbook_task)
        
        # Hook up stats tracking to orderbook updates
        orderbook_states = await get_all_orderbook_states()
        for market_ticker, state in orderbook_states.items():
            def create_stats_callback(ticker):
                def callback(notification_data):
                    _on_orderbook_update_for_stats(notification_data, ticker)
                return callback
            
            state.add_subscriber(create_stats_callback(market_ticker))
        
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
            global observation_adapter
            observation_adapter = LiveObservationAdapter(
                window_size=10,
                max_markets=1,
                temporal_context_minutes=30,
                orderbook_state_registry=registry
            )
            logger.info("LiveObservationAdapter created")
            
            # Create KalshiMultiMarketOrderManager (skip demo client init for now)
            global order_manager
            order_manager = KalshiMultiMarketOrderManager(initial_cash=10000.0)
            # Skip initialize() to avoid demo client credential issues for now
            # We'll just test the pipeline without actual trading
            logger.info("KalshiMultiMarketOrderManager created (demo client skipped)")
            
            # Create ActorService
            global actor_service
            actor_service = ActorService(
                market_tickers=market_tickers,
                model_path=config.RL_ACTOR_MODEL_PATH,  # Use config value
                queue_size=1000,
                throttle_ms=config.RL_ACTOR_THROTTLE_MS,
                event_bus=event_bus,
                observation_adapter=observation_adapter
            )
            
            # Set action selector stub (returns HOLD)
            actor_service.set_action_selector(select_action_stub)
            logger.info("Action selector stub configured")
            
            # Set order manager
            actor_service.set_order_manager(order_manager)
            logger.info("Order manager configured")
            
            # Initialize ActorService (subscribes to event bus and starts processing loop)
            await actor_service.initialize()
            logger.info("✅ ActorService initialized and ready")
        else:
            logger.info("ActorService disabled - orderbook collector only mode")
            actor_service = None
            order_manager = None
            observation_adapter = None
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
        
        # Shutdown ActorService first
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


def _on_orderbook_update_for_stats(notification_data: Dict[str, Any], market_ticker: str):
    """Track orderbook updates in statistics (sync callback)."""
    try:
        update_type = notification_data.get('update_type')
        if update_type == "snapshot":
            stats_collector.track_snapshot(market_ticker)
        elif update_type == "delta":
            stats_collector.track_delta(market_ticker)
    except Exception as e:
        logger.error(f"Error tracking stats for {market_ticker}: {e}")


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
                "debug": config.DEBUG
            },
            "stats": {
                "write_queue": write_queue.get_stats(),
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


async def force_flush_endpoint(request):
    """Force flush of write queue (admin endpoint)."""
    try:
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


async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time orderbook updates."""
    await websocket_manager.handle_connection(websocket)


# Routes
routes = [
    Route("/rl/health", health_check, methods=["GET"]),
    Route("/rl/status", status_endpoint, methods=["GET"]),
    Route("/rl/orderbook/snapshot", orderbook_snapshot_endpoint, methods=["GET"]),
    Route("/rl/admin/flush", force_flush_endpoint, methods=["POST"]),
    WebSocketRoute("/rl/ws", websocket_endpoint),
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