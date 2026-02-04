"""
TRADER V3 Standalone Application.

Clean, simple Starlette application for V3 trader.
Runs on port 8005 with minimal dependencies.
"""

import asyncio
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import V3 components
from src.kalshiflow_rl.traderv3.core.state_machine import TraderStateMachine as V3StateMachine, TraderState as V3State
from src.kalshiflow_rl.traderv3.core.event_bus import EventBus
from src.kalshiflow_rl.traderv3.core.websocket_manager import V3WebSocketManager
from src.kalshiflow_rl.traderv3.core.coordinator import V3Coordinator
from src.kalshiflow_rl.traderv3.clients.orderbook_integration import V3OrderbookIntegration
from src.kalshiflow_rl.traderv3.clients.trading_client_integration import V3TradingClientIntegration
from src.kalshiflow_rl.traderv3.clients.trades_client import TradesClient
from src.kalshiflow_rl.traderv3.clients.trades_integration import V3TradesIntegration
from src.kalshiflow_rl.traderv3.config.environment import load_config

# Import existing orderbook client
from src.kalshiflow_rl.data.orderbook_client import OrderbookClient

# Import auth for TradesClient
from kalshiflow.auth import KalshiAuth

# Import order context service for CSV export
from src.kalshiflow_rl.traderv3.services.order_context_service import get_order_context_service

# Import database and write queue for orderbook data persistence
from src.kalshiflow_rl.data.database import rl_db
from src.kalshiflow_rl.data.write_queue import get_write_queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("kalshiflow_rl.traderv3")

# Add file handler for structured log monitoring
_log_dir = Path(__file__).parent.parent.parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)
_file_handler = logging.handlers.RotatingFileHandler(
    _log_dir / "v3-trader.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=3,
)
_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logging.getLogger().addHandler(_file_handler)


def _configure_logging() -> None:
    """Suppress noisy third-party and internal loggers.

    Each suppression documents the real bug/noise source with issue numbers.
    """
    # ── Third-party loggers ──────────────────────────────────────────────
    # Suppress noisy third-party loggers to WARNING (errors still surface).
    for name in (
        "httpx",                    # HTTP client debug
        "httpcore",                 # HTTP connection pool debug
        "hpack",                    # HTTP/2 header compression debug
        "websockets",               # WebSocket protocol debug
        "asyncio",                  # Event loop debug
        "realtime._async.client",   # Supabase Realtime full JSON payloads
        "realtime._async.channel",  # Supabase Realtime channel events
        "realtime",                 # Supabase Realtime parent
        "urllib3.connectionpool",   # ~633 lines – HTTP pool debug (keepalive)
        "kalshiflow.auth",          # ~2,104 lines – RSA signature debug
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    # BUG-4 (HIGH): OpenAI and Anthropic _base_client DEBUG loggers dump entire HTML
    # pages and full request/response payloads (including system prompts, tool
    # definitions, and third-party API keys embedded in error pages).
    for name in ("openai", "openai._base_client",
                 "anthropic", "anthropic._base_client"):
        logging.getLogger(name).setLevel(logging.INFO)

    # IMP-1: Event bus DEBUG logging is ~33% of all output.
    # "Processing event: public_trade_received" and orderbook_delta events produce
    # ~9,759 lines in 3 minutes. Set to INFO to keep operational messages only.
    logging.getLogger("kalshiflow_rl.traderv3.event_bus").setLevel(logging.INFO)

    # ── Problem 1 fix: Internal V3 loggers (CORRECTED logger names) ──────
    # Round 2 suppressions had wrong logger names (services.* vs core.*, extra
    # path segments). These are the ACTUAL getLogger() names from each module.
    # All set to WARNING so ERRORs still surface.
    for name in (
        # 28.4% of output – lifecycle client heartbeat/sync chatter
        "kalshiflow_rl.traderv3.clients.lifecycle_client",
        # ~3,486 lines – "Broadcast trading state" every 0.5s
        # WAS: "kalshiflow_rl.traderv3.services.status_reporter" (wrong path)
        "kalshiflow_rl.traderv3.core.status_reporter",
        # ~2,339 lines – "Market price updated" on every tick
        # WAS: "kalshiflow_rl.traderv3.services.state_container" (wrong path)
        "kalshiflow_rl.traderv3.core.state_container",
        # ~5,651 lines – "Ticker update" per update
        # WAS: "kalshiflow_rl.traderv3.clients.market_ticker_listener" (wrong path)
        "kalshiflow_rl.traderv3.market_ticker_listener",
        # ~4,252 lines – trading sync chatter
        "kalshiflow_rl.traderv3.clients.trading_client_integration",
        # ~1,031 lines – "Retrieved X markets" every sync
        "kalshiflow_rl.traderv3.clients.demo_client",
        # ~809 lines – connection mgmt debug
        # WAS: "kalshiflow_rl.traderv3.core.websocket_manager" (wrong path)
        "kalshiflow_rl.traderv3.websocket_manager",
        # ~715 lines – lifecycle event processing
        "kalshiflow_rl.traderv3.services.event_lifecycle_service",
        # ~716 lines – "Lifecycle event stored" per DB write
        # WAS: "kalshiflow_rl.traderv3.database" (wrong path – actual is data layer)
        "kalshiflow_rl.database",
        # ~285 lines – "Orderbook unhealthy" every 5s
        # WAS: "kalshiflow_rl.traderv3.services.health_monitor" (wrong path)
        "kalshiflow_rl.traderv3.core.health_monitor",
        # Additional noisy data-layer loggers
        "kalshiflow_rl.orderbook_client",
        "kalshiflow_rl.write_queue",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


_configure_logging()

# Global coordinator instance
coordinator: V3Coordinator = None


@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager."""
    global coordinator
    
    try:
        logger.info("=" * 60)
        logger.info("TRADER V3 APPLICATION STARTING")
        logger.info("=" * 60)
        
        # Load environment configuration
        load_dotenv()
        config = load_config()
        
        # Initialize database for orderbook data persistence
        logger.info("Initializing database...")
        await rl_db.initialize()

        # Initialize order context service with database pool
        logger.info("Initializing order context service...")
        order_context_service = get_order_context_service()
        await order_context_service.initialize(db_pool=rl_db._pool)

        # Start write queue for async database writes
        logger.info("Starting write queue...")
        write_queue = get_write_queue()
        await write_queue.start()
        
        # Create core components
        logger.info("Creating V3 components...")
        
        # 1. Event bus for communication
        event_bus = EventBus()
        
        # 2. State machine for lifecycle
        state_machine = V3StateMachine(event_bus=event_bus)
        
        # 3. WebSocket manager for frontend
        websocket_manager = V3WebSocketManager(event_bus=event_bus, state_machine=state_machine)
        
        # 4. Select market tickers
        market_tickers = config.market_tickers
        if market_tickers:
            logger.info(f"Using {len(market_tickers)} target tickers")
        else:
            logger.info("No target tickers - lifecycle discovery will manage subscriptions")
        
        # 5. Create orderbook client with selected markets and V3 event bus
        # Pass V3's event bus to OrderbookClient for direct integration
        # This allows orderbook events to flow directly to V3 components without global coupling
        orderbook_client = OrderbookClient(
            market_tickers=market_tickers,
            event_bus=event_bus  # V3's isolated event bus
        )
        
        # 6. Orderbook integration layer
        orderbook_integration = V3OrderbookIntegration(
            orderbook_client=orderbook_client,
            event_bus=event_bus,
            market_tickers=market_tickers  # Use discovered/configured tickers
        )
        
        # 7. Trading client integration (optional)
        trading_client_integration = None
        if config.enable_trading_client:
            logger.info(f"Creating trading client integration (mode={config.trading_mode})...")
            
            # Import and create the demo trading client
            from src.kalshiflow_rl.traderv3.clients.demo_client import KalshiDemoTradingClient
            
            # Create trading client (paper mode for now)
            trading_client = KalshiDemoTradingClient(mode="paper")
            
            # Create integration wrapper
            trading_client_integration = V3TradingClientIntegration(
                trading_client=trading_client,
                event_bus=event_bus,
                max_orders=config.trading_max_orders,
                max_position_size=config.trading_max_position_size
            )
            
            # Start the trading client integration
            await trading_client_integration.start()

            # Wire trading client to orderbook integration for REST fallback
            orderbook_integration.set_trading_client(trading_client)

            logger.info(f"Trading client integration created (max_orders={config.trading_max_orders}, max_position={config.trading_max_position_size})")
        else:
            logger.info("Trading client disabled - orderbook only mode")

        # 8. Create trades client (required for strategies that need PUBLIC_TRADE_RECEIVED events)
        trades_integration = None

        # Trades stream always enabled (TradeFlowService + deep agent need it)
        if config.enable_trading_client:
            logger.info("Creating trades client...")

            # Create KalshiAuth for trades WebSocket
            auth = KalshiAuth.from_env()

            # Create TradesClient (connects to public trades stream)
            trades_client = TradesClient(
                ws_url=config.ws_url,
                auth=auth,
            )

            # Create V3TradesIntegration wrapper
            trades_integration = V3TradesIntegration(
                trades_client=trades_client,
                event_bus=event_bus,
            )

            logger.info("Trades stream ENABLED - strategies will receive PUBLIC_TRADE_RECEIVED events")
        else:
            logger.info("Trades stream disabled (trading client disabled)")

        # 9. Create coordinator with discovered/configured markets
        # Update config with the actual markets being used
        config.market_tickers = market_tickers

        coordinator = V3Coordinator(
            config=config,
            state_machine=state_machine,
            event_bus=event_bus,
            websocket_manager=websocket_manager,
            orderbook_integration=orderbook_integration,
            trading_client_integration=trading_client_integration,
            trades_integration=trades_integration,
        )
        
        # Start the system
        await coordinator.start()
        
        logger.info("[V3:STARTUP] environment=%s markets=%d arb=%s orchestrator=%s",
                     config.get_environment_name(), len(config.market_tickers),
                     config.arb_enabled, config.arb_orchestrator_enabled)
        logger.info("TRADER V3 ready to serve requests")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start TRADER V3: {e}")
        raise
        
    finally:
        # Cleanup on shutdown
        logger.info("TRADER V3 shutting down...")
        
        if coordinator:
            try:
                await coordinator.stop()
            except Exception as e:
                logger.error(f"Error during coordinator shutdown: {e}")
        
        # Stop write queue
        try:
            write_queue = get_write_queue()
            await write_queue.stop()
            logger.info("Write queue stopped")
        except Exception as e:
            logger.error(f"Error stopping write queue: {e}")

        # Cleanup order context service (before closing database)
        try:
            order_context_service = get_order_context_service()
            await order_context_service.cleanup()
            logger.info("Order context service cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up order context service: {e}")

        # Close database
        try:
            await rl_db.close()
            logger.info("Database closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")
        
        logger.info("[V3:SHUTDOWN] complete")
        logger.info("TRADER V3 shutdown complete")


async def health_endpoint(request):
    """Health check endpoint."""
    if not coordinator:
        return JSONResponse(
            {"status": "unhealthy", "error": "System not initialized"},
            status_code=503
        )
    
    health = coordinator.get_health()
    status_code = 200 if health["healthy"] else 503
    
    return JSONResponse(health, status_code=status_code)


async def status_endpoint(request):
    """Detailed status endpoint."""
    if not coordinator:
        return JSONResponse(
            {"error": "System not initialized"},
            status_code=503
        )

    return JSONResponse(coordinator.get_status())


async def cleanup_endpoint(request):
    """
    Cleanup endpoint - cancels open orders.

    POST /v3/cleanup
    POST /v3/cleanup?orphaned_only=true  (default, cancel only orphaned orders)
    POST /v3/cleanup?orphaned_only=false (cancel ALL orders)

    Query Parameters:
        orphaned_only: If true (default), only cancel orders without order_group_id.
                       If false, cancel all open orders regardless of order_group_id.

    Returns:
        JSON with count of cancelled/preserved orders
    """
    if not coordinator:
        return JSONResponse(
            {"error": "System not initialized"},
            status_code=503
        )

    # Check if trading client integration is available
    if not coordinator._trading_client_integration:
        return JSONResponse(
            {"error": "Trading client not configured"},
            status_code=400
        )

    # Check query parameter for cleanup mode (default: orphaned_only=true)
    orphaned_only = request.query_params.get("orphaned_only", "true").lower() == "true"

    try:
        if orphaned_only:
            # Cancel only orphaned orders (smart cleanup)
            result = await coordinator._trading_client_integration.cancel_orphaned_orders()

            cancelled_count = len(result.get("cancelled", []))
            preserved_count = len(result.get("skipped", []))
            error_count = len(result.get("errors", []))

            logger.info(
                f"Orphaned cleanup complete: {cancelled_count} cancelled, "
                f"{preserved_count} preserved, {error_count} errors"
            )

            return JSONResponse({
                "success": True,
                "mode": "orphaned_only",
                "cancelled_count": cancelled_count,
                "preserved_count": preserved_count,
                "error_count": error_count,
                "cancelled": result.get("cancelled", []),
                "preserved": result.get("skipped", []),
                "errors": result.get("errors", [])
            })
        else:
            # Cancel all open orders (legacy behavior)
            result = await coordinator._trading_client_integration.cancel_all_orders()

            cancelled_count = len(result.get("cancelled", []))
            error_count = len(result.get("errors", []))

            logger.info(f"Full cleanup complete: {cancelled_count} orders cancelled, {error_count} errors")

            return JSONResponse({
                "success": True,
                "mode": "all",
                "cancelled_count": cancelled_count,
                "error_count": error_count,
                "cancelled": result.get("cancelled", []),
                "errors": result.get("errors", [])
            })

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return JSONResponse(
            {"error": str(e), "success": False},
            status_code=500
        )


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    if not coordinator or not coordinator._websocket_manager:
        await websocket.close(code=1003, reason="System not ready")
        return

    # Delegate to WebSocket manager
    await coordinator._websocket_manager.handle_websocket(websocket)


async def export_order_contexts_endpoint(request: Request):
    """
    Export order contexts as CSV for quant analysis.

    GET /v3/export/order-contexts

    Query Parameters:
        strategy: Filter by strategy (e.g., 'rlm_no')
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        settled_only: Only include settled orders (default: true)
        format: 'csv' (default) or 'json'

    Returns:
        CSV file download or JSON array
    """
    try:
        # Parse query params
        strategy = request.query_params.get("strategy")
        from_date = request.query_params.get("from_date") or request.query_params.get("from")
        to_date = request.query_params.get("to_date") or request.query_params.get("to")
        settled_only = request.query_params.get("settled_only", "true").lower() == "true"
        output_format = request.query_params.get("format", "csv")

        # Get order context service
        order_context_service = get_order_context_service()

        # Query contexts
        contexts = await order_context_service.get_contexts_for_export(
            strategy=strategy,
            from_date=from_date,
            to_date=to_date,
            settled_only=settled_only,
        )

        if output_format == "json":
            # Convert non-JSON-serializable types
            from decimal import Decimal
            for ctx in contexts:
                for key, value in list(ctx.items()):
                    if value is None:
                        continue
                    # Convert datetime to ISO string
                    if hasattr(value, "isoformat"):
                        ctx[key] = value.isoformat()
                    # Convert Decimal to float
                    elif isinstance(value, Decimal):
                        ctx[key] = float(value)
            return JSONResponse({
                "order_contexts": contexts,
                "count": len(contexts),
                "filters": {
                    "strategy": strategy,
                    "from_date": from_date,
                    "to_date": to_date,
                    "settled_only": settled_only,
                }
            })

        # Generate CSV
        csv_content = order_context_service.generate_csv(contexts)

        from datetime import datetime
        filename = f"order_contexts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except Exception as e:
        logger.error(f"Export order contexts failed: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


async def pairs_endpoint(request: Request) -> JSONResponse:
    """List active arb pairs with current spread data."""
    if not coordinator:
        return JSONResponse({"error": "System not initialized"}, status_code=503)

    pair_registry = coordinator._pair_registry
    if not pair_registry:
        return JSONResponse({"pairs": [], "count": 0, "arb_enabled": False})

    pairs = [
        {
            "id": p.id,
            "kalshi_ticker": p.kalshi_ticker,
            "kalshi_event_ticker": p.kalshi_event_ticker,
            "poly_condition_id": p.poly_condition_id,
            "poly_token_id_yes": p.poly_token_id_yes,
            "question": p.question,
            "match_method": p.match_method,
            "match_confidence": p.match_confidence,
            "status": p.status,
        }
        for p in pair_registry.get_all_active()
    ]

    return JSONResponse({
        "pairs": pairs,
        "count": len(pairs),
        "arb_enabled": True,
        "poller_status": coordinator._poly_poller.get_status() if coordinator._poly_poller else None,
    })


async def agent_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for agent interaction/streaming."""
    # Capture reference atomically to avoid race during shutdown
    arb_strategy = coordinator._arb_strategy if coordinator else None
    if not arb_strategy:
        await websocket.close(code=1003, reason="Agent not running")
        return

    from src.kalshiflow_rl.traderv3.ws_agent_handler import AgentWebSocketHandler
    handler = AgentWebSocketHandler(arb_strategy=arb_strategy)
    await handler.handle_websocket(websocket)


# Create Starlette application
app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/v3/health", health_endpoint),
        Route("/v3/status", status_endpoint),
        Route("/v3/cleanup", cleanup_endpoint, methods=["POST"]),
        Route("/v3/export/order-contexts", export_order_contexts_endpoint),
        Route("/v3/pairs", pairs_endpoint),
        WebSocketRoute("/v3/ws", websocket_endpoint),
        WebSocketRoute("/v3/ws/agent", agent_websocket_endpoint),
    ]
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn
    
    # Load configuration to get port
    load_dotenv()
    try:
        config = load_config()
        port = config.port
        host = config.host
    except Exception:
        port = 8005
        host = "0.0.0.0"
    
    logger.info(f"Starting TRADER V3 on {host}:{port}")
    
    uvicorn.run(
        "src.kalshiflow_rl.traderv3.app:app",
        host=host,
        port=port,
        log_level="info",
        reload=False  # Don't reload in production
    )