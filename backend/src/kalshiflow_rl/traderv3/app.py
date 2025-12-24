"""
TRADER V3 Standalone Application.

Clean, simple Starlette application for V3 trader.
Runs on port 8005 with minimal dependencies.
"""

import asyncio
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import V3 components
from src.kalshiflow_rl.traderv3.core.state_machine import TraderStateMachine as V3StateMachine, TraderState as V3State
from src.kalshiflow_rl.traderv3.core.event_bus import EventBus
from src.kalshiflow_rl.traderv3.core.websocket_manager import V3WebSocketManager
from src.kalshiflow_rl.traderv3.core.coordinator import V3Coordinator
from src.kalshiflow_rl.traderv3.clients.orderbook_integration import V3OrderbookIntegration
from src.kalshiflow_rl.traderv3.config.environment import load_config

# Import existing orderbook client
from src.kalshiflow_rl.data.orderbook_client import OrderbookClient

# Import database and write queue for orderbook data persistence
from src.kalshiflow_rl.data.database import rl_db
from src.kalshiflow_rl.data.write_queue import get_write_queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("kalshiflow_rl.traderv3")

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
        
        # 4. Select market tickers (discovery or config mode)
        if config.market_tickers == ["DISCOVERY"]:
            # Discovery mode - fetch active markets
            logger.info(f"Discovery mode: fetching up to {config.max_markets} active markets...")
            from kalshiflow_rl.data.market_discovery import fetch_active_markets
            discovered_tickers = await fetch_active_markets(limit=config.max_markets)
            if discovered_tickers:
                market_tickers = discovered_tickers
                logger.info(f"Discovered {len(discovered_tickers)} active markets")
            else:
                logger.warning("No active markets discovered, using default")
                market_tickers = ["INXD-25JAN03"]
        else:
            # Config mode - use specified tickers
            market_tickers = config.market_tickers
            logger.info(f"Config mode: using {len(market_tickers)} configured markets")
        
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
        
        # 7. Create coordinator with discovered/configured markets
        # Update config with the actual markets being used
        config.market_tickers = market_tickers
        
        coordinator = V3Coordinator(
            config=config,
            state_machine=state_machine,
            event_bus=event_bus,
            websocket_manager=websocket_manager,
            orderbook_integration=orderbook_integration
        )
        
        # Start the system
        await coordinator.start()
        
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
        
        # Close database
        try:
            await rl_db.close()
            logger.info("Database closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")
        
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


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    if not coordinator or not coordinator._websocket_manager:
        await websocket.close(code=1003, reason="System not ready")
        return
    
    # Delegate to WebSocket manager
    await coordinator._websocket_manager.handle_websocket(websocket)


# Create Starlette application
app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/v3/health", health_endpoint),
        Route("/v3/status", status_endpoint),
        WebSocketRoute("/v3/ws", websocket_endpoint)
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