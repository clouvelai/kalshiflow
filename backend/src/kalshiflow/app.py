"""Main Starlette application with WebSocket support for Kalshi trade streaming"""

import os
import asyncio
import logging
from datetime import datetime
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our services
from .trade_processor import get_trade_processor
from .websocket_handler import get_websocket_manager, TradeStreamEndpoint
from .kalshi_client import KalshiWebSocketClient
from .database import get_database
from .aggregator import get_aggregator
from .market_metadata_service import initialize_metadata_service, get_metadata_service
from .auth import KalshiAuth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def health_check(request):
    """Health check endpoint to verify the server is running"""
    return JSONResponse({
        "status": "healthy",
        "service": "kalshiflow-backend",
        "version": "0.1.0"
    })

async def get_config(request):
    """Return non-sensitive configuration for the frontend"""
    return JSONResponse({
        "window_minutes": int(os.getenv("WINDOW_MINUTES", "10")),
        "hot_markets_limit": int(os.getenv("HOT_MARKETS_LIMIT", "12")),
        "recent_trades_limit": int(os.getenv("RECENT_TRADES_LIMIT", "200"))
    })

async def get_hot_markets(request):
    """Get current hot markets with metadata"""
    try:
        aggregator = get_aggregator()
        hot_markets = await aggregator.get_hot_markets_with_metadata()
        return JSONResponse({
            "hot_markets": hot_markets,
            "count": len(hot_markets),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting hot markets: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

async def get_recent_trades(request):
    """Get recent trades for debugging"""
    try:
        aggregator = get_aggregator()
        recent_trades = aggregator.get_recent_trades()
        return JSONResponse({
            "recent_trades": recent_trades,
            "count": len(recent_trades),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting recent trades: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

async def get_ticker_trades(request):
    """Get trades for a specific ticker"""
    ticker = request.path_params['ticker']
    try:
        database = get_database()
        trades = await database.get_trades_for_ticker(ticker, limit=100)
        return JSONResponse({
            "ticker": ticker,
            "trades": trades,
            "count": len(trades),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting trades for ticker {ticker}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

async def get_stats(request):
    """Get system statistics for debugging"""
    try:
        trade_processor = get_trade_processor()
        websocket_manager = get_websocket_manager()
        
        # Get stats and handle datetime serialization
        trade_stats = trade_processor.get_stats()
        
        # Convert datetime objects to ISO strings
        if trade_stats.get("started_at"):
            trade_stats["started_at"] = trade_stats["started_at"].isoformat()
        if trade_stats.get("last_trade_time"):
            trade_stats["last_trade_time"] = trade_stats["last_trade_time"].isoformat()
            
        stats = {
            "trade_processor": trade_stats,
            "websocket": websocket_manager.get_stats(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Try to get database stats, but don't fail if it errors
        try:
            database = get_database()
            db_stats = await database.get_db_stats()
            stats["database"] = db_stats
        except Exception as db_error:
            stats["database"] = {"error": str(db_error)}
        
        # Try to get metadata service stats
        try:
            metadata_service = get_metadata_service()
            if metadata_service:
                metadata_stats = await metadata_service.get_service_status()
                stats["metadata_service"] = metadata_stats
            else:
                stats["metadata_service"] = {"status": "not_initialized"}
        except Exception as meta_error:
            stats["metadata_service"] = {"error": str(meta_error)}
        
        return JSONResponse(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# Define routes
routes = [
    Route("/health", health_check, methods=["GET"]),
    Route("/api/config", get_config, methods=["GET"]),
    Route("/api/markets/hot", get_hot_markets, methods=["GET"]),
    Route("/api/trades/recent", get_recent_trades, methods=["GET"]),
    Route("/api/markets/{ticker}/trades", get_ticker_trades, methods=["GET"]),
    Route("/api/stats", get_stats, methods=["GET"]),
    WebSocketRoute("/ws/stream", TradeStreamEndpoint),
]

# Global variables for background services
kalshi_client = None
background_tasks = set()

async def startup_event():
    """Initialize services on application startup"""
    global kalshi_client
    
    logger.info("Starting Kalshi Flowboard backend services...")
    
    try:
        # Initialize trade processor
        trade_processor = get_trade_processor()
        await trade_processor.start()
        
        # Initialize WebSocket manager
        websocket_manager = get_websocket_manager()
        await websocket_manager.initialize()
        
        # Initialize metadata service
        try:
            database = get_database()
            auth = KalshiAuth.from_env()
            metadata_service = initialize_metadata_service(database, auth)
            await metadata_service.start()
            logger.info("Market metadata service started successfully")
        except Exception as e:
            logger.warning(f"Failed to start metadata service (continuing without metadata enhancement): {e}")
        
        # Create Kalshi client with trade callback
        async def trade_callback(trade):
            """Callback to handle trades from Kalshi client"""
            await trade_processor.process_trade(trade)
        
        kalshi_client = KalshiWebSocketClient.from_env(
            on_trade_callback=trade_callback
        )
        
        # Start Kalshi client as background task
        task = asyncio.create_task(kalshi_client.start())
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        
        logger.info("All services started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start services: {e}")
        raise

async def shutdown_event():
    """Clean up services on application shutdown"""
    global kalshi_client
    
    logger.info("Shutting down Kalshi Flowboard backend services...")
    
    try:
        # Stop Kalshi client
        if kalshi_client:
            await kalshi_client.stop()
        
        # Cancel all background tasks
        for task in background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        
        # Stop metadata service
        try:
            metadata_service = get_metadata_service()
            if metadata_service:
                await metadata_service.stop()
                logger.info("Metadata service stopped")
        except Exception as e:
            logger.warning(f"Error stopping metadata service: {e}")
        
        # Stop trade processor
        trade_processor = get_trade_processor()
        await trade_processor.stop()
        
        logger.info("All services shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create the Starlette application
app = Starlette(
    debug=True,
    routes=routes,
    on_startup=[startup_event],
    on_shutdown=[shutdown_event]
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)