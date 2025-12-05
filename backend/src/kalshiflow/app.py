"""Main Starlette application with WebSocket support for Kalshi trade streaming"""

import os
import asyncio
import logging
import json
from datetime import datetime, date
from decimal import Decimal
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our services
from .trade_processor import get_trade_processor
from .websocket_handler import get_websocket_manager, TradeStreamEndpoint
from .kalshi_client import KalshiWebSocketClient
from .database_factory import get_current_database, initialize_database, close_database, DatabaseFactory
from .aggregator import get_aggregator
from .market_metadata_service import initialize_metadata_service, get_metadata_service
from .time_analytics_service import get_analytics_service
from .auth import KalshiAuth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def custom_json_response(data, status_code=200):
    """Create a JSONResponse with custom serialization for Decimal and datetime objects"""
    def custom_encoder(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError(repr(obj) + " is not JSON serializable")
    
    json_str = json.dumps(data, default=custom_encoder)
    return Response(json_str, media_type="application/json", status_code=status_code)


async def health_check(request):
    """Health check endpoint to verify the server is running"""
    return JSONResponse({
        "status": "healthy",
        "service": "kalshiflow-backend",
        "version": "0.1.0"
    })

async def health_ready(request):
    """Health check endpoint indicating if the server is ready to serve requests"""
    global recovery_status
    
    if recovery_status["is_complete"]:
        return JSONResponse({
            "status": "ready",
            "service": "kalshiflow-backend",
            "recovery": {
                "enabled": recovery_status["recovery_enabled"],
                "completed_at": recovery_status["completed_at"],
                "duration_seconds": recovery_status["duration_seconds"],
                "success": recovery_status.get("stats", {}).get("analytics", {}).get("success", False) and 
                          recovery_status.get("stats", {}).get("aggregator", {}).get("success", False)
            }
        })
    else:
        return JSONResponse({
            "status": "not_ready",
            "service": "kalshiflow-backend", 
            "message": "Recovery in progress",
            "recovery": {
                "enabled": recovery_status["recovery_enabled"],
                "started_at": recovery_status["started_at"]
            }
        }, status_code=503)

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
        return custom_json_response({
            "hot_markets": hot_markets,
            "count": len(hot_markets),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting hot markets: {e}")
        return custom_json_response({"error": str(e)}, status_code=500)

async def get_recent_trades(request):
    """Get recent trades for debugging"""
    try:
        aggregator = get_aggregator()
        recent_trades = aggregator.get_recent_trades()
        return custom_json_response({
            "recent_trades": recent_trades,
            "count": len(recent_trades),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting recent trades: {e}")
        return custom_json_response({"error": str(e)}, status_code=500)

async def get_ticker_trades(request):
    """Get trades for a specific ticker"""
    ticker = request.path_params['ticker']
    try:
        database = get_current_database()
        trades = await database.get_trades_for_ticker(ticker, limit=100)
        return custom_json_response({
            "ticker": ticker,
            "trades": trades,
            "count": len(trades),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting trades for ticker {ticker}: {e}")
        return custom_json_response({"error": str(e)}, status_code=500)

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
            database = get_current_database()
            db_stats = await database.get_db_stats()
            db_stats["database_type"] = DatabaseFactory.get_database_type()
            stats["database"] = db_stats
        except Exception as db_error:
            stats["database"] = {"error": str(db_error), "database_type": DatabaseFactory.get_database_type()}
        
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
        
        # Try to get analytics service stats
        try:
            analytics_service = get_analytics_service()
            if analytics_service:
                analytics_stats = analytics_service.get_stats()
                # Convert any datetime objects to ISO strings
                for key, value in analytics_stats.items():
                    if isinstance(value, datetime):
                        analytics_stats[key] = value.isoformat()
                stats["analytics_service"] = analytics_stats
            else:
                stats["analytics_service"] = {"status": "not_initialized"}
        except Exception as analytics_error:
            stats["analytics_service"] = {"error": str(analytics_error)}
        
        return custom_json_response(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return custom_json_response({"error": str(e)}, status_code=500)

# Define routes
routes = [
    Route("/health", health_check, methods=["GET"]),
    Route("/health/ready", health_ready, methods=["GET"]),
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

# Global recovery status
recovery_status = {
    "is_complete": False,
    "recovery_enabled": True,
    "started_at": None,
    "completed_at": None,
    "duration_seconds": 0.0,
    "stats": {}
}

async def startup_event():
    """Initialize services on application startup"""
    global kalshi_client, recovery_status
    
    logger.info("Starting Kalshi Flowboard backend services...")
    
    try:
        # Initialize database first
        database = await initialize_database()
        logger.info(f"Database initialized: {DatabaseFactory.get_database_type()}")
        
        # Initialize trade processor
        trade_processor = get_trade_processor()
        await trade_processor.start()
        
        # Initialize WebSocket manager
        websocket_manager = get_websocket_manager()
        await websocket_manager.initialize()
        
        # Check if recovery is enabled via environment variable
        enable_recovery = os.getenv("ENABLE_WARM_RESTART", "true").lower() == "true"
        recovery_status["recovery_enabled"] = enable_recovery
        recovery_status["started_at"] = datetime.now().isoformat()
        
        logger.info(f"Warm restart recovery: {'ENABLED' if enable_recovery else 'DISABLED'}")
        
        # Recovery Phase: Rebuild in-memory state from database
        if enable_recovery:
            logger.info("=== WARM RESTART RECOVERY PHASE ===")
            recovery_start = datetime.now()
            
            try:
                # Recover TimeAnalyticsService first (time series buckets)
                analytics_service = get_analytics_service()
                analytics_recovery_stats = await analytics_service.recover_from_database(enable_recovery=enable_recovery)
                
                # Recover TradeAggregator (ticker states and hot markets)  
                aggregator = get_aggregator()
                aggregator_recovery_stats = await aggregator.warm_start_from_database(enable_recovery=enable_recovery)
                
                # Calculate total recovery time
                recovery_duration = (datetime.now() - recovery_start).total_seconds()
                recovery_status["duration_seconds"] = recovery_duration
                recovery_status["stats"] = {
                    "analytics": analytics_recovery_stats,
                    "aggregator": aggregator_recovery_stats
                }
                
                if analytics_recovery_stats["success"] and aggregator_recovery_stats["success"]:
                    logger.info(f"=== RECOVERY COMPLETED SUCCESSFULLY IN {recovery_duration:.2f}s ===")
                    logger.info(f"Analytics: {analytics_recovery_stats['minute_buckets_created']} minute buckets, {analytics_recovery_stats['hour_buckets_created']} hour buckets")
                    logger.info(f"Aggregator: {aggregator_recovery_stats['tickers_recovered']} tickers, {aggregator_recovery_stats['recent_trades_populated']} recent trades")
                else:
                    logger.warning("Recovery completed with some failures - see logs above")
                
            except Exception as recovery_error:
                logger.error(f"Recovery phase failed: {recovery_error}")
                logger.info("Continuing with cold start")
                recovery_status["stats"]["error"] = str(recovery_error)
            
            recovery_status["completed_at"] = datetime.now().isoformat()
            recovery_status["is_complete"] = True
            logger.info("=== RECOVERY PHASE COMPLETE ===")
        else:
            logger.info("Recovery disabled - starting with empty state (cold start)")
            recovery_status["is_complete"] = True
            recovery_status["completed_at"] = datetime.now().isoformat()
        
        # Initialize metadata service after recovery
        try:
            auth = KalshiAuth.from_env()
            metadata_service = initialize_metadata_service(database, auth)
            await metadata_service.start()
            logger.info("Market metadata service started successfully")
        except Exception as e:
            logger.warning(f"Failed to start metadata service (continuing without metadata enhancement): {e}")
        
        # Create Kalshi client with trade callback (only start after recovery is complete)
        async def trade_callback(trade):
            """Callback to handle trades from Kalshi client"""
            await trade_processor.process_trade(trade)
        
        kalshi_client = KalshiWebSocketClient.from_env(
            on_trade_callback=trade_callback
        )
        
        # Start Kalshi client as background task
        kalshi_task = asyncio.create_task(kalshi_client.start())
        background_tasks.add(kalshi_task)
        kalshi_task.add_done_callback(background_tasks.discard)
        
        logger.info("All services started successfully - ready to accept connections")
        
    except Exception as e:
        logger.error(f"Failed to start services: {e}")
        recovery_status["is_complete"] = True  # Mark as complete even on failure
        recovery_status["completed_at"] = datetime.now().isoformat()
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
        
        # Close database connection
        try:
            await close_database()
            logger.info(f"Database connection closed: {DatabaseFactory.get_database_type()}")
        except Exception as e:
            logger.warning(f"Error closing database: {e}")
        
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