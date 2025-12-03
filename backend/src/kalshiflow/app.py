"""Main Starlette application with WebSocket support for Kalshi trade streaming"""

import os
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Define routes
routes = [
    Route("/health", health_check, methods=["GET"]),
    Route("/api/config", get_config, methods=["GET"])
]

# Create the Starlette application
app = Starlette(
    debug=True,
    routes=routes
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)