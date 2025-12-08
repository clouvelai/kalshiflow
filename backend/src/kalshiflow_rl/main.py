"""
Main entry point for RL Trading Subsystem.

Provides simple command-line interface to start the RL service
with proper logging and configuration.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshiflow_rl.config import config, logger
from kalshiflow_rl.app import app


async def main():
    """Main entry point for the RL Trading Subsystem."""
    logger.info("=" * 60)
    logger.info("🚀 Kalshi Flow RL Trading Subsystem v0.1.0")
    logger.info("=" * 60)
    logger.info(f"Environment: {config.ENVIRONMENT}")
    logger.info(f"Market Ticker: {config.RL_MARKET_TICKER}")
    logger.info(f"Debug Mode: {config.DEBUG}")
    logger.info("=" * 60)
    
    try:
        # The app lifespan will handle all initialization
        logger.info("🔧 Starting RL Trading Subsystem...")
        logger.info("📊 Health endpoint: http://localhost:8001/rl/health")
        logger.info("📈 Status endpoint: http://localhost:8001/rl/status")
        logger.info("📖 Orderbook snapshot: http://localhost:8001/rl/orderbook/snapshot")
        logger.info("🔄 Use Ctrl+C to shutdown gracefully")
        logger.info("-" * 60)
        
        # Import uvicorn here to avoid startup issues
        import uvicorn
        
        # Run the ASGI application
        config_obj = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8001,  # Different from main app (8000)
            log_level="info" if not config.DEBUG else "debug",
            reload=config.DEBUG,
            reload_dirs=[str(Path(__file__).parent)] if config.DEBUG else None
        )
        
        server = uvicorn.Server(config_obj)
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown signal received")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise
    finally:
        logger.info("✅ RL Trading Subsystem shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())