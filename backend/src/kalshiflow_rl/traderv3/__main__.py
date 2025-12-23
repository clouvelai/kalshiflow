#!/usr/bin/env python3
"""
TRADER V3 Main Entry Point.
"""

if __name__ == "__main__":
    import uvicorn
    import logging
    from dotenv import load_dotenv
    from .config.environment import load_config
    
    logger = logging.getLogger("kalshiflow_rl.traderv3")
    
    # Load configuration to get port
    load_dotenv()
    try:
        config = load_config()
        port = config.port
        host = config.host
    except Exception as e:
        logger.warning(f"Could not load config: {e}, using defaults")
        port = 8005
        host = "0.0.0.0"
    
    logger.info(f"Starting TRADER V3 on {host}:{port}")
    
    uvicorn.run(
        "kalshiflow_rl.traderv3.app:app",
        host=host,
        port=port,
        log_level="info",
        reload=False  # Don't reload in production
    )