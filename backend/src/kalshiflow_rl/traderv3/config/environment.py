"""
TRADER V3 Environment Configuration.

Simple, centralized environment configuration for the V3 trader.
Loads from environment variables with sensible defaults.
"""

import os
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("kalshiflow_rl.traderv3.config.environment")


@dataclass
class V3Config:
    """
    V3 Trader configuration loaded from environment variables.
    
    Simple, flat configuration structure with sensible defaults.
    """
    
    # API Configuration
    api_url: str
    ws_url: str
    api_key_id: str
    private_key_content: str
    
    # Market Configuration
    market_tickers: List[str]
    max_markets: int = 10
    
    # Orderbook Configuration  
    orderbook_depth: int = 5
    snapshot_interval: float = 1.0  # seconds
    
    # Trading Client Configuration (optional, only for paper/live trading)
    enable_trading_client: bool = False
    trading_max_orders: int = 10
    trading_max_position_size: int = 100
    trading_mode: str = "paper"  # paper or production
    
    # State Machine Configuration
    sync_duration: float = 10.0  # seconds for Kalshi data sync
    health_check_interval: float = 5.0  # seconds
    error_recovery_delay: float = 30.0  # seconds
    
    # WebSocket Configuration
    ws_reconnect_interval: float = 5.0  # seconds
    ws_max_reconnect_attempts: int = 10
    ws_ping_interval: float = 30.0  # seconds
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8005
    
    # Logging Configuration
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "V3Config":
        """
        Load configuration from environment variables.
        
        Returns:
            V3Config instance
            
        Raises:
            ValueError: If required environment variables are missing
        """
        # Required environment variables
        api_url = os.environ.get("KALSHI_API_URL")
        ws_url = os.environ.get("KALSHI_WS_URL")
        api_key_id = os.environ.get("KALSHI_API_KEY_ID")
        private_key_content = os.environ.get("KALSHI_PRIVATE_KEY_CONTENT")
        
        # Validate required variables
        missing = []
        if not api_url:
            missing.append("KALSHI_API_URL")
        if not ws_url:
            missing.append("KALSHI_WS_URL")
        if not api_key_id:
            missing.append("KALSHI_API_KEY_ID")
        if not private_key_content:
            missing.append("KALSHI_PRIVATE_KEY_CONTENT")
            
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        # Market configuration - use same variables as RL trader
        mode = os.environ.get("RL_MODE", "discovery")
        market_limit = int(os.environ.get("RL_ORDERBOOK_MARKET_LIMIT", "10"))
        
        # Handle discovery mode or specific market tickers
        if mode == "discovery":
            # Discovery mode - OrderbookClient will auto-discover markets
            market_tickers = ["DISCOVERY"]  # Special marker for discovery mode
            logger.info(f"Discovery mode enabled with limit of {market_limit} markets")
        else:
            # Config mode - use specific market tickers
            market_tickers_str = os.environ.get("RL_MARKET_TICKERS", "INXD-25JAN03")
            market_tickers = [t.strip() for t in market_tickers_str.split(",") if t.strip()]
            if not market_tickers:
                logger.warning("No market tickers configured, using default: INXD-25JAN03")
                market_tickers = ["INXD-25JAN03"]
        
        # Optional configuration with defaults
        max_markets = market_limit  # Use the market limit as max markets
        orderbook_depth = int(os.environ.get("V3_ORDERBOOK_DEPTH", "5"))
        snapshot_interval = float(os.environ.get("V3_SNAPSHOT_INTERVAL", "1.0"))
        
        # Trading client configuration
        enable_trading_client = os.environ.get("V3_ENABLE_TRADING_CLIENT", "false").lower() == "true"
        trading_max_orders = int(os.environ.get("V3_TRADING_MAX_ORDERS", "10"))
        trading_max_position_size = int(os.environ.get("V3_TRADING_MAX_POSITION_SIZE", "100"))
        # Determine trading mode based on environment or explicit setting
        environment = os.environ.get("ENVIRONMENT", "local")
        if environment == "paper" or "demo-api" in ws_url.lower():
            trading_mode = "paper"
        else:
            trading_mode = os.environ.get("V3_TRADING_MODE", "paper")
        
        sync_duration = float(os.environ.get("V3_SYNC_DURATION", os.environ.get("V3_CALIBRATION_DURATION", "10.0")))
        health_check_interval = float(os.environ.get("V3_HEALTH_CHECK_INTERVAL", "5.0"))
        error_recovery_delay = float(os.environ.get("V3_ERROR_RECOVERY_DELAY", "30.0"))
        
        ws_reconnect_interval = float(os.environ.get("V3_WS_RECONNECT_INTERVAL", "5.0"))
        ws_max_reconnect_attempts = int(os.environ.get("V3_WS_MAX_RECONNECT_ATTEMPTS", "10"))
        ws_ping_interval = float(os.environ.get("V3_WS_PING_INTERVAL", "30.0"))
        
        host = os.environ.get("V3_HOST", "0.0.0.0")
        port = int(os.environ.get("V3_PORT", "8005"))
        
        log_level = os.environ.get("V3_LOG_LEVEL", "INFO").upper()
        
        config = cls(
            api_url=api_url,
            ws_url=ws_url,
            api_key_id=api_key_id,
            private_key_content=private_key_content,
            market_tickers=market_tickers,
            max_markets=max_markets,
            orderbook_depth=orderbook_depth,
            snapshot_interval=snapshot_interval,
            enable_trading_client=enable_trading_client,
            trading_max_orders=trading_max_orders,
            trading_max_position_size=trading_max_position_size,
            trading_mode=trading_mode,
            sync_duration=sync_duration,
            health_check_interval=health_check_interval,
            error_recovery_delay=error_recovery_delay,
            ws_reconnect_interval=ws_reconnect_interval,
            ws_max_reconnect_attempts=ws_max_reconnect_attempts,
            ws_ping_interval=ws_ping_interval,
            host=host,
            port=port,
            log_level=log_level
        )
        
        logger.info(f"Loaded V3 configuration:")
        logger.info(f"  - API URL: {api_url}")
        logger.info(f"  - WebSocket URL: {ws_url}")
        logger.info(f"  - Markets: {', '.join(market_tickers[:3])}{'...' if len(market_tickers) > 3 else ''} ({len(market_tickers)} total)")
        logger.info(f"  - Max markets: {max_markets}")
        logger.info(f"  - Sync duration: {sync_duration}s")
        logger.info(f"  - Server: {host}:{port}")
        logger.info(f"  - Log level: {log_level}")
        if enable_trading_client:
            logger.info(f"  - Trading enabled: {trading_mode} mode")
            logger.info(f"  - Max orders: {trading_max_orders}, Max position: {trading_max_position_size}")
        else:
            logger.info(f"  - Trading: DISABLED (orderbook only)")
        
        return config
    
    def is_demo_environment(self) -> bool:
        """Check if using demo API endpoints."""
        return "demo-api" in self.api_url.lower()
    
    def get_environment_name(self) -> str:
        """Get human-readable environment name."""
        if self.is_demo_environment():
            return "DEMO (Paper Trading)"
        else:
            return "PRODUCTION"
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate market configuration
        if not self.market_tickers:
            raise ValueError("No market tickers configured")
        
        if self.max_markets < 1:
            raise ValueError(f"Invalid max_markets: {self.max_markets}")
        
        if self.max_markets < len(self.market_tickers):
            logger.warning(f"Limiting markets from {len(self.market_tickers)} to {self.max_markets}")
            self.market_tickers = self.market_tickers[:self.max_markets]
        
        # Validate timing configuration
        if self.sync_duration < 1.0:
            raise ValueError(f"Sync duration too short: {self.sync_duration}s")
        
        if self.snapshot_interval < 0.1:
            raise ValueError(f"Snapshot interval too short: {self.snapshot_interval}s")
        
        # Validate server configuration
        if self.port < 1024 or self.port > 65535:
            raise ValueError(f"Invalid port: {self.port}")
        
        # Validate WebSocket configuration
        if self.ws_reconnect_interval < 1.0:
            raise ValueError(f"WebSocket reconnect interval too short: {self.ws_reconnect_interval}s")
        
        if self.ws_max_reconnect_attempts < 1:
            raise ValueError(f"Invalid max reconnect attempts: {self.ws_max_reconnect_attempts}")
        
        logger.info(f"✅ Configuration validated successfully")
        return True


def load_config() -> V3Config:
    """
    Load and validate V3 configuration from environment.
    
    Returns:
        Validated V3Config instance
        
    Raises:
        ValueError: If configuration is invalid
    """
    config = V3Config.from_env()
    config.validate()
    return config