"""
TRADER V3 Environment Configuration.

Simple, centralized environment configuration for the V3 trader.
Loads from environment variables with sensible defaults.
"""

import os
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import logging

# Import TradingStrategy for type annotation
# Use string annotation to avoid circular import
if TYPE_CHECKING:
    from ..services.trading_decision_service import TradingStrategy as TradingStrategyType

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
    trading_strategy_str: str = "hold"  # Strategy string: "hold", "whale_follower", "paper_test", "rl_model", "yes_80_90"

    # YES 80-90c Strategy Configuration (for YES_80_90 strategy)
    yes8090_min_price: int = 80  # Minimum YES ask price in cents
    yes8090_max_price: int = 90  # Maximum YES ask price in cents
    yes8090_min_liquidity: int = 10  # Minimum contracts at best ask
    yes8090_max_spread: int = 5  # Maximum bid-ask spread in cents
    yes8090_contracts: int = 100  # Contracts per trade (Tier B)
    yes8090_tier_a_contracts: int = 150  # Contracts for Tier A signals (83-87c)
    yes8090_max_concurrent: int = 100  # Maximum concurrent positions

    # RLM (Reverse Line Movement) Strategy Configuration (for RLM_NO strategy)
    # Validated +17.38% edge: When >65% trades are YES but price drops, bet NO
    rlm_yes_threshold: float = 0.65  # Minimum YES trade ratio to trigger signal
    rlm_min_trades: int = 15  # Minimum trades before evaluating signal
    rlm_min_price_drop: int = 0  # Minimum YES price drop in cents (0 = any drop)
    rlm_contracts: int = 100  # Contracts per trade
    rlm_max_concurrent: int = 1000  # Maximum concurrent positions
    rlm_allow_reentry: bool = True  # Allow adding to position on stronger signal
    rlm_orderbook_timeout: float = 2.0  # Timeout for orderbook fetch (seconds)
    rlm_tight_spread: int = 2  # Spread threshold for market order (cents)
    rlm_wide_spread: int = 3  # Spread threshold for limit order (cents)

    # Whale Detection Configuration (optional, for Follow the Whale feature)
    enable_whale_detection: bool = False
    whale_queue_size: int = 10
    whale_window_minutes: int = 5
    whale_min_size_cents: int = 10000  # $100 minimum

    # Cleanup Configuration
    cleanup_on_startup: bool = True  # Cancel orphaned orders on startup (orders without order_group_id)

    # Position/Order Checking Configuration
    allow_multiple_positions_per_market: bool = False  # Skip position check for testing
    allow_multiple_orders_per_market: bool = False  # Skip orders check for testing

    # Market selection mode: "config", "discovery", or "lifecycle" (default)
    # - config: Use static market_tickers list
    # - discovery: Auto-discover from REST API
    # - lifecycle: Market lifecycle events via WebSocket (NEW - default)
    market_mode: str = "lifecycle"

    # Lifecycle Mode Configuration (RL_MODE=lifecycle)
    lifecycle_categories: List[str] = field(default_factory=lambda: ["sports", "media_mentions", "entertainment", "crypto"])
    lifecycle_max_markets: int = 1000  # Maximum tracked markets (orderbook WS limit)
    lifecycle_sync_interval: int = 30  # Seconds between market info syncs

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
        # Mode options: "config", "discovery", "lifecycle" (default)
        market_mode = os.environ.get("RL_MODE", "lifecycle")
        market_limit = int(os.environ.get("RL_ORDERBOOK_MARKET_LIMIT", "10"))

        # Handle market selection based on mode
        if market_mode == "lifecycle":
            # Lifecycle mode - markets discovered via lifecycle WebSocket events
            market_tickers = []  # Empty - lifecycle events will populate
            logger.info(f"Lifecycle mode enabled - markets will be discovered via lifecycle events")
        elif market_mode == "discovery":
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

        # Trading strategy configuration
        # Options: "hold", "whale_follower", "paper_test", "rl_model"
        trading_strategy_str = os.environ.get("V3_TRADING_STRATEGY", "hold").lower()

        # Whale detection configuration
        enable_whale_detection = os.environ.get("V3_ENABLE_WHALE_DETECTION", "false").lower() == "true"
        whale_queue_size = int(os.environ.get("WHALE_QUEUE_SIZE", "10"))
        whale_window_minutes = int(os.environ.get("WHALE_WINDOW_MINUTES", "5"))
        whale_min_size_cents = int(os.environ.get("WHALE_MIN_SIZE_CENTS", "10000"))

        # Cleanup configuration - default True for paper trading, False for production
        cleanup_default = "true" if environment == "paper" or "demo-api" in ws_url.lower() else "false"
        cleanup_on_startup = os.environ.get("V3_CLEANUP_ON_STARTUP", cleanup_default).lower() == "true"

        # Position checking configuration - for testing multiple positions per market
        allow_multiple_positions_per_market = os.environ.get("V3_ALLOW_MULTIPLE_POSITIONS", "false").lower() == "true"
        allow_multiple_orders_per_market = os.environ.get("V3_ALLOW_MULTIPLE_ORDERS", "false").lower() == "true"

        # YES 80-90c strategy configuration
        yes8090_min_price = int(os.environ.get("YES8090_MIN_PRICE", "80"))
        yes8090_max_price = int(os.environ.get("YES8090_MAX_PRICE", "90"))
        yes8090_min_liquidity = int(os.environ.get("YES8090_MIN_LIQUIDITY", "10"))
        yes8090_max_spread = int(os.environ.get("YES8090_MAX_SPREAD", "5"))
        yes8090_contracts = int(os.environ.get("YES8090_CONTRACTS", "100"))
        yes8090_tier_a_contracts = int(os.environ.get("YES8090_TIER_A_CONTRACTS", "150"))
        yes8090_max_concurrent = int(os.environ.get("YES8090_MAX_CONCURRENT", "100"))

        # RLM (Reverse Line Movement) strategy configuration
        rlm_yes_threshold = float(os.environ.get("RLM_YES_THRESHOLD", "0.65"))
        rlm_min_trades = int(os.environ.get("RLM_MIN_TRADES", "15"))
        rlm_min_price_drop = int(os.environ.get("RLM_MIN_PRICE_DROP", "0"))
        rlm_contracts = int(os.environ.get("RLM_CONTRACTS", "100"))
        rlm_max_concurrent = int(os.environ.get("RLM_MAX_CONCURRENT", "1000"))
        rlm_allow_reentry = os.environ.get("RLM_ALLOW_REENTRY", "true").lower() == "true"
        rlm_orderbook_timeout = float(os.environ.get("RLM_ORDERBOOK_TIMEOUT", "2.0"))
        rlm_tight_spread = int(os.environ.get("RLM_TIGHT_SPREAD", "2"))
        rlm_wide_spread = int(os.environ.get("RLM_WIDE_SPREAD", "3"))

        # Lifecycle discovery mode configuration
        lifecycle_categories_str = os.environ.get("LIFECYCLE_CATEGORIES", "sports,media_mentions,entertainment,crypto")
        lifecycle_categories = [c.strip() for c in lifecycle_categories_str.split(",") if c.strip()]
        lifecycle_max_markets = int(os.environ.get("LIFECYCLE_MAX_MARKETS", "1000"))
        lifecycle_sync_interval = int(os.environ.get("LIFECYCLE_SYNC_INTERVAL", "30"))

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
            market_mode=market_mode,
            orderbook_depth=orderbook_depth,
            snapshot_interval=snapshot_interval,
            enable_trading_client=enable_trading_client,
            trading_max_orders=trading_max_orders,
            trading_max_position_size=trading_max_position_size,
            trading_mode=trading_mode,
            trading_strategy_str=trading_strategy_str,
            yes8090_min_price=yes8090_min_price,
            yes8090_max_price=yes8090_max_price,
            yes8090_min_liquidity=yes8090_min_liquidity,
            yes8090_max_spread=yes8090_max_spread,
            yes8090_contracts=yes8090_contracts,
            yes8090_tier_a_contracts=yes8090_tier_a_contracts,
            yes8090_max_concurrent=yes8090_max_concurrent,
            rlm_yes_threshold=rlm_yes_threshold,
            rlm_min_trades=rlm_min_trades,
            rlm_min_price_drop=rlm_min_price_drop,
            rlm_contracts=rlm_contracts,
            rlm_max_concurrent=rlm_max_concurrent,
            rlm_allow_reentry=rlm_allow_reentry,
            rlm_orderbook_timeout=rlm_orderbook_timeout,
            rlm_tight_spread=rlm_tight_spread,
            rlm_wide_spread=rlm_wide_spread,
            enable_whale_detection=enable_whale_detection,
            whale_queue_size=whale_queue_size,
            whale_window_minutes=whale_window_minutes,
            whale_min_size_cents=whale_min_size_cents,
            cleanup_on_startup=cleanup_on_startup,
            allow_multiple_positions_per_market=allow_multiple_positions_per_market,
            allow_multiple_orders_per_market=allow_multiple_orders_per_market,
            lifecycle_categories=lifecycle_categories,
            lifecycle_max_markets=lifecycle_max_markets,
            lifecycle_sync_interval=lifecycle_sync_interval,
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
        logger.info(f"  - Market mode: {market_mode.upper()}")
        if market_mode == "lifecycle":
            logger.info(f"    - Categories: {', '.join(lifecycle_categories)}")
            logger.info(f"    - Max tracked: {lifecycle_max_markets}")
        elif market_tickers:
            logger.info(f"  - Markets: {', '.join(market_tickers[:3])}{'...' if len(market_tickers) > 3 else ''} ({len(market_tickers)} total)")
        logger.info(f"  - Max markets: {max_markets}")
        logger.info(f"  - Sync duration: {sync_duration}s")
        logger.info(f"  - Server: {host}:{port}")
        logger.info(f"  - Log level: {log_level}")
        if enable_trading_client:
            logger.info(f"  - Trading enabled: {trading_mode} mode")
            logger.info(f"  - Trading strategy: {trading_strategy_str.upper()}")
            logger.info(f"  - Max orders: {trading_max_orders}, Max position: {trading_max_position_size}")
            logger.info(f"  - Cleanup on startup: {cleanup_on_startup}")
        else:
            logger.info(f"  - Trading: DISABLED (orderbook only)")
        if enable_whale_detection:
            logger.info(f"  - Whale detection: ENABLED")
            logger.info(f"  - Whale queue: {whale_queue_size} bets, {whale_window_minutes}min window, min ${whale_min_size_cents/100:.2f}")
        else:
            logger.info(f"  - Whale detection: DISABLED")
        if allow_multiple_positions_per_market:
            logger.warning(f"  - ALLOW_MULTIPLE_POSITIONS: ENABLED (testing mode)")
        if allow_multiple_orders_per_market:
            logger.warning(f"  - ALLOW_MULTIPLE_ORDERS: ENABLED (testing mode)")

        # Log YES 80-90 config if strategy is enabled
        if trading_strategy_str == "yes_80_90":
            logger.info(f"  - YES 80-90c Strategy: ENABLED")
            logger.info(f"    - Price range: {yes8090_min_price}-{yes8090_max_price}c")
            logger.info(f"    - Liquidity min: {yes8090_min_liquidity} contracts")
            logger.info(f"    - Max spread: {yes8090_max_spread}c")
            logger.info(f"    - Contracts: {yes8090_contracts} (Tier A: {yes8090_tier_a_contracts})")
            logger.info(f"    - Max concurrent: {yes8090_max_concurrent} positions")

        # Log RLM config if strategy is enabled
        if trading_strategy_str == "rlm_no":
            logger.info(f"  - RLM (Reverse Line Movement) Strategy: ENABLED")
            logger.info(f"    - YES threshold: {rlm_yes_threshold:.0%}")
            logger.info(f"    - Min trades: {rlm_min_trades}")
            logger.info(f"    - Min price drop: {rlm_min_price_drop}c")
            logger.info(f"    - Contracts: {rlm_contracts}")
            logger.info(f"    - Max concurrent: {rlm_max_concurrent} positions")
            logger.info(f"    - Re-entry: {'ENABLED' if rlm_allow_reentry else 'DISABLED'}")
            logger.info(f"    - Orderbook timeout: {rlm_orderbook_timeout}s")
            logger.info(f"    - Spread thresholds: tight={rlm_tight_spread}c, wide={rlm_wide_spread}c")

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

    @property
    def trading_strategy(self):
        """
        Get the trading strategy as an enum.

        Converts the string trading_strategy_str to TradingStrategy enum.
        Import is done here to avoid circular imports.

        Returns:
            TradingStrategy enum value
        """
        # Import here to avoid circular import
        from ..services.trading_decision_service import TradingStrategy

        strategy_map = {
            "hold": TradingStrategy.HOLD,
            "whale_follower": TradingStrategy.WHALE_FOLLOWER,
            "paper_test": TradingStrategy.PAPER_TEST,
            "rl_model": TradingStrategy.RL_MODEL,
            "yes_80_90": TradingStrategy.YES_80_90,
            "rlm_no": TradingStrategy.RLM_NO,
            "custom": TradingStrategy.CUSTOM,
        }

        return strategy_map.get(self.trading_strategy_str, TradingStrategy.HOLD)

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate market configuration (skip for lifecycle mode which uses empty tickers)
        if not self.market_tickers and self.market_mode != "lifecycle":
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