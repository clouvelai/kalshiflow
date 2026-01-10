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
    trading_max_orders: int = 1000
    trading_max_position_size: int = 100
    trading_mode: str = "paper"  # paper or production
    trading_strategy_str: str = "hold"  # Strategy string: "hold", "rlm_no"

    # RLM (Reverse Line Movement) Strategy Configuration (for RLM_NO strategy)
    # High reliability config: YES>70%, min_trades=25 gives 2.2% false positive rate
    # S-001 optimization: min_price_drop=5 skips weak signals (<5c has nearly zero edge)
    rlm_yes_threshold: float = 0.70  # Minimum YES trade ratio to trigger signal
    rlm_min_trades: int = 25  # Minimum trades before evaluating signal
    rlm_min_price_drop: int = 5  # Minimum YES price drop in cents (research: <5c has ~2% edge, skip)
    rlm_min_no_price: int = 35  # Minimum NO entry price in cents (backtest: <35c NO has -32% edge)
    rlm_contracts: int = 10  # Base contracts per trade (scaled by signal strength)
    rlm_max_concurrent: int = 1000  # Maximum concurrent positions
    rlm_allow_reentry: bool = True  # Allow adding to position on stronger signal
    rlm_orderbook_timeout: float = 2.0  # Timeout for orderbook fetch (seconds)
    rlm_tight_spread: int = 2  # Spread <= this: aggressive fill (ask - 1c)
    rlm_normal_spread: int = 4  # Spread <= this: price improvement (midpoint)
    rlm_max_spread: int = 10  # Spread > this: skip signal (protect from bad fills)

    # Balance Protection Configuration
    min_trader_cash: int = 1000  # Minimum balance in cents ($10.00 default). Set to 0 to disable.

    # Cleanup Configuration
    cleanup_on_startup: bool = True  # Cancel orphaned orders on startup (orders without order_group_id)

    # Position/Order Checking Configuration
    allow_multiple_positions_per_market: bool = False  # Skip position check for testing
    allow_multiple_orders_per_market: bool = False  # Skip orders check for testing

    # Order TTL Configuration
    order_ttl_enabled: bool = True  # Enable automatic cancellation of stale orders
    order_ttl_seconds: int = 90  # Cancel resting orders older than this (90s - orders either fill quickly or not at all)

    # Event Position Tracking Configuration
    # Tracks positions grouped by event_ticker to detect correlated exposure
    # For all-NO positions on binary mutually exclusive markets:
    #   P&L = YES_sum - 100 (positive = arbitrage, negative = loss)
    event_tracking_enabled: bool = True  # Enable event-level position tracking
    event_exposure_action: str = "alert"  # "alert" (warn but allow) or "block" (prevent trade)
    event_loss_threshold_cents: int = 100  # NO_sum > this = GUARANTEED_LOSS (block)
    event_risk_threshold_cents: int = 95   # NO_sum > this = HIGH_RISK (alert/warn)

    # Market selection mode: "config", "discovery", or "lifecycle" (default)
    # - config: Use static market_tickers list
    # - discovery: Auto-discover from REST API
    # - lifecycle: Market lifecycle events via WebSocket (NEW - default)
    market_mode: str = "lifecycle"

    # Lifecycle Mode Configuration (RL_MODE=lifecycle)
    # Categories: politics, media_mentions, entertainment, crypto (sports removed for now)
    # To re-add sports: LIFECYCLE_CATEGORIES=politics,media_mentions,entertainment,crypto,sports
    lifecycle_categories: List[str] = field(default_factory=lambda: ["politics", "media_mentions", "entertainment", "crypto"])
    lifecycle_max_markets: int = 1000  # Maximum tracked markets (orderbook WS limit)
    lifecycle_sync_interval: int = 30  # Seconds between market info syncs

    # API Discovery Configuration (bootstrap lifecycle with already-open markets)
    api_discovery_enabled: bool = True  # Enable REST API-based market discovery
    api_discovery_interval: int = 300  # Seconds between API discovery syncs (5 min)
    api_discovery_batch_size: int = 200  # Maximum markets to fetch per API call

    # Discovery Filtering Configuration
    # Time-to-settlement filter: Focus on short-dated markets for capital efficiency
    # Research shows: Sports (+17.9% edge), Media (+24.1%), Crypto (+12.8%) are all short-dated
    # Politics (+10.1% weak edge) is mostly long-dated - naturally excluded by time filter
    discovery_min_hours_to_settlement: float = 4.0  # Skip markets closing <4 hours (need time for RLM pattern)
    discovery_max_days_to_settlement: int = 30  # Skip markets settling >30 days out (capital efficiency)

    # Dormant Market Detection Configuration
    # Automatically unsubscribe markets with zero trading activity to free subscription slots
    dormant_detection_enabled: bool = True  # Enable/disable dormant market cleanup
    dormant_volume_threshold: int = 0  # volume_24h <= this is "dormant" (default 0 = no activity)
    dormant_grace_period_hours: float = 1.0  # Minimum hours tracked before considering dormant

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
        trading_max_orders = int(os.environ.get("V3_TRADING_MAX_ORDERS", "1000"))
        trading_max_position_size = int(os.environ.get("V3_TRADING_MAX_POSITION_SIZE", "100"))
        # Determine trading mode based on environment or explicit setting
        environment = os.environ.get("ENVIRONMENT", "local")
        if environment == "paper" or "demo-api" in ws_url.lower():
            trading_mode = "paper"
        else:
            trading_mode = os.environ.get("V3_TRADING_MODE", "paper")

        # Trading strategy configuration
        # Options: "hold", "rlm_no"
        trading_strategy_str = os.environ.get("V3_TRADING_STRATEGY", "hold").lower()

        # Balance protection configuration - minimum cash to continue trading
        min_trader_cash = int(os.environ.get("MIN_TRADER_CASH", "1000"))  # Default $10.00 in cents

        # Cleanup configuration - default True for paper trading, False for production
        cleanup_default = "true" if environment == "paper" or "demo-api" in ws_url.lower() else "false"
        cleanup_on_startup = os.environ.get("V3_CLEANUP_ON_STARTUP", cleanup_default).lower() == "true"

        # Position checking configuration - for testing multiple positions per market
        allow_multiple_positions_per_market = os.environ.get("V3_ALLOW_MULTIPLE_POSITIONS", "false").lower() == "true"
        allow_multiple_orders_per_market = os.environ.get("V3_ALLOW_MULTIPLE_ORDERS", "false").lower() == "true"

        # Order TTL configuration - cancel stale resting orders
        order_ttl_enabled = os.environ.get("V3_ORDER_TTL_ENABLED", "true").lower() == "true"
        order_ttl_seconds = int(os.environ.get("V3_ORDER_TTL_SECONDS", "90"))

        # Event Position Tracking configuration
        # Tracks positions by event_ticker to prevent guaranteed losses from correlated exposure
        event_tracking_enabled = os.environ.get("V3_EVENT_TRACKING_ENABLED", "true").lower() == "true"
        event_exposure_action = os.environ.get("V3_EVENT_EXPOSURE_ACTION", "alert").lower()
        event_loss_threshold_cents = int(os.environ.get("V3_EVENT_LOSS_THRESHOLD", "100"))
        event_risk_threshold_cents = int(os.environ.get("V3_EVENT_RISK_THRESHOLD", "95"))

        # RLM (Reverse Line Movement) strategy configuration
        # High reliability config: YES>70%, min_trades=25 (2.2% false positive rate)
        # See RLM_IMPROVEMENTS.md Section 10 for full reliability analysis
        rlm_yes_threshold = float(os.environ.get("RLM_YES_THRESHOLD", "0.70"))
        rlm_min_trades = int(os.environ.get("RLM_MIN_TRADES", "25"))
        rlm_min_price_drop = int(os.environ.get("RLM_MIN_PRICE_DROP", "5"))
        rlm_min_no_price = int(os.environ.get("RLM_MIN_NO_PRICE", "35"))
        rlm_contracts = int(os.environ.get("RLM_CONTRACTS", "3"))
        rlm_max_concurrent = int(os.environ.get("RLM_MAX_CONCURRENT", "1000"))
        rlm_allow_reentry = os.environ.get("RLM_ALLOW_REENTRY", "true").lower() == "true"
        rlm_orderbook_timeout = float(os.environ.get("RLM_ORDERBOOK_TIMEOUT", "2.0"))
        rlm_tight_spread = int(os.environ.get("RLM_TIGHT_SPREAD", "2"))
        rlm_normal_spread = int(os.environ.get("RLM_NORMAL_SPREAD", "4"))
        rlm_max_spread = int(os.environ.get("RLM_MAX_SPREAD", "10"))

        # Lifecycle discovery mode configuration
        # Default: politics, media_mentions, entertainment, crypto (sports removed for now)
        # To re-add sports: export LIFECYCLE_CATEGORIES=politics,media_mentions,entertainment,crypto,sports
        lifecycle_categories_str = os.environ.get("LIFECYCLE_CATEGORIES", "politics,media_mentions,entertainment,crypto")
        lifecycle_categories = [c.strip() for c in lifecycle_categories_str.split(",") if c.strip()]
        lifecycle_max_markets = int(os.environ.get("LIFECYCLE_MAX_MARKETS", "1000"))
        lifecycle_sync_interval = int(os.environ.get("LIFECYCLE_SYNC_INTERVAL", "30"))

        # API Discovery configuration (bootstrap with already-open markets)
        api_discovery_enabled = os.environ.get("API_DISCOVERY_ENABLED", "true").lower() == "true"
        api_discovery_interval = int(os.environ.get("API_DISCOVERY_INTERVAL", "300"))
        api_discovery_batch_size = int(os.environ.get("API_DISCOVERY_BATCH_SIZE", "200"))

        # Discovery Filtering configuration - time-to-settlement filter
        # Focus on short-dated markets for capital efficiency (4h to 30d window)
        discovery_min_hours_to_settlement = float(os.environ.get("DISCOVERY_MIN_HOURS_TO_SETTLEMENT", "4.0"))
        discovery_max_days_to_settlement = int(os.environ.get("DISCOVERY_MAX_DAYS_TO_SETTLEMENT", "30"))

        # Dormant Market Detection configuration
        # Automatically unsubscribe markets with zero 24h volume to free slots
        dormant_detection_enabled = os.environ.get("DORMANT_DETECTION_ENABLED", "true").lower() == "true"
        dormant_volume_threshold = int(os.environ.get("DORMANT_VOLUME_THRESHOLD", "0"))
        dormant_grace_period_hours = float(os.environ.get("DORMANT_GRACE_PERIOD_HOURS", "1.0"))

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
            rlm_yes_threshold=rlm_yes_threshold,
            rlm_min_trades=rlm_min_trades,
            rlm_min_price_drop=rlm_min_price_drop,
            rlm_min_no_price=rlm_min_no_price,
            rlm_contracts=rlm_contracts,
            rlm_max_concurrent=rlm_max_concurrent,
            rlm_allow_reentry=rlm_allow_reentry,
            rlm_orderbook_timeout=rlm_orderbook_timeout,
            rlm_tight_spread=rlm_tight_spread,
            rlm_normal_spread=rlm_normal_spread,
            rlm_max_spread=rlm_max_spread,
            min_trader_cash=min_trader_cash,
            cleanup_on_startup=cleanup_on_startup,
            allow_multiple_positions_per_market=allow_multiple_positions_per_market,
            allow_multiple_orders_per_market=allow_multiple_orders_per_market,
            order_ttl_enabled=order_ttl_enabled,
            order_ttl_seconds=order_ttl_seconds,
            event_tracking_enabled=event_tracking_enabled,
            event_exposure_action=event_exposure_action,
            event_loss_threshold_cents=event_loss_threshold_cents,
            event_risk_threshold_cents=event_risk_threshold_cents,
            lifecycle_categories=lifecycle_categories,
            lifecycle_max_markets=lifecycle_max_markets,
            lifecycle_sync_interval=lifecycle_sync_interval,
            api_discovery_enabled=api_discovery_enabled,
            api_discovery_interval=api_discovery_interval,
            api_discovery_batch_size=api_discovery_batch_size,
            discovery_min_hours_to_settlement=discovery_min_hours_to_settlement,
            discovery_max_days_to_settlement=discovery_max_days_to_settlement,
            dormant_detection_enabled=dormant_detection_enabled,
            dormant_volume_threshold=dormant_volume_threshold,
            dormant_grace_period_hours=dormant_grace_period_hours,
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
            if api_discovery_enabled:
                logger.info(f"    - API discovery: ENABLED (interval={api_discovery_interval}s, batch={api_discovery_batch_size})")
                logger.info(f"    - Time filter: {discovery_min_hours_to_settlement}h to {discovery_max_days_to_settlement}d (capital efficiency)")
            else:
                logger.info(f"    - API discovery: DISABLED")
            if dormant_detection_enabled:
                logger.info(f"    - Dormant detection: ENABLED (volume<={dormant_volume_threshold}, grace={dormant_grace_period_hours}h)")
            else:
                logger.info(f"    - Dormant detection: DISABLED")
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
            if min_trader_cash > 0:
                logger.info(f"  - Min cash protection: ${min_trader_cash/100:.2f} (trades skip below)")
            else:
                logger.info(f"  - Min cash protection: DISABLED")
        else:
            logger.info(f"  - Trading: DISABLED (orderbook only)")
        if allow_multiple_positions_per_market:
            logger.warning(f"  - ALLOW_MULTIPLE_POSITIONS: ENABLED (testing mode)")
        if allow_multiple_orders_per_market:
            logger.warning(f"  - ALLOW_MULTIPLE_ORDERS: ENABLED (testing mode)")
        if order_ttl_enabled:
            logger.info(f"  - Order TTL: {order_ttl_seconds}s (auto-cancel stale resting orders)")
        else:
            logger.info(f"  - Order TTL: DISABLED")

        # Log event tracking config
        if event_tracking_enabled:
            logger.info(f"  - Event tracking: ENABLED (action={event_exposure_action}, loss>{event_loss_threshold_cents}c, risk>{event_risk_threshold_cents}c)")
        else:
            logger.info(f"  - Event tracking: DISABLED")

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
            logger.info(f"    - Spread thresholds: tight={rlm_tight_spread}c, normal={rlm_normal_spread}c, max={rlm_max_spread}c")

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
            "rlm_no": TradingStrategy.RLM_NO,
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