"""
TRADER V3 Environment Configuration.

Simple, centralized environment configuration for the V3 trader.
Loads from environment variables with sensible defaults.
"""

import os
from typing import List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("kalshiflow_rl.traderv3.config.environment")

# Canonical list of allowed Kalshi API categories for lifecycle discovery.
# Validated against GET /search/tags_by_categories (2025-01-28).
# All 13 Kalshi categories: Climate and Weather, Companies, Crypto, Economics,
# Elections, Entertainment, Financials, Mentions, Politics, Science and Technology,
# Social, Sports, World
DEFAULT_LIFECYCLE_CATEGORIES = [
    "climate and weather", "companies", "crypto", "economics",
    "elections", "entertainment", "financials", "mentions",
    "politics", "science and technology", "social", "sports", "world",
]


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
    enable_trading_client: bool = True
    trading_max_orders: int = 1000
    trading_max_position_size: int = 500
    trading_mode: str = "paper"  # paper or production

    # Balance Protection Configuration
    min_trader_cash: int = 1000  # Minimum balance in cents ($10.00 default). Set to 0 to disable.

    # Cleanup Configuration
    cleanup_on_startup: bool = True  # Cancel orphaned orders on startup (orders without order_group_id)

    # Position/Order Checking Configuration
    allow_multiple_positions_per_market: bool = False  # Skip position check for testing
    allow_multiple_orders_per_market: bool = False  # Skip orders check for testing

    # Order TTL Configuration
    order_ttl_enabled: bool = True  # Enable automatic cancellation of stale orders
    order_ttl_seconds: int = 300  # Cancel resting orders older than this (5min - gives passive orders time to fill)

    # Event Position Tracking Configuration
    # Tracks positions grouped by event_ticker to detect correlated exposure
    # For all-NO positions on binary mutually exclusive markets:
    #   P&L = YES_sum - 100 (positive = arbitrage, negative = loss)
    event_tracking_enabled: bool = True  # Enable event-level position tracking
    event_exposure_action: str = "alert"  # "alert" (warn but allow) or "block" (prevent trade)
    event_loss_threshold_cents: int = 100  # NO_sum > this = GUARANTEED_LOSS (block)
    event_risk_threshold_cents: int = 95   # NO_sum > this = HIGH_RISK (alert/warn)

    # Lifecycle Mode Configuration
    lifecycle_categories: List[str] = field(default_factory=lambda: list(DEFAULT_LIFECYCLE_CATEGORIES))
    # Sports prefix filter - only allow sports markets with these event_ticker prefixes
    # Set to empty list to allow ALL sports markets
    sports_allowed_prefixes: List[str] = field(default_factory=list)
    lifecycle_max_markets: int = 1000  # Maximum tracked markets (orderbook WS limit)
    lifecycle_sync_interval: int = 30  # Seconds between market info syncs

    # API Discovery Configuration (bootstrap lifecycle with already-open markets)
    api_discovery_enabled: bool = True  # Enable REST API-based market discovery
    api_discovery_interval: int = 300  # Seconds between API discovery syncs (5 min)
    api_discovery_batch_size: int = 200  # Maximum markets to fetch per API call

    # Discovery Filtering Configuration
    discovery_min_hours_to_settlement: float = 0.5  # Skip markets closing <30min
    discovery_max_days_to_settlement: int = 90  # Skip markets settling >90 days out

    # Dormant Market Detection Configuration
    # Automatically unsubscribe markets with zero trading activity to free subscription slots
    dormant_detection_enabled: bool = True  # Enable/disable dormant market cleanup
    dormant_volume_threshold: int = 0  # volume_24h <= this is "dormant" (default 0 = no activity)
    dormant_grace_period_hours: float = 1.0  # Minimum hours tracked before considering dormant

    # Arbitrage Configuration
    arb_enabled: bool = False  # Enable cross-venue arbitrage system
    polymarket_enabled: bool = False  # Enable Polymarket price oracle
    arb_auto_trade_enabled: bool = False  # Enable auto trade execution (spread monitor still runs)
    arb_spread_threshold_cents: int = 10  # Minimum spread (cents) to trigger arb trade (raised for oracle risk)
    arb_poll_interval_seconds: float = 10.0  # Polymarket price polling interval
    arb_max_pairs: int = 50  # Maximum number of tracked pairs
    arb_max_position_per_pair: int = 100  # Max contracts per pair
    arb_cooldown_seconds: float = 30.0  # Cooldown between trades on same pair
    arb_daily_loss_limit_cents: int = 50000  # $500 daily loss limit
    arb_fee_estimate_cents: int = 7  # Kalshi fee estimate per $1 profit (~7%)
    arb_min_profit_cents: int = 200  # Minimum expected profit per trade ($2) after fees
    arb_order_ttl_seconds: int = 5  # Auto-cancel unfilled arb orders after N seconds (via Kalshi expiration_ts)
    arb_min_pair_confidence: float = 0.35  # Pre-filter threshold (LLM validates above this)
    arb_scan_interval_seconds: float = 300.0  # Agent scanner runs every 5 min
    arb_auto_index: bool = True  # Auto-build persistent pair index on startup
    arb_max_events: int = 25  # Maximum events in the pair index
    arb_active_event_tickers: List[str] = field(default_factory=list)  # Trading whitelist (empty = all)
    arb_top_n_events: int = 10  # Number of top-volume Kalshi events to track
    arb_search_pool_size: int = 50  # Events to consider for pairing before narrowing to arb_top_n_events
    arb_orchestrator_enabled: bool = False  # Enable LLM orchestrator (set True when index is solid)
    event_codex_poll_interval: float = 120.0  # Event codex sync interval (seconds)
    event_codex_candle_window: int = 60  # Candle history window (minutes) - default 1h

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
        
        # Market configuration
        market_tickers_str = os.environ.get("V3_MARKET_TICKERS", "")
        if market_tickers_str.strip():
            market_tickers = [t.strip() for t in market_tickers_str.split(",") if t.strip()]
            logger.info(f"Target tickers specified: {', '.join(market_tickers)}")
        else:
            market_tickers = []  # Lifecycle discovery will populate
            logger.info("No target tickers - lifecycle discovery will populate markets")

        # Optional configuration with defaults
        max_markets = int(os.environ.get("V3_MAX_MARKETS", "10"))
        orderbook_depth = int(os.environ.get("V3_ORDERBOOK_DEPTH", "5"))
        snapshot_interval = float(os.environ.get("V3_SNAPSHOT_INTERVAL", "1.0"))
        
        # Trading client configuration
        enable_trading_client = os.environ.get("V3_ENABLE_TRADING_CLIENT", "true").lower() == "true"
        trading_max_orders = int(os.environ.get("V3_TRADING_MAX_ORDERS", "1000"))
        trading_max_position_size = int(os.environ.get("V3_TRADING_MAX_POSITION_SIZE", "500"))
        # Determine trading mode based on environment or explicit setting
        environment = os.environ.get("ENVIRONMENT", "local")
        if environment == "paper" or "demo-api" in ws_url.lower():
            trading_mode = "paper"
        else:
            trading_mode = os.environ.get("V3_TRADING_MODE", "paper")

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

        # Lifecycle discovery mode configuration
        lifecycle_categories_str = os.environ.get("LIFECYCLE_CATEGORIES", ",".join(DEFAULT_LIFECYCLE_CATEGORIES))
        lifecycle_categories = [c.strip() for c in lifecycle_categories_str.split(",") if c.strip()]
        # Sports prefix filter - only allow sports markets with these event_ticker prefixes
        # Default: KXNFL (NFL markets only). Set empty to allow all sports.
        sports_prefixes_str = os.environ.get("SPORTS_ALLOWED_PREFIXES", "")
        sports_allowed_prefixes = [p.strip() for p in sports_prefixes_str.split(",") if p.strip()]
        lifecycle_max_markets = int(os.environ.get("LIFECYCLE_MAX_MARKETS", "1000"))
        lifecycle_sync_interval = int(os.environ.get("LIFECYCLE_SYNC_INTERVAL", "30"))

        # API Discovery configuration (bootstrap with already-open markets)
        api_discovery_enabled = os.environ.get("API_DISCOVERY_ENABLED", "true").lower() == "true"
        api_discovery_interval = int(os.environ.get("API_DISCOVERY_INTERVAL", "300"))
        api_discovery_batch_size = int(os.environ.get("API_DISCOVERY_BATCH_SIZE", "200"))

        # Discovery Filtering configuration - time-to-settlement filter
        # Focus on tradeable markets (0.5h to 90d window)
        discovery_min_hours_to_settlement = float(os.environ.get("DISCOVERY_MIN_HOURS_TO_SETTLEMENT", "0.5"))
        discovery_max_days_to_settlement = int(os.environ.get("DISCOVERY_MAX_DAYS_TO_SETTLEMENT", "90"))

        # Dormant Market Detection configuration
        # Automatically unsubscribe markets with zero 24h volume to free slots
        dormant_detection_enabled = os.environ.get("DORMANT_DETECTION_ENABLED", "true").lower() == "true"
        dormant_volume_threshold = int(os.environ.get("DORMANT_VOLUME_THRESHOLD", "0"))
        dormant_grace_period_hours = float(os.environ.get("DORMANT_GRACE_PERIOD_HOURS", "1.0"))

        # Arbitrage configuration
        arb_enabled = os.environ.get("V3_ARB_ENABLED", "false").lower() == "true"
        polymarket_enabled = os.environ.get("V3_POLYMARKET_ENABLED", "false").lower() == "true"
        arb_auto_trade_enabled = os.environ.get("V3_ARB_AUTO_TRADE_ENABLED", "false").lower() == "true"
        arb_spread_threshold_cents = int(os.environ.get("V3_ARB_SPREAD_THRESHOLD_CENTS", "10"))
        arb_poll_interval_seconds = float(os.environ.get("V3_ARB_POLL_INTERVAL_SECONDS", "3.0"))
        arb_max_pairs = int(os.environ.get("V3_ARB_MAX_PAIRS", "50"))
        arb_max_position_per_pair = int(os.environ.get("V3_ARB_MAX_POSITION_PER_PAIR", "100"))
        arb_cooldown_seconds = float(os.environ.get("V3_ARB_COOLDOWN_SECONDS", "30.0"))
        arb_daily_loss_limit_cents = int(os.environ.get("V3_ARB_DAILY_LOSS_LIMIT_CENTS", "50000"))
        arb_fee_estimate_cents = int(os.environ.get("V3_ARB_FEE_ESTIMATE_CENTS", "7"))
        arb_min_profit_cents = int(os.environ.get("V3_ARB_MIN_PROFIT_CENTS", "200"))
        arb_order_ttl_seconds = int(os.environ.get("V3_ARB_ORDER_TTL_SECONDS", "5"))
        arb_min_pair_confidence = float(os.environ.get("V3_ARB_MIN_PAIR_CONFIDENCE", "0.35"))
        arb_scan_interval_seconds = float(os.environ.get("V3_ARB_SCAN_INTERVAL_SECONDS", "300.0"))
        arb_auto_index = os.environ.get("V3_ARB_AUTO_INDEX", "true").lower() == "true"
        arb_max_events = int(os.environ.get("V3_ARB_MAX_EVENTS", "25"))
        arb_active_event_tickers_str = os.environ.get("V3_ARB_ACTIVE_EVENT_TICKERS", "")
        arb_active_event_tickers = [t.strip() for t in arb_active_event_tickers_str.split(",") if t.strip()]
        arb_top_n_events = int(os.environ.get("V3_ARB_TOP_N_EVENTS", "10"))
        arb_search_pool_size = int(os.environ.get("V3_ARB_SEARCH_POOL_SIZE", "50"))
        arb_orchestrator_enabled = os.environ.get("V3_ARB_ORCHESTRATOR_ENABLED", "true").lower() == "true"
        event_codex_poll_interval = float(os.environ.get("V3_EVENT_CODEX_POLL_INTERVAL", "120.0"))
        event_codex_candle_window = int(os.environ.get("V3_EVENT_CODEX_CANDLE_WINDOW", "60"))

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
            sports_allowed_prefixes=sports_allowed_prefixes,
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
            arb_enabled=arb_enabled,
            polymarket_enabled=polymarket_enabled,
            arb_auto_trade_enabled=arb_auto_trade_enabled,
            arb_spread_threshold_cents=arb_spread_threshold_cents,
            arb_poll_interval_seconds=arb_poll_interval_seconds,
            arb_max_pairs=arb_max_pairs,
            arb_max_position_per_pair=arb_max_position_per_pair,
            arb_cooldown_seconds=arb_cooldown_seconds,
            arb_daily_loss_limit_cents=arb_daily_loss_limit_cents,
            arb_fee_estimate_cents=arb_fee_estimate_cents,
            arb_min_profit_cents=arb_min_profit_cents,
            arb_order_ttl_seconds=arb_order_ttl_seconds,
            arb_min_pair_confidence=arb_min_pair_confidence,
            arb_scan_interval_seconds=arb_scan_interval_seconds,
            arb_auto_index=arb_auto_index,
            arb_max_events=arb_max_events,
            arb_active_event_tickers=arb_active_event_tickers,
            arb_top_n_events=arb_top_n_events,
            arb_search_pool_size=arb_search_pool_size,
            arb_orchestrator_enabled=arb_orchestrator_enabled,
            event_codex_poll_interval=event_codex_poll_interval,
            event_codex_candle_window=event_codex_candle_window,
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
        if market_tickers:
            logger.info(f"  - Target tickers: {', '.join(market_tickers[:3])}{'...' if len(market_tickers) > 3 else ''} ({len(market_tickers)} total)")
        else:
            logger.info(f"  - Discovery: lifecycle (categories={', '.join(lifecycle_categories)}, max={lifecycle_max_markets})")
            if api_discovery_enabled:
                logger.info(f"    - API discovery: interval={api_discovery_interval}s, time={discovery_min_hours_to_settlement}h-{discovery_max_days_to_settlement}d")
            if dormant_detection_enabled:
                logger.info(f"    - Dormant detection: volume<={dormant_volume_threshold}, grace={dormant_grace_period_hours}h")
        logger.info(f"  - Max markets: {max_markets}")
        logger.info(f"  - Sync duration: {sync_duration}s")
        logger.info(f"  - Server: {host}:{port}")
        logger.info(f"  - Log level: {log_level}")
        if enable_trading_client:
            logger.info(f"  - Trading enabled: {trading_mode} mode")
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

        # Log arb config
        if arb_enabled:
            logger.info(f"  - Arbitrage: ENABLED (threshold={arb_spread_threshold_cents}c, fee={arb_fee_estimate_cents}c, min_profit={arb_min_profit_cents}c, order_ttl={arb_order_ttl_seconds}s)")
            logger.info(f"    - Limits: max_pairs={arb_max_pairs}, cooldown={arb_cooldown_seconds}s, daily_loss_limit=${arb_daily_loss_limit_cents/100:.0f}")
            if polymarket_enabled:
                logger.info(f"    - Polymarket poller: interval={arb_poll_interval_seconds}s, max_position={arb_max_position_per_pair}")
            else:
                logger.warning(f"  - WARNING: arb_enabled=true but polymarket_enabled=false - arb system will have no Poly prices")
            if arb_auto_index:
                logger.info(f"    - Auto index: max_events={arb_max_events}, search_pool={arb_search_pool_size}, top_n={arb_top_n_events}")
            if arb_active_event_tickers:
                logger.info(f"    - Trading whitelist: {', '.join(arb_active_event_tickers)}")
        else:
            logger.info(f"  - Arbitrage: DISABLED")

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
        # market_tickers can be empty (lifecycle discovery populates)
        
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