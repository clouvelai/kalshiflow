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
    order_ttl_seconds: int = 3600  # Dead-letter backstop (1hr) - AccountHealth handles cleanup at 30min with order-group awareness

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
    lifecycle_max_markets: int = 5000  # Maximum tracked markets (orderbook WS limit)
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

    # Lifecycle Persistence Configuration
    lifecycle_persistence_enabled: bool = True  # V3_LIFECYCLE_PERSISTENCE - persist tracked events/markets to DB

    # Early Bird Configuration
    early_bird_enabled: bool = True              # V3_EARLY_BIRD_ENABLED - detect newly activated markets
    early_bird_min_score: float = 40.0           # V3_EARLY_BIRD_MIN_SCORE - minimum score to signal Captain
    early_bird_cooldown_seconds: float = 120.0   # V3_EARLY_BIRD_COOLDOWN - cooldown between signals per event
    early_bird_use_news: bool = True             # V3_EARLY_BIRD_USE_NEWS - use Tavily for news context
    early_bird_auto_execute: bool = False        # V3_EARLY_BIRD_AUTO_EXECUTE - auto-execute complement trades

    # Single-Event Arbitrage Configuration
    single_arb_enabled: bool = False  # Enable single-event arb system
    single_arb_event_tickers: List[str] = field(default_factory=list)  # Events to monitor
    single_arb_poll_interval: float = 5.0  # REST fallback interval (seconds)
    single_arb_captain_interval: float = 60.0  # Captain cycle interval (seconds)
    single_arb_min_edge_cents: float = 0.5  # Min edge after fees to trigger detection (low = let captain decide)
    single_arb_fee_per_contract: int = 1  # Kalshi fee per contract (demo ~1c)
    single_arb_max_contracts: int = 50  # Max contracts per leg
    single_arb_captain_enabled: bool = True  # Enable LLM Captain
    single_arb_order_ttl: int = 30  # Order TTL in seconds (auto-cancel)
    single_arb_cheval_model: str = "claude-haiku-4-5-20251001"  # ChevalDeTroie model (configurable)

    # Subaccount Configuration
    subaccount: int = 0  # V3_SUBACCOUNT - 0=primary, 1-32=sub

    # Hybrid Data Mode: stream real production market data while trading on demo
    hybrid_data_mode: bool = False            # V3_HYBRID_DATA_MODE
    prod_api_url: str = ""                    # V3_PROD_API_URL
    prod_ws_url: str = ""                     # V3_PROD_WS_URL
    prod_api_key_id: str = ""                 # V3_PROD_API_KEY_ID
    prod_private_key_content: str = ""        # V3_PROD_PRIVATE_KEY_CONTENT

    # Market Maker (Admiral) Configuration
    mm_enabled: bool = False                          # V3_MM_ENABLED - master switch
    mm_event_tickers: List[str] = field(default_factory=list)  # V3_MM_EVENT_TICKERS (comma-separated)
    mm_base_spread_cents: int = 4                     # V3_MM_BASE_SPREAD
    mm_max_position: int = 100                        # V3_MM_MAX_POSITION (per market)
    mm_max_event_exposure: int = 500                  # V3_MM_MAX_EVENT_EXPOSURE
    mm_quote_size: int = 10                           # V3_MM_QUOTE_SIZE (contracts per side)
    mm_skew_factor: float = 0.5                      # V3_MM_SKEW_FACTOR
    mm_refresh_interval: float = 5.0                  # V3_MM_REFRESH_INTERVAL (seconds)
    mm_max_drawdown_cents: int = 50000                # V3_MM_MAX_DRAWDOWN ($500 default)
    mm_admiral_enabled: bool = True                   # V3_MM_ADMIRAL_ENABLED (LLM agent)
    mm_strategic_interval: float = 300.0              # V3_MM_STRATEGIC_INTERVAL
    mm_deep_scan_interval: float = 1800.0             # V3_MM_DEEP_SCAN_INTERVAL

    # Sniper Execution Layer Configuration
    sniper_enabled: bool = False                    # V3_SNIPER_ENABLED - master kill switch
    sniper_max_position: int = 25                   # V3_SNIPER_MAX_POSITION - max contracts per market
    sniper_max_capital: int = 100000                 # V3_SNIPER_MAX_CAPITAL - max capital at risk (cents = $1000)
    sniper_cooldown: float = 10.0                   # V3_SNIPER_COOLDOWN - seconds between trades on same market
    sniper_max_trades_per_cycle: int = 5            # V3_SNIPER_MAX_TRADES_PER_CYCLE - between Captain cycles
    sniper_arb_min_edge: float = 0.5               # V3_SNIPER_ARB_MIN_EDGE - min edge cents for S1_ARB
    sniper_order_ttl: int = 30                     # V3_SNIPER_ORDER_TTL - order TTL in seconds
    sniper_leg_timeout: float = 5.0                # V3_SNIPER_LEG_TIMEOUT - per-leg placement timeout in seconds
    sniper_vpin_reject_threshold: float = 0.98     # V3_SNIPER_VPIN_REJECT_THRESHOLD - VPIN above this blocks sniper (high for Kalshi thin markets)

    # Discovery Configuration
    discovery_event_count: int = 10               # V3_DISCOVERY_EVENT_COUNT - top N events by volume (lifecycle bridge adds the rest)
    discovery_seed_events: List[str] = field(default_factory=list)  # V3_DISCOVERY_SEED_EVENTS - optional hard-coded event tickers
    discovery_max_markets_per_event: int = 50     # V3_DISCOVERY_MAX_MARKETS - skip oversized events
    discovery_refresh_interval: float = 300.0      # V3_DISCOVERY_REFRESH_INTERVAL - seconds between discovery refreshes

    # Attention-Driven Captain Configuration
    strategic_interval: float = 300.0              # V3_STRATEGIC_INTERVAL - seconds between strategic reviews
    deep_scan_interval: float = 1800.0             # V3_DEEP_SCAN_INTERVAL - seconds between deep scans
    portfolio_cache_ttl: float = 15.0              # V3_PORTFOLIO_CACHE_TTL - seconds between portfolio refreshes in trigger loop

    # Swing Detection Configuration
    swing_detection_enabled: bool = True          # V3_SWING_DETECTION_ENABLED
    swing_min_change_cents: float = 5.0           # V3_SWING_MIN_CHANGE_CENTS
    swing_volume_multiplier: float = 2.0          # V3_SWING_VOLUME_MULTIPLIER
    swing_live_window_seconds: float = 900.0      # V3_SWING_LIVE_WINDOW
    swing_max_searches_per_loop: int = 3          # V3_SWING_MAX_SEARCHES
    swing_candle_refresh_seconds: float = 3600.0  # V3_SWING_CANDLE_REFRESH

    # Captain Sizing Configuration (configurable contract sizes for prompt)
    captain_eb_complement_size: str = "100-250"     # V3_CAPTAIN_EB_COMPLEMENT_SIZE - complement strategy contract range
    captain_eb_decide_size: str = "50-150"          # V3_CAPTAIN_EB_DECIDE_SIZE - captain-decide contract range
    captain_news_size_small: str = "10-25"          # V3_CAPTAIN_NEWS_SIZE_SMALL - edge 2-5c contract range
    captain_news_size_medium: str = "25-50"         # V3_CAPTAIN_NEWS_SIZE_MEDIUM - edge 5-10c contract range
    captain_news_size_large: str = "50-100"         # V3_CAPTAIN_NEWS_SIZE_LARGE - edge >=10c contract range
    captain_max_contracts_per_market: int = 200     # V3_CAPTAIN_MAX_CONTRACTS - hard cap per market
    captain_max_capital_pct_per_event: int = 20     # V3_CAPTAIN_MAX_CAPITAL_PCT - max % of capital per event

    # Account Health Configuration
    max_drawdown_pct: float = 25.0                 # V3_MAX_DRAWDOWN_PCT - pause Captain when drawdown exceeds this

    # LLM Model Configuration (centralized tiers)
    model_captain: str = "claude-sonnet-4-20250514"      # V3_MODEL_CAPTAIN
    model_subagent: str = "claude-haiku-4-5-20251001"    # V3_MODEL_SUBAGENT
    model_utility: str = "gemini-2.0-flash"              # V3_MODEL_UTILITY
    model_embedding: str = "text-embedding-3-small"      # V3_MODEL_EMBEDDING

    # Tavily Search Configuration
    tavily_api_key: str = ""                    # TAVILY_API_KEY env var
    tavily_enabled: bool = True                 # V3_TAVILY_ENABLED (auto-disable if no key)
    tavily_search_depth: str = "advanced"       # V3_TAVILY_SEARCH_DEPTH (basic=1 credit, advanced=2)
    tavily_monthly_budget: int = 10000          # V3_TAVILY_MONTHLY_BUDGET
    tavily_news_time_range: str = "week"        # V3_TAVILY_NEWS_TIME_RANGE (day/week/month)
    tavily_max_results: int = 20                # V3_TAVILY_MAX_RESULTS

    # News Ingestion Configuration (background polling)
    news_ingestion_enabled: bool = True         # V3_NEWS_INGESTION_ENABLED
    news_max_credits_per_cycle: int = 20        # V3_NEWS_MAX_CREDITS_PER_CYCLE
    news_extract_top_n: int = 2                 # V3_NEWS_EXTRACT_TOP_N

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
        order_ttl_seconds = int(os.environ.get("V3_ORDER_TTL_SECONDS", "3600"))

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
        lifecycle_max_markets = int(os.environ.get("LIFECYCLE_MAX_MARKETS", "5000"))
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

        # Lifecycle Persistence configuration
        lifecycle_persistence_enabled = os.environ.get("V3_LIFECYCLE_PERSISTENCE", "true").lower() == "true"

        # Early Bird configuration
        early_bird_enabled = os.environ.get("V3_EARLY_BIRD_ENABLED", "true").lower() == "true"
        early_bird_min_score = float(os.environ.get("V3_EARLY_BIRD_MIN_SCORE", "40.0"))
        early_bird_cooldown_seconds = float(os.environ.get("V3_EARLY_BIRD_COOLDOWN", "120.0"))
        early_bird_use_news = os.environ.get("V3_EARLY_BIRD_USE_NEWS", "true").lower() == "true"
        early_bird_auto_execute = os.environ.get("V3_EARLY_BIRD_AUTO_EXECUTE", "false").lower() == "true"

        # Single-event arb configuration
        single_arb_enabled = os.environ.get("V3_SINGLE_ARB_ENABLED", "false").lower() == "true"
        single_arb_event_tickers_str = os.environ.get("V3_SINGLE_ARB_EVENT_TICKERS", "")
        single_arb_event_tickers = [t.strip() for t in single_arb_event_tickers_str.split(",") if t.strip()]
        single_arb_poll_interval = float(os.environ.get("V3_SINGLE_ARB_POLL_INTERVAL", "5.0"))
        single_arb_captain_interval = float(os.environ.get("V3_SINGLE_ARB_CAPTAIN_INTERVAL", "60.0"))
        single_arb_min_edge_cents = float(os.environ.get("V3_SINGLE_ARB_MIN_EDGE_CENTS", "0.5"))
        single_arb_fee_per_contract = int(os.environ.get("V3_SINGLE_ARB_FEE_PER_CONTRACT", "1"))
        single_arb_max_contracts = int(os.environ.get("V3_SINGLE_ARB_MAX_CONTRACTS", "50"))
        single_arb_captain_enabled = os.environ.get("V3_SINGLE_ARB_CAPTAIN_ENABLED", "true").lower() == "true"
        single_arb_order_ttl = int(os.environ.get("V3_SINGLE_ARB_ORDER_TTL", "30"))
        # single_arb_cheval_model is set in the LLM Model Configuration block above

        # Subaccount configuration (REQUIRED - no default)
        subaccount_raw = os.environ.get("V3_SUBACCOUNT")
        if subaccount_raw is None:
            raise ValueError(
                "V3_SUBACCOUNT is required. Set V3_SUBACCOUNT=0 for primary account "
                "or V3_SUBACCOUNT=1-32 for a dedicated subaccount."
            )
        subaccount = int(subaccount_raw)

        # Hybrid Data Mode configuration
        hybrid_data_mode = os.environ.get("V3_HYBRID_DATA_MODE", "false").lower() == "true"
        prod_api_url = os.environ.get("V3_PROD_API_URL", "")
        prod_ws_url = os.environ.get("V3_PROD_WS_URL", "")
        prod_api_key_id = os.environ.get("V3_PROD_API_KEY_ID", "")
        prod_private_key_content = os.environ.get("V3_PROD_PRIVATE_KEY_CONTENT", "")

        if hybrid_data_mode:
            missing_prod = []
            if not prod_api_url:
                missing_prod.append("V3_PROD_API_URL")
            if not prod_ws_url:
                missing_prod.append("V3_PROD_WS_URL")
            if not prod_api_key_id:
                missing_prod.append("V3_PROD_API_KEY_ID")
            if not prod_private_key_content:
                missing_prod.append("V3_PROD_PRIVATE_KEY_CONTENT")
            if missing_prod:
                raise ValueError(
                    f"V3_HYBRID_DATA_MODE=true but missing: {', '.join(missing_prod)}"
                )

        # Market Maker (Admiral) configuration
        mm_enabled = os.environ.get("V3_MM_ENABLED", "false").lower() == "true"
        mm_event_tickers_str = os.environ.get("V3_MM_EVENT_TICKERS", "")
        mm_event_tickers = [t.strip() for t in mm_event_tickers_str.split(",") if t.strip()]
        mm_base_spread_cents = int(os.environ.get("V3_MM_BASE_SPREAD", "4"))
        mm_max_position = int(os.environ.get("V3_MM_MAX_POSITION", "100"))
        mm_max_event_exposure = int(os.environ.get("V3_MM_MAX_EVENT_EXPOSURE", "500"))
        mm_quote_size = int(os.environ.get("V3_MM_QUOTE_SIZE", "10"))
        mm_skew_factor = float(os.environ.get("V3_MM_SKEW_FACTOR", "0.5"))
        mm_refresh_interval = float(os.environ.get("V3_MM_REFRESH_INTERVAL", "5.0"))
        mm_max_drawdown_cents = int(os.environ.get("V3_MM_MAX_DRAWDOWN", "50000"))
        mm_admiral_enabled = os.environ.get("V3_MM_ADMIRAL_ENABLED", "true").lower() == "true"
        mm_strategic_interval = float(os.environ.get("V3_MM_STRATEGIC_INTERVAL", "300.0"))
        mm_deep_scan_interval = float(os.environ.get("V3_MM_DEEP_SCAN_INTERVAL", "1800.0"))

        # Sniper execution layer configuration
        sniper_enabled = os.environ.get("V3_SNIPER_ENABLED", "false").lower() == "true"
        sniper_max_position = int(os.environ.get("V3_SNIPER_MAX_POSITION", "25"))
        sniper_max_capital = int(os.environ.get("V3_SNIPER_MAX_CAPITAL", "100000"))
        sniper_cooldown = float(os.environ.get("V3_SNIPER_COOLDOWN", "10.0"))
        sniper_max_trades_per_cycle = int(os.environ.get("V3_SNIPER_MAX_TRADES_PER_CYCLE", "5"))
        sniper_arb_min_edge = float(os.environ.get("V3_SNIPER_ARB_MIN_EDGE", "0.5"))
        sniper_order_ttl = int(os.environ.get("V3_SNIPER_ORDER_TTL", "30"))
        sniper_leg_timeout = float(os.environ.get("V3_SNIPER_LEG_TIMEOUT", "5.0"))
        sniper_vpin_reject_threshold = float(os.environ.get("V3_SNIPER_VPIN_REJECT_THRESHOLD", "0.98"))

        # Discovery Configuration
        discovery_event_count = int(os.environ.get("V3_DISCOVERY_EVENT_COUNT", "10"))
        discovery_seed_events_str = os.environ.get("V3_DISCOVERY_SEED_EVENTS", "")
        discovery_seed_events = [s.strip() for s in discovery_seed_events_str.split(",") if s.strip()]
        discovery_max_markets_per_event = int(os.environ.get("V3_DISCOVERY_MAX_MARKETS", "50"))
        discovery_refresh_interval = float(os.environ.get("V3_DISCOVERY_REFRESH_INTERVAL", "300.0"))

        # Attention-Driven Captain Configuration
        strategic_interval = float(os.environ.get("V3_STRATEGIC_INTERVAL", "300.0"))
        deep_scan_interval = float(os.environ.get("V3_DEEP_SCAN_INTERVAL", "1800.0"))
        portfolio_cache_ttl = float(os.environ.get("V3_PORTFOLIO_CACHE_TTL", "15.0"))

        # Swing Detection Configuration
        swing_detection_enabled = os.environ.get("V3_SWING_DETECTION_ENABLED", "true").lower() == "true"
        swing_min_change_cents = float(os.environ.get("V3_SWING_MIN_CHANGE_CENTS", "5.0"))
        swing_volume_multiplier = float(os.environ.get("V3_SWING_VOLUME_MULTIPLIER", "2.0"))
        swing_live_window_seconds = float(os.environ.get("V3_SWING_LIVE_WINDOW", "900.0"))
        swing_max_searches_per_loop = int(os.environ.get("V3_SWING_MAX_SEARCHES", "3"))
        swing_candle_refresh_seconds = float(os.environ.get("V3_SWING_CANDLE_REFRESH", "3600.0"))

        # Captain Sizing Configuration
        captain_eb_complement_size = os.environ.get("V3_CAPTAIN_EB_COMPLEMENT_SIZE", "100-250")
        captain_eb_decide_size = os.environ.get("V3_CAPTAIN_EB_DECIDE_SIZE", "50-150")
        captain_news_size_small = os.environ.get("V3_CAPTAIN_NEWS_SIZE_SMALL", "10-25")
        captain_news_size_medium = os.environ.get("V3_CAPTAIN_NEWS_SIZE_MEDIUM", "25-50")
        captain_news_size_large = os.environ.get("V3_CAPTAIN_NEWS_SIZE_LARGE", "50-100")
        captain_max_contracts_per_market = int(os.environ.get("V3_CAPTAIN_MAX_CONTRACTS", "200"))
        captain_max_capital_pct_per_event = int(os.environ.get("V3_CAPTAIN_MAX_CAPITAL_PCT", "20"))

        # Account Health Configuration
        max_drawdown_pct = float(os.environ.get("V3_MAX_DRAWDOWN_PCT", "25.0"))

        # LLM Model Configuration (centralized tiers)
        model_captain = os.environ.get("V3_MODEL_CAPTAIN", "claude-sonnet-4-20250514")
        model_subagent = os.environ.get("V3_MODEL_SUBAGENT", "claude-haiku-4-5-20251001")
        model_utility = os.environ.get("V3_MODEL_UTILITY", "gemini-2.0-flash")
        model_embedding = os.environ.get("V3_MODEL_EMBEDDING", "text-embedding-3-small")

        # Backward compat: V3_SINGLE_ARB_CHEVAL_MODEL overrides model_subagent for cheval
        cheval_override = os.environ.get("V3_SINGLE_ARB_CHEVAL_MODEL")
        if cheval_override:
            logger.warning(
                f"V3_SINGLE_ARB_CHEVAL_MODEL is deprecated, use V3_MODEL_SUBAGENT instead. "
                f"Using override: {cheval_override}"
            )
            single_arb_cheval_model = cheval_override
        else:
            single_arb_cheval_model = model_subagent

        # Tavily search configuration
        tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
        tavily_enabled = os.environ.get("V3_TAVILY_ENABLED", "true").lower() == "true"
        # Auto-disable if no API key
        if not tavily_api_key:
            tavily_enabled = False
        tavily_search_depth = os.environ.get("V3_TAVILY_SEARCH_DEPTH", "advanced")
        tavily_monthly_budget = int(os.environ.get("V3_TAVILY_MONTHLY_BUDGET", "10000"))
        tavily_news_time_range = os.environ.get("V3_TAVILY_NEWS_TIME_RANGE", "week")
        tavily_max_results = int(os.environ.get("V3_TAVILY_MAX_RESULTS", "20"))

        # News Ingestion Configuration
        news_ingestion_enabled = os.environ.get("V3_NEWS_INGESTION_ENABLED", "true").lower() == "true"
        news_max_credits_per_cycle = int(os.environ.get("V3_NEWS_MAX_CREDITS_PER_CYCLE", "20"))
        news_extract_top_n = int(os.environ.get("V3_NEWS_EXTRACT_TOP_N", "2"))

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
            lifecycle_persistence_enabled=lifecycle_persistence_enabled,
            early_bird_enabled=early_bird_enabled,
            early_bird_min_score=early_bird_min_score,
            early_bird_cooldown_seconds=early_bird_cooldown_seconds,
            early_bird_use_news=early_bird_use_news,
            early_bird_auto_execute=early_bird_auto_execute,
            mm_enabled=mm_enabled,
            mm_event_tickers=mm_event_tickers,
            mm_base_spread_cents=mm_base_spread_cents,
            mm_max_position=mm_max_position,
            mm_max_event_exposure=mm_max_event_exposure,
            mm_quote_size=mm_quote_size,
            mm_skew_factor=mm_skew_factor,
            mm_refresh_interval=mm_refresh_interval,
            mm_max_drawdown_cents=mm_max_drawdown_cents,
            mm_admiral_enabled=mm_admiral_enabled,
            mm_strategic_interval=mm_strategic_interval,
            mm_deep_scan_interval=mm_deep_scan_interval,
            single_arb_enabled=single_arb_enabled,
            single_arb_event_tickers=single_arb_event_tickers,
            single_arb_poll_interval=single_arb_poll_interval,
            single_arb_captain_interval=single_arb_captain_interval,
            single_arb_min_edge_cents=single_arb_min_edge_cents,
            single_arb_fee_per_contract=single_arb_fee_per_contract,
            single_arb_max_contracts=single_arb_max_contracts,
            single_arb_captain_enabled=single_arb_captain_enabled,
            single_arb_order_ttl=single_arb_order_ttl,
            single_arb_cheval_model=single_arb_cheval_model,
            model_captain=model_captain,
            model_subagent=model_subagent,
            model_utility=model_utility,
            model_embedding=model_embedding,
            subaccount=subaccount,
            hybrid_data_mode=hybrid_data_mode,
            prod_api_url=prod_api_url,
            prod_ws_url=prod_ws_url,
            prod_api_key_id=prod_api_key_id,
            prod_private_key_content=prod_private_key_content,
            sniper_enabled=sniper_enabled,
            sniper_max_position=sniper_max_position,
            sniper_max_capital=sniper_max_capital,
            sniper_cooldown=sniper_cooldown,
            sniper_max_trades_per_cycle=sniper_max_trades_per_cycle,
            sniper_arb_min_edge=sniper_arb_min_edge,
            sniper_order_ttl=sniper_order_ttl,
            sniper_leg_timeout=sniper_leg_timeout,
            sniper_vpin_reject_threshold=sniper_vpin_reject_threshold,
            discovery_event_count=discovery_event_count,
            discovery_seed_events=discovery_seed_events,
            discovery_max_markets_per_event=discovery_max_markets_per_event,
            discovery_refresh_interval=discovery_refresh_interval,
            strategic_interval=strategic_interval,
            deep_scan_interval=deep_scan_interval,
            portfolio_cache_ttl=portfolio_cache_ttl,
            swing_detection_enabled=swing_detection_enabled,
            swing_min_change_cents=swing_min_change_cents,
            swing_volume_multiplier=swing_volume_multiplier,
            swing_live_window_seconds=swing_live_window_seconds,
            swing_max_searches_per_loop=swing_max_searches_per_loop,
            swing_candle_refresh_seconds=swing_candle_refresh_seconds,
            captain_eb_complement_size=captain_eb_complement_size,
            captain_eb_decide_size=captain_eb_decide_size,
            captain_news_size_small=captain_news_size_small,
            captain_news_size_medium=captain_news_size_medium,
            captain_news_size_large=captain_news_size_large,
            captain_max_contracts_per_market=captain_max_contracts_per_market,
            captain_max_capital_pct_per_event=captain_max_capital_pct_per_event,
            max_drawdown_pct=max_drawdown_pct,
            tavily_api_key=tavily_api_key,
            tavily_enabled=tavily_enabled,
            tavily_search_depth=tavily_search_depth,
            tavily_monthly_budget=tavily_monthly_budget,
            tavily_news_time_range=tavily_news_time_range,
            tavily_max_results=tavily_max_results,
            news_ingestion_enabled=news_ingestion_enabled,
            news_max_credits_per_cycle=news_max_credits_per_cycle,
            news_extract_top_n=news_extract_top_n,
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
        logger.info(f"  - Models: captain={model_captain}, subagent={model_subagent}, utility={model_utility}, embedding={model_embedding}")
        if subaccount > 0:
            logger.info(f"  - Subaccount: #{subaccount}")
        else:
            logger.info(f"  - Subaccount: #0 (primary)")
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

        # Log gateway config
        if hybrid_data_mode:
            logger.info(f"  - Gateway: HYBRID MODE (prod market data + demo trading)")
            logger.info(f"    - Prod API: {prod_api_url}")
            logger.info(f"    - Demo API: {api_url}")
        else:
            logger.info(f"  - Gateway: KalshiGateway + unified WS")

        # Log Tavily config
        if tavily_enabled:
            logger.info(f"  - Tavily search: ENABLED (depth={tavily_search_depth}, budget={tavily_monthly_budget}, max_results={tavily_max_results})")
        else:
            logger.info(f"  - Tavily search: DISABLED (no TAVILY_API_KEY)")

        # Log sniper config
        if sniper_enabled:
            logger.info(f"  - Sniper: ENABLED (max_pos={sniper_max_position}, max_cap=${sniper_max_capital/100:.0f}, cooldown={sniper_cooldown}s, arb_edge>{sniper_arb_min_edge}c)")
        else:
            logger.info(f"  - Sniper: DISABLED")

        # Log swing detection config
        if swing_detection_enabled:
            logger.info(f"  - Swing detection: ENABLED (min_change={swing_min_change_cents}c, vol_mult={swing_volume_multiplier}x, window={swing_live_window_seconds}s)")
        else:
            logger.info(f"  - Swing detection: DISABLED")

        # Log discovery config
        logger.info(f"  - Discovery: top {discovery_event_count} events by volume, max_markets={discovery_max_markets_per_event}, refresh={discovery_refresh_interval}s")
        if discovery_seed_events:
            logger.info(f"    - Seed events: {','.join(discovery_seed_events)}")

        # Log MM config
        if mm_enabled:
            logger.info(f"  - Market Maker: ENABLED (events={','.join(mm_event_tickers)}, spread={mm_base_spread_cents}c, size={mm_quote_size})")
            logger.info(f"    - Admiral: {'ENABLED' if mm_admiral_enabled else 'DISABLED'} (strategic={mm_strategic_interval}s, deep_scan={mm_deep_scan_interval}s)")
            logger.info(f"    - Risk: max_pos={mm_max_position}, max_exposure={mm_max_event_exposure}, drawdown=${mm_max_drawdown_cents/100:.0f}")
        else:
            logger.info(f"  - Market Maker: DISABLED")

        # Log single-arb config
        if single_arb_enabled:
            logger.info(f"  - Single-event arb: ENABLED (events={','.join(single_arb_event_tickers)}, edge>{single_arb_min_edge_cents}c, fee={single_arb_fee_per_contract}c)")
            logger.info(f"    - Captain: {'ENABLED' if single_arb_captain_enabled else 'DISABLED'} (interval={single_arb_captain_interval}s)")
            logger.info(f"    - ChevalDeTroie model: {single_arb_cheval_model}")
        else:
            logger.info(f"  - Single-event arb: DISABLED")

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