"""
ActorService for Kalshi Trading Actor MVP.

Provides the core actor service with async queue architecture, model caching,
and serial processing for multi-market trading. Integrates with OrderbookClient
via non-blocking triggers and maintains global portfolio state.
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List, Union, TYPE_CHECKING
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import numpy as np

# Import Stable Baselines3 for model loading
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None

from ..config import config
from .event_bus import get_event_bus, EventType, MarketEvent

if TYPE_CHECKING:
    from .kalshi_multi_market_order_manager import KalshiMultiMarketOrderManager

logger = logging.getLogger("kalshiflow_rl.actor_service")

@dataclass
class ActorEvent:
    """Represents a market event for actor processing."""
    market_ticker: str
    update_type: str  # "snapshot" | "delta"
    sequence_number: int
    timestamp_ms: int
    received_at: float

@dataclass
class ActorMetrics:
    """Performance and operational metrics for the actor service."""
    events_queued: int = 0
    events_processed: int = 0
    model_predictions: int = 0
    orders_executed: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    queue_depth: int = 0
    max_queue_depth: int = 0
    errors: int = 0
    last_error: Optional[str] = None
    started_at: Optional[float] = None
    last_processed_at: Optional[float] = None


class ActorService:
    """
    Core Actor Service for multi-market trading.
    
    Features:
    - Async queue-based architecture for serial processing
    - Model caching with once-at-startup loading
    - Non-blocking integration with OrderbookClient
    - Performance monitoring and circuit breakers
    - Multi-market portfolio management
    - Dependency injection support for clean testing and architecture
    
    Architecture:
    1. Events queued from OrderbookClient triggers (non-blocking)
    2. Serial processing via single queue for all markets
    3. 4-step pipeline: build_observation → select_action → execute_action → update_positions
    4. Portfolio state management with injected dependencies
    """
    
    def __init__(
        self,
        market_tickers: Optional[List[str]] = None,
        model_path: Optional[str] = None,
        queue_size: int = 1000,
        throttle_ms: int = 250,
        event_bus: Optional[Any] = None,
        observation_adapter: Optional[Any] = None,
        strict_validation: bool = True,
        position_read_delay_ms: int = 100
    ):
        """
        Initialize Actor Service.
        
        Args:
            market_tickers: List of markets to trade (defaults to config)
            model_path: Path to trained RL model (required for RL strategy)
            queue_size: Maximum events in processing queue
            throttle_ms: Minimum milliseconds between actions per market
            event_bus: Injected event bus (optional, falls back to global)
            observation_adapter: Injected observation adapter (optional)
            strict_validation: If True, fail fast if critical dependencies missing (default: True)
            position_read_delay_ms: Delay in ms before reading positions after order execution (default: 100)
        """
        self.market_tickers = market_tickers or config.RL_MARKET_TICKERS
        self.model_path = model_path
        self.throttle_ms = throttle_ms
        self.strict_validation = strict_validation
        self.position_read_delay_ms = position_read_delay_ms
        
        # Event processing queue (serial processing)
        self._event_queue: asyncio.Queue[ActorEvent] = asyncio.Queue(maxsize=queue_size)
        self._processing = False
        self._shutdown_requested = False
        
        # Model caching (loaded once at startup)
        self._cached_model: Optional[PPO] = None
        self._model_loaded = False
        self._model_load_error: Optional[str] = None
        
        # Performance monitoring
        self.metrics = ActorMetrics()
        self._per_market_throttle: Dict[str, float] = {}
        
        # Circuit breakers and performance monitoring
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._slow_processing_count = 0
        self._max_slow_processing = 5
        self._max_errors_per_market = 10
        
        # Disabled markets tracking (for circuit breaker with re-enable)
        self._disabled_markets: Dict[str, float] = {}  # {market_ticker: disabled_at_timestamp}
        self._market_re_enable_delay_seconds = 300  # 5 minutes default
        
        # Dependency injection for services
        self._injected_event_bus = event_bus
        self._injected_observation_adapter = observation_adapter
        
        # Callbacks for integration
        self._observation_adapter: Optional[Callable] = None
        self._action_selector: Optional[Callable] = None
        self._order_manager: Optional[Union[Callable, "KalshiMultiMarketOrderManager"]] = None
        
        logger.info(
            f"ActorService initialized for {len(self.market_tickers)} markets: {', '.join(self.market_tickers)}"
        )
        logger.info(f"Queue size: {queue_size}, Throttle: {throttle_ms}ms")
        logger.info(f"Dependencies injected - event_bus: {event_bus is not None}, observation_adapter: {observation_adapter is not None}")
        logger.info(f"Strict validation: {strict_validation}, Position read delay: {position_read_delay_ms}ms")
        
        # Initialize model cache if model path provided
        if self.model_path:
            logger.info(f"Model caching enabled: {self.model_path}")
    
    async def initialize(self) -> None:
        """Initialize the actor service and load cached model."""
        logger.info("Initializing ActorService...")
        
        self.metrics.started_at = time.time()
        
        # Validate dependencies if strict validation enabled
        if self.strict_validation:
            self._validate_dependencies()
        
        # Check if model should be loaded
        # Note: With M3 ActionSelector, model loading is handled by the selector itself.
        # ActorService model loading is only for backward compatibility or if selector
        # doesn't handle its own loading.
        if self.model_path and self._needs_model():
            # Check if selector already loaded the model (RLModelSelector does this)
            from .action_selector import RLModelSelector
            if isinstance(self._action_selector, RLModelSelector):
                logger.debug(
                    f"Model loading handled by RLModelSelector, skipping ActorService model load"
                )
            else:
                # For backward compatibility or custom selectors that don't load their own model
                await self._load_and_cache_model()
        elif self.model_path and not self._needs_model():
            logger.debug(
                f"Model path provided ({self.model_path}) but action selector doesn't need it. "
                f"Model loading handled by selector (M3)."
            )
        
        # Subscribe to event bus for orderbook updates
        await self._subscribe_to_event_bus()
        
        # Start event processing loop
        asyncio.create_task(self._event_processing_loop())
        self._processing = True
        
        logger.info("✅ ActorService initialized successfully")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the actor service."""
        logger.info("Shutting down ActorService...")
        
        self._shutdown_requested = True
        
        # Wait for queue to empty with timeout
        try:
            start_time = time.time()
            while not self._event_queue.empty() and (time.time() - start_time) < 5.0:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.warning("Shutdown interrupted")
        
        self._processing = False
        logger.info("✅ ActorService shutdown complete")
    
    async def _subscribe_to_event_bus(self) -> None:
        """Subscribe to event bus for orderbook updates."""
        try:
            # Use injected event bus if available, otherwise fall back to global
            if self._injected_event_bus:
                event_bus = self._injected_event_bus
                logger.info("Using injected EventBus for subscription")
            else:
                # Fallback for backward compatibility
                event_bus = await get_event_bus()
                logger.warning("Falling back to global EventBus - consider using dependency injection")
            
            # Subscribe to both snapshot and delta events
            event_bus.subscribe(EventType.ORDERBOOK_SNAPSHOT, self._handle_event_bus_event)
            event_bus.subscribe(EventType.ORDERBOOK_DELTA, self._handle_event_bus_event)
            
            logger.info("ActorService subscribed to event bus for orderbook updates")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to event bus: {e}")
            raise
    
    def _validate_dependencies(self) -> None:
        """
        Validate that required dependencies are configured.
        
        Raises:
            ValueError: If critical dependencies are missing and strict validation is enabled
        """
        missing_deps = []
        
        # Check observation adapter (injected or callback)
        if not self._injected_observation_adapter and not self._observation_adapter:
            missing_deps.append("observation_adapter (injected or callback)")
        
        # Check action selector
        if not self._action_selector:
            missing_deps.append("action_selector")
        
        # Check order manager
        if not self._order_manager:
            missing_deps.append("order_manager")
        
        if missing_deps:
            error_msg = (
                f"ActorService missing required dependencies: {', '.join(missing_deps)}. "
                f"Use set_observation_adapter(), set_action_selector(), and set_order_manager() "
                f"to configure dependencies, or disable strict_validation for lenient mode."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug("All required dependencies validated successfully")
    
    def _is_stub_selector(self) -> bool:
        """
        Check if the current action selector is hardcoded (doesn't need model).
        
        Returns:
            True if action selector doesn't need a model, False otherwise
        """
        if not self._action_selector:
            return False
        
        # Check if it's a HardcodedSelector instance
        from .action_selector import HardcodedSelector
        if isinstance(self._action_selector, HardcodedSelector):
            return True
        
        return False
    
    def _needs_model(self) -> bool:
        """
        Check if the current action selector requires a model.
        
        Returns:
            True if model is needed (RLModelSelector), False otherwise
        """
        if not self._action_selector:
            return False
        
        # Check if it's an RLModelSelector
        from .action_selector import RLModelSelector
        if isinstance(self._action_selector, RLModelSelector):
            return True
        
        # If it's not a stub/hardcoded selector, assume it might need a model
        # (for backward compatibility with custom selectors)
        return not self._is_stub_selector()
    
    async def _handle_event_bus_event(self, event: MarketEvent) -> None:
        """
        Handle event from the event bus (replaces direct actor_event_trigger).
        
        Args:
            event: MarketEvent from the event bus
        """
        try:
            # Convert event bus event to actor event trigger
            update_type = "snapshot" if event.event_type == EventType.ORDERBOOK_SNAPSHOT else "delta"
            
            await self.trigger_event(
                market_ticker=event.market_ticker,
                update_type=update_type,
                sequence_number=event.sequence_number,
                timestamp_ms=event.timestamp_ms
            )
            
        except Exception as e:
            logger.error(f"Error handling event bus event for {event.market_ticker}: {e}")
    
    async def _load_and_cache_model(self) -> None:
        """Load RL model once at startup and cache in memory."""
        if not SB3_AVAILABLE:
            self._model_load_error = "Stable Baselines3 not available for model loading"
            logger.error(self._model_load_error)
            return
        
        if not self.model_path:
            self._model_load_error = "No model path provided"
            logger.warning(self._model_load_error)
            return
        
        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                self._model_load_error = f"Model file not found: {self.model_path}"
                logger.error(self._model_load_error)
                return
            
            logger.info(f"Loading RL model from: {self.model_path}")
            start_time = time.time()
            
            # Load model (this is the ONLY time we load the model)
            self._cached_model = PPO.load(str(model_file))
            
            load_time = time.time() - start_time
            self._model_loaded = True
            
            logger.info(f"✅ Model loaded and cached successfully in {load_time:.3f}s")
            
        except Exception as e:
            self._model_load_error = f"Failed to load model: {str(e)}"
            logger.error(f"❌ Model loading failed: {e}")
            self._model_loaded = False
    
    def get_cached_model(self) -> Optional[PPO]:
        """
        Get the cached RL model.
        
        Returns:
            Cached PPO model or None if not loaded
        """
        return self._cached_model if self._model_loaded else None
    
    def is_model_available(self) -> bool:
        """Check if cached model is available for predictions."""
        return self._model_loaded and self._cached_model is not None
    
    def get_model_load_error(self) -> Optional[str]:
        """Get model loading error if any."""
        return self._model_load_error
    
    def _should_process_market(self, market_ticker: str) -> bool:
        """
        Check if a market should be processed (not disabled by circuit breaker).
        
        Args:
            market_ticker: Market to check
            
        Returns:
            True if market should be processed, False if disabled
        """
        # Check if market is in active list
        if market_ticker not in self.market_tickers:
            return False
        
        # Check if market is disabled by circuit breaker
        if market_ticker in self._disabled_markets:
            disabled_at = self._disabled_markets[market_ticker]
            elapsed = time.time() - disabled_at
            
            # Auto re-enable after delay
            if elapsed >= self._market_re_enable_delay_seconds:
                logger.info(f"Auto re-enabling market {market_ticker} after {elapsed:.1f}s")
                del self._disabled_markets[market_ticker]
                return True
            else:
                logger.debug(f"Market {market_ticker} disabled (elapsed: {elapsed:.1f}s)")
                return False
        
        return True
    
    async def trigger_event(
        self,
        market_ticker: str,
        update_type: str,
        sequence_number: int,
        timestamp_ms: int
    ) -> bool:
        """
        Trigger actor event from OrderbookClient (non-blocking).
        
        Called after successful write_queue.enqueue_*() operations.
        
        Args:
            market_ticker: Market that was updated
            update_type: "snapshot" or "delta"
            sequence_number: Sequence number of update
            timestamp_ms: Timestamp of update
            
        Returns:
            True if event was queued, False if queue full
        """
        if self._shutdown_requested or not self._processing:
            return False
        
        # Check if we should process this market (includes circuit breaker check)
        if not self._should_process_market(market_ticker):
            return False
        
        # Note: Throttling happens at execution time, not queue time
        # This allows events to queue up but ensures actions are throttled based on actual execution
        
        current_time = time.time()
        event = ActorEvent(
            market_ticker=market_ticker,
            update_type=update_type,
            sequence_number=sequence_number,
            timestamp_ms=timestamp_ms,
            received_at=current_time
        )
        
        try:
            # Non-blocking queue put with immediate return
            self._event_queue.put_nowait(event)
            
            # Note: Throttle timestamp updated after execution, not on queue
            # This ensures throttling is based on actual action execution time
            
            self.metrics.events_queued += 1
            self.metrics.queue_depth = self._event_queue.qsize()
            self.metrics.max_queue_depth = max(self.metrics.max_queue_depth, self.metrics.queue_depth)
            
            return True
            
        except asyncio.QueueFull:
            logger.warning(f"Actor event queue full, dropping event for {market_ticker}")
            return False
        except Exception as e:
            logger.error(f"Error queuing actor event: {e}")
            return False
    
    async def _event_processing_loop(self) -> None:
        """Main event processing loop (serial processing for all markets)."""
        logger.info("Starting actor event processing loop")
        
        while not self._shutdown_requested:
            try:
                # Wait for events with timeout
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                self.metrics.queue_depth = self._event_queue.qsize()
                
                # Process single event through 4-step pipeline
                await self._process_market_update(event)
                
                # Mark task as done
                self._event_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("Event processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}\n{traceback.format_exc()}")
                self.metrics.errors += 1
                self.metrics.last_error = str(e)
                await asyncio.sleep(0.1)  # Brief pause on error
        
        logger.info("Actor event processing loop stopped")
    
    async def _process_market_update(self, event: ActorEvent) -> None:
        """
        Process a single market update through 4-step pipeline.
        
        Pipeline:
        1. build_observation - Convert orderbook state to model input
        2. select_action - Use cached model to select action  
        3. safe_execute_action - Execute action through order manager
        4. update_positions - Update portfolio tracking
        
        Args:
            event: ActorEvent to process
        """
        start_time = time.time()
        market_ticker = event.market_ticker
        
        # Check if market should be processed (circuit breaker check)
        if not self._should_process_market(market_ticker):
            logger.debug(f"Skipping processing for disabled market: {market_ticker}")
            return
        
        try:
            # Step 1: Build observation (requires LiveObservationAdapter)
            observation = await self._build_observation(market_ticker)
            if observation is None:
                logger.debug(f"No observation available for {market_ticker}")
                return
            
            # Step 2: Select action (requires ActionSelector)
            action = await self._select_action(observation, market_ticker)
            if action is None:
                logger.debug(f"No action selected for {market_ticker}")
                return
            
            # Step 3: Safe execute action (requires OrderManager)
            execution_result = await self._safe_execute_action(action, market_ticker)
            
            # Step 4: Update positions (track portfolio state)
            await self._update_positions(market_ticker, action, execution_result)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.events_processed += 1
            self.metrics.total_processing_time += processing_time
            self.metrics.avg_processing_time = self.metrics.total_processing_time / max(self.metrics.events_processed, 1)
            self.metrics.last_processed_at = time.time()
            
            # Performance monitoring
            if processing_time > 0.050:  # 50ms threshold
                self._slow_processing_count += 1
                logger.warning(
                    f"Slow processing detected: {processing_time*1000:.1f}ms for {market_ticker} "
                    f"(count: {self._slow_processing_count})"
                )
                
                if self._slow_processing_count > self._max_slow_processing:
                    logger.error("Too many slow processing events - performance circuit breaker triggered")
            
            # Reset error count for this market on success
            self._error_counts[market_ticker] = 0
            
        except Exception as e:
            self._error_counts[market_ticker] += 1
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            
            logger.error(
                f"Error processing {market_ticker} (error #{self._error_counts[market_ticker]}): {e}"
            )
            
            # Circuit breaker - disable market if too many errors
            if self._error_counts[market_ticker] > self._max_errors_per_market:
                logger.error(
                    f"Market {market_ticker} disabled due to excessive errors "
                    f"({self._error_counts[market_ticker]} errors)"
                )
                # Add to disabled markets set (can be re-enabled later)
                self._disabled_markets[market_ticker] = time.time()
                logger.info(f"Market {market_ticker} will be auto re-enabled after {self._market_re_enable_delay_seconds}s")
    
    def _check_circuit_breaker(self, market_ticker: str) -> None:
        """Check if market should be disabled due to excessive errors."""
        if self._error_counts[market_ticker] > self._max_errors_per_market:
            logger.error(
                f"Market {market_ticker} disabled due to excessive errors "
                f"({self._error_counts[market_ticker]} errors)"
            )
            # Add to disabled markets set (can be re-enabled later)
            self._disabled_markets[market_ticker] = time.time()
            logger.info(f"Market {market_ticker} will be auto re-enabled after {self._market_re_enable_delay_seconds}s")
    
    def re_enable_market(self, market_ticker: str) -> bool:
        """
        Manually re-enable a market that was disabled by circuit breaker.
        
        Args:
            market_ticker: Market to re-enable
            
        Returns:
            True if market was re-enabled, False if it wasn't disabled
        """
        if market_ticker in self._disabled_markets:
            del self._disabled_markets[market_ticker]
            logger.info(f"Market {market_ticker} manually re-enabled")
            return True
        return False
    
    def _get_default_portfolio_values(self) -> tuple:
        """
        Get default portfolio values from OrderManager or config.
        
        Returns:
            Tuple of (portfolio_value, cash_balance)
        """
        if self._order_manager:
            # Try to get initial cash from OrderManager
            if hasattr(self._order_manager, 'initial_cash'):
                initial_cash = self._order_manager.initial_cash
                return initial_cash, initial_cash
            elif hasattr(self._order_manager, 'get_cash_balance'):
                # Use current cash balance as fallback
                cash = self._order_manager.get_cash_balance()
                return cash, cash
        
        # Fall back to config default
        default_cash = getattr(config, 'RL_INITIAL_CASH', 10000.0)
        return default_cash, default_cash
    
    async def _build_observation(self, market_ticker: str) -> Optional[np.ndarray]:
        """
        Build observation for model input (Step 1).
        
        Uses injected or configured LiveObservationAdapter with portfolio data.
        Prefers injected adapter over callback adapter for consistency.
        """
        # Gather portfolio data from order manager
        position_data = {}
        portfolio_value, cash_balance = self._get_default_portfolio_values()
        order_features = None
        
        if self._order_manager:
            if hasattr(self._order_manager, 'get_positions'):
                all_positions = self._order_manager.get_positions()
                position_data = all_positions.get(market_ticker, {})
            if hasattr(self._order_manager, 'get_portfolio_value'):
                portfolio_value = self._order_manager.get_portfolio_value()
            if hasattr(self._order_manager, 'get_cash_balance'):
                cash_balance = self._order_manager.get_cash_balance()
            if hasattr(self._order_manager, 'get_order_features'):
                order_features_dict = self._order_manager.get_order_features(market_ticker)
                # Convert to numpy array: [has_open_buy, has_open_sell, time_since_order, ...]
                order_features = np.array([
                    order_features_dict.get('has_open_buy', 0.0),
                    order_features_dict.get('has_open_sell', 0.0),
                    order_features_dict.get('time_since_order', 0.0),
                    0.0,  # Placeholder for additional features
                    0.0   # Placeholder for additional features
                ], dtype=np.float32)
        
        # Prefer injected adapter (standardized path with portfolio data)
        if self._injected_observation_adapter:
            try:
                observation = await self._injected_observation_adapter.build_observation(
                    market_ticker=market_ticker,
                    position_data=position_data,
                    portfolio_value=portfolio_value,
                    cash_balance=cash_balance,
                    order_features=order_features
                )
                return observation
            except Exception as e:
                logger.error(f"Error with injected observation adapter for {market_ticker}: {e}")
                self._error_counts[market_ticker] += 1
                self.metrics.errors += 1
                self.metrics.last_error = str(e)
                self._check_circuit_breaker(market_ticker)
                # Fall through to callback adapter if available
        
        # Fallback to callback adapter (deprecated, for backward compatibility)
        if self._observation_adapter:
            logger.warning(
                f"Using deprecated callback observation adapter for {market_ticker}. "
                f"Prefer injected adapter for portfolio data support."
            )
            try:
                # Use configured observation adapter callback
                # Note: Callback may not support portfolio data - that's OK for backward compatibility
                observation = await self._observation_adapter(market_ticker)
                return observation
            except Exception as e:
                logger.error(f"Error building observation for {market_ticker}: {e}")
                self._error_counts[market_ticker] += 1
                self.metrics.errors += 1
                self.metrics.last_error = str(e)
                self._check_circuit_breaker(market_ticker)
                return None
        
        logger.debug("No observation adapter configured")
        return None
    
    async def _select_action(self, observation: np.ndarray, market_ticker: str) -> Optional[int]:
        """
        Select action using cached model (Step 2).
        
        Requires ActionSelector integration.
        """
        if not self._action_selector:
            logger.debug("No action selector configured")
            return None
        
        try:
            # Use configured action selector
            action = await self._action_selector(observation, market_ticker)
            
            if action is not None:
                self.metrics.model_predictions += 1
            
            return action
        except Exception as e:
            logger.error(f"Error selecting action for {market_ticker}: {e}")
            # Update error metrics
            self._error_counts[market_ticker] += 1
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            # Check circuit breaker
            self._check_circuit_breaker(market_ticker)
            return None
    
    async def _safe_execute_action(self, action: int, market_ticker: str) -> Optional[Dict[str, Any]]:
        """
        Safely execute trading action (Step 3).
        
        Requires OrderManager integration.
        """
        if not self._order_manager:
            logger.debug("No order manager configured")
            return None
        
        # Early return for HOLD actions (no execution needed)
        if action == 0:
            return {"status": "hold", "action": action, "market": market_ticker, "executed": False}
        
        # Check throttling BEFORE execution (for non-HOLD actions)
        if market_ticker in self._per_market_throttle:
            time_since_last = (time.time() - self._per_market_throttle[market_ticker]) * 1000
            if time_since_last < self.throttle_ms:
                logger.debug(
                    f"Market {market_ticker} throttled "
                    f"({time_since_last:.1f}ms < {self.throttle_ms}ms)"
                )
                return {"status": "throttled", "executed": False}
        
        try:
            # Get orderbook snapshot for price calculation
            orderbook_snapshot = None
            if self._injected_observation_adapter and hasattr(self._injected_observation_adapter, '_orderbook_state_registry'):
                try:
                    registry = self._injected_observation_adapter._orderbook_state_registry
                    if registry:
                        shared_state = await registry.get_shared_orderbook_state(market_ticker)
                        orderbook_snapshot = await shared_state.get_snapshot()
                except Exception as e:
                    logger.debug(f"Could not get orderbook snapshot for {market_ticker}: {e}")
                    # Fallback to global registry
                    try:
                        from ..data.orderbook_state import get_shared_orderbook_state
                        shared_state = await get_shared_orderbook_state(market_ticker)
                        orderbook_snapshot = await shared_state.get_snapshot()
                    except Exception as e2:
                        logger.warning(f"Could not get orderbook snapshot from global registry: {e2}")
            
            # Validate orderbook snapshot is available (HOLD already handled above)
            # Fail fast if missing - don't proceed with default price
            if orderbook_snapshot is None:
                logger.error(
                    f"Cannot execute action {action} for {market_ticker}: "
                    f"orderbook snapshot unavailable. Skipping execution."
                )
                self._error_counts[market_ticker] += 1
                self.metrics.errors += 1
                self.metrics.last_error = f"Missing orderbook snapshot for {market_ticker}"
                return {
                    "status": "error",
                    "error": "orderbook_snapshot_unavailable",
                    "executed": False
                }
            
            # Check if order manager is a KalshiMultiMarketOrderManager instance or a callable
            if hasattr(self._order_manager, 'execute_limit_order_action'):
                # Use KalshiMultiMarketOrderManager instance
                result = await self._order_manager.execute_limit_order_action(
                    action, market_ticker, orderbook_snapshot
                )
            else:
                # Use configured order manager callable
                result = await self._order_manager(action, market_ticker)
            
            if result and result.get("executed"):
                self.metrics.orders_executed += 1
                # Update throttle timestamp after successful execution
                # This ensures throttling is based on actual action execution time, not queue time
                self._per_market_throttle[market_ticker] = time.time()
            
            return result
        except Exception as e:
            logger.error(f"Error executing action for {market_ticker}: {e}")
            # Update error metrics
            self._error_counts[market_ticker] += 1
            self.metrics.errors += 1
            self.metrics.last_error = str(e)
            # Check circuit breaker
            self._check_circuit_breaker(market_ticker)
            return None
    
    async def _update_positions(self, market_ticker: str, action: int, execution_result: Optional[Dict[str, Any]]) -> None:
        """
        Update portfolio tracking (Step 4).
        
        Retrieves updated position data from OrderManager and logs position changes.
        
        Note: Position updates use eventual consistency model. Fills are processed
        asynchronously in KalshiMultiMarketOrderManager._process_fills(), so position
        reads immediately after order execution may be stale. A small delay is added
        to allow fill processing to complete before reading positions.
        """
        try:
            # Skip position updates for HOLD actions (no position changes)
            if not execution_result or execution_result.get("status") == "hold":
                return
            
            # Only update if we have an order manager
            if not self._order_manager:
                return
            
            # Wait for fill processing delay to account for async fill processing
            # This ensures positions reflect recent fills before reading
            if self.position_read_delay_ms > 0:
                await asyncio.sleep(self.position_read_delay_ms / 1000.0)
            
            # Retry logic for position reads (with timeout)
            max_retries = 3
            retry_delay = 0.05  # 50ms between retries
            positions = None
            portfolio_value = 0.0
            cash_balance = 0.0
            
            for attempt in range(max_retries):
                try:
                    if hasattr(self._order_manager, 'get_positions'):
                        positions = self._order_manager.get_positions()
                    if hasattr(self._order_manager, 'get_portfolio_value'):
                        portfolio_value = self._order_manager.get_portfolio_value()
                    if hasattr(self._order_manager, 'get_cash_balance'):
                        cash_balance = self._order_manager.get_cash_balance()
                    break  # Success
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.debug(f"Position read attempt {attempt + 1} failed, retrying: {e}")
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.warning(f"Failed to read positions after {max_retries} attempts: {e}")
                        return
            
            if positions is not None:
                market_position = positions.get(market_ticker, {})
                
                # Log position update
                logger.debug(
                    f"Position update for {market_ticker}: "
                    f"position={market_position.get('position', 0)}, "
                    f"portfolio_value=${portfolio_value:.2f}, "
                    f"cash=${cash_balance:.2f}"
                )
                
                # TODO: Broadcast position update via WebSocket (M5)
                # await self.broadcast_position_update(...)
                
        except Exception as e:
            logger.error(f"Error updating positions for {market_ticker}: {e}")
    
    def set_observation_adapter(self, adapter: Callable[[str], np.ndarray]) -> None:
        """Set the observation building adapter."""
        self._observation_adapter = adapter
        logger.info("Observation adapter configured")
    
    def set_action_selector(self, selector: Callable[[np.ndarray, str], int]) -> None:
        """Set the action selection strategy."""
        self._action_selector = selector
        logger.info("Action selector configured")
    
    def set_order_manager(self, manager: Union[Callable[[int, str], Dict[str, Any]], "KalshiMultiMarketOrderManager"]) -> None:
        """Set the order execution manager (accepts callable or KalshiMultiMarketOrderManager instance)."""
        self._order_manager = manager
        if hasattr(manager, 'execute_limit_order_action'):
            logger.info("KalshiMultiMarketOrderManager instance configured")
        else:
            logger.info("Order manager callable configured")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current actor service metrics."""
        return {
            "events_queued": self.metrics.events_queued,
            "events_processed": self.metrics.events_processed,
            "model_predictions": self.metrics.model_predictions,
            "orders_executed": self.metrics.orders_executed,
            "avg_processing_time_ms": self.metrics.avg_processing_time * 1000,
            "queue_depth": self.metrics.queue_depth,
            "max_queue_depth": self.metrics.max_queue_depth,
            "errors": self.metrics.errors,
            "last_error": self.metrics.last_error,
            "model_loaded": self._model_loaded,
            "model_load_error": self._model_load_error,
            "active_markets": len(self.market_tickers),
            "processing": self._processing,
            "started_at": self.metrics.started_at,
            "last_processed_at": self.metrics.last_processed_at
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current actor service status for monitoring."""
        return {
            "service": "ActorService",
            "status": "running" if self._processing and not self._shutdown_requested else "stopped",
            "markets": self.market_tickers,
            "model_available": self.is_model_available(),
            "metrics": self.get_metrics(),
            "performance": {
                "avg_processing_time_ms": self.metrics.avg_processing_time * 1000,
                "total_events": self.metrics.events_processed,
                "error_rate": self.metrics.errors / max(self.metrics.events_processed, 1),
                "throughput_events_per_sec": (
                    self.metrics.events_processed / max(time.time() - (self.metrics.started_at or time.time()), 1)
                    if self.metrics.started_at else 0
                )
            }
        }


# ===============================================================================
# Dependency Injection Support
# ===============================================================================

# Global actor service instance (maintained for backward compatibility during migration)
_actor_service: Optional[ActorService] = None
_actor_lock = asyncio.Lock()


async def get_actor_service() -> Optional[ActorService]:
    """
    Get the global actor service instance.
    
    DEPRECATED: Use dependency injection via ServiceContainer instead.
    This function is maintained for backward compatibility during migration.
    
    Returns:
        ActorService instance or None if not initialized
    """
    # Try to get from service container first
    try:
        from .service_container import get_default_container
        container = await get_default_container()
        if container.is_registered("actor_service"):
            return await container.get_service("actor_service")
    except Exception:
        pass  # Fall back to old singleton pattern
    
    async with _actor_lock:
        return _actor_service


async def initialize_actor_service(
    market_tickers: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    **kwargs
) -> ActorService:
    """
    Initialize the global actor service.
    
    DEPRECATED: Use dependency injection via ServiceContainer instead.
    This function is maintained for backward compatibility during migration.
    
    Args:
        market_tickers: Markets to trade
        model_path: Path to trained model
        **kwargs: Additional ActorService arguments
        
    Returns:
        Initialized ActorService instance
    """
    global _actor_service
    
    async with _actor_lock:
        if _actor_service is not None:
            logger.warning("ActorService already initialized")
            return _actor_service
        
        logger.info("Initializing global ActorService...")
        
        # Use lenient validation for backward compatibility (deprecated function)
        # Dependencies should be set via set_* methods after initialization
        kwargs.setdefault('strict_validation', False)
        
        _actor_service = ActorService(
            market_tickers=market_tickers,
            model_path=model_path,
            **kwargs
        )
        
        await _actor_service.initialize()
        
        logger.info("✅ Global ActorService initialized")
        return _actor_service


async def shutdown_actor_service() -> None:
    """Shutdown the global actor service."""
    global _actor_service
    
    async with _actor_lock:
        if _actor_service:
            await _actor_service.shutdown()
            _actor_service = None
            logger.info("✅ Global ActorService shutdown complete")


# ===============================================================================
# Migration Notes
# ===============================================================================
# 
# 1. Global singleton pattern replaced with dependency injection via ServiceContainer
# 2. Dependencies now injected via constructor or ServiceContainer
# 3. Backward compatibility maintained during migration period  
# 4. Event flow: OrderbookClient → EventBus → ActorService (no circular imports)
# 5. Use create_actor_service() factory for proper dependency injection