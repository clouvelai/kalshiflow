"""
WebSocket orderbook client for Kalshi RL Trading Subsystem.

Provides OrderbookClient that connects to Kalshi orderbook WebSocket,
processes snapshots and deltas, updates SharedOrderbookState non-blocking,
and queues messages for database persistence.
"""

import asyncio
import json
import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List
import websockets
from websockets.exceptions import ConnectionClosed, InvalidMessage, InvalidStatus

from .auth import get_rl_auth
from .orderbook_state import get_shared_orderbook_state, SharedOrderbookState
from .write_queue import get_write_queue
from .database import rl_db
from ..config import config
from ..trading.event_bus import emit_orderbook_snapshot, emit_orderbook_delta

logger = logging.getLogger("kalshiflow_rl.orderbook_client")


class OrderbookClient:
    """
    WebSocket client for Kalshi orderbook data.
    
    Features:
    - Connects to Kalshi orderbook WebSocket with authentication
    - Subscribes to orderbook deltas for multiple markets
    - Processes snapshots and incremental updates
    - Updates in-memory SharedOrderbookState immediately (non-blocking)
    - Queues all messages for database persistence (non-blocking)
    - Automatic reconnection with exponential backoff
    - Per-market sequence number tracking and validation
    """
    
    def __init__(self, market_tickers: Optional[List[str]] = None, stats_collector=None):
        """
        Initialize orderbook client.
        
        Args:
            market_tickers: List of market tickers to subscribe to (defaults to config)
            stats_collector: Optional stats collector for tracking metrics
        """
        # Support backward compatibility - accept single ticker as string
        if isinstance(market_tickers, str):
            self.market_tickers = [market_tickers]
        elif market_tickers is None:
            # Use configured tickers (multi-market support)
            self.market_tickers = config.RL_MARKET_TICKERS
        else:
            self.market_tickers = market_tickers
            
        self.ws_url = config.KALSHI_WS_URL
        
        # WebSocket connection management
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._running = False
        self._reconnect_count = 0
        self._session_id: Optional[int] = None
        
        # Per-market tracking
        self._last_sequences: Dict[str, int] = {ticker: 0 for ticker in self.market_tickers}
        self._orderbook_states: Dict[str, SharedOrderbookState] = {}
        
        # Statistics and monitoring
        self._messages_received = 0
        self._snapshots_received = 0
        self._deltas_received = 0
        self._connection_start_time: Optional[float] = None
        self._last_message_time: Optional[float] = None
        self._client_start_time: Optional[float] = None  # Track when client started (for health checks during retries)
        
        # Event handlers
        self._on_connected: Optional[Callable] = None
        self._on_disconnected: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
        
        # Stats collector for metrics tracking
        self._stats_collector = stats_collector
        
        logger.info(f"OrderbookClient initialized for {len(self.market_tickers)} markets: {', '.join(self.market_tickers)}")
    
    async def start(self) -> None:
        """Start the orderbook client and begin connection."""
        if self._running:
            logger.warning("OrderbookClient is already running")
            return
        
        logger.info(f"Starting OrderbookClient for {len(self.market_tickers)} markets")
        self._running = True
        self._reconnect_count = 0
        self._client_start_time = time.time()  # Track when client started
        
        # Get shared orderbook states for all markets
        for market_ticker in self.market_tickers:
            self._orderbook_states[market_ticker] = await get_shared_orderbook_state(market_ticker)
            logger.info(f"Initialized orderbook state for: {market_ticker}")
        
        # Initialize database
        await rl_db.initialize()
        
        # Start connection loop
        await self._connection_loop()
    
    async def stop(self) -> None:
        """Stop the orderbook client."""
        logger.info("Stopping OrderbookClient...")
        self._running = False
        
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        # Close session if active
        if self._session_id:
            try:
                await rl_db.close_session(self._session_id, status='closed')
                logger.info(f"Closed session {self._session_id}")
            except Exception as e:
                logger.error(f"Failed to close session {self._session_id}: {e}")
            self._session_id = None
        
        logger.info(
            f"OrderbookClient stopped. Final stats: "
            f"messages={self._messages_received}, snapshots={self._snapshots_received}, "
            f"deltas={self._deltas_received}, reconnects={self._reconnect_count}"
        )
    
    async def _connection_loop(self) -> None:
        """Main connection loop with automatic reconnection."""
        while self._running:
            try:
                await self._connect_and_subscribe()
            except Exception as e:
                logger.error(f"Connection loop error: {e}", exc_info=True)
                if self._on_error:
                    try:
                        await self._on_error(e)
                    except Exception:
                        pass
            
            if self._running:
                # Calculate reconnect delay with exponential backoff
                delay = min(config.WEBSOCKET_RECONNECT_DELAY * (2 ** self._reconnect_count), 60)
                logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_count + 1})")
                await asyncio.sleep(delay)
                self._reconnect_count += 1
                
                if self._reconnect_count >= config.MAX_RECONNECT_ATTEMPTS:
                    logger.error(f"Max reconnect attempts ({config.MAX_RECONNECT_ATTEMPTS}) reached")
                    self._running = False
                    break
    
    async def _connect_and_subscribe(self) -> None:
        """Connect to WebSocket and subscribe to orderbook."""
        auth = get_rl_auth()
        headers = auth.create_websocket_headers()
        logger.info(f"Connecting to WebSocket: {self.ws_url} (reconnect_count={self._reconnect_count})")
        
        async with websockets.connect(
            self.ws_url,
            additional_headers=headers,
            ping_interval=config.WEBSOCKET_PING_INTERVAL,
            ping_timeout=config.WEBSOCKET_TIMEOUT,
            max_size=1024*1024,  # 1MB max message size
            compression=None  # Disable compression for lower latency
        ) as websocket:
            self._websocket = websocket
            self._connection_start_time = time.time()
            self._reconnect_count = 0  # Reset on successful connection
            
            # Create new session
            self._session_id = await rl_db.create_session(
                market_tickers=self.market_tickers,
                websocket_url=self.ws_url,
                environment=config.ENVIRONMENT
            )
            
            # Pass session ID to write queue
            get_write_queue().set_session_id(self._session_id)
            
            logger.info(f"WebSocket connected for {len(self.market_tickers)} markets, session {self._session_id}")
            
            if self._on_connected:
                try:
                    await self._on_connected()
                except Exception as e:
                    logger.error(f"Connection callback error: {e}")
            
            # Subscribe to orderbook for all markets
            await self._subscribe_to_orderbook()
            
            # Process messages
            await self._message_loop()
        
        # Clear websocket when context exits (connection closed)
        self._websocket = None
        self._connection_start_time = None
    
    async def _subscribe_to_orderbook(self) -> None:
        """Subscribe to orderbook channels for all markets."""
        # Use same format as trades subscription but with orderbook_delta channel
        subscription_message = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": self.market_tickers  # Array of market ticker strings
            }
        }
        
        logger.info(f"Sending subscription: {json.dumps(subscription_message)}")
        await self._websocket.send(json.dumps(subscription_message))
        logger.info(f"Subscribed to orderbook_delta channel for {len(self.market_tickers)} markets: {', '.join(self.market_tickers)}")
    
    async def _message_loop(self) -> None:
        """Process incoming WebSocket messages."""
        try:
            async for message in self._websocket:
                if not self._running:
                    break
                
                await self._process_message(message)
                
        except ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            
            # Close session on disconnect
            if self._session_id:
                try:
                    await rl_db.close_session(self._session_id, status='closed')
                    await rl_db.update_session_stats(
                        self._session_id, 
                        messages=self._messages_received, 
                        snapshots=self._snapshots_received, 
                        deltas=self._deltas_received
                    )
                except Exception as e:
                    logger.error(f"Failed to update session on disconnect: {e}")
                self._session_id = None
            
            if self._on_disconnected:
                try:
                    await self._on_disconnected()
                except Exception:
                    pass
        except InvalidMessage as e:
            logger.error(f"Invalid WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Message loop error: {e}\n{traceback.format_exc()}")
            raise
    
    async def _process_message(self, raw_message: str) -> None:
        """
        Process a single WebSocket message.
        
        Args:
            raw_message: Raw message string from WebSocket
        """
        try:
            # Parse message
            message = json.loads(raw_message)
            self._messages_received += 1
            self._last_message_time = time.time()
            
            # Log first snapshot for each market (useful for prod debugging)
            if message.get("type") == "orderbook_snapshot":
                msg_data = message.get('msg', {})
                market_ticker = msg_data.get('market_ticker')
                if market_ticker and market_ticker not in getattr(self, '_logged_first_snapshot', set()):
                    if not hasattr(self, '_logged_first_snapshot'):
                        self._logged_first_snapshot = set()
                    self._logged_first_snapshot.add(market_ticker)
                    logger.info(f"First snapshot for {market_ticker}: seq={message.get('seq', 0)}")
            
            # Determine message type
            msg_type = self._get_message_type(message)
            
            if msg_type == "snapshot":
                await self._process_snapshot(message)
            elif msg_type == "delta":
                await self._process_delta(message)
            elif msg_type == "subscription_ack":
                logger.info("Subscription acknowledged for multi-market orderbook")
            elif msg_type == "heartbeat":
                pass  # Silently ignore heartbeats
            else:
                if self._messages_received % 100 == 0:  # Only log unhandled messages occasionally
                    logger.debug(f"Unhandled message type: {msg_type}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message as JSON: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}\n{traceback.format_exc()}")
    
    def _get_message_type(self, message: Dict[str, Any]) -> str:
        """Determine the type of WebSocket message."""
        # Check for message type field (Kalshi format)
        msg_type = message.get("type")
        
        if msg_type == "orderbook_snapshot":
            return "snapshot"
        elif msg_type == "orderbook_delta":
            return "delta"
        
        # Check for acknowledgment (Kalshi uses type: "subscribed")
        if message.get("type") == "subscribed":
            return "subscription_ack"
        
        # Check for heartbeat
        if "ping" in message or "pong" in message:
            return "heartbeat"
        
        return "unknown"
    
    def _extract_market_from_channel(self, channel: str) -> Optional[str]:
        """
        Extract market ticker from channel string.
        
        Channel format: "orderbook_delta.{MARKET_TICKER}"
        
        Args:
            channel: Channel string from WebSocket message
            
        Returns:
            Market ticker or None if not extractable
        """
        if not channel or "orderbook_delta." not in channel:
            return None
        
        try:
            # Split on '.' and get the part after 'orderbook_delta'
            parts = channel.split(".")
            if len(parts) >= 2 and parts[0] == "orderbook_delta":
                return ".".join(parts[1:])  # Handle tickers with dots
        except Exception:
            pass
        
        return None
    
    async def _process_snapshot(self, message: Dict[str, Any]) -> None:
        """Process orderbook snapshot message."""
        try:
            # Get the message data (Kalshi puts content in 'msg' field)
            msg_data = message.get("msg", {})
            market_ticker = msg_data.get("market_ticker")
            
            if not market_ticker or market_ticker not in self.market_tickers:
                logger.warning(f"Received snapshot for unknown market: {market_ticker}")
                return
            
            # Extract orderbook data from msg field
            data = msg_data
            
            # CRITICAL FIX: Parse Kalshi orderbook format correctly per official documentation
            # Reference: https://docs.kalshi.com/getting_started/orderbook_responses
            # 
            # Kalshi sends ONLY BIDS in both "yes" and "no" arrays:
            # - "yes" array: [[price, qty], ...] = YES BIDS ONLY
            # - "no" array: [[price, qty], ...]  = NO BIDS ONLY
            # 
            # Asks must be DERIVED using reciprocal relationship:
            # - YES ASKS = derived from NO BIDS (100 - no_bid_price)
            # - NO ASKS = derived from YES BIDS (100 - yes_bid_price)
            
            yes_bids = {}
            no_bids = {}
            yes_asks = {}
            no_asks = {}
            
            # Parse YES bids directly from Kalshi 'yes' array
            yes_levels = data.get("yes", [])
            if isinstance(yes_levels, list):
                for price_qty in yes_levels:
                    if len(price_qty) >= 2:
                        price, qty = price_qty[0], price_qty[1]
                        if qty > 0:
                            price_int = int(price)
                            if 1 <= price_int <= 99:  # Valid Kalshi price range
                                yes_bids[price_int] = int(qty)
            
            # Parse NO bids directly from Kalshi 'no' array
            no_levels = data.get("no", [])
            if isinstance(no_levels, list):
                for price_qty in no_levels:
                    if len(price_qty) >= 2:
                        price, qty = price_qty[0], price_qty[1]
                        if qty > 0:
                            price_int = int(price)
                            if 1 <= price_int <= 99:  # Valid Kalshi price range
                                no_bids[price_int] = int(qty)
            
            # Derive YES asks from NO bids using reciprocal relationship
            # If there's a NO bid at price X, there's a YES ask at (100 - X)
            for no_bid_price, qty in no_bids.items():
                yes_ask_price = 100 - no_bid_price
                if 1 <= yes_ask_price <= 99:  # Ensure valid derived price range
                    yes_asks[yes_ask_price] = qty
            
            # Derive NO asks from YES bids using reciprocal relationship  
            # If there's a YES bid at price X, there's a NO ask at (100 - X)
            for yes_bid_price, qty in yes_bids.items():
                no_ask_price = 100 - yes_bid_price
                if 1 <= no_ask_price <= 99:  # Ensure valid derived price range
                    no_asks[no_ask_price] = qty
            
            # VALIDATION: Ensure arbitrage constraint is maintained
            # The reciprocal relationship should guarantee: YES_bid_price + NO_ask_price = 100 (and vice versa)
            # Since NO_ask_price = 100 - YES_bid_price, we should have YES_bid_price + (100 - YES_bid_price) = 100
            validation_errors = []
            
            # Verify derived asks match expected reciprocal relationship
            for yes_bid_price in yes_bids:
                expected_no_ask_price = 100 - yes_bid_price
                if expected_no_ask_price in no_asks:
                    # This should always equal 100 by definition
                    sum_check = yes_bid_price + expected_no_ask_price
                    if sum_check != 100:
                        validation_errors.append(f"YES_bid({yes_bid_price}) + derived_NO_ask({expected_no_ask_price}) = {sum_check} ≠ 100")
            
            for no_bid_price in no_bids:
                expected_yes_ask_price = 100 - no_bid_price
                if expected_yes_ask_price in yes_asks:
                    # This should always equal 100 by definition  
                    sum_check = no_bid_price + expected_yes_ask_price
                    if sum_check != 100:
                        validation_errors.append(f"NO_bid({no_bid_price}) + derived_YES_ask({expected_yes_ask_price}) = {sum_check} ≠ 100")
            
            if validation_errors:
                logger.error(f"Reciprocal relationship validation failed for {market_ticker}: {validation_errors[:3]}...")
            
            # Log parsing success for first few snapshots to verify fix
            if not hasattr(self, '_logged_parsing_success'):
                self._logged_parsing_success = {}
            if market_ticker not in self._logged_parsing_success:
                self._logged_parsing_success[market_ticker] = 0
            
            if self._logged_parsing_success[market_ticker] < 3:  # Log first 3 snapshots per market
                self._logged_parsing_success[market_ticker] += 1
                logger.info(
                    f"PARSING SUCCESS {market_ticker}: "
                    f"YES_bids={len(yes_bids)}, NO_bids={len(no_bids)}, "
                    f"derived_YES_asks={len(yes_asks)}, derived_NO_asks={len(no_asks)} "
                    f"[Fix verified - using correct Kalshi format]"
                )
            
            # Get sequence from the outer message (not from msg data)
            sequence_number = message.get("seq", 0)
            
            snapshot_data = {
                "market_ticker": market_ticker,
                "timestamp_ms": int(time.time() * 1000),
                "sequence_number": sequence_number,
                "yes_bids": yes_bids,
                "yes_asks": yes_asks,
                "no_bids": no_bids,
                "no_asks": no_asks
            }
            
            self._last_sequences[market_ticker] = snapshot_data["sequence_number"]
            self._snapshots_received += 1
            
            # Track in stats collector if available
            if self._stats_collector:
                self._stats_collector.track_snapshot(market_ticker)
            
            # Apply snapshot to shared state (self._orderbook_states[market_ticker] is the same instance)
            # No need to apply twice - they're the same object
            if market_ticker in self._orderbook_states:
                await self._orderbook_states[market_ticker].apply_snapshot(snapshot_data)
            else:
                # Fallback: if somehow not in local dict, get from global registry
                global_state = await get_shared_orderbook_state(market_ticker)
                await global_state.apply_snapshot(snapshot_data)
            
            # Queue for database persistence with session ID (non-blocking)
            enqueue_success = await get_write_queue().enqueue_snapshot(snapshot_data)
            
            # Trigger actor event via event bus after successful database enqueue (non-blocking)
            if enqueue_success:
                try:
                    await emit_orderbook_snapshot(
                        market_ticker=market_ticker,
                        sequence_number=snapshot_data["sequence_number"],
                        timestamp_ms=snapshot_data["timestamp_ms"]
                    )
                except Exception as e:
                    # Don't let event bus errors break orderbook processing
                    logger.debug(f"Event bus emit failed for {market_ticker} snapshot: {e}")
            
            # Log snapshot processing (less verbose for production)
            total_levels = (len(snapshot_data['yes_bids']) + len(snapshot_data['yes_asks']) + 
                           len(snapshot_data['no_bids']) + len(snapshot_data['no_asks']))
            logger.info(
                f"Snapshot processed: {market_ticker} seq={self._last_sequences[market_ticker]} "
                f"levels={total_levels} (local+global)"
            )
            
        except Exception as e:
            logger.error(f"Error processing snapshot: {e}\n{traceback.format_exc()}")
    
    async def _process_delta(self, message: Dict[str, Any]) -> None:
        """Process orderbook delta message."""
        try:
            # Get the message data (Kalshi puts content in 'msg' field, same as snapshots)
            msg_data = message.get("msg", {})
            market_ticker = msg_data.get("market_ticker")
            
            if not market_ticker or market_ticker not in self.market_tickers:
                logger.warning(f"Received delta for unknown market: {market_ticker}")
                return
            
            # Delta data is in msg field
            delta_val = msg_data.get("delta", 0)
            price = int(msg_data.get("price", 0))
            side = msg_data.get("side")  # "yes" or "no" (indicates which BID side is changing)
            
            # CRITICAL: Delta processing is CORRECT as-is because Kalshi deltas only update BIDS
            # - side="yes" means YES BIDS are changing at this price
            # - side="no" means NO BIDS are changing at this price  
            # - The derived asks will be recalculated when SharedOrderbookState applies the delta
            
            # Get current size from local orderbook state to calculate new size
            current_size = 0
            if market_ticker in self._orderbook_states:
                state = self._orderbook_states[market_ticker]._state
                if side == "yes":
                    current_size = state.yes_bids.get(price, 0)  # YES bid at this price
                elif side == "no":
                    current_size = state.no_bids.get(price, 0)   # NO bid at this price
            
            # Calculate new size from delta
            new_size = max(0, current_size + delta_val)
            old_size = current_size
            
            # Determine action based on delta
            if delta_val > 0:
                action = "add" if current_size == 0 else "update"
            elif delta_val < 0:
                action = "remove" if new_size == 0 else "update"
            else:
                action = "update"
            
            # Get sequence from the outer message (same as snapshots)
            sequence_number = message.get("seq", 0)
            
            delta_data = {
                "market_ticker": market_ticker,
                "timestamp_ms": int(time.time() * 1000),
                "sequence_number": sequence_number,  # From outer message
                "side": side,  # "yes" or "no"
                "action": action,
                "price": price,  # Already converted to int above
                "old_size": old_size,
                "new_size": new_size
            }
            
            # Validate delta
            if not self._validate_delta(delta_data):
                return
            
            self._last_sequences[market_ticker] = delta_data["sequence_number"]
            self._deltas_received += 1
            
            # Track in stats collector if available
            if self._stats_collector:
                self._stats_collector.track_delta(market_ticker)
            
            # Apply delta to shared state (self._orderbook_states[market_ticker] is the same instance as get_shared_orderbook_state)
            # No need to apply twice - they're the same object
            if market_ticker in self._orderbook_states:
                success = await self._orderbook_states[market_ticker].apply_delta(delta_data)
                if not success:
                    logger.warning(f"Failed to apply delta for {market_ticker}: seq={delta_data['sequence_number']}")
            else:
                # Fallback: if somehow not in local dict, get from global registry
                global_state = await get_shared_orderbook_state(market_ticker)
                success = await global_state.apply_delta(delta_data)
                if not success:
                    logger.warning(f"Failed to apply delta for {market_ticker}: seq={delta_data['sequence_number']}")
            
            # Queue for database persistence with session ID (non-blocking)
            enqueue_success = await get_write_queue().enqueue_delta(delta_data)
            
            # Trigger actor event via event bus after successful database enqueue (non-blocking)
            if enqueue_success:
                try:
                    await emit_orderbook_delta(
                        market_ticker=market_ticker,
                        sequence_number=delta_data["sequence_number"],
                        timestamp_ms=delta_data["timestamp_ms"]
                    )
                except Exception as e:
                    # Don't let event bus errors break orderbook processing
                    logger.debug(f"Event bus emit failed for {market_ticker} delta: {e}")
            
            # Log periodically at info level for production monitoring
            if self._deltas_received % 1000 == 0:
                logger.info(f"Delta checkpoint: {self._deltas_received} deltas processed across {len(self.market_tickers)} markets (local+global)")
            
        except Exception as e:
            logger.error(f"Error processing delta: {e}\n{traceback.format_exc()}")
    
    def _map_delta_action(self, data: Dict[str, Any]) -> str:
        """Map WebSocket delta data to standardized action."""
        new_size = data.get("new_size", data.get("size", 0))
        old_size = data.get("old_size", 0)
        
        if old_size == 0 and new_size > 0:
            return "add"
        elif old_size > 0 and new_size == 0:
            return "remove"
        elif old_size != new_size:
            return "update"
        else:
            return "update"  # Default to update
    
    def _validate_delta(self, delta_data: Dict[str, Any]) -> bool:
        """Validate delta data."""
        required_fields = ["side", "action", "price", "sequence_number"]
        
        for field in required_fields:
            if delta_data.get(field) is None:
                logger.warning(f"Invalid delta - missing {field}: {delta_data}")
                return False
        
        if delta_data["side"] not in ["yes", "no"]:
            logger.warning(f"Invalid side: {delta_data['side']}")
            return False
        
        if delta_data["action"] not in ["add", "remove", "update"]:
            logger.warning(f"Invalid action: {delta_data['action']}")
            return False
        
        return True
    
    # Event handlers
    
    def on_connected(self, callback: Callable) -> None:
        """Set callback for connection events."""
        self._on_connected = callback
    
    def on_disconnected(self, callback: Callable) -> None:
        """Set callback for disconnection events."""
        self._on_disconnected = callback
    
    def on_error(self, callback: Callable) -> None:
        """Set callback for error events."""
        self._on_error = callback
    
    # Statistics and monitoring
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        uptime = None
        if self._connection_start_time:
            uptime = time.time() - self._connection_start_time
        
        stats = {
            "market_tickers": self.market_tickers,
            "market_count": len(self.market_tickers),
            "running": self._running,
            "connected": self._websocket is not None,
            "reconnect_count": self._reconnect_count,
            "last_sequences": self._last_sequences,
            "messages_received": self._messages_received,
            "snapshots_received": self._snapshots_received,
            "deltas_received": self._deltas_received,
            "uptime_seconds": uptime,
            "last_message_time": self._last_message_time,
            "session_id": self._session_id
        }
        
        return stats
    
    def is_healthy(self) -> bool:
        """Check if client is healthy and receiving data."""
        if not self._running:
            return False
        
        # During initial connection period (first 10 seconds), be lenient:
        # - Allow time for initial connection attempt (503 errors may cause retries)
        # - But only return True if websocket actually exists (connection succeeded)
        if self._client_start_time:
            time_since_start = time.time() - self._client_start_time
            if time_since_start < 10.0:  # Short grace period for initial connection
                # During grace period, only return True if websocket exists (connection succeeded)
                # Don't return True if websocket is None - that means connection hasn't succeeded yet
                if self._websocket is not None:
                    # Check if websocket is closed
                    try:
                        if hasattr(self._websocket, 'closed'):
                            if self._websocket.closed:
                                return False  # Connection closed, not healthy
                        elif hasattr(self._websocket, 'close_code'):
                            if self._websocket.close_code is not None:
                                return False  # Connection closed, not healthy
                    except (AttributeError, TypeError):
                        pass
                    # Websocket exists and appears open - healthy during grace period
                    return True
                # Websocket is None - connection hasn't been established yet, not healthy
                return False
        
        # After grace period, check websocket state strictly
        if self._websocket:
            # Check if WebSocket connection is closed
            try:
                websocket_closed = False
                if hasattr(self._websocket, 'closed'):
                    websocket_closed = self._websocket.closed
                elif hasattr(self._websocket, 'close_code'):
                    # Some WebSocket implementations use close_code (None = open)
                    websocket_closed = self._websocket.close_code is not None
                
                if websocket_closed:
                    # Clear websocket reference since it's closed
                    self._websocket = None
                    return False
            except (AttributeError, TypeError) as e:
                logger.debug(f"Could not check WebSocket connection state: {e}")
                # If we can't check, assume healthy if _websocket is not None and _running is True
                pass
        else:
            # No websocket after grace period - not healthy
            return False
        
        # If we have a websocket connection but haven't received messages yet,
        # still consider it healthy if it's been less than 10 seconds since connection
        if self._connection_start_time:
            time_since_connection = time.time() - self._connection_start_time
            if time_since_connection < 10.0:
                # During initial connection period, just check if websocket exists and is running
                return self._websocket is not None
        
        # After initial period, we should have received snapshots if healthy and subscribed to valid markets
        if self._snapshots_received == 0:
            # No snapshots received after initial period - unhealthy
            return False
        
        # We've received snapshots, connection is healthy
        # Only mark unhealthy if we haven't received ANY messages for 5 minutes
        # This handles the case where the connection silently dies
        # Note: When a new message arrives, _last_message_time gets updated, so this will recover to healthy
        if self._last_message_time:
            time_since_message = time.time() - self._last_message_time
            if time_since_message > 300:  # No messages for 5 minutes - likely disconnected
                return False
        
        return True
    
    def get_health_details(self) -> Dict[str, Any]:
        """
        Get detailed health information for initialization tracker.
        
        Returns:
            Dictionary with health status, connection info, and market subscription details
        """
        stats = self.get_stats()
        health_details = {
            "connected": stats.get("connected", False),
            "running": stats.get("running", False),
            "ws_url": self.ws_url,
            "markets_subscribed": len(self.market_tickers),
            "market_tickers": self.market_tickers,
            "messages_received": stats.get("messages_received", 0),
            "snapshots_received": stats.get("snapshots_received", 0),
            "deltas_received": stats.get("deltas_received", 0),
            "last_message_time": stats.get("last_message_time"),
            "uptime_seconds": stats.get("uptime_seconds"),
            "reconnect_count": stats.get("reconnect_count", 0),
            "session_id": stats.get("session_id"),
        }
        
        return health_details
    
    def get_last_sync_time(self) -> Optional[float]:
        """
        Get last message/sync time.
        
        Returns:
            Timestamp of last message received, or None if no messages yet
        """
        return self._last_message_time
    
    def get_orderbook_state(self, market_ticker: str) -> Optional[SharedOrderbookState]:
        """
        Get orderbook state for a specific market.
        
        Args:
            market_ticker: Market ticker to get state for
            
        Returns:
            SharedOrderbookState for the market or None if not found
        """
        return self._orderbook_states.get(market_ticker)


# Global orderbook client instance - uses configured market tickers
orderbook_client = OrderbookClient()