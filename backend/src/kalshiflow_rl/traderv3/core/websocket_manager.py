"""
TRADER V3 WebSocket Manager - Simple Status Broadcasting.

Lightweight WebSocket manager focused only on V3 trader status broadcasting.
Provides real-time updates for state machine transitions, orderbook metrics,
and system health to the frontend console.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from collections import deque
import weakref

from starlette.websockets import WebSocket, WebSocketDisconnect

from .event_bus import EventBus, EventType, StateTransitionEvent, TraderStatusEvent, WhaleQueueEvent

# Import for type hints only to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..services.whale_tracker import WhaleTracker
    from ..services.trading_decision_service import TradingDecisionService
    from ..services.whale_execution_service import WhaleExecutionService
    from .state_container import StateContainer

logger = logging.getLogger("kalshiflow_rl.traderv3.websocket_manager")


@dataclass
class WebSocketClient:
    """Represents a connected WebSocket client."""
    websocket: WebSocket
    client_id: str
    connected_at: float
    last_ping: Optional[float] = None
    subscriptions: Set[str] = None
    
    def __post_init__(self):
        if self.subscriptions is None:
            self.subscriptions = {"state_transitions", "trader_status", "orderbook_metrics"}


class V3WebSocketManager:
    """
    Simple WebSocket manager for TRADER V3 status broadcasting.
    
    Features:
    - Real-time state machine transition broadcasting
    - Orderbook metrics and health status updates
    - Console-style message formatting for frontend
    - Connection management with automatic cleanup
    - Event bus integration for seamless updates
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None, state_machine=None, whale_tracker: Optional['WhaleTracker'] = None):
        """
        Initialize WebSocket manager.

        Args:
            event_bus: EventBus instance for subscribing to events
            state_machine: State machine instance for getting current state
            whale_tracker: Optional WhaleTracker instance for sending whale queue snapshots
        """
        self._event_bus = event_bus
        self._state_machine = state_machine
        self._whale_tracker = whale_tracker
        self._state_container: Optional['StateContainer'] = None
        self._trading_service: Optional['TradingDecisionService'] = None
        self._whale_execution_service: Optional['WhaleExecutionService'] = None
        self._market_price_syncer = None  # Set via set_market_price_syncer()
        self._clients: Dict[str, WebSocketClient] = {}
        self._client_counter = 0
        self._started_at: Optional[float] = None
        self._running = False

        # Message statistics
        self._messages_sent = 0
        self._connection_count = 0
        self._active_connections = 0

        # Periodic tasks
        self._ping_task: Optional[asyncio.Task] = None
        self._ping_interval = 30.0  # seconds

        # Message coalescing - batch rapid updates within a window
        self._pending_messages: Dict[str, Dict[str, Any]] = {}  # type -> latest message data
        self._coalesce_task: Optional[asyncio.Task] = None
        self._coalesce_interval = 0.1  # 100ms batching window

        # State transition history buffer (last 20 transitions)
        # This ensures late-connecting clients can see the startup sequence
        self._state_transition_history: deque = deque(maxlen=20)

        logger.info("TRADER V3 WebSocket Manager initialized")

    def set_whale_tracker(self, whale_tracker: 'WhaleTracker') -> None:
        """
        Set the whale tracker instance for sending snapshots to new clients.

        This allows whale_tracker to be set after initialization since it's
        created later in the startup sequence.

        Args:
            whale_tracker: WhaleTracker instance
        """
        self._whale_tracker = whale_tracker
        logger.info("WhaleTracker set on WebSocket manager")

    def set_trading_service(self, trading_service: 'TradingDecisionService') -> None:
        """
        Set the trading decision service for getting followed whale IDs.

        This allows trading_service to be set after initialization since it's
        created later in the startup sequence.

        Args:
            trading_service: TradingDecisionService instance
        """
        self._trading_service = trading_service
        logger.info("TradingDecisionService set on WebSocket manager")

    def set_whale_execution_service(self, whale_execution_service: 'WhaleExecutionService') -> None:
        """
        Set the whale execution service for getting decision history.

        The WhaleExecutionService is the event-driven service that actually
        processes whales and records all decisions (followed, skipped, rate_limited).
        Its decision history is the authoritative source for the Decision Audit panel.

        Args:
            whale_execution_service: WhaleExecutionService instance
        """
        self._whale_execution_service = whale_execution_service
        logger.info("WhaleExecutionService set on WebSocket manager")

    def set_state_container(self, state_container: 'StateContainer') -> None:
        """
        Set the state container for sending trading state to new clients.

        This allows immediate trading state broadcast when clients connect,
        rather than waiting for the next periodic update.

        Args:
            state_container: StateContainer instance
        """
        self._state_container = state_container
        logger.info("StateContainer set on WebSocket manager")

    def set_market_price_syncer(self, market_price_syncer) -> None:
        """
        Set the market price syncer for including health in initial trading state.

        Args:
            market_price_syncer: MarketPriceSyncer instance
        """
        self._market_price_syncer = market_price_syncer
        logger.debug("MarketPriceSyncer set on WebSocket manager")

    async def start(self) -> None:
        """Start the WebSocket manager."""
        if self._running:
            logger.warning("WebSocket manager is already running")
            return
        
        self._running = True
        self._started_at = time.time()
        
        # Subscribe to event bus if provided
        if self._event_bus:
            self._event_bus.subscribe(EventType.SYSTEM_ACTIVITY, self._handle_system_activity)
            self._event_bus.subscribe(EventType.TRADER_STATUS, self._handle_trader_status)
            self._event_bus.subscribe(EventType.WHALE_QUEUE_UPDATED, self._handle_whale_queue_update)
            logger.info("Subscribed to event bus for real-time updates")
        
        # Start periodic tasks
        self._ping_task = asyncio.create_task(self._ping_clients())
        
        logger.info("✅ TRADER V3 WebSocket Manager started")
    
    async def stop(self) -> None:
        """Stop the WebSocket manager and disconnect all clients."""
        if not self._running:
            return
        
        logger.info("Stopping TRADER V3 WebSocket Manager...")
        self._running = False
        
        # Cancel periodic tasks
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

        # Cancel coalesce task if running
        if self._coalesce_task and not self._coalesce_task.done():
            self._coalesce_task.cancel()
            try:
                await self._coalesce_task
            except asyncio.CancelledError:
                pass

        # Disconnect all clients
        disconnect_tasks = []
        for client in list(self._clients.values()):
            disconnect_tasks.append(self._disconnect_client(client.client_id))
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        logger.info(f"✅ TRADER V3 WebSocket Manager stopped. Messages sent: {self._messages_sent}")
    
    async def handle_websocket(self, websocket: WebSocket) -> None:
        """
        Handle new WebSocket connection.
        
        Args:
            websocket: WebSocket connection to handle
        """
        try:
            # Accept the connection first
            await websocket.accept()
            
            # Generate unique client ID
            self._client_counter += 1
            client_id = f"client_{self._client_counter}_{int(time.time())}"
            
            # Create client record
            client = WebSocketClient(
                websocket=websocket,
                client_id=client_id,
                connected_at=time.time()
            )
            
            self._clients[client_id] = client
            self._connection_count += 1
            self._active_connections += 1
            
            logger.info(f"WebSocket client connected: {client_id}")
            
            # Send initial connection acknowledgment
            await self._send_to_client(client_id, {
                "type": "connection",
                "data": {
                    "client_id": client_id,
                    "timestamp": time.strftime("%H:%M:%S"),
                    "message": "Connected to TRADER V3 console"
                }
            })
            
            # Small delay to ensure connection is stable
            await asyncio.sleep(0.1)
            
            # Replay historical state transitions to bring new clients up to date
            # This ensures late-connecting clients see the full startup sequence
            # including calibration steps and state changes
            if self._state_transition_history:
                logger.info(f"Replaying {len(self._state_transition_history)} historical state transitions to client {client_id}")
                
                # Send transitions in a batch to avoid overwhelming the connection
                history_batch = {
                    "type": "history_replay", 
                    "data": {
                        "transitions": [msg["data"] for msg in self._state_transition_history],
                        "count": len(self._state_transition_history)
                    }
                }
                
                # Ensure client still exists before sending
                if client_id in self._clients:
                    await self._send_to_client(client_id, history_batch)
                    await asyncio.sleep(0.1)  # Brief pause after history replay

            # Send whale queue snapshot if whale tracker is available
            if self._whale_tracker and client_id in self._clients:
                try:
                    queue_state = self._whale_tracker.get_queue_state()
                    whale_snapshot = {
                        "type": "whale_queue",
                        "data": {
                            "queue": queue_state.get("queue", []),
                            "stats": queue_state.get("stats", {
                                "trades_seen": 0,
                                "trades_discarded": 0,
                                "discard_rate_percent": 0
                            }),
                            "timestamp": time.strftime("%H:%M:%S")
                        }
                    }
                    await self._send_to_client(client_id, whale_snapshot)
                    logger.debug(f"Sent whale queue snapshot to client {client_id}: {len(queue_state.get('queue', []))} whales")
                except Exception as e:
                    logger.warning(f"Could not send whale queue snapshot to client {client_id}: {e}")

            # Send trading state snapshot immediately (don't wait for periodic broadcast)
            # Includes market_price_syncer health if available for immediate visibility.
            if self._state_container and client_id in self._clients:
                try:
                    trading_summary = self._state_container.get_trading_summary()
                    if trading_summary.get("has_state"):
                        # Build market price syncer health if available
                        market_price_syncer_health = None
                        if self._market_price_syncer:
                            syncer_health = self._market_price_syncer.get_health_details()
                            market_price_syncer_health = {
                                "healthy": syncer_health.get("healthy", False),
                                "sync_count": syncer_health.get("sync_count", 0),
                                "tickers_synced": syncer_health.get("tickers_synced", 0),
                                "last_sync_age_seconds": syncer_health.get("last_sync_age_seconds"),
                                "sync_errors": syncer_health.get("sync_errors", 0),
                            }

                        trading_state_msg = {
                            "type": "trading_state",
                            "data": {
                                "timestamp": time.time(),
                                "version": trading_summary["version"],
                                "balance": trading_summary["balance"],
                                "portfolio_value": trading_summary["portfolio_value"],
                                "position_count": trading_summary["position_count"],
                                "order_count": trading_summary["order_count"],
                                "positions": trading_summary["positions"],
                                "open_orders": trading_summary["open_orders"],
                                "order_list": trading_summary.get("order_list", []),
                                "sync_timestamp": trading_summary["sync_timestamp"],
                                "pnl": trading_summary.get("pnl"),
                                "positions_details": trading_summary.get("positions_details", []),
                                "settlements": trading_summary.get("settlements", []),
                                "settlements_count": trading_summary.get("settlements_count", 0),
                                # Include market prices (merged into positions_details)
                                "market_prices": trading_summary.get("market_prices"),
                                # Health fields - syncer included if available
                                "position_listener": None,
                                "market_ticker_listener": None,
                                "market_price_syncer": market_price_syncer_health,
                            }
                        }
                        await self._send_to_client(client_id, trading_state_msg)
                        logger.info(f"Sent immediate trading state to client {client_id}: {trading_summary['position_count']} positions, {trading_summary.get('settlements_count', 0)} settlements")
                except Exception as e:
                    logger.warning(f"Could not send trading state to client {client_id}: {e}")

            # Now handle incoming messages
            # (Current state is already included in the historical transitions replay)
            async for message in websocket.iter_text():
                await self._handle_client_message(client_id, message)
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket client {client_id}: {e}")
        finally:
            await self._disconnect_client(client_id)
    
    async def broadcast_message(self, message_type: str, data: Dict[str, Any]) -> None:
        """
        Queue message for coalesced broadcast.

        Critical message types are sent immediately.
        Frequent updates (trading_state, trader_status, whale_queue) are
        coalesced within a 100ms window to batch rapid updates.

        Args:
            message_type: Type of message
            data: Message data
        """
        # Critical types need immediate delivery - no coalescing
        critical_types = ("state_transition", "whale_processing", "connection", "system_activity", "history_replay")
        if message_type in critical_types:
            await self._broadcast_immediate(message_type, data)
            return

        # Coalesce frequent updates (trading_state, trader_status, whale_queue)
        # Later messages of the same type replace earlier ones within the window
        self._pending_messages[message_type] = data

        # Start coalesce task if not already running
        if not self._coalesce_task or self._coalesce_task.done():
            self._coalesce_task = asyncio.create_task(self._flush_pending())

    async def _flush_pending(self) -> None:
        """Flush pending messages after coalesce interval."""
        await asyncio.sleep(self._coalesce_interval)

        # Atomically grab and clear pending messages
        messages = self._pending_messages.copy()
        self._pending_messages.clear()

        # Broadcast each coalesced message
        for msg_type, data in messages.items():
            await self._broadcast_immediate(msg_type, data)

    async def _broadcast_immediate(self, message_type: str, data: Dict[str, Any]) -> None:
        """
        Broadcast message immediately to all connected clients.

        Args:
            message_type: Type of message
            data: Message data
        """
        if not self._clients:
            return

        message = {
            "type": message_type,
            "data": data,
            "timestamp": time.time()
        }

        # Send to all connected clients
        send_tasks = []
        for client_id in list(self._clients.keys()):
            send_tasks.append(self._send_to_client(client_id, message))

        if send_tasks:
            await asyncio.gather(*send_tasks, return_exceptions=True)
    
    async def broadcast_console_message(self, level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Broadcast console-style message to all clients.
        
        Args:
            level: Log level (info, warning, error, debug)
            message: Console message
            context: Optional additional context
        """
        timestamp_str = time.strftime("%H:%M:%S", time.localtime())
        
        await self.broadcast_message("console", {
            "level": level,
            "timestamp": timestamp_str,
            "message": message,
            "context": context or {}
        })
    
    async def _handle_system_activity(self, event) -> None:
        """Handle unified system activity events from event bus."""
        logger.debug(f"Handling system activity: {event.activity_type} - {event.message}")

        # Handle whale_processing events specially for frontend animation
        if event.activity_type == "whale_processing":
            await self._handle_whale_processing(event)
            return
        
        # Extract current state from metadata if this is a state transition
        current_state = "ready"  # Default fallback state
        if event.activity_type == "state_transition" and event.metadata:
            # Get the to_state from metadata
            current_state = event.metadata.get("to_state")
            if current_state and hasattr(current_state, 'lower'):
                current_state = current_state.lower()
            elif current_state:
                current_state = str(current_state).lower()
        elif self._state_machine:
            # Fall back to state machine's current state for other activities
            try:
                if hasattr(self._state_machine, 'current_state'):
                    if hasattr(self._state_machine.current_state, 'value'):
                        current_state = self._state_machine.current_state.value.lower()
                    else:
                        current_state = str(self._state_machine.current_state).lower()
                else:
                    current_state = "ready"
            except Exception as e:
                logger.debug(f"Could not get state from state_machine: {e}, using 'ready' as fallback")
                current_state = "ready"
        
        # Ensure state is never None or undefined
        if not current_state or current_state == "none" or current_state == "unknown":
            current_state = "ready"
        
        # Format the activity message
        activity_data = {
            "timestamp": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
            "activity_type": event.activity_type,
            "message": event.message,
            "metadata": event.metadata,
            "state": current_state  # Include current state in all system activities
        }
        
        # For state transitions, extract from_state and to_state from metadata
        if event.activity_type == "state_transition" and event.metadata:
            if "from_state" in event.metadata:
                activity_data["from_state"] = event.metadata["from_state"]
            if "to_state" in event.metadata:
                activity_data["to_state"] = event.metadata["to_state"]
        
        activity_message = {
            "type": "system_activity",
            "data": activity_data
        }
        
        # Store state transitions in history for late-connecting clients
        if event.activity_type == "state_transition":
            self._state_transition_history.append(activity_message)
        
        # Broadcast to currently connected clients
        await self.broadcast_message("system_activity", activity_message["data"])

    async def _handle_trader_status(self, event: TraderStatusEvent) -> None:
        """Handle trader status events from event bus."""
        logger.debug(f"Handling trader status event: {event.state}")
        
        # Debug: Check if ping data is in the metrics
        if event.metrics:
            logger.debug(f"WebSocket manager metrics contains ping_health: {event.metrics.get('ping_health')}, last_ping_age: {event.metrics.get('last_ping_age')}")
        
        # Only send trader_status for metrics updates, not console messages
        # The frontend will update metrics silently
        await self.broadcast_message("trader_status", {
            "timestamp": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
            "state": event.state,
            "health": event.health,
            "metrics": event.metrics
        })

    async def _handle_whale_queue_update(self, event: WhaleQueueEvent) -> None:
        """
        Handle whale queue update events from event bus.

        Broadcasts whale queue state to all connected frontend clients,
        enabling the Follow the Whale feature in the V3 console.

        Includes:
        - Current whale queue
        - Followed whales and IDs
        - Decision history (why whales were followed/skipped)
        - Decision statistics

        Args:
            event: WhaleQueueEvent containing queue contents and stats
        """
        logger.debug(f"Handling whale queue update: {len(event.queue)} whales")

        # Calculate discard rate for stats
        discard_rate = 0.0
        if event.stats and event.stats.get("trades_seen", 0) > 0:
            discard_rate = (event.stats.get("trades_discarded", 0) / event.stats["trades_seen"]) * 100

        # Get followed whale IDs and full data from trading service
        # Get decision history from whale execution service (authoritative source)
        followed_whale_ids: List[str] = []
        followed_whales: List[Dict] = []
        decision_history: List[Dict] = []
        decision_stats: Dict[str, Any] = {}

        # Get followed whales from TradingDecisionService (tracks successful follows)
        if self._trading_service:
            try:
                followed_whale_ids = list(self._trading_service.get_followed_whale_ids())
                followed_whales = self._trading_service.get_followed_whales()
            except Exception as e:
                logger.debug(f"Could not get trading service data: {e}")

        # Get decision history from WhaleExecutionService (authoritative source for all decisions)
        # This includes followed, skipped, rate_limited - everything the Decision Audit needs
        if self._whale_execution_service:
            try:
                decision_history = self._whale_execution_service.get_decision_history()
                stats = self._whale_execution_service.get_stats()
                decision_stats = {
                    "whales_detected": stats.get("whales_processed", 0),
                    "whales_followed": stats.get("whales_followed", 0),
                    "whales_skipped": stats.get("whales_skipped", 0),
                    "rate_limited": stats.get("rate_limited_count", 0),
                    # Categorized skip reasons for Decision Audit panel breakdown
                    "skipped_age": stats.get("skipped_age", 0),
                    "skipped_position": stats.get("skipped_position", 0),
                    "skipped_orders": stats.get("skipped_orders", 0),
                    "already_followed": stats.get("already_followed", 0),
                    "failed": stats.get("failed", 0),
                }
            except Exception as e:
                logger.debug(f"Could not get whale execution service data: {e}")

        # Format message for frontend
        whale_data = {
            "queue": event.queue,  # Already serialized by WhaleTracker
            "stats": {
                "trades_seen": event.stats.get("trades_seen", 0) if event.stats else 0,
                "trades_discarded": event.stats.get("trades_discarded", 0) if event.stats else 0,
                "discard_rate_percent": round(discard_rate, 1),
            },
            "followed_whale_ids": followed_whale_ids,  # IDs of whales we have followed
            "followed_whales": followed_whales,  # Full data for followed trades section
            "decision_history": decision_history,  # Recent decisions with reasons
            "decision_stats": decision_stats,  # Aggregate stats (detected/followed/skipped)
            "timestamp": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
        }

        await self.broadcast_message("whale_queue", whale_data)

    async def _handle_whale_processing(self, event) -> None:
        """
        Handle whale processing events for frontend animation.

        Broadcasts processing state to all clients so they can show
        a subtle glow/pulse animation on the whale row being processed.

        Args:
            event: SystemActivityEvent with activity_type="whale_processing"
        """
        metadata = event.metadata or {}
        whale_id = metadata.get("whale_id", "")
        status = metadata.get("status", "")  # "processing" or "complete"
        action = metadata.get("action", "")  # "followed", "skipped_*", etc.

        await self.broadcast_message("whale_processing", {
            "whale_id": whale_id,
            "status": status,
            "action": action,
            "timestamp": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
        })

    async def _handle_client_message(self, client_id: str, message: str) -> None:
        """
        Handle message from WebSocket client.
        
        Args:
            client_id: Client ID
            message: Raw message string
        """
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "ping":
                # Respond to ping with pong
                await self._send_to_client(client_id, {
                    "type": "pong",
                    "timestamp": time.time()
                })
                
                # Update last ping time
                if client_id in self._clients:
                    self._clients[client_id].last_ping = time.time()
                    
            elif message_type == "subscribe":
                # Handle subscription changes
                subscriptions = set(data.get("subscriptions", []))
                if client_id in self._clients:
                    self._clients[client_id].subscriptions = subscriptions
                    logger.info(f"Client {client_id} subscribed to: {subscriptions}")
                    
            else:
                logger.debug(f"Unknown message type from client {client_id}: {message_type}")
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client {client_id}: {message}")
        except Exception as e:
            logger.error(f"Error handling message from client {client_id}: {e}")
    
    async def _send_to_client(self, client_id: str, message: Dict[str, Any]) -> None:
        """
        Send message to specific client with proper error handling.
        
        Args:
            client_id: Target client ID
            message: Message to send
        """
        if client_id not in self._clients:
            logger.debug(f"Client {client_id} not found, skipping message")
            return
        
        client = self._clients.get(client_id)
        if not client:
            return
        
        try:
            # Send message directly - starlette websockets handle their own state internally
            await client.websocket.send_text(json.dumps(message))
            self._messages_sent += 1
                
        except (RuntimeError, ConnectionError) as e:
            # Connection errors are expected when clients disconnect
            logger.debug(f"Connection error for client {client_id}: {e}")
            await self._disconnect_client(client_id)
        except Exception as e:
            # Log unexpected errors as warnings
            logger.warning(f"Unexpected error sending to client {client_id}: {e}")
            await self._disconnect_client(client_id)
    
    async def _disconnect_client(self, client_id: str) -> None:
        """
        Disconnect and remove client safely.
        
        Args:
            client_id: Client ID to disconnect
        """
        # Atomic pop - avoids check-then-pop race condition
        client = self._clients.pop(client_id, None)
        if not client:
            return
        
        # Update connection count
        self._active_connections = max(0, self._active_connections - 1)
        
        try:
            # Try to close the WebSocket connection gracefully
            await client.websocket.close()
        except Exception:
            pass  # Connection might already be closed
        
        logger.info(f"WebSocket client removed: {client_id} (active: {self._active_connections})")
    
    async def _ping_clients(self) -> None:
        """Periodic ping to maintain client connections."""
        while self._running:
            try:
                await asyncio.sleep(self._ping_interval)
                
                if not self._clients:
                    continue
                
                # Send ping to all clients
                ping_tasks = []
                for client_id in list(self._clients.keys()):
                    ping_tasks.append(self._send_to_client(client_id, {
                        "type": "ping",
                        "timestamp": time.time()
                    }))
                
                if ping_tasks:
                    await asyncio.gather(*ping_tasks, return_exceptions=True)
                
                logger.debug(f"Sent ping to {len(self._clients)} connected clients")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping task: {e}")
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics."""
        uptime = time.time() - self._started_at if self._started_at else 0
        
        return {
            "running": self._running,
            "active_connections": self._active_connections,
            "total_connections": self._connection_count,
            "messages_sent": self._messages_sent,
            "uptime_seconds": uptime,
            "messages_per_second": self._messages_sent / max(uptime, 1),
            "clients": [
                {
                    "client_id": client.client_id,
                    "connected_at": client.connected_at,
                    "connection_duration": time.time() - client.connected_at,
                    "last_ping": client.last_ping,
                    "subscriptions": list(client.subscriptions)
                }
                for client in self._clients.values()
            ]
        }
    
    def is_healthy(self) -> bool:
        """Check if WebSocket manager is healthy."""
        return self._running and self._ping_task is not None and not self._ping_task.done()
    
    def get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information."""
        stats = self.get_stats()
        return {
            "running": self._running,
            "ping_task_active": self._ping_task is not None and not self._ping_task.done(),
            "active_connections": stats["active_connections"],
            "total_connections": stats["total_connections"],
            "messages_sent": stats["messages_sent"],
            "uptime_seconds": stats["uptime_seconds"],
            "event_bus_connected": self._event_bus is not None
        }