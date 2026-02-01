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
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from collections import deque
import weakref

from starlette.websockets import WebSocket, WebSocketDisconnect


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

from .event_bus import EventBus, EventType, StateTransitionEvent, TraderStatusEvent, TradeFlowMarketUpdateEvent, TradeFlowTradeArrivedEvent

# Import for type hints only to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..services.trading_decision_service import TradingDecisionService
    from ..services.upcoming_markets_syncer import UpcomingMarketsSyncer
    from ..state.tracked_markets import TrackedMarketsState
    from .state_container import V3StateContainer
    from ..strategies import DeepAgentStrategy
    from ..deep_agent import SelfImprovingAgent
    from ..agents.reddit_entity_agent import RedditEntityAgent

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
    
    def __init__(self, event_bus: Optional[EventBus] = None, state_machine=None):
        """
        Initialize WebSocket manager.

        Args:
            event_bus: EventBus instance for subscribing to events
            state_machine: State machine instance for getting current state
        """
        self._event_bus = event_bus
        self._state_machine = state_machine
        self._state_container: Optional['V3StateContainer'] = None
        self._trading_service: Optional['TradingDecisionService'] = None
        self._deep_agent_strategy: Optional['DeepAgentStrategy'] = None  # Set via set_deep_agent_strategy()
        self._trade_flow_service = None  # Set via set_trade_flow_service()
        self._market_price_syncer = None  # Set via set_market_price_syncer()
        self._tracked_markets_state: Optional['TrackedMarketsState'] = None  # Set via set_tracked_markets_state()
        self._upcoming_markets_syncer: Optional['UpcomingMarketsSyncer'] = None  # Set via set_upcoming_markets_syncer()
        self._deep_agent: Optional['SelfImprovingAgent'] = None  # Set via set_deep_agent()
        self._reddit_agent: Optional['RedditEntityAgent'] = None  # Set via set_reddit_agent()
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

        # Trade processing heartbeat
        self._trade_processing_task: Optional[asyncio.Task] = None
        self._trade_processing_interval = 1.5  # 1.5 seconds

        # Strategy panel heartbeat
        self._strategy_panel_task: Optional[asyncio.Task] = None
        self._strategy_panel_interval = 5.0  # 5 seconds

        # State transition history buffer (last 20 transitions)
        # This ensures late-connecting clients can see the startup sequence
        self._state_transition_history: deque = deque(maxlen=20)

        # Activity feed history buffer (lifecycle events + system activities)
        # This ensures late-connecting clients can see recent activity feed events
        # when switching between Trader/Discovery views
        self._activity_feed_history: deque = deque(maxlen=100)

        logger.info("TRADER V3 WebSocket Manager initialized")

    def set_trading_service(self, trading_service: 'TradingDecisionService') -> None:
        """
        Set the trading decision service.

        Args:
            trading_service: TradingDecisionService instance
        """
        self._trading_service = trading_service
        logger.info("TradingDecisionService set on WebSocket manager")

    def set_state_container(self, state_container: 'V3StateContainer') -> None:
        """
        Set the state container for sending trading state to new clients.

        This allows immediate trading state broadcast when clients connect,
        rather than waiting for the next periodic update.

        Args:
            state_container: V3StateContainer instance
        """
        self._state_container = state_container
        logger.info("V3StateContainer set on WebSocket manager")

    def set_market_price_syncer(self, market_price_syncer) -> None:
        """
        Set the market price syncer for including health in initial trading state.

        Args:
            market_price_syncer: MarketPriceSyncer instance
        """
        self._market_price_syncer = market_price_syncer
        logger.debug("MarketPriceSyncer set on WebSocket manager")

    def set_tracked_markets_state(self, tracked_markets_state: 'TrackedMarketsState') -> None:
        """
        Set the tracked markets state for lifecycle discovery mode.

        This enables sending tracked markets snapshots to new clients
        and broadcasting lifecycle events (new markets tracked, status changes).

        Args:
            tracked_markets_state: TrackedMarketsState instance
        """
        self._tracked_markets_state = tracked_markets_state
        logger.info("TrackedMarketsState set on WebSocket manager")

    def set_deep_agent_strategy(self, strategy: 'DeepAgentStrategy') -> None:
        """
        Set the deep agent strategy for trade processing broadcasts.

        Args:
            strategy: DeepAgentStrategy instance
        """
        self._deep_agent_strategy = strategy
        logger.info("DeepAgentStrategy set on WebSocket manager")

        # Start trade processing heartbeat if manager is already running
        if self._running and (not self._trade_processing_task or self._trade_processing_task.done()):
            self._trade_processing_task = asyncio.create_task(self._trade_processing_heartbeat())
            logger.info("Started trade processing heartbeat (1.5s interval)")

        # Start strategy panel heartbeat if manager is already running
        if self._running and (not self._strategy_panel_task or self._strategy_panel_task.done()):
            self._strategy_panel_task = asyncio.create_task(self._strategy_panel_heartbeat())
            logger.info("Started strategy panel heartbeat (5s interval)")

    def set_trade_flow_service(self, service) -> None:
        """
        Set the trade flow service for providing trade state snapshots.

        When set, the WebSocketManager will prefer this service for
        trade flow market states.

        Args:
            service: TradeFlowService instance
        """
        self._trade_flow_service = service
        logger.info("TradeFlowService set on WebSocket manager")

    def set_upcoming_markets_syncer(self, syncer: 'UpcomingMarketsSyncer') -> None:
        """
        Set the upcoming markets syncer for sending snapshots on connect.

        This enables sending upcoming markets schedule to new clients when
        they connect, providing visibility into markets opening soon.

        Args:
            syncer: UpcomingMarketsSyncer instance
        """
        self._upcoming_markets_syncer = syncer
        logger.info("UpcomingMarketsSyncer set on WebSocket manager")

    def set_deep_agent(self, agent: 'SelfImprovingAgent') -> None:
        """
        Set the deep agent for sending snapshots on client connect.

        This enables sending deep agent state snapshots to new clients
        when they connect, allowing session persistence across page refreshes.

        Args:
            agent: SelfImprovingAgent instance
        """
        self._deep_agent = agent
        logger.info("SelfImprovingAgent set on WebSocket manager")

    def set_reddit_agent(self, agent: 'RedditEntityAgent') -> None:
        """
        Set the reddit entity agent for sending entity snapshots on client connect.

        This enables sending entity pipeline state (recent posts, extractions,
        price impacts) to new clients, allowing session persistence across
        page refreshes.

        Args:
            agent: RedditEntityAgent instance
        """
        self._reddit_agent = agent
        logger.info("RedditEntityAgent set on WebSocket manager")

    def _add_to_activity_feed_history(self, message_type: str, data: dict) -> None:
        """
        Track events for Activity Feed history replay.

        Stores system_activity and lifecycle_event messages so they can be
        replayed to clients when they reconnect (e.g., switching between
        Trader and Discovery views).

        Args:
            message_type: Type of message (system_activity, lifecycle_event)
            data: Message data
        """
        if message_type in ("system_activity", "lifecycle_event"):
            self._activity_feed_history.append({
                "type": message_type,
                "data": data,
                "timestamp": time.time()
            })

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
            # Trade flow events (market microstructure tracking)
            await self._event_bus.subscribe_to_trade_flow_market_update(self._handle_trade_flow_market_update)
            await self._event_bus.subscribe_to_trade_flow_trade_arrived(self._handle_trade_flow_trade_arrived)
            logger.info("Subscribed to event bus for real-time updates")
        
        # Start periodic tasks
        self._ping_task = asyncio.create_task(self._ping_clients())

        # Start trade processing heartbeat if deep agent strategy is available
        if self._deep_agent_strategy:
            self._trade_processing_task = asyncio.create_task(self._trade_processing_heartbeat())
            logger.info("Started trade processing heartbeat (1.5s interval)")

        # Start strategy panel heartbeat if deep agent strategy is available
        if self._deep_agent_strategy:
            self._strategy_panel_task = asyncio.create_task(self._strategy_panel_heartbeat())
            logger.info("Started strategy panel heartbeat (5s interval)")

        logger.info("TRADER V3 WebSocket Manager started")
    
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

        # Cancel trade processing heartbeat task
        if self._trade_processing_task and not self._trade_processing_task.done():
            self._trade_processing_task.cancel()
            try:
                await self._trade_processing_task
            except asyncio.CancelledError:
                pass

        # Cancel strategy panel heartbeat task
        if self._strategy_panel_task and not self._strategy_panel_task.done():
            self._strategy_panel_task.cancel()
            try:
                await self._strategy_panel_task
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
                                # Note: position_listener and market_ticker_listener are None here
                                # because websocket_manager doesn't have references to them.
                                # Periodic broadcasts from status_reporter include real values.
                                "position_listener": None,
                                "market_ticker_listener": None,
                                "market_price_syncer": market_price_syncer_health,
                                # Order group - must match status_reporter.py
                                "order_group": trading_summary.get("order_group"),
                                # Changes since last update - match status_reporter.py format
                                "changes": trading_summary.get("changes"),
                            }
                        }
                        await self._send_to_client(client_id, trading_state_msg)
                        logger.info(f"Sent immediate trading state to client {client_id}: {trading_summary['position_count']} positions, {trading_summary.get('settlements_count', 0)} settlements")
                except Exception as e:
                    logger.warning(f"Could not send trading state to client {client_id}: {e}")

            # Send tracked markets snapshot if in lifecycle discovery mode
            if self._tracked_markets_state and client_id in self._clients:
                await self._send_tracked_markets_snapshot(client_id)

            # Send trade flow market states snapshot if trade flow service or deep agent is available
            if (self._trade_flow_service or self._deep_agent_strategy) and client_id in self._clients:
                await self._send_trade_flow_states_snapshot(client_id)
                # Also send initial trade_processing snapshot
                if self._deep_agent_strategy:
                    await self._send_trade_processing_snapshot(client_id)

            # Send trading strategies panel snapshot if deep agent is available
            if self._deep_agent_strategy and client_id in self._clients:
                await self._send_trading_strategies_snapshot(client_id)

            # Send upcoming markets snapshot if syncer is available
            if self._upcoming_markets_syncer and client_id in self._clients:
                await self._send_upcoming_markets_snapshot(client_id)

            # Send event research snapshot (Events tab persistence)
            if self._state_container and client_id in self._clients:
                await self._send_event_research_snapshot(client_id)

            # Send activity feed history for replay (system_activity + lifecycle_event)
            # This ensures the Activity Feed is populated when switching views
            if self._activity_feed_history and client_id in self._clients:
                await self._send_activity_feed_history(client_id)

            # Send deep agent snapshot for session persistence (Agent tab)
            if self._deep_agent and client_id in self._clients:
                await self._send_deep_agent_snapshot(client_id)

            # Send entity snapshot for session persistence (Reddit posts, extractions)
            if self._reddit_agent and client_id in self._clients:
                await self._send_entity_snapshot(client_id)

            # Send event configs snapshot (active event configurations)
            if self._reddit_agent and client_id in self._clients:
                await self._send_event_configs_snapshot(client_id)

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
        Frequent updates (trading_state, trader_status) are
        coalesced within a 100ms window to batch rapid updates.

        Args:
            message_type: Type of message
            data: Message data
        """
        # Critical types need immediate delivery - no coalescing
        critical_types = (
            "state_transition", "connection", "system_activity", "history_replay",
            "deep_agent_status", "deep_agent_cycle", "deep_agent_error"
        )
        if message_type in critical_types:
            await self._broadcast_immediate(message_type, data)
            return

        # Coalesce frequent updates (trading_state, trader_status)
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

        # Forward lifecycle events to Activity Feed
        if event.activity_type == "lifecycle_event":
            metadata = event.metadata or {}
            await self.broadcast_lifecycle_event(
                event_type=metadata.get("event_type", "tracked"),
                market_ticker=metadata.get("market_ticker", ""),
                action=metadata.get("action", "tracked"),
                reason=metadata.get("reason"),
                metadata=metadata
            )
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

        # Store activity feed events for replay to reconnecting clients
        # (excludes state transitions which have separate history)
        if event.activity_type != "state_transition":
            self._add_to_activity_feed_history("system_activity", activity_message["data"])

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

    async def _handle_trade_flow_market_update(self, event: TradeFlowMarketUpdateEvent) -> None:
        """
        Handle trade flow market state update events from event bus.

        Broadcasts market trade state (YES/NO counts, price movement) to
        all connected frontend clients for the trade flow UI.

        Args:
            event: TradeFlowMarketUpdateEvent containing market state
        """
        await self.broadcast_message("trade_flow_market_state", {
            "ticker": event.market_ticker,
            "market_ticker": event.market_ticker,
            **event.state,
            "timestamp": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
        })

    async def _handle_trade_flow_trade_arrived(self, event: TradeFlowTradeArrivedEvent) -> None:
        """
        Handle trade flow trade arrived events from event bus.

        Broadcasts lightweight trade notification to all connected clients,
        triggering pulse/glow animations on the corresponding market card.

        Args:
            event: TradeFlowTradeArrivedEvent containing trade details
        """
        await self.broadcast_message("trade_flow_trade_arrived", {
            "ticker": event.market_ticker,
            "market_ticker": event.market_ticker,
            "event_ticker": event.event_ticker,
            "side": event.side,
            "count": event.count,
            "yes_price": event.price_cents,
            "timestamp": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
        })

    # ========== Trade Processing Heartbeat ==========

    async def _trade_processing_heartbeat(self) -> None:
        """
        Periodic broadcast of trade processing stats.

        Runs every 1.5 seconds to ensure stats are always visible in the UI,
        even when no tracked trades are arriving.
        """
        while self._running:
            try:
                await asyncio.sleep(self._trade_processing_interval)
                await self._broadcast_trade_processing()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trade processing heartbeat: {e}")

    async def _broadcast_trade_processing(self) -> None:
        """
        Broadcast trade processing state to all connected clients.

        Sends recent tracked trades, stats, and decision breakdown.
        """
        if not self._deep_agent_strategy:
            return

        data = self._build_trade_processing_data()
        await self.broadcast_message("trade_processing", data)

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
            # Use custom encoder to handle datetime objects in case they leak through
            await client.websocket.send_text(json.dumps(message, cls=DateTimeEncoder))
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

    # ========== Lifecycle Discovery Mode Methods ==========

    async def broadcast_tracked_markets(self) -> None:
        """
        Broadcast tracked markets state to all connected clients.

        Called when tracked markets state changes (new market tracked,
        status change, etc.). Sends full snapshot for simplicity.
        """
        if not self._tracked_markets_state:
            return

        snapshot = self._tracked_markets_state.get_snapshot()

        # Include trading attachments for each tracked market
        if self._state_container:
            for market in snapshot.get("markets", []):
                ticker = market.get("ticker")
                if ticker:
                    trading = self._state_container.get_trading_attachment_for_market(ticker)
                    market["trading"] = trading

        await self.broadcast_message("tracked_markets", snapshot)

    async def broadcast_lifecycle_event(
        self,
        event_type: str,
        market_ticker: str,
        action: str,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Broadcast a lifecycle event to all connected clients.

        Used for real-time updates when markets are tracked/untracked
        or status changes occur.

        Args:
            event_type: Type of event (created, determined, settled, closed)
            market_ticker: Affected market ticker
            action: Action taken (tracked, rejected, unsubscribed)
            reason: Optional reason for the action
            metadata: Optional additional metadata
        """
        event_data = {
            "event_type": event_type,
            "market_ticker": market_ticker,
            "action": action,
            "reason": reason,
            "metadata": metadata or {},
            "timestamp": time.strftime("%H:%M:%S"),
        }
        # Store for replay to reconnecting clients
        self._add_to_activity_feed_history("lifecycle_event", event_data)
        await self.broadcast_message("lifecycle_event", event_data)

    async def broadcast_market_info_update(
        self,
        ticker: str,
        price: int,
        volume: int,
        open_interest: Optional[int] = None,
        yes_bid: Optional[int] = None,
        yes_ask: Optional[int] = None
    ) -> None:
        """
        Broadcast market info update for a single tracked market.

        Called by TrackedMarketsSyncer when market info is refreshed.
        Provides real-time price/volume updates for the lifecycle grid.

        Args:
            ticker: Market ticker
            price: Current YES price in cents
            volume: Volume traded
            open_interest: Optional open interest
            yes_bid: Optional best YES bid
            yes_ask: Optional best YES ask
        """
        update_data = {
            "ticker": ticker,
            "price": price,
            "volume": volume,
            "open_interest": open_interest,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "timestamp": time.time(),
        }
        await self.broadcast_message("market_info_update", update_data)

    # ========== Trading Strategies Panel Heartbeat ==========

    async def _strategy_panel_heartbeat(self) -> None:
        """
        Periodic broadcast of trading strategies panel data.

        Runs every 5 seconds to provide strategy status, performance metrics,
        and skip breakdown to the frontend Trading Strategies Panel.
        """
        while self._running:
            try:
                await asyncio.sleep(self._strategy_panel_interval)
                await self._broadcast_trading_strategies()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in strategy panel heartbeat: {e}")

    async def _broadcast_trading_strategies(self) -> None:
        """Broadcast deep agent strategy panel data to all connected clients."""
        if not self._deep_agent_strategy:
            return

        try:
            panel_data = self._build_trading_strategies_data()
            await self.broadcast_message("trading_strategies", panel_data)
        except Exception as e:
            logger.error(f"Error broadcasting trading strategies: {e}")


    async def _send_snapshot(
        self,
        client_id: str,
        message_type: str,
        data_source: Any,
        get_data: Any,
        log_name: Optional[str] = None
    ) -> None:
        """
        Generic snapshot sender for client initialization.

        Consolidates the common pattern used by all _send_*_snapshot methods:
        check source exists, check client exists, try/except, log.

        Args:
            client_id: Target client ID
            message_type: WebSocket message type (e.g., "tracked_markets")
            data_source: Object to check for existence (return early if None)
            get_data: Callable that returns the snapshot data (can be sync or async)
            log_name: Optional name for logging (defaults to message_type)
        """
        if not data_source or client_id not in self._clients:
            return

        try:
            data = get_data()
            if asyncio.iscoroutine(data):
                data = await data
            await self._send_to_client(client_id, {"type": message_type, "data": data})
            logger.debug(f"Sent {log_name or message_type} snapshot to client {client_id}")
        except Exception as e:
            logger.warning(f"Could not send {log_name or message_type} snapshot to {client_id}: {e}")

    async def _send_tracked_markets_snapshot(self, client_id: str) -> None:
        """Send tracked markets snapshot to a specific client."""
        await self._send_snapshot(
            client_id,
            "tracked_markets",
            self._tracked_markets_state,
            lambda: self._build_tracked_markets_data(),
            log_name="tracked_markets"
        )

    def _build_tracked_markets_data(self) -> dict:
        """Build tracked markets snapshot data with trading attachments."""
        snapshot = self._tracked_markets_state.get_snapshot()
        # Include trading attachments for each tracked market
        if self._state_container:
            for market in snapshot.get("markets", []):
                ticker = market.get("ticker")
                if ticker:
                    trading = self._state_container.get_trading_attachment_for_market(ticker)
                    market["trading"] = trading
        return snapshot

    async def _send_trade_flow_states_snapshot(self, client_id: str) -> None:
        """Send trade flow market states snapshot to a specific client."""
        data_source = self._trade_flow_service or self._deep_agent_strategy
        await self._send_snapshot(
            client_id,
            "trade_flow_states_snapshot",
            data_source,
            lambda: self._build_trade_flow_states_data(),
            log_name="trade flow states"
        )

    def _build_trade_flow_states_data(self) -> dict:
        """Build trade flow market states snapshot data.

        Prefers TradeFlowService (dedicated trade feed) over
        DeepAgentStrategy (fallback).
        """
        if self._trade_flow_service:
            market_states = self._trade_flow_service.get_market_states(limit=100)
        elif self._deep_agent_strategy:
            market_states = self._deep_agent_strategy.get_market_states(limit=100)
        else:
            market_states = []
        return {
            "markets": market_states,
            "count": len(market_states),
            "timestamp": time.strftime("%H:%M:%S"),
        }

    async def _send_trade_processing_snapshot(self, client_id: str) -> None:
        """Send trade processing snapshot to a specific client."""
        await self._send_snapshot(
            client_id,
            "trade_processing",
            self._deep_agent_strategy,
            lambda: self._build_trade_processing_data(),
            log_name="trade_processing"
        )

    def _build_trade_processing_data(self) -> dict:
        """Build trade processing snapshot data."""
        stats = self._deep_agent_strategy.get_trade_processing_stats()
        recent_trades = self._deep_agent_strategy.get_recent_tracked_trades(limit=20)
        decision_history = self._deep_agent_strategy.get_decision_history(limit=20)

        total = stats.get("trades_processed", 0) + stats.get("trades_filtered", 0)
        filter_rate = round(stats.get("trades_filtered", 0) / total * 100, 1) if total > 0 else 0.0

        # Get low_balance counter from trading service if available
        low_balance = 0
        if self._trading_service:
            decision_stats = self._trading_service.get_decision_stats()
            low_balance = decision_stats.get("low_balance", 0)

        return {
            "recent_trades": recent_trades,
            "stats": {
                "trades_seen": total,
                "trades_filtered": stats.get("trades_filtered", 0),
                "trades_tracked": stats.get("trades_processed", 0),
                "filter_rate_percent": filter_rate,
            },
            "decisions": {
                "detected": stats.get("signals_detected", 0),
                "executed": stats.get("signals_executed", 0),
                "rate_limited": stats.get("rate_limited_count", 0),
                "skipped": stats.get("signals_skipped", 0),
                "reentries": stats.get("reentries", 0),
                "low_balance": low_balance,
            },
            "decision_history": decision_history,
            "last_updated": time.time(),
            "timestamp": time.strftime("%H:%M:%S"),
        }

    async def _send_trading_strategies_snapshot(self, client_id: str) -> None:
        """Send trading strategies panel snapshot to a specific client."""
        await self._send_snapshot(
            client_id,
            "trading_strategies",
            self._deep_agent_strategy,
            lambda: self._build_trading_strategies_data(),
            log_name="trading_strategies"
        )

    def _build_trading_strategies_data(self) -> dict:
        """Build trading strategies panel data from deep agent."""
        stats = self._deep_agent_strategy.get_stats()
        consolidated = self._deep_agent_strategy.get_consolidated_view()
        return {
            "strategies": {
                "deep_agent": {
                    "status": {
                        "running": stats.get("running", False),
                        "healthy": stats.get("healthy", False),
                        "uptime_seconds": stats.get("uptime_seconds", 0),
                    },
                    "performance": {
                        "cycles_run": stats.get("cycles_run", 0),
                        "trades_executed": stats.get("trades_executed", 0),
                        "win_rate": stats.get("win_rate", 0.0),
                    },
                    "consolidated_view": consolidated,
                }
            },
            "recent_decisions": self._deep_agent_strategy.get_decision_history(limit=15),
            "last_updated": time.time(),
            "timestamp": time.strftime("%H:%M:%S"),
        }

    async def _send_upcoming_markets_snapshot(self, client_id: str) -> None:
        """Send upcoming markets snapshot to a specific client."""
        await self._send_snapshot(
            client_id,
            "upcoming_markets",
            self._upcoming_markets_syncer,
            lambda: self._upcoming_markets_syncer.get_snapshot_message()["data"],
            log_name="upcoming_markets"
        )

    async def _send_event_research_snapshot(self, client_id: str) -> None:
        """
        Send cached event research results to a specific client.

        This enables the Events tab to show research that was broadcast
        before the client connected. The snapshot wraps the cached results
        in a trading_state message format that the frontend expects.
        """
        if not self._state_container:
            return

        event_research = self._state_container.get_event_research_results()
        if not event_research:
            return

        # Send as a trading_state snapshot with event_research included
        # This matches the format the frontend expects in useV3WebSocket.js
        await self._send_snapshot(
            client_id,
            "trading_state",
            event_research,
            lambda: {"event_research": event_research},
            log_name="event_research"
        )

    async def _send_activity_feed_history(self, client_id: str) -> None:
        """Send activity feed history to a specific client."""
        await self._send_snapshot(
            client_id,
            "activity_feed_history",
            self._activity_feed_history,
            lambda: self._build_activity_feed_data(),
            log_name="activity_feed_history"
        )

    def _build_activity_feed_data(self) -> dict:
        """Build activity feed history data."""
        history_list = list(self._activity_feed_history)
        return {
            "events": history_list,
            "count": len(history_list)
        }

    async def _send_deep_agent_snapshot(self, client_id: str) -> None:
        """
        Send deep agent snapshot to a specific client.

        This enables session persistence for the Deep Agent page,
        restoring cycle count, tool calls, trades, and settlements
        after a page refresh.
        """
        if not self._deep_agent:
            return
        if client_id not in self._clients:
            return

        try:
            snapshot = self._deep_agent.get_snapshot()
            await self._send_to_client(client_id, {
                "type": "deep_agent_snapshot",
                "data": snapshot
            })
            logger.info(
                f"Sent deep agent snapshot to {client_id}: "
                f"cycle {snapshot.get('cycle_count', 0)}, "
                f"{snapshot.get('trades_executed', 0)} trades"
            )
        except Exception as e:
            logger.warning(f"Could not send deep agent snapshot to {client_id}: {e}")

    async def _send_entity_snapshot(self, client_id: str) -> None:
        """
        Send extraction pipeline snapshot to a specific client.

        Restores Reddit posts and extraction signals after a page refresh.
        """
        if not self._reddit_agent:
            return
        if client_id not in self._clients:
            return

        try:
            # Get snapshot from reddit agent (posts + extractions)
            entity_snapshot = self._reddit_agent.get_entity_snapshot()

            # Build stats (camelCase for frontend)
            stats = entity_snapshot.get("stats", {})

            # Build combined snapshot
            snapshot_data = {
                "reddit_posts": entity_snapshot.get("reddit_posts", []),
                "extractions": entity_snapshot.get("extractions", []),
                # Legacy field: frontend may still read "entities"
                "entities": entity_snapshot.get("extractions", []),
                "stats": stats,
                "is_active": entity_snapshot.get("is_active", False),
            }

            await self._send_to_client(client_id, {
                "type": "entity_snapshot",
                "data": snapshot_data
            })
            logger.info(
                f"Sent extraction snapshot to {client_id}: "
                f"{len(snapshot_data['reddit_posts'])} posts, "
                f"{len(snapshot_data['extractions'])} extractions, "
                f"stats={stats}"
            )
        except Exception as e:
            logger.warning(f"Could not send entity snapshot to {client_id}: {e}")

    async def _send_event_configs_snapshot(self, client_id: str) -> None:
        """Send active event configs to a new client."""
        if not self._reddit_agent or client_id not in self._clients:
            return

        try:
            configs = self._reddit_agent.get_event_configs()
            if not configs:
                return

            await self._send_to_client(client_id, {
                "type": "event_configs_snapshot",
                "data": {"configs": configs}
            })
            logger.debug(
                f"Sent event configs snapshot to {client_id}: {len(configs)} configs"
            )
        except Exception as e:
            logger.warning(f"Could not send event configs snapshot to {client_id}: {e}")

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