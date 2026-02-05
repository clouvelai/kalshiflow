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
        self._event_bus = event_bus
        self._state_machine = state_machine
        self._state_container: Optional['V3StateContainer'] = None
        self._trading_service: Optional['TradingDecisionService'] = None
        self._trade_flow_service = None
        self._market_price_syncer = None
        self._tracked_markets_state: Optional['TrackedMarketsState'] = None
        self._upcoming_markets_syncer: Optional['UpcomingMarketsSyncer'] = None
        self._single_arb_coordinator = None
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
        self._ping_interval = 30.0

        # Message coalescing
        self._pending_messages: Dict[str, Dict[str, Any]] = {}
        self._coalesce_task: Optional[asyncio.Task] = None
        self._coalesce_interval = 0.1

        # State transition history buffer (last 20 transitions)
        self._state_transition_history: deque = deque(maxlen=20)

        # Activity feed history buffer
        self._activity_feed_history: deque = deque(maxlen=100)

        logger.info("TRADER V3 WebSocket Manager initialized")

    def set_trading_service(self, trading_service: 'TradingDecisionService') -> None:
        self._trading_service = trading_service
        logger.info("TradingDecisionService set on WebSocket manager")

    def set_state_container(self, state_container: 'V3StateContainer') -> None:
        self._state_container = state_container
        logger.info("V3StateContainer set on WebSocket manager")

    def set_market_price_syncer(self, market_price_syncer) -> None:
        self._market_price_syncer = market_price_syncer
        logger.debug("MarketPriceSyncer set on WebSocket manager")

    def set_tracked_markets_state(self, tracked_markets_state: 'TrackedMarketsState') -> None:
        self._tracked_markets_state = tracked_markets_state
        logger.info("TrackedMarketsState set on WebSocket manager")

    def set_trade_flow_service(self, service) -> None:
        self._trade_flow_service = service
        logger.info("TradeFlowService set on WebSocket manager")

    def set_upcoming_markets_syncer(self, syncer: 'UpcomingMarketsSyncer') -> None:
        self._upcoming_markets_syncer = syncer
        logger.info("UpcomingMarketsSyncer set on WebSocket manager")

    def set_single_arb_coordinator(self, coordinator) -> None:
        self._single_arb_coordinator = coordinator
        logger.info("SingleArbCoordinator set on WebSocket manager")

    def _add_to_activity_feed_history(self, message_type: str, data: dict) -> None:
        """Track events for Activity Feed history replay."""
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

        if self._event_bus:
            self._event_bus.subscribe(EventType.SYSTEM_ACTIVITY, self._handle_system_activity)
            self._event_bus.subscribe(EventType.TRADER_STATUS, self._handle_trader_status)
            await self._event_bus.subscribe_to_trade_flow_market_update(self._handle_trade_flow_market_update)
            await self._event_bus.subscribe_to_trade_flow_trade_arrived(self._handle_trade_flow_trade_arrived)
            logger.info("Subscribed to event bus for real-time updates")

        self._ping_task = asyncio.create_task(self._ping_clients())

        logger.info("TRADER V3 WebSocket Manager started")

    async def stop(self) -> None:
        """Stop the WebSocket manager and disconnect all clients."""
        if not self._running:
            return

        logger.info("Stopping TRADER V3 WebSocket Manager...")
        self._running = False

        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

        if self._coalesce_task and not self._coalesce_task.done():
            self._coalesce_task.cancel()
            try:
                await self._coalesce_task
            except asyncio.CancelledError:
                pass

        disconnect_tasks = []
        for client in list(self._clients.values()):
            disconnect_tasks.append(self._disconnect_client(client.client_id))

        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        logger.info(f"TRADER V3 WebSocket Manager stopped. Messages sent: {self._messages_sent}")

    async def handle_websocket(self, websocket: WebSocket) -> None:
        """Handle new WebSocket connection."""
        try:
            await websocket.accept()

            self._client_counter += 1
            client_id = f"client_{self._client_counter}_{int(time.time())}"

            client = WebSocketClient(
                websocket=websocket,
                client_id=client_id,
                connected_at=time.time()
            )

            self._clients[client_id] = client
            self._connection_count += 1
            self._active_connections += 1

            logger.info(f"WebSocket client connected: {client_id}")

            await self._send_to_client(client_id, {
                "type": "connection",
                "data": {
                    "client_id": client_id,
                    "timestamp": time.strftime("%H:%M:%S"),
                    "message": "Connected to TRADER V3 console"
                }
            })

            await asyncio.sleep(0.1)

            # Replay historical state transitions
            if self._state_transition_history:
                history_batch = {
                    "type": "history_replay",
                    "data": {
                        "transitions": [msg["data"] for msg in self._state_transition_history],
                        "count": len(self._state_transition_history)
                    }
                }
                if client_id in self._clients:
                    await self._send_to_client(client_id, history_batch)
                    await asyncio.sleep(0.1)

            # Send trading state snapshot
            if self._state_container and client_id in self._clients:
                try:
                    trading_summary = self._state_container.get_trading_summary()
                    if trading_summary.get("has_state"):
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
                                "market_prices": trading_summary.get("market_prices"),
                                "position_listener": None,
                                "market_ticker_listener": None,
                                "market_price_syncer": market_price_syncer_health,
                                "order_group": trading_summary.get("order_group"),
                                "changes": trading_summary.get("changes"),
                            }
                        }
                        await self._send_to_client(client_id, trading_state_msg)
                        logger.info(f"Sent immediate trading state to client {client_id}")
                except Exception as e:
                    logger.warning(f"Could not send trading state to client {client_id}: {e}")

            # Send tracked markets snapshot
            if self._tracked_markets_state and client_id in self._clients:
                await self._send_tracked_markets_snapshot(client_id)

            # Send trade flow market states snapshot
            if self._trade_flow_service and client_id in self._clients:
                await self._send_trade_flow_states_snapshot(client_id)

            # Send upcoming markets snapshot
            if self._upcoming_markets_syncer and client_id in self._clients:
                await self._send_upcoming_markets_snapshot(client_id)

            # Send event research snapshot
            if self._state_container and client_id in self._clients:
                await self._send_event_research_snapshot(client_id)

            # Send activity feed history
            if self._activity_feed_history and client_id in self._clients:
                await self._send_activity_feed_history(client_id)

            # Send captain paused state to new client
            if self._single_arb_coordinator and client_id in self._clients:
                is_paused = self._single_arb_coordinator.is_captain_paused()
                await self._send_to_client(client_id, {
                    "type": "captain_paused",
                    "data": {"paused": is_paused}
                })

            # Handle incoming messages
            async for message in websocket.iter_text():
                await self._handle_client_message(client_id, message)

        except WebSocketDisconnect:
            logger.info(f"WebSocket client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket client {client_id}: {e}")
        finally:
            await self._disconnect_client(client_id)

    async def broadcast_message(self, message_type: str, data: Dict[str, Any]) -> None:
        """Queue message for coalesced broadcast."""
        critical_types = (
            "state_transition", "connection", "system_activity", "history_replay",
            "trade_flow_market_state", "trade_flow_trade_arrived",
            "agent_message",
        )
        if message_type in critical_types:
            await self._broadcast_immediate(message_type, data)
            return

        self._pending_messages[message_type] = data

        if not self._coalesce_task or self._coalesce_task.done():
            self._coalesce_task = asyncio.create_task(self._flush_pending())

    async def _flush_pending(self) -> None:
        """Flush pending messages after coalesce interval."""
        await asyncio.sleep(self._coalesce_interval)
        messages = self._pending_messages.copy()
        self._pending_messages.clear()
        for msg_type, data in messages.items():
            await self._broadcast_immediate(msg_type, data)

    async def _broadcast_immediate(self, message_type: str, data: Dict[str, Any]) -> None:
        """Broadcast message immediately to all connected clients."""
        if not self._clients:
            return

        message = {
            "type": message_type,
            "data": data,
            "timestamp": time.time()
        }

        send_tasks = []
        for client_id in list(self._clients.keys()):
            send_tasks.append(self._send_to_client(client_id, message))

        if send_tasks:
            await asyncio.gather(*send_tasks, return_exceptions=True)

    async def broadcast_console_message(self, level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Broadcast console-style message to all clients."""
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

        current_state = "ready"
        if event.activity_type == "state_transition" and event.metadata:
            current_state = event.metadata.get("to_state")
            if current_state and hasattr(current_state, 'lower'):
                current_state = current_state.lower()
            elif current_state:
                current_state = str(current_state).lower()
        elif self._state_machine:
            try:
                if hasattr(self._state_machine, 'current_state'):
                    if hasattr(self._state_machine.current_state, 'value'):
                        current_state = self._state_machine.current_state.value.lower()
                    else:
                        current_state = str(self._state_machine.current_state).lower()
            except Exception:
                current_state = "ready"

        if not current_state or current_state == "none" or current_state == "unknown":
            current_state = "ready"

        activity_data = {
            "timestamp": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
            "activity_type": event.activity_type,
            "message": event.message,
            "metadata": event.metadata,
            "state": current_state
        }

        if event.activity_type == "state_transition" and event.metadata:
            if "from_state" in event.metadata:
                activity_data["from_state"] = event.metadata["from_state"]
            if "to_state" in event.metadata:
                activity_data["to_state"] = event.metadata["to_state"]

        activity_message = {
            "type": "system_activity",
            "data": activity_data
        }

        if event.activity_type == "state_transition":
            self._state_transition_history.append(activity_message)

        if event.activity_type != "state_transition":
            self._add_to_activity_feed_history("system_activity", activity_message["data"])

        await self.broadcast_message("system_activity", activity_message["data"])

    async def _handle_trader_status(self, event: TraderStatusEvent) -> None:
        """Handle trader status events from event bus."""
        await self.broadcast_message("trader_status", {
            "timestamp": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
            "state": event.state,
            "health": event.health,
            "metrics": event.metrics
        })

    async def _handle_trade_flow_market_update(self, event: TradeFlowMarketUpdateEvent) -> None:
        """Handle trade flow market state update events."""
        await self.broadcast_message("trade_flow_market_state", {
            "ticker": event.market_ticker,
            "market_ticker": event.market_ticker,
            **event.state,
            "timestamp": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
        })

    async def _handle_trade_flow_trade_arrived(self, event: TradeFlowTradeArrivedEvent) -> None:
        """Handle trade flow trade arrived events."""
        await self.broadcast_message("trade_flow_trade_arrived", {
            "ticker": event.market_ticker,
            "market_ticker": event.market_ticker,
            "event_ticker": event.event_ticker,
            "side": event.side,
            "count": event.count,
            "yes_price": event.price_cents,
            "timestamp": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
        })

    async def _handle_client_message(self, client_id: str, message: str) -> None:
        """Handle message from WebSocket client."""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "ping":
                await self._send_to_client(client_id, {
                    "type": "pong",
                    "timestamp": time.time()
                })
                if client_id in self._clients:
                    self._clients[client_id].last_ping = time.time()

            elif message_type == "subscribe":
                subscriptions = set(data.get("subscriptions", []))
                if client_id in self._clients:
                    self._clients[client_id].subscriptions = subscriptions
                    logger.info(f"Client {client_id} subscribed to: {subscriptions}")

            elif message_type == "captain_pause":
                if self._single_arb_coordinator:
                    self._single_arb_coordinator.pause_captain()
                    await self.broadcast_message("captain_paused", {"paused": True})
                    logger.info(f"Captain paused by client {client_id}")

            elif message_type == "captain_resume":
                if self._single_arb_coordinator:
                    self._single_arb_coordinator.resume_captain()
                    await self.broadcast_message("captain_paused", {"paused": False})
                    logger.info(f"Captain resumed by client {client_id}")

            else:
                logger.debug(f"Unknown message type from client {client_id}: {message_type}")

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client {client_id}: {message}")
        except Exception as e:
            logger.error(f"Error handling message from client {client_id}: {e}")

    async def _send_to_client(self, client_id: str, message: Dict[str, Any]) -> None:
        """Send message to specific client."""
        if client_id not in self._clients:
            return

        client = self._clients.get(client_id)
        if not client:
            return

        try:
            await client.websocket.send_text(json.dumps(message, cls=DateTimeEncoder))
            self._messages_sent += 1
        except (RuntimeError, ConnectionError) as e:
            logger.debug(f"Connection error for client {client_id}: {e}")
            await self._disconnect_client(client_id)
        except Exception as e:
            logger.warning(f"Unexpected error sending to client {client_id}: {type(e).__name__}: {e!r}")
            await self._disconnect_client(client_id)

    async def _disconnect_client(self, client_id: str) -> None:
        """Disconnect and remove client safely."""
        client = self._clients.pop(client_id, None)
        if not client:
            return
        self._active_connections = max(0, self._active_connections - 1)
        try:
            await client.websocket.close()
        except Exception:
            pass
        logger.info(f"WebSocket client removed: {client_id} (active: {self._active_connections})")

    async def _ping_clients(self) -> None:
        """Periodic ping to maintain client connections."""
        while self._running:
            try:
                await asyncio.sleep(self._ping_interval)
                if not self._clients:
                    continue
                ping_tasks = []
                for client_id in list(self._clients.keys()):
                    ping_tasks.append(self._send_to_client(client_id, {
                        "type": "ping",
                        "timestamp": time.time()
                    }))
                if ping_tasks:
                    await asyncio.gather(*ping_tasks, return_exceptions=True)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping task: {e}")

    # ========== Lifecycle Discovery Mode Methods ==========

    async def broadcast_tracked_markets(self) -> None:
        """Broadcast tracked markets state to all connected clients."""
        if not self._tracked_markets_state:
            return
        snapshot = self._tracked_markets_state.get_snapshot()
        if self._state_container:
            for market in snapshot.get("markets", []):
                ticker = market.get("ticker")
                if ticker:
                    trading = self._state_container.get_trading_attachment_for_market(ticker)
                    market["trading"] = trading
        await self.broadcast_message("tracked_markets", snapshot)

    async def broadcast_lifecycle_event(
        self, event_type: str, market_ticker: str, action: str,
        reason: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Broadcast a lifecycle event to all connected clients."""
        event_data = {
            "event_type": event_type,
            "market_ticker": market_ticker,
            "action": action,
            "reason": reason,
            "metadata": metadata or {},
            "timestamp": time.strftime("%H:%M:%S"),
        }
        self._add_to_activity_feed_history("lifecycle_event", event_data)
        await self.broadcast_message("lifecycle_event", event_data)

    async def broadcast_market_info_update(
        self, ticker: str, price: int, volume: int,
        open_interest: Optional[int] = None,
        yes_bid: Optional[int] = None, yes_ask: Optional[int] = None
    ) -> None:
        """Broadcast market info update for a single tracked market."""
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

    # ========== Snapshot Methods ==========

    async def _send_snapshot(
        self, client_id: str, message_type: str, data_source: Any,
        get_data: Any, log_name: Optional[str] = None
    ) -> None:
        """Generic snapshot sender for client initialization."""
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
            client_id, "tracked_markets", self._tracked_markets_state,
            lambda: self._build_tracked_markets_data(), log_name="tracked_markets"
        )

    def _build_tracked_markets_data(self) -> dict:
        """Build tracked markets snapshot data with trading attachments."""
        snapshot = self._tracked_markets_state.get_snapshot()
        if self._state_container:
            for market in snapshot.get("markets", []):
                ticker = market.get("ticker")
                if ticker:
                    trading = self._state_container.get_trading_attachment_for_market(ticker)
                    market["trading"] = trading
        return snapshot

    async def _send_trade_flow_states_snapshot(self, client_id: str) -> None:
        """Send trade flow market states snapshot to a specific client."""
        await self._send_snapshot(
            client_id, "trade_flow_states_snapshot", self._trade_flow_service,
            lambda: self._build_trade_flow_states_data(), log_name="trade flow states"
        )

    def _build_trade_flow_states_data(self) -> dict:
        """Build trade flow market states snapshot data."""
        if self._trade_flow_service:
            market_states = self._trade_flow_service.get_market_states(limit=100)
        else:
            market_states = []
        return {
            "markets": market_states,
            "count": len(market_states),
            "timestamp": time.strftime("%H:%M:%S"),
        }

    async def _send_upcoming_markets_snapshot(self, client_id: str) -> None:
        """Send upcoming markets snapshot to a specific client."""
        await self._send_snapshot(
            client_id, "upcoming_markets", self._upcoming_markets_syncer,
            lambda: self._upcoming_markets_syncer.get_snapshot_message()["data"],
            log_name="upcoming_markets"
        )

    async def _send_event_research_snapshot(self, client_id: str) -> None:
        """Send cached event research results to a specific client."""
        if not self._state_container:
            return
        event_research = self._state_container.get_event_research_results()
        if not event_research:
            return
        await self._send_snapshot(
            client_id, "trading_state", event_research,
            lambda: {"event_research": event_research}, log_name="event_research"
        )

    async def _send_activity_feed_history(self, client_id: str) -> None:
        """Send activity feed history to a specific client."""
        await self._send_snapshot(
            client_id, "activity_feed_history", self._activity_feed_history,
            lambda: self._build_activity_feed_data(), log_name="activity_feed_history"
        )

    def _build_activity_feed_data(self) -> dict:
        """Build activity feed history data."""
        history_list = list(self._activity_feed_history)
        return {"events": history_list, "count": len(history_list)}

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
