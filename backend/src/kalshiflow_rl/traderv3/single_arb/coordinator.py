"""
SingleArbCoordinator - Wires EventArbIndex + EventArbMonitor + ArbCaptain.

Startup sequence:
1. Create EventArbIndex
2. For each hardcoded event ticker: load_event() via REST
3. Subscribe all market tickers to orderbook WS
4. Create EventArbMonitor (subscribes to EventBus)
5. Start REST poller fallback
6. Create ArbCaptain with memory store
7. Start Captain cycle loop
8. Wire WebSocket broadcasting
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

from .index import EventArbIndex
from .monitor import EventArbMonitor
from .tools import set_dependencies as set_tool_dependencies

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.coordinator")

# Default memory data directory
DEFAULT_MEMORY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "memory", "data"
)


class SingleArbCoordinator:
    """
    Coordinator for the single-event arb system.

    Wires all components together and manages lifecycle.
    """

    def __init__(
        self,
        config,
        event_bus,
        websocket_manager,
        orderbook_integration,
        trading_client=None,
        health_monitor=None,
    ):
        self._config = config
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager
        self._orderbook_integration = orderbook_integration
        self._trading_client = trading_client
        self._health_monitor = health_monitor

        self._index: Optional[EventArbIndex] = None
        self._monitor: Optional[EventArbMonitor] = None
        self._captain = None
        self._memory_store = None

        self._running = False
        self._started_at: Optional[float] = None

    async def start(self) -> None:
        """Start the single-arb system."""
        if self._running:
            return

        logger.info("Starting single-event arb system...")

        # 1. Create index
        fee_per_contract = getattr(self._config, "single_arb_fee_per_contract", 7)
        min_edge = getattr(self._config, "single_arb_min_edge_cents", 3.0)
        self._index = EventArbIndex(
            fee_per_contract_cents=fee_per_contract,
            min_edge_cents=min_edge,
        )

        # 2. Load events
        event_tickers = getattr(self._config, "single_arb_event_tickers", [])
        if not event_tickers:
            logger.warning("No single_arb_event_tickers configured")
            return

        # Need a trading client for REST calls
        client = self._get_trading_client()
        if not client:
            logger.error("No trading client available for single-arb REST calls")
            return

        loaded_events = 0
        for event_ticker in event_tickers:
            state = await self._index.load_event(event_ticker, client)
            if state:
                loaded_events += 1
            else:
                logger.warning(f"Failed to load event: {event_ticker}")

        if loaded_events == 0:
            logger.error("No events loaded - single-arb system cannot start")
            return

        # 3. Subscribe all market tickers to orderbook WS
        market_tickers = self._index.market_tickers
        if market_tickers and self._orderbook_integration:
            for ticker in market_tickers:
                try:
                    await self._orderbook_integration.subscribe_market(ticker)
                except Exception as e:
                    logger.warning(f"Failed to subscribe {ticker} to orderbook: {e}")

            logger.info(f"Subscribed {len(market_tickers)} markets to orderbook WS")

        # 4. Setup memory store
        self._setup_memory()

        # 5. Set tool dependencies
        set_tool_dependencies(
            index=self._index,
            trading_client=client,
            memory_store=self._memory_store,
            config=self._config,
        )

        # 6. Create monitor
        self._monitor = EventArbMonitor(
            index=self._index,
            event_bus=self._event_bus,
            trading_client=client,
            config=self._config,
            broadcast_callback=self._broadcast,
            opportunity_callback=self._on_opportunity,
        )
        await self._monitor.start()

        # 7. Create and start Captain (if enabled)
        captain_enabled = getattr(self._config, "single_arb_captain_enabled", True)
        if captain_enabled:
            try:
                from .captain import ArbCaptain
                captain_interval = getattr(self._config, "single_arb_captain_interval", 60.0)
                self._captain = ArbCaptain(
                    cycle_interval=captain_interval,
                    event_callback=self._emit_agent_event,
                )
                await self._captain.start()
                logger.info("ArbCaptain started")
            except Exception as e:
                logger.error(f"Failed to start ArbCaptain: {e}")

        # 8. Broadcast initial snapshot
        await self._broadcast_snapshot()

        # Register with health monitor
        if self._health_monitor:
            self._health_monitor.register_component(
                "single_arb", self, critical=False
            )

        self._running = True
        self._started_at = time.time()

        logger.info(
            f"Single-event arb system started: "
            f"{loaded_events} events, {len(market_tickers)} markets"
        )

    async def stop(self) -> None:
        """Stop the single-arb system."""
        self._running = False

        if self._captain:
            await self._captain.stop()
        if self._monitor:
            await self._monitor.stop()

        logger.info("Single-event arb system stopped")

    def _get_trading_client(self):
        """Get the underlying trading client for REST calls."""
        if self._trading_client:
            # V3TradingClientIntegration wraps the actual client
            if hasattr(self._trading_client, "_client"):
                return self._trading_client._client
            return self._trading_client
        return None

    def _setup_memory(self) -> None:
        """Setup dual memory store (file + vector)."""
        from .memory import FileMemoryStore, VectorMemoryService, DualMemoryStore
        from kalshiflow_rl.data.database import rl_db

        file_store = FileMemoryStore(data_dir=DEFAULT_MEMORY_DIR)

        # Try to create vector store (best-effort)
        vector_store = None
        try:
            vector_store = VectorMemoryService(db=rl_db)
            logger.info("VectorMemoryService initialized (pgvector)")
        except Exception as e:
            logger.warning(f"VectorMemoryService unavailable: {e}")

        self._memory_store = DualMemoryStore(
            file_store=file_store,
            vector_store=vector_store,
        )
        logger.info(f"DualMemoryStore initialized (vector={'yes' if vector_store else 'no'})")

    async def _broadcast(self, message: Dict) -> None:
        """Broadcast message to all frontend WebSocket clients."""
        if self._websocket_manager:
            try:
                await self._websocket_manager.broadcast(message)
            except Exception as e:
                logger.debug(f"Broadcast error: {e}")

    async def _broadcast_snapshot(self) -> None:
        """Broadcast full snapshot to frontend."""
        if not self._index:
            return

        snapshot = self._index.get_snapshot()
        await self._broadcast({
            "type": "event_arb_snapshot",
            "data": snapshot,
        })

    async def _on_opportunity(self, opportunity) -> None:
        """Handle detected arb opportunity."""
        # Broadcast to frontend
        await self._broadcast({
            "type": "arb_opportunity",
            "data": opportunity.to_dict(),
        })

    async def _emit_agent_event(self, event_data: Dict) -> None:
        """Forward agent events to frontend WebSocket."""
        await self._broadcast(event_data)

    # Health check interface
    @property
    def is_healthy(self) -> bool:
        return self._running and self._index is not None

    def get_health_details(self) -> Dict:
        """Health details for the health monitor."""
        details = {
            "running": self._running,
            "started_at": self._started_at,
        }

        if self._index:
            details["events"] = len(self._index.events)
            details["markets"] = len(self._index.market_tickers)

        if self._monitor:
            details["monitor"] = self._monitor.get_stats()

        if self._captain:
            details["captain"] = self._captain.get_stats()

        return details
