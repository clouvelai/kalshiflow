"""
SingleArbCoordinator - Wires EventArbIndex + EventArbMonitor + ArbCaptain.

Startup sequence:
1. Create EventArbIndex
2. For each hardcoded event ticker: load_event() via REST
3. Subscribe all market tickers to orderbook WS
4. Setup memory store (file + vector) + seed AGENTS.md
5. Create session order group
6. Set tool dependencies
7. Create EventArbMonitor (subscribes to EventBus)
8. Create ArbCaptain with FilesystemBackend for persistent /memories/
9. Broadcast initial snapshot
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

        self._order_group_id: Optional[str] = None
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

        # 5. Create session order group for the captain
        self._order_group_id = None
        try:
            resp = await client.create_order_group(contracts_limit=10000)
            self._order_group_id = resp.get("order_group_id")
            if self._order_group_id:
                logger.info(f"[SINGLE_ARB] Order group created: {self._order_group_id[:8]}...")
            else:
                logger.warning("[SINGLE_ARB] Order group response missing order_group_id")
        except Exception as e:
            logger.warning(f"[SINGLE_ARB] Order group creation failed: {e}")

        # 6. Set tool dependencies
        order_ttl = getattr(self._config, "single_arb_order_ttl", 60)
        set_tool_dependencies(
            index=self._index,
            trading_client=client,
            memory_store=self._memory_store,
            config=self._config,
            order_group_id=self._order_group_id,
            order_ttl=order_ttl,
        )

        # 7. Create monitor
        self._monitor = EventArbMonitor(
            index=self._index,
            event_bus=self._event_bus,
            trading_client=client,
            config=self._config,
            broadcast_callback=self._broadcast,
            opportunity_callback=self._on_opportunity,
        )
        await self._monitor.start()

        # 8. Create and start Captain (if enabled)
        captain_enabled = getattr(self._config, "single_arb_captain_enabled", True)
        if captain_enabled:
            try:
                from .captain import ArbCaptain
                captain_interval = getattr(self._config, "single_arb_captain_interval", 60.0)
                self._captain = ArbCaptain(
                    cycle_interval=captain_interval,
                    event_callback=self._emit_agent_event,
                    memory_data_dir=DEFAULT_MEMORY_DIR,
                )
                await self._captain.start()
                logger.info("ArbCaptain started")
            except Exception as e:
                logger.error(f"Failed to start ArbCaptain: {e}")

        # 9. Broadcast initial snapshot
        await self._broadcast_snapshot()

        # Register with health monitor
        if self._health_monitor:
            self._health_monitor.register_component(
                "single_arb", self, critical=False
            )

        self._running = True
        self._started_at = time.time()

        logger.info(
            f"[SINGLE_ARB:STARTUP] Single-event arb system started: "
            f"events={loaded_events} markets={len(market_tickers)} captain={captain_enabled}"
        )

    async def stop(self) -> None:
        """Stop the single-arb system."""
        self._running = False

        if self._captain:
            await self._captain.stop()
        if self._monitor:
            await self._monitor.stop()

        # Reset order group (cancels all resting orders in group)
        if getattr(self, "_order_group_id", None):
            try:
                client = self._get_trading_client()
                if client:
                    await client.reset_order_group(self._order_group_id)
                    logger.info(f"[SINGLE_ARB] Order group reset: {self._order_group_id[:8]}...")
            except Exception as e:
                logger.debug(f"Order group reset failed: {e}")

        logger.info("[SINGLE_ARB:SHUTDOWN] Single-event arb system stopped")

    def _get_trading_client(self):
        """Get the underlying trading client for REST calls."""
        if self._trading_client:
            # V3TradingClientIntegration wraps the actual client
            if hasattr(self._trading_client, "_client"):
                return self._trading_client._client
            return self._trading_client
        return None

    def _setup_memory(self) -> None:
        """Setup dual memory store (file + vector) and seed AGENTS.md."""
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

        # Seed AGENTS.md if it doesn't exist (loaded into Captain system prompt each cycle)
        agents_md_path = os.path.join(DEFAULT_MEMORY_DIR, "AGENTS.md")
        if not os.path.exists(agents_md_path):
            with open(agents_md_path, "w") as f:
                f.write(
                    "# Trading Learnings\n"
                    "\n"
                    "## Strategy Notes\n"
                    "- Primary: single-event arb (probability sum violations)\n"
                    "- Long arb: sum of YES asks < 100 - total_fees -> buy all YES\n"
                    "- Short arb: sum of YES bids > 100 + total_fees -> buy all NO\n"
                    "- Fee ~7c per contract per leg\n"
                    "- Start with 5 contracts per leg. Scale what proves profitable.\n"
                    "\n"
                    "## Patterns Observed\n"
                    "(Record patterns here as you discover them)\n"
                    "\n"
                    "## Mistakes & Lessons\n"
                    "(Record mistakes and what you learned)\n"
                    "\n"
                    "## Market-Specific Notes\n"
                    "(Record notes about specific events/markets)\n"
                )
            logger.info("Created seed AGENTS.md")

        # Seed BOTS.md if it doesn't exist (ChevalDeTroie bot registry)
        bots_md_path = os.path.join(DEFAULT_MEMORY_DIR, "BOTS.md")
        if not os.path.exists(bots_md_path):
            with open(bots_md_path, "w") as f:
                f.write(
                    "# Bot Registry\n"
                    "\n"
                    "## Classification Schema\n"
                    "- **MM_BOT**: Market maker (maintains quotes on both sides)\n"
                    "- **ARB_BOT**: Arbitrage bot (responds to edge conditions)\n"
                    "- **MOMENTUM_BOT**: Trades with recent price direction\n"
                    "- **WHALE**: Large position accumulator\n"
                    "- **UNKNOWN_AUTO**: Automated but pattern unclear\n"
                    "\n"
                    "## Identified Entities\n"
                    "\n"
                    "No bots identified yet.\n"
                    "\n"
                    "## Anomalies Log\n"
                    "\n"
                    "(Record unusual market behavior here)\n"
                )
            logger.info("Created seed BOTS.md")

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

    # Captain pause/resume interface
    def pause_captain(self) -> None:
        """Pause the Captain after current cycle completes."""
        if self._captain:
            self._captain.pause()

    def resume_captain(self) -> None:
        """Resume Captain cycles."""
        if self._captain:
            self._captain.resume()

    def is_captain_paused(self) -> bool:
        """Check if Captain is paused."""
        return self._captain.is_paused if self._captain else False

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
