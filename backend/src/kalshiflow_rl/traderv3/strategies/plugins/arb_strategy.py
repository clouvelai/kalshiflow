"""
Arbitrage Strategy - Trading layer for cross-venue arbitrage.

Owns SpreadMonitor (hot path for sub-100ms execution).
Orchestrator is gated behind config.arb_orchestrator_enabled (default True).
"""

import logging
from typing import Any, Dict, Optional

from ...services.spread_monitor import SpreadMonitor
from ...services.pair_registry import PairRegistry
from ...core.event_bus import EventBus
from ...core.websocket_manager import V3WebSocketManager
from ...config.environment import V3Config

logger = logging.getLogger("kalshiflow_rl.traderv3.strategies.plugins.arb_strategy")


class ArbStrategy:
    """
    Arbitrage strategy: SpreadMonitor always runs, orchestrator gated by config.

    Components:
    - SpreadMonitor: Fast hot path for automated spread-based trading (always on)
    - ArbOrchestrator v2: Captain + EventAnalyst + MemoryCurator (enabled by default)
    """

    def __init__(
        self,
        event_bus: EventBus,
        pair_registry: PairRegistry,
        config: V3Config,
        trading_client=None,
        websocket_manager: Optional[V3WebSocketManager] = None,
        state_container=None,
        supabase_client=None,
        orderbook_integration=None,
        poly_client=None,
    ):
        self._event_bus = event_bus
        self._pair_registry = pair_registry
        self._config = config
        self._trading_client = trading_client
        self._websocket_manager = websocket_manager
        self._state_container = state_container
        self._supabase = supabase_client
        self._orderbook_integration = orderbook_integration
        self._poly_client = poly_client

        self._spread_monitor: Optional[SpreadMonitor] = None
        self._orchestrator = None
        self._running = False

    async def start(self) -> None:
        """Start spread monitor. Orchestrator must be started separately via start_orchestrator()."""
        if self._running:
            return

        self._running = True

        # SpreadMonitor always starts (hot path)
        self._spread_monitor = SpreadMonitor(
            event_bus=self._event_bus,
            pair_registry=self._pair_registry,
            config=self._config,
            trading_client=self._trading_client,
            websocket_manager=self._websocket_manager,
            supabase_client=self._supabase,
            orderbook_integration=self._orderbook_integration,
        )
        await self._spread_monitor.start()
        logger.info("Spread monitor started")

    async def start_orchestrator(self) -> None:
        """Start the LLM orchestrator (call after pair index + event codex are populated)."""
        if not self._config.arb_orchestrator_enabled:
            logger.info("Arb orchestrator disabled by config")
            return
        await self._start_orchestrator()

    def set_event_codex(self, event_codex) -> None:
        """Wire EventCodex into the orchestrator's data tools (called after late init)."""
        try:
            from ...deep_agent.tools.data_tools import set_shared_data
            set_shared_data(event_codex=event_codex)
            logger.info("EventCodex wired into orchestrator data tools")
        except Exception as e:
            logger.warning(f"Failed to wire EventCodex: {e}")

    async def _start_orchestrator(self) -> None:
        """Start the LLM orchestrator (only when explicitly enabled)."""
        try:
            from ...deep_agent.orchestrator import ArbOrchestrator, ArbOrchestratorConfig
            from ...deep_agent.tools.kalshi_tools import set_trading_client
            from ...deep_agent.tools.poly_tools import set_poly_client
            from ...deep_agent.tools.db_tools import set_dependencies
            from ...deep_agent.tools.data_tools import set_shared_data
            from ...deep_agent.tools.trade_tools import set_trade_deps
            from ...deep_agent.tools.memory_tools import set_memory_store
            from ...deep_agent.memory.file_store import FileMemoryStore
            from ...deep_agent.memory.dual_store import DualMemoryStore
            from ...deep_agent.vector_memory import VectorMemoryService

            # 1. Wire Kalshi trading client
            if self._trading_client:
                set_trading_client(self._trading_client)

            # 2. Wire Polymarket client
            if self._poly_client:
                set_poly_client(self._poly_client)

            # 3. Wire DB tools (legacy tools still used by orchestrator)
            set_dependencies(
                supabase=self._supabase,
                pair_registry=self._pair_registry,
                spread_monitor=self._spread_monitor,
                state_container=self._state_container,
            )

            # 4. Create memory stores
            file_store = FileMemoryStore()
            file_store.clear_validations()  # Fresh start — force re-validation of all events

            # Initialize embedding model + vector memory
            embedding_model = None
            try:
                from langchain_openai import OpenAIEmbeddings
                embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
            except Exception as e:
                logger.warning(f"Embedding model not available: {e}")

            vector_service = VectorMemoryService(
                supabase_client=self._supabase,
                embedding_model=embedding_model,
                agent_id="arb_agent",
            )

            dual_store = DualMemoryStore(
                file_store=file_store,
                vector_service=vector_service,
            )

            # 5. Wire memory tools
            set_memory_store(dual_store)

            # 6. Wire data snapshot tools
            set_shared_data(
                pair_registry=self._pair_registry,
                spread_monitor=self._spread_monitor,
                file_store=file_store,
                # event_codex wired later via set_event_codex()
            )

            # 7. Wire trade tools
            broadcast_fn = self._broadcast_agent_event if self._websocket_manager else None
            set_trade_deps(
                trading_client=self._trading_client,
                pair_registry=self._pair_registry,
                spread_monitor=self._spread_monitor,
                supabase=self._supabase,
                file_store=file_store,
                event_callback=broadcast_fn,
            )

            # 8. Create and start orchestrator
            orchestrator_config = ArbOrchestratorConfig(
                scan_interval_seconds=self._config.arb_scan_interval_seconds,
            )
            self._orchestrator = ArbOrchestrator(config=orchestrator_config)

            if self._websocket_manager:
                self._orchestrator.set_event_callback(self._broadcast_agent_event)

            await self._orchestrator.start()
            logger.info("Arb orchestrator v2 started")

        except Exception as e:
            logger.error(f"Orchestrator failed to start (spread monitor still running): {e}")
            self._orchestrator = None

    async def _broadcast_agent_event(self, event_data: Dict[str, Any]) -> None:
        """Broadcast agent events to frontend via WebSocket."""
        if self._websocket_manager:
            await self._websocket_manager.broadcast_message("agent_message", event_data)

    async def stop(self) -> None:
        """Stop all components."""
        if not self._running:
            return

        self._running = False

        if self._orchestrator:
            try:
                await self._orchestrator.stop()
            except Exception as e:
                logger.error(f"Error stopping orchestrator: {e}")

        if self._spread_monitor:
            try:
                await self._spread_monitor.stop()
            except Exception as e:
                logger.error(f"Error stopping spread monitor: {e}")

        logger.info("ArbStrategy stopped")

    async def handle_user_message(self, message: str) -> str:
        """Forward user message to orchestrator."""
        if self._orchestrator:
            return await self._orchestrator.handle_user_message(message)
        return "Orchestrator not running (set ARB_ORCHESTRATOR_ENABLED=true to enable)"

    def get_status(self) -> Dict[str, Any]:
        """Get combined status."""
        return {
            "running": self._running,
            "orchestrator_enabled": self._config.arb_orchestrator_enabled,
            "spread_monitor": self._spread_monitor.get_status() if self._spread_monitor else None,
            "orchestrator": self._orchestrator.get_status() if self._orchestrator else None,
        }

    def is_healthy(self) -> bool:
        """Healthy if at least spread monitor is running."""
        return self._running and bool(
            self._spread_monitor and self._spread_monitor.is_healthy()
        )
