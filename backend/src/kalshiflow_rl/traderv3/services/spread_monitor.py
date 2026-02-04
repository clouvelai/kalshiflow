"""
Spread Monitor - Hot Execution Path.

Subscribes to Kalshi orderbook and Polymarket price events via EventBus.
Maintains live spread state for each paired market.
When abs(spread) > threshold: executes trade DIRECTLY (no LLM, pure Python).

This is the performance-critical path: event -> spread calc -> order < 100ms.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.event_bus import EventBus
from ..core.events.types import EventType
from ..core.events.arb_events import PolyPriceEvent, KalshiApiPriceEvent, SpreadUpdateEvent, SpreadTradeExecutedEvent
from ..services.pair_registry import PairRegistry
from ..config.environment import V3Config

logger = logging.getLogger("kalshiflow_rl.traderv3.services.spread_monitor")


@dataclass
class SpreadState:
    """Live spread state for a single paired market."""
    pair_id: str
    kalshi_ticker: str
    question: str = ""

    # Effective prices (used for spread calculation - best available from each venue)
    kalshi_yes_bid: Optional[int] = None  # best bid cents
    kalshi_yes_ask: Optional[int] = None  # best ask cents
    kalshi_yes_mid: Optional[int] = None  # midpoint cents
    poly_yes_cents: Optional[int] = None
    poly_no_cents: Optional[int] = None

    # --- Kalshi WS (orderbook BBO) ---
    kalshi_ws_yes_bid: Optional[int] = None
    kalshi_ws_yes_ask: Optional[int] = None
    kalshi_ws_yes_mid: Optional[int] = None
    kalshi_ws_updated_at: float = 0.0

    # --- Kalshi API (REST orderbook poll) ---
    kalshi_api_yes_bid: Optional[int] = None
    kalshi_api_yes_ask: Optional[int] = None
    kalshi_api_yes_mid: Optional[int] = None
    kalshi_api_updated_at: float = 0.0

    # --- Poly WS ---
    poly_ws_yes_cents: Optional[int] = None
    poly_ws_updated_at: float = 0.0

    # --- Poly API ---
    poly_api_yes_cents: Optional[int] = None
    poly_api_updated_at: float = 0.0

    # Computed spread
    spread_cents: Optional[int] = None  # kalshi_mid - poly_yes

    # Cooldown tracking
    last_trade_at: float = 0.0
    trades_count: int = 0

    @property
    def tradeable(self) -> bool:
        """Both venues have effective prices (API-sourced)."""
        return self.kalshi_yes_mid is not None and self.poly_yes_cents is not None

    @property
    def tradeable_reason(self) -> Optional[str]:
        """Human-readable reason why pair is not tradeable, or None if tradeable."""
        if self.kalshi_yes_mid is None and self.poly_yes_cents is None:
            return "No Kalshi or Poly liquidity"
        if self.kalshi_yes_mid is None:
            return "No Kalshi liquidity"
        if self.poly_yes_cents is None:
            return "No Poly liquidity"
        return None

    def is_fresh(self, max_age: float = 30.0) -> bool:
        """Both venues have recent data from ANY source."""
        now = time.time()
        kalshi_age = min(
            (now - self.kalshi_ws_updated_at) if self.kalshi_ws_updated_at else 9999,
            (now - self.kalshi_api_updated_at) if self.kalshi_api_updated_at else 9999,
        )
        poly_age = min(
            (now - self.poly_ws_updated_at) if self.poly_ws_updated_at else 9999,
            (now - self.poly_api_updated_at) if self.poly_api_updated_at else 9999,
        )
        return (
            self.kalshi_yes_mid is not None
            and self.poly_yes_cents is not None
            and kalshi_age < max_age
            and poly_age < max_age
        )

    def compute_spread(self) -> Optional[int]:
        """Compute spread in cents. Positive = Kalshi expensive, Negative = Kalshi cheap."""
        if self.kalshi_yes_mid is not None and self.poly_yes_cents is not None:
            self.spread_cents = self.kalshi_yes_mid - self.poly_yes_cents
            return self.spread_cents
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize spread state to a dict for WebSocket/API responses."""
        now = time.time()
        return {
            "pair_id": self.pair_id,
            "kalshi_ticker": self.kalshi_ticker,
            "question": self.question,
            # Effective prices
            "kalshi_yes_bid": self.kalshi_yes_bid,
            "kalshi_yes_ask": self.kalshi_yes_ask,
            "kalshi_yes_mid": self.kalshi_yes_mid,
            "poly_yes_cents": self.poly_yes_cents,
            "poly_no_cents": self.poly_no_cents,
            "spread_cents": self.spread_cents,
            # Kalshi WS source
            "kalshi_ws_yes_bid": self.kalshi_ws_yes_bid,
            "kalshi_ws_yes_ask": self.kalshi_ws_yes_ask,
            "kalshi_ws_yes_mid": self.kalshi_ws_yes_mid,
            "kalshi_ws_age_ms": round((now - self.kalshi_ws_updated_at) * 1000) if self.kalshi_ws_updated_at else None,
            # Kalshi API source
            "kalshi_api_yes_bid": self.kalshi_api_yes_bid,
            "kalshi_api_yes_ask": self.kalshi_api_yes_ask,
            "kalshi_api_yes_mid": self.kalshi_api_yes_mid,
            "kalshi_api_age_ms": round((now - self.kalshi_api_updated_at) * 1000) if self.kalshi_api_updated_at else None,
            # Poly WS source
            "poly_ws_yes_cents": self.poly_ws_yes_cents,
            "poly_ws_age_ms": round((now - self.poly_ws_updated_at) * 1000) if self.poly_ws_updated_at else None,
            # Poly API source
            "poly_api_yes_cents": self.poly_api_yes_cents,
            "poly_api_age_ms": round((now - self.poly_api_updated_at) * 1000) if self.poly_api_updated_at else None,
            # Tradeable status
            "tradeable": self.tradeable,
            "tradeable_reason": self.tradeable_reason,
        }


class SpreadMonitor:
    """
    Hot execution path for cross-venue spread monitoring.

    Event-driven spread computation with direct order execution (no LLM).
    Includes per-pair cooldowns and configurable thresholds.
    """

    def __init__(
        self,
        event_bus: EventBus,
        pair_registry: PairRegistry,
        config: V3Config,
        trading_client=None,
        websocket_manager=None,
        supabase_client=None,
        orderbook_integration=None,
    ):
        self._event_bus = event_bus
        self._pair_registry = pair_registry
        self._config = config
        self._trading_client = trading_client
        self._websocket_manager = websocket_manager
        self._supabase = supabase_client
        self._orderbook_integration = orderbook_integration

        # Live spread state per pair
        self._spreads: Dict[str, SpreadState] = {}

        # Metrics
        self._spread_updates = 0
        self._trades_executed = 0
        self._trades_skipped_cooldown = 0
        self._daily_pnl_cents = 0

        self._running = False

    async def start(self) -> None:
        """Start spread monitoring by subscribing to events."""
        if self._running:
            return

        self._running = True

        # Initialize spread state for all active pairs
        for pair in self._pair_registry.get_all_active():
            self._spreads[pair.id] = SpreadState(
                pair_id=pair.id,
                kalshi_ticker=pair.kalshi_ticker,
                question=pair.question,
            )

        # Subscribe to events
        self._event_bus.subscribe(EventType.ORDERBOOK_SNAPSHOT, self._on_orderbook)
        self._event_bus.subscribe(EventType.ORDERBOOK_DELTA, self._on_orderbook)
        self._event_bus.subscribe(EventType.POLY_PRICE_UPDATE, self._on_poly_price)
        self._event_bus.subscribe(EventType.KALSHI_API_PRICE_UPDATE, self._on_kalshi_api)
        self._event_bus.subscribe(EventType.PAIR_MATCHED, self._on_pair_matched)

        logger.info(f"Spread monitor started: {len(self._spreads)} pairs, threshold={self._config.arb_spread_threshold_cents}c")

    def register_pair(self, pair) -> None:
        """Register a new pair for spread monitoring at runtime.

        Called when the agent discovers a new pair (via PAIR_MATCHED event)
        or when upsert_paired_market adds to the in-memory registry.
        Also subscribes the pair's Kalshi ticker to the orderbook integration
        so we receive BBO data for it.
        """
        if pair.id in self._spreads:
            return
        self._spreads[pair.id] = SpreadState(
            pair_id=pair.id,
            kalshi_ticker=pair.kalshi_ticker,
            question=pair.question,
        )
        logger.info(f"Registered new pair for spread monitoring: {pair.kalshi_ticker}")

        # Subscribe to orderbook for this ticker (fire-and-forget)
        if self._orderbook_integration:
            asyncio.create_task(self._subscribe_pair_orderbook(pair.kalshi_ticker))

    async def _subscribe_pair_orderbook(self, ticker: str) -> None:
        """Subscribe a pair's Kalshi ticker to the orderbook integration."""
        try:
            await self._orderbook_integration.subscribe_market(ticker)
            logger.info(f"Subscribed new arb pair to orderbook: {ticker}")
        except Exception as e:
            logger.warning(f"Failed to subscribe arb pair {ticker} to orderbook: {e}")

    async def _on_pair_matched(self, event) -> None:
        """Handle PAIR_MATCHED events - dynamically add new pairs."""
        if not self._running:
            return
        pair = self._pair_registry.get_by_id(event.pair_id)
        if pair:
            self.register_pair(pair)

    async def stop(self) -> None:
        """Stop spread monitoring."""
        self._running = False

        # Unsubscribe from all events to prevent leaks
        self._event_bus.unsubscribe(EventType.ORDERBOOK_SNAPSHOT, self._on_orderbook)
        self._event_bus.unsubscribe(EventType.ORDERBOOK_DELTA, self._on_orderbook)
        self._event_bus.unsubscribe(EventType.POLY_PRICE_UPDATE, self._on_poly_price)
        self._event_bus.unsubscribe(EventType.KALSHI_API_PRICE_UPDATE, self._on_kalshi_api)
        self._event_bus.unsubscribe(EventType.PAIR_MATCHED, self._on_pair_matched)

        logger.info(f"Spread monitor stopped (trades={self._trades_executed}, updates={self._spread_updates})")

    async def _on_orderbook(self, market_ticker: str, metadata: Dict) -> None:
        """Handle Kalshi orderbook snapshot/delta events.

        EventBus passes (market_ticker, metadata) for orderbook events.
        Reads BBO directly from event metadata (attached at emission time
        by the orderbook client, no async lock acquisition needed).
        """
        if not self._running:
            return

        pair = self._pair_registry.get_by_kalshi(market_ticker)
        if not pair:
            return

        state = self._spreads.get(pair.id)
        if not state:
            return

        # Read BBO from metadata (attached by orderbook_client at emission time)
        yes_bid = metadata.get("yes_bid")
        yes_ask = metadata.get("yes_ask")
        yes_mid = metadata.get("yes_mid")

        # WS is display-only: update WS fields but NOT effective prices
        if yes_bid is not None:
            state.kalshi_ws_yes_bid = yes_bid
        if yes_ask is not None:
            state.kalshi_ws_yes_ask = yes_ask
        if yes_mid is not None:
            state.kalshi_ws_yes_mid = yes_mid

        state.kalshi_ws_updated_at = time.time()
        # No spread recalc: WS is informational only, API drives spread

    async def _on_poly_price(self, event: PolyPriceEvent) -> None:
        """Handle Polymarket price update events."""
        if not self._running:
            return

        state = self._spreads.get(event.pair_id)
        if not state:
            return

        # Track per-source prices and timestamps independently
        source = getattr(event, 'source', 'ws')
        now = time.time()
        if source == "ws":
            state.poly_ws_yes_cents = event.poly_yes_cents
            state.poly_ws_updated_at = now
        else:
            state.poly_api_yes_cents = event.poly_yes_cents
            state.poly_api_updated_at = now

        # Effective price always set from incoming event
        state.poly_yes_cents = event.poly_yes_cents
        state.poly_no_cents = event.poly_no_cents
        await self._update_spread(state)

    async def _on_kalshi_api(self, event: KalshiApiPriceEvent) -> None:
        """Handle Kalshi REST API price update events."""
        if not self._running:
            return

        state = self._spreads.get(event.pair_id)
        if not state:
            # Try lookup by ticker (poller uses pair_id from registry)
            pair = self._pair_registry.get_by_kalshi(event.kalshi_ticker)
            if pair:
                state = self._spreads.get(pair.id)
        if not state:
            return

        # Track Kalshi API source independently
        state.kalshi_api_yes_bid = event.yes_bid
        state.kalshi_api_yes_ask = event.yes_ask
        state.kalshi_api_yes_mid = event.yes_mid
        state.kalshi_api_updated_at = time.time()

        # API is the primary signal: always update effective prices
        if event.yes_bid is not None:
            state.kalshi_yes_bid = event.yes_bid
        if event.yes_ask is not None:
            state.kalshi_yes_ask = event.yes_ask
        if event.yes_mid is not None:
            state.kalshi_yes_mid = event.yes_mid

        # Trigger spread recalculation
        if state.kalshi_yes_mid is not None:
            await self._update_spread(state)

    async def _update_spread(self, state: SpreadState) -> None:
        """Recompute spread and check if trade should execute."""
        spread = state.compute_spread()
        if spread is None:
            return

        self._spread_updates += 1

        spread_event = SpreadUpdateEvent(
            pair_id=state.pair_id,
            kalshi_ticker=state.kalshi_ticker,
            kalshi_yes_bid=state.kalshi_yes_bid,
            kalshi_yes_ask=state.kalshi_yes_ask,
            kalshi_yes_mid=state.kalshi_yes_mid,
            poly_yes_cents=state.poly_yes_cents,
            poly_no_cents=state.poly_no_cents,
            spread_cents=spread,
            question=state.question,
        )
        await self._event_bus.emit(EventType.SPREAD_UPDATE, spread_event)

        if self._websocket_manager:
            await self._websocket_manager.broadcast_message("spread_update", state.to_dict())

        pair = self._pair_registry.get_by_id(state.pair_id)
        threshold = (pair.threshold_override_cents if pair and pair.threshold_override_cents
                     else self._config.arb_spread_threshold_cents)

        if abs(spread) >= threshold and state.is_fresh():
            await self._maybe_execute_trade(state, spread, threshold)

    async def _maybe_execute_trade(self, state: SpreadState, spread: int, threshold: int) -> None:
        """Check constraints and execute arb trade if conditions met."""
        now = time.time()

        # Event ticker whitelist check
        if self._config.arb_active_event_tickers:
            pair = self._pair_registry.get_by_id(state.pair_id)
            if pair and pair.kalshi_event_ticker not in self._config.arb_active_event_tickers:
                return  # Event not in whitelist

        # Cooldown check
        if (now - state.last_trade_at) < self._config.arb_cooldown_seconds:
            self._trades_skipped_cooldown += 1
            return

        if not self._trading_client:
            logger.debug(f"Spread {spread}c on {state.kalshi_ticker} exceeds threshold but no trading client")
            return

        # Positive spread: Kalshi YES expensive vs Poly -> buy YES on Kalshi at ask
        # (we'll hedge by notionally selling on Poly where it's cheaper).
        # Negative spread: Kalshi YES cheap vs Poly -> sell YES on Kalshi at bid.
        #
        # For paper trading MVP we only execute the Kalshi leg.
        if spread > 0:
            # Kalshi YES overpriced: sell YES on Kalshi (collect bid price)
            side = "yes"
            action = "sell"
            price_cents = state.kalshi_yes_bid if state.kalshi_yes_bid else state.kalshi_yes_mid
        else:
            # Kalshi YES underpriced: buy YES on Kalshi (pay ask price)
            side = "yes"
            action = "buy"
            price_cents = state.kalshi_yes_ask if state.kalshi_yes_ask else state.kalshi_yes_mid

        # Guard: if we still have no price, skip
        if price_cents is None:
            logger.warning(f"No Kalshi price for {state.kalshi_ticker}, skipping trade (spread={spread}c)")
            return

        contracts = min(10, self._config.arb_max_position_per_pair)  # Use config limit

        reasoning = f"Spread {spread}c (threshold {threshold}c) on {state.kalshi_ticker}"

        try:
            order_result = await self._trading_client.create_order(
                ticker=state.kalshi_ticker,
                side=side,
                action=action,
                count=contracts,
                type="limit",
                yes_price=price_cents if side == "yes" else None,
                no_price=price_cents if side == "no" else None,
            )

            order_id = order_result.get("order", {}).get("order_id") if order_result else None

            if not order_id:
                logger.warning(f"Arb order may have failed for {state.kalshi_ticker}: no order_id returned")

            state.last_trade_at = now
            state.trades_count += 1
            self._trades_executed += 1

            logger.info(f"ARB TRADE: {action} {contracts} {side} {state.kalshi_ticker} @ {price_cents}c (spread={spread}c, order={order_id})")

            trade_event = SpreadTradeExecutedEvent(
                pair_id=state.pair_id,
                kalshi_ticker=state.kalshi_ticker,
                side=side,
                action=action,
                contracts=contracts,
                price_cents=price_cents,
                spread_at_entry=spread,
                kalshi_mid=state.kalshi_yes_mid or 0,
                poly_mid=state.poly_yes_cents or 0,
                kalshi_order_id=order_id,
                reasoning=reasoning,
            )
            await self._event_bus.emit(EventType.SPREAD_TRADE_EXECUTED, trade_event)

            if self._websocket_manager:
                await self._websocket_manager.broadcast_message("arb_trade_executed", {
                    "pair_id": state.pair_id,
                    "kalshi_ticker": state.kalshi_ticker,
                    "side": side,
                    "action": action,
                    "contracts": contracts,
                    "price_cents": price_cents,
                    "spread_at_entry": spread,
                    "order_id": order_id,
                    "reasoning": reasoning,
                })

            if self._supabase:
                asyncio.create_task(self._log_trade(trade_event))

        except Exception as e:
            logger.error(f"Arb trade execution failed: {e}")

    async def _log_trade(self, event: SpreadTradeExecutedEvent) -> None:
        """Write trade to arb_trades table (best-effort)."""
        try:
            self._supabase.table("arb_trades").insert({
                "pair_id": event.pair_id,
                "kalshi_ticker": event.kalshi_ticker,
                "side": event.side,
                "action": event.action,
                "contracts": event.contracts,
                "price_cents": event.price_cents,
                "spread_at_entry": event.spread_at_entry,
                "kalshi_mid": event.kalshi_mid,
                "poly_mid": event.poly_mid,
                "kalshi_order_id": event.kalshi_order_id,
                "reasoning": event.reasoning,
                "status": "executed",
            }).execute()
        except Exception as e:
            logger.debug(f"Failed to log arb trade: {e}")

    def get_spread_dashboard(self) -> List[Dict[str, Any]]:
        """Get current spreads for all pairs, sorted by absolute spread descending."""
        result = []
        for s in self._spreads.values():
            entry = s.to_dict()
            entry["is_fresh"] = s.is_fresh()
            entry["trades_count"] = s.trades_count
            entry["last_trade_at"] = s.last_trade_at
            result.append(entry)
        return sorted(result, key=lambda x: abs(x.get("spread_cents") or 0), reverse=True)

    def get_spread_state(self, pair_id: str) -> Optional[SpreadState]:
        """Get live spread state for a single pair (used by PairIndexService)."""
        return self._spreads.get(pair_id)

    def get_status(self) -> Dict[str, Any]:
        """Get monitor status for health endpoints."""
        return {
            "running": self._running,
            "pairs_monitored": len(self._spreads),
            "spread_updates": self._spread_updates,
            "trades_executed": self._trades_executed,
            "trades_skipped_cooldown": self._trades_skipped_cooldown,
            "threshold_cents": self._config.arb_spread_threshold_cents,
            "cooldown_seconds": self._config.arb_cooldown_seconds,
        }

    def is_healthy(self) -> bool:
        """Health check."""
        return self._running
