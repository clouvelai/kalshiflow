"""AttentionRouter - Event-driven signal scoring and priority queue.

Subscribes to EventBus (same pattern as Monitor), reads signals from
EventArbIndex, and maintains a priority queue of AttentionItems.
Captain reads from the router instead of polling all state every cycle.

Pure Python, no LLM calls. Signals already exist on MarketMeta/EventMeta.

Scoring weights:
  | Signal                          | Weight | Source                            |
  |---------------------------------|--------|-----------------------------------|
  | Edge magnitude > threshold      | 30     | EventMeta.long_edge()/short_edge()|
  | Edge delta since last check     | 15     | Diff against previous snapshot    |
  | Position P&L threshold crossing | 20     | Portfolio state                   |
  | VPIN spike (>0.7 or >0.85)     | 10     | MarketMeta.micro.vpin             |
  | Sweep detected                  | 10     | MarketMeta.micro.sweep_active     |
  | Whale trade                     | 5      | MarketMeta.micro.whale_trade_count|
  | Time to close < 1h with pos    | 10     | _time_to_close() + position check |

Score >= 40 → emit. Urgency: >=70 immediate, >=55 high, else normal.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, TYPE_CHECKING

from ..core.events.types import EventType
from .models import AttentionItem
from .context_builder import _time_to_close, _compute_regime

if TYPE_CHECKING:
    from .index import EventArbIndex, EventMeta, MarketMeta

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.attention")

# Scoring thresholds
SCORE_EMIT_THRESHOLD = 40
URGENCY_IMMEDIATE_THRESHOLD = 70
URGENCY_HIGH_THRESHOLD = 55

# Signal weights
WEIGHT_EDGE_MAGNITUDE = 30
WEIGHT_EDGE_DELTA = 15
WEIGHT_POSITION_PNL = 20
WEIGHT_VPIN_SPIKE = 10
WEIGHT_SWEEP = 10
WEIGHT_WHALE = 5
WEIGHT_TIME_PRESSURE = 10

# Edge thresholds (cents)
EDGE_MIN_CENTS = 2.0           # Below this, no edge signal
EDGE_HIGH_CENTS = 5.0          # Above this, full weight
VPIN_ALERT_THRESHOLD = 0.70    # First alert level
VPIN_CRITICAL_THRESHOLD = 0.85 # Critical level

# Debounce
BATCH_INTERVAL_SECONDS = 5.0
ITEM_DEFAULT_TTL = 120.0

# Acknowledgment cooldown — after Captain drains an item, suppress re-emission
# for this many seconds unless the score increases materially.
ACKNOWLEDGE_COOLDOWN = 120.0
ACKNOWLEDGE_SCORE_INCREASE_PCT = 0.25  # 25% score increase to re-emit during cooldown

# P&L thresholds (cents per contract)
PNL_PROFIT_THRESHOLD = 10   # +10c/ct → take profit signal
PNL_LOSS_THRESHOLD = -12    # -12c/ct → cut loss signal

# Settlement time threshold (hours)
SETTLEMENT_WARN_HOURS = 1.0


class AttentionRouter:
    """Event-driven signal scoring and priority queue for Captain.

    Subscribes to EventBus, evaluates signals on each update,
    and emits AttentionItems when scores exceed threshold.
    Captain drains items via drain_items() instead of polling all state.
    """

    def __init__(
        self,
        index: "EventArbIndex",
        config,
        auto_action_callback: Optional[Callable] = None,
        broadcast_callback: Optional[Callable] = None,
    ):
        self._index = index
        self._config = config
        self._auto_action_callback = auto_action_callback
        self._broadcast = broadcast_callback

        # Stats counters
        self._eval_count = 0
        self._emit_count = 0

        # Priority queue keyed by (event_ticker, category)
        self._items: Dict[str, AttentionItem] = {}

        # Acknowledged items: key → (timestamp, score_at_drain)
        # Prevents re-emission of the same signal within cooldown window
        self._acknowledged: Dict[str, tuple] = {}

        # Previous state for delta computation
        self._prev_edges: Dict[str, Dict[str, float]] = {}  # event_ticker -> {long, short}
        self._prev_whale_counts: Dict[str, int] = {}  # market_ticker -> count

        # Position state (updated by Captain before waiting)
        self._positions: Dict[str, Dict] = {}  # ticker -> {side, qty, pnl_per_ct, event_ticker}

        # Notification event for Captain's _wait_for_trigger
        self._notify = asyncio.Event()

        # Background batch loop
        self._running = False
        self._batch_task: Optional[asyncio.Task] = None

    def subscribe(self, event_bus) -> None:
        """Subscribe to EventBus channels (same pattern as Monitor)."""
        event_bus.subscribe(EventType.ORDERBOOK_SNAPSHOT, self._on_orderbook)
        event_bus.subscribe(EventType.ORDERBOOK_DELTA, self._on_orderbook)
        event_bus.subscribe(EventType.MARKET_TRADE, self._on_trade)
        event_bus.subscribe(EventType.TICKER_UPDATE, self._on_ticker)

    def unsubscribe(self, event_bus) -> None:
        """Unsubscribe from EventBus."""
        event_bus.unsubscribe(EventType.ORDERBOOK_SNAPSHOT, self._on_orderbook)
        event_bus.unsubscribe(EventType.ORDERBOOK_DELTA, self._on_orderbook)
        event_bus.unsubscribe(EventType.MARKET_TRADE, self._on_trade)
        event_bus.unsubscribe(EventType.TICKER_UPDATE, self._on_ticker)

    async def start(self) -> None:
        """Start the batch emit loop."""
        if self._running:
            return
        self._running = True
        self._batch_task = asyncio.create_task(self._batch_loop())
        logger.info("[ATTENTION] Router started")

    async def stop(self) -> None:
        """Stop the batch emit loop."""
        self._running = False
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        logger.info(f"[ATTENTION] Router stopped (pending={len(self._items)})")

    # ------------------------------------------------------------------
    # Public interface for Captain
    # ------------------------------------------------------------------

    def drain_items(self, min_urgency: str = "high") -> List[AttentionItem]:
        """Return and clear qualifying items. Captain calls this.

        Drained items are recorded in _acknowledged with their score so that
        _emit_item can suppress re-emission within the cooldown window.
        """
        urgency_rank = {"immediate": 0, "high": 1, "normal": 2}
        min_rank = urgency_rank.get(min_urgency, 1)
        now = time.time()

        result = []
        to_remove = []
        for key, item in self._items.items():
            # Remove expired
            if item.is_expired:
                to_remove.append(key)
                continue
            # Filter by urgency
            item_rank = urgency_rank.get(item.urgency, 2)
            if item_rank <= min_rank:
                result.append(item)
                to_remove.append(key)

        for key in to_remove:
            item = self._items.pop(key, None)
            # Record acknowledged items (not expired ones) with their score
            if item and not item.is_expired:
                self._acknowledged[key] = (now, item.score)

        # Reset notification
        self._notify.clear()

        return sorted(result, key=lambda x: x.score, reverse=True)

    def pending_items(self) -> List[AttentionItem]:
        """Peek at all non-expired items without draining."""
        return sorted(
            [i for i in self._items.values() if not i.is_expired],
            key=lambda x: x.score,
            reverse=True,
        )

    def update_positions(self, positions: List[Dict]) -> None:
        """Update position state for P&L scoring. Called by Captain."""
        self._positions = {}
        for p in positions:
            ticker = p.get("ticker", "")
            if ticker and p.get("quantity", 0) > 0:
                self._positions[ticker] = {
                    "side": p.get("side", ""),
                    "qty": p.get("quantity", 0),
                    "pnl_per_ct": p.get("pnl_per_ct", 0),
                    "event_ticker": p.get("event_ticker", ""),
                }

    def inject_item(self, item: AttentionItem) -> None:
        """Public API for external systems (Sniper, AutoActions) to inject items."""
        self._emit_item(item)

    @property
    def has_items(self) -> bool:
        """True if there are any non-expired qualifying items."""
        for item in self._items.values():
            if not item.is_expired and item.urgency in ("immediate", "high"):
                return True
        return False

    @property
    def notify_event(self) -> asyncio.Event:
        """Event that Captain waits on for reactive triggers."""
        return self._notify

    def clear_notify(self) -> None:
        """Clear the notification event (public API for Captain)."""
        self._notify.clear()

    # ------------------------------------------------------------------
    # EventBus callbacks
    # ------------------------------------------------------------------

    async def _on_orderbook(self, market_ticker: str, metadata: Dict) -> None:
        """Recompute scores when orderbook updates."""
        if not self._running:
            return
        event_ticker = self._index.get_event_for_ticker(market_ticker)
        if not event_ticker:
            return
        self._evaluate_event(event_ticker)

    async def _on_trade(self, market_ticker: str, metadata: Dict) -> None:
        """Check for whale trades and sweeps."""
        if not self._running:
            return
        event_ticker = self._index.get_event_for_ticker(market_ticker)
        if not event_ticker:
            return

        event = self._index.events.get(event_ticker)
        if not event:
            return
        market = event.markets.get(market_ticker)
        if not market:
            return

        # Whale detection (delta-based)
        prev_whales = self._prev_whale_counts.get(market_ticker, 0)
        cur_whales = market.micro.whale_trade_count
        if cur_whales > prev_whales:
            count = metadata.get("count", 0)
            self._emit_item(AttentionItem(
                event_ticker=event_ticker,
                market_ticker=market_ticker,
                category="whale_activity",
                summary=f"whale trade {count}ct on {market_ticker}",
                score=WEIGHT_WHALE * min(count / 100, 2.0),
                data={"count": count, "side": metadata.get("taker_side", "")},
                ttl_seconds=60.0,
            ))
        self._prev_whale_counts[market_ticker] = cur_whales

        # Sweep detection
        if market.micro.sweep_active:
            self._emit_item(AttentionItem(
                event_ticker=event_ticker,
                market_ticker=market_ticker,
                category="sweep_detected",
                summary=f"sweep {market.micro.sweep_direction} {market.micro.sweep_size}ct across {market.micro.sweep_levels} levels",
                score=WEIGHT_SWEEP * 1.5,
                data={
                    "direction": market.micro.sweep_direction,
                    "size": market.micro.sweep_size,
                    "levels": market.micro.sweep_levels,
                },
                ttl_seconds=30.0,
            ))

    async def _on_ticker(self, market_ticker: str, metadata: Dict) -> None:
        """Trigger evaluation on ticker updates."""
        if not self._running:
            return
        event_ticker = self._index.get_event_for_ticker(market_ticker)
        if event_ticker:
            self._evaluate_event(event_ticker)

    # ------------------------------------------------------------------
    # Scoring engine
    # ------------------------------------------------------------------

    def _evaluate_event(self, event_ticker: str) -> None:
        """Evaluate all signals for an event and emit items if scoring threshold met."""
        self._eval_count += 1
        event = self._index.events.get(event_ticker)
        if not event or not event.all_markets_have_data:
            return

        fee = self._index.fee_per_contract

        # 1. Edge magnitude
        long_edge = event.long_edge(fee) or 0.0
        short_edge = event.short_edge(fee) or 0.0
        best_edge = max(long_edge, short_edge)
        direction = "long" if long_edge >= short_edge else "short"

        edge_score = 0.0
        if best_edge >= EDGE_HIGH_CENTS:
            edge_score = WEIGHT_EDGE_MAGNITUDE
        elif best_edge >= EDGE_MIN_CENTS:
            edge_score = WEIGHT_EDGE_MAGNITUDE * (best_edge - EDGE_MIN_CENTS) / (EDGE_HIGH_CENTS - EDGE_MIN_CENTS)

        # 2. Edge delta
        prev = self._prev_edges.get(event_ticker, {})
        prev_best = max(prev.get("long", 0), prev.get("short", 0))
        edge_delta = best_edge - prev_best
        self._prev_edges[event_ticker] = {"long": long_edge, "short": short_edge}

        delta_score = 0.0
        if edge_delta > 1.0:
            delta_score = min(WEIGHT_EDGE_DELTA, WEIGHT_EDGE_DELTA * edge_delta / 5.0)

        # 3. VPIN spike (worst across markets)
        max_vpin = max((m.micro.vpin for m in event.markets.values()), default=0.0)
        vpin_score = 0.0
        if max_vpin >= VPIN_CRITICAL_THRESHOLD:
            vpin_score = WEIGHT_VPIN_SPIKE
        elif max_vpin >= VPIN_ALERT_THRESHOLD:
            vpin_score = WEIGHT_VPIN_SPIKE * 0.5

        # 4. Position P&L
        pnl_score = 0.0
        pnl_summary = ""
        for ticker, pos in self._positions.items():
            if pos.get("event_ticker") != event_ticker:
                continue
            pnl_ct = pos.get("pnl_per_ct", 0)
            if pnl_ct >= PNL_PROFIT_THRESHOLD:
                pnl_score = max(pnl_score, WEIGHT_POSITION_PNL)
                pnl_summary = f"pnl={pnl_ct:+d}c/ct (take profit)"
            elif pnl_ct <= PNL_LOSS_THRESHOLD:
                pnl_score = max(pnl_score, WEIGHT_POSITION_PNL)
                pnl_summary = f"pnl={pnl_ct:+d}c/ct (cut loss)"

        # 5. Time to close with position
        ttc = _time_to_close(event)
        time_score = 0.0
        time_summary = ""
        has_position = any(
            p.get("event_ticker") == event_ticker for p in self._positions.values()
        )
        if ttc is not None and ttc < SETTLEMENT_WARN_HOURS and has_position:
            time_score = WEIGHT_TIME_PRESSURE
            time_summary = f"ttc={ttc:.1f}h with open position"

        # Composite score
        total_score = edge_score + delta_score + vpin_score + pnl_score + time_score

        if total_score < SCORE_EMIT_THRESHOLD:
            return

        # Determine urgency
        if total_score >= URGENCY_IMMEDIATE_THRESHOLD:
            urgency = "immediate"
        elif total_score >= URGENCY_HIGH_THRESHOLD:
            urgency = "high"
        else:
            urgency = "normal"

        # Build summary
        parts = []
        if best_edge >= EDGE_MIN_CENTS:
            regime = self._event_regime(event)
            spread = self._event_avg_spread(event)
            parts.append(f"{direction} edge={best_edge:.1f}c regime={regime} spread={spread}c")
        if edge_delta > 1.0:
            parts.append(f"edge_delta=+{edge_delta:.1f}c")
        if max_vpin >= VPIN_ALERT_THRESHOLD:
            parts.append(f"vpin={max_vpin:.2f}")
        if pnl_summary:
            parts.append(pnl_summary)
        if time_summary:
            parts.append(time_summary)

        # Determine primary category
        if pnl_score >= WEIGHT_POSITION_PNL:
            category = "position_risk"
        elif time_score > 0:
            category = "settlement_approaching"
        elif max_vpin >= VPIN_CRITICAL_THRESHOLD:
            category = "regime_change"
        elif edge_delta > 2.0:
            category = "edge_emergence"
        elif best_edge >= EDGE_MIN_CENTS:
            category = "arb_opportunity"
        else:
            category = "arb_opportunity"

        item = AttentionItem(
            event_ticker=event_ticker,
            urgency=urgency,
            category=category,
            summary=", ".join(parts),
            score=total_score,
            data={
                "long_edge": round(long_edge, 1),
                "short_edge": round(short_edge, 1),
                "direction": direction,
                "edge_delta": round(edge_delta, 1),
                "max_vpin": round(max_vpin, 3),
                "regime": self._event_regime(event),
                "ttc_hours": ttc,
            },
        )
        self._emit_item(item)

    # ------------------------------------------------------------------
    # Item management
    # ------------------------------------------------------------------

    def _emit_item(self, item: AttentionItem) -> None:
        """Add or replace item in the queue. Debounced by key.

        Suppresses re-emission of items that Captain recently drained
        unless the score has increased materially (>= 25%).
        """
        key = item.key

        # Check acknowledgment cooldown
        ack = self._acknowledged.get(key)
        if ack:
            ack_time, ack_score = ack
            elapsed = time.time() - ack_time
            if elapsed < ACKNOWLEDGE_COOLDOWN:
                # Still in cooldown — only re-emit if score increased materially
                if ack_score > 0 and item.score < ack_score * (1.0 + ACKNOWLEDGE_SCORE_INCREASE_PCT):
                    return  # Suppressed
                # Material score increase — allow re-emission and clear ack
                del self._acknowledged[key]

        self._emit_count += 1
        existing = self._items.get(key)

        # Merge: keep higher score, update data
        if existing and not existing.is_expired:
            if item.score > existing.score:
                self._items[key] = item
            else:
                # Update data on existing but keep score
                existing.data.update(item.data)
                existing.created_at = time.time()  # Refresh TTL
        else:
            self._items[key] = item

        # Run auto-action callback (if registered) — tracked async task
        # Defer Captain notification until auto_action completes so that
        # item.data["auto_handled"] is set before Captain drains.
        if self._auto_action_callback:
            try:
                coro = self._auto_action_callback(item)
                if asyncio.iscoroutine(coro):
                    task = asyncio.create_task(coro)
                    task.add_done_callback(
                        lambda t: self._on_auto_action_done(t, item)
                    )
                    return  # Notification deferred to callback
            except Exception as e:
                logger.debug(f"[ATTENTION] Auto-action callback error: {e}")

        # Signal Captain (only when no auto-action callback or callback setup failed)
        if item.urgency in ("immediate", "high"):
            self._notify.set()

    def _on_auto_action_done(self, task: asyncio.Task, item: AttentionItem) -> None:
        """Done callback for auto-action tasks.

        Logs errors and signals Captain AFTER auto_action has processed,
        preventing race where Captain drains before auto_handled is set.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"[ATTENTION] Auto-action failed: {exc}")
        # Now that auto_action has processed (or failed), notify Captain
        if item.urgency in ("immediate", "high"):
            self._notify.set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _event_regime(event: "EventMeta") -> str:
        """Quick event regime from markets."""
        return _compute_regime(event)

    @staticmethod
    def _event_avg_spread(event: "EventMeta") -> int:
        """Average spread across markets with data."""
        spreads = [m.spread for m in event.markets.values() if m.spread is not None]
        return round(sum(spreads) / len(spreads)) if spreads else 0

    async def _batch_loop(self) -> None:
        """Periodic cleanup of expired items + broadcast attention snapshot."""
        broadcast_counter = 0
        while self._running:
            try:
                await asyncio.sleep(BATCH_INTERVAL_SECONDS)
                if not self._running:
                    break
                # Expire old items
                expired = [k for k, v in self._items.items() if v.is_expired]
                for k in expired:
                    del self._items[k]

                # Expire old acknowledgments
                now = time.time()
                expired_acks = [
                    k for k, (ts, _) in self._acknowledged.items()
                    if now - ts >= ACKNOWLEDGE_COOLDOWN
                ]
                for k in expired_acks:
                    del self._acknowledged[k]

                # Broadcast attention snapshot every ~10s (2 batch intervals)
                broadcast_counter += 1
                if broadcast_counter >= 2 and self._broadcast:
                    broadcast_counter = 0
                    try:
                        pending = self.pending_items()
                        await self._broadcast({
                            "type": "attention_snapshot",
                            "data": {
                                "items": [i.to_dict() for i in pending],
                                "stats": {
                                    "total_evaluated": self._eval_count,
                                    "items_emitted_total": self._emit_count,
                                    "pending_count": len(pending),
                                },
                            },
                        })
                    except Exception as e:
                        logger.debug(f"[ATTENTION] Broadcast error: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[ATTENTION] Batch loop error: {e}")

    def get_stats(self) -> Dict:
        """Stats for health endpoint."""
        return {
            "pending_items": len(self._items),
            "acknowledged_items": len(self._acknowledged),
            "items_by_urgency": {
                "immediate": sum(1 for i in self._items.values() if i.urgency == "immediate"),
                "high": sum(1 for i in self._items.values() if i.urgency == "high"),
                "normal": sum(1 for i in self._items.values() if i.urgency == "normal"),
            },
            "tracked_positions": len(self._positions),
            "tracked_events": len(self._prev_edges),
        }
