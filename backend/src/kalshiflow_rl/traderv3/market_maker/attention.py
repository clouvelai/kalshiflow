"""MMAttentionRouter - Signal scoring and routing for market maker.

Monitors market data and quote engine events, scores them, and emits
MMAttentionItems that trigger reactive Captain cycles via the bridge.

Signal categories:
  - fill: Our quote was filled (adjust inventory, rebalance)
  - inventory_warning: Position approaching limits
  - vpin_spike: Toxic flow detected, may need to pull quotes
  - spread_change: Market spread changed significantly
  - fill_storm: Many fills in short window
"""

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional

from .models import MMAttentionItem

logger = logging.getLogger("kalshiflow_rl.traderv3.market_maker.attention")


class MMAttentionRouter:
    """Scores market signals and bridges them to Captain's AttentionRouter."""

    def __init__(
        self,
        min_score: float = 30.0,
        max_items: int = 20,
        captain_inject: Optional[Callable] = None,
    ):
        self._min_score = min_score
        self._max_items = max_items
        self._items: List[MMAttentionItem] = []
        self._dedup: Dict[str, float] = {}  # key -> last_emitted_ts
        self._notify_event = asyncio.Event()
        self._captain_inject = captain_inject  # Bridge to Captain AttentionRouter

        # Dedup cooldowns per category
        self._cooldowns = {
            "fill": 5.0,            # Allow rapid fill signals
            "inventory_warning": 60.0,
            "vpin_spike": 30.0,
            "spread_change": 60.0,
            "fill_storm": 30.0,
        }

    @property
    def notify_event(self) -> asyncio.Event:
        return self._notify_event

    @property
    def pending_items(self) -> List[MMAttentionItem]:
        """Return non-expired items, sorted by score descending."""
        now = time.time()
        self._items = [i for i in self._items if not i.is_expired]
        return sorted(self._items, key=lambda x: x.score, reverse=True)

    def drain(self) -> List[MMAttentionItem]:
        """Return and clear all pending items."""
        items = self.pending_items
        self._items = []
        self._notify_event.clear()
        return items

    # ------------------------------------------------------------------
    # Signal Emitters
    # ------------------------------------------------------------------

    def on_fill(
        self,
        event_ticker: str,
        market_ticker: str,
        side: str,
        price_cents: int,
        quantity: int,
        position_after: int,
    ) -> None:
        """Emit fill signal."""
        score = 40.0 + min(quantity * 2, 30)  # Bigger fills = higher score

        self._emit(MMAttentionItem(
            event_ticker=event_ticker,
            market_ticker=market_ticker,
            urgency="high",
            category="fill",
            summary=f"Fill: {side} {quantity}@{price_cents}c, pos now {position_after}",
            data={
                "side": side,
                "price": price_cents,
                "quantity": quantity,
                "position_after": position_after,
            },
            score=score,
            ttl_seconds=60.0,
        ))

    def on_inventory_warning(
        self,
        event_ticker: str,
        market_ticker: str,
        position: int,
        max_position: int,
    ) -> None:
        """Emit inventory warning when position is near limits."""
        pct_used = abs(position) / max_position if max_position > 0 else 0
        if pct_used < 0.7:
            return  # Only warn above 70%

        score = 50.0 + pct_used * 30
        urgency = "immediate" if pct_used >= 0.9 else "high"

        self._emit(MMAttentionItem(
            event_ticker=event_ticker,
            market_ticker=market_ticker,
            urgency=urgency,
            category="inventory_warning",
            summary=f"Inventory {pct_used*100:.0f}% of limit: pos={position}/{max_position}",
            data={"position": position, "max_position": max_position, "pct_used": pct_used},
            score=score,
            ttl_seconds=120.0,
        ))

    def on_vpin_spike(
        self,
        event_ticker: str,
        market_ticker: str,
        vpin: float,
        threshold: float,
    ) -> None:
        """Emit VPIN spike signal."""
        if vpin < threshold * 0.8:
            return  # Only signal when close to threshold

        score = 60.0 + (vpin - threshold * 0.8) * 100
        urgency = "immediate" if vpin >= threshold else "high"

        self._emit(MMAttentionItem(
            event_ticker=event_ticker,
            market_ticker=market_ticker,
            urgency=urgency,
            category="vpin_spike",
            summary=f"VPIN={vpin:.3f} (threshold={threshold})",
            data={"vpin": vpin, "threshold": threshold},
            score=min(score, 95),
            ttl_seconds=60.0,
        ))

    def on_spread_change(
        self,
        event_ticker: str,
        market_ticker: str,
        old_spread: int,
        new_spread: int,
    ) -> None:
        """Emit spread change signal when market spread changes significantly."""
        if old_spread is None or new_spread is None:
            return
        change = abs(new_spread - old_spread)
        if change < 2:
            return  # Ignore small changes

        score = 30.0 + change * 5
        direction = "widened" if new_spread > old_spread else "tightened"

        self._emit(MMAttentionItem(
            event_ticker=event_ticker,
            market_ticker=market_ticker,
            urgency="normal",
            category="spread_change",
            summary=f"Spread {direction}: {old_spread} → {new_spread}c",
            data={"old_spread": old_spread, "new_spread": new_spread},
            score=min(score, 70),
            ttl_seconds=120.0,
        ))

    def on_fill_storm(
        self,
        event_ticker: str,
        fill_count: int,
        window_seconds: float,
    ) -> None:
        """Emit fill storm signal."""
        self._emit(MMAttentionItem(
            event_ticker=event_ticker,
            urgency="immediate",
            category="fill_storm",
            summary=f"Fill storm: {fill_count} fills in {window_seconds:.0f}s",
            data={"fill_count": fill_count, "window": window_seconds},
            score=80.0,
            ttl_seconds=60.0,
        ))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, item: MMAttentionItem) -> None:
        """Emit an item if it passes score and dedup checks."""
        if item.score < self._min_score:
            return

        # Dedup check
        cooldown = self._cooldowns.get(item.category, 30.0)
        now = time.time()
        last = self._dedup.get(item.key, 0)
        if now - last < cooldown:
            return

        self._dedup[item.key] = now
        self._items.append(item)

        # Trim to max
        if len(self._items) > self._max_items:
            self._items = sorted(self._items, key=lambda x: x.score, reverse=True)[:self._max_items]

        # Notify local event (legacy compatibility)
        self._notify_event.set()

        # Bridge to Captain AttentionRouter
        if self._captain_inject:
            try:
                from ..single_arb.models import AttentionItem
                captain_item = AttentionItem(
                    event_ticker=item.event_ticker,
                    market_ticker=item.market_ticker,
                    urgency=item.urgency,
                    category=f"mm_{item.category}",
                    summary=f"[MM] {item.summary}",
                    data=item.data,
                    score=item.score,
                    ttl_seconds=item.ttl_seconds,
                )
                self._captain_inject(captain_item)
            except Exception as e:
                logger.debug(f"[MM_ATTENTION] Captain bridge error: {e}")

        logger.info(
            f"[MM_ATTENTION] Emitted {item.category} score={item.score:.0f} "
            f"urgency={item.urgency}: {item.summary}"
        )
