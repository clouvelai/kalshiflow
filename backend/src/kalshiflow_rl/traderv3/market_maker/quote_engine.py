"""QuoteEngine - Deterministic quoting loop for market making.

Runs every refresh_interval seconds. For each market:
  1. Compute fair value (via fair_value module)
  2. Compute spread (max of base, vol-adjusted, break-even)
  3. Compute skew from inventory
  4. Compute bid/ask prices
  5. Check risk gates (position limit, event exposure, drawdown, VPIN)
  6. Execute: cancel+replace if prices changed

No LLM calls on the hot path. Admiral configures QuoteConfig; engine executes.

Architecture:
  - QuoteEngine owns the requote loop
  - Receives fill notifications via on_fill() callback
  - Gateway (KalshiGateway) used for order placement/cancellation
  - MMIndex provides market data and tracks active quotes
"""

import asyncio
import logging
import time
from collections import deque
from typing import Any, Callable, Coroutine, Dict, List, Optional

from . import fee_calculator
from . import fair_value as fv_module
from . import inventory_manager as inv_mgr
from .index import MMIndex
from .models import ActiveQuote, QuoteConfig, QuoteState

logger = logging.getLogger("kalshiflow_rl.traderv3.market_maker.quote_engine")

# Type for fill notification callback (market_ticker, side, action, price, qty)
FillCallback = Callable[[str, str, str, int, int], Coroutine[Any, Any, None]]

# Type for WS broadcast callback
WSBroadcastCallback = Callable[[str, Dict], Coroutine[Any, Any, None]]


class QuoteEngine:
    """Deterministic quoting engine.

    Maintains two-sided quotes across all markets in the MMIndex.
    Runs a periodic loop that recomputes and replaces quotes as needed.
    """

    def __init__(
        self,
        index: MMIndex,
        gateway,  # KalshiGateway
        config: QuoteConfig,
        max_drawdown_cents: int = 50000,
        order_ttl: int = 60,
        ws_broadcast: Optional[WSBroadcastCallback] = None,
    ):
        self._index = index
        self._gateway = gateway
        self._config = config
        self._max_drawdown_cents = max_drawdown_cents
        self._order_ttl = order_ttl
        self._ws_broadcast = ws_broadcast

        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Fill tracking for fill-storm detection
        self._recent_fills: deque = deque(maxlen=100)

        # Track starting balance for drawdown
        self._starting_balance_cents: Optional[int] = None
        self._current_balance_cents: int = 0

    @property
    def config(self) -> QuoteConfig:
        return self._config

    @config.setter
    def config(self, new_config: QuoteConfig) -> None:
        self._config = new_config

    @property
    def state(self) -> QuoteState:
        return self._index.quote_state

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, balance_cents: int = 0) -> None:
        """Start the quoting loop."""
        if self._running:
            return
        self._running = True
        self._starting_balance_cents = balance_cents
        self._current_balance_cents = balance_cents
        # Clear any stale quote tracking from previous sessions
        self._index.clear_all_quotes()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("[QUOTE_ENGINE] Started quoting loop")

    async def stop(self) -> None:
        """Stop the quoting loop and pull all quotes."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Pull all quotes on shutdown
        await self._pull_all_quotes("shutdown")
        logger.info("[QUOTE_ENGINE] Stopped")

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Periodic requote loop."""
        while self._running:
            try:
                if self._config.enabled and not self.state.quotes_pulled:
                    await self._requote_cycle()
                await asyncio.sleep(self._config.refresh_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[QUOTE_ENGINE] Error in requote cycle: {e}", exc_info=True)
                await asyncio.sleep(self._config.refresh_interval)

    async def _requote_cycle(self) -> None:
        """Single requote pass across all markets."""
        self.state.total_requote_cycles += 1
        self.state.last_requote_at = time.time()

        # Check drawdown circuit breaker
        if self._check_drawdown():
            await self._pull_all_quotes("drawdown_breaker")
            return

        # Check fill storm
        self._check_fill_storm()

        active_count = 0

        for event_ticker, event in self._index.events.items():
            # Check event-level exposure
            event_exposure = self._index.total_event_exposure(event_ticker)
            exposure_ok = inv_mgr.check_event_exposure(
                event_exposure, self._config.max_event_exposure
            )

            for market_ticker, market in event.markets.items():
                if market.status not in ("open", "active"):
                    continue
                if not market.has_data:
                    continue

                # Compute fair value
                fair_val = fv_module.estimate_fair_value(market, event)
                if fair_val is None:
                    continue
                self._index.set_fair_value(market_ticker, fair_val)

                # Compute spread
                spread = self._compute_spread(fair_val, market)

                # Get per-market config (with overrides)
                max_pos = self._config.get_market_config(
                    market_ticker, "max_position", self._config.max_position
                )
                quote_size = self._config.get_market_config(
                    market_ticker, "quote_size", self._config.quote_size
                )

                # Compute skew
                inv = self._index.get_inventory(market_ticker)
                skew = inv_mgr.compute_skew(
                    inv.position, self._config.skew_factor, self._config.skew_cap_cents
                )

                # Compute quote prices
                bid_price = int(round(fair_val - spread / 2 + skew))
                ask_price = int(round(fair_val + spread / 2 + skew))

                # Clamp to valid range
                bid_price = max(1, min(99, bid_price))
                ask_price = max(1, min(99, ask_price))

                # Ensure bid < ask
                if bid_price >= ask_price:
                    if fair_val <= 50:
                        ask_price = min(99, bid_price + 1)
                    else:
                        bid_price = max(1, ask_price - 1)

                # Risk gates
                vpin_ok = market.micro.vpin < self._config.pull_quotes_threshold
                if not vpin_ok:
                    await self._cancel_market_quotes(market_ticker, "vpin_threshold")
                    continue

                if not exposure_ok:
                    # Still allow closing quotes
                    one_side = inv_mgr.should_one_side_only(inv.position, max_pos)
                    if one_side is None:
                        await self._cancel_market_quotes(market_ticker, "event_exposure")
                        continue

                # Position limit - determine which sides to quote
                one_side = inv_mgr.should_one_side_only(inv.position, max_pos)

                # Place/replace quotes
                current = self._index.get_quotes(market_ticker)

                # Bid side
                if one_side != "ask":  # Quote bid unless we should only quote ask
                    bid_ok = inv_mgr.check_position_limit(inv.position, "bid", max_pos)
                    if bid_ok:
                        await self._update_quote(
                            market_ticker, "bid", "yes", "buy",
                            bid_price, quote_size, current.get("bid"),
                        )
                        active_count += 1
                    else:
                        await self._cancel_side(market_ticker, "bid", current.get("bid"))
                else:
                    await self._cancel_side(market_ticker, "bid", current.get("bid"))

                # Ask side
                if one_side != "bid":  # Quote ask unless we should only quote bid
                    ask_ok = inv_mgr.check_position_limit(inv.position, "ask", max_pos)
                    if ask_ok:
                        # Ask = sell YES = buy NO at (100 - ask_price)
                        await self._update_quote(
                            market_ticker, "ask", "no", "buy",
                            100 - ask_price, quote_size, current.get("ask"),
                        )
                        active_count += 1
                    else:
                        await self._cancel_side(market_ticker, "ask", current.get("ask"))
                else:
                    await self._cancel_side(market_ticker, "ask", current.get("ask"))

        self.state.active_quotes = active_count

        # Broadcast requote cycle summary
        if self._ws_broadcast:
            try:
                await self._ws_broadcast("mm_requote_cycle", {
                    "cycle": self.state.total_requote_cycles,
                    "active_quotes": active_count,
                    "spread_multiplier": self.state.spread_multiplier,
                    "quotes_pulled": self.state.quotes_pulled,
                    "pull_reason": self.state.pull_reason,
                    "fees_paid_cents": self.state.fees_paid_cents,
                    "timestamp": time.time(),
                })
            except Exception:
                pass
            # Broadcast full snapshot so frontend stays current
            try:
                snapshot = self._index.get_full_snapshot()
                await self._ws_broadcast("mm_snapshot", snapshot)
            except Exception as e:
                logger.warning(f"[QUOTE_ENGINE] Snapshot broadcast failed: {e}")
            # Broadcast flat inventory for the inventory panel
            try:
                inventory_markets = []
                for ticker in self._index.market_tickers:
                    inv = self._index.get_inventory(ticker)
                    snap = self._index.get_market_snapshot(ticker)
                    quotes = self._index.get_quotes(ticker)
                    event_ticker = self._index.get_event_for_ticker(ticker) or ""
                    mid = snap.yes_bid if snap else None
                    title = (snap.title if snap else ticker)[:60]
                    unrealized = 0.0
                    if inv.position != 0 and mid is not None:
                        if inv.position > 0:
                            unrealized = (mid - inv.avg_entry_cents) * inv.position
                        else:
                            unrealized = (inv.avg_entry_cents - mid) * abs(inv.position)
                    bid_q = quotes.get("bid")
                    ask_q = quotes.get("ask")
                    inventory_markets.append({
                        "ticker": ticker,
                        "event_ticker": event_ticker,
                        "title": title,
                        "position": inv.position,
                        "avg_entry_cents": round(inv.avg_entry_cents, 1),
                        "realized_pnl_cents": round(inv.realized_pnl_cents, 1),
                        "unrealized_pnl_cents": round(unrealized, 1),
                        "total_buys": inv.total_buys,
                        "total_sells": inv.total_sells,
                        "mid_cents": snap.fair_value or mid if snap else None,
                        "bid_quote": {"price": bid_q.price_cents, "size": bid_q.size} if bid_q else None,
                        "ask_quote": {"price": 100 - ask_q.price_cents, "size": ask_q.size} if ask_q else None,
                    })
                await self._ws_broadcast("mm_inventory_update", {
                    "markets": inventory_markets,
                    "timestamp": time.time(),
                })
            except Exception as e:
                logger.warning(f"[QUOTE_ENGINE] Inventory broadcast failed: {e}")

    # ------------------------------------------------------------------
    # Quote Placement
    # ------------------------------------------------------------------

    async def _update_quote(
        self,
        market_ticker: str,
        quote_side: str,  # "bid" or "ask"
        order_side: str,  # "yes" or "no"
        action: str,      # "buy" or "sell"
        price_cents: int,
        size: int,
        existing: Optional[ActiveQuote],
    ) -> None:
        """Place or replace a quote. Cancel+replace (not amend) for MVP."""
        # Check if existing quote is same price/size - no action needed
        if existing and existing.price_cents == price_cents and existing.size == size:
            return

        # Cancel existing if present
        if existing and existing.order_id:
            try:
                await self._gateway.cancel_order(existing.order_id)
                self._index.clear_quote(market_ticker, quote_side)
            except Exception as e:
                # 404 = already expired/filled — expected for TTL orders
                if "404" in str(e) or "not_found" in str(e):
                    logger.debug(f"[QUOTE_ENGINE] Cancel 404 (expired): {existing.order_id}")
                else:
                    logger.warning(f"[QUOTE_ENGINE] Cancel failed for {existing.order_id}: {e}")
                self._index.clear_quote(market_ticker, quote_side)

        # Place new quote
        try:
            expiration_ts = int(time.time()) + self._order_ttl if self._order_ttl > 0 else None
            resp = await self._gateway.create_order(
                ticker=market_ticker,
                action=action,
                side=order_side,
                count=size,
                price=price_cents,
                expiration_ts=expiration_ts,
            )

            order_id = getattr(resp, 'order', None) and resp.order.order_id or ""
            new_quote = ActiveQuote(
                order_id=order_id,
                side=order_side,
                action=action,
                price_cents=price_cents,
                size=size,
                placed_at=time.time(),
            )
            self._index.set_quote(market_ticker, quote_side, new_quote)

            if self._ws_broadcast:
                try:
                    await self._ws_broadcast("mm_quote_placed", {
                        "market_ticker": market_ticker,
                        "quote_side": quote_side,
                        "price_cents": price_cents,
                        "size": size,
                        "order_id": order_id,
                    })
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"[QUOTE_ENGINE] Place quote failed for {market_ticker} {quote_side}: {e}")

    async def _cancel_side(
        self, market_ticker: str, quote_side: str, existing: Optional[ActiveQuote]
    ) -> None:
        """Cancel a specific side quote if it exists."""
        if not existing or not existing.order_id:
            return
        try:
            await self._gateway.cancel_order(existing.order_id)
            self._index.clear_quote(market_ticker, quote_side)
        except Exception as e:
            if "404" in str(e) or "not_found" in str(e):
                logger.debug(f"[QUOTE_ENGINE] Cancel 404 (expired): {existing.order_id}")
            else:
                logger.warning(f"[QUOTE_ENGINE] Cancel failed for {existing.order_id}: {e}")
            self._index.clear_quote(market_ticker, quote_side)

    async def _cancel_market_quotes(self, market_ticker: str, reason: str) -> None:
        """Cancel all quotes for a market."""
        quotes = self._index.get_quotes(market_ticker)
        for side, q in list(quotes.items()):
            if q.order_id:
                try:
                    await self._gateway.cancel_order(q.order_id)
                except Exception:
                    pass
        self._index.clear_all_quotes(market_ticker)

    # ------------------------------------------------------------------
    # Pull / Resume
    # ------------------------------------------------------------------

    async def pull_all_quotes(self, reason: str = "manual") -> int:
        """Public method to pull all quotes (called by Admiral tool)."""
        return await self._pull_all_quotes(reason)

    async def _pull_all_quotes(self, reason: str) -> int:
        """Cancel all active quotes across all markets."""
        order_ids = self._index.get_all_active_order_ids()
        cancelled = 0

        # Batch cancel if possible
        if order_ids:
            try:
                for i in range(0, len(order_ids), 20):
                    batch = order_ids[i:i + 20]
                    await self._gateway.batch_cancel_orders(batch)
                    cancelled += len(batch)
            except Exception:
                # Fall back to individual cancels
                for oid in order_ids:
                    try:
                        await self._gateway.cancel_order(oid)
                        cancelled += 1
                    except Exception:
                        pass

        # Clear all quote tracking
        self._index.clear_all_quotes()
        self.state.quotes_pulled = True
        self.state.pull_reason = reason
        self.state.active_quotes = 0

        logger.info(f"[QUOTE_ENGINE] Pulled {cancelled} quotes (reason: {reason})")

        if self._ws_broadcast:
            try:
                await self._ws_broadcast("mm_quotes_pulled", {
                    "cancelled": cancelled,
                    "reason": reason,
                })
            except Exception:
                pass

        return cancelled

    def resume_quotes(self) -> None:
        """Resume quoting after a pull."""
        self.state.quotes_pulled = False
        self.state.pull_reason = ""
        self.state.spread_multiplier = 1.0
        logger.info("[QUOTE_ENGINE] Quotes resumed")

    # ------------------------------------------------------------------
    # Fill Handling
    # ------------------------------------------------------------------

    def on_fill(
        self, market_ticker: str, side: str, action: str, price_cents: int, quantity: int
    ) -> None:
        """Handle a fill notification. Updates inventory and telemetry."""
        # Update inventory
        self._index.record_fill(market_ticker, side, action, price_cents, quantity)

        # Update telemetry
        # Determine if this was our bid or ask fill
        quotes = self._index.get_quotes(market_ticker)
        if side == "yes" and action == "buy":
            self.state.total_fills_bid += 1
        elif side == "no" and action == "buy":
            self.state.total_fills_ask += 1

        # Track fees
        fee = fee_calculator.maker_fee(price_cents)
        self.state.fees_paid_cents += fee * quantity

        # Track fill for storm detection
        self._recent_fills.append(time.time())

        # Clear the filled quote
        if side == "yes" and action == "buy":
            self._index.clear_quote(market_ticker, "bid")
        elif side == "no" and action == "buy":
            self._index.clear_quote(market_ticker, "ask")

        logger.info(
            f"[QUOTE_ENGINE:FILL] {market_ticker} {side} {action} "
            f"{quantity}@{price_cents}c (fee: {fee:.2f}c)"
        )

    def update_balance(self, balance_cents: int) -> None:
        """Update tracked balance (from portfolio refresh)."""
        self._current_balance_cents = balance_cents
        if self._starting_balance_cents is None:
            self._starting_balance_cents = balance_cents

    # ------------------------------------------------------------------
    # Spread Computation
    # ------------------------------------------------------------------

    def _compute_spread(self, fair_value: float, market) -> float:
        """Compute effective spread considering base, break-even, and volatility."""
        base = self._config.base_spread_cents

        # Break-even spread (must cover both-side maker fees)
        be_spread = fee_calculator.break_even_spread(fair_value)

        # Volatility adjustment: widen spread when VPIN is elevated
        vpin = market.micro.vpin
        vol_adj = 0.0
        if vpin > 0.5:
            vol_adj = (vpin - 0.5) * 4  # 0-2c extra for VPIN 0.5-1.0

        # Fill storm multiplier
        spread = max(base, be_spread + 1, base + vol_adj) * self.state.spread_multiplier

        return spread

    # ------------------------------------------------------------------
    # Risk Gates
    # ------------------------------------------------------------------

    def _check_drawdown(self) -> bool:
        """Check if drawdown exceeds limit. Returns True if breaker should trip."""
        if self._starting_balance_cents is None or self._starting_balance_cents == 0:
            return False

        total_pnl = self._index.total_realized_pnl() + self._index.total_unrealized_pnl()
        if total_pnl < -self._max_drawdown_cents:
            logger.warning(
                f"[QUOTE_ENGINE:DRAWDOWN] P&L {total_pnl}c exceeds "
                f"max drawdown {self._max_drawdown_cents}c"
            )
            return True
        return False

    def _check_fill_storm(self) -> None:
        """Detect fill storm and widen spreads if needed."""
        now = time.time()
        window = self._config.fill_storm_window
        threshold = self._config.fill_storm_threshold

        recent = sum(1 for t in self._recent_fills if now - t < window)
        if recent >= threshold:
            if not self.state.fill_storm_active:
                logger.warning(
                    f"[QUOTE_ENGINE:FILL_STORM] {recent} fills in {window}s, widening spreads 2x"
                )
                self.state.fill_storm_active = True
                self.state.spread_multiplier = 2.0
        else:
            if self.state.fill_storm_active:
                logger.info("[QUOTE_ENGINE:FILL_STORM] Storm subsided, normalizing spreads")
                self.state.fill_storm_active = False
                self.state.spread_multiplier = 1.0
