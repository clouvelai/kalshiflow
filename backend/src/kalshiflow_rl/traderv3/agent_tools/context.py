"""Tool context classes - testable logic backing module-level @tool functions.

Each context class encapsulates all dependencies for a tool category.
Module-level @tool functions delegate to a single context instance,
reducing 10+ globals to 1 per tool file.
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..gateway.client import KalshiGateway
    from ..single_arb.index import EventArbIndex
    from ..single_arb.memory.file_store import FileMemoryStore
    from ..single_arb.event_understanding import UnderstandingBuilder
    from ..core.event_bus import EventBus
    from .session import TradingSession

logger = logging.getLogger("kalshiflow_rl.traderv3.agent_tools.context")


class CaptainToolContext:
    """Context for Captain observation tools.

    Provides read-only market data methods. Does not place orders.
    """

    def __init__(
        self,
        gateway: "KalshiGateway",
        index: "EventArbIndex",
        file_store: "FileMemoryStore",
        session: "TradingSession",
        understanding_builder: Optional["UnderstandingBuilder"] = None,
        broadcast_callback=None,
    ):
        self.gateway = gateway
        self.index = index
        self.file_store = file_store
        self.session = session
        self.understanding_builder = understanding_builder
        self.broadcast_callback = broadcast_callback

    async def get_events_summary(self) -> List[Dict[str, Any]]:
        """Compact summary of all monitored events."""
        fee = self.index._fee_per_contract
        summary = []

        for et, event in self.index.events.items():
            sum_bid = event.market_sum_bid()
            sum_ask = event.market_sum_ask()
            long_e = event.long_edge(fee)
            short_e = event.short_edge(fee)

            total_vol5m = sum(m.micro.volume_5m for m in event.markets.values())
            total_whale = sum(m.micro.whale_trade_count for m in event.markets.values())

            item: Dict[str, Any] = {
                "event_ticker": et,
                "title": event.title,
                "market_count": event.markets_total,
                "markets_with_data": event.markets_with_data,
                "all_markets_have_data": event.all_markets_have_data,
                "sum_yes_bid": sum_bid,
                "sum_yes_ask": sum_ask,
                "long_edge": round(long_e, 1) if long_e is not None else None,
                "short_edge": round(short_e, 1) if short_e is not None else None,
                "volume_5m": total_vol5m,
                "whale_trades": total_whale,
            }

            # Mentions data
            mentions_data = event.mentions_data
            if mentions_data and mentions_data.get("lexeme_pack"):
                lexeme_pack = mentions_data["lexeme_pack"]
                entity = lexeme_pack.get("entity", "")
                item["is_mentions_market"] = True
                item["mentions_entity"] = entity

                baseline = mentions_data.get("baseline_estimates", {})
                current = mentions_data.get("current_estimates", {})

                if entity and entity in baseline:
                    item["baseline_probability"] = round(
                        baseline[entity].get("probability", 0.0), 3
                    )
                else:
                    item["baseline_probability"] = None

                if entity and entity in current:
                    item["current_probability"] = round(
                        current[entity].get("probability", 0.0), 3
                    )
                else:
                    item["current_probability"] = None

                last_sim = mentions_data.get("last_simulation_ts", 0)
                item["simulation_stale"] = (time.time() - last_sim) > 300
                item["has_baseline"] = bool(baseline)

            summary.append(item)

        return summary

    async def get_event_snapshot(self, event_ticker: str) -> Dict[str, Any]:
        """Full snapshot for a specific event."""
        snapshot = self.index.get_event_snapshot(event_ticker)
        if not snapshot:
            return {"error": f"Event {event_ticker} not found in index"}

        # Trim noisy fields
        keep = {
            "ticker", "title", "yes_bid", "yes_ask", "yes_bid_size", "yes_ask_size",
            "yes_levels", "no_levels", "spread", "freshness_seconds",
            "last_trade_price", "last_trade_side", "trade_count", "micro",
        }
        if "markets" in snapshot:
            trimmed = {}
            for ticker, mkt in snapshot["markets"].items():
                trimmed[ticker] = {k: v for k, v in mkt.items() if k in keep}
            snapshot["markets"] = trimmed

        for key in ("subtitle", "loaded_at"):
            snapshot.pop(key, None)

        return snapshot

    async def get_market_orderbook(self, market_ticker: str) -> Dict[str, Any]:
        """Full orderbook depth for a single market."""
        event_ticker = self.index.get_event_for_ticker(market_ticker)
        if not event_ticker:
            return {"error": f"Market {market_ticker} not tracked"}

        event = self.index.events.get(event_ticker)
        if not event:
            return {"error": f"Event {event_ticker} not found"}

        market = event.markets.get(market_ticker)
        if not market:
            return {"error": f"Market {market_ticker} not found in event"}

        return {
            "ticker": market.ticker,
            "title": market.title,
            "yes_levels": market.yes_levels,
            "no_levels": market.no_levels,
            "yes_bid": market.yes_bid,
            "yes_ask": market.yes_ask,
            "yes_bid_size": market.yes_bid_size,
            "yes_ask_size": market.yes_ask_size,
            "spread": market.spread,
            "source": market.source,
            "freshness_seconds": round(market.freshness_seconds, 1),
        }

    async def get_trade_history(
        self, ticker: Optional[str] = None, limit: int = 20
    ) -> Dict[str, Any]:
        """Fills, settlements, P&L via gateway."""
        try:
            fills = await self.gateway.get_fills(ticker=ticker, limit=limit)
            fill_dicts = [
                {
                    "ticker": f.ticker,
                    "side": f.side,
                    "action": f.action,
                    "count": f.count,
                    "yes_price": f.yes_price,
                    "no_price": f.no_price,
                    "order_id": f.order_id,
                    "created_time": f.created_time,
                }
                for f in fills
            ]

            settlements = await self.gateway.get_settlements(limit=50)
            settlement_dicts = [
                {
                    "ticker": s.ticker,
                    "market_result": s.market_result,
                    "revenue": s.revenue,
                    "payout": s.payout,
                    "settled_time": s.settled_time,
                }
                for s in settlements[:20]
            ]

            total_revenue = sum(s.revenue for s in settlements)
            total_payout = sum(s.payout for s in settlements)

            return {
                "fills": fill_dicts,
                "fill_count": len(fill_dicts),
                "settlements": settlement_dicts,
                "settlement_count": len(settlement_dicts),
                "pnl_summary": {
                    "total_revenue_cents": total_revenue,
                    "total_payout_cents": total_payout,
                    "net_pnl_cents": total_payout - total_revenue,
                },
            }
        except Exception as e:
            return {"error": f"Trade history failed: {e}"}

    async def get_positions(self) -> Dict[str, Any]:
        """Positions with realtime P&L from live orderbook."""
        try:
            all_positions = await self.gateway.get_positions()
            tracked = set(self.index.market_tickers) if self.index else set()

            positions = []
            for pos in all_positions:
                if pos.ticker not in tracked:
                    continue

                position_count = pos.position
                qty = abs(position_count)
                side = "yes" if position_count > 0 else "no"

                cost = pos.total_traded
                realized_pnl = pos.realized_pnl
                fees = pos.fees_paid

                current_value = 0
                exit_price = None
                event_ticker = self.index.get_event_for_ticker(pos.ticker)

                if event_ticker and qty > 0:
                    event = self.index.events.get(event_ticker)
                    if event:
                        market = event.markets.get(pos.ticker)
                        if market:
                            if side == "yes":
                                exit_price = market.yes_bid
                            else:
                                exit_price = 100 - market.yes_ask if market.yes_ask else None
                            if exit_price:
                                current_value = exit_price * qty

                unrealized_pnl = current_value - cost if cost > 0 else 0

                positions.append({
                    "ticker": pos.ticker,
                    "event_ticker": event_ticker,
                    "side": side,
                    "quantity": qty,
                    "cost": cost,
                    "exit_price": exit_price,
                    "current_value": current_value,
                    "unrealized_pnl": unrealized_pnl,
                    "realized_pnl": realized_pnl,
                    "fees_paid": fees,
                })

            positions.sort(key=lambda p: p["unrealized_pnl"], reverse=True)

            return {
                "positions": positions,
                "count": len(positions),
                "total_cost": sum(p["cost"] for p in positions),
                "total_value": sum(p["current_value"] for p in positions),
                "total_unrealized_pnl": sum(p["unrealized_pnl"] for p in positions),
            }
        except Exception as e:
            return {"error": f"Get positions failed: {e}"}

    async def get_balance(self) -> Dict[str, Any]:
        """Account balance."""
        try:
            bal = await self.gateway.get_balance()
            return {
                "balance_cents": bal.balance,
                "balance_dollars": round(bal.balance / 100, 2),
            }
        except Exception as e:
            return {"error": str(e)}

    async def update_understanding(
        self, event_ticker: str, force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Rebuild structured understanding for an event."""
        if not self.understanding_builder:
            return {"error": "UnderstandingBuilder not available"}

        event = self.index.events.get(event_ticker)
        if not event:
            return {"error": f"Event {event_ticker} not found"}

        try:
            understanding = await self.understanding_builder.build(
                event, force_refresh=force_refresh
            )
            event.understanding = understanding.to_dict()

            if self.broadcast_callback:
                try:
                    await self.broadcast_callback({
                        "type": "event_understanding_update",
                        "data": {
                            "event_ticker": event_ticker,
                            "understanding": understanding.to_dict(),
                        },
                    })
                except Exception:
                    pass

            return {
                "status": "updated",
                "event_ticker": event_ticker,
                "trading_summary": understanding.trading_summary,
                "key_factors": understanding.key_factors,
                "participants": len(understanding.participants),
                "extensions": list(understanding.extensions.keys()),
                "version": understanding.version,
                "stale": understanding.stale,
            }
        except Exception as e:
            return {"error": f"Understanding build failed: {e}"}


class CommandoToolContext:
    """Context for TradeCommando execution tools.

    Places orders, manages positions, records learnings.
    """

    def __init__(
        self,
        gateway: "KalshiGateway",
        index: "EventArbIndex",
        file_store: "FileMemoryStore",
        session: "TradingSession",
        event_bus: Optional["EventBus"] = None,
        broadcast_callback=None,
    ):
        self.gateway = gateway
        self.index = index
        self.file_store = file_store
        self.session = session
        self.event_bus = event_bus
        self.broadcast_callback = broadcast_callback

    async def place_order(
        self,
        ticker: str,
        side: str,
        contracts: int,
        price_cents: int,
        reasoning: str,
        action: str = "buy",
    ) -> Dict[str, Any]:
        """Place a single order via gateway."""
        if side not in ("yes", "no"):
            return {"error": f"Invalid side: {side}"}
        if action not in ("buy", "sell"):
            return {"error": f"Invalid action: {action}"}
        if not (1 <= contracts <= 100):
            return {"error": f"Contracts must be 1-100, got {contracts}"}
        if not (1 <= price_cents <= 99):
            return {"error": f"Price must be 1-99 cents, got {price_cents}"}

        try:
            expiration_ts = int(time.time()) + self.session.order_ttl

            resp = await self.gateway.create_order(
                ticker=ticker,
                action=action,
                side=side,
                count=contracts,
                price=price_cents,
                order_group_id=self.session.order_group_id or None,
                expiration_ts=expiration_ts,
            )

            order = resp.order
            order_id = order.order_id
            status = order.status

            if order_id:
                self.session.captain_order_ids.add(order_id)

            # Record to memory
            self._record_trade(
                f"TRADE: {action} {contracts} {side} {ticker} @{price_cents}c | {reasoning}",
                {
                    "order_id": order_id,
                    "ticker": ticker,
                    "side": side,
                    "action": action,
                    "contracts": contracts,
                    "price_cents": price_cents,
                    "status": status,
                },
            )

            # Broadcast to frontend via EventBus
            await self._broadcast_trade_executed(
                order_id=order_id,
                event_ticker=self.index.get_event_for_ticker(ticker) if self.index else None,
                ticker=ticker,
                direction=None,
                side=side,
                action=action,
                contracts=contracts,
                price_cents=price_cents,
                status=status,
                reasoning=reasoning,
            )

            return {
                "order_id": order_id,
                "status": status,
                "ticker": ticker,
                "side": side,
                "action": action,
                "contracts": contracts,
                "price_cents": price_cents,
                "ttl_seconds": self.session.order_ttl,
                "order_group": self.session.order_group_id[:8] if self.session.order_group_id else None,
            }

        except Exception as e:
            self._record_trade(
                f"FAILED ORDER: {action} {contracts} {side} {ticker} @{price_cents}c | {reasoning} | error: {e}",
                {"ticker": ticker, "side": side, "action": action, "status": "failed", "error": str(e)},
            )
            return {"error": f"Order failed: {e}"}

    async def execute_arb(
        self,
        event_ticker: str,
        direction: str,
        max_contracts: int,
        reasoning: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute multi-leg arb trade via gateway."""
        event_state = self.index.events.get(event_ticker)
        if not event_state:
            return {"error": f"Event {event_ticker} not found"}

        if not event_state.all_markets_have_data:
            return {"error": f"Not all markets have data ({event_state.markets_with_data}/{event_state.markets_total})"}

        if direction not in ("long", "short"):
            return {"error": f"Invalid direction: {direction}"}

        try:
            balance = await self.gateway.get_balance()
            balance_cents = balance.balance
        except Exception as e:
            return {"error": f"Balance check failed: {e}"}

        # Build legs
        legs = []
        total_cost = 0
        errors = []

        for book in event_state.markets.values():
            if direction == "long":
                if book.yes_ask is None:
                    errors.append(f"{book.ticker}: no YES ask")
                    continue
                side = "yes"
                price = book.yes_ask
            else:
                if book.yes_bid is None:
                    errors.append(f"{book.ticker}: no YES bid")
                    continue
                side = "no"
                price = 100 - book.yes_bid

            contracts = min(max_contracts, book.yes_ask_size if direction == "long" else book.yes_bid_size)
            contracts = max(contracts, 1)

            leg_cost = contracts * price
            if total_cost + leg_cost > balance_cents:
                errors.append(f"{book.ticker}: insufficient balance")
                continue

            legs.append({
                "ticker": book.ticker,
                "title": book.title,
                "side": side,
                "contracts": contracts,
                "price_cents": price,
            })
            total_cost += leg_cost

        if dry_run:
            return {
                "status": "preview",
                "event_ticker": event_ticker,
                "direction": direction,
                "legs_planned": len(legs),
                "legs_total": event_state.markets_total,
                "estimated_cost_cents": total_cost,
                "balance_cents": balance_cents,
                "balance_after_cents": balance_cents - total_cost,
                "legs": legs,
                "errors": errors,
                "reasoning": reasoning,
            }

        # Execute legs
        legs_executed = []
        exec_cost = 0

        for leg in legs:
            try:
                resp = await self.gateway.create_order(
                    ticker=leg["ticker"],
                    action="buy",
                    side=leg["side"],
                    count=leg["contracts"],
                    price=leg["price_cents"],
                    order_group_id=self.session.order_group_id or None,
                    expiration_ts=int(time.time()) + self.session.order_ttl,
                )
                order = resp.order
                order_id = order.order_id

                if order_id:
                    self.session.captain_order_ids.add(order_id)

                legs_executed.append({
                    **leg,
                    "order_id": order_id,
                    "status": order.status,
                })
                exec_cost += leg["contracts"] * leg["price_cents"]

            except Exception as e:
                errors.append(f"{leg['ticker']}: order failed: {e}")

        status = "completed" if len(legs_executed) == event_state.markets_total else "partial"
        if not legs_executed:
            status = "failed"

        result = {
            "status": status,
            "event_ticker": event_ticker,
            "direction": direction,
            "legs_executed": len(legs_executed),
            "legs_total": event_state.markets_total,
            "total_cost_cents": exec_cost,
            "legs": legs_executed,
            "errors": errors,
            "reasoning": reasoning,
        }

        # Record to memory
        prefix = "ARB" if status != "failed" else "FAILED ARB"
        self._record_trade(
            f"{prefix}: {direction} {event_ticker} | {len(legs_executed)}/{event_state.markets_total} legs | cost={exec_cost}c | {reasoning}",
            {
                "event_ticker": event_ticker,
                "direction": direction,
                "legs_executed": len(legs_executed),
                "total_cost_cents": exec_cost,
                "status": status,
                "order_ids": [leg.get("order_id") for leg in legs_executed],
            },
        )

        # Broadcast each leg
        for leg in legs_executed:
            await self._broadcast_trade_executed(
                order_id=leg.get("order_id", ""),
                event_ticker=event_ticker,
                ticker=leg.get("ticker", ""),
                direction=direction,
                side=leg.get("side", ""),
                action="buy",
                contracts=leg.get("contracts", 0),
                price_cents=leg.get("price_cents", 0),
                status=leg.get("status", ""),
                reasoning=reasoning,
            )

        return result

    async def cancel_order(self, order_id: str, reason: str = "") -> Dict[str, Any]:
        """Cancel a resting order."""
        try:
            await self.gateway.cancel_order(order_id)

            if reason:
                self._record_trade(
                    f"CANCEL: order {order_id[:8]}... | {reason}",
                    {"order_id": order_id, "action": "cancel"},
                )
            return {"status": "cancelled", "order_id": order_id}

        except Exception as e:
            err_str = str(e).lower()
            if "not found" in err_str or "404" in err_str:
                return {"status": "already_gone", "order_id": order_id}
            return {"error": f"Cancel failed: {e}"}

    async def get_resting_orders(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """Get resting orders with queue positions."""
        try:
            orders = await self.gateway.get_orders(ticker=ticker, status="resting")

            queue_map = {}
            if orders:
                try:
                    mt = ticker or orders[0].ticker
                    if mt:
                        qps = await self.gateway.get_queue_positions(market_tickers=mt)
                        for qp in qps:
                            queue_map[qp.order_id] = qp.queue_position
                except Exception:
                    pass

            order_dicts = [
                {
                    "order_id": o.order_id,
                    "ticker": o.ticker,
                    "side": o.side,
                    "action": o.action,
                    "price": o.price or o.yes_price,
                    "remaining_count": o.remaining_count,
                    "created_time": o.created_time,
                    "expiration_time": o.expiration_time,
                    "queue_position": queue_map.get(o.order_id),
                }
                for o in orders
            ]

            return {"count": len(order_dicts), "orders": order_dicts}

        except Exception as e:
            return {"error": f"Get orders failed: {e}"}

    async def get_recent_trades(self, event_ticker: str) -> Dict[str, Any]:
        """Recent public trades across all markets in an event."""
        event = self.index.events.get(event_ticker)
        if not event:
            return {"error": f"Event {event_ticker} not found"}

        all_trades = []
        total_count = 0
        for m in event.markets.values():
            total_count += m.trade_count
            for t in m.recent_trades[:5]:
                all_trades.append({
                    "ticker": m.ticker,
                    "price": t.get("yes_price"),
                    "count": t.get("count", 1),
                    "side": t.get("taker_side"),
                    "ts": t.get("ts"),
                })

        all_trades.sort(key=lambda t: t.get("ts", 0), reverse=True)

        most_active = event.most_active_market()
        return {
            "event_ticker": event_ticker,
            "trades": all_trades[:15],
            "trade_count_total": total_count,
            "most_active_ticker": most_active.ticker if most_active else None,
        }

    async def get_market_orderbook(self, market_ticker: str) -> Dict[str, Any]:
        """Orderbook depth - delegates to same index logic."""
        event_ticker = self.index.get_event_for_ticker(market_ticker)
        if not event_ticker:
            return {"error": f"Market {market_ticker} not tracked"}

        event = self.index.events.get(event_ticker)
        if not event:
            return {"error": f"Event {event_ticker} not found"}

        market = event.markets.get(market_ticker)
        if not market:
            return {"error": f"Market {market_ticker} not found in event"}

        return {
            "ticker": market.ticker,
            "title": market.title,
            "yes_levels": market.yes_levels,
            "no_levels": market.no_levels,
            "yes_bid": market.yes_bid,
            "yes_ask": market.yes_ask,
            "yes_bid_size": market.yes_bid_size,
            "yes_ask_size": market.yes_ask_size,
            "spread": market.spread,
            "source": market.source,
            "freshness_seconds": round(market.freshness_seconds, 1),
        }

    async def get_balance(self) -> Dict[str, Any]:
        """Account balance."""
        try:
            bal = await self.gateway.get_balance()
            return {
                "balance_cents": bal.balance,
                "balance_dollars": round(bal.balance / 100, 2),
            }
        except Exception as e:
            return {"error": str(e)}

    async def get_positions(self) -> Dict[str, Any]:
        """Same as CaptainToolContext.get_positions - needed by commando."""
        try:
            all_positions = await self.gateway.get_positions()
            tracked = set(self.index.market_tickers) if self.index else set()

            positions = []
            for pos in all_positions:
                if pos.ticker not in tracked:
                    continue

                position_count = pos.position
                qty = abs(position_count)
                side = "yes" if position_count > 0 else "no"
                cost = pos.total_traded

                current_value = 0
                exit_price = None
                event_ticker = self.index.get_event_for_ticker(pos.ticker)

                if event_ticker and qty > 0:
                    event = self.index.events.get(event_ticker)
                    if event:
                        market = event.markets.get(pos.ticker)
                        if market:
                            if side == "yes":
                                exit_price = market.yes_bid
                            else:
                                exit_price = 100 - market.yes_ask if market.yes_ask else None
                            if exit_price:
                                current_value = exit_price * qty

                unrealized_pnl = current_value - cost if cost > 0 else 0

                positions.append({
                    "ticker": pos.ticker,
                    "event_ticker": event_ticker,
                    "side": side,
                    "quantity": qty,
                    "cost": cost,
                    "exit_price": exit_price,
                    "current_value": current_value,
                    "unrealized_pnl": unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                    "fees_paid": pos.fees_paid,
                })

            positions.sort(key=lambda p: p["unrealized_pnl"], reverse=True)

            return {
                "positions": positions,
                "count": len(positions),
                "total_cost": sum(p["cost"] for p in positions),
                "total_value": sum(p["current_value"] for p in positions),
                "total_unrealized_pnl": sum(p["unrealized_pnl"] for p in positions),
            }
        except Exception as e:
            return {"error": f"Get positions failed: {e}"}

    def record_learning(
        self, content: str, category: str = "learning", target_file: str = "AGENTS.md"
    ) -> Dict[str, Any]:
        """Record a learning to journal."""
        try:
            self.file_store.append(
                content=content,
                memory_type=category,
                metadata={"target_file": target_file},
            )
            return {"status": "stored", "category": category, "target_file": target_file}
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_trade(self, content: str, metadata: Dict) -> None:
        """Record trade to journal (best-effort)."""
        try:
            self.file_store.append(content=content, memory_type="trade", metadata=metadata)
        except Exception:
            pass

    async def _broadcast_trade_executed(self, **kwargs) -> None:
        """Broadcast trade executed event to frontend."""
        if not self.broadcast_callback:
            return
        try:
            await self.broadcast_callback({
                "type": "arb_trade_executed",
                "data": {
                    "order_id": kwargs.get("order_id", ""),
                    "event_ticker": kwargs.get("event_ticker"),
                    "kalshi_ticker": kwargs.get("ticker", ""),
                    "direction": kwargs.get("direction"),
                    "side": kwargs.get("side", ""),
                    "action": kwargs.get("action", "buy"),
                    "contracts": kwargs.get("contracts", 0),
                    "price_cents": kwargs.get("price_cents", 0),
                    "status": kwargs.get("status", ""),
                    "reasoning": kwargs.get("reasoning", ""),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            })
        except Exception:
            pass


class MentionsToolContext:
    """Context for MentionsSpecialist tools.

    Wraps the existing mentions simulator and index for clean access.
    """

    def __init__(
        self,
        index: "EventArbIndex",
        file_store: "FileMemoryStore",
        config: Any = None,
        mentions_data_dir: Optional[str] = None,
        broadcast_callback=None,
    ):
        self.index = index
        self.file_store = file_store
        self.config = config
        self.mentions_data_dir = mentions_data_dir
        self.broadcast_callback = broadcast_callback
