"""Context builder for Captain V2.

Serializes live EventArbIndex state into Pydantic models.
Zero API calls for market data - reads directly from in-memory index.
Only build_portfolio_state() requires a single API call (balance + positions).
"""

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

from .models import (
    AttentionItem,
    CycleDiff,
    EventSemantics,
    EventSnapshot,
    MarketSnapshot,
    MarketState,
    PortfolioState,
    Position,
    SniperStatus,
)

if TYPE_CHECKING:
    from .index import EventArbIndex, EventMeta, MarketMeta
    from .sniper import Sniper

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.context_builder")

# Token budget constants (approximate: 1 token ~ 4 chars)
REACTIVE_TOKEN_BUDGET = 4000
STRATEGIC_TOKEN_BUDGET = 8000
DEEP_SCAN_TOKEN_BUDGET = 16000


def _estimate_tokens(text: str) -> int:
    """Approximate token count. ~4 chars per token for English text."""
    return len(text) // 4


class ContextBudget:
    """Token-budgeted section accumulator.

    Replaces the repeated _add() closure in each build_*_context method.
    Tracks token usage and drops sections that would exceed the budget.
    """

    def __init__(self, budget: int):
        self.budget = budget
        self.parts: list[str] = []
        self.tokens_used: int = 0
        self.dropped: list[str] = []

    def add(self, section_name: str, text: str) -> bool:
        """Add section if within budget. Returns True if added."""
        cost = _estimate_tokens(text)
        if self.tokens_used + cost > self.budget * 0.9:
            logger.debug(f"Context truncated: {section_name} omitted, {self.tokens_used}/{self.budget} tokens")
            self.dropped.append(section_name)
            return False
        self.parts.append(text)
        self.tokens_used += cost
        return True

    def build(self, suffix: str = "") -> str:
        """Join all parts, appending truncation warning and suffix."""
        if self.dropped:
            self.parts.append(
                f"\nCONTEXT TRUNCATED: Omitted sections: {self.dropped} due to token budget. "
                "Data may be incomplete -- call get_market_state for fresh data if needed."
            )
        if suffix:
            self.parts.append(suffix)
        return "\n".join(self.parts)


def _fmt_balance(portfolio: "PortfolioState", subaccount: int = 0) -> str:
    """Format balance for prompt with subaccount-aware diagnostics."""
    if portfolio.balance_dollars is None:
        if subaccount > 0:
            return f"$ERROR (subaccount #{subaccount} balance fetch failed — verify it exists via /portfolio/subaccounts/balances)"
        return "$unavailable"
    if portfolio.balance_cents == 0 and subaccount > 0:
        return f"$0.00 (WARNING: subaccount #{subaccount} has zero balance — transfer funds)"
    return f"${portfolio.balance_dollars:,.2f}"


def _market_snapshot(m: "MarketMeta") -> MarketSnapshot:
    """Build MarketSnapshot from a live MarketMeta. Pure Python, no I/O."""
    return MarketSnapshot(
        ticker=m.ticker,
        title=m.title,
        yes_bid=m.yes_bid,
        yes_ask=m.yes_ask,
        yes_bid_size=m.yes_bid_size,
        yes_ask_size=m.yes_ask_size,
        spread=m.spread,
        microprice=round(m.microprice, 2) if m.microprice is not None else None,
        vpin=round(m.micro.vpin, 3),
        book_imbalance=round(m.micro.book_imbalance, 3),
        volume_5m=m.micro.volume_5m,
        whale_trade_count=m.micro.whale_trade_count,
        trade_count=m.trade_count,
        freshness_seconds=round(m.freshness_seconds, 1),
        last_price=m.last_price,
        regime=_market_regime(m),
    )


def _market_regime(m: "MarketMeta") -> str:
    """Detect per-market regime from microstructure signals. Pure Python."""
    if m.micro.vpin > 0.98:
        return "toxic"
    if m.micro.sweep_active:
        return "sweep"
    if m.micro.total_bid_depth + m.micro.total_ask_depth < 5:
        return "thin"
    return "normal"


def _compute_regime(event: "EventMeta") -> str:
    """Detect event-level regime (worst-case across markets). Pure Python."""
    for m in event.markets.values():
        if m.micro.vpin > 0.98:
            return "toxic"
        if m.micro.sweep_active:
            return "sweep"
    total_depth = sum(
        m.micro.total_bid_depth + m.micro.total_ask_depth
        for m in event.markets.values()
    )
    if total_depth < 20:
        return "thin"
    return "normal"


def _time_to_close(event: "EventMeta") -> Optional[float]:
    """Compute hours until earliest market close. Pure Python."""
    now = time.time()
    closest = None
    for m in event.markets.values():
        if not m.close_time:
            continue
        try:
            ct = m.close_time.replace("Z", "+00:00")
            close_dt = datetime.fromisoformat(ct)
            hours = (close_dt.timestamp() - now) / 3600
            if closest is None or hours < closest:
                closest = hours
        except (ValueError, TypeError):
            continue
    return round(closest, 1) if closest is not None else None


def _event_semantics(event: "EventMeta") -> Optional[EventSemantics]:
    """Extract EventSemantics from EventUnderstanding dict, if available."""
    u = event.understanding
    if not u:
        return None
    if not isinstance(u, dict):
        return None
    participants = u.get("participants", [])
    who = [p.get("name", "") for p in participants if isinstance(p, dict) and p.get("name")]
    key_factors = u.get("key_factors", [])
    # key_factors can be list of strings or list of dicts with "factor" key
    search_terms = []
    if key_factors:
        for f in key_factors[:3]:
            if isinstance(f, str):
                search_terms.append(f[:50])
            elif isinstance(f, dict):
                search_terms.append(f.get("factor", "")[:50])
    return EventSemantics(
        what=u.get("trading_summary", "")[:200],
        who=who[:5],
        when=u.get("timeline", {}).get("status", "") if isinstance(u.get("timeline"), dict) else "",
        domain=u.get("domain", ""),
        settlement_summary="",
        search_terms=search_terms,
        news_summary=(u.get("news_articles", [{}])[0].get("title", "") if u.get("news_articles") else ""),
        news_fetched_at=u.get("news_fetched_at"),
    )


class ContextBuilder:
    """Builds structured Pydantic context from live in-memory state.

    Used by Captain V2 to inject MarketState, PortfolioState, SniperStatus
    into the cycle prompt as structured JSON.
    """

    def __init__(self, index: "EventArbIndex", subaccount: int = 0):
        self._index = index
        self._subaccount = subaccount
        self._prev_state: Optional[Dict] = None

    def build_market_state(self) -> MarketState:
        """Build complete market state from live index. Zero API calls."""
        index = self._index
        fee = index.fee_per_contract
        events = []
        total_markets = 0

        for et, event in index.events.items():
            markets = {}
            for mt, m in event.markets.items():
                markets[mt] = _market_snapshot(m)
            total_markets += len(markets)

            long_e = event.long_edge(fee)
            short_e = event.short_edge(fee)
            ttc = _time_to_close(event)

            events.append(EventSnapshot(
                event_ticker=et,
                title=event.title,
                category=event.category,
                mutually_exclusive=event.mutually_exclusive,
                market_count=event.markets_total,
                markets=markets,
                long_edge=round(long_e, 1) if long_e is not None else None,
                short_edge=round(short_e, 1) if short_e is not None else None,
                sum_yes_bid=event.market_sum_bid(),
                sum_yes_ask=event.market_sum_ask(),
                total_volume_5m=sum(m.micro.volume_5m for m in event.markets.values()),
                regime=_compute_regime(event),
                time_to_close_hours=ttc,
                semantics=_event_semantics(event),
            ))

        return MarketState(
            events=events,
            total_events=len(events),
            total_markets=total_markets,
        )

    async def build_portfolio_state(self, gateway) -> PortfolioState:
        """Build portfolio state. 1 API call (balance) + 1 (positions).

        Positions are scoped by the gateway's subaccount parameter,
        so no event_ticker filter is needed.
        """
        index = self._index

        try:
            balance_obj = await gateway.get_balance()
            balance_cents = balance_obj.balance
        except Exception as e:
            logger.warning(f"Balance fetch failed: {e}")
            balance_cents = None

        positions = []
        total_pnl = 0
        total_cost = 0
        try:
            raw_positions = await gateway.get_positions()
            for pos in raw_positions:
                ticker = pos.ticker if hasattr(pos, "ticker") else pos.get("ticker")
                position_count = pos.position if hasattr(pos, "position") else pos.get("position", 0)
                if position_count == 0:
                    continue
                qty = abs(position_count)
                side = "yes" if position_count > 0 else "no"
                cost = pos.total_traded if hasattr(pos, "total_traded") else pos.get("total_traded", 0)
                et = index.get_event_for_ticker(ticker)
                exit_price = None
                if et:
                    event = index.events.get(et)
                    if event:
                        market = event.markets.get(ticker)
                        if market:
                            if side == "yes":
                                exit_price = market.yes_bid
                            else:
                                exit_price = 100 - market.yes_ask if market.yes_ask else None
                current_value = exit_price * qty if exit_price else 0
                unrealized_pnl = current_value - cost if cost > 0 else 0
                positions.append(Position(
                    ticker=ticker,
                    event_ticker=et,
                    side=side,
                    quantity=qty,
                    cost_cents=cost,
                    exit_price=exit_price,
                    current_value_cents=current_value,
                    unrealized_pnl_cents=unrealized_pnl,
                ))
                total_pnl += unrealized_pnl
                total_cost += cost
        except Exception as e:
            logger.warning(f"Position fetch failed: {e}")

        return PortfolioState(
            balance_cents=balance_cents or 0,
            balance_dollars=round(balance_cents / 100, 2) if balance_cents is not None else None,
            positions=sorted(positions, key=lambda p: p.unrealized_pnl_cents, reverse=True),
            total_positions=len(positions),
            total_unrealized_pnl_cents=total_pnl,
            total_cost_cents=total_cost,
        )

    def build_sniper_status(self, sniper: Optional["Sniper"]) -> SniperStatus:
        """Build sniper status. Pure Python, reads Sniper state directly."""
        if not sniper:
            return SniperStatus(enabled=False)

        s = sniper.state
        cfg = sniper.config
        last_action = None
        if s.recent_actions:
            last = s.recent_actions[-1]
            last_action = (
                f"{last.event_ticker} {last.direction} "
                f"edge={last.edge_cents:.1f}c legs={last.legs_filled}/{last.legs_attempted} "
                f"latency={last.latency_ms:.0f}ms"
            )

        return SniperStatus(
            enabled=cfg.enabled,
            total_trades=s.total_trades,
            total_arbs_executed=s.total_arbs_executed,
            capital_in_flight=s.capital_active,
            capital_in_positions=0,
            capital_deployed_lifetime=s.capital_deployed_lifetime,
            total_partial_unwinds=s.total_partial_unwinds,
            last_rejection_reason=s.last_rejection_reason,
            last_action_summary=last_action,
            config_subset={
                "max_position": cfg.max_position,
                "max_capital": cfg.max_capital,
                "arb_min_edge": cfg.arb_min_edge,
                "cooldown": cfg.cooldown,
                "max_trades_per_cycle": cfg.max_trades_per_cycle,
            },
        )

    def build_reactive_context(
        self,
        items: List[AttentionItem],
        portfolio: PortfolioState,
        sniper_status: Optional[SniperStatus] = None,
    ) -> str:
        """Build compact prompt for reactive Captain cycles.

        Contains ONLY attention items + relevant positions. ~200-400 tokens.
        """
        cb = ContextBudget(REACTIVE_TOKEN_BUDGET)

        # Attention items (highest priority)
        attn_lines = ["ATTENTION:"]
        for i, item in enumerate(items, 1):
            attn_lines.append(f"  {i}. {item.to_prompt()}")
        cb.add("ATTENTION", "\n".join(attn_lines))

        # Compact portfolio
        if portfolio.positions:
            portfolio_lines = [
                f"PORTFOLIO: balance={_fmt_balance(portfolio, self._subaccount)}, "
                f"{portfolio.total_positions} positions, "
                f"pnl=${portfolio.total_unrealized_pnl_cents / 100:+.2f}"
            ]
            # Only show positions relevant to attention items
            attention_events = {item.event_ticker for item in items}
            relevant = [
                p for p in portfolio.positions
                if p.event_ticker in attention_events
            ]
            if relevant:
                for p in relevant[:5]:
                    pnl_ct = round(p.unrealized_pnl_cents / p.quantity) if p.quantity else 0
                    portfolio_lines.append(
                        f"  {p.ticker} {p.side} {p.quantity}ct "
                        f"exit={p.exit_price} pnl={pnl_ct:+d}c/ct"
                    )
            cb.add("PORTFOLIO", "\n".join(portfolio_lines))
        else:
            cb.add("PORTFOLIO", f"PORTFOLIO: balance={_fmt_balance(portfolio, self._subaccount)}, no positions")

        # Sniper (one line)
        if sniper_status and sniper_status.enabled:
            cb.add("SNIPER",
                f"SNIPER: {sniper_status.total_arbs_executed} arbs, "
                f"capital_active={sniper_status.capital_in_flight + sniper_status.capital_in_positions}c"
            )

        # Sweep detail section
        sweep_items = [i for i in items if i.category == "sweep_detected"]
        if sweep_items:
            sweep_lines = ["SWEEP_DETAIL:"]
            for sw in sweep_items:
                d = sw.data
                detail_parts = [f"  {sw.market_ticker or sw.event_ticker}"]
                if d.get("direction"):
                    detail_parts.append(f"direction={d['direction']}")
                if d.get("size"):
                    detail_parts.append(f"size={d['size']}ct")
                if d.get("levels"):
                    detail_parts.append(f"levels={d['levels']}")
                sweep_lines.append(" ".join(detail_parts))
            cb.add("SWEEP_DETAIL", "\n".join(sweep_lines))

        # After ATTENTION items, if any are early_bird, add detail
        early_bird_items = [i for i in items if i.category == "early_bird"]
        if early_bird_items:
            eb_lines = ["EARLY_BIRD_DETAIL:"]
            for eb in early_bird_items:
                d = eb.data
                detail_parts = [f"  {eb.market_ticker}"]
                if d.get("strategy"):
                    detail_parts.append(f"strategy={d['strategy']}")
                if d.get("fair_value") is not None:
                    detail_parts.append(f"fair_value={d['fair_value']:.0f}c")
                if d.get("early_bird_score"):
                    detail_parts.append(f"score={d['early_bird_score']:.0f}")
                eb_lines.append(" ".join(detail_parts))
            cb.add("EARLY_BIRD_DETAIL", "\n".join(eb_lines))

        return cb.build(
            suffix="ACTION: Respond to attention items above ONLY. Trade, exit, or note why you pass. "
            "Do NOT plan strategy, research unrelated events, or call write_todos in reactive mode."
        )

    def build_strategic_context(
        self,
        portfolio: PortfolioState,
        pending_items: List[AttentionItem],
        sniper_status: Optional[SniperStatus] = None,
        task_section: str = "",
        health: Optional[dict] = None,
        early_bird_opportunities: Optional[List[dict]] = None,
        mm_state: Optional[dict] = None,
        decision_accuracy: Optional[dict] = None,
    ) -> str:
        """Build prompt for strategic Captain cycles (every 5 min).

        Summary of portfolio, sub-threshold attention items, sniper stats, task ledger.
        Health is pre-injected so the LLM doesn't need to call get_account_health.
        """
        cb = ContextBudget(STRATEGIC_TOKEN_BUDGET)

        # Portfolio summary (high priority)
        if portfolio.positions:
            portfolio_lines = [
                f"PORTFOLIO: balance={_fmt_balance(portfolio, self._subaccount)}, "
                f"{portfolio.total_positions} pos, "
                f"pnl=${portfolio.total_unrealized_pnl_cents / 100:+.2f}"
            ]
            for p in portfolio.positions[:5]:
                pnl_ct = round(p.unrealized_pnl_cents / p.quantity) if p.quantity else 0
                portfolio_lines.append(
                    f"  {p.ticker} {p.side} {p.quantity}ct "
                    f"exit={p.exit_price} pnl={pnl_ct:+d}c/ct"
                )
            cb.add("PORTFOLIO", "\n".join(portfolio_lines))
        else:
            cb.add("PORTFOLIO", f"PORTFOLIO: balance={_fmt_balance(portfolio, self._subaccount)}, no positions")

        # Health (pre-injected — saves a tool call)
        if health:
            cb.add("HEALTH",
                f"HEALTH: drawdown={health.get('drawdown_pct', 0):.1f}%, "
                f"realized_pnl={health.get('total_realized_pnl_cents', 0)}c, "
                f"settlements={health.get('settlement_count_session', 0)}"
            )

        # Decision quality feedback (closes the learning loop)
        if decision_accuracy and decision_accuracy.get("total_decisions", 0) > 0:
            total = decision_accuracy["total_decisions"]
            with_outcomes = decision_accuracy.get("decisions_with_outcomes", 0)
            correct = decision_accuracy.get("direction_correct_count", 0)
            accuracy = decision_accuracy.get("direction_accuracy_pct", 0)
            avg_pnl = decision_accuracy.get("avg_hypothetical_pnl", 0)
            fill_pct = decision_accuracy.get("would_have_filled_pct", 0)
            cb.add("DECISIONS",
                f"DECISIONS (24h): {total} orders, {correct}/{with_outcomes} direction-correct "
                f"({accuracy:.0f}%), avg_pnl={avg_pnl:+.1f}c/trade, "
                f"{fill_pct:.0f}% would-have-filled"
            )

        # Early bird opportunities
        if early_bird_opportunities:
            eb_lines = [f"EARLY_BIRD: {len(early_bird_opportunities)} opportunities"]
            for opp in early_bird_opportunities[:5]:
                opp_parts = [f"  {opp.get('market_ticker', '?')}"]
                if opp.get("score"):
                    opp_parts.append(f"score={opp['score']:.0f}")
                if opp.get("strategy"):
                    opp_parts.append(f"strategy={opp['strategy']}")
                if opp.get("fair_value") is not None:
                    opp_parts.append(f"fair_value={opp['fair_value']:.0f}c")
                if opp.get("age_seconds"):
                    opp_parts.append(f"age={opp['age_seconds']:.0f}s")
                eb_lines.append(" ".join(opp_parts))
            cb.add("EARLY_BIRD", "\n".join(eb_lines))

        # Pending attention (sub-threshold items for awareness)
        if pending_items:
            attn_lines = [f"PENDING_ATTENTION: {len(pending_items)} items"]
            for item in pending_items[:5]:
                attn_lines.append(f"  - {item.to_prompt()}")
            if any(item.category == "early_bird" for item in pending_items):
                attn_lines.append("TIP: Early bird opportunities detected. Call get_early_bird_opportunities for details.")
            cb.add("PENDING_ATTENTION", "\n".join(attn_lines))

        # Sniper stats
        if sniper_status and sniper_status.enabled:
            sniper_parts = [
                f"{sniper_status.total_arbs_executed} arbs",
                f"{sniper_status.total_trades} trades",
            ]
            if sniper_status.last_rejection_reason:
                sniper_parts.append(f"last_reject={sniper_status.last_rejection_reason}")
            capital = sniper_status.capital_in_flight + sniper_status.capital_in_positions
            sniper_parts.append(f"capital=${capital / 100:.2f}")
            cb.add("SNIPER", f"SNIPER: {', '.join(sniper_parts)}")

        # MM state (expanded for strategic orchestration)
        if mm_state:
            enabled = mm_state.get("enabled", False)
            fills = mm_state.get("total_fills", 0)
            net_pnl = mm_state.get("net_pnl_cents", 0)
            pulled = mm_state.get("quotes_pulled", False)
            spread_captured = mm_state.get("spread_captured_cents", 0)
            adverse = mm_state.get("adverse_selection_cents", 0)
            mm_events = mm_state.get("mm_event_tickers", [])
            mm_text = (
                f"MM: {'ON' if enabled else 'OFF'}, "
                f"fills={fills}, spread_captured={spread_captured:.0f}c, adverse={adverse:.0f}c, "
                f"net_pnl={net_pnl:.0f}c"
                f"{', PULLED' if pulled else ''}"
            )
            if mm_events:
                mm_text += f"\n  MM_EVENTS: {', '.join(mm_events)}"
            cb.add("MM", mm_text)

        # Task ledger
        if task_section:
            cb.add("TASKS", task_section)

        return cb.build(
            suffix="ACTION: Search news on active events. Evaluate early bird opportunities. When MM active: get_quote_performance and tune. Manage positions."
        )

    def build_deep_scan_context(
        self,
        market_state: MarketState,
        portfolio: PortfolioState,
        sniper_status: Optional[SniperStatus] = None,
        health: Optional[dict] = None,
        trade_memories: Optional[List[str]] = None,
        news_memories: Optional[List[str]] = None,
        task_section: str = "",
        market_movers: Optional[List[dict]] = None,
        swing_patterns: Optional[List[dict]] = None,
        early_bird_opportunities: Optional[List[dict]] = None,
        mm_state: Optional[dict] = None,
        decision_accuracy: Optional[dict] = None,
    ) -> str:
        """Build prompt for deep scan Captain cycles (every 30 min).

        All events summary (compact), full positions, sniper perf, health, memories.
        """
        import json
        cb = ContextBudget(DEEP_SCAN_TOKEN_BUDGET)

        # Compact events summary (high priority)
        events_compact = []
        for ev in market_state.events:
            events_compact.append({
                "t": ev.event_ticker,
                "le": ev.long_edge,
                "se": ev.short_edge,
                "regime": ev.regime,
                "vol": ev.total_volume_5m,
                "mkts": ev.market_count,
                "ttc": ev.time_to_close_hours,
            })
        cb.add("EVENTS", f"EVENTS: {json.dumps(events_compact, separators=(',', ':'))}")

        # Full portfolio (high priority)
        if portfolio.positions:
            portfolio_lines = [
                f"PORTFOLIO: balance={_fmt_balance(portfolio, self._subaccount)}, "
                f"{portfolio.total_positions} pos, "
                f"pnl=${portfolio.total_unrealized_pnl_cents / 100:+.2f}, "
                f"cost=${portfolio.total_cost_cents / 100:.2f}"
            ]
            for p in portfolio.positions:
                pnl_ct = round(p.unrealized_pnl_cents / p.quantity) if p.quantity else 0
                portfolio_lines.append(
                    f"  {p.ticker} ({p.event_ticker}) {p.side} {p.quantity}ct "
                    f"cost={p.cost_cents}c exit={p.exit_price} pnl={pnl_ct:+d}c/ct"
                )
            cb.add("PORTFOLIO", "\n".join(portfolio_lines))
        else:
            cb.add("PORTFOLIO", f"PORTFOLIO: balance={_fmt_balance(portfolio, self._subaccount)}, no positions")

        # Sniper performance
        if sniper_status and sniper_status.enabled:
            cb.add("SNIPER_PERF",
                f"SNIPER_PERF: arbs={sniper_status.total_arbs_executed}, "
                f"trades={sniper_status.total_trades}, "
                f"unwinds={sniper_status.total_partial_unwinds}, "
                f"capital_lifetime={sniper_status.capital_deployed_lifetime}c"
            )

        # Health
        if health:
            cb.add("HEALTH",
                f"HEALTH: drawdown={health.get('drawdown_pct', 0):.1f}%, "
                f"realized_pnl={health.get('total_realized_pnl_cents', 0)}c, "
                f"settlements={health.get('settlement_count_session', 0)}"
            )

        # Decision quality feedback (closes the learning loop)
        if decision_accuracy and decision_accuracy.get("total_decisions", 0) > 0:
            total = decision_accuracy["total_decisions"]
            with_outcomes = decision_accuracy.get("decisions_with_outcomes", 0)
            correct = decision_accuracy.get("direction_correct_count", 0)
            accuracy = decision_accuracy.get("direction_accuracy_pct", 0)
            avg_pnl = decision_accuracy.get("avg_hypothetical_pnl", 0)
            fill_pct = decision_accuracy.get("would_have_filled_pct", 0)
            cb.add("DECISIONS",
                f"DECISIONS (24h): {total} orders, {correct}/{with_outcomes} direction-correct "
                f"({accuracy:.0f}%), avg_pnl={avg_pnl:+.1f}c/trade, "
                f"{fill_pct:.0f}% would-have-filled"
            )

        # Early bird opportunities
        if early_bird_opportunities:
            eb_lines = [f"EARLY_BIRD: {len(early_bird_opportunities)} opportunities"]
            for opp in early_bird_opportunities[:5]:
                opp_parts = [f"  {opp.get('market_ticker', '?')}"]
                if opp.get("score"):
                    opp_parts.append(f"score={opp['score']:.0f}")
                if opp.get("strategy"):
                    opp_parts.append(f"strategy={opp['strategy']}")
                if opp.get("fair_value") is not None:
                    opp_parts.append(f"fair_value={opp['fair_value']:.0f}c")
                if opp.get("age_seconds"):
                    opp_parts.append(f"age={opp['age_seconds']:.0f}s")
                eb_lines.append(" ".join(opp_parts))
            cb.add("EARLY_BIRD", "\n".join(eb_lines))

        # Trade outcome learnings (lower priority — may be truncated)
        if trade_memories:
            mem_lines = ["TRADE_LEARNINGS:"]
            for mem in trade_memories[:5]:
                mem_lines.append(f"  - {mem[:120]}")
            cb.add("TRADE_LEARNINGS", "\n".join(mem_lines))

        # News learnings (signal-quality-boosted recall)
        if news_memories:
            news_lines = ["NEWS_LEARNINGS:"]
            for mem in news_memories[:5]:
                news_lines.append(f"  - {mem[:120]}")
            cb.add("NEWS_LEARNINGS", "\n".join(news_lines))

        # News-price impact learnings
        if market_movers:
            impact_lines = ["NEWS_IMPACT (last 48h):"]
            for m in market_movers[:3]:
                impact_lines.append(
                    f"  - \"{m.get('news_title', '?')}\" moved {m.get('market_ticker', '?')} "
                    f"{m.get('change_cents', 0)}c {m.get('direction', '?')}"
                )
            cb.add("NEWS_IMPACT", "\n".join(impact_lines))

        # Swing-news pattern index (known news->price relationships)
        if swing_patterns:
            pattern_lines = ["NEWS_PATTERNS (known news->price relationships):"]
            for p in swing_patterns[:5]:
                pattern_lines.append(
                    f"  - \"{p.get('news_title', '?')[:80]}\" -> {p.get('direction', '?')} "
                    f"{p.get('change_cents', 0):.0f}c on {p.get('event_ticker', '?')} "
                    f"(confidence={p.get('causal_confidence', 0):.1f})"
                )
            cb.add("NEWS_PATTERNS", "\n".join(pattern_lines))

        # MM state (detailed for deep scan)
        if mm_state:
            enabled = mm_state.get("enabled", False)
            fills_bid = mm_state.get("total_fills_bid", 0)
            fills_ask = mm_state.get("total_fills_ask", 0)
            net_pnl = mm_state.get("net_pnl_cents", 0)
            spread_captured = mm_state.get("spread_captured_cents", 0)
            adverse = mm_state.get("adverse_selection_cents", 0)
            pulled = mm_state.get("quotes_pulled", False)
            mm_events = mm_state.get("mm_event_tickers", [])
            mm_lines = [
                f"MM: {'ON' if enabled else 'OFF'}, "
                f"fills={fills_bid}bid/{fills_ask}ask, "
                f"spread_captured={spread_captured:.0f}c, adverse={adverse:.0f}c, "
                f"net_pnl={net_pnl:.0f}c"
                f"{', PULLED' if pulled else ''}"
            ]
            if mm_events:
                mm_lines.append(f"  MM_EVENTS: {', '.join(mm_events)}")
            # Per-market inventory for non-zero positions
            mm_positions = mm_state.get("positions", [])
            for mp in mm_positions:
                if mp.get("position", 0) != 0:
                    mm_lines.append(
                        f"  MM_POS: {mp['ticker']} pos={mp['position']} "
                        f"pnl={mp.get('realized_pnl_cents', 0):.0f}c"
                    )
            cb.add("MM", "\n".join(mm_lines))

        # Task ledger
        if task_section:
            cb.add("TASKS", task_section)

        return cb.build(
            suffix="ACTION: Full review. Evaluate all events against news. "
            "Full MM review: get_quote_performance, recall MM memories, search_news on MM events. "
            "Check early bird opportunities. Manage positions."
        )

    def compute_diffs(self, current: MarketState) -> CycleDiff:
        """Compute changes since last cycle. Pure Python."""
        # Build current snapshot for comparison
        cur = {
            "prices": {},
            "volumes": {},
            "timestamp": time.time(),
        }
        for ev in current.events:
            for mt, m in ev.markets.items():
                if m.yes_bid is not None and m.yes_ask is not None:
                    cur["prices"][mt] = round((m.yes_bid + m.yes_ask) / 2, 1)
                cur["volumes"][ev.event_ticker] = ev.total_volume_5m

        prev = self._prev_state
        self._prev_state = cur

        if not prev:
            return CycleDiff(has_changes=False)

        elapsed = cur["timestamp"] - prev.get("timestamp", cur["timestamp"])
        price_moves = []
        for mt, price in cur["prices"].items():
            old = prev.get("prices", {}).get(mt)
            if old is not None:
                delta = price - old
                if abs(delta) >= 1:
                    sign = "+" if delta > 0 else ""
                    price_moves.append(f"{mt} {sign}{delta:.0f}c ({old:.0f}->{price:.0f})")

        volume_spikes = []
        for et, vol in cur["volumes"].items():
            old_vol = prev.get("volumes", {}).get(et, 0)
            if vol > old_vol + 50:
                volume_spikes.append(f"{et} +{vol - old_vol} vol")

        has_changes = bool(price_moves or volume_spikes)
        return CycleDiff(
            elapsed_seconds=round(elapsed, 0),
            price_moves=price_moves[:5],
            volume_spikes=volume_spikes[:3],
            has_changes=has_changes,
        )
