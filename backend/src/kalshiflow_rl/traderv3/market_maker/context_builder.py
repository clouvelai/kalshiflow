"""Context builder for Admiral market maker.

Constructs mode-specific prompts for each Admiral invocation cycle.
Zero API calls for market data - reads from MMIndex.
"""

import logging
import time
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, TYPE_CHECKING

from .models import (
    InventoryResult,
    MMAttentionItem,
    MMDeepScanContext,
    MMReactiveContext,
    MMStrategicContext,
    QuotePerformanceResult,
)

if TYPE_CHECKING:
    from .index import MMIndex
    from .quote_engine import QuoteEngine
    from ..single_arb.index import EventMeta

logger = logging.getLogger("kalshiflow_rl.traderv3.market_maker.context_builder")


@dataclass
class EventSemantics:
    """WHAT/WHO/WHEN context for an event. Built from EventUnderstanding."""
    what: str = ""
    who: List[str] = dc_field(default_factory=list)
    when: str = ""
    domain: str = ""
    search_terms: List[str] = dc_field(default_factory=list)
    news_summary: str = ""


def _event_semantics(event: "EventMeta") -> Optional[EventSemantics]:
    """Extract EventSemantics from EventUnderstanding dict, if available."""
    u = getattr(event, "understanding", None)
    if not u or not isinstance(u, dict):
        return None
    participants = u.get("participants", [])
    who = [p.get("name", "") for p in participants if isinstance(p, dict) and p.get("name")]
    key_factors = u.get("key_factors", [])
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
        search_terms=search_terms,
        news_summary=(u.get("news_articles", [{}])[0].get("title", "") if u.get("news_articles") else ""),
    )


class MMContextBuilder:
    """Builds mode-specific context strings for Admiral LLM cycles."""

    def __init__(self, index: "MMIndex", quote_engine: "QuoteEngine"):
        self._index = index
        self._engine = quote_engine

    def build_reactive(
        self,
        cycle_num: int,
        items: List[MMAttentionItem],
        balance_cents: int = 0,
    ) -> str:
        """Build compact reactive context (~200-400 tokens).

        Includes: attention items, inventory snapshot, quote state.
        """
        lines = [
            f"=== ADMIRAL REACTIVE CYCLE #{cycle_num} ===",
            f"Time: {time.strftime('%H:%M:%S')}",
            f"Balance: ${balance_cents / 100:.2f}",
            "",
            "--- SIGNALS ---",
        ]

        for item in items:
            lines.append(item.to_prompt())

        # Compact inventory
        lines.append("")
        lines.append("--- INVENTORY ---")
        for et, event in self._index.events.items():
            for ticker in event.markets:
                inv = self._index.get_inventory(ticker)
                if inv.position != 0:
                    lines.append(f"  {ticker}: pos={inv.position} pnl={inv.realized_pnl_cents:.0f}c")

        # Quote state
        state = self._engine.state
        lines.append("")
        lines.append(f"--- QUOTES: {state.active_quotes} active, pulled={state.quotes_pulled} ---")
        if state.fill_storm_active:
            lines.append("  *** FILL STORM ACTIVE - spreads widened 2x ***")
        lines.append(f"  spread_mult={state.spread_multiplier}x, fills_bid={state.total_fills_bid}, fills_ask={state.total_fills_ask}")

        return "\n".join(lines)

    def build_strategic(
        self,
        cycle_num: int,
        balance_cents: int = 0,
        pending_items: Optional[List[MMAttentionItem]] = None,
        tasks: str = "",
    ) -> str:
        """Build strategic context (~400-600 tokens).

        Includes: all markets with quotes + inventory, performance stats, pending items.
        """
        lines = [
            f"=== ADMIRAL STRATEGIC REVIEW #{cycle_num} ===",
            f"Time: {time.strftime('%H:%M:%S')}",
            f"Balance: ${balance_cents / 100:.2f}",
            "",
        ]

        # Per-market state
        lines.append("--- MARKETS ---")
        for et, event in self._index.events.items():
            lines.append(f"Event: {event.title} ({et})")
            sem = _event_semantics(event)
            if sem:
                lines.append(f"  Context: {sem.what[:100]} [{sem.domain}]")
            for ticker, market in event.markets.items():
                inv = self._index.get_inventory(ticker)
                fv = self._index.get_fair_value(ticker)
                quotes = self._index.get_quotes(ticker)
                bid_q = quotes.get("bid")
                ask_q = quotes.get("ask")

                bid_str = f"B:{bid_q.price_cents}c/{bid_q.size}" if bid_q else "B:--"
                ask_str = f"A:{ask_q.price_cents}c/{ask_q.size}" if ask_q else "A:--"
                fv_str = f"FV:{fv:.1f}" if fv else "FV:--"

                lines.append(
                    f"  {ticker}: {bid_str} {ask_str} {fv_str} "
                    f"mid={market.yes_mid or '--'} spread={market.spread or '--'} "
                    f"pos={inv.position} pnl={inv.realized_pnl_cents:.0f}c "
                    f"vpin={market.micro.vpin:.2f}"
                )

        # Performance
        state = self._engine.state
        lines.append("")
        lines.append("--- PERFORMANCE ---")
        lines.append(f"  Fills: bid={state.total_fills_bid} ask={state.total_fills_ask}")
        lines.append(f"  Spread captured: {state.spread_captured_cents:.1f}c")
        lines.append(f"  Fees paid: {state.fees_paid_cents:.1f}c")
        lines.append(f"  Requote cycles: {state.total_requote_cycles}")
        lines.append(f"  Spread multiplier: {state.spread_multiplier}x")

        # Config
        config = self._engine.config
        lines.append("")
        lines.append(f"--- CONFIG: spread={config.base_spread_cents}c size={config.quote_size} skew={config.skew_factor} ---")

        if pending_items:
            lines.append("")
            lines.append("--- PENDING SIGNALS ---")
            for item in pending_items[:5]:
                lines.append(f"  {item.to_prompt()}")

        if tasks:
            lines.append("")
            lines.append(tasks)

        return "\n".join(lines)

    def build_deep_scan(
        self,
        cycle_num: int,
        balance_cents: int = 0,
        memories: Optional[List[str]] = None,
    ) -> str:
        """Build deep scan context (~800-1200 tokens).

        Full market data, performance, inventory, memories.
        """
        lines = [
            f"=== ADMIRAL DEEP SCAN #{cycle_num} ===",
            f"Time: {time.strftime('%H:%M:%S %Z')}",
            f"Balance: ${balance_cents / 100:.2f}",
            "",
        ]

        # Full market state
        lines.append("--- FULL MARKET STATE ---")
        for et, event in self._index.events.items():
            lines.append(f"\nEvent: {event.title} ({et})")
            lines.append(f"  ME={event.mutually_exclusive} markets={len(event.markets)}")

            sem = _event_semantics(event)
            if sem:
                lines.append(f"  What: {sem.what[:200]}")
                if sem.who:
                    lines.append(f"  Who: {', '.join(sem.who[:5])}")
                if sem.when:
                    lines.append(f"  When: {sem.when}")
                lines.append(f"  Domain: {sem.domain}")
                if sem.search_terms:
                    lines.append(f"  Key factors: {', '.join(sem.search_terms)}")
                if sem.news_summary:
                    lines.append(f"  Recent news: {sem.news_summary[:150]}")

            sum_mid = event.market_sum()
            if sum_mid is not None:
                lines.append(f"  sum_mid={sum_mid:.1f} deviation={abs(sum_mid - 100):.1f}")

            for ticker, market in event.markets.items():
                inv = self._index.get_inventory(ticker)
                fv = self._index.get_fair_value(ticker)
                quotes = self._index.get_quotes(ticker)
                bid_q = quotes.get("bid")
                ask_q = quotes.get("ask")

                lines.append(f"  {ticker}: {market.title}")
                lines.append(
                    f"    BBO: {market.yes_bid or '--'}@{market.yes_bid_size}"
                    f" / {market.yes_ask or '--'}@{market.yes_ask_size}"
                    f" spread={market.spread or '--'}"
                )
                lines.append(f"    FV={fv or '--'} microprice={market.microprice or '--'}")
                lines.append(
                    f"    Micro: vpin={market.micro.vpin:.3f} obi={market.micro.book_imbalance:.3f}"
                    f" vol5m={market.micro.volume_5m} whales={market.micro.whale_trade_count}"
                )
                if bid_q:
                    lines.append(f"    Our bid: {bid_q.price_cents}c x{bid_q.size} Q:{bid_q.queue_position or '?'}")
                if ask_q:
                    lines.append(f"    Our ask: {ask_q.price_cents}c x{ask_q.size} Q:{ask_q.queue_position or '?'}")
                lines.append(
                    f"    Inventory: pos={inv.position} avg={inv.avg_entry_cents:.1f}c"
                    f" realized={inv.realized_pnl_cents:.1f}c unrealized={inv.unrealized_pnl_cents:.1f}c"
                )

        # Full performance
        state = self._engine.state
        lines.append("\n--- PERFORMANCE ---")
        lines.append(f"  Total fills: bid={state.total_fills_bid} ask={state.total_fills_ask}")
        lines.append(f"  Spread captured: {state.spread_captured_cents:.2f}c")
        lines.append(f"  Adverse selection: {state.adverse_selection_cents:.2f}c")
        lines.append(f"  Fees: {state.fees_paid_cents:.2f}c")
        net = state.spread_captured_cents - state.adverse_selection_cents - state.fees_paid_cents
        lines.append(f"  Net P&L from MM: {net:.2f}c")
        lines.append(f"  Requote cycles: {state.total_requote_cycles}")
        lines.append(f"  Quote uptime: {state.quote_uptime_pct:.1f}%")

        # Aggregate P&L
        total_realized = self._index.total_realized_pnl()
        total_unrealized = self._index.total_unrealized_pnl()
        lines.append(f"\n--- P&L ---")
        lines.append(f"  Realized: {total_realized:.2f}c (${total_realized/100:.2f})")
        lines.append(f"  Unrealized: {total_unrealized:.2f}c (${total_unrealized/100:.2f})")

        # Config
        config = self._engine.config
        lines.append(f"\n--- CONFIG ---")
        lines.append(f"  spread={config.base_spread_cents}c size={config.quote_size}")
        lines.append(f"  skew_factor={config.skew_factor} skew_cap={config.skew_cap_cents}c")
        lines.append(f"  max_pos={config.max_position} max_exposure={config.max_event_exposure}")
        lines.append(f"  refresh={config.refresh_interval}s vpin_pull={config.pull_quotes_threshold}")

        if memories:
            lines.append(f"\n--- MEMORY ({len(memories)} results) ---")
            for mem in memories[:5]:
                lines.append(f"  {mem[:200]}")

        return "\n".join(lines)
