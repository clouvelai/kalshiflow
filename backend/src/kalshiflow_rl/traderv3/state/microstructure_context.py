"""
Microstructure Context - Clean representation of trading/orderbook state for LLM consumption.

This module provides a unified data structure for exposing trading microstructure signals
to the agentic research pipeline. It encapsulates trade flow, orderbook, and position state
in a format suitable for LLM reasoning.

Architecture:
    - MicrostructureContext is built from various V3 trader data sources
    - Provides to_prompt_string() for formatting in LLM prompts
    - Can be serialized for logging/storage

Data Sources:
    - Trade flow: From PUBLIC_TRADE_RECEIVED event accumulation (INDEPENDENT per-strategy)
    - Orderbook: From V3OrderbookIntegration.get_orderbook_signals() (SHARED)
    - Position: From V3StateContainer.trading_state (SHARED)

Architecture Decision (2026-01-08):
    Trade flow signals are currently accumulated INDEPENDENTLY per-strategy (e.g., RLM and
    agentic research each maintain their own TradeFlowState). This is intentional for the
    discovery phase - we don't yet know what signals agentic research actually needs.

    FUTURE: Once requirements are validated, consolidate into a shared TradeSignalAggregator
    service (similar to OrderbookSignalAggregator) that computes common signals once:
    - yes_ratio, no_ratio, momentum, trade_velocity, whale_count, price_drop_from_open

    See: .claude/plans/joyful-tickling-hummingbird.md for full Architecture Decision Record.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class TradeFlowState:
    """Per-market trade flow accumulation state."""
    yes_trades: int = 0
    no_trades: int = 0
    first_yes_price: Optional[int] = None  # True market open price
    last_yes_price: Optional[int] = None
    tmo_price: Optional[int] = None  # Fetched TMO price (if available)
    first_trade_at: Optional[float] = None
    last_trade_at: Optional[float] = None

    @property
    def total_trades(self) -> int:
        return self.yes_trades + self.no_trades

    @property
    def yes_ratio(self) -> float:
        """Ratio of YES trades to total trades (0.0-1.0)."""
        if self.total_trades == 0:
            return 0.5
        return self.yes_trades / self.total_trades

    @property
    def price_drop_from_open(self) -> int:
        """Price drop from open in cents. Positive = price fell."""
        open_price = self.tmo_price or self.first_yes_price
        if open_price is None or self.last_yes_price is None:
            return 0
        return open_price - self.last_yes_price

    def update_from_trade(
        self,
        side: str,
        price_cents: int,
        timestamp: float,
    ) -> None:
        """Update state from a public trade event."""
        # Convert NO price to YES price
        yes_price = price_cents if side == "yes" else (100 - price_cents)

        # Track first price (if not using TMO)
        if self.first_yes_price is None:
            self.first_yes_price = yes_price
            self.first_trade_at = timestamp

        # Accumulate counts
        if side == "yes":
            self.yes_trades += 1
        else:
            self.no_trades += 1

        self.last_yes_price = yes_price
        self.last_trade_at = timestamp

    def reset(self) -> None:
        """Reset state for a new market."""
        self.yes_trades = 0
        self.no_trades = 0
        self.first_yes_price = None
        self.last_yes_price = None
        self.tmo_price = None
        self.first_trade_at = None
        self.last_trade_at = None


@dataclass
class MicrostructureContext:
    """
    Clean, typed representation of trading/orderbook state for LLM consumption.

    This is the primary interface for exposing microstructure data to the agentic
    research pipeline. All fields are typed and documented for LLM understanding.
    """

    # === TRADE FLOW SIGNALS ===
    yes_trades: int = 0                # Count of YES trades observed
    no_trades: int = 0                 # Count of NO trades observed
    yes_ratio: float = 0.5             # YES trades / total trades (0.0-1.0)
    total_trades: int = 0              # Total trade count
    price_drop_from_open: int = 0      # TMO/open price - current (cents, positive = fell)
    last_yes_price: int = 50           # Most recent YES price (cents)

    # === ORDERBOOK SIGNALS (from 10-second buckets) ===
    no_spread: int = 0                 # Current NO spread (cents)
    yes_spread: int = 0                # Current YES spread (cents)
    volume_imbalance: float = 0.0      # (bid_vol - ask_vol) / total → [-1, 1]
    bbo_depth_bid: int = 0             # Best bid size (contracts)
    bbo_depth_ask: int = 0             # Best ask size (contracts)
    spread_volatility: float = 0.0     # Spread instability (0-2)
    large_order_count: int = 0         # Orders ≥10k contracts in recent window

    # === POSITION CONTEXT (optional) ===
    current_position: Optional[int] = None       # Existing position count
    position_side: Optional[str] = None          # "yes" or "no"
    unrealized_pnl_cents: Optional[int] = None   # Unrealized P&L in cents

    # === TIMESTAMPS ===
    captured_at: float = field(default_factory=time.time)
    trade_flow_age_seconds: float = 0.0    # Time since last trade flow update
    orderbook_age_seconds: float = 0.0     # Time since last orderbook update

    @classmethod
    def from_components(
        cls,
        trade_flow: Optional[TradeFlowState] = None,
        orderbook_context: Optional[Any] = None,  # OrderbookContext
        orderbook_signals: Optional[Dict[str, Any]] = None,  # Raw signals from aggregator
        position_count: Optional[int] = None,
        position_side: Optional[str] = None,
        unrealized_pnl: Optional[int] = None,
    ) -> 'MicrostructureContext':
        """
        Build MicrostructureContext from V3 trader components.

        Args:
            trade_flow: TradeFlowState from per-market accumulation
            orderbook_context: OrderbookContext from order_context module
            orderbook_signals: Raw signals dict from OrderbookSignalAggregator
            position_count: Current position count from state container
            position_side: Current position side ("yes" or "no")
            unrealized_pnl: Unrealized P&L in cents

        Returns:
            MicrostructureContext with all available data
        """
        ctx = cls()
        current_time = time.time()

        # === TRADE FLOW ===
        if trade_flow:
            ctx.yes_trades = trade_flow.yes_trades
            ctx.no_trades = trade_flow.no_trades
            ctx.yes_ratio = trade_flow.yes_ratio
            ctx.total_trades = trade_flow.total_trades
            ctx.price_drop_from_open = trade_flow.price_drop_from_open
            ctx.last_yes_price = trade_flow.last_yes_price or 50

            if trade_flow.last_trade_at:
                ctx.trade_flow_age_seconds = current_time - trade_flow.last_trade_at

        # === ORDERBOOK (from OrderbookContext) ===
        if orderbook_context:
            ctx.no_spread = orderbook_context.no_spread or 0
            ctx.yes_spread = orderbook_context.yes_spread or 0
            ctx.bbo_depth_bid = orderbook_context.no_bid_size_at_bbo or 0
            ctx.bbo_depth_ask = orderbook_context.no_ask_size_at_bbo or 0

            # Volume imbalance from bid_imbalance field
            if orderbook_context.bid_imbalance is not None:
                ctx.volume_imbalance = orderbook_context.bid_imbalance

            # Calculate orderbook age
            if orderbook_context.captured_at:
                ctx.orderbook_age_seconds = current_time - orderbook_context.captured_at

        # === ORDERBOOK SIGNALS (from aggregator buckets) ===
        if orderbook_signals:
            # Spread volatility from 10-second buckets
            ctx.spread_volatility = orderbook_signals.get("spread_volatility", 0.0)
            ctx.large_order_count = orderbook_signals.get("large_order_count", 0)

            # Override spread if not set from OrderbookContext
            if ctx.no_spread == 0 and "no_spread_close" in orderbook_signals:
                ctx.no_spread = orderbook_signals.get("no_spread_close", 0)
            if ctx.yes_spread == 0 and "yes_spread_close" in orderbook_signals:
                ctx.yes_spread = orderbook_signals.get("yes_spread_close", 0)

            # Volume imbalance from bucket if not set
            if ctx.volume_imbalance == 0.0 and "volume_imbalance" in orderbook_signals:
                ctx.volume_imbalance = orderbook_signals.get("volume_imbalance", 0.0)

        # === POSITION ===
        ctx.current_position = position_count
        ctx.position_side = position_side
        ctx.unrealized_pnl_cents = unrealized_pnl

        ctx.captured_at = current_time
        return ctx

    def to_prompt_string(self) -> str:
        """
        Format microstructure context for LLM prompt consumption.

        Returns a multi-line string suitable for inclusion in LLM prompts,
        with human-readable formatting and explanatory context.
        """
        lines = []

        # === TRADE FLOW ===
        lines.append(f"- Trade Flow: {self.yes_ratio:.0%} YES trades ({self.total_trades} total)")

        if self.price_drop_from_open != 0:
            direction = "down" if self.price_drop_from_open > 0 else "up"
            lines.append(f"- Price Movement: {abs(self.price_drop_from_open)}c {direction} from open")
        else:
            lines.append("- Price Movement: Unchanged from open")

        # === ORDERBOOK ===
        lines.append(f"- Orderbook Spread: NO={self.no_spread}c, YES={self.yes_spread}c")

        # Volume imbalance interpretation
        if abs(self.volume_imbalance) > 0.3:
            if self.volume_imbalance > 0:
                imbalance_desc = "strong buy pressure"
            else:
                imbalance_desc = "strong sell pressure"
        elif abs(self.volume_imbalance) > 0.1:
            if self.volume_imbalance > 0:
                imbalance_desc = "mild buy pressure"
            else:
                imbalance_desc = "mild sell pressure"
        else:
            imbalance_desc = "balanced"
        lines.append(f"- Volume Imbalance: {self.volume_imbalance:+.2f} ({imbalance_desc})")

        lines.append(f"- Liquidity: {self.bbo_depth_bid} bid / {self.bbo_depth_ask} ask contracts")

        if self.large_order_count > 0:
            lines.append(f"- Large Orders: {self.large_order_count} (≥10k contracts)")

        if self.spread_volatility > 0.5:
            lines.append(f"- Spread Volatility: {self.spread_volatility:.2f} (unstable)")

        # === POSITION (if any) ===
        if self.current_position is not None and self.current_position != 0:
            pnl_str = ""
            if self.unrealized_pnl_cents is not None:
                pnl_sign = "+" if self.unrealized_pnl_cents >= 0 else ""
                pnl_str = f", {pnl_sign}${self.unrealized_pnl_cents/100:.2f} unrealized"
            lines.append(f"- Current Position: {self.current_position} {self.position_side}{pnl_str}")

        # === DATA FRESHNESS ===
        if self.trade_flow_age_seconds > 60:
            lines.append(f"- Trade Data Age: {self.trade_flow_age_seconds:.0f}s (stale)")
        if self.orderbook_age_seconds > 10:
            lines.append(f"- Orderbook Age: {self.orderbook_age_seconds:.0f}s")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            # Trade flow
            "yes_trades": self.yes_trades,
            "no_trades": self.no_trades,
            "yes_ratio": round(self.yes_ratio, 4),
            "total_trades": self.total_trades,
            "price_drop_from_open": self.price_drop_from_open,
            "last_yes_price": self.last_yes_price,
            # Orderbook
            "no_spread": self.no_spread,
            "yes_spread": self.yes_spread,
            "volume_imbalance": round(self.volume_imbalance, 4),
            "bbo_depth_bid": self.bbo_depth_bid,
            "bbo_depth_ask": self.bbo_depth_ask,
            "spread_volatility": round(self.spread_volatility, 4),
            "large_order_count": self.large_order_count,
            # Position
            "current_position": self.current_position,
            "position_side": self.position_side,
            "unrealized_pnl_cents": self.unrealized_pnl_cents,
            # Timestamps
            "captured_at": self.captured_at,
            "trade_flow_age_seconds": round(self.trade_flow_age_seconds, 2),
            "orderbook_age_seconds": round(self.orderbook_age_seconds, 2),
        }

    def is_data_fresh(
        self,
        max_trade_age_seconds: float = 120.0,
        max_orderbook_age_seconds: float = 30.0,
    ) -> bool:
        """Check if microstructure data is fresh enough for trading decisions."""
        trade_fresh = self.trade_flow_age_seconds <= max_trade_age_seconds
        orderbook_fresh = self.orderbook_age_seconds <= max_orderbook_age_seconds
        return trade_fresh and orderbook_fresh

    def has_sufficient_data(self, min_trades: int = 5) -> bool:
        """Check if there's enough data for meaningful analysis."""
        return self.total_trades >= min_trades

    @property
    def is_empty(self) -> bool:
        """Check if context has no meaningful data."""
        return self.total_trades == 0 and self.no_spread == 0 and self.yes_spread == 0
