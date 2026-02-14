"""Pydantic models and dataclasses for the Admiral market maker.

All tool returns and context injections use these models. Serialized to JSON
for the LLM prompt via .model_dump_json().

Dataclasses are used for internal mutable state (QuoteConfig, QuoteState, etc.)
while Pydantic BaseModel is used for serialization boundaries (tool outputs, WS).
"""

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# QuoteConfig (mutable by Admiral, like SniperConfig)
# ---------------------------------------------------------------------------

@dataclass
class QuoteConfig:
    """Market-making configuration tunable by the Admiral.

    Hot-reloaded by QuoteEngine each refresh cycle. Admiral calls
    configure_quotes tool to modify these values.
    """
    enabled: bool = True
    base_spread_cents: int = 4          # Minimum spread width
    quote_size: int = 10                # Contracts per side per market
    skew_factor: float = 0.5           # Inventory skew multiplier
    skew_cap_cents: float = 5.0        # Max skew offset in cents
    max_position: int = 100             # Max contracts per market (one side)
    max_event_exposure: int = 500       # Max total contracts across event
    refresh_interval: float = 5.0       # Seconds between requote cycles
    cancel_on_fill: bool = True         # Cancel opposite side on fill
    pull_quotes_threshold: float = 0.95 # VPIN above this → pull all quotes
    fill_storm_threshold: int = 10      # Fills in 30s → widen 2x
    fill_storm_window: float = 30.0     # Window for fill storm detection
    market_overrides: Dict[str, Dict] = dc_field(default_factory=dict)

    def get_market_config(self, ticker: str, key: str, default: Any = None) -> Any:
        """Get a per-market override or fall back to base config."""
        override = self.market_overrides.get(ticker, {})
        if key in override:
            return override[key]
        return getattr(self, key, default)


# ---------------------------------------------------------------------------
# Per-Market Quoting State (internal, mutable)
# ---------------------------------------------------------------------------

@dataclass
class ActiveQuote:
    """Tracks a single resting quote (bid or ask) we placed."""
    order_id: str = ""
    side: str = ""           # "yes" or "no"
    action: str = ""         # "buy" or "sell"
    price_cents: int = 0
    size: int = 0
    placed_at: float = 0.0
    queue_position: Optional[int] = None


@dataclass
class MarketInventory:
    """Position state for a single market."""
    position: int = 0               # Net position (+ = long YES, - = short YES / long NO)
    avg_entry_cents: float = 0.0
    realized_pnl_cents: float = 0.0
    unrealized_pnl_cents: float = 0.0
    total_buys: int = 0
    total_sells: int = 0
    cost_basis_cents: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.position == 0

    def record_fill(self, side: str, action: str, price_cents: int, quantity: int) -> None:
        """Update inventory from a fill.

        Args:
            side: "yes" or "no"
            action: "buy" or "sell"
            price_cents: Fill price in cents
            quantity: Number of contracts filled
        """
        # Normalize to YES-equivalent position change
        if side == "yes" and action == "buy":
            delta = quantity
            cost = price_cents * quantity
        elif side == "yes" and action == "sell":
            delta = -quantity
            cost = -price_cents * quantity
        elif side == "no" and action == "buy":
            delta = -quantity  # Buying NO = selling YES equivalent
            cost = (100 - price_cents) * quantity  # NO cost
        else:  # no + sell
            delta = quantity
            cost = -(100 - price_cents) * quantity

        old_pos = self.position
        self.position += delta

        if delta > 0:
            self.total_buys += quantity
        else:
            self.total_sells += quantity

        # Update cost basis and realized P&L
        if old_pos >= 0 and delta > 0:
            # Adding to long
            self.cost_basis_cents += cost
        elif old_pos <= 0 and delta < 0:
            # Adding to short
            self.cost_basis_cents += cost
        else:
            # Closing or flipping - realize P&L
            if old_pos != 0:
                avg = self.cost_basis_cents / abs(old_pos) if old_pos != 0 else 0
                closed = min(abs(delta), abs(old_pos))
                if old_pos > 0:
                    self.realized_pnl_cents += closed * (price_cents - avg)
                else:
                    self.realized_pnl_cents += closed * (avg - price_cents)
                remaining = abs(old_pos) - closed
                if remaining > 0:
                    self.cost_basis_cents = avg * remaining * (1 if old_pos > 0 else -1)
                else:
                    self.cost_basis_cents = 0
                    if abs(delta) > abs(old_pos):
                        # Flipped sides
                        overshoot = abs(delta) - abs(old_pos)
                        self.cost_basis_cents = price_cents * overshoot * (1 if delta > 0 else -1)

        if self.position != 0:
            self.avg_entry_cents = abs(self.cost_basis_cents / self.position)
        else:
            self.avg_entry_cents = 0.0


# ---------------------------------------------------------------------------
# QuoteState (telemetry, like SniperState)
# ---------------------------------------------------------------------------

@dataclass
class QuoteState:
    """Aggregate telemetry for the QuoteEngine."""
    active_quotes: int = 0
    total_fills_bid: int = 0
    total_fills_ask: int = 0
    total_requote_cycles: int = 0
    spread_captured_cents: float = 0.0
    adverse_selection_cents: float = 0.0
    fees_paid_cents: float = 0.0
    quote_uptime_pct: float = 0.0
    quotes_pulled: bool = False
    pull_reason: str = ""
    last_requote_at: float = 0.0
    fill_storm_active: bool = False
    spread_multiplier: float = 1.0  # 1.0 = normal, 2.0 = widened for fill storm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_quotes": self.active_quotes,
            "total_fills_bid": self.total_fills_bid,
            "total_fills_ask": self.total_fills_ask,
            "total_requote_cycles": self.total_requote_cycles,
            "spread_captured_cents": round(self.spread_captured_cents, 2),
            "adverse_selection_cents": round(self.adverse_selection_cents, 2),
            "fees_paid_cents": round(self.fees_paid_cents, 2),
            "quote_uptime_pct": round(self.quote_uptime_pct, 1),
            "quotes_pulled": self.quotes_pulled,
            "pull_reason": self.pull_reason,
            "last_requote_at": self.last_requote_at,
            "fill_storm_active": self.fill_storm_active,
            "spread_multiplier": self.spread_multiplier,
        }


# ---------------------------------------------------------------------------
# Pydantic Models (tool outputs, WS messages)
# ---------------------------------------------------------------------------

class MMMarketSnapshot(BaseModel):
    """Per-market state for Admiral tools / WS broadcast."""
    ticker: str
    title: str
    subtitle: str = ""
    status: str = "open"
    has_data: bool = False
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None
    yes_bid_size: int = 0
    yes_ask_size: int = 0
    spread: Optional[int] = None
    microprice: Optional[float] = None
    fair_value: Optional[float] = None
    vpin: float = 0.0
    book_imbalance: float = 0.0
    volume_5m: int = 0

    # Our quotes
    our_bid_price: Optional[int] = None
    our_bid_size: int = 0
    our_bid_queue: Optional[int] = None
    our_ask_price: Optional[int] = None
    our_ask_size: int = 0
    our_ask_queue: Optional[int] = None

    # Inventory
    position: int = 0
    avg_entry_cents: float = 0.0
    unrealized_pnl_cents: float = 0.0

    # Full depth for orderbook viz
    yes_levels: List[List[int]] = Field(default_factory=list)
    no_levels: List[List[int]] = Field(default_factory=list)

    # Maker fee at current mid
    maker_fee_cents: float = 0.0


class MMEventSnapshot(BaseModel):
    """Per-event state for Admiral tools / WS broadcast."""
    event_ticker: str
    title: str
    category: str = ""
    mutually_exclusive: bool = True
    market_count: int = 0
    markets: Dict[str, MMMarketSnapshot] = Field(default_factory=dict)
    total_position_contracts: int = 0
    total_unrealized_pnl_cents: float = 0.0
    total_realized_pnl_cents: float = 0.0


class MMStateResult(BaseModel):
    """Result of get_mm_state tool."""
    events: List[MMEventSnapshot] = Field(default_factory=list)
    quote_config: Dict = Field(default_factory=dict)
    quote_state: Dict = Field(default_factory=dict)


class InventoryResult(BaseModel):
    """Result of get_inventory tool."""
    markets: List[Dict] = Field(default_factory=list)
    total_position_contracts: int = 0
    total_realized_pnl_cents: float = 0.0
    total_unrealized_pnl_cents: float = 0.0
    total_fees_paid_cents: float = 0.0
    balance_cents: int = 0
    balance_dollars: float = 0.0
    event_exposure: int = 0
    max_event_exposure: int = 0


class QuotePerformanceResult(BaseModel):
    """Result of get_quote_performance tool."""
    total_fills_bid: int = 0
    total_fills_ask: int = 0
    total_requote_cycles: int = 0
    spread_captured_cents: float = 0.0
    adverse_selection_cents: float = 0.0
    fees_paid_cents: float = 0.0
    net_pnl_cents: float = 0.0
    quote_uptime_pct: float = 0.0
    fill_rate_pct: float = 0.0
    avg_spread_captured: float = 0.0
    spread_multiplier: float = 1.0
    fill_storm_active: bool = False


class ConfigureQuotesResult(BaseModel):
    """Result of configure_quotes / set_market_override tools."""
    status: str = "updated"
    config: Dict = Field(default_factory=dict)
    changes: List[str] = Field(default_factory=list)


class PullQuotesResult(BaseModel):
    """Result of pull_quotes / resume_quotes tools."""
    status: str = ""  # "pulled" or "resumed"
    cancelled_orders: int = 0
    reason: str = ""


# ---------------------------------------------------------------------------
# Attention / Context Models (dataclasses, internal)
# ---------------------------------------------------------------------------

@dataclass
class MMAttentionItem:
    """A scored signal for the Admiral's attention.

    Produced by MMAttentionRouter, consumed by Admiral's reactive loop.
    """
    event_ticker: str
    market_ticker: Optional[str] = None
    urgency: str = "normal"              # immediate, high, normal
    category: str = "fill"               # fill, inventory_warning, vpin_spike, spread_change, fill_storm
    summary: str = ""
    data: Dict[str, Any] = dc_field(default_factory=dict)
    score: float = 0.0
    created_at: float = dc_field(default_factory=lambda: __import__("time").time())
    ttl_seconds: float = 120.0

    @property
    def is_expired(self) -> bool:
        import time
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def key(self) -> str:
        return f"{self.event_ticker}:{self.market_ticker or ''}:{self.category}"

    def to_prompt(self) -> str:
        urgency_tag = self.urgency.upper()
        parts = [f"[{urgency_tag}]", self.event_ticker]
        if self.market_ticker:
            parts.append(self.market_ticker)
        parts.append(self.summary)
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_ticker": self.event_ticker,
            "market_ticker": self.market_ticker,
            "urgency": self.urgency,
            "category": self.category,
            "summary": self.summary,
            "data": self.data,
            "score": round(self.score, 1),
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
        }


@dataclass
class MMReactiveContext:
    """Context for reactive Admiral cycles (fill/VPIN driven)."""
    cycle_num: int = 0
    items: List[MMAttentionItem] = dc_field(default_factory=list)
    inventory: Optional[InventoryResult] = None
    quote_state: Optional[Dict] = None


@dataclass
class MMStrategicContext:
    """Context for strategic Admiral cycles (every 5 min)."""
    cycle_num: int = 0
    inventory: Optional[InventoryResult] = None
    performance: Optional[QuotePerformanceResult] = None
    pending_items: List[MMAttentionItem] = dc_field(default_factory=list)
    tasks: str = ""


@dataclass
class MMDeepScanContext:
    """Context for deep scan Admiral cycles (every 30 min)."""
    cycle_num: int = 0
    mm_state: Optional[MMStateResult] = None
    inventory: Optional[InventoryResult] = None
    performance: Optional[QuotePerformanceResult] = None
    health: Optional[Dict] = None
    memories: Optional[List[str]] = None


# Re-export news models from single_arb (shared, not duplicated)
from ..single_arb.models import NewsArticle, ImpactPattern, SwingNewsResult  # noqa: E402, F401
