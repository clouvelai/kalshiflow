"""Pydantic models for Captain V2 tool inputs/outputs and context injection.

All tool returns and context injections use these models. No more ad-hoc dicts.
Serialized to JSON for the LLM prompt via .model_dump_json().
"""

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --- Market Data Models ---

class MarketSnapshot(BaseModel):
    """Per-market state derived from MarketMeta."""
    ticker: str
    title: str
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None
    yes_bid_size: int = 0
    yes_ask_size: int = 0
    spread: Optional[int] = None
    microprice: Optional[float] = None
    vpin: float = 0.0
    book_imbalance: float = 0.0
    volume_5m: int = 0
    whale_trade_count: int = 0
    trade_count: int = 0
    freshness_seconds: float = 0.0
    last_price: Optional[int] = None
    regime: str = "normal"  # normal, toxic, sweep, thin


class EventSnapshot(BaseModel):
    """Per-event state derived from EventMeta."""
    event_ticker: str
    title: str
    category: str = ""
    mutually_exclusive: bool = True
    market_count: int = 0
    markets: Dict[str, MarketSnapshot] = Field(default_factory=dict)
    long_edge: Optional[float] = None
    short_edge: Optional[float] = None
    sum_yes_bid: Optional[float] = None
    sum_yes_ask: Optional[float] = None
    total_volume_5m: int = 0
    regime: str = "normal"  # normal, toxic, sweep, thin
    time_to_close_hours: Optional[float] = None
    semantics: Optional["EventSemantics"] = None


class EventSemantics(BaseModel):
    """WHAT/WHO/WHEN context for an event. Built from EventUnderstanding."""
    what: str = ""
    who: List[str] = Field(default_factory=list)
    when: str = ""
    domain: str = ""
    settlement_summary: str = ""
    search_terms: List[str] = Field(default_factory=list)
    news_summary: str = ""
    news_fetched_at: Optional[float] = None


class MarketState(BaseModel):
    """Complete market context injected into Captain prompt each cycle."""
    events: List[EventSnapshot] = Field(default_factory=list)
    total_events: int = 0
    total_markets: int = 0


# --- Portfolio Models ---

class Position(BaseModel):
    """Single open position."""
    ticker: str
    event_ticker: Optional[str] = None
    side: str  # "yes" or "no"
    quantity: int = 0
    cost_cents: int = 0
    exit_price: Optional[int] = None
    current_value_cents: int = 0
    unrealized_pnl_cents: int = 0


class PortfolioState(BaseModel):
    """Balance + positions. Single API call to construct."""
    balance_cents: int = 0
    balance_dollars: Optional[float] = 0.0
    positions: List[Position] = Field(default_factory=list)
    total_positions: int = 0
    total_unrealized_pnl_cents: int = 0
    total_cost_cents: int = 0


# --- Order/Execution Models ---

class OrderResult(BaseModel):
    """Order response WITH new_balance so agent never needs a separate balance query."""
    order_id: str = ""
    status: str = ""  # placed, resting, executed, failed
    ticker: str = ""
    side: str = ""
    action: str = ""
    contracts: int = 0
    price_cents: int = 0
    ttl_seconds: int = 0
    new_balance_cents: Optional[int] = None
    new_balance_dollars: Optional[float] = None
    error: Optional[str] = None


class ArbLegResult(BaseModel):
    """Single leg of an arb execution."""
    ticker: str
    title: str = ""
    side: str
    contracts: int
    price_cents: int
    order_id: str = ""
    status: str = ""


class ArbResult(BaseModel):
    """Multi-leg arb response WITH new_balance."""
    status: str = ""  # preview, completed, partial, failed
    event_ticker: str = ""
    direction: str = ""
    legs_executed: int = 0
    legs_total: int = 0
    legs: List[ArbLegResult] = Field(default_factory=list)
    total_cost_cents: int = 0
    new_balance_cents: Optional[int] = None
    new_balance_dollars: Optional[float] = None
    errors: List[str] = Field(default_factory=list)
    conflict_warnings: List[str] = Field(default_factory=list)


class RestingOrder(BaseModel):
    """Queue state for a resting order."""
    order_id: str
    ticker: str = ""
    side: str = ""
    action: str = ""
    price_cents: int = 0
    remaining_count: int = 0
    queue_position: Optional[int] = None


# --- News/Memory Models ---

class NewsArticle(BaseModel):
    """Normalized news article from Tavily or DDG."""
    title: str = ""
    url: str = ""
    content: str = ""
    raw_content: str = ""  # Full article body from Tavily Extract or advanced search
    published_date: str = ""
    score: float = 0.0
    source: str = ""
    similar_patterns: List["ImpactPattern"] = Field(default_factory=list)


class NewsSearchResult(BaseModel):
    """Search response."""
    query: str = ""
    articles: List[NewsArticle] = Field(default_factory=list)
    count: int = 0
    stored_in_memory: bool = False
    cached: bool = False


class ImpactPattern(BaseModel):
    """A historical news article that was similar AND moved prices."""
    news_title: str = ""
    news_url: str = ""
    direction: str = ""        # "up" / "down"
    change_cents: float = 0.0  # how much it moved prices
    confidence: float = 0.0    # causal confidence
    similarity: float = 0.0    # how similar to current article
    event_ticker: str = ""     # which event it was on
    market_ticker: str = ""


class SwingNewsResult(BaseModel):
    """Enhanced news search result with pattern enrichment."""
    query: str = ""
    articles: List[NewsArticle] = Field(default_factory=list)
    count: int = 0
    stored_in_memory: bool = False
    cached: bool = False
    depth: str = ""  # ultra_fast, fast, advanced
    patterns_found: int = 0
    tavily_answer: str = ""  # LLM synthesis from advanced search


class MemoryEntry(BaseModel):
    """Recalled memory item."""
    content: str = ""
    memory_type: str = ""
    similarity: float = 0.0
    age_hours: Optional[float] = None


class RecallResult(BaseModel):
    """Memory search result."""
    query: str = ""
    results: List[MemoryEntry] = Field(default_factory=list)
    count: int = 0


# --- Sniper Models ---

class SniperStatus(BaseModel):
    """Execution layer status."""
    enabled: bool = False
    total_trades: int = 0
    total_arbs_executed: int = 0
    capital_in_flight: int = 0
    capital_in_positions: int = 0
    capital_deployed_lifetime: int = 0
    total_partial_unwinds: int = 0
    last_rejection_reason: Optional[str] = None
    last_action_summary: Optional[str] = None
    config_subset: Dict = Field(default_factory=dict)


# --- Diff Models ---

class CycleDiff(BaseModel):
    """Changes since last cycle."""
    elapsed_seconds: float = 0.0
    price_moves: List[str] = Field(default_factory=list)
    volume_spikes: List[str] = Field(default_factory=list)
    new_activity: List[str] = Field(default_factory=list)
    has_changes: bool = False


# --- Account Health Models ---

class SettlementSummary(BaseModel):
    """Single settlement record."""
    ticker: str = ""
    result: str = ""  # "yes", "no", or "voided"
    revenue_cents: int = 0      # gross payout (keep for backward compat)
    pnl_cents: int = 0          # net P&L = revenue - costs
    settled_at: Optional[str] = None


class StalePosition(BaseModel):
    """Position in a settled or closed market."""
    ticker: str = ""
    event_ticker: str = ""
    side: str = ""
    quantity: int = 0
    reason: str = ""  # "market_settled", "market_closed", "event_closed"


class DecisionAccuracyStats(BaseModel):
    """Aggregated decision accuracy metrics from the decision ledger."""
    total_decisions: int = 0
    decisions_with_outcomes: int = 0
    direction_accuracy_pct: float = 0.0
    avg_hypothetical_pnl_cents: float = 0.0
    would_have_filled_pct: float = 0.0
    by_source: Dict = Field(default_factory=dict)
    by_cycle_mode: Dict = Field(default_factory=dict)


class AccountHealthStatus(BaseModel):
    """Complete account health snapshot from AccountHealthService."""
    status: str = "healthy"  # healthy, warning, critical
    balance_cents: int = 0
    balance_dollars: float = 0.0
    balance_peak_cents: int = 0
    drawdown_pct: float = 0.0
    balance_trend: str = "stable"  # rising, falling, stable
    settlement_count_session: int = 0
    total_realized_pnl_cents: int = 0
    recent_settlements: List[SettlementSummary] = Field(default_factory=list)
    stale_positions: List[StalePosition] = Field(default_factory=list)
    stale_orders_cleaned: int = 0
    orphaned_groups_cleaned: int = 0
    alerts: List[Dict] = Field(default_factory=list)
    activity_log: List[Dict] = Field(default_factory=list)
    decision_accuracy: Optional[DecisionAccuracyStats] = None


# Rebuild forward refs for nested models
EventSnapshot.model_rebuild()
NewsArticle.model_rebuild()


# --- Attention Router Models (dataclasses, not Pydantic - internal only) ---

@dataclass
class AttentionItem:
    """A scored signal that may warrant Captain's attention.

    Produced by AttentionRouter, consumed by Captain's reactive/strategic loops.
    """
    event_ticker: str
    market_ticker: Optional[str] = None  # None = event-level signal
    urgency: str = "normal"              # immediate, high, normal
    category: str = "arb_opportunity"
    summary: str = ""                    # human-readable, ~20 words
    data: Dict[str, Any] = dc_field(default_factory=dict)
    score: float = 0.0                   # composite 0-100
    created_at: float = dc_field(default_factory=lambda: __import__("time").time())
    ttl_seconds: float = 120.0           # auto-expire

    @property
    def is_expired(self) -> bool:
        import time
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def key(self) -> str:
        """Dedup key: (event_ticker, category)."""
        return f"{self.event_ticker}:{self.category}"

    def to_prompt(self) -> str:
        """Compact string for injection into Captain prompt."""
        urgency_tag = self.urgency.upper()
        parts = [f"[{urgency_tag}]", self.event_ticker]
        if self.market_ticker:
            parts.append(self.market_ticker)
        parts.append(self.summary)
        if self.data.get("auto_handled"):
            parts.append(f"(auto: {self.data['auto_handled']})")
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


# --- Context Mode Dataclasses ---

@dataclass
class ReactiveContext:
    """Context for reactive Captain cycles (attention-driven)."""
    cycle_num: int = 0
    items: List[AttentionItem] = dc_field(default_factory=list)
    portfolio: Optional[PortfolioState] = None
    sniper: Optional[SniperStatus] = None

@dataclass
class StrategicContext:
    """Context for strategic Captain cycles (every 5 min)."""
    cycle_num: int = 0
    portfolio: Optional[PortfolioState] = None
    pending_items: List[AttentionItem] = dc_field(default_factory=list)
    sniper: Optional[SniperStatus] = None
    tasks: str = ""  # task ledger section

@dataclass
class DeepScanContext:
    """Context for deep scan Captain cycles (every 30 min)."""
    cycle_num: int = 0
    market_state: Optional[MarketState] = None
    portfolio: Optional[PortfolioState] = None
    sniper: Optional[SniperStatus] = None
    health: Optional[Dict] = None
    memories: Optional[List[str]] = None
