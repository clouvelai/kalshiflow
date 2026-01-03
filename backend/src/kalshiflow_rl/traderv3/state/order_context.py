"""
Order Context Capture - Data structures for persisting order context.

Captures comprehensive signal, orderbook, and position context at order time
for post-hoc quant analysis of settled trades.

Architecture:
    - Context is STAGED in memory when order is placed (has all context available)
    - Context is PERSISTED to DB only when order is confirmed FILLED
    - Settlement data is LINKED later when market settles

Key Data Flow:
    1. TradingDecisionService._execute_buy() -> stage_context() (in memory)
    2. StateContainer._sync_trading_attachments() detects fill -> persist_on_fill() (to DB)
    3. StateContainer._capture_settlement_for_attachment() -> link_settlement() (update DB)
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger("kalshiflow_rl.traderv3.state.order_context")


class SpreadTier(Enum):
    """Spread classification for orderbook state."""
    TIGHT = "tight"      # <= 2c
    NORMAL = "normal"    # <= 4c
    WIDE = "wide"        # > 4c
    UNKNOWN = "unknown"


@dataclass
class OrderbookSnapshot:
    """
    Top-of-book snapshot at order time.

    Captures the YES side orderbook state when order was placed.
    """
    best_bid_cents: Optional[int] = None
    best_ask_cents: Optional[int] = None
    bid_ask_spread_cents: Optional[int] = None
    bid_size_contracts: Optional[int] = None
    ask_size_contracts: Optional[int] = None
    spread_tier: SpreadTier = SpreadTier.UNKNOWN

    @classmethod
    def from_orderbook_state(cls, snapshot: Dict[str, Any]) -> 'OrderbookSnapshot':
        """Create from shared orderbook state snapshot."""
        if not snapshot:
            return cls()

        yes_bids = snapshot.get("yes_bids", {})
        yes_asks = snapshot.get("yes_asks", {})

        # Get best bid/ask (lowest ask, highest bid)
        best_bid = max(yes_bids.keys()) if yes_bids else None
        best_ask = min(yes_asks.keys()) if yes_asks else None

        spread = None
        spread_tier = SpreadTier.UNKNOWN
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            if spread <= 2:
                spread_tier = SpreadTier.TIGHT
            elif spread <= 4:
                spread_tier = SpreadTier.NORMAL
            else:
                spread_tier = SpreadTier.WIDE

        return cls(
            best_bid_cents=best_bid,
            best_ask_cents=best_ask,
            bid_ask_spread_cents=spread,
            bid_size_contracts=yes_bids.get(best_bid) if best_bid else None,
            ask_size_contracts=yes_asks.get(best_ask) if best_ask else None,
            spread_tier=spread_tier,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_bid_cents": self.best_bid_cents,
            "best_ask_cents": self.best_ask_cents,
            "bid_ask_spread_cents": self.bid_ask_spread_cents,
            "bid_size_contracts": self.bid_size_contracts,
            "ask_size_contracts": self.ask_size_contracts,
            "spread_tier": self.spread_tier.value,
        }


@dataclass
class PositionContext:
    """Position and account context at order time."""
    existing_position_count: int = 0
    existing_position_side: Optional[str] = None  # 'yes' or 'no'
    is_reentry: bool = False
    entry_number: int = 1
    balance_cents: int = 0
    open_position_count: int = 0  # Total positions across all markets

    def to_dict(self) -> Dict[str, Any]:
        return {
            "existing_position_count": self.existing_position_count,
            "existing_position_side": self.existing_position_side,
            "is_reentry": self.is_reentry,
            "entry_number": self.entry_number,
            "balance_cents": self.balance_cents,
            "open_position_count": self.open_position_count,
        }


@dataclass
class MarketContext:
    """Market context at signal time."""
    market_category: Optional[str] = None
    market_close_ts: Optional[float] = None  # Unix timestamp
    hours_to_settlement: Optional[float] = None
    trades_in_market: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_category": self.market_category,
            "market_close_ts": self.market_close_ts,
            "hours_to_settlement": self.hours_to_settlement,
            "trades_in_market": self.trades_in_market,
        }


@dataclass
class StagedOrderContext:
    """
    In-memory staging of order context before fill confirmation.

    Created when order is placed, persisted to DB only when fill is confirmed.
    Contains all context that is available at order placement time.
    """
    # Order Identification
    order_id: str
    market_ticker: str
    session_id: Optional[str] = None

    # Strategy & Signal
    strategy: str = "unknown"
    signal_id: Optional[str] = None
    signal_detected_at: Optional[float] = None
    signal_params: Dict[str, Any] = field(default_factory=dict)

    # Market Context
    market: MarketContext = field(default_factory=MarketContext)

    # Price Context
    no_price_at_signal: Optional[int] = None  # 100 - yes_price
    bucket_5c: Optional[int] = None

    # Orderbook Context
    orderbook: OrderbookSnapshot = field(default_factory=OrderbookSnapshot)

    # Position Context
    position: PositionContext = field(default_factory=PositionContext)

    # Order Details
    action: str = "buy"  # 'buy' or 'sell'
    side: str = "yes"    # 'yes' or 'no'
    order_price_cents: Optional[int] = None
    order_quantity: int = 0
    order_type: str = "limit"

    # Timestamps
    placed_at: float = field(default_factory=time.time)

    # Metadata
    strategy_version: Optional[str] = None

    def compute_derived_fields(self) -> None:
        """Compute derived fields like bucket_5c from raw data."""
        if self.no_price_at_signal is not None:
            self.bucket_5c = (self.no_price_at_signal // 5) * 5

    def to_db_dict(
        self,
        fill_count: int,
        fill_avg_price_cents: Optional[int],
        filled_at: float,
    ) -> Dict[str, Any]:
        """
        Convert to dictionary for database insertion.

        Called when fill is confirmed, with fill details provided.
        """
        placed_dt = datetime.fromtimestamp(self.placed_at, tz=timezone.utc)
        filled_dt = datetime.fromtimestamp(filled_at, tz=timezone.utc)

        # Calculate timing fields
        time_to_fill_ms = int((filled_at - self.placed_at) * 1000)

        # Calculate slippage (for NO side orders)
        slippage_cents = None
        signal_price = self.signal_params.get("last_yes_price")
        if signal_price is not None and fill_avg_price_cents is not None:
            # For NO side: we want NO price to be low (good fill)
            # Signal gave us YES price, we're buying NO at (100 - yes_price)
            if self.side == "no":
                expected_no_price = 100 - signal_price
                slippage_cents = fill_avg_price_cents - expected_no_price
            else:
                slippage_cents = fill_avg_price_cents - signal_price

        return {
            "order_id": self.order_id,
            "market_ticker": self.market_ticker,
            "session_id": self.session_id,
            # Strategy & Signal
            "strategy": self.strategy,
            "signal_id": self.signal_id,
            "signal_detected_at": datetime.fromtimestamp(self.signal_detected_at, tz=timezone.utc) if self.signal_detected_at else None,
            "signal_params": self.signal_params,
            # Market Context
            "market_category": self.market.market_category,
            "market_close_ts": datetime.fromtimestamp(self.market.market_close_ts, tz=timezone.utc) if self.market.market_close_ts else None,
            "hours_to_settlement": self.market.hours_to_settlement,
            "trades_in_market": self.market.trades_in_market,
            # Price Context
            "no_price_at_signal": self.no_price_at_signal,
            "bucket_5c": self.bucket_5c,
            # Orderbook
            "best_bid_cents": self.orderbook.best_bid_cents,
            "best_ask_cents": self.orderbook.best_ask_cents,
            "bid_ask_spread_cents": self.orderbook.bid_ask_spread_cents,
            "spread_tier": self.orderbook.spread_tier.value,
            "bid_size_contracts": self.orderbook.bid_size_contracts,
            "ask_size_contracts": self.orderbook.ask_size_contracts,
            # Position
            "existing_position_count": self.position.existing_position_count,
            "existing_position_side": self.position.existing_position_side,
            "is_reentry": self.position.is_reentry,
            "entry_number": self.position.entry_number,
            "balance_cents": self.position.balance_cents,
            "open_position_count": self.position.open_position_count,
            # Order
            "action": self.action,
            "side": self.side,
            "order_price_cents": self.order_price_cents,
            "order_quantity": self.order_quantity,
            "order_type": self.order_type,
            # Timing
            "placed_at": placed_dt,
            "hour_of_day_utc": placed_dt.hour,
            "day_of_week": placed_dt.weekday(),
            "calendar_week": placed_dt.strftime("%Y-W%W"),
            # Fill
            "fill_count": fill_count,
            "fill_avg_price_cents": fill_avg_price_cents,
            "filled_at": filled_dt,
            "time_to_fill_ms": time_to_fill_ms,
            "slippage_cents": slippage_cents,
            # Metadata
            "strategy_version": self.strategy_version,
        }


def extract_rlm_signal_params(
    market_trade_state: Any,
    is_reentry: bool = False,
    position_scale: str = "1x",
    aggressive_pricing: bool = False,
) -> Dict[str, Any]:
    """
    Extract RLM-specific signal parameters from MarketTradeState.

    Args:
        market_trade_state: MarketTradeState object from RLM service
        is_reentry: Whether this is adding to existing position
        position_scale: Position size multiplier ("1x", "1.5x", "2x")
        aggressive_pricing: Whether aggressive pricing was used

    Returns:
        Dictionary of RLM signal parameters for JSONB storage
    """
    return {
        "yes_trades": market_trade_state.yes_trades,
        "no_trades": market_trade_state.no_trades,
        "total_trades": market_trade_state.total_trades,
        "yes_ratio": round(market_trade_state.yes_ratio, 4),
        "price_drop_cents": market_trade_state.price_drop,
        "true_market_open": market_trade_state.true_market_open,
        "last_yes_price": market_trade_state.last_yes_price,
        "using_tmo": market_trade_state.using_tmo,
        "is_reentry": is_reentry,
        "entry_yes_ratio": market_trade_state.entry_yes_ratio,
        "position_scale": position_scale,
        "signal_trigger_count": market_trade_state.signal_trigger_count,
        "aggressive_pricing": aggressive_pricing,
    }
