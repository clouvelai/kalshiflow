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


class BBODepthTier(Enum):
    """Classification of liquidity depth at best bid/offer."""
    THICK = "thick"      # >= 100 contracts
    NORMAL = "normal"    # 20-99 contracts
    THIN = "thin"        # < 20 contracts
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
    def from_orderbook_state(
        cls,
        snapshot: Dict[str, Any],
        tight_spread: int = 2,
        normal_spread: int = 4,
    ) -> 'OrderbookSnapshot':
        """Create from shared orderbook state snapshot.

        Args:
            snapshot: Dict from SharedOrderbookState.get_snapshot()
            tight_spread: Spread threshold for TIGHT tier (default: 2)
            normal_spread: Spread threshold for NORMAL tier (default: 4)
        """
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
            if spread <= tight_spread:
                spread_tier = SpreadTier.TIGHT
            elif spread <= normal_spread:
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
class OrderbookContext:
    """
    Comprehensive orderbook context for trade-time pricing decisions.

    Captures both YES and NO sides with depth information (levels 1-5),
    derived metrics, and freshness indicators.
    """
    # === NO SIDE (Primary for NO orders) ===
    no_best_bid: Optional[int] = None
    no_best_ask: Optional[int] = None
    no_spread: Optional[int] = None

    # Queue depth at BBO (fill probability)
    no_bid_size_at_bbo: Optional[int] = None
    no_ask_size_at_bbo: Optional[int] = None

    # Second level (robustness if BBO disappears)
    no_second_bid: Optional[int] = None
    no_second_ask: Optional[int] = None

    # Total liquidity (levels 1-5)
    no_bid_total_volume: int = 0
    no_ask_total_volume: int = 0

    # === YES SIDE (Cross-reference) ===
    yes_best_bid: Optional[int] = None
    yes_best_ask: Optional[int] = None
    yes_spread: Optional[int] = None

    # === DERIVED METRICS ===
    spread_tier: SpreadTier = SpreadTier.UNKNOWN
    bid_imbalance: Optional[float] = None  # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    bbo_depth_tier: BBODepthTier = BBODepthTier.UNKNOWN  # Ask side depth (entry liquidity for buying NO)
    bid_depth_tier: BBODepthTier = BBODepthTier.UNKNOWN  # Bid side depth (exit liquidity for selling NO)

    # === FRESHNESS ===
    last_update_ms: Optional[int] = None  # Timestamp of last orderbook update
    captured_at: float = field(default_factory=time.time)  # When this context was built
    is_stale: bool = False  # True if orderbook data is >60s old (per-market staleness)

    @classmethod
    def from_orderbook_snapshot(
        cls,
        snapshot: Dict[str, Any],
        stale_threshold_seconds: float = 5.0,
        tight_spread: int = 2,
        normal_spread: int = 4,
    ) -> 'OrderbookContext':
        """
        Create OrderbookContext from SharedOrderbookState.get_snapshot() result.

        Args:
            snapshot: Dict from SharedOrderbookState.get_snapshot()
            stale_threshold_seconds: Max age before data is considered stale
            tight_spread: Spread threshold for TIGHT tier (default: 2)
            normal_spread: Spread threshold for NORMAL tier (default: 4)

        Returns:
            OrderbookContext with all fields populated
        """
        if not snapshot:
            return cls(is_stale=True)

        # Extract raw data
        no_bids = snapshot.get("no_bids", {})
        no_asks = snapshot.get("no_asks", {})
        yes_bids = snapshot.get("yes_bids", {})
        yes_asks = snapshot.get("yes_asks", {})
        last_update_time = snapshot.get("last_update_time", 0)

        # Calculate staleness
        current_time_ms = int(time.time() * 1000)
        is_stale = (current_time_ms - last_update_time) > (stale_threshold_seconds * 1000) if last_update_time else True

        # === NO SIDE ===
        # Sort and extract levels
        no_bid_prices = sorted(no_bids.keys(), reverse=True)  # Descending (best first)
        no_ask_prices = sorted(no_asks.keys())  # Ascending (best first)

        no_best_bid = no_bid_prices[0] if no_bid_prices else None
        no_best_ask = no_ask_prices[0] if no_ask_prices else None
        no_second_bid = no_bid_prices[1] if len(no_bid_prices) > 1 else None
        no_second_ask = no_ask_prices[1] if len(no_ask_prices) > 1 else None

        no_spread = (no_best_ask - no_best_bid) if (no_best_bid is not None and no_best_ask is not None) else None

        no_bid_size_at_bbo = no_bids.get(no_best_bid) if no_best_bid is not None else None
        no_ask_size_at_bbo = no_asks.get(no_best_ask) if no_best_ask is not None else None

        no_bid_total_volume = sum(no_bids.values())
        no_ask_total_volume = sum(no_asks.values())

        # === YES SIDE ===
        yes_bid_prices = sorted(yes_bids.keys(), reverse=True)
        yes_ask_prices = sorted(yes_asks.keys())

        yes_best_bid = yes_bid_prices[0] if yes_bid_prices else None
        yes_best_ask = yes_ask_prices[0] if yes_ask_prices else None
        yes_spread = (yes_best_ask - yes_best_bid) if (yes_best_bid is not None and yes_best_ask is not None) else None

        # === DERIVED METRICS ===
        # Spread tier (based on NO side for NO orders)
        spread_tier = SpreadTier.UNKNOWN
        if no_spread is not None:
            if no_spread <= tight_spread:
                spread_tier = SpreadTier.TIGHT
            elif no_spread <= normal_spread:
                spread_tier = SpreadTier.NORMAL
            else:
                spread_tier = SpreadTier.WIDE

        # Bid imbalance: positive = more bid pressure (bullish for that side)
        bid_imbalance = None
        total_volume = no_bid_total_volume + no_ask_total_volume
        if total_volume > 0:
            bid_imbalance = (no_bid_total_volume - no_ask_total_volume) / total_volume

        # BBO depth tier (based on ask side for buying NO)
        bbo_depth_tier = BBODepthTier.UNKNOWN
        if no_ask_size_at_bbo is not None:
            if no_ask_size_at_bbo >= 100:
                bbo_depth_tier = BBODepthTier.THICK
            elif no_ask_size_at_bbo >= 20:
                bbo_depth_tier = BBODepthTier.NORMAL
            else:
                bbo_depth_tier = BBODepthTier.THIN

        # Bid depth tier (based on bid side for exit liquidity when selling NO)
        bid_depth_tier = BBODepthTier.UNKNOWN
        if no_bid_size_at_bbo is not None:
            if no_bid_size_at_bbo >= 100:
                bid_depth_tier = BBODepthTier.THICK
            elif no_bid_size_at_bbo >= 20:
                bid_depth_tier = BBODepthTier.NORMAL
            else:
                bid_depth_tier = BBODepthTier.THIN

        return cls(
            # NO side
            no_best_bid=no_best_bid,
            no_best_ask=no_best_ask,
            no_spread=no_spread,
            no_bid_size_at_bbo=no_bid_size_at_bbo,
            no_ask_size_at_bbo=no_ask_size_at_bbo,
            no_second_bid=no_second_bid,
            no_second_ask=no_second_ask,
            no_bid_total_volume=no_bid_total_volume,
            no_ask_total_volume=no_ask_total_volume,
            # YES side
            yes_best_bid=yes_best_bid,
            yes_best_ask=yes_best_ask,
            yes_spread=yes_spread,
            # Derived
            spread_tier=spread_tier,
            bid_imbalance=bid_imbalance,
            bbo_depth_tier=bbo_depth_tier,
            bid_depth_tier=bid_depth_tier,
            # Freshness
            last_update_ms=last_update_time,
            captured_at=time.time(),
            is_stale=is_stale,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            # NO side
            "no_best_bid": self.no_best_bid,
            "no_best_ask": self.no_best_ask,
            "no_spread": self.no_spread,
            "no_bid_size_at_bbo": self.no_bid_size_at_bbo,
            "no_ask_size_at_bbo": self.no_ask_size_at_bbo,
            "no_second_bid": self.no_second_bid,
            "no_second_ask": self.no_second_ask,
            "no_bid_total_volume": self.no_bid_total_volume,
            "no_ask_total_volume": self.no_ask_total_volume,
            # YES side
            "yes_best_bid": self.yes_best_bid,
            "yes_best_ask": self.yes_best_ask,
            "yes_spread": self.yes_spread,
            # Derived
            "spread_tier": self.spread_tier.value,
            "bid_imbalance": round(self.bid_imbalance, 4) if self.bid_imbalance is not None else None,
            "bbo_depth_tier": self.bbo_depth_tier.value,
            "bid_depth_tier": self.bid_depth_tier.value,
            # Freshness
            "last_update_ms": self.last_update_ms,
            "captured_at": self.captured_at,
            "is_stale": self.is_stale,
        }

    def should_skip_due_to_staleness(self) -> bool:
        """Check if order should be skipped due to stale orderbook data."""
        return self.is_stale

    def get_recommended_entry_price(
        self,
        aggressive: bool = False,
        max_spread: int = 10,
        trade_rate: Optional[float] = None,
    ) -> Optional[int]:
        """
        Calculate recommended NO entry price based on orderbook context.

        Uses spread-aware pricing similar to existing RLM logic but with
        additional depth considerations and queue position awareness.

        Args:
            aggressive: If True, price closer to ask for faster fill
            max_spread: Maximum spread allowed (returns None if exceeded)
            trade_rate: Actual trade rate (trades/second) for queue wait estimation

        Returns:
            Recommended price in cents, or None if conditions not met
        """
        if self.no_best_bid is None or self.no_best_ask is None:
            return None

        if self.no_spread is None or self.no_spread > max_spread:
            return None

        # Use BBO sizes for depth weighting (more relevant for immediate fills)
        # Total volume (all levels) is used only for wide spread fallback
        bid_vol_bbo = self.no_bid_size_at_bbo or 0
        ask_vol_bbo = self.no_ask_size_at_bbo or 0
        bbo_total = bid_vol_bbo + ask_vol_bbo

        # Base pricing on spread tier
        if self.spread_tier == SpreadTier.TIGHT:
            # Tight spread (<= 2c): price near ask
            base_price = self.no_best_ask - 1 if not aggressive else self.no_best_ask
        elif self.spread_tier == SpreadTier.NORMAL:
            # Normal spread (<= 4c): use BBO depth-weighted midpoint
            if bbo_total > 0:
                # Weight: bid_vol / (bid_vol + ask_vol) shifts toward bid when ask is heavy
                weight = bid_vol_bbo / bbo_total
                # Weighted mid: interpolate between bid and ask based on BBO volume balance
                # When ask > bid (negative pressure), weight < 0.5, mid shifts toward bid (cheaper)
                weighted_mid = int(self.no_best_bid + weight * self.no_spread)
                base_price = weighted_mid if not aggressive else self.no_best_ask - 1
            else:
                # Fallback to simple midpoint
                midpoint = (self.no_best_bid + self.no_best_ask) // 2
                base_price = midpoint if not aggressive else self.no_best_ask - 1
        else:
            # Wide spread (> 4c): bid + 75-85% of spread, adjusted by BBO depth
            if bbo_total > 0:
                # Adjust spread percentage based on BBO volume imbalance
                weight = bid_vol_bbo / bbo_total
                # When ask heavy (weight < 0.5), be less aggressive (lower %)
                # When bid heavy (weight > 0.5), can be more aggressive (higher %)
                base_pct = 0.85 if aggressive else 0.75
                adjusted_pct = base_pct * (0.5 + weight)  # Range: base_pct * [0.5, 1.5]
                adjusted_pct = min(0.95, max(0.50, adjusted_pct))  # Clamp to reasonable range
                base_price = self.no_best_bid + int(self.no_spread * adjusted_pct)
            else:
                spread_pct = 0.85 if aggressive else 0.75
                base_price = self.no_best_bid + int(self.no_spread * spread_pct)

        # Adjust for thin liquidity at BBO
        if self.bbo_depth_tier == BBODepthTier.THIN and not aggressive:
            # If liquidity is thin, be more aggressive to ensure fill
            if base_price < self.no_best_ask:
                base_price = min(base_price + 1, self.no_best_ask)

        # Queue position awareness: estimate wait time based on actual trade rate
        # If queue would take >30 seconds to fill, be more aggressive (cross spread)
        if trade_rate is not None and self.no_ask_size_at_bbo is not None:
            # Estimate queue wait time in seconds
            # trade_rate = actual trades per second (from RLM's trade tracking)
            # estimated_wait = queue_size / trade_rate
            if trade_rate > 0:
                estimated_wait_seconds = self.no_ask_size_at_bbo / trade_rate
                # If queue would take >30 seconds to fill, be more aggressive
                if estimated_wait_seconds > 30 and base_price < self.no_best_ask:
                    logger.debug(
                        f"Queue wait ~{estimated_wait_seconds:.0f}s > 30s, increasing aggression"
                    )
                    base_price = min(base_price + 1, self.no_best_ask)

        # Ensure price is within valid Kalshi bounds [1, 99]
        return max(1, min(99, base_price))


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

        # Calculate slippage: difference between fill price and order price
        # This measures execution quality (order-to-fill), not signal-to-fill
        slippage_cents = None
        if self.order_price_cents is not None and fill_avg_price_cents is not None:
            # Positive slippage = paid more than order price (bad)
            # Negative slippage = paid less than order price (good, price improved)
            slippage_cents = fill_avg_price_cents - self.order_price_cents

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
