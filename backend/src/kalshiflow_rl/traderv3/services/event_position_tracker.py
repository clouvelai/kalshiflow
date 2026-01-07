"""
Event Position Tracker for V3 Trader.

Tracks positions grouped by event_ticker to detect and prevent correlated
exposure in mutually exclusive markets. This is a risk management layer
that can alert or block trades that would create guaranteed losses.

Key Math (for all-NO positions on binary mutually exclusive markets):
    NO_sum = N * 100 - YES_sum  (where N = number of markets)
    P&L per contract = 100 - NO_sum = YES_sum - 100

    If YES_sum > 100 -> NO is cheap -> ARBITRAGE (guaranteed profit)
    If YES_sum < 100 -> NO is expensive -> GUARANTEED LOSS

Thresholds (for binary markets, N=2):
    ARBITRAGE: YES_sum > 105c (NO is cheap, safe to hold)
    NORMAL: 95c <= YES_sum <= 105c (fair pricing)
    HIGH_RISK: YES_sum < 95c (NO is expensive)
    GUARANTEED_LOSS: YES_sum < 100c (will lose money)

Usage:
    tracker = EventPositionTracker(tracked_markets, state_container, config)

    # Before executing a trade
    allowed, reason, risk_level = await tracker.check_before_trade(ticker, "no", 50)
    if not allowed and config.event_exposure_action == "block":
        # Reject trade

    # Get all event groups for UI display
    groups = tracker.get_event_groups()
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from ..state.tracked_markets import TrackedMarketsState
    from ..core.state_container import V3StateContainer
    from ..config.environment import V3Config

logger = logging.getLogger("kalshiflow_rl.traderv3.services.event_position_tracker")


# Risk level type
RiskLevel = Literal["ARBITRAGE", "NORMAL", "HIGH_RISK", "GUARANTEED_LOSS"]


@dataclass
class MarketPositionSummary:
    """Summary of position in a single market within an event."""
    ticker: str
    yes_price: int  # Current YES price in cents
    no_price: int   # Current NO price in cents (100 - yes_price)
    position_count: int  # Number of contracts (positive = YES, negative = NO)
    position_side: str  # "yes", "no", or "none"
    has_position: bool


@dataclass
class EventGroup:
    """
    Aggregated view of positions within an event.

    For RLM strategy, we primarily care about all-NO positions in
    mutually exclusive binary markets.
    """
    event_ticker: str
    markets: Dict[str, MarketPositionSummary] = field(default_factory=dict)
    yes_sum: int = 0    # Sum of YES prices in cents
    no_sum: int = 0     # Sum of NO prices (N*100 - yes_sum for binary)
    risk_level: RiskLevel = "NORMAL"
    position_count: int = 0  # Total contracts across all markets
    has_mixed_sides: bool = False  # True if both YES and NO positions exist
    pnl_estimate: int = 0  # Estimated P&L in cents (YES_sum - 100 for binary)

    @property
    def market_count(self) -> int:
        """Number of markets in this event."""
        return len(self.markets)

    @property
    def markets_with_positions(self) -> int:
        """Number of markets with open positions."""
        return sum(1 for m in self.markets.values() if m.has_position)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_ticker": self.event_ticker,
            "markets": {
                ticker: {
                    "ticker": m.ticker,
                    "yes_price": m.yes_price,
                    "no_price": m.no_price,
                    "position_count": m.position_count,
                    "position_side": m.position_side,
                    "has_position": m.has_position,
                }
                for ticker, m in self.markets.items()
            },
            "yes_sum": self.yes_sum,
            "no_sum": self.no_sum,
            "risk_level": self.risk_level,
            "position_count": self.position_count,
            "has_mixed_sides": self.has_mixed_sides,
            "pnl_estimate": self.pnl_estimate,
            "market_count": self.market_count,
            "markets_with_positions": self.markets_with_positions,
        }


class EventPositionTracker:
    """
    Track and analyze positions grouped by event_ticker.

    This is a derived aggregation layer (not primary state) that queries
    TrackedMarketsState for market info and StateContainer for positions,
    computing event-level risk metrics on demand with version-based caching.

    Responsibilities:
        1. Group tracked markets by event_ticker
        2. Calculate YES/NO sums for each event group
        3. Classify risk levels (ARBITRAGE, NORMAL, HIGH_RISK, GUARANTEED_LOSS)
        4. Pre-trade exposure checks to prevent guaranteed losses
    """

    def __init__(
        self,
        tracked_markets: 'TrackedMarketsState',
        state_container: 'V3StateContainer',
        config: 'V3Config',
    ):
        """
        Initialize event position tracker.

        Args:
            tracked_markets: TrackedMarketsState for market info and prices
            state_container: V3StateContainer for current positions
            config: V3Config for threshold settings
        """
        self._tracked_markets = tracked_markets
        self._state_container = state_container
        self._config = config

        # Version-based caching
        self._cache: Optional[Dict[str, EventGroup]] = None
        self._cache_version: int = 0

        # Stats tracking
        self._check_count = 0
        self._block_count = 0
        self._alert_count = 0

        logger.info(
            f"EventPositionTracker initialized "
            f"(enabled={config.event_tracking_enabled}, "
            f"action={config.event_exposure_action})"
        )

    def _get_current_version(self) -> int:
        """
        Get combined version from tracked markets and trading state.

        This version changes when either:
        - Markets are added/removed/updated (tracked_markets._version)
        - Positions change (state_container.trading_state_version)
        """
        tm_version = self._tracked_markets._version if self._tracked_markets else 0
        ts_version = self._state_container.trading_state_version
        return tm_version + ts_version

    def get_event_groups(self) -> Dict[str, EventGroup]:
        """
        Get all event groups with computed risk levels.

        Uses version-based caching to avoid redundant computation.

        Returns:
            Dict mapping event_ticker -> EventGroup
        """
        current_version = self._get_current_version()

        # Return cached if version hasn't changed
        if self._cache is not None and self._cache_version == current_version:
            return self._cache

        # Recompute
        self._cache = self._compute_event_groups()
        self._cache_version = current_version

        return self._cache

    def _compute_event_groups(self) -> Dict[str, EventGroup]:
        """
        Compute event groups from current tracked markets and positions.

        Returns:
            Dict mapping event_ticker -> EventGroup
        """
        groups: Dict[str, EventGroup] = {}

        if not self._tracked_markets:
            return groups

        # Get current positions
        trading_state = self._state_container.trading_state
        positions = trading_state.positions if trading_state else {}

        # Group markets by event_ticker (use public get_all() method)
        for market in self._tracked_markets.get_all():
            ticker = market.ticker
            event_ticker = market.event_ticker
            if not event_ticker:
                continue  # Skip markets without event_ticker

            # Create group if needed
            if event_ticker not in groups:
                groups[event_ticker] = EventGroup(event_ticker=event_ticker)

            group = groups[event_ticker]

            # Get YES price (use yes_ask if available, else price field)
            yes_price = market.yes_ask if market.yes_ask > 0 else market.price
            if yes_price <= 0:
                yes_price = 50  # Default to 50c if no price data

            no_price = 100 - yes_price

            # Check if we have a position in this market
            position = positions.get(ticker, {})
            position_count = position.get("position", 0)

            if position_count > 0:
                position_side = "yes"
            elif position_count < 0:
                position_side = "no"
            else:
                position_side = "none"

            has_position = position_count != 0

            # Add market to group
            group.markets[ticker] = MarketPositionSummary(
                ticker=ticker,
                yes_price=yes_price,
                no_price=no_price,
                position_count=abs(position_count),
                position_side=position_side,
                has_position=has_position,
            )

            # Update group aggregates
            group.yes_sum += yes_price
            group.position_count += abs(position_count)

            # Check for mixed sides
            if has_position:
                existing_sides = {
                    m.position_side for m in group.markets.values()
                    if m.has_position and m.position_side != "none"
                }
                group.has_mixed_sides = len(existing_sides) > 1

        # Calculate NO sum and risk level for each group
        for event_ticker, group in groups.items():
            n_markets = group.market_count
            if n_markets == 0:
                continue

            # NO_sum = N * 100 - YES_sum for binary markets
            group.no_sum = n_markets * 100 - group.yes_sum

            # P&L estimate for all-NO positions: YES_sum - 100
            # (positive = arbitrage profit, negative = loss)
            if n_markets == 2:  # Binary event
                group.pnl_estimate = group.yes_sum - 100
            else:
                # For N-ary events, P&L = YES_sum - 100 (same formula)
                group.pnl_estimate = group.yes_sum - 100

            # Classify risk (pass has_mixed_sides to skip all-NO math for mixed)
            group.risk_level = self._classify_risk(
                group.yes_sum,
                n_markets,
                group.markets_with_positions,
                group.has_mixed_sides,
            )

        return groups

    def _classify_risk(
        self,
        yes_sum: int,
        n_markets: int,
        markets_with_positions: int,
        has_mixed_sides: bool = False,
    ) -> RiskLevel:
        """
        Classify risk for positions in mutually exclusive markets.

        The math for binary events (N=2) with ALL-NO positions:
            NO_sum = 200 - YES_sum
            P&L = 100 - NO_sum = YES_sum - 100

            YES_sum > 100 -> P&L positive -> ARBITRAGE
            YES_sum < 100 -> P&L negative -> LOSS

        NOTE: For mixed-side positions (YES + NO in same event), the all-NO
        math doesn't apply. Mixed positions create directional exposure where
        P&L varies by outcome (not guaranteed). We return NORMAL for mixed.

        Args:
            yes_sum: Sum of YES prices in cents
            n_markets: Number of markets in the event
            markets_with_positions: Number of markets with open positions
            has_mixed_sides: True if both YES and NO positions exist

        Returns:
            Risk level classification
        """
        # Only compute risk if we have positions in multiple markets
        if markets_with_positions < 2:
            return "NORMAL"

        # Mixed-side positions have different math (directional, not guaranteed)
        # Skip all-NO risk classification - the formula doesn't apply
        if has_mixed_sides:
            return "NORMAL"

        # For binary events (most common case) with ALL-NO positions
        # Check thresholds from HIGHEST to LOWEST to avoid dead code
        #
        # Use config thresholds (default: risk=95, loss=100)
        # Arbitrage threshold = 200 - risk_threshold (symmetric around 100)
        risk_threshold = getattr(self._config, 'event_risk_threshold_cents', 95)
        loss_threshold = getattr(self._config, 'event_loss_threshold_cents', 100)
        arb_threshold = 200 - risk_threshold  # 105 by default

        # Thresholds reference (for binary, N=2, default thresholds):
        # | YES_sum  | NO_sum (200-YES) | Risk Level       | Action           |
        # |----------|------------------|------------------|------------------|
        # | >= 105c  | <= 95c           | ARBITRAGE        | Safe - NO cheap  |
        # | 100-104c | 96-100c          | NORMAL           | Fair pricing     |
        # | 95-99c   | 101-105c         | HIGH_RISK        | Alert            |
        # | < 95c    | > 105c           | GUARANTEED_LOSS  | BLOCK            |

        if yes_sum >= arb_threshold:
            # YES_sum >= 105 means NO is cheap = arbitrage (guaranteed profit)
            return "ARBITRAGE"
        elif yes_sum >= loss_threshold:
            # YES_sum 100-104 = normal/fair pricing
            return "NORMAL"
        elif yes_sum >= risk_threshold:
            # YES_sum 95-99 = NO getting expensive, high risk
            return "HIGH_RISK"
        else:
            # YES_sum < 95 means NO_sum > 105 = guaranteed loss
            return "GUARANTEED_LOSS"

    async def check_before_trade(
        self,
        ticker: str,
        side: str,
        price: int,
    ) -> Tuple[bool, str, RiskLevel]:
        """
        Pre-trade exposure check for event-level risk.

        Called before executing a trade to check if it would create
        a guaranteed loss or high-risk exposure across an event.

        Args:
            ticker: Market ticker to trade
            side: "yes" or "no"
            price: Trade price in cents

        Returns:
            Tuple of:
                - allowed: bool - True if trade should proceed
                - reason: str - Human-readable explanation
                - risk_level: RiskLevel - Current/projected risk level
        """
        self._check_count += 1

        # Skip if tracking disabled
        if not getattr(self._config, 'event_tracking_enabled', True):
            return (True, "Event tracking disabled", "NORMAL")

        # Get market info
        market = self._tracked_markets.get_market(ticker) if self._tracked_markets else None
        if not market or not market.event_ticker:
            return (True, "Market not tracked or no event_ticker", "NORMAL")

        event_ticker = market.event_ticker

        # Get event groups
        groups = self.get_event_groups()
        group = groups.get(event_ticker)

        if not group:
            return (True, "No event group found", "NORMAL")

        # Handle mixed-side positions - all-NO math doesn't apply
        if group.has_mixed_sides:
            reason = (
                f"Mixed-side positions in {event_ticker} - "
                f"all-NO risk math not applicable (directional exposure)"
            )
            logger.info(reason)
            return (True, reason, "NORMAL")

        # Only check for NO trades (RLM strategy focus)
        if side != "no":
            return (True, f"YES trade - event check skipped", group.risk_level)

        # Check if adding this NO would create mixed-side exposure
        # (i.e., there are existing YES positions in other markets)
        existing_yes_positions = any(
            m.position_side == "yes" and m.ticker != ticker
            for m in group.markets.values()
        )
        if existing_yes_positions:
            reason = (
                f"Adding NO to {ticker} would create mixed-side exposure in {event_ticker} - "
                f"allowing trade (directional exposure, not guaranteed loss)"
            )
            logger.info(reason)
            return (True, reason, "NORMAL")

        # Calculate projected YES sum if we add this NO position
        # Adding NO at price X means the YES price is (100 - X)
        projected_yes_price = 100 - price

        # Get current YES price for this market
        current_market = group.markets.get(ticker)
        current_yes_price = current_market.yes_price if current_market else 50

        # Project new YES sum (replace current with projected)
        projected_yes_sum = group.yes_sum - current_yes_price + projected_yes_price

        # Check if this would create positions in multiple markets
        markets_with_pos = group.markets_with_positions
        current_has_pos = current_market.has_position if current_market else False
        if not current_has_pos:
            markets_with_pos += 1  # This trade would add a new position

        # Classify projected risk (has_mixed_sides=False since we filtered above)
        projected_risk = self._classify_risk(
            projected_yes_sum,
            group.market_count,
            markets_with_pos,
            has_mixed_sides=False,  # Already filtered out mixed scenarios
        )

        # Decide based on projected risk
        if projected_risk == "GUARANTEED_LOSS":
            self._block_count += 1
            reason = (
                f"BLOCKED: Adding NO @ {price}c to {ticker} would create "
                f"guaranteed loss (YES_sum={projected_yes_sum}c < 100c)"
            )
            logger.warning(reason)
            return (False, reason, projected_risk)

        elif projected_risk == "HIGH_RISK":
            self._alert_count += 1
            reason = (
                f"HIGH_RISK: NO @ {price}c on {ticker} "
                f"(YES_sum={projected_yes_sum}c, event={event_ticker})"
            )
            logger.warning(reason)
            # Allow but alert - actual blocking depends on config
            action = getattr(self._config, 'event_exposure_action', 'alert')
            allowed = action != "block"
            return (allowed, reason, projected_risk)

        elif projected_risk == "ARBITRAGE":
            reason = (
                f"ARBITRAGE: NO is cheap on {event_ticker} "
                f"(YES_sum={projected_yes_sum}c > 105c)"
            )
            logger.info(reason)
            return (True, reason, projected_risk)

        else:
            return (True, f"Normal exposure on {event_ticker}", projected_risk)

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        groups = self.get_event_groups()

        # Count by risk level
        risk_counts = {"ARBITRAGE": 0, "NORMAL": 0, "HIGH_RISK": 0, "GUARANTEED_LOSS": 0}
        for group in groups.values():
            if group.markets_with_positions >= 2:  # Only count if multi-market exposure
                risk_counts[group.risk_level] += 1

        return {
            "enabled": getattr(self._config, 'event_tracking_enabled', True),
            "action": getattr(self._config, 'event_exposure_action', 'alert'),
            "event_groups": len(groups),
            "events_with_multi_positions": sum(
                1 for g in groups.values() if g.markets_with_positions >= 2
            ),
            "risk_counts": risk_counts,
            "checks": self._check_count,
            "blocks": self._block_count,
            "alerts": self._alert_count,
            "cache_version": self._cache_version,
        }

    def get_event_groups_for_broadcast(self) -> Dict[str, Any]:
        """
        Get event groups formatted for WebSocket broadcast.

        Returns minimal data needed for frontend display.
        """
        groups = self.get_event_groups()

        # Include groups with multiple markets (for badge counter) or interesting risk levels
        broadcast_groups = {}
        for event_ticker, group in groups.items():
            # Show event grouping for all multi-market events (enables "1/5" badge)
            # Also show events with non-normal risk (ARB/RISK/LOSS badges)
            if group.market_count > 1 or group.risk_level != "NORMAL":
                broadcast_groups[event_ticker] = group.to_dict()

        return {
            "event_groups": broadcast_groups,
            "stats": {
                "total_events": len(groups),
                "multi_position_events": len(broadcast_groups),
            },
        }
