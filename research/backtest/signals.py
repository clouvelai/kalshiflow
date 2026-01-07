"""
Reusable Signal Building Blocks for Trading Strategies.

Purpose:
    Provides helper functions that extract features from MarketState.
    These are the building blocks that strategies use to construct signals.

Key Responsibilities:
    1. **Flow Metrics** - YES/NO trade ratios, volume ratios
    2. **Price Movement** - Price drop/rise from open
    3. **Time Metrics** - Hours to settlement, minutes since first trade
    4. **Velocity** - Trade velocity, volume velocity
    5. **Price Access** - Current YES/NO prices

Architecture Position:
    Used by strategy implementations to compute features:

        from signals import yes_ratio, price_drop, current_no_price

        class MyStrategy:
            def on_trade(self, trade, state):
                if yes_ratio(state) > 0.7 and price_drop(state) > 5:
                    return SignalEntry(...)

Design Principles:
    - Pure functions: Each function takes state and returns a value
    - No side effects: Functions don't modify state
    - Point-in-time: Functions only use data available at current moment
    - Null-safe: Functions handle missing data gracefully
"""

from typing import Optional, TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from .state import MarketState


# =============================================================================
# Flow Metrics
# =============================================================================

def yes_ratio(state: 'MarketState') -> float:
    """
    Ratio of YES trades to total trades.

    Returns:
        Float between 0.0 and 1.0. Returns 0.0 if no trades.

    Example:
        If market has 70 YES trades and 30 NO trades:
        >>> yes_ratio(state)
        0.7
    """
    if state.total_trades == 0:
        return 0.0
    return state.yes_trades / state.total_trades


def no_ratio(state: 'MarketState') -> float:
    """
    Ratio of NO trades to total trades.

    Returns:
        Float between 0.0 and 1.0. Returns 0.0 if no trades.

    Equivalent to: 1.0 - yes_ratio(state)
    """
    if state.total_trades == 0:
        return 0.0
    return state.no_trades / state.total_trades


def yes_volume_ratio(state: 'MarketState') -> float:
    """
    Ratio of YES contracts to total contracts traded.

    Differs from yes_ratio because it weights by trade size.

    Returns:
        Float between 0.0 and 1.0. Returns 0.0 if no volume.

    Example:
        If 100 YES contracts and 50 NO contracts traded:
        >>> yes_volume_ratio(state)
        0.667
    """
    if state.total_contracts == 0:
        return 0.0
    return state.yes_contracts / state.total_contracts


def no_volume_ratio(state: 'MarketState') -> float:
    """
    Ratio of NO contracts to total contracts traded.

    Returns:
        Float between 0.0 and 1.0. Returns 0.0 if no volume.
    """
    if state.total_contracts == 0:
        return 0.0
    return state.no_contracts / state.total_contracts


# =============================================================================
# Price Movement
# =============================================================================

def price_drop(state: 'MarketState') -> int:
    """
    YES price drop from open to current (in cents).

    Positive value = price went down (bearish on YES).
    This is the key RLM signal: YES trades dominate but price drops.

    Returns:
        Integer cents. Positive = price dropped. Returns 0 if prices unavailable.

    Example:
        If YES opened at 60 cents and is now at 52 cents:
        >>> price_drop(state)
        8
    """
    if state.open_price is None or state.last_yes_price is None:
        return 0
    return state.open_price - state.last_yes_price


def price_rise(state: 'MarketState') -> int:
    """
    YES price rise from open to current (in cents).

    Positive value = price went up (bullish on YES).

    Returns:
        Integer cents. Positive = price rose. Returns 0 if prices unavailable.

    Equivalent to: -price_drop(state)
    """
    return -price_drop(state)


def price_change_pct(state: 'MarketState') -> float:
    """
    Percentage price change from open.

    Returns:
        Float percentage (e.g., -5.0 for 5% drop). Returns 0.0 if unavailable.

    Example:
        If YES opened at 60 and is now 54 (drop of 6):
        >>> price_change_pct(state)
        -10.0  # 6/60 * 100
    """
    if state.open_price is None or state.last_yes_price is None:
        return 0.0
    if state.open_price == 0:
        return 0.0
    return ((state.last_yes_price - state.open_price) / state.open_price) * 100


# =============================================================================
# Trade/Volume Counts
# =============================================================================

def total_trades(state: 'MarketState') -> int:
    """
    Total number of trades in market.

    Returns:
        Integer count of all trades (YES + NO).
    """
    return state.total_trades


def total_volume(state: 'MarketState') -> int:
    """
    Total contracts traded in market.

    Returns:
        Integer count of all contracts (YES + NO contracts).
    """
    return state.total_contracts


def yes_trades(state: 'MarketState') -> int:
    """Total number of YES side trades."""
    return state.yes_trades


def no_trades(state: 'MarketState') -> int:
    """Total number of NO side trades."""
    return state.no_trades


def yes_contracts(state: 'MarketState') -> int:
    """Total YES contracts traded."""
    return state.yes_contracts


def no_contracts(state: 'MarketState') -> int:
    """Total NO contracts traded."""
    return state.no_contracts


# =============================================================================
# Time Metrics
# =============================================================================

def hours_to_settlement(state: 'MarketState', current_time: datetime) -> Optional[float]:
    """
    Hours until market settlement/close.

    Args:
        state: MarketState with close_time
        current_time: Current timestamp

    Returns:
        Float hours, or None if close_time unknown.

    Example:
        If market closes in 90 minutes:
        >>> hours_to_settlement(state, now)
        1.5
    """
    if state.close_time is None:
        return None
    delta = state.close_time - current_time
    return delta.total_seconds() / 3600


def minutes_to_settlement(state: 'MarketState', current_time: datetime) -> Optional[float]:
    """
    Minutes until market settlement/close.

    Args:
        state: MarketState with close_time
        current_time: Current timestamp

    Returns:
        Float minutes, or None if close_time unknown.
    """
    if state.close_time is None:
        return None
    delta = state.close_time - current_time
    return delta.total_seconds() / 60


def minutes_since_first_trade(state: 'MarketState', current_time: datetime) -> Optional[float]:
    """
    Minutes since first trade in market.

    Useful for detecting "young" markets where signals are less reliable.

    Args:
        state: MarketState with first_trade_time
        current_time: Current timestamp

    Returns:
        Float minutes, or None if first_trade_time unknown.
    """
    if state.first_trade_time is None:
        return None
    delta = current_time - state.first_trade_time
    return delta.total_seconds() / 60


def market_age_hours(state: 'MarketState', current_time: datetime) -> Optional[float]:
    """
    Hours since first trade in market.

    Args:
        state: MarketState with first_trade_time
        current_time: Current timestamp

    Returns:
        Float hours, or None if first_trade_time unknown.
    """
    minutes = minutes_since_first_trade(state, current_time)
    if minutes is None:
        return None
    return minutes / 60


# =============================================================================
# Velocity Metrics
# =============================================================================

def trade_velocity(state: 'MarketState', current_time: datetime, window_minutes: float = 5.0) -> float:
    """
    Approximate trades per minute.

    Uses total trades / time since first trade.
    For windowed velocity, would need trade timestamps in state.

    Args:
        state: MarketState
        current_time: Current timestamp
        window_minutes: Not used in simple calculation (future enhancement)

    Returns:
        Float trades per minute. Returns 0.0 if < 1 minute of data.
    """
    minutes = minutes_since_first_trade(state, current_time)
    if minutes is None or minutes < 1:
        return 0.0
    return state.total_trades / minutes


def volume_velocity(state: 'MarketState', current_time: datetime) -> float:
    """
    Contracts per minute.

    Args:
        state: MarketState
        current_time: Current timestamp

    Returns:
        Float contracts per minute. Returns 0.0 if < 1 minute of data.
    """
    minutes = minutes_since_first_trade(state, current_time)
    if minutes is None or minutes < 1:
        return 0.0
    return state.total_contracts / minutes


# =============================================================================
# Current Prices
# =============================================================================

def current_yes_price(state: 'MarketState') -> Optional[int]:
    """
    Current YES price (last trade price).

    Returns:
        Integer cents, or None if no YES trades yet.
    """
    return state.last_yes_price


def current_no_price(state: 'MarketState') -> Optional[int]:
    """
    Current NO price (100 - last YES price).

    This is the price you'd pay to buy NO.

    Returns:
        Integer cents, or None if no trades yet.

    Example:
        If last YES price is 65 cents:
        >>> current_no_price(state)
        35
    """
    if state.last_yes_price is None:
        return None
    return 100 - state.last_yes_price


def open_price(state: 'MarketState') -> Optional[int]:
    """
    Opening YES price (first trade price).

    Returns:
        Integer cents, or None if no trades yet.
    """
    return state.open_price


# =============================================================================
# Composite Signals
# =============================================================================

def rlm_score(state: 'MarketState') -> float:
    """
    Reverse Line Movement score.

    Combines YES ratio and price drop into single score.
    Higher score = stronger RLM signal.

    Score = yes_ratio * (price_drop / 10)

    Returns:
        Float score >= 0. Higher = stronger RLM signal.

    Example:
        With yes_ratio=0.75 and price_drop=8:
        >>> rlm_score(state)
        0.6  # 0.75 * 0.8
    """
    ratio = yes_ratio(state)
    drop = max(0, price_drop(state))
    return ratio * (drop / 10)


def flow_price_divergence(state: 'MarketState') -> float:
    """
    Measure of divergence between flow direction and price direction.

    Positive = YES flow dominates but price dropping (bearish divergence)
    Negative = NO flow dominates but price rising (bullish divergence)

    Returns:
        Float between -1 and +1 approximately.

    Formula:
        (yes_ratio - 0.5) * (-price_change_pct / 10)

    Example:
        YES ratio 0.8, price dropped 10%:
        >>> flow_price_divergence(state)
        0.3  # (0.8-0.5) * 1.0
    """
    ratio = yes_ratio(state)
    pct_change = price_change_pct(state)

    # Flow bias: +0.3 if 80% YES, -0.3 if 20% YES
    flow_bias = ratio - 0.5

    # Price direction: +1 if price dropped 10%, -1 if rose 10%
    price_direction = -pct_change / 10

    return flow_bias * price_direction
