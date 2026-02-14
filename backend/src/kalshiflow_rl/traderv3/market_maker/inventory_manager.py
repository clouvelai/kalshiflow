"""Inventory manager for market making.

Handles position tracking, inventory skewing, and exposure limit checks.
The QuoteEngine queries this to compute skew adjustments and enforce risk gates.
"""

import logging
from typing import Dict, Optional

from .models import MarketInventory, QuoteConfig

logger = logging.getLogger("kalshiflow_rl.traderv3.market_maker.inventory_manager")


def compute_skew(
    position: int,
    skew_factor: float,
    skew_cap_cents: float,
) -> float:
    """Compute quote skew based on inventory.

    Positive position (long) → negative skew (lower both quotes to sell).
    Negative position (short) → positive skew (raise both quotes to buy).

    Args:
        position: Net position (+ = long YES, - = short YES).
        skew_factor: Multiplier for position-to-skew conversion.
        skew_cap_cents: Maximum skew offset in cents.

    Returns:
        Skew in cents. Negative = lower prices, positive = raise prices.
    """
    raw_skew = -position * skew_factor
    return max(-skew_cap_cents, min(skew_cap_cents, raw_skew))


def check_position_limit(
    position: int,
    side: str,
    max_position: int,
) -> bool:
    """Check if a new quote on the given side would exceed position limits.

    Args:
        position: Current net position.
        side: "bid" (would add long) or "ask" (would add short).
        max_position: Maximum absolute position.

    Returns:
        True if the quote is allowed, False if it would exceed limits.
    """
    if side == "bid" and position >= max_position:
        return False
    if side == "ask" and position <= -max_position:
        return False
    return True


def check_event_exposure(
    current_exposure: int,
    max_exposure: int,
) -> bool:
    """Check if event-level exposure is within limits.

    Args:
        current_exposure: Total absolute position across all event markets.
        max_exposure: Maximum event-level exposure.

    Returns:
        True if within limits.
    """
    return current_exposure < max_exposure


def should_one_side_only(
    position: int,
    max_position: int,
) -> Optional[str]:
    """Determine if we should only quote one side due to position limits.

    Args:
        position: Current net position.
        max_position: Maximum absolute position.

    Returns:
        "bid" if we should only quote bid (to reduce short),
        "ask" if we should only quote ask (to reduce long),
        None if both sides are fine.
    """
    if position >= max_position:
        return "ask"  # Only quote ask to reduce long position
    if position <= -max_position:
        return "bid"  # Only quote bid to reduce short position
    return None


def compute_unrealized_pnl(
    position: int,
    avg_entry: float,
    current_mid: Optional[float],
) -> float:
    """Compute unrealized P&L for a position.

    Args:
        position: Net position (+ = long, - = short).
        avg_entry: Average entry price in cents.
        current_mid: Current midpoint price in cents.

    Returns:
        Unrealized P&L in cents.
    """
    if current_mid is None or position == 0:
        return 0.0
    if position > 0:
        return position * (current_mid - avg_entry)
    else:
        return abs(position) * (avg_entry - current_mid)
