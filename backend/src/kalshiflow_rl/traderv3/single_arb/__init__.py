"""
Single-Event Arbitrage System.

Exploits probability completeness violations within mutually exclusive Kalshi events.
When the sum of YES prices across all outcomes deviates from 100 cents, a guaranteed
arb opportunity exists.

Also includes MentionsSpecialist for counting literal mentions in mentions markets.
"""

from .index import EventArbIndex, EventMeta, MarketMeta, ArbOpportunity, ArbLeg
from .monitor import EventArbMonitor
from .coordinator import SingleArbCoordinator
from .mentions_tools import (
    LexemePackLite,
    set_mentions_dependencies,
    restore_mentions_state_from_disk,
)

__all__ = [
    "EventArbIndex",
    "EventMeta",
    "MarketMeta",
    "ArbOpportunity",
    "ArbLeg",
    "EventArbMonitor",
    "SingleArbCoordinator",
    "LexemePackLite",
    "set_mentions_dependencies",
    "restore_mentions_state_from_disk",
]
