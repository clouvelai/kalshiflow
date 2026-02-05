"""
Single-Event Arbitrage System.

Exploits probability completeness violations within mutually exclusive Kalshi events.
When the sum of YES prices across all outcomes deviates from 100 cents, a guaranteed
arb opportunity exists.
"""

from .index import EventArbIndex, EventMeta, MarketMeta, ArbOpportunity, ArbLeg
from .monitor import EventArbMonitor
from .coordinator import SingleArbCoordinator

__all__ = [
    "EventArbIndex",
    "EventMeta",
    "MarketMeta",
    "ArbOpportunity",
    "ArbLeg",
    "EventArbMonitor",
    "SingleArbCoordinator",
]
