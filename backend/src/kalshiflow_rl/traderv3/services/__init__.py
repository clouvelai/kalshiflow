"""
V3 Trader Services.

Services that provide specific functionality for the V3 trader.
"""

from .trading_decision_service import (
    TradingDecisionService,
    TradingStrategy,
    TradingDecision
)
from .whale_tracker import WhaleTracker, BigBet

__all__ = [
    "TradingDecisionService",
    "TradingStrategy",
    "TradingDecision",
    "WhaleTracker",
    "BigBet",
]