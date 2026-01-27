"""
Deep Agent - Self-Improving Trading Agent for Kalshi.

This module implements a self-improving agent that:
1. Observes market price impact signals from Reddit entity pipeline
2. Acts by trading or holding based on signal strength
3. Reflects on outcomes and learns
4. Persists learnings to memory files

The agent uses LangChain for tool calling and streaming, with
custom tools for market data, price impacts, and trade execution.
"""

from .agent import SelfImprovingAgent, DeepAgentConfig
from .tools import (
    get_price_impacts,
    get_markets,
    trade,
    get_session_state,
    read_memory,
    write_memory,
)
from .reflection import ReflectionEngine

__all__ = [
    "SelfImprovingAgent",
    "DeepAgentConfig",
    "ReflectionEngine",
    "get_price_impacts",
    "get_markets",
    "trade",
    "get_session_state",
    "read_memory",
    "write_memory",
]
