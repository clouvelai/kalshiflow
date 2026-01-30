"""
Deep Agent - Self-Improving Trading Agent for Kalshi.

This module implements a self-improving agent that:
1. Observes market price impact signals from Reddit entity pipeline
2. Acts by trading or holding based on signal strength
3. Reflects on outcomes and learns
4. Persists learnings to memory files

The agent uses the Anthropic Claude API directly for tool calling,
with custom tools for market data, price impacts, and trade execution.
"""

from .agent import SelfImprovingAgent, DeepAgentConfig
from .reflection import ReflectionEngine

__all__ = [
    "SelfImprovingAgent",
    "DeepAgentConfig",
    "ReflectionEngine",
]
