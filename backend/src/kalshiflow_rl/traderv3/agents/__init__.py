"""
Agent Module for TRADER V3.

Contains agents for the entity-based trading pipeline:
- BaseAgent: Abstract base class for all agents
- RedditEntityAgent: Streams Reddit posts, extracts entities with sentiment
- PriceImpactAgent: Transforms entity sentiment into price impact signals
"""

from .base_agent import BaseAgent, AgentStatus
from .reddit_entity_agent import RedditEntityAgent, RedditEntityAgentConfig
from .price_impact_agent import PriceImpactAgent, PriceImpactAgentConfig

__all__ = [
    # Base
    "BaseAgent",
    "AgentStatus",
    # Agents
    "RedditEntityAgent",
    "RedditEntityAgentConfig",
    "PriceImpactAgent",
    "PriceImpactAgentConfig",
]
