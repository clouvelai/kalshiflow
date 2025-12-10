"""
Market-agnostic RL environments for Kalshi trading.

This package provides a fresh implementation of session-based, market-agnostic
reinforcement learning environments for Kalshi prediction markets.

Key architectural principles:
- Market-agnostic: Model never sees market tickers or market-specific metadata
- Session-based: Episodes generated from session_id data with guaranteed continuity  
- Unified metrics: Same position tracking for training and inference matching Kalshi API
- Primitive actions: Simple action space allows agent to discover complex strategies
- Simplified rewards: Reward = portfolio value change only

Classes:
    MarketAgnosticKalshiEnv: Core RL environment using session-based episodes
    SessionDataLoader: Loads historical data by session_id for episode generation
    SessionData: Data structure holding session orderbook data and metadata
    PrimitiveActionSpace: 5-action stateless action space for immediate market orders
    PrimitiveActions: Enumeration of the 5 primitive actions (HOLD + 4 NOW)
"""

from .market_agnostic_env import MarketAgnosticKalshiEnv, SessionConfig
from .session_data_loader import SessionDataLoader, SessionData
from .feature_extractors import (
    extract_market_agnostic_features,
    extract_temporal_features,
    build_observation_from_session_data,
)
from .action_space import PrimitiveActionSpace, PrimitiveActions, primitive_action_space

__all__ = [
    'MarketAgnosticKalshiEnv',
    'SessionConfig',
    'SessionDataLoader', 
    'SessionData',
    'extract_market_agnostic_features',
    'extract_temporal_features',
    'build_observation_from_session_data',
    'PrimitiveActionSpace',
    'PrimitiveActions',
    'primitive_action_space',
]