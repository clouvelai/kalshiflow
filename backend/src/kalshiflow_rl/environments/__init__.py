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
"""

from .market_agnostic_env import MarketAgnosticKalshiEnv
from .session_data_loader import SessionDataLoader, SessionData
from .feature_extractors import (
    extract_market_agnostic_features,
    extract_temporal_features,
    build_observation_from_session_data,
)

__all__ = [
    'MarketAgnosticKalshiEnv',
    'SessionDataLoader', 
    'SessionData',
    'extract_market_agnostic_features',
    'extract_temporal_features',
    'build_observation_from_session_data',
]