"""
Market-agnostic feature extraction for Kalshi RL environments.

This module provides universal feature extraction functions that work identically
across all Kalshi markets. Features are normalized to probability space [0,1] and
the model never sees market tickers or market-specific metadata.

CRITICAL PRICE FORMAT CONVENTION:
- INPUT: Raw orderbook data with prices in integer cents (1-99)
- OUTPUT: Normalized features in probability space (0.01-0.99) 
- CONVERSION: All price features = cents / 100.0
- PURPOSE: Model learns universal patterns without market-specific bias
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import logging

from .session_data_loader import SessionData, SessionDataPoint

logger = logging.getLogger(__name__)


def extract_market_agnostic_features(orderbook_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract market-agnostic features from orderbook data.
    
    All features are normalized to [0,1] probability space and work identically
    across different Kalshi markets. The model never sees market tickers.
    
    PRICE CONVERSION: Input prices in cents (1-99) → Output probability (0.01-0.99)
    - best_yes_price_norm = yes_price_cents / 100.0  
    - spread_norm = spread_cents / 100.0
    - mid_price_norm = mid_price_cents / 100.0
    
    Args:
        orderbook_data: Orderbook snapshot with prices in integer cents (1-99)
        
    Returns:
        Dictionary of normalized features in [0,1] probability range:
        - All price features converted from cents to probability
        - Volume features normalized to [0,1] 
        - Ratios and indicators naturally in [-1,1] or [0,1]
    """
    # Implementation placeholder - will be completed in M4
    features = {
        # Price and spread features (cents → probability conversion)
        'best_yes_price_norm': 0.5,        # yes_price_cents / 100.0 → [0.01, 0.99]
        'best_no_price_norm': 0.5,         # no_price_cents / 100.0 → [0.01, 0.99]  
        'spread_norm': 0.1,                # spread_cents / 100.0 → [0.01, 0.99]
        'mid_price_norm': 0.5,             # mid_price_cents / 100.0 → [0.01, 0.99]
        
        # Volume and liquidity features
        'yes_volume_norm': 0.5,            # YES side volume / max_volume
        'no_volume_norm': 0.5,             # NO side volume / max_volume
        'volume_imbalance': 0.0,           # (yes_vol - no_vol) / total_vol
        'total_liquidity_norm': 0.5,      # Total book depth / max_depth
        
        # Order book shape features  
        'book_depth_ratio': 0.5,          # Levels with orders / max_levels
        'price_clustering': 0.5,          # Clustering around round prices
        'volatility_estimate': 0.1,       # Recent price movement
        
        # Arbitrage and efficiency features
        'arbitrage_opportunity': 0.0,     # YES + NO price deviation from 100
        'market_efficiency': 1.0,         # How close to efficient pricing
        'momentum_indicator': 0.0,        # Recent directional bias
    }
    
    logger.debug(f"Extracted {len(features)} market-agnostic features")
    return features


def extract_temporal_features(
    current_data: SessionDataPoint,
    historical_data: List[SessionDataPoint]
) -> Dict[str, float]:
    """
    Extract temporal features from session data.
    
    Captures time gaps, activity bursts, and market momentum that
    are universal across all Kalshi markets.
    
    Args:
        current_data: Current session data point
        historical_data: Previous session data points for context
        
    Returns:
        Dictionary of temporal features in [0,1] range
    """
    # Implementation placeholder - will be completed in M4
    features = {
        # Time-based features
        'time_since_last_update': 0.1,    # Normalized time gap
        'activity_burst_indicator': 0.0,  # Recent activity spike
        'quiet_period_indicator': 0.0,    # Prolonged low activity
        'time_of_day_norm': 0.5,          # Hour of day / 24
        
        # Activity momentum features
        'update_frequency': 0.5,          # Updates per minute, normalized
        'volume_acceleration': 0.0,       # Change in volume velocity
        'price_momentum': 0.0,            # Sustained directional movement
        'volatility_regime': 0.1,         # Current volatility vs historical
        
        # Multi-market coordination features
        'cross_market_activity': 0.5,    # Activity across all markets
        'market_divergence': 0.0,        # How much markets are diverging
        'synchronization_score': 1.0,    # How synchronized updates are
    }
    
    logger.debug(f"Extracted {len(features)} temporal features")
    return features


def extract_portfolio_features(
    position_data: Dict[str, Any],
    portfolio_value: float,
    cash_balance: float
) -> Dict[str, float]:
    """
    Extract portfolio state features for observation.
    
    Provides position and portfolio context without exposing market identities.
    
    Args:
        position_data: Current positions across markets
        portfolio_value: Total portfolio value
        cash_balance: Available cash
        
    Returns:
        Dictionary of portfolio features in [0,1] range
    """
    # Implementation placeholder - will be completed in M4
    features = {
        # Portfolio composition
        'cash_ratio': 0.8,                # Cash / total_portfolio_value
        'position_ratio': 0.2,           # Position value / total_portfolio_value
        'leverage': 0.1,                  # Total position size / portfolio_value
        
        # Position characteristics
        'position_count': 0.1,            # Number of positions / max_positions
        'position_diversity': 0.5,       # Diversity across markets
        'long_short_ratio': 0.5,         # Long positions / total positions
        
        # Risk metrics
        'concentration_risk': 0.2,       # Largest position / total portfolio
        'correlation_risk': 0.1,         # Estimated portfolio correlation
        'unrealized_pnl_ratio': 0.0,     # Unrealized P&L / portfolio_value
    }
    
    logger.debug(f"Extracted {len(features)} portfolio features") 
    return features


def build_observation_from_session_data(
    session_data: SessionDataPoint,
    historical_data: List[SessionDataPoint],
    position_data: Dict[str, Any],
    portfolio_value: float,
    cash_balance: float,
    max_markets: int = 5
) -> np.ndarray:
    """
    Build complete observation vector from session data.
    
    This is the shared function used by both training and inference
    to ensure consistency. All features are market-agnostic.
    
    PRICE CONVERSION: Automatically converts input prices from cents to probability:
    - session_data contains raw cents (1-99) from database
    - extract_market_agnostic_features() converts to probability (0.01-0.99)
    - Model receives normalized features only
    
    Args:
        session_data: Current session data point (prices in cents)
        historical_data: Historical context for temporal features (prices in cents)
        position_data: Current portfolio positions
        portfolio_value: Total portfolio value in dollars
        cash_balance: Available cash in dollars
        max_markets: Maximum markets to include in observation
        
    Returns:
        Observation vector as numpy array with all features in [0,1] or [-1,1] range
    """
    # Implementation placeholder - will be completed in M4
    observation_parts = []
    
    # Extract features for each active market (up to max_markets)
    market_count = 0
    for market_ticker, market_data in session_data.markets_data.items():
        if market_count >= max_markets:
            break
            
        # Get market-agnostic features (no ticker exposed to model)
        market_features = extract_market_agnostic_features(market_data)
        observation_parts.extend(list(market_features.values()))
        market_count += 1
    
    # Pad with zeros if fewer markets than max_markets
    features_per_market = 15  # Number of features in extract_market_agnostic_features
    while market_count < max_markets:
        observation_parts.extend([0.0] * features_per_market)
        market_count += 1
    
    # Add temporal features
    temporal_features = extract_temporal_features(session_data, historical_data)
    observation_parts.extend(list(temporal_features.values()))
    
    # Add portfolio features
    portfolio_features = extract_portfolio_features(position_data, portfolio_value, cash_balance)
    observation_parts.extend(list(portfolio_features.values()))
    
    # Convert to numpy array
    observation = np.array(observation_parts, dtype=np.float32)
    
    logger.debug(f"Built observation vector with {len(observation)} features")
    return observation


def calculate_observation_space_size(max_markets: int = 5) -> int:
    """
    Calculate the size of the observation space.
    
    This determines the observation_space dimension for the Gym environment.
    
    Args:
        max_markets: Maximum markets per episode
        
    Returns:
        Total observation space size
    """
    market_features = len(extract_market_agnostic_features({}))  # 15 features
    temporal_features = len(extract_temporal_features(None, []))  # 11 features
    portfolio_features = len(extract_portfolio_features({}, 0.0, 0.0))  # 9 features
    
    total_size = (max_markets * market_features) + temporal_features + portfolio_features
    
    logger.info(f"Observation space size: {total_size} ({max_markets} markets × {market_features} + {temporal_features} temporal + {portfolio_features} portfolio)")
    return total_size


def validate_feature_consistency(
    features1: Dict[str, float],
    features2: Dict[str, float],
    tolerance: float = 1e-6
) -> Tuple[bool, List[str]]:
    """
    Validate that feature extraction is consistent between calls.
    
    This helps ensure that training and inference use identical features.
    
    Args:
        features1: First feature extraction result
        features2: Second feature extraction result  
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Tuple of (is_consistent, list_of_differences)
    """
    # Implementation placeholder - will be completed in M4
    differences = []
    
    # Check all keys are present
    if set(features1.keys()) != set(features2.keys()):
        differences.append("Feature keys differ")
    
    # Check value consistency (for testing with same inputs)
    for key in features1.keys():
        if key in features2:
            if abs(features1[key] - features2[key]) > tolerance:
                differences.append(f"Feature '{key}' differs: {features1[key]} vs {features2[key]}")
    
    return len(differences) == 0, differences