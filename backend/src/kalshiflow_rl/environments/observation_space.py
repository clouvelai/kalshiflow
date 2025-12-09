"""
Unified observation space definition for Kalshi RL Trading Subsystem.

This module provides the critical build_observation_from_orderbook() function
that MUST be used by both training environment and inference actor to ensure
identical feature distribution between training and production.

CRITICAL ARCHITECTURAL REQUIREMENT:
- This function is the single source of truth for observations
- Both KalshiTradingEnv and TradingActor MUST use this same function
- Any changes to features must be made here to maintain training/inference consistency
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from decimal import Decimal
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger("kalshiflow_rl.observation_space")

# Observation space constants
MAX_MARKETS = 10  # Maximum number of markets supported
PRICE_LEVELS_PER_SIDE = 5  # Number of top price levels to include per side
HISTORICAL_WINDOW = 10  # Number of historical observations to include
FEATURE_NORMALIZATION_SCALE = 100.0  # Scale for normalizing price features


class ObservationConfig:
    """Configuration for observation space."""
    
    def __init__(self):
        self.max_markets = MAX_MARKETS
        self.price_levels_per_side = PRICE_LEVELS_PER_SIDE
        self.historical_window = HISTORICAL_WINDOW
        self.include_position_state = True
        self.include_market_metadata = True
        self.normalize_features = True
        self.feature_scale = FEATURE_NORMALIZATION_SCALE
        
        # Feature categories - helps with interpretability
        self.spread_features_count = 2  # yes_spread, no_spread per market
        self.mid_price_features_count = 2  # yes_mid, no_mid per market
        self.volume_features_count = 4  # yes_bid_vol, yes_ask_vol, no_bid_vol, no_ask_vol per market
        self.price_level_features_count = self.price_levels_per_side * 4 * 2  # 5 levels × 4 sides × (price, size) per market
        self.market_activity_features_count = 3  # sequence_delta, time_delta, total_volume per market
        self.position_features_count = 3  # position_yes, position_no, unrealized_pnl per market
        
        self.features_per_market = (
            self.spread_features_count +
            self.mid_price_features_count + 
            self.volume_features_count +
            self.price_level_features_count +
            self.market_activity_features_count +
            self.position_features_count
        )
        
        # Global features (not per-market)
        self.global_features_count = 5  # total_portfolio_value, total_position_count, timestamp_normalized, market_count, active_market_ratio


def create_observation_space(config: ObservationConfig) -> gym.Space:
    """
    Create Gymnasium observation space for multi-market RL environment.
    
    Args:
        config: Observation configuration
        
    Returns:
        Gymnasium Box space for observations
    """
    # Calculate total observation size
    market_features_size = config.max_markets * config.features_per_market
    total_size = market_features_size + config.global_features_count
    
    # Use normalized feature space [-1, 1] for better neural network training
    observation_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(total_size,),
        dtype=np.float32,
        seed=42
    )
    
    logger.info(f"Created observation space: {total_size} features "
                f"({config.max_markets} markets × {config.features_per_market} + {config.global_features_count} global)")
    
    return observation_space


def build_observation_from_orderbook(
    orderbook_states: Union[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    position_states: Optional[Dict[str, Dict[str, float]]] = None,
    config: Optional[ObservationConfig] = None,
    previous_observations: Optional[List[np.ndarray]] = None
) -> np.ndarray:
    """
    Build observation from orderbook state(s) and position state(s).
    
    This is the CRITICAL UNIFIED FUNCTION that must be used by:
    1. KalshiTradingEnv during training (historical data)
    2. TradingActor during inference (live data)
    
    Any changes to feature extraction MUST be made here to maintain
    training/inference consistency.
    
    Args:
        orderbook_states: Either:
            - Single orderbook state dict for single market
            - List of orderbook state dicts for multiple markets
            - Dict mapping market_ticker -> orderbook state dict
        position_states: Dict mapping market_ticker -> position info
            Expected format: {market_ticker: {"position_yes": float, "position_no": float, "unrealized_pnl": float}}
        config: Observation configuration (uses default if None)
        previous_observations: Historical observations for temporal features (optional)
        
    Returns:
        Normalized feature vector as numpy array
    """
    if config is None:
        config = ObservationConfig()
    
    if position_states is None:
        position_states = {}
    
    # Normalize input to dict format: market_ticker -> orderbook_state
    market_states = _normalize_orderbook_input(orderbook_states)
    
    # Initialize observation vector
    observation = np.zeros(config.max_markets * config.features_per_market + config.global_features_count, dtype=np.float32)
    
    # Track global stats
    global_stats = {
        'total_portfolio_value': 0.0,
        'total_position_count': 0,
        'active_markets': 0,
        'current_timestamp': 0
    }
    
    # Extract features for each market
    market_idx = 0
    for market_ticker, orderbook_state in market_states.items():
        if market_idx >= config.max_markets:
            logger.warning(f"Too many markets ({len(market_states)}), truncating to {config.max_markets}")
            break
        
        market_features = _extract_market_features(
            market_ticker=market_ticker,
            orderbook_state=orderbook_state,
            position_state=position_states.get(market_ticker, {}),
            config=config
        )
        
        # Insert market features into observation vector
        start_idx = market_idx * config.features_per_market
        end_idx = start_idx + config.features_per_market
        observation[start_idx:end_idx] = market_features
        
        # Update global stats
        position_info = position_states.get(market_ticker, {})
        position_yes = position_info.get('position_yes', 0.0)
        position_no = position_info.get('position_no', 0.0)
        unrealized_pnl = position_info.get('unrealized_pnl', 0.0)
        
        global_stats['total_portfolio_value'] += unrealized_pnl
        if abs(position_yes) > 0.01 or abs(position_no) > 0.01:
            global_stats['total_position_count'] += 1
        if orderbook_state.get('total_volume', 0) > 0:
            global_stats['active_markets'] += 1
        global_stats['current_timestamp'] = max(global_stats['current_timestamp'], 
                                                orderbook_state.get('last_update_time', 0))
        
        market_idx += 1
    
    # Extract global features
    global_features = _extract_global_features(global_stats, len(market_states), config)
    
    # Insert global features at the end
    global_start_idx = config.max_markets * config.features_per_market
    observation[global_start_idx:] = global_features
    
    # Apply normalization if enabled
    if config.normalize_features:
        observation = _normalize_observation(observation, config)
    
    logger.debug(f"Built observation for {len(market_states)} markets: "
                f"shape={observation.shape}, range=[{observation.min():.3f}, {observation.max():.3f}]")
    
    return observation


def _normalize_orderbook_input(orderbook_states: Union[Dict, List]) -> Dict[str, Dict[str, Any]]:
    """Normalize different input formats to dict[market_ticker, orderbook_state]."""
    if isinstance(orderbook_states, dict):
        # Check if it's a single orderbook state or a mapping
        if 'market_ticker' in orderbook_states:
            # Single orderbook state
            ticker = orderbook_states['market_ticker']
            return {ticker: orderbook_states}
        else:
            # Mapping of market_ticker -> orderbook_state
            return orderbook_states
    elif isinstance(orderbook_states, list):
        # List of orderbook states
        result = {}
        for state in orderbook_states:
            if 'market_ticker' in state:
                result[state['market_ticker']] = state
            else:
                logger.warning("Orderbook state missing market_ticker field")
        return result
    else:
        raise ValueError(f"Invalid orderbook_states format: {type(orderbook_states)}")


def _extract_market_features(
    market_ticker: str,
    orderbook_state: Dict[str, Any],
    position_state: Dict[str, float],
    config: ObservationConfig
) -> np.ndarray:
    """
    Extract features for a single market.
    
    Returns normalized feature vector for this market.
    """
    features = []
    
    # 1. Spread features (2 features)
    yes_spread = orderbook_state.get('yes_spread', 0) or 0
    no_spread = orderbook_state.get('no_spread', 0) or 0
    features.extend([yes_spread, no_spread])
    
    # 2. Mid price features (2 features)
    yes_mid = orderbook_state.get('yes_mid_price', 50.0) or 50.0
    no_mid = orderbook_state.get('no_mid_price', 50.0) or 50.0
    features.extend([yes_mid, no_mid])
    
    # 3. Volume features (4 features)
    # Calculate volume at best prices
    yes_bids = orderbook_state.get('yes_bids', {})
    yes_asks = orderbook_state.get('yes_asks', {})
    no_bids = orderbook_state.get('no_bids', {})
    no_asks = orderbook_state.get('no_asks', {})
    
    yes_bid_vol = _get_top_level_volume(yes_bids, reverse=True)
    yes_ask_vol = _get_top_level_volume(yes_asks, reverse=False)
    no_bid_vol = _get_top_level_volume(no_bids, reverse=True)
    no_ask_vol = _get_top_level_volume(no_asks, reverse=False)
    
    features.extend([yes_bid_vol, yes_ask_vol, no_bid_vol, no_ask_vol])
    
    # 4. Price level features (5 levels × 4 sides × 2 (price, size) = 40 features)
    price_level_features = []
    
    for book, is_reversed in [(yes_bids, True), (yes_asks, False), (no_bids, True), (no_asks, False)]:
        levels = _get_top_price_levels(book, config.price_levels_per_side, is_reversed)
        for price, size in levels:
            price_level_features.extend([price, size])
    
    features.extend(price_level_features)
    
    # 5. Market activity features (3 features)
    last_sequence = orderbook_state.get('last_sequence', 0)
    last_update_time = orderbook_state.get('last_update_time', 0)
    total_volume = orderbook_state.get('total_volume', 0)
    
    # Normalize sequence and timestamp (use relative values for better training)
    sequence_normalized = min(last_sequence / 1000.0, 100.0)  # Cap at 100
    time_normalized = (last_update_time % (24 * 3600 * 1000)) / (24 * 3600 * 1000)  # Daily cycle [0,1]
    volume_normalized = min(total_volume / 10000.0, 50.0)  # Cap at 50
    
    features.extend([sequence_normalized, time_normalized, volume_normalized])
    
    # 6. Position features (3 features)
    position_yes = position_state.get('position_yes', 0.0)
    position_no = position_state.get('position_no', 0.0)
    unrealized_pnl = position_state.get('unrealized_pnl', 0.0)
    
    # Normalize positions (assume max position ±1000)
    position_yes_norm = np.clip(position_yes / 1000.0, -1.0, 1.0)
    position_no_norm = np.clip(position_no / 1000.0, -1.0, 1.0)
    pnl_norm = np.clip(unrealized_pnl / 1000.0, -5.0, 5.0)  # Cap at ±$5000
    
    features.extend([position_yes_norm, position_no_norm, pnl_norm])
    
    # Verify feature count matches expectation
    expected_count = config.features_per_market
    if len(features) != expected_count:
        logger.error(f"Market {market_ticker} feature count mismatch: "
                    f"expected {expected_count}, got {len(features)}")
        # Pad or truncate to match expected size
        if len(features) < expected_count:
            features.extend([0.0] * (expected_count - len(features)))
        else:
            features = features[:expected_count]
    
    return np.array(features, dtype=np.float32)


def _extract_global_features(
    global_stats: Dict[str, float],
    active_market_count: int,
    config: ObservationConfig
) -> np.ndarray:
    """Extract global portfolio and timing features."""
    features = []
    
    # Portfolio value (normalized)
    portfolio_value_norm = np.clip(global_stats['total_portfolio_value'] / 10000.0, -10.0, 10.0)
    features.append(portfolio_value_norm)
    
    # Position count (normalized)
    position_count_norm = min(global_stats['total_position_count'] / config.max_markets, 1.0)
    features.append(position_count_norm)
    
    # Timestamp (daily cycle normalized)
    timestamp_norm = (global_stats['current_timestamp'] % (24 * 3600 * 1000)) / (24 * 3600 * 1000)
    features.append(timestamp_norm)
    
    # Market count (normalized)
    market_count_norm = min(active_market_count / config.max_markets, 1.0)
    features.append(market_count_norm)
    
    # Active market ratio
    active_ratio = global_stats['active_markets'] / max(active_market_count, 1)
    features.append(active_ratio)
    
    return np.array(features, dtype=np.float32)


def _get_top_level_volume(book: Dict[int, int], reverse: bool) -> float:
    """Get volume at the best price level."""
    if not book:
        return 0.0
    
    if reverse:
        # For bids, highest price is best
        best_price = max(book.keys())
    else:
        # For asks, lowest price is best
        best_price = min(book.keys())
    
    return float(book.get(best_price, 0))


def _get_top_price_levels(
    book: Dict[int, int], 
    num_levels: int, 
    reverse: bool
) -> List[Tuple[float, float]]:
    """
    Get top N price levels from order book.
    
    Args:
        book: Price levels {price: size}
        num_levels: Number of levels to return
        reverse: True for bids (descending), False for asks (ascending)
        
    Returns:
        List of (price, size) tuples
    """
    if not book:
        return [(0.0, 0.0)] * num_levels
    
    # Sort prices
    sorted_prices = sorted(book.keys(), reverse=reverse)
    
    levels = []
    for i in range(num_levels):
        if i < len(sorted_prices):
            price = sorted_prices[i]
            size = book[price]
            # Normalize price to [0, 100] range and size to reasonable scale
            price_norm = float(price) / 100.0
            size_norm = min(float(size) / 1000.0, 10.0)  # Cap at 10
            levels.append((price_norm, size_norm))
        else:
            levels.append((0.0, 0.0))
    
    return levels


def _normalize_observation(observation: np.ndarray, config: ObservationConfig) -> np.ndarray:
    """
    Apply final normalization to observation vector.
    
    Ensures all features are in reasonable range for neural network training.
    """
    # Clip extreme values
    observation = np.clip(observation, -10.0, 10.0)
    
    # Apply tanh normalization to keep values in [-1, 1]
    observation = np.tanh(observation / config.feature_scale * 10.0)
    
    return observation


def get_observation_feature_names(config: Optional[ObservationConfig] = None) -> List[str]:
    """
    Get human-readable names for observation features.
    
    Useful for debugging and model interpretability.
    
    Args:
        config: Observation configuration
        
    Returns:
        List of feature names matching observation vector order
    """
    if config is None:
        config = ObservationConfig()
    
    feature_names = []
    
    # Market features
    for market_idx in range(config.max_markets):
        prefix = f"market_{market_idx}"
        
        # Spread features
        feature_names.extend([f"{prefix}_yes_spread", f"{prefix}_no_spread"])
        
        # Mid price features
        feature_names.extend([f"{prefix}_yes_mid", f"{prefix}_no_mid"])
        
        # Volume features
        feature_names.extend([
            f"{prefix}_yes_bid_vol", f"{prefix}_yes_ask_vol",
            f"{prefix}_no_bid_vol", f"{prefix}_no_ask_vol"
        ])
        
        # Price level features
        for level_idx in range(config.price_levels_per_side):
            for side in ['yes_bid', 'yes_ask', 'no_bid', 'no_ask']:
                feature_names.extend([
                    f"{prefix}_{side}_L{level_idx}_price",
                    f"{prefix}_{side}_L{level_idx}_size"
                ])
        
        # Market activity features
        feature_names.extend([
            f"{prefix}_sequence_norm", f"{prefix}_time_norm", f"{prefix}_volume_norm"
        ])
        
        # Position features
        feature_names.extend([
            f"{prefix}_position_yes", f"{prefix}_position_no", f"{prefix}_unrealized_pnl"
        ])
    
    # Global features
    feature_names.extend([
        "global_portfolio_value", "global_position_count", "global_timestamp_norm",
        "global_market_count", "global_active_ratio"
    ])
    
    return feature_names


# Create default observation space and config for easy import
default_config = ObservationConfig()
default_observation_space = create_observation_space(default_config)