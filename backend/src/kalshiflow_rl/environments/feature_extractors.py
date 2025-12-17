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
    if not orderbook_data:
        # Return zero features for empty data
        return _get_default_market_features()
    
    features = {}
    
    # === PRICE FEATURES (CENTS → PROBABILITY CONVERSION) ===
    
    # Extract YES/NO sides data
    yes_bids = orderbook_data.get('yes_bids', {})
    yes_asks = orderbook_data.get('yes_asks', {})
    no_bids = orderbook_data.get('no_bids', {})
    no_asks = orderbook_data.get('no_asks', {})
    
    # Get best prices (highest bid, lowest ask)
    best_yes_bid = max(map(int, yes_bids.keys())) if yes_bids else None
    best_yes_ask = min(map(int, yes_asks.keys())) if yes_asks else None
    best_no_bid = max(map(int, no_bids.keys())) if no_bids else None
    best_no_ask = min(map(int, no_asks.keys())) if no_asks else None
    
    # Convert cents to probability space [0.01, 0.99]
    features['best_yes_bid_norm'] = (best_yes_bid / 100.0) if best_yes_bid else 0.5
    features['best_yes_ask_norm'] = (best_yes_ask / 100.0) if best_yes_ask else 0.5  
    features['best_no_bid_norm'] = (best_no_bid / 100.0) if best_no_bid else 0.5
    features['best_no_ask_norm'] = (best_no_ask / 100.0) if best_no_ask else 0.5
    
    # Calculate spreads in cents (absolute cost)
    yes_spread_cents = (best_yes_ask - best_yes_bid) if (best_yes_ask and best_yes_bid) else 10
    no_spread_cents = (best_no_ask - best_no_bid) if (best_no_ask and best_no_bid) else 10
    
    # DEPRECATED: Old normalized spreads (keeping for backward compatibility temporarily)
    yes_spread = yes_spread_cents / 100.0
    no_spread = no_spread_cents / 100.0
    features['yes_spread_norm'] = min(max(yes_spread, 0.001), 0.99)  # Clamp to valid range
    features['no_spread_norm'] = min(max(no_spread, 0.001), 0.99)
    
    # Mid-prices in probability space
    yes_mid = ((best_yes_bid + best_yes_ask) / 2.0 / 100.0) if (best_yes_bid and best_yes_ask) else 0.5
    no_mid = ((best_no_bid + best_no_ask) / 2.0 / 100.0) if (best_no_bid and best_no_ask) else 0.5
    
    features['yes_mid_price_norm'] = min(max(yes_mid, 0.01), 0.99)
    features['no_mid_price_norm'] = min(max(no_mid, 0.01), 0.99)
    
    # === SPREAD-AWARE FEATURES (COMPATIBLE WITH TRAINING) ===
    
    # Add back specific features to match the 52-feature model used in training
    # 1. Direct spread costs in cents (normalized)
    features['yes_spread_cents'] = yes_spread_cents / 100.0  
    features['no_spread_cents'] = no_spread_cents / 100.0
    
    # 2. Relative spread as percentage of mid-price
    features['yes_spread_pct'] = (yes_spread_cents / (yes_mid * 100.0)) if yes_mid > 0 else 0.5
    features['no_spread_pct'] = (no_spread_cents / (no_mid * 100.0)) if no_mid > 0 else 0.5
    
    # 3. Spread regime classification
    def classify_spread_regime(spread_cents):
        if spread_cents < 2:
            return 0.0  # Ultra-tight
        elif spread_cents < 5:
            return 0.25  # Tight
        elif spread_cents < 10:
            return 0.5  # Medium
        elif spread_cents < 20:
            return 0.75  # Wide
        else:
            return 1.0  # Very wide
    
    features['yes_spread_regime'] = classify_spread_regime(yes_spread_cents)
    features['no_spread_regime'] = classify_spread_regime(no_spread_cents)
    
    # 4. Breakeven distance
    features['yes_breakeven_move'] = yes_spread_cents / 100.0
    features['no_breakeven_move'] = no_spread_cents / 100.0
    
    # Calculate volume for liquidity features
    yes_bid_volume = sum(yes_bids.values()) if yes_bids else 0
    yes_ask_volume = sum(yes_asks.values()) if yes_asks else 0
    no_bid_volume = sum(no_bids.values()) if no_bids else 0
    no_ask_volume = sum(no_asks.values()) if no_asks else 0
    
    yes_total_volume = yes_bid_volume + yes_ask_volume
    no_total_volume = no_bid_volume + no_ask_volume
    
    # === VOLUME AND LIQUIDITY FEATURES ===
    
    # Note: yes/no_bid_volume already calculated above for liquidity scores
    total_yes_volume = yes_bid_volume + yes_ask_volume
    total_no_volume = no_bid_volume + no_ask_volume
    total_volume = total_yes_volume + total_no_volume
    
    # Normalize volumes (using log scale for wide ranges)
    max_volume_ref = 10000.0  # Reference maximum for normalization
    # REMOVED: yes_volume_norm and no_volume_norm (redundant with total_volume + imbalance)
    features['total_volume_norm'] = min(np.log(1 + total_volume) / np.log(1 + max_volume_ref), 1.0)
    
    # Volume imbalance: (yes_vol - no_vol) / total_vol ∈ [-1, 1]
    features['volume_imbalance'] = ((total_yes_volume - total_no_volume) / total_volume) if total_volume > 0 else 0.0
    
    # Side-specific imbalances: (bid_vol - ask_vol) / total_side_vol ∈ [-1, 1]
    features['yes_side_imbalance'] = ((yes_bid_volume - yes_ask_volume) / total_yes_volume) if total_yes_volume > 0 else 0.0
    features['no_side_imbalance'] = ((no_bid_volume - no_ask_volume) / total_no_volume) if total_no_volume > 0 else 0.0
    
    # === ORDER BOOK SHAPE FEATURES ===
    
    # Book depth (number of price levels with orders)
    yes_depth = len([p for p in list(yes_bids.keys()) + list(yes_asks.keys()) if yes_bids.get(p, 0) + yes_asks.get(p, 0) > 0])
    no_depth = len([p for p in list(no_bids.keys()) + list(no_asks.keys()) if no_bids.get(p, 0) + no_asks.get(p, 0) > 0])
    
    max_depth_ref = 20.0  # Reference maximum depth
    features['yes_book_depth_norm'] = min(yes_depth / max_depth_ref, 1.0)
    features['no_book_depth_norm'] = min(no_depth / max_depth_ref, 1.0)
    
    # Top-of-book liquidity concentration (top 3 levels vs total)
    def get_top_concentration(bids_dict, asks_dict, top_n=3):
        all_levels = list(bids_dict.items()) + list(asks_dict.items())
        if not all_levels:
            return 0.5
        
        sorted_by_volume = sorted(all_levels, key=lambda x: x[1], reverse=True)
        top_volume = sum(vol for _, vol in sorted_by_volume[:top_n])
        total_volume = sum(vol for _, vol in all_levels)
        
        return (top_volume / total_volume) if total_volume > 0 else 0.5
    
    features['yes_liquidity_concentration'] = get_top_concentration(yes_bids, yes_asks)
    features['no_liquidity_concentration'] = get_top_concentration(no_bids, no_asks)
    
    # === ARBITRAGE AND EFFICIENCY FEATURES ===
    
    # Arbitrage opportunity: |YES_mid + NO_mid - 1.0| (should be close to 0)
    total_mid_price = features['yes_mid_price_norm'] + features['no_mid_price_norm']
    features['arbitrage_opportunity'] = abs(total_mid_price - 1.0)
    
    # Market efficiency: How close the sum is to 1.0 (higher = more efficient)
    features['market_efficiency'] = max(0.0, 1.0 - features['arbitrage_opportunity'])
    
    # REMOVED: cross_side_efficiency (redundant with arbitrage_opportunity)
    
    logger.debug(f"Extracted {len(features)} market-agnostic features for orderbook")
    return features


def _get_default_market_features() -> Dict[str, float]:
    """Return default feature values for empty/invalid orderbook data."""
    return {
        # Price features (default to mid-market)
        'best_yes_bid_norm': 0.49,
        'best_yes_ask_norm': 0.51, 
        'best_no_bid_norm': 0.49,
        'best_no_ask_norm': 0.51,
        'yes_spread_norm': 0.02,  # Deprecated but kept for backward compatibility
        'no_spread_norm': 0.02,   # Deprecated but kept for backward compatibility
        'yes_mid_price_norm': 0.5,
        'no_mid_price_norm': 0.5,
        
        # Spread-aware features (defaults for medium spreads)
        'yes_spread_cents': 0.02,
        'no_spread_cents': 0.02,
        'yes_spread_pct': 0.04,  # 4% of mid-price
        'no_spread_pct': 0.04,
        'yes_spread_regime': 0.25,  # Tight regime
        'no_spread_regime': 0.25,
        'yes_breakeven_move': 0.02,
        'no_breakeven_move': 0.02,
        
        # Volume features (minimal activity) - REMOVED yes/no_volume_norm
        'total_volume_norm': 0.01,
        'volume_imbalance': 0.0,
        'yes_side_imbalance': 0.0,
        'no_side_imbalance': 0.0,
        
        # Book shape features (minimal depth)
        'yes_book_depth_norm': 0.1,
        'no_book_depth_norm': 0.1,
        'yes_liquidity_concentration': 0.5,
        'no_liquidity_concentration': 0.5,
        
        # Efficiency features (perfect efficiency as default) - REMOVED cross_side_efficiency
        'arbitrage_opportunity': 0.0,
        'market_efficiency': 1.0
    }


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
    features = {}
    
    # === TIME-BASED FEATURES ===
    
    # Time since last update (normalized)
    time_gap_seconds = current_data.time_gap if current_data.time_gap > 0 else 0.0
    max_gap_ref = 300.0  # 5 minutes as reference maximum
    features['time_since_last_update'] = min(time_gap_seconds / max_gap_ref, 1.0)
    
    # Time of day normalization (business hours focus)
    hour_of_day = current_data.timestamp.hour
    # Market hours typically 9 AM - 4 PM ET, normalize to [0,1]
    if 9 <= hour_of_day <= 16:
        features['time_of_day_norm'] = (hour_of_day - 9) / 7.0  # Business hours [0,1]
    else:
        features['time_of_day_norm'] = 0.0  # Outside business hours
    
    # REMOVED: day_of_week_norm (low signal for prediction markets)
    
    # === ACTIVITY MOMENTUM FEATURES ===
    
    # Use pre-computed activity score from SessionDataPoint
    current_activity = current_data.activity_score
    features['current_activity_score'] = current_activity
    
    if len(historical_data) >= 1:
        prev_activity = historical_data[-1].activity_score
        features['activity_change'] = np.tanh(current_activity - prev_activity)  # [-1,1]
        
        # Recent update frequency (last 5 data points)
        recent_points = historical_data[-5:] if len(historical_data) >= 5 else historical_data
        if len(recent_points) >= 2:
            time_span = (current_data.timestamp_ms - recent_points[0].timestamp_ms) / 1000.0
            update_freq = len(recent_points) / max(time_span, 1.0)  # Updates per second
            max_freq_ref = 2.0  # 2 updates per second as reference max
            features['update_frequency'] = min(update_freq / max_freq_ref, 1.0)
        else:
            features['update_frequency'] = 0.1
    else:
        features['activity_change'] = 0.0
        features['update_frequency'] = 0.1
    
    # === ACTIVITY PATTERN DETECTION ===
    
    # Activity burst detection (last 10 points)
    recent_window = historical_data[-10:] + [current_data] if historical_data else [current_data]
    activity_scores = [point.activity_score for point in recent_window]
    
    if len(activity_scores) >= 3:
        mean_activity = np.mean(activity_scores)
        std_activity = np.std(activity_scores)
        
        # Burst: current activity > mean + 1.5*std
        burst_threshold = mean_activity + 1.5 * std_activity
        features['activity_burst_indicator'] = 1.0 if current_activity > burst_threshold else 0.0
        
        # REMOVED: quiet_period_indicator (inverse of activity_burst_indicator)
        
        # Activity trend (increasing/decreasing)
        if len(activity_scores) >= 5:
            trend_slope = np.polyfit(range(len(activity_scores)), activity_scores, 1)[0]
            features['activity_trend'] = np.tanh(trend_slope * 10.0)  # Scale and bound to [-1,1]
        else:
            features['activity_trend'] = 0.0
    else:
        features['activity_burst_indicator'] = 0.0
        features['activity_trend'] = 0.0
    
    # === PRICE MOMENTUM FEATURES ===
    
    # Use pre-computed momentum from SessionDataPoint
    features['price_momentum'] = current_data.momentum
    
    # Volatility regime based on recent price movements
    if len(historical_data) >= 5:
        # Calculate volatility from mid-price changes across all markets
        recent_points = historical_data[-5:] + [current_data]
        price_changes = []
        
        for i in range(1, len(recent_points)):
            prev_point = recent_points[i-1]
            curr_point = recent_points[i]
            
            # Track all mid-price changes across markets
            for market_ticker in curr_point.mid_prices:
                if market_ticker in prev_point.mid_prices:
                    prev_yes_mid, _ = prev_point.mid_prices[market_ticker]
                    curr_yes_mid, _ = curr_point.mid_prices[market_ticker]
                    
                    if prev_yes_mid is not None and curr_yes_mid is not None:
                        price_change = abs(float(curr_yes_mid) - float(prev_yes_mid))
                        price_changes.append(price_change)
        
        if price_changes:
            volatility = np.std(price_changes)
            max_vol_ref = 5.0  # 5 cents as reference maximum volatility
            features['volatility_regime'] = min(volatility / max_vol_ref, 1.0)
        else:
            features['volatility_regime'] = 0.1
    else:
        features['volatility_regime'] = 0.1
    
    # === SINGLE-MARKET ACTIVITY FEATURES ===
    
    # For single-market training, we focus on the target market's activity patterns
    # rather than cross-market correlations which don't apply
    
    # REMOVED: activity_consistency (low signal value, overlaps with volatility_regime)
    
    # Price stability (how much prices are changing recently)
    if len(historical_data) >= 3:
        price_changes = []
        for i in range(max(0, len(historical_data) - 3), len(historical_data)):
            if i > 0:
                curr_point = historical_data[i]
                prev_point = historical_data[i-1]
                
                # Look at any market's price changes (should be single market in MarketSessionView)
                for market in curr_point.mid_prices:
                    if market in prev_point.mid_prices:
                        curr_mid = curr_point.mid_prices[market][0]  # YES mid price
                        prev_mid = prev_point.mid_prices[market][0]
                        
                        if curr_mid is not None and prev_mid is not None:
                            change = abs(float(curr_mid) - float(prev_mid))
                            price_changes.append(change)
        
        if price_changes:
            avg_price_change = np.mean(price_changes)
            max_change_ref = 5.0  # 5 cents as reference maximum
            features['price_stability'] = max(0.0, 1.0 - (avg_price_change / max_change_ref))
        else:
            features['price_stability'] = 0.5
    else:
        features['price_stability'] = 0.5
    
    # REMOVED: activity_persistence (overlaps with activity_trend)
    
    logger.debug(f"Extracted {len(features)} temporal features for timestamp {current_data.timestamp_ms}")
    return features


def extract_portfolio_features(
    position_data: Dict[str, Any],
    portfolio_value: float,
    cash_balance: float,
    current_prices: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Extract portfolio state features for observation.
    
    Provides position and portfolio context without exposing market identities.
    Uses Kalshi position convention: +contracts = YES position, -contracts = NO position
    
    Args:
        position_data: Current positions across markets (Kalshi format: {ticker: {position: int, cost_basis: float, realized_pnl: float}})
        portfolio_value: Total portfolio value in dollars
        cash_balance: Available cash in dollars
        current_prices: Optional dict mapping ticker to mid price in probability space [0,1]
        
    Returns:
        Dictionary of portfolio features in [0,1] or [-1,1] range
    """
    features = {}
    
    # Handle empty portfolio
    if portfolio_value <= 0.01:
        return _get_default_portfolio_features()
    
    # === PORTFOLIO COMPOSITION ===
    
    # Cash vs position allocation
    features['cash_ratio'] = min(max(cash_balance / portfolio_value, 0.0), 1.0)
    
    position_value = portfolio_value - cash_balance
    features['position_ratio'] = min(max(position_value / portfolio_value, 0.0), 1.0)
    
    # === POSITION CHARACTERISTICS ===
    
    if not position_data:
        # No positions
        features.update({
            'position_count_norm': 0.0,
            'average_position_size_norm': 0.0,
            'long_position_ratio': 0.0,
            'short_position_ratio': 0.0,
            'net_position_bias': 0.0,
            'position_concentration': 0.0,
            'largest_position_norm': 0.0,
            'unrealized_pnl_ratio': 0.0,
        })
    else:
        # Analyze positions
        positions = []
        long_positions = 0
        short_positions = 0
        total_abs_positions = 0
        position_values = []
        unrealized_pnl = 0.0
        
        for ticker, pos_info in position_data.items():
            # Handle both dict and PositionInfo objects
            if hasattr(pos_info, 'position'):  # PositionInfo object
                position = pos_info.position
                cost_basis = pos_info.cost_basis
                realized_pnl = pos_info.realized_pnl
            else:  # Dict format (backward compatibility)
                position = pos_info.get('position', 0)
                cost_basis = pos_info.get('cost_basis', 0.0)
                realized_pnl = pos_info.get('realized_pnl', 0.0)
            
            if position != 0:
                positions.append(position)
                total_abs_positions += abs(position)
                
                if position > 0:
                    long_positions += 1
                else:
                    short_positions += 1
                
                # Calculate position value using current market prices
                # Use actual market price from current_prices, fallback to 0.5 (50 cents) if not available
                if current_prices and ticker in current_prices:
                    mid_price_probability = current_prices[ticker]  # Already in [0,1] probability space
                    position_value = abs(position) * (mid_price_probability * 100.0)  # Convert back to cents for calculation
                else:
                    # Fallback to 50 cents if price not available
                    position_value = abs(position) * 50.0
                position_values.append(position_value)
                
                # Unrealized P&L estimation (cost basis vs current value)
                unrealized_pnl += position_value - cost_basis
        
        total_positions = len(positions)
        
        # Position count (normalized)
        max_positions_ref = 10.0
        features['position_count_norm'] = min(total_positions / max_positions_ref, 1.0)
        
        # Average position size
        if total_positions > 0:
            avg_position_size = total_abs_positions / total_positions
            max_size_ref = 1000.0  # Reference maximum position size
            features['average_position_size_norm'] = min(avg_position_size / max_size_ref, 1.0)
        else:
            features['average_position_size_norm'] = 0.0
        
        # Long vs short position ratios
        if total_positions > 0:
            features['long_position_ratio'] = long_positions / total_positions
            features['short_position_ratio'] = short_positions / total_positions
        else:
            features['long_position_ratio'] = 0.0
            features['short_position_ratio'] = 0.0
        
        # Net position bias: (long_contracts - short_contracts) / total_abs_contracts
        net_position = sum(positions)
        features['net_position_bias'] = (net_position / total_abs_positions) if total_abs_positions > 0 else 0.0
        features['net_position_bias'] = np.tanh(features['net_position_bias'])  # Bound to [-1,1]
        
        # Position concentration (largest position / total portfolio)
        if position_values:
            largest_position_value = max(position_values)
            features['position_concentration'] = min(largest_position_value / portfolio_value, 1.0)
            
            # Largest position size (normalized)
            features['largest_position_norm'] = min(max(positions, key=abs) / 1000.0, 1.0) if positions else 0.0
        else:
            features['position_concentration'] = 0.0
            features['largest_position_norm'] = 0.0
        
        # Unrealized P&L ratio
        features['unrealized_pnl_ratio'] = np.tanh(unrealized_pnl / portfolio_value) if portfolio_value > 0 else 0.0
    
    # === RISK METRICS ===
    
    # REMOVED: position_diversity (not relevant for single-market training)
    
    # Leverage approximation (total position value / portfolio value)
    if position_data:
        total_position_value = 0.0
        for ticker, pos_info in position_data.items():
            # Handle both dict and PositionInfo objects
            if hasattr(pos_info, 'position'):  # PositionInfo object
                position = abs(pos_info.position)
            else:  # Dict format
                position = abs(pos_info.get('position', 0))
            if current_prices and ticker in current_prices:
                mid_price_probability = current_prices[ticker]
                position_val = position * (mid_price_probability * 100.0)  # Convert back to cents
            else:
                position_val = position * 50.0  # Fallback to 50 cents
            total_position_value += position_val
        
        features['leverage'] = min(total_position_value / portfolio_value, 2.0) if portfolio_value > 0 else 0.0
        features['leverage'] = features['leverage'] / 2.0  # Normalize to [0,1]
    else:
        features['leverage'] = 0.0
    
    logger.debug(f"Extracted {len(features)} portfolio features (portfolio_value: ${portfolio_value:.2f}, positions: {len(position_data) if position_data else 0})")
    return features


def _get_default_portfolio_features() -> Dict[str, float]:
    """Return default portfolio features for empty/minimal portfolios."""
    return {
        # Portfolio composition (all cash, no positions)
        'cash_ratio': 1.0,
        'position_ratio': 0.0,
        
        # Position characteristics (no positions)
        'position_count_norm': 0.0,
        'average_position_size_norm': 0.0,
        'long_position_ratio': 0.0,
        'short_position_ratio': 0.0,
        'net_position_bias': 0.0,
        'position_concentration': 0.0,
        'largest_position_norm': 0.0,
        'unrealized_pnl_ratio': 0.0,
        
        # Risk metrics (no risk)
        'leverage': 0.0
    }


def build_observation_from_session_data(
    session_data: SessionDataPoint,
    historical_data: List[SessionDataPoint],
    position_data: Dict[str, Any],
    portfolio_value: float,
    cash_balance: float,
    max_markets: int = 1,
    order_features: Optional[np.ndarray] = None
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
        order_features: Optional order state features (5 elements)
        
    Returns:
        Observation vector as numpy array with all features in [0,1] or [-1,1] range
    """
    observation_parts = []
    
    # === MARKET FEATURES (UP TO max_markets) ===
    
    # Sort markets by activity for consistent ordering
    # Use total volume as activity metric
    markets_by_activity = []
    for market_ticker, market_data in session_data.markets_data.items():
        total_volume = market_data.get('total_volume', 0)
        markets_by_activity.append((total_volume, market_ticker, market_data))
    
    # Sort by volume descending (most active first)
    markets_by_activity.sort(key=lambda x: x[0], reverse=True)
    
    # Extract features for each active market (up to max_markets)
    market_count = 0
    for volume, market_ticker, market_data in markets_by_activity:
        if market_count >= max_markets:
            break
            
        # Get market-agnostic features (no ticker exposed to model)
        # This automatically converts cents → probability
        market_features = extract_market_agnostic_features(market_data)
        observation_parts.extend(list(market_features.values()))
        market_count += 1
    
    # Pad with default features if fewer markets than max_markets
    default_features = _get_default_market_features()
    features_per_market = len(default_features)
    
    while market_count < max_markets:
        observation_parts.extend(list(default_features.values()))
        market_count += 1
    
    # === TEMPORAL FEATURES ===
    
    temporal_features = extract_temporal_features(session_data, historical_data)
    observation_parts.extend(list(temporal_features.values()))
    
    # === PORTFOLIO FEATURES ===
    
    # Extract current market prices from session data for accurate position valuation
    current_prices = {}
    for market_ticker, market_data in session_data.markets_data.items():
        # Calculate mid-price in probability space for each market
        yes_bids = market_data.get('yes_bids', {})
        yes_asks = market_data.get('yes_asks', {})
        
        # Get best prices
        best_yes_bid = max(map(int, yes_bids.keys())) if yes_bids else None
        best_yes_ask = min(map(int, yes_asks.keys())) if yes_asks else None
        
        # Calculate YES mid-price in probability space
        if best_yes_bid is not None and best_yes_ask is not None:
            yes_mid_cents = (best_yes_bid + best_yes_ask) / 2.0
            yes_mid_probability = yes_mid_cents / 100.0  # Convert cents to probability
            current_prices[market_ticker] = min(max(yes_mid_probability, 0.01), 0.99)  # Clamp to valid range
        else:
            current_prices[market_ticker] = 0.5  # Default to 50% if no prices available
    
    portfolio_features = extract_portfolio_features(position_data, portfolio_value, cash_balance, current_prices)
    observation_parts.extend(list(portfolio_features.values()))
    
    # === ORDER FEATURES ===
    
    # Add order state features if provided (from OrderManager)
    if order_features is not None:
        observation_parts.extend(order_features.tolist())
    else:
        # Default order features (no open orders)
        default_order_features = [0.0, 0.0, 0.0, 0.0, 0.0]  # [has_open_buy, has_open_sell, buy_distance, sell_distance, time_since_order]
        observation_parts.extend(default_order_features)
    
    # === FINAL OBSERVATION VECTOR ===
    
    # Convert to numpy array and validate
    observation = np.array(observation_parts, dtype=np.float32)
    
    # Sanity check: ensure all values are finite and in reasonable ranges
    observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
    observation = np.clip(observation, -2.0, 2.0)  # Clip extreme values
    
    order_feature_count = 5 if order_features is not None else 5  # Always 5 order features
    logger.debug(
        f"Built observation vector with {len(observation)} features: "
        f"{market_count} markets × {features_per_market} + {len(temporal_features)} temporal + "
        f"{len(portfolio_features)} portfolio + {order_feature_count} order"
    )
    
    return observation


def calculate_observation_space_size(max_markets: int = 1) -> int:
    """
    Calculate the size of the observation space.
    
    This determines the observation_space dimension for the Gym environment.
    
    Args:
        max_markets: Maximum markets per episode
        
    Returns:
        Total observation space size
    """
    # Calculate feature counts using actual implementations
    market_features = len(extract_market_agnostic_features({}))
    
    # Create dummy data for temporal features calculation
    from datetime import datetime
    dummy_current = SessionDataPoint(
        timestamp=datetime.now(),
        timestamp_ms=int(datetime.now().timestamp() * 1000),
        markets_data={}
    )
    temporal_features = len(extract_temporal_features(dummy_current, []))
    
    portfolio_features = len(extract_portfolio_features({}, 1000.0, 800.0, {}))
    
    # Order features (5 features: has_open_buy, has_open_sell, buy_distance, sell_distance, time_since_order)
    order_features = 5
    
    # Global features removed - were redundant with temporal features
    global_features = 0
    
    total_size = (max_markets * market_features) + temporal_features + portfolio_features + order_features + global_features
    
    logger.info(
        f"Observation space size: {total_size} "
        f"({max_markets} markets × {market_features} + {temporal_features} temporal + "
        f"{portfolio_features} portfolio + {order_features} order + {global_features} global)"
    )
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
    differences = []
    
    # Check all keys are present
    keys1 = set(features1.keys())
    keys2 = set(features2.keys())
    
    if keys1 != keys2:
        missing_in_1 = keys2 - keys1
        missing_in_2 = keys1 - keys2
        
        if missing_in_1:
            differences.append(f"Features missing in first dict: {sorted(missing_in_1)}")
        if missing_in_2:
            differences.append(f"Features missing in second dict: {sorted(missing_in_2)}")
    
    # Check value consistency for common keys
    common_keys = keys1 & keys2
    
    for key in sorted(common_keys):  # Sort for deterministic error messages
        value1 = features1[key]
        value2 = features2[key]
        
        # Handle NaN values
        if np.isnan(value1) and np.isnan(value2):
            continue  # Both NaN is considered consistent
        elif np.isnan(value1) or np.isnan(value2):
            differences.append(f"Feature '{key}' NaN inconsistency: {value1} vs {value2}")
            continue
        
        # Handle infinite values
        if np.isinf(value1) and np.isinf(value2) and np.sign(value1) == np.sign(value2):
            continue  # Both same infinity is consistent
        elif np.isinf(value1) or np.isinf(value2):
            differences.append(f"Feature '{key}' infinity inconsistency: {value1} vs {value2}")
            continue
        
        # Numerical comparison
        abs_diff = abs(value1 - value2)
        
        # Use relative tolerance for large values
        if max(abs(value1), abs(value2)) > 1.0:
            relative_tolerance = tolerance * max(abs(value1), abs(value2))
            if abs_diff > relative_tolerance:
                differences.append(f"Feature '{key}' differs (relative): {value1} vs {value2} (diff: {abs_diff:.2e}, tol: {relative_tolerance:.2e})")
        else:
            # Use absolute tolerance for small values
            if abs_diff > tolerance:
                differences.append(f"Feature '{key}' differs (absolute): {value1} vs {value2} (diff: {abs_diff:.2e}, tol: {tolerance:.2e})")
    
    # Additional validation: check feature value ranges
    out_of_range = []
    
    for key, value in features1.items():
        if np.isfinite(value):  # Only check finite values
            # Most features should be in [0,1] or [-1,1] range
            if key.endswith('_imbalance') or key.endswith('_bias') or key in ['price_momentum', 'activity_change', 'activity_trend', 'net_position_bias']:
                # These features can be in [-1,1] range
                if not (-1.1 <= value <= 1.1):  # Small tolerance for rounding
                    out_of_range.append(f"Feature '{key}' out of [-1,1] range: {value}")
            else:
                # Most other features should be in [0,1] range
                if not (-0.1 <= value <= 1.1):  # Small tolerance for rounding
                    out_of_range.append(f"Feature '{key}' out of [0,1] range: {value}")
    
    if out_of_range:
        differences.extend(out_of_range)
    
    is_consistent = len(differences) == 0
    
    if not is_consistent:
        logger.warning(f"Feature consistency validation failed: {len(differences)} differences found")
        for diff in differences[:5]:  # Log first 5 differences
            logger.warning(f"  - {diff}")
        if len(differences) > 5:
            logger.warning(f"  - ... and {len(differences) - 5} more differences")
    else:
        logger.debug(f"Feature consistency validation passed for {len(common_keys)} features")
    
    return is_consistent, differences


def validate_observation_vector(observation: np.ndarray) -> Tuple[bool, List[str]]:
    """
    Validate that an observation vector is well-formed.
    
    Checks for NaN values, infinite values, and reasonable ranges.
    
    Args:
        observation: Observation vector to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check shape
    if observation.ndim != 1:
        issues.append(f"Observation should be 1D, got {observation.ndim}D")
    
    # Check for NaN values
    nan_count = np.isnan(observation).sum()
    if nan_count > 0:
        issues.append(f"Observation contains {nan_count} NaN values")
    
    # Check for infinite values
    inf_count = np.isinf(observation).sum()
    if inf_count > 0:
        issues.append(f"Observation contains {inf_count} infinite values")
    
    # Check value ranges (most features should be reasonable)
    extreme_values = np.abs(observation) > 10.0
    extreme_count = extreme_values.sum()
    if extreme_count > 0:
        issues.append(f"Observation contains {extreme_count} values with |value| > 10.0")
        extreme_indices = np.where(extreme_values)[0]
        if len(extreme_indices) <= 5:
            for idx in extreme_indices:
                issues.append(f"  - Index {idx}: {observation[idx]:.3f}")
    
    # Check for all-zero observation (might indicate data problems)
    if np.allclose(observation, 0.0):
        issues.append("Observation vector is all zeros (possible data issue)")
    
    is_valid = len(issues) == 0
    
    if not is_valid:
        logger.warning(f"Observation validation failed: {len(issues)} issues found")
        for issue in issues[:5]:
            logger.warning(f"  - {issue}")
    
    return is_valid, issues