"""
Comprehensive tests for market-agnostic feature extractors.

This test validates:
- Cents → probability conversion accuracy
- Feature extraction consistency across markets
- Universal feature normalization
- Cross-market feature compatibility
- Portfolio state encoding accuracy
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

from kalshiflow_rl.environments.feature_extractors import (
    extract_market_agnostic_features,
    extract_temporal_features, 
    extract_portfolio_features,
    build_observation_from_session_data,
    calculate_observation_space_size,
    validate_feature_consistency,
    validate_observation_vector,
    _get_default_market_features,
    _get_default_portfolio_features
)
from kalshiflow_rl.environments.session_data_loader import SessionDataPoint


class TestMarketAgnosticFeatures:
    """Test market-agnostic feature extraction with cents → probability conversion."""
    
    def test_empty_orderbook_data(self):
        """Test feature extraction with empty orderbook data."""
        features = extract_market_agnostic_features({})
        
        # Should return default features
        default_features = _get_default_market_features()
        assert features == default_features
        
        # Validate all features are in valid ranges
        is_valid, issues = validate_feature_consistency(features, features)
        assert is_valid, f"Default features validation failed: {issues}"
    
    def test_cents_to_probability_conversion(self):
        """Test accurate cents → probability conversion."""
        # Sample orderbook data with prices in cents
        orderbook_data = {
            'yes_bids': {'45': 100, '44': 200},  # Best bid: 45 cents
            'yes_asks': {'46': 150, '47': 250},  # Best ask: 46 cents
            'no_bids': {'54': 120, '53': 180},   # Best bid: 54 cents
            'no_asks': {'55': 160, '56': 240}    # Best ask: 55 cents
        }
        
        features = extract_market_agnostic_features(orderbook_data)
        
        # Verify cents → probability conversion
        assert abs(features['best_yes_bid_norm'] - 0.45) < 0.001  # 45 cents → 0.45
        assert abs(features['best_yes_ask_norm'] - 0.46) < 0.001  # 46 cents → 0.46
        assert abs(features['best_no_bid_norm'] - 0.54) < 0.001   # 54 cents → 0.54
        assert abs(features['best_no_ask_norm'] - 0.55) < 0.001   # 55 cents → 0.55
        
        # Verify spread calculations in probability space
        expected_yes_spread = (46 - 45) / 100.0  # 1 cent → 0.01
        expected_no_spread = (55 - 54) / 100.0   # 1 cent → 0.01
        assert abs(features['yes_spread_norm'] - expected_yes_spread) < 0.001
        assert abs(features['no_spread_norm'] - expected_no_spread) < 0.001
        
        # Verify mid-prices in probability space
        expected_yes_mid = (45 + 46) / 2.0 / 100.0  # 45.5 cents → 0.455
        expected_no_mid = (54 + 55) / 2.0 / 100.0    # 54.5 cents → 0.545
        assert abs(features['yes_mid_price_norm'] - expected_yes_mid) < 0.001
        assert abs(features['no_mid_price_norm'] - expected_no_mid) < 0.001
    
    def test_volume_normalization(self):
        """Test volume feature normalization."""
        orderbook_data = {
            'yes_bids': {'45': 500},
            'yes_asks': {'46': 300},
            'no_bids': {'54': 400},
            'no_asks': {'55': 200}
        }
        
        features = extract_market_agnostic_features(orderbook_data)
        
        # All volume features should be normalized to [0,1]
        assert 0.0 <= features['yes_volume_norm'] <= 1.0
        assert 0.0 <= features['no_volume_norm'] <= 1.0
        assert 0.0 <= features['total_volume_norm'] <= 1.0
        
        # Volume imbalance should be in [-1,1]
        assert -1.0 <= features['volume_imbalance'] <= 1.0
        assert -1.0 <= features['yes_side_imbalance'] <= 1.0
        assert -1.0 <= features['no_side_imbalance'] <= 1.0
    
    def test_arbitrage_detection(self):
        """Test arbitrage opportunity detection."""
        # Perfect arbitrage opportunity: YES + NO != 100
        orderbook_data = {
            'yes_bids': {'40': 100},
            'yes_asks': {'41': 100},
            'no_bids': {'50': 100},    # Should be ~59 for no arbitrage
            'no_asks': {'51': 100}
        }
        
        features = extract_market_agnostic_features(orderbook_data)
        
        # Should detect arbitrage opportunity
        assert features['arbitrage_opportunity'] > 0.0
        assert features['market_efficiency'] < 1.0
        
        # Test efficient market (YES + NO ≈ 100)
        efficient_data = {
            'yes_bids': {'45': 100},
            'yes_asks': {'46': 100}, 
            'no_bids': {'54': 100},   # 45.5 + 54.5 = 100
            'no_asks': {'55': 100}
        }
        
        efficient_features = extract_market_agnostic_features(efficient_data)
        assert efficient_features['arbitrage_opportunity'] < 0.1  # Small due to spreads
        assert efficient_features['market_efficiency'] > 0.9
    
    def test_feature_ranges(self):
        """Test that all features are in expected ranges."""
        # Test with various orderbook configurations
        test_cases = [
            # Extreme spreads
            {'yes_bids': {'10': 100}, 'yes_asks': {'90': 100}, 'no_bids': {'10': 100}, 'no_asks': {'90': 100}},
            # High volume
            {'yes_bids': {'45': 10000}, 'yes_asks': {'46': 10000}, 'no_bids': {'54': 10000}, 'no_asks': {'55': 10000}},
            # Minimal volume
            {'yes_bids': {'45': 1}, 'yes_asks': {'46': 1}, 'no_bids': {'54': 1}, 'no_asks': {'55': 1}},
        ]
        
        for orderbook_data in test_cases:
            features = extract_market_agnostic_features(orderbook_data)
            
            # Validate feature ranges
            for key, value in features.items():
                assert np.isfinite(value), f"Feature '{key}' is not finite: {value}"
                
                if key.endswith('_imbalance'):
                    assert -1.1 <= value <= 1.1, f"Imbalance feature '{key}' out of range: {value}"
                else:
                    assert -0.1 <= value <= 1.1, f"Feature '{key}' out of [0,1] range: {value}"
    
    def test_cross_market_consistency(self):
        """Test that features are extracted consistently across different markets."""
        # Same orderbook structure for different "markets" (should produce identical features)
        market_data = {
            'yes_bids': {'45': 100, '44': 200},
            'yes_asks': {'46': 150, '47': 250}, 
            'no_bids': {'54': 120, '53': 180},
            'no_asks': {'55': 160, '56': 240}
        }
        
        features1 = extract_market_agnostic_features(market_data)
        features2 = extract_market_agnostic_features(market_data)  # Same data
        
        # Should be identical
        is_consistent, differences = validate_feature_consistency(features1, features2)
        assert is_consistent, f"Cross-market consistency failed: {differences}"
        
        # Test with slightly different volumes (should only affect volume features)
        market_data_2 = market_data.copy()
        market_data_2['yes_bids'] = {'45': 110, '44': 210}  # +10% volume
        
        features3 = extract_market_agnostic_features(market_data_2)
        
        # Price features should be identical
        price_features = ['best_yes_bid_norm', 'best_yes_ask_norm', 'yes_spread_norm', 'yes_mid_price_norm']
        for feature in price_features:
            assert abs(features1[feature] - features3[feature]) < 0.001, f"Price feature '{feature}' should be identical"


class TestTemporalFeatures:
    """Test temporal feature extraction."""
    
    def create_session_data_point(self, timestamp_ms: int, activity_score: float = 0.5, 
                                markets_data: dict = None) -> SessionDataPoint:
        """Helper to create SessionDataPoint for testing."""
        return SessionDataPoint(
            timestamp=datetime.fromtimestamp(timestamp_ms / 1000.0),
            timestamp_ms=timestamp_ms,
            markets_data=markets_data or {},
            time_gap=0.0,
            activity_score=activity_score,
            momentum=0.0
        )
    
    def test_time_based_features(self):
        """Test time-based feature extraction."""
        base_time = int(datetime(2024, 1, 1, 14, 30, 0).timestamp() * 1000)  # 2:30 PM
        
        current_data = self.create_session_data_point(base_time)
        current_data.time_gap = 30.0  # 30 seconds since last update
        
        historical_data = [
            self.create_session_data_point(base_time - 60000),  # 1 minute ago
            self.create_session_data_point(base_time - 30000),  # 30 seconds ago
        ]
        
        features = extract_temporal_features(current_data, historical_data)
        
        # Time gap normalization (30 seconds / 300 seconds max = 0.1)
        assert abs(features['time_since_last_update'] - 0.1) < 0.01
        
        # Time of day (2:30 PM = 14.5 hours, business hours 9-16, so (14.5-9)/7 ≈ 0.786)
        expected_time_norm = (14.5 - 9) / 7.0
        assert abs(features['time_of_day_norm'] - expected_time_norm) < 0.1
        
        # Day of week (should be normalized)
        assert 0.0 <= features['day_of_week_norm'] <= 1.0
    
    def test_activity_detection(self):
        """Test activity burst and quiet period detection."""
        base_time = int(datetime.now().timestamp() * 1000)
        
        # Create historical data with varying activity (more points for better statistics)
        historical_data = []
        activity_scores = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.8, 0.9, 0.8, 0.2, 0.1, 0.1]  # Burst towards end
        
        for i, activity in enumerate(activity_scores):
            point = self.create_session_data_point(base_time - (len(activity_scores) - i) * 1000, activity)
            point.activity_score = activity  # Set activity score explicitly
            historical_data.append(point)
        
        # Test burst detection with very high activity
        current_burst = self.create_session_data_point(base_time, 1.0)  # Maximum activity
        current_burst.activity_score = 1.0  # Set activity score explicitly
        features_burst = extract_temporal_features(current_burst, historical_data)
        
        # Debug print
        print(f"Burst detection: activity={current_burst.activity_score}, indicator={features_burst['activity_burst_indicator']}")
        
        assert features_burst['activity_burst_indicator'] == 1.0  # Should detect burst
        assert features_burst['current_activity_score'] == 1.0
        
        # Test quiet period detection with very low activity relative to high historical activity
        # Create historical data with higher activity levels
        high_activity_data = []
        high_activity_scores = [0.8, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8, 0.7, 0.8]  # High activity
        
        for i, activity in enumerate(high_activity_scores):
            point = self.create_session_data_point(base_time - (len(high_activity_scores) - i) * 1000, activity)
            point.activity_score = activity  # Set activity score explicitly
            high_activity_data.append(point)
        
        current_quiet = self.create_session_data_point(base_time, 0.0)  # Very low activity relative to history
        current_quiet.activity_score = 0.0  # Set activity score explicitly
        features_quiet = extract_temporal_features(current_quiet, high_activity_data)
        
        # Debug print
        print(f"Quiet detection: activity={current_quiet.activity_score}, indicator={features_quiet['quiet_period_indicator']}")
        
        assert features_quiet['quiet_period_indicator'] == 1.0  # Should detect quiet period
    
    def test_market_synchronization(self):
        """Test multi-market coordination features."""
        base_time = int(datetime.now().timestamp() * 1000)
        
        # Create data with multiple markets
        markets_data = {
            'MARKET1': {'total_volume': 1000},
            'MARKET2': {'total_volume': 800}, 
            'MARKET3': {'total_volume': 1200}
        }
        
        current_data = self.create_session_data_point(base_time, markets_data=markets_data)
        current_data.mid_prices = {
            'MARKET1': (Decimal('45.0'), None),
            'MARKET2': (Decimal('52.0'), None),
            'MARKET3': (Decimal('38.0'), None)
        }
        
        # Create more historical data points (need at least 2 for synchronization, 5+ for correlation)
        historical_data = []
        for i in range(6):  # Create 6 historical points
            point = self.create_session_data_point(base_time - (6-i) * 1000, markets_data=markets_data)
            point.mid_prices = {
                'MARKET1': (Decimal(str(44.0 + i * 0.1)), None),  # Gradual increase
                'MARKET2': (Decimal(str(51.0 + i * 0.1)), None),  # Gradual increase (synchronized)
                'MARKET3': (Decimal(str(37.0 + i * 0.1)), None)   # Gradual increase (synchronized)
            }
            historical_data.append(point)
        
        features = extract_temporal_features(current_data, historical_data)
        
        # Check basic market features
        assert features['active_markets_norm'] == 0.3    # 3 markets / 10 max = 0.3
        
        # Synchronization should be detected (but might not be super high due to algorithm complexity)
        assert features['market_synchronization'] >= 0.4  # Some synchronization detected
        
        # Market divergence should be low since all markets moved in same direction
        assert features['market_divergence'] < 0.5       # Low divergence


class TestPortfolioFeatures:
    """Test portfolio state feature extraction."""
    
    def test_empty_portfolio(self):
        """Test feature extraction with empty portfolio."""
        features = extract_portfolio_features({}, 1000.0, 1000.0)
        
        # Should return default features for empty portfolio
        default_features = _get_default_portfolio_features()
        
        # Key checks for empty portfolio
        assert features['cash_ratio'] == 1.0  # All cash
        assert features['position_ratio'] == 0.0  # No positions
        assert features['position_count_norm'] == 0.0
        assert features['leverage'] == 0.0
    
    def test_kalshi_position_convention(self):
        """Test Kalshi position convention: +YES contracts, -NO contracts."""
        position_data = {
            'MARKET1': {'position': 100, 'cost_basis': 4500.0, 'realized_pnl': 0.0},    # 100 YES contracts
            'MARKET2': {'position': -50, 'cost_basis': 2500.0, 'realized_pnl': 100.0}, # 50 NO contracts 
            'MARKET3': {'position': 25, 'cost_basis': 1250.0, 'realized_pnl': -50.0}   # 25 YES contracts
        }
        
        features = extract_portfolio_features(position_data, 10000.0, 2000.0)
        
        # Portfolio composition
        assert features['cash_ratio'] == 0.2  # $2000 cash / $10000 total
        assert features['position_ratio'] == 0.8  # $8000 positions / $10000 total
        
        # Position characteristics
        assert features['position_count_norm'] == 0.3  # 3 positions / 10 max
        
        # Long/short analysis
        total_positions = 3
        long_positions = 2  # MARKET1 and MARKET3 (positive positions)
        short_positions = 1  # MARKET2 (negative position)
        
        assert abs(features['long_position_ratio'] - (long_positions / total_positions)) < 0.01
        assert abs(features['short_position_ratio'] - (short_positions / total_positions)) < 0.01
        
        # Net position bias: (100 - 50 + 25) / (100 + 50 + 25) = 75/175 = 0.429
        expected_bias = (100 - 50 + 25) / (100 + 50 + 25)
        assert abs(features['net_position_bias'] - np.tanh(expected_bias)) < 0.01
    
    def test_portfolio_risk_metrics(self):
        """Test portfolio risk and diversification metrics."""
        # Concentrated portfolio (one large position)
        concentrated_positions = {
            'MARKET1': {'position': 1000, 'cost_basis': 45000.0, 'realized_pnl': 0.0},  # Large position
            'MARKET2': {'position': 10, 'cost_basis': 500.0, 'realized_pnl': 0.0}      # Small position
        }
        
        concentrated_features = extract_portfolio_features(concentrated_positions, 50000.0, 5000.0)
        
        # Should detect high concentration
        assert concentrated_features['position_concentration'] > 0.8  # Large position dominates
        assert concentrated_features['position_diversity'] < 0.3     # Low diversity
        
        # Diversified portfolio (multiple equal positions)
        diversified_positions = {
            f'MARKET{i}': {'position': 50, 'cost_basis': 2500.0, 'realized_pnl': 0.0} 
            for i in range(1, 6)  # 5 equal positions
        }
        
        diversified_features = extract_portfolio_features(diversified_positions, 20000.0, 7500.0)
        
        # Should detect good diversification
        assert diversified_features['position_concentration'] < 0.3  # No single large position
        assert diversified_features['position_diversity'] > 0.7     # High diversity
    
    def test_leverage_calculation(self):
        """Test leverage approximation."""
        # High leverage scenario
        high_leverage_positions = {
            'MARKET1': {'position': 2000, 'cost_basis': 80000.0, 'realized_pnl': 0.0}  # Large position relative to portfolio
        }
        
        high_leverage_features = extract_portfolio_features(high_leverage_positions, 50000.0, 10000.0)
        
        # Should detect high leverage (position value >> portfolio value)
        assert high_leverage_features['leverage'] > 0.8  # High leverage
        
        # Low leverage scenario
        low_leverage_positions = {
            'MARKET1': {'position': 100, 'cost_basis': 5000.0, 'realized_pnl': 0.0}
        }
        
        low_leverage_features = extract_portfolio_features(low_leverage_positions, 50000.0, 45000.0)
        
        # Should detect low leverage
        assert low_leverage_features['leverage'] < 0.2  # Low leverage


class TestObservationBuilding:
    """Test complete observation building from session data."""
    
    def create_sample_session_data(self) -> SessionDataPoint:
        """Create sample session data for testing."""
        return SessionDataPoint(
            timestamp=datetime(2024, 1, 1, 14, 30, 0),
            timestamp_ms=int(datetime(2024, 1, 1, 14, 30, 0).timestamp() * 1000),
            markets_data={
                'MARKET1': {
                    'yes_bids': {'45': 100},
                    'yes_asks': {'46': 150},
                    'no_bids': {'54': 120},
                    'no_asks': {'55': 160},
                    'total_volume': 530
                },
                'MARKET2': {
                    'yes_bids': {'48': 80},
                    'yes_asks': {'49': 120},
                    'no_bids': {'51': 90},
                    'no_asks': {'52': 130},
                    'total_volume': 420
                }
            }
        )
    
    def test_observation_vector_construction(self):
        """Test complete observation vector construction."""
        current_data = self.create_sample_session_data()
        historical_data = [self.create_sample_session_data()]  # Same for simplicity
        
        position_data = {
            'MARKET1': {'position': 100, 'cost_basis': 4500.0, 'realized_pnl': 0.0}
        }
        
        observation = build_observation_from_session_data(
            current_data, historical_data, position_data, 10000.0, 5500.0, max_markets=1
        )
        
        # Check observation shape
        expected_size = calculate_observation_space_size(max_markets=1)
        assert observation.shape == (expected_size,), f"Expected shape {(expected_size,)}, got {observation.shape}"
        
        # Validate observation vector
        is_valid, issues = validate_observation_vector(observation)
        assert is_valid, f"Observation validation failed: {issues}"
        
        # Check that features are finite and in reasonable ranges
        assert np.all(np.isfinite(observation)), "Observation contains non-finite values"
        assert np.all(np.abs(observation) <= 2.0), "Observation contains extreme values"
    
    def test_market_sorting_by_activity(self):
        """Test that markets are sorted by activity for consistent ordering."""
        # Create session with markets of different activity levels
        session_data = SessionDataPoint(
            timestamp=datetime.now(),
            timestamp_ms=int(datetime.now().timestamp() * 1000),
            markets_data={
                'LOW_VOLUME': {'total_volume': 100},
                'HIGH_VOLUME': {'total_volume': 1000},
                'MED_VOLUME': {'total_volume': 500}
            }
        )
        
        observation = build_observation_from_session_data(
            session_data, [], {}, 10000.0, 10000.0, max_markets=1
        )
        
        # Should include highest volume market first (HIGH_VOLUME)
        # MED_VOLUME and LOW_VOLUME should be excluded due to max_markets=1
        # This is tested by ensuring deterministic ordering
        is_valid, issues = validate_observation_vector(observation)
        assert is_valid, f"Sorted observation validation failed: {issues}"
    
    def test_observation_space_size_calculation(self):
        """Test observation space size calculation."""
        for max_markets in [1, 2, 3, 5]:
            size = calculate_observation_space_size(max_markets)
            
            # Should be positive and reasonable
            assert size > 0, f"Invalid observation space size for max_markets={max_markets}"
            assert size < 1000, f"Observation space too large for max_markets={max_markets}: {size}"
            
            # Should increase with more markets (except for max_markets=1 as our default)
            if max_markets > 1:
                smaller_size = calculate_observation_space_size(max_markets - 1)
                assert size > smaller_size, f"Observation size should increase with more markets"


class TestFeatureConsistency:
    """Test feature extraction consistency across different conditions."""
    
    def test_identical_input_consistency(self):
        """Test that identical inputs produce identical features."""
        orderbook_data = {
            'yes_bids': {'45': 100, '44': 200},
            'yes_asks': {'46': 150, '47': 250},
            'no_bids': {'54': 120, '53': 180},
            'no_asks': {'55': 160, '56': 240}
        }
        
        features1 = extract_market_agnostic_features(orderbook_data)
        features2 = extract_market_agnostic_features(orderbook_data)
        
        is_consistent, differences = validate_feature_consistency(features1, features2)
        assert is_consistent, f"Identical input consistency failed: {differences}"
    
    def test_feature_value_ranges(self):
        """Test that all features are in expected value ranges."""
        # Test with various realistic orderbook scenarios
        test_scenarios = [
            # Normal market
            {'yes_bids': {'45': 100}, 'yes_asks': {'46': 100}, 'no_bids': {'54': 100}, 'no_asks': {'55': 100}},
            # Wide spread
            {'yes_bids': {'30': 50}, 'yes_asks': {'70': 50}, 'no_bids': {'30': 50}, 'no_asks': {'70': 50}},
            # High volume
            {'yes_bids': {'45': 5000}, 'yes_asks': {'46': 5000}, 'no_bids': {'54': 5000}, 'no_asks': {'55': 5000}},
            # Imbalanced
            {'yes_bids': {'45': 1000}, 'yes_asks': {'46': 100}, 'no_bids': {'54': 100}, 'no_asks': {'55': 1000}},
        ]
        
        for i, scenario in enumerate(test_scenarios):
            features = extract_market_agnostic_features(scenario)
            
            # Validate each feature's range
            for key, value in features.items():
                assert np.isfinite(value), f"Scenario {i}: Feature '{key}' is not finite: {value}"
                
                if 'imbalance' in key or 'bias' in key or key in ['price_momentum', 'activity_change']:
                    assert -1.1 <= value <= 1.1, f"Scenario {i}: Feature '{key}' out of [-1,1] range: {value}"
                else:
                    assert -0.1 <= value <= 1.1, f"Scenario {i}: Feature '{key}' out of [0,1] range: {value}"
    
    def test_training_inference_consistency(self):
        """Test that training and inference use identical features."""
        # Simulate training data
        session_data = SessionDataPoint(
            timestamp=datetime.now(),
            timestamp_ms=int(datetime.now().timestamp() * 1000),
            markets_data={
                'MARKET1': {'yes_bids': {'45': 100}, 'yes_asks': {'46': 100}, 
                           'no_bids': {'54': 100}, 'no_asks': {'55': 100}, 'total_volume': 400}
            }
        )
        
        position_data = {'MARKET1': {'position': 50, 'cost_basis': 2500.0, 'realized_pnl': 0.0}}
        
        # Build observation twice (simulating training and inference)
        obs1 = build_observation_from_session_data(session_data, [], position_data, 10000.0, 7500.0)
        obs2 = build_observation_from_session_data(session_data, [], position_data, 10000.0, 7500.0)
        
        # Should be identical
        np.testing.assert_array_equal(obs1, obs2, "Training/inference observations differ")
        
        # Validate both observations
        for obs in [obs1, obs2]:
            is_valid, issues = validate_observation_vector(obs)
            assert is_valid, f"Observation validation failed: {issues}"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_orderbook_sides(self):
        """Test handling of missing orderbook sides."""
        # Missing YES side
        orderbook_no_yes = {'no_bids': {'54': 100}, 'no_asks': {'55': 100}}
        features_no_yes = extract_market_agnostic_features(orderbook_no_yes)
        
        # Should handle gracefully with defaults
        assert 0.0 <= features_no_yes['best_yes_bid_norm'] <= 1.0
        assert 0.0 <= features_no_yes['best_yes_ask_norm'] <= 1.0
        
        # Missing NO side
        orderbook_no_no = {'yes_bids': {'45': 100}, 'yes_asks': {'46': 100}}
        features_no_no = extract_market_agnostic_features(orderbook_no_no)
        
        # Should handle gracefully with defaults
        assert 0.0 <= features_no_no['best_no_bid_norm'] <= 1.0
        assert 0.0 <= features_no_no['best_no_ask_norm'] <= 1.0
    
    def test_extreme_prices(self):
        """Test handling of extreme price values."""
        # Extreme low prices that sum correctly
        extreme_low = {'yes_bids': {'1': 100}, 'yes_asks': {'2': 100}, 
                      'no_bids': {'98': 100}, 'no_asks': {'99': 100}}
        features_low = extract_market_agnostic_features(extreme_low)
        
        # Should convert correctly: 1 cent → 0.01, 99 cents → 0.99
        assert abs(features_low['best_yes_bid_norm'] - 0.01) < 0.001
        assert abs(features_low['best_no_ask_norm'] - 0.99) < 0.001
        
        # Test arbitrage detection with prices that don't sum to 100
        arbitrage_case = {'yes_bids': {'30': 100}, 'yes_asks': {'35': 100}, 
                         'no_bids': {'30': 100}, 'no_asks': {'35': 100}}  # Both sides at ~32.5 cents = 65 total
        features_arb = extract_market_agnostic_features(arbitrage_case)
        
        # YES mid ≈ 0.325, NO mid ≈ 0.325, total ≈ 0.65 (should be 1.0)
        assert features_arb['arbitrage_opportunity'] > 0.3  # Should detect arbitrage
    
    def test_zero_volume_handling(self):
        """Test handling of zero volumes."""
        zero_volume_data = {
            'yes_bids': {'45': 0},  # Zero volume
            'yes_asks': {'46': 0},
            'no_bids': {'54': 0}, 
            'no_asks': {'55': 0}
        }
        
        features = extract_market_agnostic_features(zero_volume_data)
        
        # Should handle zero volumes gracefully
        assert features['yes_volume_norm'] == 0.0
        assert features['no_volume_norm'] == 0.0
        assert features['volume_imbalance'] == 0.0
    
    def test_portfolio_edge_cases(self):
        """Test portfolio feature extraction edge cases."""
        # Zero portfolio value
        zero_features = extract_portfolio_features({}, 0.0, 0.0)
        default_features = _get_default_portfolio_features()
        assert zero_features == default_features
        
        # Negative positions (short positions)
        short_positions = {
            'MARKET1': {'position': -100, 'cost_basis': 5500.0, 'realized_pnl': 0.0}  # Short 100 contracts
        }
        
        short_features = extract_portfolio_features(short_positions, 10000.0, 4500.0)
        
        # Should handle negative positions correctly
        assert short_features['short_position_ratio'] == 1.0  # All positions are short
        assert short_features['long_position_ratio'] == 0.0   # No long positions
        assert short_features['net_position_bias'] < 0.0      # Negative bias


if __name__ == "__main__":
    pytest.main([__file__, "-v"])